"""Is there natural concept drift in ExtraSensory, and can the autoencoder see it without labels?

No drift is injected. Per client and seed: train the MLP and the autoencoder on the training rows (as in the
other experiments), calibrate the autoencoder alarm on the calibration day, then score every later row in time
order (calibration day, FPR day, post-drift segment) and summarise per day:

- macro F1 and masked BCE of the frozen classifier  (does the model get worse over time?)
- mean reconstruction error, share of rows above the alarm threshold and in alarm state  (does the AE see it?)
- mean prediction entropy
- feature shift: mean |day mean| of the z-scored features (all / motion sensors)  (does the input move?)
- label shift: mean |day positive rate - training positive rate| over the labels reported that day
  (does what the user does move?)

Then per client: trend of F1 and AE error over the post-drift days (Spearman), early vs late halves, the day of
the first AE alarm, and a "retrain" check: a model trained on train + early post-drift rows, scored on the late
rows, against the frozen model (if retraining helps, the change is learnable and the drift is real).

Days are 24 h bins counted from the end of the pre-drift segment (day 0 = first post-drift day; the FPR and
calibration days are -1 and -2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon

from src.data import load_clients, load_config
from src.detectors import ThresholdAlarm, prediction_entropy, reconstruction_error, train_autoencoder
from src.metrics import macro_f1
from src.models import masked_bce, predict_proba, train_classifier
from src.utils import save_results, set_seed


def day_stats(c, X, Y, t, day, clf, ae, alarm, motion, p_train, cfg) -> dict:
    ncfg, thr = cfg["natural_drift_check"], cfg["classifier"]["threshold"]
    P = predict_proba(clf, X)
    err = reconstruction_error(ae, X)
    labeled = (~np.isnan(Y)).any(axis=1)
    rec = {"day": day, "n": len(X), "n_labeled": int(labeled.sum()),
           "ae_err": float(err.mean()), "ae_above": float((err > alarm.threshold).mean()),
           "ae_alarm": float(len(alarm.alarm_state(err)) / len(err)),
           "entropy": float(prediction_entropy(P).mean()),
           "shift_all": float(np.abs(X[:, c.observed].mean(axis=0)).mean()),
           "shift_motion": float(np.abs(X[:, motion & c.observed].mean(axis=0)).mean()),
           "f1": np.nan, "bce": np.nan, "label_shift": np.nan}
    if labeled.sum() >= ncfg["min_day_labeled"]:
        rec["f1"] = macro_f1(Y, P, thr)
        rec["bce"] = float(masked_bce(torch.logit(torch.from_numpy(P).clamp(1e-6, 1 - 1e-6)),
                                      torch.from_numpy(Y)))
        reported = ~np.isnan(Y)
        with np.errstate(invalid="ignore"):
            p_day = np.where(reported.sum(axis=0) > 0, np.nansum(Y == 1, axis=0) / reported.sum(axis=0), np.nan)
        rec["label_shift"] = float(np.nanmean(np.abs(p_day - p_train)))
    return rec


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    clients, features, labels = load_clients(cfg)
    ncfg, ccfg, acfg = cfg["natural_drift_check"], cfg["classifier"], cfg["autoencoder"]
    motion = np.array([f.startswith(tuple(cfg["drift"]["columns_prefixes"])) for f in features])

    daily, per_client = [], []
    for ci, c in enumerate(clients.values()):
        t_end_pre = c.t["fpr"][-1]
        X = np.vstack([c.X[s] for s in ("calib", "fpr", "post")])
        Y = np.vstack([c.Y[s] for s in ("calib", "fpr", "post")])
        t = np.concatenate([c.t[s] for s in ("calib", "fpr", "post")])
        day = np.floor((t - t_end_pre) / 86400).astype(int)
        day[: len(c.X["calib"])] = -2          # the calibration day, whatever its exact hours
        day[len(c.X["calib"]): len(c.X["calib"]) + len(c.X["fpr"])] = -1
        days = [d for d in np.unique(day) if (day == d).sum() >= ncfg["min_day_rows"]]
        post_days = [d for d in days if d >= 0]
        if len(post_days) < ncfg["min_post_days"]:
            continue
        with np.errstate(invalid="ignore"):
            Yt = c.Y["train"]
            p_train = np.nansum(Yt == 1, axis=0) / (~np.isnan(Yt)).sum(axis=0)

        # retrain check: late half of the post-drift rows, scored by the frozen and by a retrained model
        n_post = len(c.X["post"])
        early, late = slice(0, n_post // 2), slice(n_post // 2, n_post)
        X_re = np.vstack([c.X["train"], c.X["post"][early]])
        Y_re = np.vstack([c.Y["train"], c.Y["post"][early]])
        k = min(n_post // 2, len(c.X["train"]) // 2)   # same size as train: drop the k oldest training rows
        X_rc = np.vstack([c.X["train"][k:], c.X["post"][early]])
        Y_rc = np.vstack([c.Y["train"][k:], c.Y["post"][early]])

        rows, frozen_late, retrained_late, recent_late = [], [], [], []
        for seed in ncfg["seeds"]:
            clf = train_classifier(c.X["train"], c.Y["train"], ccfg, seed)
            ae = train_autoencoder(c.X["train"], acfg, seed)
            alarm = ThresholdAlarm.calibrate(reconstruction_error(ae, c.X["calib"]), cfg["alarm"])
            for d in days:
                m = day == d
                rows.append({"client": ci, "uuid": c.uuid[:8], "seed": seed,
                             **day_stats(c, X[m], Y[m], t[m], d, clf, ae, alarm, motion, p_train, cfg)})
            frozen_late.append(macro_f1(c.Y["post"][late], predict_proba(clf, c.X["post"][late]), ccfg["threshold"]))
            clf_re = train_classifier(X_re, Y_re, ccfg, seed)
            retrained_late.append(macro_f1(c.Y["post"][late], predict_proba(clf_re, c.X["post"][late]), ccfg["threshold"]))
            clf_rc = train_classifier(X_rc, Y_rc, ccfg, seed)
            recent_late.append(macro_f1(c.Y["post"][late], predict_proba(clf_rc, c.X["post"][late]), ccfg["threshold"]))

        df = pd.DataFrame(rows)
        daily.append(df)
        g = df.groupby("day").mean(numeric_only=True)   # mean over seeds
        ref, post = g.loc[g.index < 0], g.loc[g.index >= 0]
        half = len(post) // 2
        post_f1 = post.f1.dropna()
        first_alarm = post.index[post.ae_alarm > 0]
        rec = {"client": ci, "uuid": c.uuid[:8], "n_post_days": len(post), "n_f1_days": len(post_f1),
               "f1_ref": ref.f1.mean(), "f1_post": post.f1.mean(),
               "f1_early": post.f1.iloc[:half].mean(), "f1_late": post.f1.iloc[half:].mean(),
               "bce_ref": ref.bce.mean(), "bce_early": post.bce.iloc[:half].mean(), "bce_late": post.bce.iloc[half:].mean(),
               "ae_ref": ref.ae_err.mean(), "ae_early": post.ae_err.iloc[:half].mean(), "ae_late": post.ae_err.iloc[half:].mean(),
               "ae_alarm_post": post.ae_alarm.mean(), "first_alarm_day": first_alarm[0] if len(first_alarm) else np.nan,
               "shift_motion_ref": ref.shift_motion.mean(), "shift_motion_late": post.shift_motion.iloc[half:].mean(),
               "label_shift_ref": ref.label_shift.mean(), "label_shift_late": post.label_shift.iloc[half:].mean(),
               "frozen_late": float(np.mean(frozen_late)), "retrained_late": float(np.mean(retrained_late)),
               "recent_late": float(np.mean(recent_late))}
        for name, a, b in (("f1_trend", post_f1.index, post_f1.values),
                           ("ae_trend", post.index, post.ae_err.values),
                           ("ae_f1_corr", post.ae_err[post_f1.index].values, post_f1.values),
                           ("label_f1_corr", post.label_shift[post_f1.index].values, post_f1.values)):
            if len(a) >= 3:
                r, p = spearmanr(a, b)
                rec[name], rec[name + "_p"] = float(r), float(p)
            else:
                rec[name], rec[name + "_p"] = np.nan, np.nan
        per_client.append(rec)
        print(f"{c.uuid[:8]}: {len(post)} post days  F1 ref {rec['f1_ref']:.3f} early {rec['f1_early']:.3f} "
              f"late {rec['f1_late']:.3f}  AE ref {rec['ae_ref']:.3f} late {rec['ae_late']:.3f}  "
              f"alarm {rec['ae_alarm_post']:.2f}  retrain {rec['frozen_late']:.3f}->{rec['retrained_late']:.3f} recent {rec['recent_late']:.3f}",
              flush=True)

    daily = pd.concat(daily, ignore_index=True)
    pc = pd.DataFrame(per_client)
    pc["f1_delta"] = pc.f1_late - pc.f1_early
    pc["f1_drop_from_ref"] = pc.f1_post - pc.f1_ref
    pc["ae_delta"] = pc.ae_late - pc.ae_ref
    pc["retrain_gain"] = pc.retrained_late - pc.frozen_late
    pc["recent_gain"] = pc.recent_late - pc.frozen_late
    save_results(daily, "natural_drift_daily", cfg)
    path = save_results(pc, "natural_drift_check", cfg)

    pd.set_option("display.width", 250)
    cols = ["uuid", "n_post_days", "f1_ref", "f1_early", "f1_late", "f1_trend", "f1_trend_p", "ae_ref", "ae_late",
            "ae_trend", "ae_trend_p", "ae_f1_corr", "ae_alarm_post", "first_alarm_day", "label_shift_late",
            "label_f1_corr", "frozen_late", "retrained_late", "recent_late"]
    print("\n" + pc[cols].round(3).to_string(index=False))

    def test(x):
        x = x.dropna()
        return f"median {x.median():+.3f}, mean {x.mean():+.3f}, {int((x > 0).sum())}/{len(x)} > 0, Wilcoxon p = {wilcoxon(x).pvalue:.3f}"

    n = len(pc)
    print(f"\nclients: {n}")
    print(f"F1 late - early:            {test(pc.f1_delta)}")
    print(f"F1 post - clean reference:  {test(pc.f1_drop_from_ref)}")
    print(f"AE error late - reference:  {test(pc.ae_delta)}")
    print(f"retrained - frozen (late):  {test(pc.retrain_gain)}")
    print(f"recent - frozen (late):     {test(pc.recent_gain)}   (same training size; the drift test)")
    print(f"F1 trend < 0 (p<.05): {int(((pc.f1_trend < 0) & (pc.f1_trend_p < .05)).sum())}/{n}   "
          f"AE trend > 0 (p<.05): {int(((pc.ae_trend > 0) & (pc.ae_trend_p < .05)).sum())}/{n}   "
          f"both: {int(((pc.f1_trend < 0) & (pc.f1_trend_p < .05) & (pc.ae_trend > 0) & (pc.ae_trend_p < .05)).sum())}/{n}")
    print(f"AE alarm on any post day: {int((pc.ae_alarm_post > 0).sum())}/{n}   "
          f"AE err vs F1 corr < 0 (p<.05): {int(((pc.ae_f1_corr < 0) & (pc.ae_f1_corr_p < .05)).sum())}/{n}   "
          f"label shift vs F1 corr < 0 (p<.05): {int(((pc.label_f1_corr < 0) & (pc.label_f1_corr_p < .05)).sum())}/{n}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
