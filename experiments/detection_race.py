"""Detection race (H1, H2): the drift alarms on every usable user.

For each user x drift type x magnitude x threshold method, run the autoencoder, entropy and FedConD
detectors on the user's stream and record detection lag and false-alarm rates (see src/evaluation.py).

Control: each detector is also run on the same stream without drift. Its "lag" there (control_lag) is when
it would have said "drift" anyway; a detection is only real if it comes before that (real = lag < control_lag,
or detected while the control never fires).

The classifier and autoencoder are trained once per user on the clean training rows. FedConD's result does
not depend on the threshold method; it is repeated in both for easy comparison.

Usage: python -m experiments.detection_race [--feasibility-users]   (the 5 feasibility users only, quick)
"""
import argparse

import numpy as np
import pandas as pd

from src.data import load_clients, load_config
from src.detectors import train_autoencoder
from src.evaluation import DETECTORS, drift_columns, evaluate_run, make_stream, run_detectors
from src.models import train_classifier
from src.utils import save_results, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feasibility-users", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    clients, features, _ = load_clients(cfg)
    users = cfg["feasibility_users"] if args.feasibility_users else sorted(u[:8] for u in clients)
    out_name = "detection_feasibility" if args.feasibility_users else "detection_race"
    ecfg, dcfg = cfg["detection_race"], cfg["drift"]
    B = cfg["fedcond_test"]["batch_rows"]
    clip = cfg["data"]["clip_z"]

    rows = []
    for prefix in users:
        c = next(c for u, c in clients.items() if u.startswith(prefix))
        clf = train_classifier(c.X["train"], c.Y["train"], cfg["classifier"], cfg["seed"])
        ae = train_autoencoder(c.X["train"], cfg["autoencoder"], cfg["seed"])
        cols = drift_columns(features, c, dcfg["columns_prefixes"])
        X0, Y0, onset0 = make_stream(c, cols, "abrupt", 0.0, dcfg, clip)  # same stream, no drift
        control = {}
        for method in ecfg["threshold_methods"]:
            runs = run_detectors(c, X0, Y0, clf, ae, {**cfg, "alarm": {**cfg["alarm"], "method": method}})
            for name in DETECTORS:
                control[method, name] = evaluate_run(runs[name], onset0, len(c.X["fpr"]), len(X0), B)["lag"]

        for kind in ecfg["kinds"]:
            for m in dcfg["magnitudes"]:
                X, Y, onset = make_stream(c, cols, kind, m, dcfg, clip)
                for method in ecfg["threshold_methods"]:
                    run_cfg = {**cfg, "alarm": {**cfg["alarm"], "method": method}}
                    runs = run_detectors(c, X, Y, clf, ae, run_cfg)
                    for name in DETECTORS:
                        res = evaluate_run(runs[name], onset, len(c.X["fpr"]), len(X), B)
                        rows.append({"user": prefix, "kind": kind, "magnitude": m, "threshold": method,
                                     "detector": name, "onset_row": onset, **res,
                                     "control_lag": control[method, name]})
        print(f"done {prefix}")

    df = pd.DataFrame(rows)
    path = save_results(df, out_name, cfg)

    df["detected"] = df.lag.notna()
    df["real"] = df.detected & ((df.lag < df.control_lag) | df.control_lag.isna())
    summary = (df.groupby(["threshold", "kind", "magnitude", "detector"])
                 .agg(median_lag=("lag", "median"), detected=("detected", "sum"),
                      real=("real", "sum"), fa_fpr=("fa_fpr", "mean"),
                      fa_post=("fa_post", "mean"), fa_all=("fa_all", "mean"))
                 .reset_index())
    save_results(summary, out_name + "_summary", cfg)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(f"\n(detected / real = number of users out of {len(users)})")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
