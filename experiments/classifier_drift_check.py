"""Feasibility step 2: does injected drift visibly hurt a per-user activity classifier?

For each test user, train the MLP on the pre-drift training rows, then report macro F1 on:
- train:     the training rows (does it learn at all?)
- fpr:       the clean held-out rows just before the post-drift segment
- post:      post-drift rows from the drift onset on, clean and with abrupt / gradual drift of each magnitude
             (drift is added to the motion-sensor columns this client actually has)

All post-drift scores use the same rows, so any difference comes from the injected drift only.
"""
import numpy as np
import pandas as pd

from src.data import load_clients, load_config
from src.drift import inject, onset_row
from src.metrics import macro_f1
from src.models import predict_proba, train_classifier
from src.utils import save_results, set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    clients, features, _ = load_clients(cfg)
    ccfg, dcfg = cfg["classifier"], cfg["drift"]
    drift_cols = np.array([f.startswith(tuple(dcfg["columns_prefixes"])) for f in features])

    rows = []
    for prefix in cfg["feasibility_users"]:
        c = next(c for u, c in clients.items() if u.startswith(prefix))
        model = train_classifier(c.X["train"], c.Y["train"], ccfg, cfg["seed"])

        def f1(X, Y):
            return macro_f1(Y, predict_proba(model, X), ccfg["threshold"])

        cols = np.flatnonzero(drift_cols & c.observed)  # never shift a sensor this phone lacks
        onset = onset_row(len(c.X["post"]), dcfg["onset_frac"])
        Y_after = c.Y["post"][onset:]
        rec = {"user": prefix, "train": f1(c.X["train"], c.Y["train"]), "fpr": f1(c.X["fpr"], c.Y["fpr"]),
               "post_clean": f1(c.X["post"][onset:], Y_after)}
        for kind in ("abrupt", "gradual"):
            for m in dcfg["magnitudes"]:
                X = inject(c.X["post"], cols, onset, m, kind, dcfg["gradual_width"])
                rec[f"{kind}_m{m}"] = f1(X[onset:], Y_after)
        rows.append(rec)

    df = pd.DataFrame(rows)
    path = save_results(df, "classifier_drift_check", cfg)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print("\nmean:\n" + df.drop(columns="user").mean().round(3).to_string())
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
