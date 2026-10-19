"""Feasibility step 6: does reacting to drift improve accuracy in federated learning?

A small asynchronous FedConD federation (src/fedcond.py) is run once per trigger: none (never reacts),
fedcond (FedConD's supervised test), autoencoder (FedConD-AE) and oracle (fires exactly at the true onset).
Everything except the trigger is identical, including the seed. Some clients get injected drift.

Reported per client: prequential macro F1 of the received global model before and after the drift onset,
trigger fires, detection lag and final lambda. If the oracle clearly beats no reaction, earlier detection can
pay off (room for H3).
"""
import time

import numpy as np
import pandas as pd

from src.data import load_clients, load_config
from src.evaluation import drift_columns
from src.fedcond import evaluate, make_fed_client, run
from src.utils import save_results, set_seed


def main():
    cfg = load_config()
    fcfg = cfg["federated"]
    clients, features, _ = load_clients(cfg)
    chosen = sorted(clients.values(), key=lambda c: len(c.X["post"]), reverse=True)[:fcfg["n_clients"]]

    rows = []
    for seed in fcfg["seeds"]:
        set_seed(seed)
        drifted = set(np.random.default_rng(seed).choice(len(chosen), fcfg["n_drifted"], replace=False))
        for trigger in fcfg["triggers"]:
            t0 = time.time()
            fed = [make_fed_client(c, i in drifted, drift_columns(features, c, cfg["drift"]["columns_prefixes"]),
                                   cfg, seed) for i, c in enumerate(chosen)]
            run(fed, trigger, cfg, seed)
            for fc in fed:
                rows.append({"seed": seed, "trigger": trigger, **evaluate(fc, cfg["classifier"]["threshold"])})
            print(f"seed {seed} {trigger:<12} {time.time() - t0:5.0f} s")

    df = pd.DataFrame(rows)
    path = save_results(df, "federated_feasibility", cfg)
    summary = (df.groupby(["drifted", "trigger"])
                 .agg(f1_before=("f1_before", "mean"), f1_after=("f1_after", "mean"),
                      fires_before=("fires_before", "mean"), fires_after=("fires_after", "mean"),
                      lag=("lag", "median"), final_lambda=("final_lambda", "mean"))
                 .reindex(fcfg["triggers"], level="trigger").reset_index())
    save_results(summary, "federated_feasibility_summary", cfg)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
