"""Reaction test: with a perfect (oracle) drift alarm, does any reaction improve accuracy after drift?

Drift scheme with all usable clients; `n_drifted` of them get abrupt drift on the motion sensors. Training rows
are fully labeled; afterwards only `label_rate` of the stream rows keep their labels. The oracle fires exactly
at the drift onset; the reactions compared are none, FedConD's (lambda += lambda_step) and recentering (label-free
input correction). Reported: test-then-train macro F1 before and after the onset, drifted and clean clients.

Usage: python -m experiments.reaction_test [--workers N]
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from src.data import load_clients, load_config
from src.evaluation import drift_columns
from src.fedcond import evaluate, make_fed_client, run
from src.utils import save_results, set_seed


def run_job(seed: int, method: str) -> list[dict]:
    torch.set_num_threads(1)
    cfg = load_config()
    rcfg = cfg["reaction_test"]
    cfg["federated"] = {**cfg["federated"], "kind": rcfg["kind"], "magnitude": rcfg["magnitude"]}
    set_seed(seed)
    clients, features, _ = load_clients(cfg)
    order = sorted(clients)
    drifted = set(np.random.default_rng(seed).choice(len(order), rcfg["n_drifted"], replace=False))
    fed = [make_fed_client(clients[u], k in drifted, drift_columns(features, clients[u], cfg["drift"]["columns_prefixes"]),
                           cfg, seed, with_ae=False, label_rate=rcfg["label_rate"], label_seed=k)
           for k, u in enumerate(order)]
    t0 = time.time()
    run(fed, method, cfg, seed)
    print(f"seed {seed} {method:<16} {time.time() - t0:6.0f} s", flush=True)
    return [{"seed": seed, "method": method, **evaluate(fc, cfg["classifier"]["threshold"])} for fc in fed]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config()
    rcfg = cfg["reaction_test"]
    load_clients(cfg)
    jobs = [(seed, m) for seed in rcfg["seeds"] for m in rcfg["methods"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_job, *zip(*jobs)))
    df = pd.DataFrame([r for rows in results for r in rows])
    path = save_results(df, "reaction_test", cfg)
    summary = (df.groupby(["drifted", "method"])
                 .agg(f1_before=("f1_before", "mean"), f1_after=("f1_after", "mean"), fires=("fires_after", "mean"))
                 .reindex(rcfg["methods"], level="method").reset_index())
    save_results(summary, "reaction_test_summary", cfg)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
