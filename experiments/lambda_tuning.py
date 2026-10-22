"""Tune FedProx's mu and FedConD's base lambda on validation rows.

Setup of experiment 1 (paper_time scheme: per-client 60/20/20 by time, no injected drift, all labels). For each
value of the grid, the final global model is scored on every client's validation rows (macro F1, mean over
clients and seeds). The best value per method goes into config.yaml (paper_setup.fedprox_mu, federated.lambda_base).

Usage: python -m experiments.lambda_tuning [--workers N]
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch

from src.data import load_clients, load_config
from src.fedcond import make_paper_client, predict, run
from src.metrics import macro_f1
from src.utils import save_results, set_seed

SCHEME = "paper_time"


def run_job(seed: int, method: str, value: float) -> list[dict]:
    torch.set_num_threads(1)
    cfg = load_config()
    if method == "fedprox":
        cfg["paper_setup"]["fedprox_mu"] = value
    else:
        cfg["federated"]["lambda_base"] = value
    set_seed(seed)
    clients, _, _ = load_clients(cfg, SCHEME)
    order = sorted(clients)
    fed = [make_paper_client(clients[u], cfg, seed, with_ae=False) for u in order]
    t0 = time.time()
    w, model = run(fed, method, cfg, seed)
    thr = cfg["classifier"]["threshold"]
    rows = [{"seed": seed, "method": method, "value": value, "uuid": u[:8], "fires": len(fc.fired),
             "final_lambda": fc.lam, "val_macro": macro_f1(clients[u].Y["val"], predict(model, w, clients[u].X["val"]), thr)}
            for u, fc in zip(order, fed)]
    print(f"seed {seed} {method:<8} {value:<4} {time.time() - t0:6.0f} s", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config()
    tcfg = cfg["lambda_tuning"]
    load_clients(cfg, SCHEME)
    jobs = [(seed, m, v) for m in tcfg["methods"] for v in tcfg["grid"] for seed in tcfg["seeds"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_job, *zip(*jobs)))
    df = pd.DataFrame([r for rows in results for r in rows])
    path = save_results(df, "lambda_tuning", cfg)
    per_seed = df.groupby(["method", "value", "seed"]).agg(val_macro=("val_macro", "mean"), fires=("fires", "mean"),
                                                            final_lambda=("final_lambda", "mean"))
    summary = (per_seed.groupby(["method", "value"])
                       .agg(val_macro=("val_macro", "mean"), seed_sd=("val_macro", "std"), fires=("fires", "mean"),
                            final_lambda=("final_lambda", "mean")).reset_index())
    save_results(summary, "lambda_tuning_summary", cfg)
    print(summary.round(4).to_string(index=False))
    for m, g in summary.groupby("method"):
        best = g.loc[g.val_macro.idxmax()]
        print(f"best {m}: {best.value} (val macro F1 {best.val_macro:.4f})")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
