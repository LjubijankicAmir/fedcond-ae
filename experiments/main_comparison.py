"""Main comparison (H3): injected drift plus label scarcity, all methods.

Drift scheme, all usable clients; `n_drifted` of them (drawn per seed) get drift of the given kind on the motion
sensors. Training rows are fully labeled; afterwards each stream row keeps its labels with probability
`label_rate` (hidden labels are unavailable for training and for FedConD's score; grading uses the true labels).
Methods: FedAvg, FedProx, FedConD, FedConD-AE, its two ablations (alarm with FedConD's reaction only; alarm with
the correction only) and the oracle alarm with the correction (reference). Reported: test-then-train macro F1
before and after the drift onset, on drifted and on clean clients.

Usage: python -m experiments.main_comparison [--workers N] [--methods m1 m2 ...] [--smoke]
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

# sensitivity variants: name -> (method actually run, federated settings overridden)
VARIANTS = {"fedcond_step01": ("fedcond", {"lambda_step": 0.1})}   # the paper's smallest lambda step


def run_job(seed: int, kind: str, label_rate: float, method: str) -> list[dict]:
    torch.set_num_threads(1)
    cfg = load_config()
    mcfg = cfg["main_comparison"]
    base, overrides = VARIANTS.get(method, (method, {}))
    cfg["federated"] = {**cfg["federated"], "kind": kind, "magnitude": mcfg["magnitude"], **overrides}
    set_seed(seed)
    clients, features, _ = load_clients(cfg)
    order = sorted(clients)
    drifted = set(np.random.default_rng(seed).choice(len(order), mcfg["n_drifted"], replace=False))
    fed = [make_fed_client(clients[u], k in drifted, drift_columns(features, clients[u], cfg["drift"]["columns_prefixes"]),
                           cfg, seed, with_ae=method.startswith("fedcond_ae"), label_rate=label_rate, label_seed=k)
           for k, u in enumerate(order)]
    t0 = time.time()
    run(fed, base, cfg, seed)
    print(f"seed {seed} {kind:<8} labels {label_rate:<5} {method:<20} {time.time() - t0:6.0f} s", flush=True)
    return [{"seed": seed, "kind": kind, "label_rate": label_rate, "method": method,
             **evaluate(fc, cfg["classifier"]["threshold"])} for fc in fed]


def summarize(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Per kind x label rate x drifted x method: F1 after onset (mean over clients, then over seeds), its spread
    over seeds, fires after onset and detection lag."""
    per_seed = (df.groupby(["kind", "label_rate", "drifted", "method", "seed"])
                  .agg(f1_before=("f1_before", "mean"), f1_after=("f1_after", "mean"),
                       micro_after=("micro_after", "mean"), fires_after=("fires_after", "mean"), lag=("lag", "median")))
    out = (per_seed.groupby(["kind", "label_rate", "drifted", "method"])
                   .agg(f1_before=("f1_before", "mean"), f1_after=("f1_after", "mean"), seed_sd=("f1_after", "std"),
                        micro_after=("micro_after", "mean"), fires_after=("fires_after", "mean"), lag=("lag", "median")))
    return (out.reindex(methods, level="method")
               .sort_index(level=["kind", "label_rate", "drifted"], ascending=[True, False, False], sort_remaining=False)
               .reset_index())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--methods", nargs="+", default=None, help="subset of main_comparison.methods to (re)run")
    parser.add_argument("--smoke", action="store_true", help="one seed, one kind, one label rate, two methods")
    args = parser.parse_args()
    cfg = load_config()
    mcfg = cfg["main_comparison"]
    load_clients(cfg)
    methods = args.methods or mcfg["methods"]
    if args.smoke:
        jobs = [(mcfg["seeds"][0], mcfg["kinds"][0], mcfg["label_rates"][-1], m) for m in ("fedcond", "fedcond_ae")]
    else:
        jobs = [(seed, kind, r, m) for kind in mcfg["kinds"] for r in mcfg["label_rates"]
                for seed in mcfg["seeds"] for m in methods]
        if args.methods is None:   # sensitivity variants: first kind only
            jobs += [(seed, mcfg["kinds"][0], r, m) for r in mcfg["label_rates"]
                     for seed in mcfg["seeds"] for m in mcfg.get("sensitivity", [])]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_job, *zip(*jobs)))
    df = pd.DataFrame([r for rows in results for r in rows])
    name = "main_comparison_smoke" if args.smoke else "main_comparison"
    path = save_results(df, name, cfg)
    summary = summarize(df, [m for m in mcfg["methods"] + mcfg.get("sensitivity", []) if m in set(df.method)])
    save_results(summary, name + "_summary", cfg)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
