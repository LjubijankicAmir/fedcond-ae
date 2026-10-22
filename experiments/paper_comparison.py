"""Experiment 1: FedAvg, FedProx, FedConD and FedConD-AE in a setup close to the FedConD paper's ExtraSensory one.

No injected drift (natural drift only). Per client 60/20/20 train/val/test, by time and at random. The first
`initial_frac` of each client's training rows is available at the start; the rest streams in, B rows per local
round. After the stream ends, the final global model is scored on every client's test rows (macro and micro F1),
as in the paper's Table I (average and variance over clients). Validation rows are reserved for tuning.

Validation goal: the paper's ranking FedConD > FedProx > FedAvg.

Usage: python -m experiments.paper_comparison [--workers N] [--methods m1 m2 ...]
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch

from src.data import load_clients, load_config
from src.fedcond import make_paper_client, predict, run
from src.metrics import macro_f1, micro_f1
from src.utils import save_results, set_seed


def run_job(scheme: str, seed: int, method: str, label_rate: float = 1.0) -> list[dict]:
    """One run; label_rate < 1 hides labels in the stream (see make_paper_client). Returns one row per client."""
    torch.set_num_threads(1)
    cfg = load_config()
    set_seed(seed)
    clients, _, _ = load_clients(cfg, scheme)
    order = sorted(clients)
    fed = [make_paper_client(clients[u], cfg, seed, with_ae=method.startswith("fedcond_ae"), label_rate=label_rate,
                             label_seed=k) for k, u in enumerate(order)]
    t0 = time.time()
    w, model = run(fed, method, cfg, seed)
    thr = cfg["classifier"]["threshold"]
    rows = []
    for u, fc in zip(order, fed):
        c = clients[u]
        corrections = fc.corrections + ([fc.latest_ok] if fc.correct_from is not None else [])
        rec = {"scheme": scheme, "seed": seed, "method": method, "label_rate": label_rate, "uuid": u[:8],
               "fires": len(fc.fired), "scored_rounds": fc.scored, "final_lambda": fc.lam, "updates": fc.updates,
               "corrections_accepted": int(sum(corrections)), "corrections_rejected": int(len(corrections) - sum(corrections))}
        for split in ("val", "test"):
            P = predict(model, w, c.X[split])
            rec[f"{split}_macro"] = macro_f1(c.Y[split], P, thr)
            rec[f"{split}_micro"] = micro_f1(c.Y[split], P, thr)
        rows.append(rec)
    print(f"{scheme:<13} labels {label_rate:<4} seed {seed} {method:<11} {time.time() - t0:6.0f} s", flush=True)
    return rows


def summarize(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Per scheme x method: mean over clients (Table I 'Avg') and variance over clients ('Var'), each averaged
    over seeds, plus the spread of the client mean across seeds."""
    per_seed = (df.groupby(["scheme", "method", "seed"])
                  .agg(macro=("test_macro", "mean"), micro=("test_micro", "mean"),
                       macro_var=("test_macro", "var"), micro_var=("test_micro", "var"),
                       fires=("fires", "mean")))
    out = (per_seed.groupby(["scheme", "method"])
                   .agg(macro=("macro", "mean"), macro_seed_sd=("macro", "std"), macro_var=("macro_var", "mean"),
                        micro=("micro", "mean"), micro_seed_sd=("micro", "std"), micro_var=("micro_var", "mean"),
                        fires=("fires", "mean")))
    return out.reindex(methods, level="method").reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--methods", nargs="+", default=None, help="subset of paper_setup.methods to (re)run")
    args = parser.parse_args()
    cfg = load_config()
    pcfg = cfg["paper_setup"]
    methods = args.methods or pcfg["methods"]
    for scheme in pcfg["schemes"]:
        load_clients(cfg, scheme)  # build the caches once before the workers start

    jobs = [(s, seed, m) for s in pcfg["schemes"] for seed in pcfg["seeds"] for m in methods]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_job, *zip(*jobs)))
    df = pd.DataFrame([r for rows in results for r in rows])
    path = save_results(df, "paper_comparison", cfg)
    summary = summarize(df, methods)
    save_results(summary, "paper_comparison_summary", cfg)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
