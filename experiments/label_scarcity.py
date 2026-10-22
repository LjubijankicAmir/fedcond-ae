"""Experiment 4: label scarcity (H3).

Same setup as experiment 1 (paper_comparison, time-ordered split, no injected drift), but after the fully
labeled start, each streamed row keeps its labels only with probability `label_rate` - a user who labels a lot
during setup and rarely afterwards. Hidden labels are unavailable for local training and for FedConD's drift
score; the final global model is still scored on the clients' true test labels.

Usage: python -m experiments.label_scarcity [--workers N]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from experiments.paper_comparison import run_job
from src.data import load_clients, load_config
from src.utils import save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config()
    pcfg = cfg["paper_setup"]
    scheme = pcfg["label_scheme"]
    load_clients(cfg, scheme)

    jobs = [(scheme, seed, m, r) for r in pcfg["label_rates"] for seed in pcfg["seeds"] for m in pcfg["methods"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_job, *zip(*jobs)))
    df = pd.DataFrame([r for rows in results for r in rows])
    path = save_results(df, "label_scarcity", cfg)

    per_seed = df.groupby(["label_rate", "method", "seed"]).agg(
        macro=("test_macro", "mean"), micro=("test_micro", "mean"),
        fires=("fires", "mean"), scored=("scored_rounds", "mean"), updates=("updates", "mean"))
    summary = (per_seed.groupby(["label_rate", "method"])
                       .agg(macro=("macro", "mean"), macro_seed_sd=("macro", "std"), micro=("micro", "mean"),
                            fires=("fires", "mean"), scored_rounds=("scored", "mean"), rounds=("updates", "mean"))
                       .reindex(pcfg["methods"], level="method")
                       .sort_index(level="label_rate", ascending=False, sort_remaining=False).reset_index())
    save_results(summary, "label_scarcity_summary", cfg)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
