"""Feasibility step 3: visual sanity check of the injected drift.

Plots one accelerometer feature of one test user around the drift onset, clean vs drifted,
for abrupt and gradual drift. Thin lines are the per-minute values, thick lines a trailing 30-minute mean
(past rows only, as a detector would see them).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_clients, load_config
from src.drift import inject, onset_row
from src.utils import result_stem

USER = 0                                   # index into config feasibility_users
FEATURE = "raw_acc:magnitude_stats:mean"   # average acceleration magnitude in the 20 s window
BEFORE, AFTER = 300, 600                   # rows shown around the onset
SMOOTH = 30                                # trailing-mean window (rows)

CLEAN, DRIFTED, INK, MUTED = "#a3a29c", "#2a78d6", "#0b0b0b", "#52514e"


def main():
    cfg = load_config()
    dcfg = cfg["drift"]
    clients, features, _ = load_clients(cfg)
    prefix = cfg["feasibility_users"][USER]
    c = next(c for u, c in clients.items() if u.startswith(prefix))
    cols = np.flatnonzero(np.array([f.startswith(tuple(dcfg["columns_prefixes"])) for f in features]) & c.observed)
    j = features.index(FEATURE)
    m = dcfg["magnitudes"][0]

    X = c.X["post"]
    onset = onset_row(len(X), dcfg["onset_frac"])
    lo, hi = max(0, onset - BEFORE), min(len(X), onset + AFTER)
    x = np.arange(lo, hi) - onset
    print(f"user {prefix}: onset at post-drift row {onset} of {len(X)} "
          f"(timestamp {int(c.t['post'][onset])}), magnitude {m}, {len(cols)} drifted columns")

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, sharey=True)
    for ax, kind in zip(axes, ("abrupt", "gradual")):
        drifted = inject(X, cols, onset, m, kind, dcfg["gradual_width"])[lo:hi, j]
        clean = X[lo:hi, j]
        for y, color, label in ((drifted, DRIFTED, f"drifted (+{m:g})"), (clean, CLEAN, "clean")):
            ax.plot(x, y, color=color, lw=0.6, alpha=0.45)
            ax.plot(x, pd.Series(y).rolling(SMOOTH, min_periods=1).mean(), color=color, lw=2, label=label)
        ax.axvline(0, color=INK, lw=1, ls="--")
        ax.text(3, 0.97, "onset", transform=ax.get_xaxis_transform(), va="top", color=INK, fontsize=9)
        if kind == "gradual":
            ax.axvline(dcfg["gradual_width"], color=MUTED, lw=1, ls=":")
            ax.text(dcfg["gradual_width"] + 3, 0.97, "ramp complete", transform=ax.get_xaxis_transform(),
                    va="top", color=MUTED, fontsize=9)
        ax.set_title(f"{kind.capitalize()} drift", loc="left", fontsize=11, color=INK)
        ax.set_ylabel("z-score (training std)", color=MUTED)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    axes[1].set_xlabel("minutes (rows) relative to drift onset", color=MUTED)
    fig.suptitle(f"{FEATURE}, user {prefix}", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()

    path = f"{result_stem('drift_example', cfg)}.png"
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
