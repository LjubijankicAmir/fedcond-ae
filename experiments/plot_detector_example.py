"""Feasibility step 4: show the three drift detectors on one user's data stream.

One panel per detector, sharing the time axis (rows = minutes, 0 = drift onset). Each panel shows the
detector's score, its alarm threshold (label-free detectors only), the alarms it raised, and its detection
lag and false-alarm rate per 30-minute round (see src/evaluation.py). Everything left of the onset is clean data, so alarms there are false alarms.

Usage: python -m experiments.plot_detector_example [USER_PREFIX] [abrupt|gradual] [MAGNITUDE]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.data import load_clients, load_config
from src.detectors import train_autoencoder
from src.evaluation import drift_columns, evaluate_run, make_stream, run_detectors
from src.models import train_classifier
from src.utils import result_stem, set_seed

SCORE, ALARM, INK, MUTED, GRID, DRIFT_BG = "#2a78d6", "#d03b3b", "#0b0b0b", "#52514e", "#e6e5e0", "#f3f2ee"

PANELS = {
    "autoencoder": ("Autoencoder (label-free, ours)", "reconstruction error\nper minute (log scale)"),
    "entropy": ("Prediction entropy (label-free)", "classifier uncertainty\nper minute (0-1)"),
    "fedcond": ("FedConD test (needs labels)", "macro F1 per round\n(30 minutes)"),
}


def main():
    user = sys.argv[1] if len(sys.argv) > 1 else None
    kind = sys.argv[2] if len(sys.argv) > 2 else "abrupt"
    cfg = load_config()
    magnitude = float(sys.argv[3]) if len(sys.argv) > 3 else cfg["drift"]["magnitudes"][0]
    user = user or cfg["feasibility_users"][0]
    set_seed(cfg["seed"])

    clients, features, _ = load_clients(cfg)
    c = next(c for u, c in clients.items() if u.startswith(user))
    clf = train_classifier(c.X["train"], c.Y["train"], cfg["classifier"], cfg["seed"])
    ae = train_autoencoder(c.X["train"], cfg["autoencoder"], cfg["seed"])
    cols = drift_columns(features, c, cfg["drift"]["columns_prefixes"])
    X, Y, onset = make_stream(c, cols, kind, magnitude, cfg["drift"], cfg["data"]["clip_z"])
    runs = run_detectors(c, X, Y, clf, ae, cfg)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, (name, (title, ylabel)) in zip(axes, PANELS.items()):
        run = runs[name]
        x = run.score_rows - onset
        ax.axvspan(0, len(X) - onset, color=DRIFT_BG, zorder=0)
        if name == "fedcond":
            ax.plot(x, run.scores, "o-", color=SCORE, lw=0.8, ms=2.5)
            flagged = np.isin(run.score_rows, run.alarm_rows)
            ax.plot(x[flagged], run.scores[flagged], "v", color=ALARM, ms=7, zorder=3)
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.plot(x, run.scores, color=SCORE, lw=0.5)
            ax.axhline(run.threshold, color=ALARM, lw=1.2)
            ax.plot(run.alarm_rows - onset, np.full(len(run.alarm_rows), run.threshold), "v",
                    color=ALARM, ms=6, zorder=3)
            if name == "autoencoder":
                ax.set_yscale("log")
        ax.axvline(0, color=INK, lw=1.2, ls="--")

        res = evaluate_run(run, onset, len(c.X["fpr"]), len(X), cfg["fedcond_test"]["batch_rows"])
        lag_txt = "never" if np.isnan(res["lag"]) else f"after {res['lag']:.0f} min"
        ax.set_title(f"{title}   ·   drift detected: {lag_txt}   ·   "
                     f"false alarms: {res['fa_all']:.0%} of {res['clean_rounds']} clean rounds",
                     loc="left", fontsize=10.5, color=INK)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("minutes relative to drift onset (left of the dashed line: clean data)", color=MUTED)
    legend = [
        Line2D([], [], color=SCORE, lw=1.5, label="detector score"),
        Line2D([], [], color=ALARM, lw=1.2, label="alarm threshold (set on clean calibration data)"),
        Line2D([], [], color=ALARM, marker="v", ls="", ms=7, label="alarm raised"),
        Line2D([], [], color=INK, lw=1.2, ls="--", label="drift onset"),
        Patch(color=DRIFT_BG, label="drifted data"),
    ]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Drift detectors on user {user}: {kind} drift of +{magnitude:g} std on all motion sensors",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.935))

    path = f"{result_stem(f'detector_example_{user}_{kind}_m{magnitude:g}', cfg)}.png"
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
