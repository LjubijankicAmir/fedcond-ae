"""Evaluation metrics."""
from __future__ import annotations

import numpy as np


def macro_f1(Y: np.ndarray, P: np.ndarray, threshold: float = 0.5) -> float:
    """Macro F1 over the active labels of Y.

    Y: (rows, labels) with 1 / 0 / NaN; NaN entries are ignored.
    P: predicted probabilities, same shape.
    A label is active if it has at least one reported positive in Y (otherwise its F1 is undefined).
    """
    scores = []
    for j in range(Y.shape[1]):
        reported = ~np.isnan(Y[:, j])
        y = Y[reported, j] == 1
        if not y.any():
            continue
        p = P[reported, j] >= threshold
        tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
        scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else float("nan")


def micro_f1(Y: np.ndarray, P: np.ndarray, threshold: float = 0.5) -> float:
    """Micro F1: all reported (row, label) decisions pooled into one F1. NaN entries in Y are ignored."""
    reported = ~np.isnan(Y)
    y, p = Y[reported] == 1, P[reported] >= threshold
    tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else float("nan")
