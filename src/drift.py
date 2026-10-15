"""Synthetic drift injection on z-scored features.

Features are z-scored with the client's training statistics, so adding m to a column shifts it by
m training standard deviations. The result is clipped like the preprocessed data, as drifted sensor
readings would pass through the same preprocessing.
"""
from __future__ import annotations

import numpy as np


def onset_row(n_rows: int, onset_frac: float) -> int:
    """Row index (within the post-drift segment) where the drift starts."""
    return int(n_rows * onset_frac)


def drift_profile(n_rows: int, onset: int, kind: str, width: int) -> np.ndarray:
    """Per-row shift multiplier in [0, 1]: 0 before onset, then 1 (abrupt) or a linear ramp over `width` rows (gradual)."""
    i = np.arange(n_rows) - onset
    if kind == "abrupt":
        return (i >= 0).astype(np.float32)
    if kind == "gradual":
        return np.clip((i + 1) / width, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"unknown drift kind: {kind}")


def inject(X: np.ndarray, cols: list[int], onset: int, magnitude: float, kind: str, width: int,
           clip: float) -> np.ndarray:
    """Return a copy of X with `magnitude` added to `cols` from row `onset` on, clipped to +-clip."""
    X = X.copy()
    X[:, cols] += magnitude * drift_profile(len(X), onset, kind, width)[:, None]
    return np.clip(X, -clip, clip)
