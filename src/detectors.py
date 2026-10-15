"""Drift detectors (triggers).

- FedConDTest: FedConD's supervised test (Chen et al. 2021, Eq. 4) on per-round evaluation scores.
- Autoencoder: label-free; per-row reconstruction error of the client's own sensor data.
- prediction_entropy: label-free; per-row uncertainty of the classifier's predictions.

Both label-free scores are turned into alarms by the same rule (ThresholdAlarm): a threshold calibrated on
clean calibration rows, and an alarm once `confirm` consecutive rows exceed it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import norm
from torch import nn


class FedConDTest:
    """Yates-corrected z-test of the newest round score against the mean of the previous scores (queue).

    Drift = the new score is significantly *lower* than the history mean (one-sided). Every score is added
    to the queue after testing, whether or not drift was flagged.
    """

    def __init__(self, queue_size: int, alpha: float):
        self.queue: deque[float] = deque(maxlen=queue_size)
        self.alpha = alpha

    def update(self, score: float) -> bool:
        drift = False
        if self.queue and not np.isnan(score):
            a = len(self.queue)
            s_bar = float(np.mean(self.queue))
            s_hat = (sum(self.queue) + score) / (a + 1)
            delta = 1.0 / (a + 1)
            var = s_hat * (1.0 - s_hat) * delta
            if var > 0 and score < s_bar:
                gamma = (s_bar - score - 0.5 * delta) / np.sqrt(var)
                drift = bool(norm.sf(gamma) < self.alpha)
        if not np.isnan(score):
            self.queue.append(score)
        return drift


class Autoencoder(nn.Module):
    """Symmetric MLP autoencoder: n -> hidden... -> bottleneck -> ...hidden -> n (linear output)."""

    def __init__(self, n_in: int, hidden: list[int]):
        super().__init__()
        sizes = [n_in, *hidden]
        enc = [m for a, b in zip(sizes, sizes[1:]) for m in (nn.Linear(a, b), nn.ReLU())]
        rev = sizes[::-1]
        dec = [m for a, b in zip(rev, rev[1:]) for m in (nn.Linear(a, b), nn.ReLU())][:-1]
        self.net = nn.Sequential(*enc, *dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_autoencoder(X: np.ndarray, cfg: dict, seed: int) -> Autoencoder:
    """Train on the client's own (unlabeled) rows to reconstruct them, MSE loss."""
    torch.manual_seed(seed)
    model = Autoencoder(X.shape[1], cfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    X = torch.from_numpy(X)
    gen = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(cfg["epochs"]):
        for idx in torch.randperm(len(X), generator=gen).split(cfg["batch_size"]):
            opt.zero_grad()
            nn.functional.mse_loss(model(X[idx]), X[idx]).backward()
            opt.step()
    return model


@torch.no_grad()
def reconstruction_error(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    """Per-row mean squared reconstruction error."""
    model.eval()
    X = torch.from_numpy(X)
    return ((model(X) - X) ** 2).mean(dim=1).numpy()


def prediction_entropy(P: np.ndarray) -> np.ndarray:
    """Per-row mean binary entropy (in bits, 0..1) of the predicted label probabilities."""
    P = np.clip(P, 1e-7, 1 - 1e-7)
    return (-(P * np.log2(P) + (1 - P) * np.log2(1 - P))).mean(axis=1)


@dataclass
class ThresholdAlarm:
    threshold: float
    confirm: int

    @classmethod
    def calibrate(cls, calib_scores: np.ndarray, cfg: dict) -> ThresholdAlarm:
        """Threshold from clean calibration scores: mean + k*std, or an empirical quantile."""
        if cfg["method"] == "std":
            threshold = calib_scores.mean() + cfg["k"] * calib_scores.std()
        elif cfg["method"] == "quantile":
            threshold = np.quantile(calib_scores, cfg["quantile"])
        else:
            raise ValueError(f"unknown threshold method: {cfg['method']}")
        return cls(float(threshold), cfg["confirm"])

    def _runs(self, scores: np.ndarray) -> np.ndarray:
        """Length of the current run of consecutive above-threshold rows, per row."""
        runs, run = np.zeros(len(scores), dtype=int), 0
        for i, above in enumerate(scores > self.threshold):
            run = run + 1 if above else 0
            runs[i] = run
        return runs

    def alarms(self, scores: np.ndarray) -> np.ndarray:
        """Row indices where an alarm starts: the `confirm`-th consecutive row above threshold (one per run)."""
        return np.flatnonzero(self._runs(scores) == self.confirm)

    def alarm_state(self, scores: np.ndarray) -> np.ndarray:
        """Row indices that are in alarm: at least `confirm` consecutive rows above threshold up to that row."""
        return np.flatnonzero(self._runs(scores) >= self.confirm)
