"""Activity classifier: a small multi-label MLP trained with NaN-masked binary cross-entropy."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: list[int], dropout: float):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits, one per label


def masked_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """BCE over reported labels only: NaN (= not reported) is excluded, not treated as 0."""
    mask = ~torch.isnan(y)
    if not mask.any():
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], y[mask])


def train_classifier(X: np.ndarray, Y: np.ndarray, cfg: dict, seed: int) -> MLP:
    """Train a fresh MLP on (X, Y) for a fixed number of epochs."""
    torch.manual_seed(seed)
    model = MLP(X.shape[1], Y.shape[1], cfg["hidden"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    gen = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(cfg["epochs"]):
        for idx in torch.randperm(len(X), generator=gen).split(cfg["batch_size"]):
            opt.zero_grad()
            masked_bce(model(X[idx]), Y[idx]).backward()
            opt.step()
    return model


@torch.no_grad()
def predict_proba(model: MLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X))).numpy()
