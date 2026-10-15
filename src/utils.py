"""Seeding and result saving."""
from __future__ import annotations

import datetime
import random

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import ROOT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def result_stem(name: str, cfg: dict) -> str:
    """Path stem results/<name>_<date>; also writes the config it came from as <stem>.config.yaml."""
    stem = ROOT / "results" / f"{name}_{datetime.date.today():%Y%m%d}"
    stem.parent.mkdir(exist_ok=True)
    with open(f"{stem}.config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(stem)


def save_results(df: pd.DataFrame, name: str, cfg: dict) -> str:
    """Write results/<name>_<date>.csv plus its config."""
    path = f"{result_stem(name, cfg)}.csv"
    df.to_csv(path, index=False)
    return path
