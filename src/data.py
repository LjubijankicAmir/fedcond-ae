"""Load and preprocess the ExtraSensory per-user files.

One file = one user = one federated client; one row = one minute. Per client:

1. sort rows by timestamp
2. keep the same feature columns for every client (drop columns mostly missing across the dataset)
3. drop rows with too many missing features
4. split by time: pre-drift (first 60% of rows) | post-drift (last 40%)
5. inside pre-drift, by timestamp: train | calib (one full day) | fpr (the last full day before post-drift)
6. mean imputation and z-scoring with the client's own training statistics, then clipping to +-clip_z

Clients without enough data for this split (too little training time, or almost no rows recorded in the
calibration or FPR day) are excluded.

Alternative "paper" schemes (experiment 1, close to the FedConD paper): per client 60/20/20 train/val/test,
either by time ("paper_time") or random ("paper_random", each part kept in time order). Steps 1-3 and 6 are the
same; only clients without any usable rows are excluded.

Run as `python -m src.data` to print a per-user summary.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CACHE_VERSION = 5  # bump when the Client format or preprocessing code changes
SPLITS = ("train", "calib", "fpr", "post")  # splits of the default "drift" scheme
SCHEMES = ("drift", "paper_time", "paper_random")


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class Client:
    uuid: str
    X: dict[str, np.ndarray]  # split -> (rows, features) float32, imputed and z-scored
    Y: dict[str, np.ndarray]  # split -> (rows, labels) float32; 1 / 0 / NaN (= not reported)
    t: dict[str, np.ndarray]  # split -> Unix timestamps of the rows
    n_raw: int                # rows before cleaning
    n_dropped: int            # rows dropped for too many missing features
    observed: np.ndarray      # (features,) bool; False = no training values for this client (column set to 0)

    @property
    def n_unobserved(self) -> int:
        return int((~self.observed).sum())

    @property
    def n_rows(self) -> int:
        return sum(len(x) for x in self.X.values())

    @property
    def days(self) -> float:
        t = np.concatenate(list(self.t.values()))
        return (t.max() - t.min()) / 86400

    @property
    def train_hours(self) -> float:
        t = self.t["train"]
        return (t[-1] - t[0]) / 3600 if len(t) else 0.0


def read_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Read every user file, sorted by timestamp. Keys are user UUIDs."""
    users = {}
    for f in sorted(Path(raw_dir).glob("*.features_labels.csv.gz")):
        df = pd.read_csv(f)
        users[f.name.split(".")[0]] = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if not users:
        raise FileNotFoundError(f"no *.features_labels.csv.gz files in {raw_dir}")
    return users


def select_features(users: dict[str, pd.DataFrame], max_missing: float) -> list[str]:
    """Feature columns whose missing rate over all users' rows pooled is <= max_missing."""
    df0 = next(iter(users.values()))
    cols = [c for c in df0.columns if c not in ("timestamp", "label_source") and not c.startswith("label:")]
    missing = pd.concat([df[cols] for df in users.values()]).isna().mean()
    return [c for c in cols if missing[c] <= max_missing]


def select_labels(users: dict[str, pd.DataFrame], min_users: int) -> list[str]:
    """Label columns that are positive at least once for >= min_users users."""
    cols = [c for c in next(iter(users.values())).columns if c.startswith("label:")]
    n_users = sum((df[cols] == 1).any() for df in users.values())
    return [c for c in cols if n_users[c] >= min_users]


def split_bounds(t: np.ndarray, pre_frac: float, calib_hours: float, fpr_hours: float) -> dict[str, tuple[int, int]]:
    """Row ranges [start, end) of each split, given the client's sorted timestamps.

    Pre-drift = the first round(n * pre_frac) rows. Inside it, by time: fpr = rows in the last `fpr_hours`,
    calib = rows in the `calib_hours` before that, train = everything earlier.
    """
    n_pre = round(len(t) * pre_frac)
    t_pre, end = t[:n_pre], t[n_pre - 1]
    fpr_start = int(np.searchsorted(t_pre, end - fpr_hours * 3600, side="right"))
    calib_start = int(np.searchsorted(t_pre, end - (fpr_hours + calib_hours) * 3600, side="right"))
    return {"train": (0, calib_start), "calib": (calib_start, fpr_start), "fpr": (fpr_start, n_pre),
            "post": (n_pre, len(t))}


def paper_split(n: int, scheme: str, cfg: dict) -> dict[str, np.ndarray]:
    """Row indices of a per-client train/val/test split (60/20/20 by default), each part in time order."""
    if scheme == "paper_time":
        idx = np.arange(n)
    elif scheme == "paper_random":
        idx = np.random.default_rng(cfg["paper_split_seed"]).permutation(n)
    else:
        raise ValueError(f"unknown split scheme: {scheme}")
    a = round(n * cfg["paper_train_frac"])
    b = a + round(n * cfg["paper_val_frac"])
    return {"train": np.sort(idx[:a]), "val": np.sort(idx[a:b]), "test": np.sort(idx[b:])}


def preprocess_user(uuid: str, df: pd.DataFrame, features: list[str], labels: list[str], cfg: dict,
                    scheme: str = "drift") -> Client | None:
    """Clean, split and scale one user's data; None if no rows survive the missing-value filter."""
    n_raw = len(df)
    df = df[df[features].isna().mean(axis=1) <= cfg["max_row_missing"]].reset_index(drop=True)
    if df.empty:
        return None

    X = df[features].to_numpy(np.float64)
    Y = df[labels].to_numpy(np.float32)
    t = df["timestamp"].to_numpy()
    if scheme == "drift":
        bounds = split_bounds(t, cfg["pre_drift_frac"], cfg["calib_hours"], cfg["fpr_hours"])
        idx = {s: np.arange(*b) for s, b in bounds.items()}
    else:
        idx = paper_split(len(df), scheme, cfg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns, handled below
        mean = np.nanmean(X[idx["train"]], axis=0)
        std = np.nanstd(X[idx["train"]], axis=0)
    unobserved = np.isnan(mean)
    std = np.where(np.isnan(std) | (std < 1e-8), 1.0, std)
    X = (X - np.nan_to_num(mean)) / std
    X = np.nan_to_num(X, nan=0.0)  # mean imputation: the training mean is 0 after z-scoring
    X[:, unobserved] = 0.0         # no training values -> the column carries no information for this client
    X = np.clip(X, -cfg["clip_z"], cfg["clip_z"])

    return Client(
        uuid=uuid,
        X={s: X[i].astype(np.float32) for s, i in idx.items()},
        Y={s: Y[i] for s, i in idx.items()},
        t={s: t[i] for s, i in idx.items()},
        n_raw=n_raw,
        n_dropped=n_raw - len(df),
        observed=~unobserved,
    )


def exclusion_reason(c: Client | None, cfg: dict, scheme: str = "drift") -> str | None:
    """Why a client cannot be used, or None if it can."""
    if c is None:
        return "no rows left after the missing-value filter"
    if scheme != "drift":
        return None
    if c.train_hours < cfg["min_train_hours"]:
        return f"training data covers only {c.train_hours:.0f} h"
    for s in ("calib", "fpr"):
        if len(c.X[s]) < cfg["min_window_rows"]:
            return f"only {len(c.X[s])} rows in the {s} day"
    return None


def load_all(cfg: dict | None = None, scheme: str = "drift"
             ) -> tuple[dict[str, Client], dict[str, str], list[str], list[str]]:
    """Return (usable clients by UUID, {excluded UUID: reason}, feature names, label names).

    Cached in data/processed/, keyed by the data config, the split scheme and CACHE_VERSION.
    """
    cfg = (cfg or load_config())["data"]
    key = hashlib.sha1(json.dumps([CACHE_VERSION, scheme, cfg], sort_keys=True).encode()).hexdigest()[:10]
    cache = ROOT / cfg["processed_dir"] / f"clients_{key}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    users = read_raw(ROOT / cfg["raw_dir"])
    features = select_features(users, cfg["max_col_missing"])
    labels = select_labels(users, cfg["min_label_users"])
    clients, excluded = {}, {}
    for u, df in users.items():
        c = preprocess_user(u, df, features, labels, cfg, scheme)
        reason = exclusion_reason(c, cfg, scheme)
        if reason:
            excluded[u] = reason
        else:
            clients[u] = c

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump((clients, excluded, features, labels), f)
    return clients, excluded, features, labels


def load_clients(cfg: dict | None = None, scheme: str = "drift") -> tuple[dict[str, Client], list[str], list[str]]:
    """Return (usable clients by UUID, feature names, label names)."""
    clients, _, features, labels = load_all(cfg, scheme)
    return clients, features, labels


def summary(clients: dict[str, Client]) -> pd.DataFrame:
    rows = []
    for c in clients.values():
        labeled = {s: int((~np.isnan(c.Y[s])).any(axis=1).sum()) for s in SPLITS}
        rows.append({
            "uuid": c.uuid, "raw_rows": c.n_raw, "dropped": c.n_dropped, "rows": c.n_rows,
            "days": round(c.days, 1), "train_hours": round(c.train_hours), "unobserved_cols": c.n_unobserved,
            **{f"n_{s}": len(c.X[s]) for s in SPLITS},
            "labeled_train": labeled["train"], "labeled_post": labeled["post"],
        })
    return pd.DataFrame(rows).sort_values("labeled_post", ascending=False).reset_index(drop=True)


def main():
    clients, excluded, features, labels = load_all()
    df = summary(clients)
    groups = pd.Series([f.split(":")[0] for f in features]).value_counts()

    pd.set_option("display.width", 200)
    print(df.assign(uuid=df.uuid.str[:8]).to_string())
    print(f"\nclients: {len(clients)}   rows kept: {df.rows.sum()} / {df.raw_rows.sum()} "
          f"({df.dropped.sum()} dropped, {df.dropped.sum() / df.raw_rows.sum():.1%})")
    print(f"excluded ({len(excluded)}):")
    for u, reason in sorted(excluded.items()):
        print(f"  {u[:8]}: {reason}")
    print(f"features: {len(features)}  ->  {groups.to_dict()}")
    print(f"labels ({len(labels)}): {[l.removeprefix('label:') for l in labels]}")
    print(f"days covered: {df.days.min()}-{df.days.max()} (median {df.days.median()})")
    print(f"\nmost labeled post-drift rows (first-test candidates): {df.uuid.head(5).str[:8].tolist()}")

    out = ROOT / "results" / "data_summary.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    # Import through the package so cached pickles reference src.data.Client, not __main__.Client.
    from src.data import main
    main()
