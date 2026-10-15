"""Load and preprocess the ExtraSensory per-user files.

One file = one user = one federated client; one row = one minute. Per client:

1. sort rows by timestamp
2. keep the same feature columns for every client (drop columns mostly missing across the dataset)
3. drop rows with too many missing features
4. split by time: pre-drift (first 60%) | post-drift (last 40%)
5. inside pre-drift: train | calib | fpr, where calib and fpr are the two time-halves of the last 15%
6. mean imputation and z-scoring with the client's own training statistics

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
CACHE_VERSION = 2  # bump when the Client format or preprocessing code changes
SPLITS = ("train", "calib", "fpr", "post")


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
        return (self.t["post"][-1] - self.t["train"][0]) / 86400


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


def split_bounds(n: int, pre_frac: float, val_frac: float) -> dict[str, tuple[int, int]]:
    """Row ranges [start, end) of each split for a client with n rows."""
    n_pre = round(n * pre_frac)
    n_train = n_pre - round(n_pre * val_frac)
    n_calib_end = n_train + (n_pre - n_train) // 2
    return {"train": (0, n_train), "calib": (n_train, n_calib_end), "fpr": (n_calib_end, n_pre), "post": (n_pre, n)}


def preprocess_user(uuid: str, df: pd.DataFrame, features: list[str], labels: list[str], cfg: dict) -> Client:
    n_raw = len(df)
    df = df[df[features].isna().mean(axis=1) <= cfg["max_row_missing"]].reset_index(drop=True)

    X = df[features].to_numpy(np.float64)
    Y = df[labels].to_numpy(np.float32)
    t = df["timestamp"].to_numpy()
    bounds = split_bounds(len(df), cfg["pre_drift_frac"], cfg["val_frac"])

    lo, hi = bounds["train"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns, handled below
        mean = np.nanmean(X[lo:hi], axis=0)
        std = np.nanstd(X[lo:hi], axis=0)
    unobserved = np.isnan(mean)
    std = np.where(np.isnan(std) | (std < 1e-8), 1.0, std)
    X = (X - np.nan_to_num(mean)) / std
    X = np.nan_to_num(X, nan=0.0)  # mean imputation: the training mean is 0 after z-scoring
    X[:, unobserved] = 0.0         # no training values -> the column carries no information for this client

    sl = {s: slice(*b) for s, b in bounds.items()}
    return Client(
        uuid=uuid,
        X={s: X[sl[s]].astype(np.float32) for s in SPLITS},
        Y={s: Y[sl[s]] for s in SPLITS},
        t={s: t[sl[s]] for s in SPLITS},
        n_raw=n_raw,
        n_dropped=n_raw - len(df),
        observed=~unobserved,
    )


def load_clients(cfg: dict | None = None) -> tuple[dict[str, Client], list[str], list[str]]:
    """Return (clients by UUID, feature names, label names). Cached in data/processed/."""
    cfg = (cfg or load_config())["data"]
    key = hashlib.sha1(json.dumps([CACHE_VERSION, cfg], sort_keys=True).encode()).hexdigest()[:10]
    cache = ROOT / cfg["processed_dir"] / f"clients_{key}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    users = read_raw(ROOT / cfg["raw_dir"])
    features = select_features(users, cfg["max_col_missing"])
    labels = select_labels(users, cfg["min_label_users"])
    clients = {u: preprocess_user(u, df, features, labels, cfg) for u, df in users.items()}
    # Users with every row dropped (e.g. a phone without gyroscope and magnetometer) are excluded.
    clients = {u: c for u, c in clients.items() if c.n_rows > 0}

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump((clients, features, labels), f)
    return clients, features, labels


def summary(clients: dict[str, Client]) -> pd.DataFrame:
    rows = []
    for c in clients.values():
        labeled = {s: int((~np.isnan(c.Y[s])).any(axis=1).sum()) for s in SPLITS}
        rows.append({
            "uuid": c.uuid, "raw_rows": c.n_raw, "dropped": c.n_dropped, "rows": c.n_rows,
            "days": round(c.days, 1), "unobserved_cols": c.n_unobserved,
            **{f"n_{s}": len(c.X[s]) for s in SPLITS},
            "labeled_train": labeled["train"], "labeled_post": labeled["post"],
        })
    return pd.DataFrame(rows).sort_values("labeled_post", ascending=False).reset_index(drop=True)


def main():
    clients, features, labels = load_clients()
    df = summary(clients)
    groups = pd.Series([f.split(":")[0] for f in features]).value_counts()

    pd.set_option("display.width", 200)
    print(df.assign(uuid=df.uuid.str[:8]).to_string())
    excluded = sorted(p.name.split(".")[0][:8] for p in (ROOT / load_config()["data"]["raw_dir"]).glob("*.csv.gz")
                      if p.name.split(".")[0] not in clients)
    print(f"\nclients: {len(clients)} (excluded, no rows left: {excluded})   rows kept: {df.rows.sum()} / {df.raw_rows.sum()} "
          f"({df.dropped.sum()} dropped, {df.dropped.sum() / df.raw_rows.sum():.1%})")
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
