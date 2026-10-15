"""Run the drift detectors on one client's data stream; measure detection lag and false alarms.

Stream = the client's FPR rows followed by its post-drift rows, with drift injected into the post-drift part.
The calibration rows (just before the FPR rows) set the label-free thresholds and warm up FedConD's score
queue, so every detector starts the stream with the same knowledge of "normal".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data import Client
from src.detectors import FedConDTest, ThresholdAlarm, prediction_entropy, reconstruction_error
from src.drift import inject, onset_row
from src.metrics import macro_f1
from src.models import predict_proba

DETECTORS = ("autoencoder", "entropy", "fedcond")


@dataclass
class DetectorRun:
    scores: np.ndarray      # per row (autoencoder, entropy) or per round (fedcond)
    score_rows: np.ndarray  # stream row of each score; for rounds, the round's last row
    threshold: float        # NaN for fedcond (its test has no fixed threshold)
    alarm_rows: np.ndarray  # stream rows where an alarm starts (for plotting)
    state_rows: np.ndarray  # stream rows in alarm state; a round with any of them answers "drift"


def drift_columns(features: list[str], c: Client, prefixes: list[str]) -> np.ndarray:
    """Indices of the drifted feature columns that this client actually has."""
    return np.flatnonzero(np.array([f.startswith(tuple(prefixes)) for f in features]) & c.observed)


def make_stream(c: Client, cols: np.ndarray, kind: str, magnitude: float, dcfg: dict, clip: float):
    """Return (X, Y, onset): FPR rows + post-drift rows with drift injected; onset is a stream row index."""
    onset_post = onset_row(len(c.X["post"]), dcfg["onset_frac"])
    X_post = inject(c.X["post"], cols, onset_post, magnitude, kind, dcfg["gradual_width"], clip)
    X = np.vstack([c.X["fpr"], X_post])
    Y = np.vstack([c.Y["fpr"], c.Y["post"]])
    return X, Y, len(c.X["fpr"]) + onset_post


def _rounds(n_rows: int, size: int) -> list[slice]:
    return [slice(i, i + size) for i in range(0, n_rows - size + 1, size)]


def run_detectors(c: Client, X: np.ndarray, Y: np.ndarray, clf, ae, cfg: dict) -> dict[str, DetectorRun]:
    runs = {}
    rows = np.arange(len(X))
    for name, score in (("autoencoder", lambda Z: reconstruction_error(ae, Z)),
                        ("entropy", lambda Z: prediction_entropy(predict_proba(clf, Z)))):
        alarm = ThresholdAlarm.calibrate(score(c.X["calib"]), cfg["alarm"])
        s = score(X)
        runs[name] = DetectorRun(s, rows, alarm.threshold, alarm.alarms(s), alarm.alarm_state(s))

    fcfg, thr = cfg["fedcond_test"], cfg["classifier"]["threshold"]
    test = FedConDTest(fcfg["queue_size"], fcfg["alpha"])
    P_calib = predict_proba(clf, c.X["calib"])
    for r in _rounds(len(P_calib), fcfg["batch_rows"]):  # warm-up: fill the queue, alarms ignored
        test.update(macro_f1(c.Y["calib"][r], P_calib[r], thr))
    P = predict_proba(clf, X)
    rounds = _rounds(len(X), fcfg["batch_rows"])
    s = np.array([macro_f1(Y[r], P[r], thr) for r in rounds])
    ends = np.array([r.stop - 1 for r in rounds])
    flags = np.array([test.update(v) for v in s], dtype=bool)
    runs["fedcond"] = DetectorRun(s, ends, float("nan"), ends[flags], ends[flags])
    return runs


def evaluate_run(run: DetectorRun, onset: int, n_fpr: int, n_rows: int, round_rows: int) -> dict:
    """Detection lag and false-alarm rates of one detector run, on a common per-round scale.

    The stream is cut into consecutive rounds of `round_rows` rows. In every round each detector answers
    "drift: yes/no": yes if any of its alarm-state rows falls in the round. The answer takes effect at the
    round's end, as in the federated loop.
    - lag: end of the round of the first alarm-state row at or after onset, minus onset (rows); NaN if never.
    - fa_*: share of fully clean rounds (ending before onset) answered "yes"; split into rounds inside the
      FPR rows, rounds in the clean part of the post-drift segment, and all clean rounds.
    """
    after = run.state_rows[run.state_rows >= onset]
    lag = float((after[0] // round_rows + 1) * round_rows - 1 - onset) if len(after) else float("nan")

    n_rounds = n_rows // round_rows
    starts = np.arange(n_rounds) * round_rows
    ends = starts + round_rows - 1
    flagged = np.zeros(n_rounds, dtype=bool)
    idx = np.unique(run.state_rows // round_rows)
    flagged[idx[idx < n_rounds]] = True

    def rate(mask):
        return float(flagged[mask].mean()) if mask.any() else float("nan")

    clean = ends < onset
    return {
        "lag": lag,
        "fa_fpr": rate(clean & (ends < n_fpr)),
        "fa_post": rate(clean & (starts >= n_fpr)),
        "fa_all": rate(clean),
        "clean_rounds": int(clean.sum()),
    }
