"""Federated learning on streaming client data: FedAvg, FedProx, FedConD and FedConD-AE.

Every method shares the same data stream, model, local training and evaluation; they differ only in how the
server combines client models and in the proximal weight lambda of the local objective
    h_k(w_k) = BCE_k(w_k) + lambda / 2 * ||w_k - w||^2   (w = the global model the client received).

Clients receive their data as a stream, B new rows per local round. In every local round, client k
1. receives a global model w,
2. predicts its B new rows with w (stored for test-then-train evaluation; the round's macro F1 is FedConD's score),
3. (FedConD only) asks its drift trigger whether drift happened; if so lambda_k += lambda_step,
4. trains w_k for `local_epochs` on all rows seen so far, minimising h_k,
5. uploads w_k.

Server, by method:
- fedavg / fedprox (synchronous, McMahan et al. 2017 / Li et al. 2020): each round a share `concurrency` of the
  clients with data left is sampled at random; the new global model is the sample-weighted average of their
  models. lambda = 0 (FedAvg) or a fixed mu (FedProx).
- fedcond / fedcond_ae (asynchronous, Chen et al. 2021): a share `concurrency` of the clients trains at the same
  time; each upload is applied at once, w <- w + (n_k / N) * (w_k - w_received); the next request goes to the
  client with the fewest updates. Jobs finish in dispatch order, so a client may start from a global model that
  is a few updates old (no staleness weighting, as in the paper). The two differ only in the drift trigger
  (`trigger_fired`): FedConD's supervised test or the autoencoder. "none" (never fires) and "oracle" (fires at
  the true drift onset) are reference triggers.

Reaction when a trigger fires: "leash" (FedConD: lambda_k += lambda_step) or "recenter" (label-free input
correction, see `recenter`; lambda unchanged).

Before streaming, every method starts from the same warm-up: synchronous FedAvg rounds on the rows available at
the start (lambda = 0).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.data import Client
from src.detectors import Autoencoder, FedConDTest, ThresholdAlarm, reconstruction_error, train_autoencoder
from src.drift import inject, onset_row
from src.metrics import macro_f1, micro_f1
from src.models import MLP, masked_bce

# method -> (server type, drift trigger, reaction)
METHODS = {
    "fedavg": ("sync", None, None),
    "fedprox": ("sync", None, None),
    "fedcond": ("async", "fedcond", "leash"),
    "fedcond_ae": ("async", "autoencoder", "recenter_or_leash"),   # FedConD-AE: label-free alarm + reaction
    "none": ("async", "none", None),
    "oracle": ("async", "oracle", "leash"),
    "oracle_recenter": ("async", "oracle", "recenter"),
    "fedcond_ae_leash": ("async", "autoencoder", "leash"),         # ablation: the alarm with FedConD's reaction
    "fedcond_ae_recenter": ("async", "autoencoder", "recenter"),   # ablation: the alarm with the correction only
}


@dataclass
class FedClient:
    uuid: str
    X: np.ndarray              # rows available at the start, followed by the stream rows (drift injected if any)
    Y: np.ndarray              # labels the client can use (some hidden as NaN when labels are scarce)
    stream_start: int          # first stream row (= number of rows available at the start)
    active_from: int           # first row where trigger decisions are acted on
    onset: int                 # drift onset row (len(X) if no drift; for clean clients: used for evaluation)
    drifted: bool
    ae_alarms: np.ndarray      # rows where an autoencoder alarm starts
    test: FedConDTest
    lam: float
    ae: Autoencoder | None = None                    # the client's autoencoder (judges the "recenter" correction)
    ae_threshold: float = float("nan")
    pos: int = 0               # rows seen so far
    updates: int = 0
    fired: list[int] = field(default_factory=list)   # rows where the trigger took effect (round's last row)
    scored: int = 0                                  # rounds with a defined FedConD score (enough labels)
    Y_true: np.ndarray | None = None                 # all labels, for evaluation only (None: same as Y)
    correct_from: int | None = None                  # "recenter" reaction: row of the latest alarm
    X_orig: np.ndarray | None = None                 # "recenter" reaction: uncorrected rows
    corr_base: np.ndarray | None = None              # "recenter" reaction: correction frozen at earlier alarms
    corrections: list[bool] = field(default_factory=list)   # "recenter": per earlier alarm, correction accepted?
    latest_ok: bool = False                          # "recenter": correction of the latest alarm accepted so far
    P: np.ndarray | None = None                      # test-then-train predictions for the stream rows

    def __post_init__(self):
        self.pos = self.stream_start
        self.P = np.full((len(self.X) - self.stream_start, self.Y.shape[1]), np.nan, dtype=np.float32)


def _ae_alarms(X_fit: np.ndarray, X_calib: np.ndarray, X_stream: np.ndarray, offset: int, cfg: dict,
               seed: int) -> dict:
    """Train the client's autoencoder, calibrate its threshold and find the alarm rows in the stream."""
    ae = train_autoencoder(X_fit, cfg["autoencoder"], seed)
    alarm = ThresholdAlarm.calibrate(reconstruction_error(ae, X_calib), cfg["alarm"])
    return {"ae": ae, "ae_threshold": alarm.threshold, "ae_alarms": offset + alarm.alarms(reconstruction_error(ae, X_stream))}


NO_AE = {"ae": None, "ae_threshold": float("nan"), "ae_alarms": np.array([], dtype=int)}


def make_fed_client(c: Client, drifted: bool, drift_cols: np.ndarray, cfg: dict, seed: int,
                    with_ae: bool = True, label_rate: float = 1.0, label_seed: int = 0) -> FedClient:
    """Drift scheme: start = training rows; stream = calibration day + FPR day + post-drift (drift injected).

    Triggers are acted on from the FPR day; during the calibration day FedConD's score queue fills up and the
    autoencoder threshold is set on these rows. label_rate < 1: training rows keep all labels, each stream row
    keeps its labels with probability label_rate (evaluation uses the true labels).
    """
    dcfg, fcfg = cfg["drift"], cfg["federated"]
    X_post = c.X["post"]
    onset_post = onset_row(len(X_post), dcfg["onset_frac"])
    if drifted:
        X_post = inject(X_post, drift_cols, onset_post, fcfg["magnitude"], fcfg["kind"], dcfg["gradual_width"],
                        cfg["data"]["clip_z"])
    X = np.vstack([c.X["train"], c.X["calib"], c.X["fpr"], X_post])
    Y = np.vstack([c.Y["train"], c.Y["calib"], c.Y["fpr"], c.Y["post"]])
    n_train, n_calib = len(c.X["train"]), len(c.X["calib"])
    pre = n_train + n_calib + len(c.X["fpr"])
    ae = _ae_alarms(c.X["train"], c.X["calib"], X[n_train:], n_train, cfg, seed) if with_ae else NO_AE
    ft = cfg["fedcond_test"]
    Y_used = hide_labels(Y, n_train, label_rate, label_seed) if label_rate < 1 else Y
    return FedClient(uuid=c.uuid, X=X, Y=Y_used, stream_start=n_train, active_from=n_train + n_calib,
                     onset=pre + onset_post, drifted=drifted, **ae,
                     test=FedConDTest(ft["queue_size"], ft["alpha"]), lam=fcfg["lambda_base"], Y_true=Y)


def hide_labels(Y: np.ndarray, start: int, rate: float, seed: int) -> np.ndarray:
    """Copy of Y where each row from `start` on keeps its labels with probability `rate` (else all NaN).

    One uniform number per row decides, so the rows kept at a lower rate are a subset of those kept at a higher
    rate, and the same rows are hidden for every method.
    """
    Y = Y.copy()
    u = np.random.default_rng(seed).random(len(Y))
    hide = (np.arange(len(Y)) >= start) & (u >= rate)
    Y[hide] = np.nan
    return Y


def make_paper_client(c: Client, cfg: dict, seed: int, with_ae: bool = True, label_rate: float = 1.0,
                      label_seed: int = 0) -> FedClient:
    """Paper scheme (no injected drift): the first `initial_frac` of the training rows is available at the start,
    the rest streams in. The autoencoder is trained on the first 2/3 of the initial rows and calibrated on the
    last 1/3. Validation and test rows are not part of the stream.

    label_rate < 1: the starting rows keep all their labels; each stream row keeps its labels with probability
    label_rate (hidden from training and from FedConD's score; evaluation uses the true labels).
    """
    X, Y = c.X["train"], c.Y["train"]
    n0 = round(len(X) * cfg["paper_setup"]["initial_frac"])
    if label_rate < 1:
        Y = hide_labels(Y, n0, label_rate, label_seed)
    n_fit = round(n0 * 2 / 3)
    ae = _ae_alarms(X[:n_fit], X[n_fit:n0], X[n0:], n0, cfg, seed) if with_ae else NO_AE
    ft = cfg["fedcond_test"]
    return FedClient(uuid=c.uuid, X=X, Y=Y, stream_start=n0, active_from=n0, onset=len(X), drifted=False,
                     **ae, test=FedConDTest(ft["queue_size"], ft["alpha"]), lam=cfg["federated"]["lambda_base"])


def recenter_shift(fc: FedClient, stop: int, window: int, min_rows: int) -> np.ndarray | None:
    """Estimated input shift at the latest alarm: mean of the first `window` rows since the alarm (as many as have
    arrived, at least `min_rows`) minus mean of the `window` rows before it; None while too few rows exist, and
    None if the autoencoder does not confirm it (see `shift_confirmed`)."""
    a = fc.correct_from
    after = fc.X_orig[a:min(a + window, stop)]
    if len(after) < min_rows:
        return None
    shift = after.mean(axis=0) - fc.X_orig[max(0, a - window):a].mean(axis=0)
    return shift if shift_confirmed(fc, after, shift) else None


def shift_confirmed(fc: FedClient, after: np.ndarray, shift: np.ndarray) -> bool:
    """The autoencoder judges the correction, without labels: the rows since the alarm must look abnormal as a
    whole (mean reconstruction error above the alarm threshold, after the correction frozen so far) and the new
    shift must make them look more normal (lower error). A false alarm on an ordinary day fails the first test,
    so the day-to-day change in activity mix is never subtracted. Without an autoencoder (oracle) always true."""
    if fc.ae is None:
        return True
    base = after - fc.corr_base
    err_now = reconstruction_error(fc.ae, base).mean()
    return bool(err_now > fc.ae_threshold and reconstruction_error(fc.ae, base - shift).mean() < err_now)


def recenter(fc: FedClient, stop: int, window: int, min_rows: int, clip: float) -> None:
    """Label-free input correction after a drift alarm (in place, rows correct_from..stop-1).

    The shift measured at the latest alarm (`recenter_shift`) is added to the correction frozen at earlier alarms
    (`corr_base`) and subtracted from every row since the latest alarm. The estimate uses the first `window` rows
    after the alarm and then stays fixed (a full day on each side keeps the daily activity mix comparable). It is
    frozen only once its window is complete (see the alarm handling in `run`): a later alarm inside the window
    either supersedes a so-far-rejected one or is ignored while a confirmed correction is still being measured.
    """
    a = fc.correct_from
    shift = recenter_shift(fc, stop, window, min_rows)
    fc.latest_ok = shift is not None
    total = fc.corr_base + (0.0 if shift is None else shift)
    fc.X[a:stop] = np.clip(fc.X_orig[a:stop] - total, -clip, clip)


def trigger_fired(trigger: str, fc: FedClient, rows: slice, score: float) -> bool:
    """The only part that differs between FedConD and FedConD-AE."""
    if trigger == "fedcond":
        return fc.test.update(score)
    if trigger == "autoencoder":
        return bool(((fc.ae_alarms >= rows.start) & (fc.ae_alarms < rows.stop)).any())
    if trigger == "oracle":
        return fc.drifted and rows.start <= fc.onset < rows.stop
    return False


def local_train(model: MLP, w: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, lam: float, epochs: int,
                ccfg: dict, gen: torch.Generator) -> torch.Tensor:
    """Start from global weights w, minimise masked BCE + lam/2 * ||w_k - w||^2; return the new weights."""
    vector_to_parameters(w.clone(), model.parameters())  # copy: the parameters must not alias (and modify) w
    opt = torch.optim.Adam(model.parameters(), lr=ccfg["lr"])
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=gen).split(ccfg["batch_size"]):
            opt.zero_grad()
            loss = masked_bce(model(X[idx]), Y[idx])
            if lam > 0:
                loss = loss + lam / 2 * (parameters_to_vector(model.parameters()) - w).pow(2).sum()
            loss.backward()
            opt.step()
    return parameters_to_vector(model.parameters()).detach()


@torch.no_grad()
def predict(model: MLP, w: torch.Tensor, X: np.ndarray) -> np.ndarray:
    vector_to_parameters(w.clone(), model.parameters())
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def _weighted_average(models: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    total = float(sum(weights))
    return sum(m * (n / total) for m, n in zip(models, weights))


def run(clients: list[FedClient], method: str, cfg: dict, seed: int) -> tuple[torch.Tensor, MLP]:
    """Simulate `method` on the clients' streams. Mutates the clients (predictions, fires, lambda).

    Returns the final global weights and a model object to evaluate them with (see `predict`).
    """
    server, trigger, reaction = METHODS[method]
    fcfg, ccfg = cfg["federated"], cfg["classifier"]
    B, thr = cfg["fedcond_test"]["batch_rows"], ccfg["threshold"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(clients[0].X.shape[1], clients[0].Y.shape[1], ccfg["hidden"], ccfg["dropout"])
    w = parameters_to_vector(model.parameters()).detach()
    data = [(torch.from_numpy(fc.X), torch.from_numpy(fc.Y)) for fc in clients]
    n_parallel = max(1, round(fcfg["concurrency"] * len(clients)))

    # Warm-up (identical for all methods): synchronous FedAvg rounds on the rows available at the start.
    for _ in range(fcfg["warmup_rounds"]):
        w = _weighted_average(
            [local_train(model, w, X[:fc.stream_start], Y[:fc.stream_start], 0.0, 1, ccfg, gen)
             for fc, (X, Y) in zip(clients, data)],
            [fc.stream_start for fc in clients])

    def has_data(fc):
        return fc.pos + B <= len(fc.X)

    def local_round(i: int, w_recv: torch.Tensor, lam: float) -> torch.Tensor:
        """Steps 2-5 of a local round (trigger handling included for async methods)."""
        fc, (X, Y) = clients[i], data[i]
        rows = slice(fc.pos, fc.pos + B)
        if fc.correct_from is not None:  # "recenter": correct the new rows (X shares memory with fc.X)
            recenter(fc, rows.stop, fcfg["recenter_window"], fcfg["recenter_min_rows"], cfg["data"]["clip_z"])
        P = predict(model, w_recv, fc.X[rows])
        fc.P[rows.start - fc.stream_start:rows.stop - fc.stream_start] = P
        if trigger is not None:
            score = macro_f1(fc.Y[rows], P, thr)
            fc.scored += not np.isnan(score)
            fired = trigger_fired(trigger, fc, rows, score)
            if fired and rows.start >= fc.active_from:
                fc.fired.append(rows.stop - 1)
                if reaction == "leash":
                    fc.lam += fcfg["lambda_step"]
                elif reaction in ("recenter", "recenter_or_leash"):
                    leash = fcfg["lambda_step"] if reaction == "recenter_or_leash" else 0.0
                    if fc.correct_from is None:
                        fc.X_orig, fc.corr_base = fc.X.copy(), np.zeros(fc.X.shape[1], dtype=fc.X.dtype)
                        fc.correct_from, fc.lam = rows.start, fc.lam + leash
                    elif rows.stop - fc.correct_from >= fcfg["recenter_window"]:
                        # the previous alarm's window is complete, its verdict final: freeze it, measure anew
                        shift = recenter_shift(fc, rows.stop, fcfg["recenter_window"], fcfg["recenter_min_rows"])
                        fc.corrections.append(shift is not None)
                        fc.corr_base = fc.corr_base + (0.0 if shift is None else shift)
                        if shift is not None:
                            fc.lam -= leash   # explained by an input shift: FedConD's leash is taken back
                        fc.correct_from, fc.lam = rows.start, fc.lam + leash
                    elif not fc.latest_ok:
                        # the previous alarm looks false so far: this alarm supersedes it (nothing frozen)
                        fc.correct_from, fc.lam = rows.start, fc.lam + leash
                    # else: a confirmed correction is still being measured; this alarm is the same episode
            lam = fc.lam
        fc.pos = rows.stop
        fc.updates += 1
        return local_train(model, w_recv, X[:fc.pos], Y[:fc.pos], lam, fcfg["local_epochs"], ccfg, gen)

    if server == "sync":
        mu = cfg["paper_setup"]["fedprox_mu"] if method == "fedprox" else 0.0
        while avail := [i for i, fc in enumerate(clients) if has_data(fc)]:
            chosen = rng.choice(avail, min(n_parallel, len(avail)), replace=False)
            w = _weighted_average([local_round(i, w, mu) for i in chosen], [clients[i].pos for i in chosen])
        return w, model

    def next_client(busy):
        free = [i for i, fc in enumerate(clients) if i not in busy and has_data(fc)]
        return min(free, key=lambda i: clients[i].updates) if free else None

    in_flight: deque[tuple[int, torch.Tensor]] = deque()
    for _ in range(n_parallel):
        i = next_client({j for j, _ in in_flight})
        if i is not None:
            in_flight.append((i, w.clone()))
    while in_flight:
        i, w_recv = in_flight.popleft()
        w_k = local_round(i, w_recv, clients[i].lam)
        w = w + (clients[i].pos / sum(c.pos for c in clients)) * (w_k - w_recv)
        j = next_client({k for k, _ in in_flight})
        if j is not None:
            in_flight.append((j, w.clone()))
    return w, model


def evaluate(fc: FedClient, thr: float) -> dict:
    """Test-then-train macro (and micro) F1 of the received global model: before (active part to onset) and after
    the onset."""
    seen = fc.pos - fc.stream_start
    Y_all = fc.Y if fc.Y_true is None else fc.Y_true
    P, Y = fc.P[:seen], Y_all[fc.stream_start:fc.pos]
    a, o = fc.active_from - fc.stream_start, fc.onset - fc.stream_start
    first_fire_after = [r for r in fc.fired if r >= fc.onset]
    corrections = fc.corrections + ([fc.latest_ok] if fc.correct_from is not None else [])
    return {
        "uuid": fc.uuid[:8], "drifted": fc.drifted,
        "f1_before": macro_f1(Y[a:o], P[a:o], thr), "f1_after": macro_f1(Y[o:], P[o:], thr),
        "micro_before": micro_f1(Y[a:o], P[a:o], thr), "micro_after": micro_f1(Y[o:], P[o:], thr),
        "fires_before": sum(r < fc.onset for r in fc.fired), "fires_after": len(first_fire_after),
        "lag": (first_fire_after[0] - fc.onset) if first_fire_after else np.nan,
        "final_lambda": fc.lam, "updates": fc.updates,
        "corrections_accepted": int(sum(corrections)), "corrections_rejected": int(len(corrections) - sum(corrections)),
    }
