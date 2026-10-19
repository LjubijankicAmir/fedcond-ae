"""Asynchronous federated learning with drift-triggered proximal regularisation (FedConD, Chen et al. 2021).

Clients receive their data as a stream, B new rows per local round. In a local round, client k
1. receives the current global model w,
2. predicts its B new rows with w (prequential evaluation; the round's macro F1 is FedConD's score),
3. asks its drift trigger whether drift happened; if so lambda_k += lambda_step,
4. trains w_k for `local_epochs` on all rows seen so far, minimising BCE + lambda_k / 2 * ||w_k - w||^2,
5. uploads w_k; the server applies w <- w + (n_k / N) * (w_k - w).

FedConD and FedConD-AE share all of this; only the trigger differs (see `trigger_fired`).

Asynchrony: a share `concurrency` of the clients trains at the same time. Jobs finish in dispatch order and each
finished client is replaced by the client with the fewest updates so far, so a client may start from a global
model that is already a few updates old (no staleness weighting, as in the paper).

Stream per client: calibration day + FPR day + post-drift rows. During the calibration day triggers are not
acted on (FedConD's score queue fills up; the AE threshold was set on these rows).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.data import Client
from src.detectors import FedConDTest, ThresholdAlarm, reconstruction_error, train_autoencoder
from src.drift import inject, onset_row
from src.metrics import macro_f1
from src.models import MLP, masked_bce

TRIGGERS = ("none", "fedcond", "autoencoder", "oracle")


@dataclass
class FedClient:
    uuid: str
    X: np.ndarray              # training rows followed by the stream rows (drift injected if `drifted`)
    Y: np.ndarray
    stream_start: int          # first stream row (= number of training rows)
    active_from: int           # first row where trigger decisions are acted on (start of the FPR day)
    onset: int                 # drift onset row (for clean clients: same position, used for evaluation)
    drifted: bool
    ae_alarms: np.ndarray      # rows where an autoencoder alarm starts
    test: FedConDTest
    lam: float
    pos: int = 0               # rows seen so far
    updates: int = 0
    fired: list[int] = field(default_factory=list)   # rows where the trigger took effect (round's last row)
    P: np.ndarray | None = None                      # prequential predictions for the stream rows


def make_fed_client(c: Client, drifted: bool, drift_cols: np.ndarray, cfg: dict, seed: int) -> FedClient:
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

    ae = train_autoencoder(c.X["train"], cfg["autoencoder"], seed)
    alarm = ThresholdAlarm.calibrate(reconstruction_error(ae, c.X["calib"]), cfg["alarm"])
    ae_alarms = n_train + alarm.alarms(reconstruction_error(ae, X[n_train:]))

    ft = cfg["fedcond_test"]
    return FedClient(uuid=c.uuid, X=X, Y=Y, stream_start=n_train, active_from=n_train + n_calib,
                     onset=pre + onset_post, drifted=drifted, ae_alarms=ae_alarms,
                     test=FedConDTest(ft["queue_size"], ft["alpha"]), lam=fcfg["lambda_base"], pos=n_train,
                     P=np.full((len(X) - n_train, Y.shape[1]), np.nan, dtype=np.float32))


def trigger_fired(trigger: str, fc: FedClient, rows: slice, score: float) -> bool:
    """The only part that differs between the compared methods."""
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
            prox = (parameters_to_vector(model.parameters()) - w).pow(2).sum()
            (masked_bce(model(X[idx]), Y[idx]) + lam / 2 * prox).backward()
            opt.step()
    return parameters_to_vector(model.parameters()).detach()


@torch.no_grad()
def predict(model: MLP, w: torch.Tensor, X: np.ndarray) -> np.ndarray:
    vector_to_parameters(w.clone(), model.parameters())
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def run(clients: list[FedClient], trigger: str, cfg: dict, seed: int) -> list[FedClient]:
    """Simulate FedConD with the given trigger. Mutates and returns the clients (predictions, fires, lambda)."""
    fcfg, ccfg = cfg["federated"], cfg["classifier"]
    B, thr = cfg["fedcond_test"]["batch_rows"], ccfg["threshold"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = MLP(clients[0].X.shape[1], clients[0].Y.shape[1], ccfg["hidden"], ccfg["dropout"])
    w = parameters_to_vector(model.parameters()).detach()
    data = [(torch.from_numpy(fc.X), torch.from_numpy(fc.Y)) for fc in clients]

    # Warm-up: synchronous FedAvg rounds on the training rows, so streaming starts from a trained global model.
    n_train = np.array([fc.stream_start for fc in clients], dtype=np.float64)
    for _ in range(fcfg["warmup_rounds"]):
        updates = [local_train(model, w, X[:fc.stream_start], Y[:fc.stream_start], fc.lam, 1, ccfg, gen)
                   for fc, (X, Y) in zip(clients, data)]
        w = sum(u * (n / n_train.sum()) for u, n in zip(updates, n_train))

    def has_data(fc):
        return fc.pos + B <= len(fc.X)

    def next_client(busy):
        free = [i for i, fc in enumerate(clients) if i not in busy and has_data(fc)]
        return min(free, key=lambda i: clients[i].updates) if free else None

    in_flight: deque[tuple[int, torch.Tensor]] = deque()
    n_parallel = max(1, round(fcfg["concurrency"] * len(clients)))
    for _ in range(n_parallel):
        i = next_client({j for j, _ in in_flight})
        if i is not None:
            in_flight.append((i, w.clone()))

    while in_flight:
        i, w_recv = in_flight.popleft()
        fc, (X, Y) = clients[i], data[i]
        rows = slice(fc.pos, fc.pos + B)

        P = predict(model, w_recv, fc.X[rows])
        fc.P[rows.start - fc.stream_start:rows.stop - fc.stream_start] = P
        fired = trigger_fired(trigger, fc, rows, macro_f1(fc.Y[rows], P, thr))
        if fired and rows.start >= fc.active_from:
            fc.lam += fcfg["lambda_step"]
            fc.fired.append(rows.stop - 1)

        fc.pos = rows.stop
        w_k = local_train(model, w_recv, X[:fc.pos], Y[:fc.pos], fc.lam, fcfg["local_epochs"], ccfg, gen)
        n_total = sum(c.pos for c in clients)
        w = w + (fc.pos / n_total) * (w_k - w_recv)
        fc.updates += 1

        j = next_client({k for k, _ in in_flight})
        if j is not None:
            in_flight.append((j, w.clone()))
    return clients


def evaluate(fc: FedClient, thr: float) -> dict:
    """Prequential macro F1 of the received global model: clean part (FPR day to onset) and after the onset."""
    seen = fc.pos - fc.stream_start
    P, Y = fc.P[:seen], fc.Y[fc.stream_start:fc.pos]
    a, o = fc.active_from - fc.stream_start, fc.onset - fc.stream_start
    first_fire_after = [r for r in fc.fired if r >= fc.onset]
    return {
        "uuid": fc.uuid[:8], "drifted": fc.drifted,
        "f1_before": macro_f1(Y[a:o], P[a:o], thr), "f1_after": macro_f1(Y[o:], P[o:], thr),
        "fires_before": sum(r < fc.onset for r in fc.fired), "fires_after": len(first_fire_after),
        "lag": (first_fire_after[0] - fc.onset) if first_fire_after else np.nan,
        "final_lambda": fc.lam, "updates": fc.updates,
    }
