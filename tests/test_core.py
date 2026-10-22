"""Unit tests for the core building blocks. Run with: .venv/bin/python -m pytest"""
import numpy as np
import pytest
import torch
from scipy.stats import norm
from torch.nn.utils import parameters_to_vector

from src.detectors import FedConDTest, ThresholdAlarm
from src.drift import inject
from src.fedcond import METHODS, FedClient, _weighted_average, local_train, run
from src.metrics import macro_f1, micro_f1
from src.models import MLP


# --- FedConD drift test (Eq. 4) -------------------------------------------------------------------------------

def test_fedcond_test_matches_eq4():
    test = FedConDTest(queue_size=20, alpha=0.05)
    history = [0.9] * 20
    for s in history:
        test.update(s)
    new = 0.85
    s_bar, s_hat, delta = np.mean(history), np.mean(history + [new]), 1 / 21
    gamma = (abs(s_bar - new) - 0.5 * delta) / np.sqrt(s_hat * (1 - s_hat) * delta)
    assert test.update(new) == (norm.sf(gamma) < 0.05)


def test_fedcond_test_flags_only_large_drops():
    test = FedConDTest(queue_size=20, alpha=0.05)
    for _ in range(20):
        assert not test.update(0.9)
    assert not test.update(0.85)   # small drop: not significant
    assert test.update(0.2)        # large drop: drift
    assert not test.update(1.0)    # improvement is never drift (one-sided)


def test_fedcond_test_skips_undefined_scores():
    test = FedConDTest(queue_size=20, alpha=0.05)
    test.update(0.9)
    assert not test.update(float("nan"))
    assert list(test.queue) == [0.9]


# --- Local training and aggregation ----------------------------------------------------------------------------

def _toy_model_and_data(seed=0):
    torch.manual_seed(seed)
    model = MLP(8, 3, [16], 0.0)
    X = torch.randn(200, 8)
    Y = (torch.rand(200, 3) > 0.7).float()
    return model, X, Y


def test_local_train_does_not_modify_global_weights():
    model, X, Y = _toy_model_and_data()
    w = parameters_to_vector(model.parameters()).detach().clone()
    w_before = w.clone()
    w_k = local_train(model, w, X, Y, 0.1, 2, {"lr": 1e-3, "batch_size": 32}, torch.Generator().manual_seed(0))
    assert torch.equal(w, w_before)
    assert not torch.equal(w_k, w)


def test_proximal_term_shrinks_the_update():
    model, X, Y = _toy_model_and_data()
    w = parameters_to_vector(model.parameters()).detach().clone()
    sizes = [(local_train(model, w, X, Y, lam, 2, {"lr": 1e-2, "batch_size": 32},
                          torch.Generator().manual_seed(0)) - w).norm() for lam in (0.0, 10.0)]
    assert sizes[1] < 0.5 * sizes[0]


def test_weighted_average():
    a, b = torch.tensor([0.0, 2.0]), torch.tensor([4.0, 6.0])
    assert torch.allclose(_weighted_average([a, b], [1, 3]), torch.tensor([3.0, 5.0]))


# --- Drift injection, alarms, metrics --------------------------------------------------------------------------

def test_inject_abrupt_and_gradual():
    X = np.zeros((10, 3), dtype=np.float32)
    abrupt = inject(X, [0, 1], onset=4, magnitude=2.0, kind="abrupt", width=4, clip=5.0)
    assert (abrupt[:4] == 0).all() and (abrupt[4:, :2] == 2).all() and (abrupt[:, 2] == 0).all()
    gradual = inject(X, [0], onset=4, magnitude=2.0, kind="gradual", width=4, clip=5.0)
    assert np.allclose(gradual[:, 0], [0, 0, 0, 0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0])
    assert inject(X + 4.5, [0], 0, 2.0, "abrupt", 4, clip=5.0)[:, 0].max() == 5.0


def test_threshold_alarm_needs_consecutive_rows():
    alarm = ThresholdAlarm(threshold=1.0, confirm=3)
    scores = np.array([0, 2, 2, 0, 2, 2, 2, 2, 0])
    assert alarm.alarms(scores).tolist() == [6]
    assert alarm.alarm_state(scores).tolist() == [6, 7]


def test_f1_ignores_unreported_labels():
    Y = np.array([[1, 0], [0, np.nan], [1, np.nan]], dtype=np.float32)
    P = np.array([[0.9, 0.9], [0.9, 0.9], [0.1, 0.9]], dtype=np.float32)
    # label 0: tp=1, fp=1, fn=1 -> F1 = 0.5; label 1 has no reported positive -> skipped
    assert macro_f1(Y, P) == pytest.approx(0.5)
    # pooled: tp=1, fp=2 (row 0 label 1, row 1 label 0), fn=1 -> 2/(2+2+1)
    assert micro_f1(Y, P) == pytest.approx(0.4)


# --- Federated simulation (all methods, tiny synthetic clients) ------------------------------------------------

CFG = {
    "federated": {"concurrency": 0.5, "local_epochs": 1, "lambda_base": 0.1, "lambda_step": 0.2,
                  "warmup_rounds": 1, "recenter_window": 20, "recenter_min_rows": 10},
    "data": {"clip_z": 5.0},
    "classifier": {"hidden": [8], "dropout": 0.0, "lr": 1e-3, "batch_size": 16, "threshold": 0.5},
    "fedcond_test": {"batch_rows": 10, "queue_size": 20, "alpha": 0.05},
    "paper_setup": {"fedprox_mu": 0.1},
}


def _toy_clients(n_clients=4):
    rng = np.random.default_rng(0)
    clients = []
    for k in range(n_clients):
        X = rng.normal(size=(100, 6)).astype(np.float32)
        Y = (rng.random((100, 2)) > 0.6).astype(np.float32)
        clients.append(FedClient(uuid=f"c{k}", X=X, Y=Y, stream_start=30, active_from=30, onset=60,
                                 drifted=k == 0, ae_alarms=np.array([65]), test=FedConDTest(20, 0.05),
                                 lam=0.1))
    return clients


@pytest.mark.parametrize("method", list(METHODS))
def test_every_method_streams_all_data(method):
    clients = _toy_clients()
    w, model = run(clients, method, CFG, seed=0)
    assert torch.isfinite(w).all()
    for fc in clients:
        assert fc.pos == len(fc.X)                  # whole stream consumed (70 rows = 7 rounds of 10)
        assert fc.updates == 7
        assert not np.isnan(fc.P).any()             # every stream row predicted before training on it
    if method == "oracle":
        assert clients[0].fired == [69] and clients[0].lam == pytest.approx(0.3)
    if method == "oracle_recenter":
        assert clients[0].fired == [69] and clients[0].lam == pytest.approx(0.1) and clients[0].correct_from == 60
    if method in ("fedcond_ae", "fedcond_ae_leash"):   # fedcond_ae: no autoencoder on the toy clients, so the
        assert all(fc.fired == [69] and fc.lam == pytest.approx(0.3) for fc in clients)   # alarm's step is pending
    if method in ("fedavg", "fedprox", "none"):
        assert all(not fc.fired for fc in clients)


def test_hide_labels_keeps_start_and_nests_rates():
    from src.fedcond import hide_labels
    Y = np.ones((1000, 2), dtype=np.float32)
    kept = {r: ~np.isnan(hide_labels(Y, 100, r, seed=0)).any(axis=1) for r in (0.5, 0.1, 0.0)}
    assert all(k[:100].all() for k in kept.values())               # starting rows always keep their labels
    assert not kept[0.0][100:].any()
    assert abs(kept[0.5][100:].mean() - 0.5) < 0.06 and abs(kept[0.1][100:].mean() - 0.1) < 0.04
    assert (kept[0.1] <= kept[0.5]).all()                            # 10% rows are a subset of the 50% rows


def test_recenter_removes_a_constant_shift():
    from src.fedcond import recenter
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3)).astype(np.float32)
    X[200:, :2] += 2.0                                   # drift on two features from row 200
    fc = FedClient(uuid="c", X=X, Y=np.zeros((400, 1), np.float32), stream_start=0, active_from=0, onset=200,
                   drifted=True, ae_alarms=np.array([], dtype=int), test=FedConDTest(20, 0.05), lam=0.1)
    fc.correct_from, fc.X_orig, fc.corr_base = 200, fc.X.copy(), np.zeros(3, np.float32)
    recenter(fc, stop=400, window=200, min_rows=50, clip=5.0)
    assert np.allclose(fc.X[200:].mean(axis=0), X[:200].mean(axis=0), atol=1e-5)
    assert np.array_equal(fc.X[:200], X[:200])         # rows before the alarm untouched


def test_recenter_accumulates_over_alarms():
    """A false alarm before the drift and one after it must not undo the correction (toy stream, 10-row rounds)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3)).astype(np.float32) * 0.01
    X[100:, 0] += 3.0                                    # drift on feature 0 from row 100
    fc = FedClient(uuid="c", X=X, Y=np.zeros((200, 1), np.float32), stream_start=20, active_from=20, onset=100,
                   drifted=True, ae_alarms=np.array([40, 100, 160]), test=FedConDTest(20, 0.05), lam=0.1)
    cfg = {**CFG, "federated": {**CFG["federated"], "recenter_window": 30, "recenter_min_rows": 10}}
    run([fc], "fedcond_ae_recenter", cfg, seed=0)
    assert fc.fired == [49, 109, 169]
    assert abs(fc.X[80:100, 0].mean()) < 0.1             # false alarm before the drift: ~no correction
    assert abs(fc.X[130:200, 0].mean()) < 0.1            # drift removed, and kept after the third alarm


def test_autoencoder_rejects_correction_on_a_false_alarm():
    """With an autoencoder on the client, a correction is applied only when the rows since the alarm look
    abnormal to it and the shift makes them look normal again."""
    from src.detectors import train_autoencoder, reconstruction_error
    from src.fedcond import recenter, recenter_shift
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 4)).astype(np.float32)
    ae = train_autoencoder(X[:300], {"hidden": [8, 4], "lr": 1e-2, "batch_size": 32, "epochs": 30}, seed=0)
    err = reconstruction_error(ae, X[:300])
    thr = float(err.mean() + 3 * err.std())

    def client(X):
        fc = FedClient(uuid="c", X=X.copy(), Y=np.zeros((600, 1), np.float32), stream_start=0, active_from=0,
                       onset=400, drifted=True, ae_alarms=np.array([], dtype=int), test=FedConDTest(20, 0.05),
                       lam=0.1, ae=ae, ae_threshold=thr)
        fc.correct_from, fc.X_orig, fc.corr_base = 400, fc.X.copy(), np.zeros(4, np.float32)
        return fc

    fc = client(X)                                        # false alarm: nothing changed at row 400
    assert recenter_shift(fc, stop=600, window=200, min_rows=50) is None
    recenter(fc, stop=600, window=200, min_rows=50, clip=5.0)
    assert np.array_equal(fc.X, X) and not fc.latest_ok

    X2 = X.copy()
    X2[400:, :2] += 3.0                                   # real shift from row 400
    fc = client(X2)
    recenter(fc, stop=600, window=200, min_rows=50, clip=5.0)
    assert fc.latest_ok and np.allclose(fc.X[400:].mean(axis=0), X2[200:400].mean(axis=0), atol=0.05)


def test_alarm_inside_window_supersedes_a_rejected_one_only():
    """Toy stream, 10-row rounds, window 80, at least 51 rows before a shift is measured. A false alarm (row 160)
    is superseded by the true one (row 200) inside its window (too few rows yet: rejected so far), so the shift is
    measured from the true onset. A third alarm (row 260) arrives while the autoencoder confirms the correction
    being measured and is ignored. Nothing is frozen early."""
    from src.detectors import train_autoencoder, reconstruction_error
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 3)).astype(np.float32)
    X[200:, 0] += 3.0
    # bottleneck narrower than the input, so the autoencoder cannot learn the identity and does see the shift
    ae = train_autoencoder(X[:100], {"hidden": [2], "lr": 1e-2, "batch_size": 32, "epochs": 30}, seed=0)
    err = reconstruction_error(ae, X[:100])
    fc = FedClient(uuid="c", X=X, Y=np.zeros((400, 1), np.float32), stream_start=100, active_from=100, onset=200,
                   drifted=True, ae_alarms=np.array([160, 200, 260]), test=FedConDTest(20, 0.05), lam=0.1,
                   ae=ae, ae_threshold=float(err.mean() + 3 * err.std()))
    cfg = {**CFG, "federated": {**CFG["federated"], "recenter_window": 80, "recenter_min_rows": 51}}
    run([fc], "fedcond_ae_recenter", cfg, seed=0)
    assert fc.fired == [169, 209, 269]
    assert fc.correct_from == 200 and fc.corrections == [] and fc.latest_ok
    assert abs(fc.X[280:, 0].mean()) < 0.6 and abs(fc.X[100:200, 0].mean()) < 0.6   # 3.0 shift gone (80-row windows are noisy)


def test_fedcond_ae_takes_the_leash_back_only_for_explained_alarms():
    """Same toy stream as the supersede test, full FedConD-AE reaction: the superseded false alarm keeps its leash step,
    the true alarm's step is pending (its window is still open at the end), the ignored alarm adds none."""
    from src.detectors import train_autoencoder, reconstruction_error
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 3)).astype(np.float32)
    X[200:, 0] += 3.0
    ae = train_autoencoder(X[:100], {"hidden": [2], "lr": 1e-2, "batch_size": 32, "epochs": 30}, seed=0)
    err = reconstruction_error(ae, X[:100])
    fc = FedClient(uuid="c", X=X, Y=np.zeros((400, 1), np.float32), stream_start=100, active_from=100, onset=200,
                   drifted=True, ae_alarms=np.array([160, 200, 260]), test=FedConDTest(20, 0.05), lam=0.1,
                   ae=ae, ae_threshold=float(err.mean() + 3 * err.std()))
    cfg = {**CFG, "federated": {**CFG["federated"], "recenter_window": 80, "recenter_min_rows": 51}}
    run([fc], "fedcond_ae", cfg, seed=0)
    assert fc.fired == [169, 209, 269] and fc.correct_from == 200
    assert fc.lam == pytest.approx(0.5)

    # a confirmed alarm whose window completes gives its step back (no autoencoder: always confirmed)
    fc = FedClient(uuid="c", X=X, Y=np.zeros((400, 1), np.float32), stream_start=100, active_from=100, onset=200,
                   drifted=True, ae_alarms=np.array([120, 220]), test=FedConDTest(20, 0.05), lam=0.1)
    run([fc], "fedcond_ae", cfg, seed=0)
    assert fc.corrections == [True] and fc.lam == pytest.approx(0.3)
