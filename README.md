# FedConD-AE

Practical part of the master's thesis *Federated Learning with Concept Drift on Mobile Sensor Data*
(International Burch University, Sarajevo).

FedConD (Chen et al., 2021, [arXiv:2109.00151](https://arxiv.org/abs/2109.00151)) is an asynchronous federated
learning method that handles concept drift on each client. Both of its drift parts need true labels at run time:
the alarm watches the model's accuracy on new data, and the reaction (tightening the proximal term λ) is triggered by
that alarm. Phone users rarely label their activity, so in practice the alarm goes blind.

**FedConD-AE** keeps FedConD unchanged (server, client scheduling, local objective, training) and replaces the two
label-dependent parts with label-free ones:

- **Alarm.** A small autoencoder per client, trained on the client's own early sensor data, with a threshold set on
  one calibration day (mean + 3 std of the reconstruction error). Three consecutive minutes above the threshold raise
  the alarm.
- **Reaction.** The client measures how much each feature's daily mean moved since the day before the alarm and
  subtracts that shift from the rows since the alarm ("re-learn normal"). The correction is applied only if the
  autoencoder confirms it: the rows since the alarm must look abnormal as a whole and the correction must make them
  look normal again. Alarms the autoencoder cannot explain this way get FedConD's own reaction (λ += 0.2), so FedConD
  is the fallback inside the method.

No code of the original FedConD is public; it is reimplemented here from the paper, together with FedAvg and FedProx.
All methods share one code base, so differences in results can only come from the method.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q      # 24 tests, a few seconds
```

Python 3.14, PyTorch (CPU is enough, the data is small). Every random choice is seeded from `config.yaml`.

## Data

The experiments use the [ExtraSensory dataset](http://extrasensory.ucsd.edu/), "primary data: features and labels".
Download `ExtraSensory.per_uuid_features_labels.zip` and extract the 60 `<UUID>.features_labels.csv.gz` files into
`data/raw/` (keep them gzipped). The dataset is not part of the repository.

```bash
.venv/bin/python -m src.data    # preprocess all users, print a per-user summary, cache to data/processed/
```

Preprocessing per user (`src/data.py`): rows sorted by time; the same 158 feature columns for every user (columns
missing in more than 20% of all rows are dropped); rows with more than 30% missing features dropped; mean imputation
and z-scoring with the user's own training statistics, clipped to ±5; 30 labels (positive for at least 30 users),
multi-label, missing labels masked out of the loss and the metrics. 45 of the 60 users have enough data for the
drift scheme; all 59 non-empty users are used in the paper scheme.

Two split schemes, both chronological per user:

- **drift scheme** (main experiments): first 60% of rows = training rows (all labels) + one calibration day + one
  clean day; last 40% = post-drift segment, where synthetic drift is injected on 10% of the clients.
- **paper scheme** (replication): 60/20/20 train/validation/test per user, by time (`paper_time`) or at random
  (`paper_random`), as in the FedConD paper.

## Reproducing the results

Each script writes `results/<name>_<date>.csv` plus a `<name>_summary_<date>.csv` and the exact `config.yaml` it
ran with. Times are for a 10-core laptop CPU; `--workers N` runs that many federated simulations in parallel.

| Result | Command | Time | Output |
|---|---|---|---|
| Replication in the paper's setup (no drift, all labels, micro + macro F1) | `python -m experiments.paper_comparison --workers 6` | 30 min | `paper_comparison_*` |
| Natural drift in ExtraSensory (frozen model and autoencoder error per day) | `python -m experiments.natural_drift_check` | 30 min | `natural_drift_check_*`, `natural_drift_daily_*` |
| Detection race, H1 and H2 (lag and false alarms of the three alarms, 45 users) | `python -m experiments.detection_race` | 20 min | `detection_race_*` |
| FedProx μ and FedConD base λ tuning on validation rows | `python -m experiments.lambda_tuning --workers 6` | 25 min | `lambda_tuning_*` |
| Reactions under a perfect alarm (none / FedConD's / correction) | `python -m experiments.reaction_test --workers 3` | 35 min | `reaction_test_*` |
| Label scarcity without drift (paper scheme, 100% to 0% labels) | `python -m experiments.label_scarcity --workers 6` | 1 h | `label_scarcity_*` |
| **Main comparison, H3** (drift + label scarcity, all methods, 5 seeds) | `python -m experiments.main_comparison --workers 7` | 5.5 h | `main_comparison_*` |

Figures: `python -m experiments.plot_drift_example` (one feature before and after injected drift) and
`python -m experiments.plot_detector_example [user] [abrupt|gradual] [magnitude]` (the three alarms on one stream).

All commands run with `.venv/bin/python` from the repository root.

### The main comparison

45 clients, 5 of them drifted (drawn per seed), 5 seeds, abrupt and gradual drift. Drift = +2 training standard
deviations on all 83 motion-sensor features (accelerometer, gyroscope, magnetometer) from 25% into the post-drift
segment on; gradual drift ramps up over 120 minutes. Training rows are fully labeled; after that each streamed row
keeps its labels with probability 100%, 5% or 0%. Hidden labels are unavailable for training and for FedConD's
alarm, and are used only for scoring. Scoring is test-then-train: every batch of 30 new rows is predicted by the
received global model before the client trains on it. Reported: macro F1 (main metric, 30 unbalanced labels) and
micro F1 after the drift onset, on drifted clients, clean clients and all clients.

Method names in the result files:

| key | method |
|---|---|
| `fedavg`, `fedprox` | McMahan et al. 2017, Li et al. 2020 (synchronous) |
| `fedcond` | FedConD (asynchronous; supervised alarm, λ += 0.2 on alarm) |
| `fedcond_ae` | **FedConD-AE** (autoencoder alarm, confirmed correction, FedConD's reaction as fallback) |
| `fedcond_ae_leash` | ablation: autoencoder alarm with FedConD's reaction only |
| `fedcond_ae_recenter` | ablation: autoencoder alarm with the correction only |
| `oracle_recenter` | reference: an alarm that fires exactly at the true onset, with the correction |
| `fedcond_step01` | sensitivity: FedConD with λ += 0.1, the paper's smallest step |

## Layout

```
config.yaml       all settings and seeds, one section per experiment
src/data.py       loading, cleaning, splitting, scaling
src/models.py     the activity classifier (MLP, masked BCE)
src/drift.py      synthetic drift injection
src/detectors.py  the alarms: FedConD's test, the autoencoder, prediction entropy
src/evaluation.py lag and false-alarm evaluation of the alarms on one stream
src/fedcond.py    the federated simulation: FedAvg, FedProx, FedConD, FedConD-AE and its ablations
src/metrics.py    macro and micro F1 with missing labels ignored
experiments/      one runnable script per result (see the table above)
tests/            unit tests of the alarm test, the proximal training, drift injection, the reaction
results/          result files (only the final ones are committed)
```
