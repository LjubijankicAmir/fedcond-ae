# FedConD-AE

Practical part of the master's thesis *Federated Learning with Concept Drift on Mobile Sensor Data*
(International Burch University, Sarajevo).

FedConD (Chen et al., 2021, [arXiv:2109.00151](https://arxiv.org/abs/2109.00151)) is an asynchronous federated
learning method that detects concept drift on each client by watching the model's accuracy, which requires labels.
FedConD-AE keeps FedConD unchanged except for the drift trigger, which is replaced by a per-client autoencoder
that detects drift from the rise in reconstruction error and needs no labels.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data

The experiments use the [ExtraSensory dataset](http://extrasensory.ucsd.edu/), "primary data: features and labels".
Download `ExtraSensory.per_uuid_features_labels.zip` and extract the 60 `<UUID>.features_labels.csv.gz` files
into `data/raw/` (keep them gzipped).

## Layout

```
config.yaml     all settings and seeds
src/            library code (data loading, models, drift injection, detectors, FedConD)
experiments/    runnable scripts, one per experiment
```

## Usage

```bash
.venv/bin/python -m src.data    # preprocess all users and print a per-user summary
```
