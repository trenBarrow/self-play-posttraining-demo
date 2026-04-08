# Self-Play Posttraining Demo

## About

This repository is a code-only mirror of a cross-protocol command anomaly detection and self-play research pipeline. It includes dataset generation, canonicalization, model training, scoring, packaging, and tests for F´ and MAVLink-oriented workflows. Large runtime trees, vendored dependencies, and generated artifacts are intentionally excluded.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py --help
bash scripts/poster_demo.sh --dry-run
```

Run a small end-to-end example after dependencies are installed:

```bash
bash scripts/poster_demo.sh --rows 24 --seed 7 --protocol-mode fprime
```

## Startup

The live stack helpers use Docker Compose and expect the external runtime trees under `gds/`, which are not included in this code-only mirror.

```bash
bash scripts/fprime_real/up.sh
bash scripts/fprime_real/down.sh

bash scripts/mavlink_real/up.sh
bash scripts/mavlink_real/down.sh
```

## Test

Run a quick non-live pass with:

```bash
python3 -m pytest \
  tests/test_canonical_schema.py \
  tests/test_blue_runtime.py \
  tests/test_feature_policy.py
```

Run the full suite with:

```bash
python3 -m pytest
```
