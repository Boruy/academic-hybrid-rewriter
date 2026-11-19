# Six-layer Academic Rewriter

The [three rewrite levels](docs/Three_Level_Model.md) provide a word-only mode and separately trained medium and deep neural modes.

An academic text revision model with exactly **2 LSTM + 2 Transformer + 2 Mamba** sequence blocks. The model trains from scratch with a training-only tokenizer. Architecture and commands are documented in [Six_Layer_Model.md](docs/Six_Layer_Model.md).

This repository contains the datasets and their provenance, the six-layer model, training and full-split evaluation code, and the supporting configuration and checks. Checkpoints, prediction files and generated reports stay local in ignored directories. Historical baseline implementations, presentations and report pipelines have been removed.

## Project files

| Path | Purpose |
| --- | --- |
| `data/academic-pairs-v1/` | Ready-to-use training, validation and test splits with a frozen manifest |
| `data/raw/`, `data/processed/` | Original IteraTeR release, source hashes and real revision splits |
| `data/curation/` | Preserved audit records and frozen arXivEdits references; not an approved training corpus |
| `docs/` | Model, dataset and environment documentation |
| `configs/six_layer_experiment.json` | Versioned training settings |
| `src/academic_rewriter/` | Six-layer model, data preparation, training and evaluation |
| `scripts/` | Setup, pinned data download, training launcher and integrity check |
| `tests/` | Model, data-isolation, checkpoint and generation regression checks |

## Installation

Use Python 3.10 or later. Run these commands yourself in PowerShell from the project root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
```

For an existing offline environment, use `scripts/setup.ps1 -Offline`. See [environment setup](docs/Environment_Setup.md). Optional SacreBLEU support can be installed with `.\.venv\Scripts\python.exe -m pip install -e '.[evaluation]'`; other recorded metrics do not require it.

## Train

The prepared dataset already exists: **12,491 training pairs** (12,000 synthetic + 491 real), **98 real validation pairs** and **87 real test pairs**. Synthetic pairs are unverified weak supervision and are used only for training. Do not regenerate or repartition this frozen version.

Use a fresh output directory for each run:

```powershell
.\.venv\Scripts\python.exe scripts/train_six_layer.py --output runs/six_layer_manual_v2 --print-command
.\.venv\Scripts\python.exe scripts/train_six_layer.py --output runs/six_layer_manual_v2
```

The launcher checks the generator and split hashes before training. The settings use width 192, four attention heads, batch size 16, LR 3e-4, up to 2,000 updates, validation every 250 updates, best-validation checkpoint selection and early stopping. Existing weights are protected against accidental replacement.

## Test and export all predictions

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_manual_v2 --split validation --limit 0
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_manual_v2 --split test --limit 0
```

Both commands default to `data/academic-pairs-v1`, batch size 8 and a budget of 192 new tokens. `--limit 0` evaluates every row. Results are written to `CHECKPOINT/evaluation/SPLIT-192/`:

- `predictions.csv`: every ID, original `source`, model `prediction` and human `reference`, encoded as UTF-8 with a BOM for Excel.
- `predictions.jsonl`: the same texts plus token-length, truncation and generation-budget diagnostics.
- `metrics.json`: aggregate cross-entropy, overlap, copy and number-preservation diagnostics.

Use `--output runs/evaluation_v2` to choose a new results directory and preserve an earlier evaluation. The CSV retains complete sentences, including commas, quotes and line breaks.

## Checks

Execution remains user-controlled. The agent provides commands and does not run training, inference, data generators, tests or project scripts without an explicit execution request.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_integrity.ps1
```

## Data and evidence

See the [dataset card](docs/Dataset_Card.md), [paired-data README](data/academic-pairs-v1/README.md) and [third-party notices](THIRD_PARTY_NOTICES.md). The synthetic generator and frozen data bytes are preserved. New study versions must use new paths. The arXivEdits references are not independent human approvals or ready-to-train examples.

The Mamba blocks use a sequential PyTorch reference recurrence. Official-kernel parity and optimized speedups are unverified. The Transformer blocks retain quadratic attention. The real validation/test sets have already been inspected and are diagnostic; lower loss or preserved numbers do not establish useful academic rewriting.

`SHA256SUMS.txt` covers the actual bytes of tracked project files. Environments, caches and user-run artifacts are excluded. Project changes follow the commit-and-push workflow in [AGENTS.md](AGENTS.md).
