# LexiFAR

[Project overview in Chinese](docs/Project_Overview_ZH.md)

**LexiFAR** (Lexically Guided Feature Fusion for Academic Rewriting) revises English research sentences. A shared word edit module passes edit locations and probability-weighted replacement vectors between stages. Features from all four stages fuse into a single rewritten sentence. See the [LexiFAR architecture](docs/LexiFAR_Architecture.md).

The model has **2 LSTM + 2 Transformer + 2 Mamba** sequence blocks and trains from scratch with a training-only tokenizer. See [LexiFAR model and training](docs/LexiFAR_Training.md) for the layer configuration and commands.

This repository contains the datasets and their provenance, the six-layer model, training and full-split evaluation code, and the supporting configuration and checks. Checkpoints, prediction files and generated reports stay local in ignored directories. Historical baseline implementations, presentations and report pipelines have been removed.

## Project files

| Path | Purpose |
| --- | --- |
| `data/academic-pairs-v1/` | Ready-to-use training, validation and test splits with a frozen manifest |
| `data/word-formality-v1/` | Single-word replacements and retain cases for the joint word objective |
| `data/grammar-edits-v1/` | Training-only synthetic preposition edits and retain examples |
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

## Train one checkpoint

The prepared dataset already exists: **12,491 training pairs** (12,000 synthetic + 491 real), **98 real validation pairs** and **87 real test pairs**. Synthetic pairs are unverified weak supervision and are used only for training. Do not regenerate or repartition this frozen version.

The default training command writes to `runs/six_layer_joint_v6_edits`. Rerunning it replaces the checkpoint and training logs in that directory.

```powershell
.\.venv\Scripts\python.exe scripts/train_six_layer.py --print-command
.\.venv\Scripts\python.exe scripts/train_six_layer.py
```

Each sentence batch contains eight real revisions and eight synthetic examples. The final head learns to mix vocabulary generation with copying source tokens. The joint objective combines word supervision, an auxiliary sentence objective and the fused final sentence objective. The launcher verifies the data and generator hashes. The settings use width 192, four attention heads, LR 3e-4, up to 2,000 updates, validation every 250 updates, best-validation checkpoint selection and early stopping.

The shared word head learns additional edit-position supervision from the real training revisions. At inference it can flag source spans without a dictionary entry. P4 uses the final guide to favor copying positions outside the predicted edit regions.

The terminal prints each optimizer update and its losses. Pass `--log-every 10` for a line every ten updates.

The focused grammar experiment continues the completed v6 checkpoint and writes to `runs/six_layer_joint_v7_grammar`. It samples the grammar supplement and gives edited reference tokens extra weight in the final sentence loss. See the [grammar continuation commands](docs/LexiFAR_Training.md#focused-grammar-continuation).

The span-edit experiment trains P4 to select complete word ranges and generate their replacement text. It copies the untouched source text verbatim, including spacing and punctuation. See the [span-edit commands](docs/LexiFAR_Training.md#whole-word-span-editing).

## Test and export final predictions

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_joint_v6_edits --split validation --limit 0
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_joint_v6_edits --split test --limit 0
```

Both commands default to `data/academic-pairs-v1`, batch size 8 and a budget of 192 new tokens. `--limit 0` evaluates every row. Results are written to `CHECKPOINT/evaluation/SPLIT-192/`:

- `predictions.csv`: source, human reference and one fused final `prediction`, encoded as UTF-8 with a BOM for Excel.
- `predictions.jsonl`: the same comparison plus token lengths, truncation and generation diagnostics.
- `metrics.json`: final sentence metrics and a copy-input baseline. Pass `--word-test PATH` to add aggregate scores for independently reviewed word pairs.

Use `--output runs/evaluation_v2` to choose a new results directory and preserve an earlier evaluation. The CSV retains complete sentences, including commas, quotes and line breaks.

Open the [offline results viewer](docs/results/index.html) in a local browser and select the completed run directory. It reads the training and validation logs plus each pair of `evaluation/<group>/metrics.json` and `predictions.jsonl` files. Switch between evaluation groups, search by ID or sentence, and compare the source, human reference and raw prediction with word-level highlighting. The training view plots the 20-step moving average and the fused validation cross-entropy. The viewer skips evaluation files that predate the current completion record or weights.

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
