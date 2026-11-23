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

Use Python 3.10 or later. Run the blocks yourself in PowerShell. Each block sets the checkout path to `P:\MLProject`. Change that first path if your checkout is elsewhere.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'scripts/setup.ps1'
    if ($LASTEXITCODE -ne 0) { throw 'Installation failed. See the output above.' }
}
```

For an existing offline environment, run the offline setup block. To add SacreBLEU to the recorded metrics, use the optional evaluation installation block. [Offline and optional evaluation commands](docs/Environment_Setup.md)

## Train one checkpoint

The prepared dataset already exists: **12,491 training pairs** (12,000 synthetic + 491 real), **98 real validation pairs** and **87 real test pairs**. Synthetic pairs are unverified weak supervision and are used only for training. Do not regenerate or repartition this frozen version.

The default training command writes to `runs/six_layer_joint_v6_edits`. Rerunning it replaces the checkpoint and training logs in that directory.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $config = 'configs/six_layer_experiment.json'
    $run = 'runs/six_layer_joint_v6_edits'
    & $python scripts/train_six_layer.py --config $config --output $run --print-command
    if ($LASTEXITCODE -ne 0) { throw 'v6 configuration check failed. See the output above.' }
    & $python scripts/train_six_layer.py --config $config --output $run
    if ($LASTEXITCODE -ne 0) { throw 'v6 training failed. See the output above.' }
}
```

Each sentence batch contains eight real revisions and eight synthetic examples. The final head learns to mix vocabulary generation with copying source tokens. The joint objective combines word supervision, an auxiliary sentence objective and the fused final sentence objective. The launcher verifies the data and generator hashes. The settings use width 192, four attention heads, LR 3e-4, up to 2,000 updates, validation every 250 updates, best-validation checkpoint selection and early stopping.

The shared word head learns additional edit-position supervision from the real training revisions. At inference it can flag source spans without a dictionary entry. P4 uses the final guide to favor copying positions outside the predicted edit regions.

The terminal prints each optimizer update and its losses. The launcher's `--log-every` option sets the print interval in updates.

The focused grammar experiment continues the completed v6 checkpoint and writes to `runs/six_layer_joint_v7_grammar`. It samples the grammar supplement and gives edited reference tokens extra weight in the final sentence loss. See the [grammar continuation commands](docs/LexiFAR_Training.md#focused-grammar-continuation).

The v8 span-edit configuration trains the full six-layer model from random weights and writes to `runs/six_layer_joint_v8_spans_scratch`. P4 learns to select complete word ranges and generate their replacements while copying untouched source text verbatim. See the [manual training commands](docs/LexiFAR_Training.md#whole-word-span-editing).

The reviewed v2 and author-revision v3 studies train v8 on larger real training pools. [Reviewed expansion workflow](docs/LexiFAR_Training.md#train-on-the-reviewed-expansion) · [Author revision workflow](docs/LexiFAR_Training.md#train-on-the-author-revision-expansion)

## Test and export final predictions

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $data = 'data/academic-pairs-v1'
    $run = 'runs/six_layer_joint_v6_edits'
    if (-not (Test-Path -LiteralPath "$run/result.json" -PathType Leaf -ErrorAction Stop)) {
        throw 'Complete v6 training before evaluation.'
    }
    foreach ($split in @('validation', 'test')) {
        & $python -m academic_rewriter.test --checkpoint $run --data $data --split $split --batch-size 8 --max-new-tokens 192 --limit 0 --output "$run/evaluation/${split}-192"
        if ($LASTEXITCODE -ne 0) { throw "v6 $split evaluation failed. See the output above." }
    }
}
```

The block evaluates every row with batch size 8 and a budget of 192 new tokens. It writes the following files under `runs/six_layer_joint_v6_edits/evaluation/validation-192/` and `runs/six_layer_joint_v6_edits/evaluation/test-192/`.

- `predictions.csv`: source, human reference and one fused final `prediction`, encoded as UTF-8 with a BOM for Excel.
- `predictions.jsonl`: the same comparison plus token lengths, truncation and generation diagnostics.
- `metrics.json` records final sentence metrics and a copy-input baseline. The evaluator can also report aggregate scores for independently reviewed word pairs. [Complete word evaluation command](docs/LexiFAR_Architecture.md#test-the-fused-rewrite)

The CSV retains complete sentences, including commas, quotes and line breaks. Use the separate export example to keep earlier evaluation files. [Full-split and separate export commands](docs/LexiFAR_Training.md#evaluate-the-entire-split)

Open the [offline results viewer](docs/results/index.html) in a local browser and select the completed run directory. It reads the training and validation logs plus each pair of `evaluation/<group>/metrics.json` and `predictions.jsonl` files. Switch between evaluation groups, search by ID or sentence, and compare the source, human reference and raw prediction with word-level highlighting. The training view plots the 20-step moving average and the fused validation cross-entropy. The viewer skips evaluation files that predate the current completion record or weights.

## Checks

Execution remains user-controlled. The agent provides commands and does not run training, inference, data generators, tests or project scripts without an explicit execution request.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    & $python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed. See the output above.' }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'scripts/verify_integrity.ps1'
    if ($LASTEXITCODE -ne 0) { throw 'Repository integrity check failed. See the output above.' }
}
```

## Data and evidence

See the [dataset card](docs/Dataset_Card.md), [paired-data README](data/academic-pairs-v1/README.md) and [third-party notices](THIRD_PARTY_NOTICES.md). The synthetic generator and frozen data bytes are preserved. New study versions must use new paths. The arXivEdits references are not independent human approvals or ready-to-train examples.

The Mamba blocks use a sequential PyTorch reference recurrence. Official-kernel parity and optimized speedups are unverified. The Transformer blocks retain quadratic attention. The real validation/test sets have already been inspected and are diagnostic; lower loss or preserved numbers do not establish useful academic rewriting.

`SHA256SUMS.txt` covers the actual bytes of tracked project files. Environments, caches and user-run artifacts are excluded. Project changes follow the commit-and-push workflow in [AGENTS.md](AGENTS.md).
