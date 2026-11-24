# LexiFAR

[Project overview in Chinese](docs/Project_Overview_ZH.md)

**LexiFAR** (Lexically Guided Feature Fusion for Academic Rewriting) revises English research sentences. A shared word edit module passes edit locations and probability-weighted replacement vectors between stages. Features from all four stages fuse into a single rewritten sentence. See the [LexiFAR architecture](docs/LexiFAR_Architecture.md).

The default backbone has **2 LSTM + 2 Transformer + 2 Mamba** sequence blocks and trains from scratch with a training-only tokenizer. The training guide includes a comparison with two tanh RNN blocks and a middle self-attention ablation. [LexiFAR model and training](docs/LexiFAR_Training.md#compare-lstm-and-tanh-rnn-front-ends)

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
| `configs/six_layer_span_edits.json` | v8 settings used to generate the v3 experiment config |
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

## Train the current checkpoint

The current recipe trains v8 from random weights on the v3 author revision dataset. The local v3 build contains 10,264 real and 12,000 synthetic training pairs. The recipe doubles the P4 loss weight on inserted and replacement text and selects the weights with the lowest P4 validation loss. [Complete v8 + v3 PowerShell workflow](docs/LexiFAR_Training.md#train-on-the-author-revision-expansion)

## Rewrite and evaluate with the selected checkpoint

Use the checkpoint under `runs/six_layer_joint_v8_academic_pairs_v3_payload_2_best`. Training exports all 98 validation predictions. After selecting the checkpoint, evaluate all 87 test sentences. [Single-sentence command](docs/LexiFAR_Architecture.md#generate-a-final-rewrite) ? [Full test command](docs/LexiFAR_Training.md#evaluate-the-entire-split)

The evaluator writes these files under `evaluation/validation-194/` or `evaluation/test-194/`.

- `predictions.csv` aligns the source, human reference and final prediction. It uses UTF-8 with a BOM for Excel.
- `predictions.jsonl` includes applied edits, token lengths, truncation and fact diagnostics.
- `metrics.json` records final sentence metrics and a copy-input baseline. [Complete word evaluation command](docs/LexiFAR_Architecture.md#test-the-fused-rewrite)

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
