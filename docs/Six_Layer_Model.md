# Six-layer academic rewriter

The model contains exactly **six sequence layers**:

| Layer | Module | Function |
| --- | --- | --- |
| 1 | Unidirectional LSTM | Local sequential features and recurrent memory |
| 2 | Unidirectional LSTM | Further recurrent composition |
| 3 | Transformer block | Causal multi-head self-attention and feed-forward network |
| 4 | Transformer block | Further contextual interaction |
| 5 | Mamba block | Selective state-space sequence mixing |
| 6 | Mamba block | Final selective sequence mixing |

Each block has normalization and a residual path. Token, position and source/target segment embeddings precede the blocks; output normalization and a tied vocabulary projection follow them. These are not additional sequence layers. There is no pretrained encoder or decoder. The repository supports this six-layer architecture only.

## Conditional generation

Training uses `[BOS] source [EOS] [BOS] target [EOS]`. Only target tokens contribute to shifted cross-entropy; source tokens provide context. Attention is causal, and the LSTM is unidirectional. Right padding is masked from the loss and recurrent state updates.

Generation consumes the source prefix and greedily predicts target tokens. LSTM and Mamba states are cached; Transformer blocks retain their past normalized states. The returned sequence begins with an initial BOS marker that is excluded from generated-token counts. Sampling and beam search are not implemented.

The Mamba module is the plain-PyTorch Mamba-1 reference recurrence, not an optimized official CUDA kernel. No speed or numerical-parity claim is made. Transformer attention remains quadratic.

## Data and tokenizer

`data/academic-pairs-v1` contains 12,491 training pairs: 12,000 synthetic examples and 491 existing real revisions. The 98 validation and 87 test pairs are the unchanged real splits from `data/processed`.

Synthetic examples use 48 authored semantic frames across 24 technical domains, with shared facts, hypothetical quantities and protected terms. Records retain generator IDs, slots, hashes and the flags `synthetic=true`, `human_verified=false`, `split=train`. Automated quantity, negation, scope and uncertainty guards do not establish independent semantic validity. Synthetic examples never enter validation or test.

The byte-level BPE tokenizer is fitted to training text only with a full byte alphabet and a 4,096-token vocabulary. The model starts from random weights. `--tiny` uses a small model and word-level tokenizer for pipeline checks. Cross-entropy is comparable only when the tokenizer and evaluation protocol match.

The [dataset card](Dataset_Card.md) documents real-data provenance and preserved curation references. The frozen arXivEdits partition and source hashes are retained; those references are not active training data or independently approved examples.

## Training

Run commands yourself from the repository root. Project code is not executed by the agent unless explicitly requested.

```powershell
.\.venv\Scripts\python.exe scripts/train_six_layer.py --output runs/six_layer_manual_v2 --print-command
.\.venv\Scripts\python.exe scripts/train_six_layer.py --output runs/six_layer_manual_v2
```

The launcher reads `configs/six_layer_experiment.json` and checks the exact generator, dataset and held-out fingerprints before training. It uses width 192, four attention heads, feed-forward width 768, dropout 0.1, batch 16, LR 3e-4, seed 42, source/target limits of 192 tokens and up to 2,000 updates. Each epoch is shuffled without replacement. Truncation is rejected.

Validation runs every 250 updates. The best-validation checkpoint includes step zero, and early stopping follows four unimproved checks. Per-step loss, moving-average loss, gradient norm, learning rate and validation records are saved alongside the tokenizer, model config and `weights.pt`.

Use a new output directory for another run. `--init-checkpoint` on `academic_rewriter.run` loads a saved six-layer model, tokenizer and config with a fresh optimizer. Checkpoint module names and tensor layouts are preserved so that existing six-layer checkpoints remain compatible. Runtime reload verification is left to the user.

## Evaluate the entire split

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_manual_v2 --split validation --batch-size 8 --max-new-tokens 192 --limit 0
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_manual_v2 --split test --batch-size 8 --max-new-tokens 192 --limit 0
```

The default data directory is `data/academic-pairs-v1`. A zero limit evaluates every row: 98 validation cases or 87 test cases. Only completed checkpoints with `result.json`, `six_layer_config.json`, `tokenizer/` and `weights.pt` can be evaluated.

Each command writes `CHECKPOINT/evaluation/SPLIT-192/predictions.csv`, `predictions.jsonl` and `metrics.json`. The CSV includes all originals, predictions and human references, with complete text and Excel-compatible UTF-8 encoding. JSONL also records observed budget hits and source/reference truncation. A budget hit alone does not prove a sentence is incomplete.

Repeated evaluation replaces files in its selected output directory. Pass `--output runs/evaluation_v2` to preserve an earlier evaluation. Checkpoints and predictions are local runtime artifacts and are ignored by Git.

## Data reproduction and checks

The existing data and generator hashes are frozen. The following generator command is for first creation of a new study path only; do not rerun it over `data/academic-pairs-v1`.

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.synthetic --output data/academic-pairs-v2 --count 12000
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_integrity.ps1
```

For a changed corpus, use a new configuration and output directory that point to its new manifest. Tests cover causal isolation, cached/full-logit agreement, padding, target-only loss, all-block gradients, checkpoint reload/continuation, full prediction exports and synthetic-data isolation. These commands remain pending until the user runs them.

## Evidence boundaries

The real validation/test cases have already been inspected and are diagnostic rather than fresh confirmation. A changed output is not automatically an improvement. Overlap, copy rate and number preservation do not prove academic style, fluency or factual equivalence. Independent quality ratings and a successful-rewrite rate remain unavailable.
