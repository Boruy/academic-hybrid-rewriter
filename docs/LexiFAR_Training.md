# LexiFAR model and training

See the [LexiFAR architecture](LexiFAR_Architecture.md) for the four stages, word feedback and final fusion.

The model contains exactly **six sequence layers**:

| Layer | Module | Function |
| --- | --- | --- |
| 1 | Unidirectional LSTM | Local sequential features and recurrent memory |
| 2 | Unidirectional LSTM | Further recurrent composition |
| 3 | Transformer block | Causal multi-head self-attention and feed-forward network |
| 4 | Transformer block | Further contextual interaction |
| 5 | Mamba block | Selective state-space sequence mixing |
| 6 | Mamba block | Final selective sequence mixing |

Each block has normalization and a residual path. Token, position and source/target segment embeddings precede the blocks. The joint model has prediction projections after blocks 1, 2, 4 and 6. A shared word edit head supplies boundary weights and probability-weighted replacement vectors for the three transitions. For the final prediction, the four projected features are concatenated at the same token position and projected back to the hidden width. The result uses a vocabulary projection tied to the token embeddings. There is no pretrained encoder or decoder.

## Conditional generation

Sentence training uses `[BOS] source [EOS] [BOS] target [EOS]`. Only target tokens contribute to shifted cross-entropy for the auxiliary prediction after block 4 and for the fused prediction. The word loss comes from a separate batch of contextual word labels in the same optimizer step. Source-only word guidance is fused before the second LSTM, the Transformers and the Mamba blocks. Attention is causal, and the LSTM is unidirectional. Right padding is masked from the loss and recurrent state updates.

Generation consumes the source prefix and greedily predicts target tokens. LSTM and Mamba states are cached; Transformer blocks retain their past normalized states. The returned sequence begins with an initial BOS marker that is excluded from generated-token counts. Sampling and beam search are not implemented.

The Mamba module is the plain-PyTorch Mamba-1 reference recurrence, not an optimized official CUDA kernel. No speed or numerical-parity claim is made. Transformer attention remains quadratic.

## Data and tokenizer

`data/academic-pairs-v1` contains 12,491 training pairs: 12,000 synthetic examples and 491 existing real revisions. The 98 validation and 87 test pairs are the unchanged real splits from `data/processed`.

Synthetic examples use 48 authored semantic frames across 24 technical domains, with shared facts, hypothetical quantities and protected terms. Records retain generator IDs, slots, hashes and the flags `synthetic=true`, `human_verified=false`, `split=train`. Automated quantity, negation, scope and uncertainty guards do not establish independent semantic validity. Synthetic examples never enter validation or test.

The byte-level BPE tokenizer is fitted to sentence and word training text with a full byte alphabet and a 4,096-token vocabulary limit. The model starts from random weights. `--tiny` uses a small model and word-level tokenizer for pipeline checks. Cross-entropy is comparable only when the tokenizer and evaluation protocol match.

Fresh training initializes the token, absolute-position and source/target segment embeddings at a standard deviation of 0.02 before adding them together. The padding token starts at zero. A fresh run records `fresh_input_embedding_std` in `result.json`. Loading a checkpoint or continuing training restores all three learned tables from its saved weights.

The [dataset card](Dataset_Card.md) documents real-data provenance and preserved curation references. The frozen arXivEdits partition and source hashes are retained; those references are not active training data or independently approved examples.

## Training

Run commands yourself from the repository root. Project code is not executed by the agent unless explicitly requested.

```powershell
.\.venv\Scripts\python.exe scripts/train_six_layer.py --print-command
.\.venv\Scripts\python.exe scripts/train_six_layer.py
```

The launcher reads `configs/six_layer_experiment.json` and checks the generator and data fingerprints before training. It uses width 192, four attention heads, feed-forward width 768, dropout 0.1, batch 16, LR 3e-4, seed 42, source/target limits of 192 tokens and up to 2,000 joint updates. Sentence batches are drawn from shuffled epochs and word batches are sampled from the training word corpus. Sentence truncation is rejected.

To compare the updated initialization with the saved November 21 run, start a fresh run with the same configuration in a separate directory. Compare the raw generations on the same 98 validation examples with the copy-input baseline, including the four examples containing numeric strings.

```powershell
$python = '.\.venv\Scripts\python.exe'
$run = 'runs/six_layer_joint_v4_embscale'
& $python scripts/train_six_layer.py --output $run
& $python -m academic_rewriter.test --checkpoint $run --split validation --limit 0
```

Each update prints the step, total loss, 20-step moving average, auxiliary P3 cross-entropy, fused P4 cross-entropy, mean word cross-entropy, learning rate and elapsed time. `--log-every 10` changes the terminal interval to ten updates. The trainer records every step in `training_progress.jsonl` and reports when validation starts and finishes.

Validation runs every 250 updates and monitors the fused final token cross-entropy. The best-validation checkpoint includes step zero, and early stopping follows four unimproved checks. The checkpoint directory stores `weights.pt`, the tokenizer and a config with the saved stage layout, fusion and candidate guidance settings. The training and validation logs record auxiliary losses where labels are available.

The default config passes `--overwrite` and reruns training in `runs/six_layer_joint_v4`. Use `--output` to choose the directory to write. The trainer publishes `result.json` after it saves the weights and logs, marking the run ready for evaluation.

`--init-checkpoint` on `academic_rewriter.run` continues the saved architecture and tokenizer with a fresh optimizer and writes to a different directory. A joint continuation requires the saved word replacement vocabulary. Existing single-depth checkpoints remain available in their original mode.

## Evaluate the entire split

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_joint_v4 --split validation --batch-size 8 --max-new-tokens 192 --limit 0
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_joint_v4 --split test --batch-size 8 --max-new-tokens 192 --limit 0
```

The default data directory is `data/academic-pairs-v1`. A zero limit evaluates every row: 98 validation cases or 87 test cases. Only completed checkpoints with `result.json`, `six_layer_config.json`, `tokenizer/` and `weights.pt` can be evaluated.

Each command writes `CHECKPOINT/evaluation/SPLIT-192/predictions.csv`, `predictions.jsonl` and `metrics.json`. With a fused checkpoint, each row aligns the source, human reference and final `prediction`. JSONL adds observed budget hits and source/reference truncation. A budget hit alone does not prove a sentence is incomplete.

Repeated evaluation replaces files in its selected output directory. Pass `--output runs/evaluation_v2` to preserve an earlier evaluation. Checkpoints and predictions are local runtime artifacts and are ignored by Git.

To measure the effect of the replacement vectors, train a position-only checkpoint with the same data and seed, then evaluate it with the same decode settings. `metrics.json` records `candidate_guidance`.

```powershell
.\.venv\Scripts\python.exe scripts/train_six_layer.py --word-guidance position --output runs/six_layer_joint_v4_position
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_joint_v4_position --split test --limit 0
```

## Data reproduction and checks

The existing data and generator hashes are frozen. The following generator command is for first creation of a new study path only; do not rerun it over `data/academic-pairs-v1`.

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.synthetic --output data/academic-pairs-v2 --count 12000
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_integrity.ps1
```

For a changed corpus, use a new configuration and output directory that point to its new manifest. Tests cover causal isolation, candidate alignment and gradients, cached/full-logit agreement, padding, target-only loss, checkpoint reload and continuation, full prediction exports and synthetic-data isolation. These commands remain pending until the user runs them.

## Evidence boundaries

The real validation/test cases have already been inspected and are diagnostic rather than fresh confirmation. A changed output is not automatically an improvement. Overlap, copy rate and number preservation do not prove academic style, fluency or factual equivalence. Independent quality ratings and a successful-rewrite rate remain unavailable.
