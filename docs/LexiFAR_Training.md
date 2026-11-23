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

Sentence training uses `[BOS] source [EOS] [BOS] target [EOS]`. Only target tokens contribute to the shifted token loss for the auxiliary prediction after block 4 and for the fused prediction. The word loss comes from a separate batch of contextual word labels in the same optimizer step. Source-only word guidance is fused before the second LSTM, the Transformers and the Mamba blocks. Attention is causal, and the LSTM is unidirectional. Right padding is masked from the loss and recurrent state updates.

With `source_copy` enabled, the fused prediction learns a per-token gate between the vocabulary distribution and attention over preceding or current source tokens. The copy distribution sums mass at repeated token IDs and excludes sentence markers, padding and target tokens. P4 minimizes the negative log probability of the reference token under the mixture.

`contextual_edits` adds a generic edit decision to the shared word head so it can score spans throughout the source. Training aligns each real source and reference at word and punctuation boundaries. Replaced or deleted spans supervise `EDIT` and matched spans supervise `KEEP`. These automatic labels use the frozen real training pairs. The guides use the source at training and inference. P4 subtracts the unexpanded E3 edit weight from each source position's copy attention logit. This keeps the copy bias local to the predicted word positions.

Generation consumes the source prefix and greedily predicts target tokens. LSTM and Mamba states, the Transformers' past normalized states and the copy head's source positions are cached. The returned sequence begins with an initial BOS marker that is excluded from generated-token counts.

The Mamba module is the plain-PyTorch Mamba-1 reference recurrence, not an optimized official CUDA kernel. No speed or numerical-parity claim is made. Transformer attention remains quadratic.

## Data and tokenizer

`data/academic-pairs-v1` contains 12,491 training pairs: 12,000 synthetic examples and 491 existing real revisions. The 98 validation and 87 test pairs are the unchanged real splits from `data/processed`.

Synthetic examples use 48 authored semantic frames across 24 technical domains, with shared facts, hypothetical quantities and protected terms. Records retain generator IDs, slots, hashes and the flags `synthetic=true`, `human_verified=false`, `split=train`. Automated quantity, negation, scope and uncertainty guards do not establish independent semantic validity. Synthetic examples never enter validation or test.

The byte-level BPE tokenizer is fitted to sentence and word training text with a full byte alphabet and a 4,096-token vocabulary limit. The default model starts from random weights. `--tiny` uses smaller model dimensions for pipeline checks. Tiny plain sentence models use a word-level tokenizer and span-edit models use byte-level BPE. Cross-entropy is comparable only when the tokenizer, target representation and evaluation protocol match.

Fresh training initializes the token, absolute-position and source/target segment embeddings at a standard deviation of 0.02 before adding them together. The padding token starts at zero. A fresh run records `fresh_input_embedding_std` in `result.json`. Loading a checkpoint or continuing training restores all three learned tables from its saved weights.

The [dataset card](Dataset_Card.md) documents real-data provenance and preserved curation references. The frozen arXivEdits partition and source hashes are retained; those references are not active training data or independently approved examples.

## Training

Run commands yourself from the repository root. Project code is not executed by the agent unless explicitly requested.

```powershell
.\.venv\Scripts\python.exe scripts/train_six_layer.py --print-command
.\.venv\Scripts\python.exe scripts/train_six_layer.py
```

The launcher reads `configs/six_layer_experiment.json` and checks the generator and data fingerprints before training. It uses width 192, four attention heads, feed-forward width 768, dropout 0.1, batch 16, LR 3e-4, seed 42, source/target limits of 192 tokens and up to 2,000 joint updates. Sentence truncation is rejected.

The default uses balanced sentence batches of eight real revisions and eight synthetic pairs and samples a separate batch from the word corpus. Within each sentence batch, examples are sampled without replacement from their respective groups. Across updates, they can recur. An odd sentence batch allocates the extra pair to real revisions. `result.json` records the number of real and synthetic pairs seen across all updates.

Each update prints the step, total loss, 20-step moving average, auxiliary P3 cross-entropy, fused P4 cross-entropy, mean word cross-entropy, contextual edit loss when supervised, learning rate and elapsed time. `--log-every 10` changes the terminal interval to ten updates. The trainer records every step in `training_progress.jsonl` and reports when validation starts and finishes.

Validation runs every 250 updates and monitors the fused final token cross-entropy. The best-validation checkpoint includes step zero, and early stopping follows four unimproved checks. The checkpoint directory stores `weights.pt`, the tokenizer and a config with the saved stage layout, fusion and candidate guidance settings. The training and validation logs record auxiliary losses where labels are available.

The default config passes `--overwrite` and reruns training in `runs/six_layer_joint_v6_edits`. Use `--output` to choose the directory to write. The trainer publishes `result.json` after it saves the weights and logs, marking the run ready for evaluation.

`--init-checkpoint` on `academic_rewriter.run` continues the saved architecture and tokenizer with a fresh optimizer and writes to a different directory. A joint continuation uses the saved word replacement vocabulary, source-copy setting and contextual-edit setting. Start a fresh joint run with `--source-copy --contextual-edits` to train the expanded word head and the guided copy path together.

## Focused grammar continuation

The saved v6 validation outputs retain `in LayoutLM` with guidance both enabled and disabled. They also retain `corpus this kind` and contain malformed suffixes on otherwise recognizable source phrases. Alignment of the 491 real training pairs yielded no single-word `in` to `into` change and no isolated insertion of `of`.

The v7 recipe continues the local `runs/six_layer_joint_v6_edits` checkpoint with a fresh optimizer for 600 updates at LR `1e-4`. Each sentence batch contains eight real revisions, six original synthetic pairs and two [grammar pairs](../data/grammar-edits-v1/README.md). The supplement has 42 edited examples and 22 retain examples. The launcher checks its fingerprint.

P4 weights reference subwords in replaced or inserted spans by three. For deletions it weights the next reference word, or the final end token when the removed material reaches the end. Eligible labels come from real training pairs and the grammar supplement. Other target tokens have weight one. The recipe normalizes the P4 loss by the sum of target weights and uses ordinary validation loss for logging. It saves the final update as the v7 checkpoint.

```powershell
$python = '.\.venv\Scripts\python.exe'
$run = 'runs/six_layer_joint_v7_grammar'
& $python scripts/train_six_layer.py --config configs/six_layer_grammar_edits.json
& $python -m academic_rewriter.test --checkpoint $run --split validation --limit 0
```

Compare `evaluation/validation-192/predictions.csv` with the v6 file by ID. Inspect `065afd42f64dbb51` for the expression about physical appearance, `00a677997fbda9ac` for the preposition after `incorporate`, and `bb11df8bbef966fe` for the missing `of`. Check complete words, scientific meaning and necessary edits in the raw text. The v7 training run and this review are pending.

## Whole-word span editing

The v8 configuration starts the entire six-layer model, including the 193 edit-boundary embeddings, from random weights. It fits a new byte-level BPE tokenizer on the training sentence pairs, grammar supplement and word corpus. The full sentence training pool contains 12,555 pairs. Each batch samples eight real revisions, six original synthetic pairs and two grammar pairs. The 76 word examples provide separate supervision.

The recipe uses 2,000 optimizer updates, batch size 16, a learning rate of `3e-4` and seed 42. Validation runs every 250 updates. The final checkpoint is written to `runs/six_layer_joint_v8_spans_scratch`.

P4 learns ordered edit records from the existing training references. All untouched source spans are copied verbatim. Whole words may be replaced, deleted or preceded by an insertion. The renderer retains the source and records `edit_error` if the program is incomplete, has an invalid range or would join words across the edit boundary. For references with too many separate edits for the target budget, training can encode one edit spanning the source. The current 12,740 pairs fit as aligned edits with the v7 tokenizer.

For span checkpoints, the single-sentence rewrite returns the source when the candidate changes a recorded quantity, cardinal number word, recognized numeric comparison or technical identifier. It reports the first match in `fact_error` and `fallback`. Batch evaluations log `fact_error` on the assembled model prediction and summarize it in `fact_error_counts`. These checks cover recognizable expressions such as `seven`, `times faster than` and identifiers with mixed capitals or digits. Review the edited spans and their technical meaning when comparing models.

```powershell
$python = '.\.venv\Scripts\python.exe'
$run = 'runs/six_layer_joint_v8_spans_scratch'
& $python -m unittest discover -s tests -p 'test_span_edits.py' -v
& $python scripts/train_six_layer.py --config configs/six_layer_span_edits.json
& $python -m academic_rewriter.test --checkpoint $run --split validation --max-new-tokens 194 --limit 0
```

The earlier 1,000-step continuation from v7 is available in `runs/six_layer_joint_v8_spans`. For a separate continuation, pass `--init-checkpoint` with a compatible checkpoint and choose a separate `--output` directory. Compare the source, reference and assembled prediction in `evaluation/validation-194/predictions.csv`. The JSONL records `edits` with source character offsets and `edit_error`. Inspect those fields to distinguish a deliberate retain decision from an invalid attempt.

The local v1 from-scratch run finished 2,000 updates. It recorded a minimum P4 validation cross-entropy of 5.564 and saved the final step with a validation cross-entropy of 6.566. Its 98 saved validation predictions score 0.747 ROUGE-L. The input-copy baseline scores 0.823.

## Train on the reviewed expansion

Build `.cache/academic-pairs-v2` after both curators return completed decisions for the same training candidates. The [data card](Dataset_Card.md#reviewed-training-expansion) gives the review criteria and build command. The generated config uses the same from-scratch v8 settings and adds the approved revisions to the real training pool. It writes to `runs/six_layer_joint_v8_academic_pairs_v2`.

```powershell
$python = '.\.venv\Scripts\python.exe'
$data = '.cache/academic-pairs-v2'
$run = 'runs/six_layer_joint_v8_academic_pairs_v2'
& $python -m unittest discover -s tests -p 'test_academic_pair_expansion.py' -v
& $python scripts/train_six_layer.py --config "$data/experiment.json" --print-command
& $python scripts/train_six_layer.py --config "$data/experiment.json"
& $python -m academic_rewriter.test --checkpoint $run --data $data --split validation --max-new-tokens 194 --limit 0
```

Use the same validation cases to compare which needed edits each model completes, where it makes unnecessary edits, and whether it preserves facts and qualifiers. The independent reviews, dataset build and training comparison are pending.

## Train on the author revision expansion

The v3 data build produces a local experiment config for a controlled comparison with the v8 run on v1. It uses the same six-layer architecture, random model initialization, batch composition, learning rate, update budget and validation cadence. Each batch draws its eight real pairs from the expanded author revision pool. The 98 validation and 87 test cases have the original bytes. Training verifies and loads the v1-trained BPE tokenizer saved during the 192-token encoding check. [v3 data build](Dataset_Card.md#author-revision-expansion)

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m unittest discover -s tests -p 'test_academic_pair_expansion.py' -v
if ($LASTEXITCODE -ne 0) { throw 'Expansion regression test failed. See the output above.' }
& $python scripts/expand_academic_pairs.py --author-revisions
```

After the builder reports at least 10,000 real pairs and creates `experiment.json`, train and evaluate v8.

```powershell
$python = '.\.venv\Scripts\python.exe'
$data = '.cache/academic-pairs-v3'
$run = 'runs/six_layer_joint_v8_academic_pairs_v3'
if (-not (Test-Path -LiteralPath "$data/experiment.json" -PathType Leaf)) {
    throw 'Complete the v3 data build before training.'
}
& $python scripts/train_six_layer.py --config "$data/experiment.json" --print-command
if ($LASTEXITCODE -ne 0) { throw 'Training config check failed. See the output above.' }
& $python scripts/train_six_layer.py --config "$data/experiment.json"
if ($LASTEXITCODE -ne 0) { throw 'Training failed. See the output above.' }
& $python -m academic_rewriter.test --checkpoint $run --data $data --split validation --max-new-tokens 194 --limit 0
```

Compare the validation loss trajectory and inspect the same validation sentences for needed edits, unintended changes and preservation of technical claims. The model training, regression test and generation comparison are manual steps.

### Keep the best validation checkpoint

The v1 from-scratch run recorded its lowest P4 validation loss at step 500. For a v3 run using the lowest validation loss, pass `--checkpoint-selection best-validation`. The command below stops after four consecutive validation checks without improvement and writes the selected weights to a dedicated run directory. Build v3 before running it.

```powershell
$python = '.\.venv\Scripts\python.exe'
$data = '.cache/academic-pairs-v3'
$run = 'runs/six_layer_joint_v8_academic_pairs_v3_best'
if (-not (Test-Path -LiteralPath "$data/experiment.json" -PathType Leaf)) {
    throw 'Complete the v3 data build before training.'
}
& $python -m unittest discover -s tests -p 'test_span_edits.py' -v
if ($LASTEXITCODE -ne 0) { throw 'Span editing regression test failed. See the output above.' }
& $python scripts/train_six_layer.py --config "$data/experiment.json" --checkpoint-selection best-validation --early-stopping-patience 4 --output $run
if ($LASTEXITCODE -ne 0) { throw 'Training failed. See the output above.' }
& $python -m academic_rewriter.test --checkpoint $run --data $data --split validation --max-new-tokens 194 --limit 0
```

Use the new `validation-194` predictions to check needed edits, malformed replacements and protected-expression changes alongside the copy baseline. This training and validation comparison awaits manual execution.

## Evaluate the entire split

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_joint_v6_edits --split validation --batch-size 8 --max-new-tokens 192 --limit 0
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_joint_v6_edits --split test --batch-size 8 --max-new-tokens 192 --limit 0
```

The default data directory is `data/academic-pairs-v1`. A zero limit evaluates every row: 98 validation cases or 87 test cases. Only completed checkpoints with `result.json`, `six_layer_config.json`, `tokenizer/` and `weights.pt` can be evaluated.

Each command writes `CHECKPOINT/evaluation/SPLIT-192/predictions.csv`, `predictions.jsonl` and `metrics.json`. With a fused checkpoint, each row aligns the source, human reference and final `prediction`. JSONL adds observed budget hits and source/reference truncation. A budget hit alone does not prove a sentence is incomplete.

Start the qualitative review with the LayoutLM preposition edit and the corpus description reporting 7,000 documents. Check that the raw generation retains the model name, technical relationships and number while making the local edit.

```powershell
$ids = '00a677997fbda9ac', '4766dceaeff49be0'
Get-Content runs/six_layer_joint_v6_edits/evaluation/validation-192/predictions.jsonl -Encoding UTF8 |
    ForEach-Object { ConvertFrom-Json -InputObject $_ } |
    Where-Object { $_.id -in $ids } |
    Format-List source,reference,prediction
```

Repeated evaluation replaces files in its selected output directory. Pass `--output runs/evaluation_v2` to preserve an earlier evaluation. Checkpoints and predictions are local runtime artifacts and are ignored by Git.

To inspect the contribution of E1–E3 during generation, decode the same checkpoint with `--no-word-guidance`. Compare the two `predictions.csv` files by ID. Check whether the required edit was made and whether nearby terms, numbers and clauses were preserved. The second run writes to `evaluation/validation-192-no-word-guidance/` by default.

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_joint_v6_edits --split validation --limit 0 --no-word-guidance
```

To measure the effect of the replacement vectors, train a position-only checkpoint with the same data and seed, then evaluate it with the same decode settings. `metrics.json` records `candidate_guidance`.

```powershell
.\.venv\Scripts\python.exe scripts/train_six_layer.py --word-guidance position --output runs/six_layer_joint_v6_edits_position
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_joint_v6_edits_position --split test --limit 0
```

## Data reproduction and checks

The existing data and generator hashes are frozen. The following generator command is for first creation of a new study path only; do not rerun it over `data/academic-pairs-v1`.

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.synthetic --output data/academic-pairs-v2 --count 12000
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_integrity.ps1
```

For a changed corpus, use a new configuration and output directory that point to its new manifest. Tests cover causal isolation, source copy probabilities, edit alignment and gradients, cached/full-logit agreement, padding, balanced sampling, target-only loss, checkpoint reload and continuation, full prediction exports and synthetic-data isolation. These commands remain pending until the user runs them.

## Evidence boundaries

The real validation/test cases have already been inspected and are diagnostic rather than fresh confirmation. A changed output is not automatically an improvement. Overlap, copy rate and number preservation do not prove academic style, fluency or factual equivalence. Independent quality ratings and a successful-rewrite rate remain unavailable.
