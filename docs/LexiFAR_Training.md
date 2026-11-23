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

Run each PowerShell block yourself. The blocks use `P:\MLProject` as the checkout path. If your checkout is elsewhere, change the first `Set-Location` in the block. Set up `.venv` before training or evaluation. The explicit `--config` path supplies the complete recipe and lets the launcher verify the frozen data. [Environment setup](Environment_Setup.md)

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

The launcher reads `configs/six_layer_experiment.json` and checks the generator and data fingerprints before training. It uses width 192, four attention heads, feed-forward width 768, dropout 0.1, batch 16, LR 3e-4, seed 42, source/target limits of 192 tokens and up to 2,000 joint updates. Sentence truncation is rejected.

The default uses balanced sentence batches of eight real revisions and eight synthetic pairs and samples a separate batch from the word corpus. Within each sentence batch, examples are sampled without replacement from their respective groups. Across updates, they can recur. An odd sentence batch allocates the extra pair to real revisions. `result.json` records the number of real and synthetic pairs seen across all updates.

Each update prints the step, total loss, 20-step moving average, auxiliary P3 cross-entropy, fused P4 cross-entropy, mean word cross-entropy, contextual edit loss when supervised, learning rate and elapsed time. The launcher's `--log-every` option sets the terminal interval in updates. The trainer records every step in `training_progress.jsonl` and reports when validation starts and finishes.

Validation runs every 250 updates and monitors the fused final token cross-entropy. The best-validation checkpoint includes step zero, and early stopping follows four unimproved checks. The checkpoint directory stores `weights.pt`, the tokenizer and a config with the saved stage layout, fusion and candidate guidance settings. The training and validation logs record auxiliary losses where labels are available.

The default config passes `--overwrite` and reruns training in `runs/six_layer_joint_v6_edits`. Use `--output` to choose the directory to write. The trainer publishes `result.json` after it saves the weights and logs, marking the run ready for evaluation.

The v6 recipe trains the expanded word head and guided copy path together. A continuation uses the saved architecture, tokenizer, word replacement vocabulary, source-copy setting and contextual-edit setting, and starts a fresh optimizer. The v7 and v8 continuation commands specify both the parent checkpoint and a dedicated output directory.

## Focused grammar continuation

The saved v6 validation outputs retain `in LayoutLM` with guidance both enabled and disabled. They also retain `corpus this kind` and contain malformed suffixes on otherwise recognizable source phrases. Alignment of the 491 real training pairs yielded no single-word `in` to `into` change and no isolated insertion of `of`.

The v7 recipe continues the local `runs/six_layer_joint_v6_edits` checkpoint with a fresh optimizer for 600 updates at LR `1e-4`. Each sentence batch contains eight real revisions, six original synthetic pairs and two [grammar pairs](../data/grammar-edits-v1/README.md). The supplement has 42 edited examples and 22 retain examples. The launcher checks its fingerprint.

P4 weights reference subwords in replaced or inserted spans by three. For deletions it weights the next reference word, or the final end token when the removed material reaches the end. Eligible labels come from real training pairs and the grammar supplement. Other target tokens have weight one. The recipe normalizes the P4 loss by the sum of target weights and uses ordinary validation loss for logging. It saves the final update as the v7 checkpoint.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $data = 'data/academic-pairs-v1'
    $parent = 'runs/six_layer_joint_v6_edits'
    $run = 'runs/six_layer_joint_v7_grammar'
    if (-not (Test-Path -LiteralPath "$parent/result.json" -PathType Leaf -ErrorAction Stop)) {
        throw 'Complete the v6 training block before starting v7.'
    }
    & $python scripts/train_six_layer.py --config 'configs/six_layer_grammar_edits.json' --init-checkpoint $parent --output $run --checkpoint-selection final --early-stopping-patience 0
    if ($LASTEXITCODE -ne 0) { throw 'v7 training failed. See the output above.' }
    & $python -m academic_rewriter.test --checkpoint $run --data $data --split validation --batch-size 8 --max-new-tokens 192 --limit 0 --output "$run/evaluation/validation-192"
    if ($LASTEXITCODE -ne 0) { throw 'v7 validation failed. See the output above.' }
}
```

Compare `evaluation/validation-192/predictions.csv` with the v6 file by ID. Inspect `065afd42f64dbb51` for the expression about physical appearance, `00a677997fbda9ac` for the preposition after `incorporate`, and `bb11df8bbef966fe` for the missing `of`. Check complete words, scientific meaning and necessary edits in the raw text.

## Whole-word span editing

The v8 configuration starts the entire six-layer model, including the 193 edit-boundary embeddings, from random weights. It fits a new byte-level BPE tokenizer on the training sentence pairs, grammar supplement and word corpus. The full sentence training pool contains 12,555 pairs. Each batch samples eight real revisions, six original synthetic pairs and two grammar pairs. The 76 word examples provide separate supervision.

The recipe uses 2,000 optimizer updates, batch size 16, a learning rate of `3e-4` and seed 42. Validation runs every 250 updates. The final checkpoint is written to `runs/six_layer_joint_v8_spans_scratch`.

P4 learns ordered edit records from the existing training references. All untouched source spans are copied verbatim. Whole words may be replaced, deleted or preceded by an insertion. The renderer retains the source and records `edit_error` if the program is incomplete, has an invalid range or would join words across the edit boundary. For references with too many separate edits for the target budget, training can encode one edit spanning the source. The current 12,740 pairs fit as aligned edits with the v7 tokenizer.

For span checkpoints, the single-sentence rewrite returns the source when the candidate changes a recorded quantity, cardinal number word, recognized numeric comparison or technical identifier. It reports the first match in `fact_error` and `fallback`. Batch evaluations log `fact_error` on the assembled model prediction and summarize it in `fact_error_counts`. These checks cover recognizable expressions such as `seven`, `times faster than` and identifiers with mixed capitals or digits. Review the edited spans and their technical meaning when comparing models.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $data = 'data/academic-pairs-v1'
    $run = 'runs/six_layer_joint_v8_spans_scratch'
    & $python -m unittest discover -s tests -p 'test_span_edits.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Span editing regression test failed. See the output above.' }
    & $python scripts/train_six_layer.py --config 'configs/six_layer_span_edits.json' --output $run --checkpoint-selection final --early-stopping-patience 0
    if ($LASTEXITCODE -ne 0) { throw 'v8 training failed. See the output above.' }
    & $python -m academic_rewriter.test --checkpoint $run --data $data --split validation --batch-size 8 --max-new-tokens 194 --limit 0 --output "$run/evaluation/validation-194"
    if ($LASTEXITCODE -ne 0) { throw 'v8 validation failed. See the output above.' }
}
```

The earlier 1,000-step continuation from v7 is available in `runs/six_layer_joint_v8_spans`. The following command runs a separate continuation using the current 2,000-step v8 recipe and the completed v7 parent.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $data = 'data/academic-pairs-v1'
    $parent = 'runs/six_layer_joint_v7_grammar'
    $run = 'runs/six_layer_joint_v8_spans_from_v7'
    if (-not (Test-Path -LiteralPath "$parent/result.json" -PathType Leaf -ErrorAction Stop)) {
        throw 'Complete the v7 training block before continuing with span edits.'
    }
    & $python scripts/train_six_layer.py --config 'configs/six_layer_span_edits.json' --init-checkpoint $parent --output $run --checkpoint-selection final --early-stopping-patience 0
    if ($LASTEXITCODE -ne 0) { throw 'v8 continuation failed. See the output above.' }
    & $python -m academic_rewriter.test --checkpoint $run --data $data --split validation --batch-size 8 --max-new-tokens 194 --limit 0 --output "$run/evaluation/validation-194"
    if ($LASTEXITCODE -ne 0) { throw 'v8 continuation validation failed. See the output above.' }
}
```

Compare the source, reference and assembled prediction in `evaluation/validation-194/predictions.csv`. The JSONL records `edits` with source character offsets and `edit_error`. Inspect those fields to distinguish a deliberate retain decision from an invalid attempt.

The local v1 from-scratch run finished 2,000 updates. It recorded a minimum P4 validation cross-entropy of 5.564 and saved the final step with a validation cross-entropy of 6.566. Its 98 saved validation predictions score 0.747 ROUGE-L. The input-copy baseline scores 0.823.

## Train on the reviewed expansion

The v2 workflow uses the two completed curator exports at the paths below or reuses an existing `.cache/academic-pairs-v2` study. Its generated config adds the approved revisions to the real training pool and uses the from-scratch v8 settings. [Review criteria and export instructions](Dataset_Card.md#reviewed-training-expansion)

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $data = '.cache/academic-pairs-v2'
    $config = "$data/experiment.json"
    $reviews = '.cache/academic-refinement/review-v2/train'
    $run = 'runs/six_layer_joint_v8_academic_pairs_v2'
    & $python -m unittest discover -s tests -p 'test_academic_pair_expansion.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Expansion regression test failed. See the output above.' }
    if (-not (Test-Path -LiteralPath $config -PathType Leaf -ErrorAction Stop)) {
        foreach ($review in @("$reviews/train_curator_1.jsonl", "$reviews/train_curator_2.jsonl")) {
            if (-not (Test-Path -LiteralPath $review -PathType Leaf -ErrorAction Stop)) {
                throw "Place the completed curator export at $review."
            }
        }
        & $python scripts/expand_academic_pairs.py --version 2 --reviews "$reviews/train_curator_1.jsonl" "$reviews/train_curator_2.jsonl"
        if ($LASTEXITCODE -ne 0) { throw 'v2 dataset build failed. See the output above.' }
    }
    & $python scripts/train_six_layer.py --config $config --output $run --checkpoint-selection final --early-stopping-patience 0 --print-command
    if ($LASTEXITCODE -ne 0) { throw 'v2 configuration check failed. See the output above.' }
    & $python scripts/train_six_layer.py --config $config --output $run --checkpoint-selection final --early-stopping-patience 0
    if ($LASTEXITCODE -ne 0) { throw 'v2 training failed. See the output above.' }
    & $python -m academic_rewriter.test --checkpoint $run --data $data --split validation --batch-size 8 --max-new-tokens 194 --limit 0 --output "$run/evaluation/validation-194"
    if ($LASTEXITCODE -ne 0) { throw 'v2 validation failed. See the output above.' }
}
```

Use the same validation cases to compare which needed edits each model completes, where it makes unnecessary edits, and whether it preserves facts and qualifiers. The independent reviews, dataset build and training comparison are pending.

## Train on the author revision expansion

The v3 study uses the same six-layer architecture, random model initialization, batch composition, learning rate, update budget and validation cadence as the v8 run on v1. Each batch draws its eight real pairs from a pool of 10,264 real training pairs in the completed local v3 build. The 98 validation and 87 test cases have the original bytes. Training verifies and loads the v1-trained BPE tokenizer saved during the 192-token encoding check. The build requires the pinned training alignment at `.cache/arxivedits/train_alignment.json`. [Complete alignment download and v3 build commands](Dataset_Card.md#author-revision-expansion)

### Keep the best validation checkpoint

The run on v1 reached its lowest P4 validation loss at step 500. This block builds v3 if needed, saves the weights with the lowest P4 validation loss and stops after four consecutive validation checks without improvement. It then exports every validation prediction for that checkpoint.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $alignment = '.cache/arxivedits/train_alignment.json'
    $data = '.cache/academic-pairs-v3'
    $config = "$data/experiment.json"
    $run = 'runs/six_layer_joint_v8_academic_pairs_v3_best'
    & $python -m unittest discover -s tests -p 'test_academic_pair_expansion.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Expansion regression test failed. See the output above.' }
    & $python -m unittest discover -s tests -p 'test_span_edits.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Span editing regression test failed. See the output above.' }
    if (-not (Test-Path -LiteralPath $config -PathType Leaf -ErrorAction Stop)) {
        if (-not (Test-Path -LiteralPath $alignment -PathType Leaf -ErrorAction Stop)) {
            throw "Download the training alignment to $alignment using docs/Dataset_Card.md."
        }
        & $python scripts/expand_academic_pairs.py --version 3 --author-revisions $alignment
        if ($LASTEXITCODE -ne 0) { throw 'v3 dataset build failed. See the output above.' }
    }
    & $python scripts/train_six_layer.py --config $config --output $run --checkpoint-selection best-validation --early-stopping-patience 4 --print-command
    if ($LASTEXITCODE -ne 0) { throw 'v3 configuration check failed. See the output above.' }
    & $python scripts/train_six_layer.py --config $config --output $run --checkpoint-selection best-validation --early-stopping-patience 4
    if ($LASTEXITCODE -ne 0) { throw 'v3 training failed. See the output above.' }
    & $python -m academic_rewriter.test --checkpoint $run --data $data --split validation --batch-size 8 --max-new-tokens 194 --limit 0 --output "$run/evaluation/validation-194"
    if ($LASTEXITCODE -ne 0) { throw 'v3 validation failed. See the output above.' }
}
```

Compare the new `validation-194` predictions with the same validation cases from v1. Check needed edits, malformed replacements and protected-expression changes alongside the copy baseline.

### Save the final update for the fixed-step comparison

Use a completed v3 data build. This block uses the full 2,000-update budget and saves the final weights in a separate directory.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $data = '.cache/academic-pairs-v3'
    $config = "$data/experiment.json"
    $run = 'runs/six_layer_joint_v8_academic_pairs_v3'
    if (-not (Test-Path -LiteralPath $config -PathType Leaf -ErrorAction Stop)) {
        throw 'Complete the v3 dataset build in docs/Dataset_Card.md before training.'
    }
    & $python scripts/train_six_layer.py --config $config --output $run --checkpoint-selection final --early-stopping-patience 0
    if ($LASTEXITCODE -ne 0) { throw 'v3 training failed. See the output above.' }
    & $python -m academic_rewriter.test --checkpoint $run --data $data --split validation --batch-size 8 --max-new-tokens 194 --limit 0 --output "$run/evaluation/validation-194"
    if ($LASTEXITCODE -ne 0) { throw 'v3 validation failed. See the output above.' }
}
```

### Evaluate the selected v3 checkpoint on the test split

Run this after selecting the checkpoint with validation. The example uses the best-validation run and evaluates all 87 test sentences. For the fixed-update checkpoint, set `$run` to `runs/six_layer_joint_v8_academic_pairs_v3` in this block.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $data = '.cache/academic-pairs-v3'
    $run = 'runs/six_layer_joint_v8_academic_pairs_v3_best'
    if (-not (Test-Path -LiteralPath "$run/result.json" -PathType Leaf -ErrorAction Stop)) {
        throw "Complete the v3 training run at $run before evaluating the test split."
    }
    & $python -m academic_rewriter.test --checkpoint $run --data $data --split test --batch-size 8 --max-new-tokens 194 --limit 0 --output "$run/evaluation/test-194"
    if ($LASTEXITCODE -ne 0) { throw 'v3 test evaluation failed. See the output above.' }
}
```

### Export validation predictions from an existing v3 run

Use this block to evaluate a completed fixed-update checkpoint. The output contains all 98 validation predictions and their edit diagnostics.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $data = '.cache/academic-pairs-v3'
    $run = 'runs/six_layer_joint_v8_academic_pairs_v3'
    if (-not (Test-Path -LiteralPath "$run/result.json" -PathType Leaf -ErrorAction Stop)) {
        throw "Complete the v3 training run at $run before evaluating the validation split."
    }
    & $python -m academic_rewriter.test --checkpoint $run --data $data --split validation --batch-size 8 --max-new-tokens 194 --limit 0 --output "$run/evaluation/validation-194"
    if ($LASTEXITCODE -ne 0) { throw 'v3 validation failed. See the output above.' }
}
```

### Compare source-copy settings on the completed v3 checkpoint

With the same tokenizer and frozen validation split, the v3 run reached P4 validation cross-entropy 4.853 at step 2,000. Its saved predictions scored 0.800 whitespace ROUGE-L versus 0.823 for copying the input. All 11 exact copies rendered generated edit records that left the text unchanged. Another 33 predictions only deleted text. The model returned three insertions across the split, while the aligned references contain 22. For example, in `00a677997fbda9ac`, it deleted `the` and missed the reference change from `in LayoutLM` to `into LayoutLM`.

Both v8 runs use objective weights 0.5 for P3 sentence CE, 0.5 for P4 edit-program CE, 0.25 for word CE and 0.25 for source edit-position loss. The source copy head learns a gate that mixes source-token probabilities into P4. For span edits, it can supply replacement text drawn from the source. Compare the same saved weights with copy scales 1.0 and 0.5 to estimate the effect of that gate on generation.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $data = '.cache/academic-pairs-v3'
    $run = 'runs/six_layer_joint_v8_academic_pairs_v3'
    if (-not (Test-Path -LiteralPath "$run/result.json" -PathType Leaf -ErrorAction Stop)) {
        throw "Complete the v3 training run at $run before comparing copy settings."
    }
    foreach ($scale in @('1.0', '0.5')) {
        $evaluation = "$run/evaluation/validation-194-copy-scale-$scale"
        & $python -m academic_rewriter.test --checkpoint $run --data $data --split validation --batch-size 8 --max-new-tokens 194 --limit 0 --span-copy-scale $scale --output $evaluation
        if ($LASTEXITCODE -ne 0) { throw "v3 validation failed at copy scale $scale. See the output above." }
        $metrics = Get-Content -LiteralPath "$evaluation/metrics.json" -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
        [pscustomobject]@{
            CopyScale = $metrics.span_copy_scale
            CopyRate = $metrics.copy_rate
            Retained = [int]$metrics.edit_status_counts.retain
            UnchangedEdit = [int]$metrics.edit_status_counts.noop_edit
            DeletionOnly = $metrics.deletion_only_count
            Insertions = $metrics.edit_operation_counts.insert
            RougeL = $metrics.whitespace_rouge_l_f1
        }
    }
}
```

`edit_status` distinguishes a deliberate retain, an edit that renders the same text, an invalid program and a changed sentence. Compare the same validation IDs for correct replacements, missed edits and `fact_error` alongside these counts. The decoding scale redistributes pointer probability to the vocabulary distribution. The reported cross-entropy still uses the trained model distribution.

### Compare span replacement weighting during continuation

The v3 run drew 16,000 real examples in 2,000 updates from 10,264 unique pairs and its P4 validation loss was still falling at the final update. The commands below continue from that checkpoint with a fresh optimizer and sampling seed 43. The first run keeps unit token weights. The second gives inserted and replacement text twice the loss weight of edit boundaries and control tokens, with normalization by the sum of target weights. Both runs use the same unweighted P4 validation loss for checkpoint selection.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $data = '.cache/academic-pairs-v3'
    $config = "$data/experiment.json"
    $parent = 'runs/six_layer_joint_v8_academic_pairs_v3'
    if (-not (Test-Path -LiteralPath $config -PathType Leaf -ErrorAction Stop)) {
        throw 'Complete the v3 data build in docs/Dataset_Card.md before continuing training.'
    }
    if (-not (Test-Path -LiteralPath "$parent/result.json" -PathType Leaf -ErrorAction Stop)) {
        throw "Complete the v3 training run at $parent before continuing training."
    }
    & $python -m unittest discover -s tests -p 'test_span_edits.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Span editing regression test failed. See the output above.' }
    foreach ($weight in @('1.0', '2.0')) {
        $run = "runs/six_layer_joint_v8_academic_pairs_v3_payload_$weight"
        & $python scripts/train_six_layer.py --config $config --init-checkpoint $parent --output $run --steps 1000 --seed 43 --span-payload-weight $weight --checkpoint-selection best-validation --early-stopping-patience 4
        if ($LASTEXITCODE -ne 0) { throw "v3 continuation failed at payload weight $weight. See the output above." }
        & $python -m academic_rewriter.test --checkpoint $run --data $data --split validation --batch-size 8 --max-new-tokens 194 --limit 0 --span-copy-scale 1.0 --output "$run/evaluation/validation-194"
        if ($LASTEXITCODE -ne 0) { throw "v3 validation failed at payload weight $weight. See the output above." }
    }
}
```

Check `selected_step` in each result. Step zero corresponds to the parent weights. Compare edit corrections and factual preservation on the same validation cases before choosing the checkpoint for the test split. The continuation and both decoding comparisons await manual execution.

## Evaluate the entire split

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $data = 'data/academic-pairs-v1'
    $run = 'runs/six_layer_joint_v6_edits'
    if (-not (Test-Path -LiteralPath "$run/result.json" -PathType Leaf -ErrorAction Stop)) {
        throw 'Complete the v6 training block before evaluation.'
    }
    foreach ($split in @('validation', 'test')) {
        & $python -m academic_rewriter.test --checkpoint $run --data $data --split $split --batch-size 8 --max-new-tokens 192 --limit 0 --output "$run/evaluation/${split}-192"
        if ($LASTEXITCODE -ne 0) { throw "v6 $split evaluation failed. See the output above." }
    }
}
```

The command uses `data/academic-pairs-v1`. A zero limit evaluates all 98 validation cases or all 87 test cases. Evaluation loads a completed checkpoint with `result.json`, `six_layer_config.json`, `tokenizer/` and `weights.pt`.

Each command writes `CHECKPOINT/evaluation/SPLIT-192/predictions.csv`, `predictions.jsonl` and `metrics.json`. With a fused checkpoint, each row aligns the source, human reference and final `prediction`. JSONL adds observed budget hits and source/reference truncation. A budget hit alone does not prove a sentence is incomplete.

Start the qualitative review with the LayoutLM preposition edit and the corpus description reporting 7,000 documents. Check that the raw generation retains the model name, technical relationships and number while making the local edit.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $ids = '00a677997fbda9ac', '4766dceaeff49be0'
    $predictions = 'runs/six_layer_joint_v6_edits/evaluation/validation-192/predictions.jsonl'
    Get-Content -LiteralPath $predictions -Encoding UTF8 -ErrorAction Stop |
        ForEach-Object { ConvertFrom-Json -InputObject $_ } |
        Where-Object { $_.id -in $ids } |
        Format-List id, source, reference, prediction
}
```

Repeated evaluation writes to the specified directory. This command exports a second set of v6 validation predictions to a dedicated directory. Checkpoints and predictions are local runtime artifacts and are ignored by Git.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $run = 'runs/six_layer_joint_v6_edits'
    & $python -m academic_rewriter.test --checkpoint $run --data 'data/academic-pairs-v1' --split validation --batch-size 8 --max-new-tokens 192 --limit 0 --output "$run/evaluation/validation-192-repeat"
    if ($LASTEXITCODE -ne 0) { throw 'v6 validation export failed. See the output above.' }
}
```

To inspect the contribution of E1–E3 during generation, run the command below on the same v6 checkpoint. Compare `evaluation/validation-192-no-word-guidance/predictions.csv` with `evaluation/validation-192/predictions.csv` by ID. Check whether the required edit was made and whether nearby terms, numbers and clauses were preserved.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $run = 'runs/six_layer_joint_v6_edits'
    & $python -m academic_rewriter.test --checkpoint $run --data 'data/academic-pairs-v1' --split validation --batch-size 8 --max-new-tokens 192 --limit 0 --no-word-guidance --output "$run/evaluation/validation-192-no-word-guidance"
    if ($LASTEXITCODE -ne 0) { throw 'v6 validation without word guidance failed. See the output above.' }
}
```

To measure the effect of the replacement vectors, train a position-only checkpoint with the same data and seed, then evaluate it with the same decode settings. `metrics.json` records `candidate_guidance`.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $run = 'runs/six_layer_joint_v6_edits_position'
    & $python scripts/train_six_layer.py --config 'configs/six_layer_experiment.json' --word-guidance position --output $run
    if ($LASTEXITCODE -ne 0) { throw 'Position-only training failed. See the output above.' }
    & $python -m academic_rewriter.test --checkpoint $run --data 'data/academic-pairs-v1' --split test --batch-size 8 --max-new-tokens 192 --limit 0 --output "$run/evaluation/test-192"
    if ($LASTEXITCODE -ne 0) { throw 'Position-only test evaluation failed. See the output above.' }
}
```

## Data reproduction and checks

The generator command creates a separate local study at `.cache/academic-pairs-synthetic-v2` using the frozen IteraTeR splits as input. It uses a new path to preserve the published v1 data.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $study = '.cache/academic-pairs-synthetic-v2'
    if (Test-Path -LiteralPath $study -ErrorAction Stop) {
        throw "Use a new versioned study path. $study already exists."
    }
    & $python -m academic_rewriter.synthetic --original 'data/processed' --output $study --count 12000 --seed 42
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic dataset generation failed. See the output above.' }
}
```

For a changed corpus, use a new configuration and output directory that point to its new manifest.

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

Tests cover causal isolation, source copy probabilities, edit alignment and gradients, cached/full-logit agreement, padding, balanced sampling, target-only loss, checkpoint reload and continuation, full prediction exports and synthetic-data isolation. Run the tests and study commands manually.

## Evidence boundaries

The real validation/test cases have already been inspected and are diagnostic rather than fresh confirmation. A changed output is not automatically an improvement. Overlap, copy rate and number preservation do not prove academic style, fluency or factual equivalence. Independent quality ratings and a successful-rewrite rate remain unavailable.
