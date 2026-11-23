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

Sentence training uses `[BOS] source [EOS] [BOS] target [EOS]`. P3 predicts the full reference and P4 predicts the edit program. Only target positions contribute to each shifted loss. The word loss comes from a separate batch of contextual word labels in the same optimizer step. Source-only word guidance is fused before the second LSTM, the Transformers and the Mamba blocks. Attention is causal, and the LSTM is unidirectional. Right padding is masked from the loss and recurrent state updates.

With `source_copy` enabled, the fused prediction learns a per-token gate between the vocabulary distribution and attention over preceding or current source tokens. The copy distribution sums mass at repeated token IDs and excludes sentence markers, padding and target tokens. P4 minimizes the negative log probability of the reference token under the mixture.

`contextual_edits` adds a generic edit decision to the shared word head so it can score spans throughout the source. Training aligns each real source and reference at word and punctuation boundaries. Replaced or deleted spans supervise `EDIT` and matched spans supervise `KEEP`. These automatic labels use the frozen real training pairs. The guides use the source at training and inference. P4 subtracts the unexpanded E3 edit weight from each source position's copy attention logit. This keeps the copy bias local to the predicted word positions.

Generation consumes the source prefix and greedily predicts target tokens. LSTM and Mamba states, the Transformers' past normalized states and the copy head's source positions are cached. The returned sequence begins with an initial BOS marker that is excluded from generated-token counts.

The Mamba module is the plain-PyTorch Mamba-1 reference recurrence, not an optimized official CUDA kernel. No speed or numerical-parity claim is made. Transformer attention remains quadratic.

## Data and tokenizer

The frozen v1 corpus supplies the original 12,000 synthetic and 491 real training pairs. The active v3 study appends author revisions. Both use the same 98 validation and 87 test pairs from `data/processed`.

Synthetic examples use 48 authored semantic frames across 24 technical domains, with shared facts, hypothetical quantities and protected terms. Records retain generator IDs, slots, hashes and the flags `synthetic=true`, `human_verified=false`, `split=train`. Automated quantity, negation, scope and uncertainty guards do not establish independent semantic validity. Synthetic examples never enter validation or test.

The byte-level BPE tokenizer is fitted to sentence and word training text with a full byte alphabet and a 4,096-token vocabulary limit. The default model starts from random weights. `--tiny` uses smaller model dimensions for pipeline checks. Tiny plain sentence models use a word-level tokenizer and span-edit models use byte-level BPE. Cross-entropy is comparable only when the tokenizer, target representation and evaluation protocol match.

Fresh training initializes the token, absolute-position and source/target segment embeddings at a standard deviation of 0.02 before adding them together. The padding token starts at zero. A fresh run records `fresh_input_embedding_std` in `result.json`. Loading a checkpoint or continuing training restores all three learned tables from its saved weights.

The [dataset card](Dataset_Card.md) documents real-data provenance and preserved curation references. The frozen arXivEdits partition and source hashes are retained; those references are not active training data or independently approved examples.

## Whole-word span editing

The v8 final head predicts ordered edit records. Each record selects a complete word range and supplies replacement text, a deletion or an insertion. Unselected text, including whitespace and punctuation, is copied verbatim. The renderer checks the edit ranges and records `edit_error` when the program fails. The JSONL export includes applied edits and their source character offsets.

The single-sentence command returns the source when an edit changes a protected quantity, cardinal number word, recognized numeric comparison or technical identifier. Batch evaluation exports raw predictions with `edit_status` and `fact_error`. See the [single-sentence command](LexiFAR_Architecture.md#generate-a-final-rewrite).

## Train on the author revision expansion

Train v8 from random weights with `.cache/academic-pairs-v3`. The completed local build contains 10,264 real and 12,000 synthetic sentence pairs, plus the separate grammar and word training supplements used by the recipe. Each sentence batch draws eight real pairs, six original synthetic pairs and two grammar pairs. The launcher verifies the corpus and loads the saved v1-trained byte-level BPE tokenizer. Validation and test contain the original 98 and 87 real pairs.

The current recipe gives P4 insertion and replacement tokens twice the weight of its edit boundaries and control tokens. Validation still uses unweighted P4 cross-entropy. The run uses seed 42 and up to 2,000 updates, checks validation every 250 updates and stops after four consecutive checks without improvement. The selected checkpoint and validation predictions go under `runs/six_layer_joint_v8_academic_pairs_v3_payload_2_best`. This weighted recipe awaits manual execution.

Run the PowerShell blocks from an installed checkout. Adjust the first `Set-Location` if your checkout is elsewhere. The training block builds v3 if needed and the pinned alignment is cached. [Environment setup](Environment_Setup.md) ? [Complete alignment download and v3 build commands](Dataset_Card.md#author-revision-expansion)

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $alignment = '.cache/arxivedits/train_alignment.json'
    $data = '.cache/academic-pairs-v3'
    $config = "$data/experiment.json"
    $run = 'runs/six_layer_joint_v8_academic_pairs_v3_payload_2_best'
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
    & $python scripts/train_six_layer.py --config $config --output $run --steps 2000 --seed 42 --span-payload-weight 2.0 --checkpoint-selection best-validation --early-stopping-patience 4
    if ($LASTEXITCODE -ne 0) { throw 'v3 training failed. See the output above.' }
    & $python -m academic_rewriter.test --checkpoint $run --data $data --split validation --batch-size 8 --max-new-tokens 194 --limit 0 --span-copy-scale 1.0 --output "$run/evaluation/validation-194"
    if ($LASTEXITCODE -ne 0) { throw 'v3 validation failed. See the output above.' }
}
```

The completed run records `selected_step` in `result.json` and saves the selected model in `weights.pt`. Check needed edits, malformed replacements and protected-expression changes alongside the copy baseline in `evaluation/validation-194`.

## Evaluate the entire split

Run this after selecting the current checkpoint on validation. It evaluates all 87 frozen test sentences and writes `predictions.csv`, `predictions.jsonl` and `metrics.json` under `evaluation/test-194`.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $data = '.cache/academic-pairs-v3'
    $run = 'runs/six_layer_joint_v8_academic_pairs_v3_payload_2_best'
    if (-not (Test-Path -LiteralPath "$run/result.json" -PathType Leaf -ErrorAction Stop)) {
        throw "Complete the v3 training run at $run before evaluating the test split."
    }
    & $python -m academic_rewriter.test --checkpoint $run --data $data --split test --batch-size 8 --max-new-tokens 194 --limit 0 --span-copy-scale 1.0 --output "$run/evaluation/test-194"
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

The two completed v8 runs used objective weights 0.5 for P3 sentence CE, 0.5 for P4 edit-program CE, 0.25 for word CE and 0.25 for source edit-position loss. The source copy head learns a gate that mixes source-token probabilities into P4. For span edits, it can supply replacement text drawn from the source. Compare the same saved weights with copy scales 1.0 and 0.5 to estimate the effect of that gate on generation.

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
