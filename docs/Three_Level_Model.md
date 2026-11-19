# Three rewrite levels

| Level | Model | Scope |
| --- | --- | --- |
| Shallow | Contextual word model | Replaces individual informal words and can keep a word when context favors retention. The rest of the input stays byte for byte intact. |
| Medium | 2 LSTM and 2 Transformer blocks | Generates a revised sentence using the first four sequence blocks, with its own training and checkpoint. Intended for local rephrasing. |
| Deep | 2 LSTM, 2 Transformer and 2 Mamba blocks | Generates a revised sentence using all six sequence blocks. Intended for broader sentence restructuring. |

The neural modes use the same sentence-pair objective and corpus. The active blocks define the neural depth. Edit scope in generated text is an empirical property to measure. Each neural checkpoint saves its depth in `six_layer_config.json`. Old six-layer checkpoints default to deep. A continuation uses the depth of its parent checkpoint. The medium model retains the six-block weight layout and skips the inactive Mamba blocks during training and decoding.

## Train and try the word model

Run these PowerShell commands from the repository root. Use a fresh artifact path for a new fit.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m academic_rewriter.lexical train --data data/word-formality-v1/train.jsonl --output .cache/three_level_v1/word_model.json
& $python -m academic_rewriter.rewrite --depth shallow --lexical-model .cache/three_level_v1/word_model.json --text 'The proposed method gets better results than the baseline.'
```

The training examples teach both substitutions and retention. The model learns replacements and local context cues from the [word corpus](../data/word-formality-v1/README.md). Unsupported or ambiguous contexts can leave the original wording intact. The JSON output shows the source, prediction and original character offsets for each replacement. Word-level training uses the versioned word corpus separately from the frozen sentence splits.

For a word evaluation, prepare a new, independently reviewed JSONL file. Each row has `source` and `target`. A target can replace exactly one word or copy the source. Include `focus` on retain rows to identify the ambiguous word and add `focus_start` when it occurs more than once. Keep review material in `.cache` and exclude the training examples. The included word corpus provides training supervision only.

```powershell
& $python -m academic_rewriter.lexical evaluate --model .cache/three_level_v1/word_model.json --data .cache/three_level_v1/reviewed_word_pairs.jsonl
```

## Train and try the neural modes

Use the sentence launcher to verify the existing frozen data and train into fresh directories. If a completed deep checkpoint is already available, use its path in the deep rewrite command.

```powershell
& $python scripts/train_six_layer.py --depth medium --output .cache/three_level_v1/medium
& $python scripts/train_six_layer.py --depth deep --output .cache/three_level_v1/deep
& $python -m academic_rewriter.rewrite --depth medium --checkpoint .cache/three_level_v1/medium --text 'The proposed method gets better results than the baseline.'
& $python -m academic_rewriter.rewrite --depth deep --checkpoint .cache/three_level_v1/deep --text 'The proposed method gets better results than the baseline.'
```

The single-sentence command returns the source if the candidate is empty, lacks an end token, or changes the numeric strings. It reports the reason and candidate in JSON. Assess meaning and academic tone separately.

## Check the implementation and compare checkpoints

```powershell
& $python -m unittest discover -s tests -v
& $python -m academic_rewriter.test --checkpoint .cache/three_level_v1/medium --data data/academic-pairs-v1 --split validation --batch-size 8 --max-new-tokens 192 --limit 0
& $python -m academic_rewriter.test --checkpoint .cache/three_level_v1/deep --data data/academic-pairs-v1 --split validation --batch-size 8 --max-new-tokens 192 --limit 0
```

The split evaluator exports raw neural generations for diagnostics. The existing real splits have been inspected and can support exploratory comparison. Independent reviews are still needed to establish style improvement, meaning preservation and the appropriate edit scope for each level.
