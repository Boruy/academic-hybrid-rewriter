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

## Evaluation and loss informed by Doolittle

Diao et al. evaluate paragraph-level academic formalization with several measures. Apply the same dimensions to this project's sentence-level task and report them separately. Paragraph scores from the original benchmark are specific to its data and evaluation models. See [Doolittle, Sections 4.1 and 6.2](https://aclanthology.org/2023.emnlp-main.809.pdf).

| Dimension | Doolittle measure | Use for the three levels |
| --- | --- | --- |
| Academic formality and edits | CoLA and AESW RoBERTa acceptability classifiers (`ACC-cola`, `ACC-aesw`), plus SARI and GLEU | Prioritize change in formality from input to output. Use AESW as a scientific-writing proxy and CoLA as a grammatical acceptability check. SARI and GLEU complement review when human rewrites are available. |
| Fluency | Perplexity (`PPL`) under GPT-2 Large tuned on formal academic writing | Use the same external language model for source and output. Report lower PPL as better under that model. |
| Meaning preservation | Subword embedding similarity to input (`SIM-input`) and human rewrite (`SIM-gold`) | Report both. A copy can score highly against its input. Review numbers, units, negation, uncertainty and claims separately. |
| Conditional generation quality | BARTScore (`BARTS`) | Record the scoring direction and fixed scorer checkpoint. Treat the score as a supplementary diagnostic. |

The current evaluator provides ROUGE-L, optional BLEU, copy rate and number preservation as diagnostics. The external Doolittle measures above describe a proposed comparison. Pin scorer checkpoints, tokenization and score direction for any experiment using them. Report the original input as a copy baseline and give medium and deep the same inputs and decoding budget.

For shallow, use independently reviewed single-word examples and retain cases. Report focused replacement accuracy on cases needing a change, unnecessary edit rate on retain cases and exact sentence accuracy from `academic_rewriter.lexical evaluate`. For medium and deep, use two independent reviewers with system names hidden. Rate academic formality by precision and register, fluency by grammar and coherence, and meaning preservation by comparing output with input. Use a 1–5 scale for each dimension and log changed facts separately. Score input formality and fluency too, include already acceptable inputs, and tag expected edit scope before inspecting outputs. Doolittle used GPT-4 for separate 1–5 judgments on a test subset. Report any such scores separately from independent human judgments. See [Doolittle, Section 6.2 and Appendix E](https://aclanthology.org/2023.emnlp-main.809.pdf).

The current medium and deep models minimize **target-token cross-entropy** conditioned on the source. The shallow classifier fits local context counts for replacement and retention. Keep these as baselines. Validation cross-entropy measures prediction of reference tokens. External PPL and formality scores measure properties of the generated text.

Doolittle applies **metric-oriented reinforcement learning (MORL)** after preparing a rewriting policy. Its reward combines the AESW positive-class logit, output PPL, SIM-input and BARTScore. PPO updates the policy using this reward on generated rewrites and uses a KL penalty relative to the pre-RL policy. For example, with AESW as the classifier, the published Galactica notebook computes `0.01 * BARTScore(input, output) - PPL(output) / 200 + 0.01 * SIM(input, output) + 0.01 * AESW_logit(output)`. This formula describes the example implementation. See [Doolittle, Section 5 and Appendix B](https://aclanthology.org/2023.emnlp-main.809.pdf) and the [authors' Galactica notebook](https://github.com/shizhediao/Doolittle/blob/main/MORL-Galactica.ipynb).

Use the metrics first to compare the supervised checkpoints. A separate MORL experiment can start from the same supervised deep checkpoint, compare against that starting checkpoint, and choose reward scales and checkpoints on validation data. Use the final test split after selection. Report each score and factual-error rate alongside the reward so a fluency or formality gain remains visible alongside any loss of meaning. A fresh confirmatory comparison requires separately versioned, independently reviewed data.
