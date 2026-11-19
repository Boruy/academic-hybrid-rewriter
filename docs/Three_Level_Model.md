# Three rewrite levels

Train the shallow, medium and deep outputs in one run and select the output mode at inference. The four internal prediction stages share the six sequence blocks and follow the supplied segmentation diagram's cascade. Each of the first three predictions supplies a word boundary guide for the following stage.

| Level | Model | Scope |
| --- | --- | --- |
| Shallow | P2 after the second LSTM | Predicts a replacement or retain for each eligible word after using P1 to locate potential edits. Each replacement preserves the surrounding text, including punctuation and spacing. |
| Medium | P3 after the two Transformer blocks | Generates a revised sentence using the first two word guides. P3 also emits a source word score for the final guide. |
| Deep | P4 after the two Mamba blocks | Generates a revised sentence using all three word guides. |

## Architecture and loss

```mermaid
flowchart LR
    S[Source tokens] --> L1[First LSTM block] --> P1[P1 coarse word prediction]
    P1 --> E1[E1 word boundary guide] --> F2[Word-guided fusion]
    L1 --> F2 --> L2[Second LSTM block] --> P2[P2 shallow word output]
    P2 --> E2[E2 word boundary guide] --> F3[Word-guided fusion]
    L2 --> F3 --> T[Two Transformer blocks] --> P3[P3 medium sentence and word scores]
    P3 --> E3[E3 word boundary guide] --> F4[Word-guided fusion]
    T --> F4 --> M[Two Mamba blocks] --> P4[P4 deep sentence output]
```

Each prediction has a `MidFPred` projection with two linear layers and GELU. The shared single-word edit module takes the place of CDEA. At P1, P2 and P3 it reads the original word span and its context from the current source features, then predicts a replacement or retention. It places the probability of a change on the source subwords. One-dimensional dilation minus erosion highlights their boundaries. The resulting map reweights source features before the next stage. P3's word scores come from the same stage features as its sentence prediction. The guides use only the input sentence during training and inference.

Each optimizer update uses a batch of [sentence pairs](../data/academic-pairs-v1/README.md) and a batch of [word pairs](../data/word-formality-v1/README.md). The default loss is `0.5 × CE(P3 sentence) + 0.5 × CE(P4 sentence) + 0.25 × mean(CE(P1 word), CE(P2 word), CE(P3 word))`. The sentence objectives supervise reference output tokens. The word objectives supervise the labeled replacement or retain decision at each word readout. The logs record all three word losses. The mean medium and deep cross-entropy on real validation data selects the checkpoint.

## Train once and select a mode

Run from the repository root in PowerShell. Use a fresh checkpoint directory for a new run.

```powershell
$python = '.\.venv\Scripts\python.exe'
$checkpoint = 'runs/six_layer_joint_v2'
$sentence = 'The proposed method gets better results than the baseline.'
& $python scripts/train_six_layer.py --output $checkpoint
foreach ($depth in 'shallow', 'medium', 'deep') {
    & $python -m academic_rewriter.rewrite --depth $depth --checkpoint $checkpoint --text $sentence
}
```

Joint checkpoints record the stage layout and replacement vocabulary in `six_layer_config.json`. Earlier joint and single-depth checkpoints load with their saved layout, and standalone word artifacts load with `--lexical-model`. Start a fresh run to train the four-stage layout. For medium and deep, the single-sentence command returns the source if the candidate is empty, lacks an end token, or changes the numeric strings. The shallow output records each edited word and its original character offsets and retains words outside the input token budget.

## Test one checkpoint across all three modes

```powershell
& $python -m academic_rewriter.test --checkpoint $checkpoint --split test --limit 0
& $python -m unittest discover -s tests -p test_joint_rewriter.py -v
```

The test command evaluates all 87 real test pairs by default. Use `--split validation` for the 98 validation pairs. The CSV aligns the original text, human reference and all three predictions on each row. JSONL includes shallow edits and separate generation diagnostics. `metrics.json` reports metrics by mode, a copy-input baseline and the cumulative block counts of the prediction stages (`[1, 2, 4, 6]` for the new model). The batch test exports raw neural generations for diagnostics. The existing real splits have already been inspected and support exploratory comparison.

To score shallow replacement decisions, pass `--word-test .cache/reviewed_word_pairs.jsonl` with distinct, independently reviewed real cases. Each row needs `source` and `target`. For retain cases, include `focus` and, for repeated words, `focus_start`. Report focused replacement accuracy, unnecessary edit rate and exact sentence accuracy. The included word corpus is training only.

## Evaluation and loss informed by Doolittle

Diao et al. evaluate paragraph-level academic formalization with several measures. Apply the same dimensions to this project's sentence-level task and report them separately. Paragraph scores from the original benchmark are specific to its data and evaluation models. See [Doolittle, Sections 4.1 and 6.2](https://aclanthology.org/2023.emnlp-main.809.pdf).

| Dimension | Doolittle measure | Use for the three levels |
| --- | --- | --- |
| Academic formality and edits | CoLA and AESW RoBERTa acceptability classifiers (`ACC-cola`, `ACC-aesw`), plus SARI and GLEU | Prioritize change in formality from input to output. Use AESW as a scientific-writing proxy and CoLA as a grammatical acceptability check. SARI and GLEU complement review when human rewrites are available. |
| Fluency | Perplexity (`PPL`) under GPT-2 Large tuned on formal academic writing | Use the same external language model for source and output. Report lower PPL as better under that model. |
| Meaning preservation | Subword embedding similarity to input (`SIM-input`) and human rewrite (`SIM-gold`) | Report both. A copy can score highly against its input. Review numbers, units, negation, uncertainty and claims separately. |
| Conditional generation quality | BARTScore (`BARTS`) | Record the scoring direction and fixed scorer checkpoint. Treat the score as a supplementary diagnostic. |

The current evaluator provides ROUGE-L, optional BLEU, copy rate and number preservation as diagnostics. The external Doolittle measures above describe a proposed comparison. Pin scorer checkpoints, tokenization and score direction for any experiment using them. Report the original input as a copy baseline and give medium and deep the same inputs and decoding budget.

For medium and deep, use two independent reviewers with system names hidden. Rate academic formality by precision and register, fluency by grammar and coherence, and meaning preservation by comparing output with input. Use a 1–5 scale for each dimension and log changed facts separately. Score input formality and fluency too, include already acceptable inputs, and tag expected edit scope before inspecting outputs. Doolittle used GPT-4 for separate 1–5 judgments on a test subset. Report any such scores separately from independent human judgments. See [Doolittle, Section 6.2 and Appendix E](https://aclanthology.org/2023.emnlp-main.809.pdf).

The joint supervised checkpoint is the baseline for this comparison. Validation cross-entropy measures prediction of reference tokens. External PPL and formality scores measure properties of the generated text.

Doolittle applies **metric-oriented reinforcement learning (MORL)** after preparing a rewriting policy. Its reward combines the AESW positive-class logit, output PPL, SIM-input and BARTScore. PPO updates the policy using this reward on generated rewrites and uses a KL penalty relative to the pre-RL policy. For example, with AESW as the classifier, the published Galactica notebook computes `0.01 * BARTScore(input, output) - PPL(output) / 200 + 0.01 * SIM(input, output) + 0.01 * AESW_logit(output)`. This formula describes the example implementation. See [Doolittle, Section 5 and Appendix B](https://aclanthology.org/2023.emnlp-main.809.pdf) and the [authors' Galactica notebook](https://github.com/shizhediao/Doolittle/blob/main/MORL-Galactica.ipynb).

Use the metrics first to compare the supervised checkpoints. A separate MORL experiment can start from the same supervised deep checkpoint, compare against that starting checkpoint, and choose reward scales and checkpoints on validation data. Use the final test split after selection. Report each score and factual-error rate alongside the reward so a fluency or formality gain remains visible alongside any loss of meaning. A fresh confirmatory comparison requires separately versioned, independently reviewed data.
