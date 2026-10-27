# Multidimensional revision diagnostics

Measured on the same 87 inspected test cases across 10 connected document groups. All generator checkpoints and predictions are unchanged. Results use the 128-new-token generation profile.

| System / control | Source BERTScore F1 | Reference BERTScore F1 | Min directional NLI entailment | Word edit rate |
| --- | ---: | ---: | ---: | ---: |
| BART baseline | 0.9975 [0.9968, 0.9982] | 0.9792 [0.9748, 0.9832] | 0.8713 [0.8115, 0.9265] | 0.0861% [0.0127, 0.1707] |
| Frozen hybrid | 0.9974 [0.9968, 0.9981] | 0.9791 [0.9747, 0.9831] | 0.8705 [0.8102, 0.9259] | 0.3616% [0.1973, 0.4984] |
| Full hybrid | 0.9975 [0.9968, 0.9982] | 0.9792 [0.9748, 0.9831] | 0.8712 [0.8115, 0.9262] | 0.3081% [0.1323, 0.4713] |
| Copy source (control) | 1.0000 [1.0000, 1.0000] | 0.9815 [0.9768, 0.9856] | 0.9653 [0.9534, 0.9752] | 0.0000% [0.0000, 0.0000] |
| Human reference (control) | 0.9815 [0.9768, 0.9856] | 1.0000 [1.0000, 1.0000] | 0.5211 [0.4404, 0.6087] | 20.8010% [15.8369, 25.8587] |

Brackets are 95% percentile document-group bootstrap intervals (1,000 draws, seed 42). They describe this pilot; they do not establish an architectural advantage. Word edit rate is Unicode word-token Levenshtein distance divided by source word count; it can exceed 100% for large insertions.

## Interpretation

The full hybrid has source BERTScore near 0.9975 while changing only about 0.31% of source word tokens on average. Copying scores approximately 1.0. High preservation scores therefore do not establish useful academic revision. Do not rank overall rewriting quality using these metrics alone.

Human references receive lower NLI entailment than copies. Possible causes include genuine information changes in revisions and limitations of the NLI model on technical language. A low automatic value alone does not prove a reference is wrong. The reference control is not an adjudicated semantic gold standard.

## Metric definitions and provenance

- BERTScore uses the independent distilbert/distilroberta-base encoder, layer 5, no IDF, no baseline rescaling and the slow tokenizer. This is a lightweight pilot encoder, not the default/largest model recommended by the authors. Raw scores are not calibrated semantic-equivalence probabilities.
- NLI uses cross-encoder/nli-deberta-v3-small with explicit contradiction/entailment/neutral label IDs 0/1/2. Both source→output and output→source probabilities are retained. Their minimum and geometric mean are descriptive aggregates. This is not a reproduction of the specialized published Mutual Implication Score model.
- Input lengths are checked before neural scoring; scoring fails instead of silently truncating. The recorded run has zero truncated metric inputs/pairs.
- Summary JSON records exact model commits, model/config file SHA-256 values, library versions and source fingerprints. Per-case JSONL retains all directional values and heuristic change flags. Generic change flags are review prompts, not semantic judgments.
- Corpus SacreBLEU is included with its tokenizer/version signature. It remains a reference-overlap diagnostic.

## Dimensions still requiring evidence

- Academic style: pending independent human Academic/Non-academic labels. The classifier training and scoring pipeline is implemented, but no trained or calibrated classifier is claimed. Revision intent and generic formality are not academic-style ground truth.
- Fluency: pending independent human ratings; no causal-LM perplexity or grammar score has been fabricated. Generator teacher-forced CE is not an independent fluency model.
- Successful rewriting: pending two independent reviewers and rubric/threshold calibration. Blank forms produce null, not zero, success rates. The blinded pilot packets cover 24 inspected cases × five candidates = 120 items per reviewer. This is not a fresh 100–200-case study.

## Reproduce

~~~powershell
.\.venv\Scripts\python.exe -m pip install -e '.[evaluation,analysis]'
.\.venv\Scripts\python.exe scripts/download_evaluation_models.py
.\.venv\Scripts\python.exe -m academic_rewriter.assess --neural
.\.venv\Scripts\python.exe scripts/report_multidimensional.py
.\.venv\Scripts\python.exe scripts/report_stage_two.py
~~~

Models are downloaded to .cache/evaluation-models and are excluded from GitHub. Assessment loads only pinned local snapshots and does not upload source text. Config: configs/evaluation.json. Use a different output directory for a lexical-only diagnostic so it does not replace the published neural assessment.

See [the complete protocol](../../docs/Multidimensional_Evaluation.md), [the human rubric](../../docs/Human_Evaluation_Rubric.md) and [the interactive report](../stage-two/index.html#multidimensional).

## Research basis

- [Babakov et al., ACL SRW 2022](https://aclanthology.org/2022.acl-srw.23/): content-preservation metric comparison and the specialized MIS approach.
- [Lai et al., HumEval 2022](https://aclanthology.org/2022.humeval-1.9/): human evaluation across formality, content and fluency.
- [Pauli et al., Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1175/): limitations of metric meta-evaluation for style transfer.
- [BERTScore implementation](https://github.com/Tiiiger/bert_score) and [NLI model card](https://huggingface.co/cross-encoder/nli-deberta-v3-small).
