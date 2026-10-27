# Multidimensional academic rewriting evaluation

The project evaluates content preservation, academic improvement and fluency separately. Reference overlap and edit size remain diagnostics. No single automatic metric establishes a successful rewrite.

## Current measured and pending dimensions

| Dimension | Implemented measure | Current evidence |
| --- | --- | --- |
| Content preservation | Source–output BERTScore F1; both NLI directions; minimum and geometric mean entailment | Measured for 87 cases × three models and two controls |
| Academic improvement | Independent binary classifier, calibrated class probabilities and source→output gain | Training/scoring code ready; independent human labels unavailable |
| Fluency | Independent human rating | Blank blind-review packets ready; no independent grammar/PPL score claimed |
| Reference similarity | Reference–output BERTScore, whitespace ROUGE-L and signed/versioned corpus SacreBLEU | Measured |
| Revision amount | Word-token Levenshtein edit rate, exact copying, unchanged word sequence | Measured |
| Successful rewriting | Calibrated multidimensional human rule | Pending independent ratings and threshold calibration |

The results are in [output/multidimensional](../output/multidimensional/README.md) and the [interactive report](../output/stage-two/index.html#multidimensional). The present data is already inspected and is labeled inspected_test_diagnostic. Generator weights and the saved 128-token predictions are unchanged. New metrics cannot themselves repair rewriting quality.

## Semantic metrics

BERTScore uses distilbert/distilroberta-base at exact commit fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b, layer 5, no IDF, no baseline rescaling and slow tokenization. This is an explicitly chosen lightweight pilot encoder, not the default or highest-capacity encoder recommended by the BERTScore authors. Source and reference comparisons are separately named. Raw F1 is not a calibrated probability of equal meaning.

NLI uses cross-encoder/nli-deberta-v3-small at exact commit fa2804872c3b4bd748f38c0185cc85775361e735. Its documented label order is contradiction=0, entailment=1, neutral=2; code checks semantic labels against the model config. For each source/output pair, the report retains both entailment and contradiction probabilities in both directions, plus the minimum and geometric mean of the entailment probabilities. This generic bidirectional NLI diagnostic is not the specialized MIS model introduced by Babakov et al. No universal entailment threshold is assumed.

All scoring runs locally. Input lengths are checked before scoring; an overlength input fails rather than being silently truncated. The summary records model-file hashes, exact revisions, package versions, device and fingerprints of configs, scoring code, predictions and the data splits. Numbers, negation, hedging and citation changes are separate heuristic review prompts, not semantic verdicts.

The copy-source control deliberately demonstrates that excellent preservation can coincide with zero revision. Human-reference controls are not assumed to preserve every source claim: revisions may alter information, and automatic NLI also has domain limitations. Their lower NLI values require manual inspection, rather than a declaration that every reference is wrong.

## Paired comparisons and splits

Every model must cover the same unique IDs and exactly match the source/reference text. The evaluator aligns by IDs and rejects missing, duplicate or mismatched records. The three generators use the same saved splits and decoding conditions. Classifier annotations must be separately document-disjoint from all generator splits.

The default 95% intervals use 1,000 bootstrap draws of the 10 connected document groups with seed 42. All sentence pairs within a sampled group stay together. Model differences use matched cases and group resamples. These are descriptive intervals for a small pilot, not evidence of architectural superiority or a substitute for multiple training seeds.

For a future confirmation study, supply a new data/prediction config with evaluation_status=fresh_confirmatory. The loader rejects document IDs, group IDs or normalized exact texts that overlap the existing training, validation or inspected test sets. Exact checks cannot rule out all semantic near duplicates or undisclosed external overlap. Choose 100–200 genuinely new examples and document the sampling process. The current 87 cases must not be relabeled as fresh.

## Independent academic classifier

The existing style intent label means a revision category, not Academic vs Non-academic. The heuristic detector and an unvalidated formality classifier are not replacements. No numeric academic score is filled in while annotations are missing.

Prepare human annotations matching [configs/style_annotations.schema.json](../configs/style_annotations.schema.json): nonempty text, academic=0/1, document ID, train/validation/test split, label_source=human and guideline_version. Labels should reflect precision, objective expression and appropriate qualification across varied topics, rather than whether a sentence happens to mention scientific concepts. Keep a human label provenance record.

~~~powershell
.\.venv\Scripts\python.exe -m academic_rewriter.train_style --data path/to/independent_annotations.jsonl --output .cache/style_classifier_v1
~~~

The trainer rejects generator text/document overlap, cross-split document leakage, duplicate annotation text, missing classes and revision-intent labels. It trains a fresh binary classification head on the independent pretrained encoder, selects a checkpoint on validation cross-entropy, chooses a scalar temperature on validation data and reports the held-out classifier test metrics. Using the same validation set for selection and temperature fitting can overfit; independent calibration and generalization checks remain necessary. No test labels guide training or temperature selection.

To score a trained classifier, set style_classifier to an object with checkpoint=.cache/style_classifier_v1 in the evaluation config. The scorer rechecks annotation overlap and returns P(academic) for source/output plus their difference. These are classifier probabilities, not 1–5 academic-quality ratings or proof of improved precision and hedging.

## Independent human review

See the [anchored rubric](Human_Evaluation_Rubric.md). Two offline forms have been prepared for a 24-case diagnostic pilot: [reviewer 1](../output/human-evaluation/rater_1.html), [reviewer 2](../output/human-evaluation/rater_2.html). Each includes 120 randomly ordered candidates from three models and copy/reference controls, without system names or automatic scores. They contain no prefilled ratings.

The mapping is private in .cache/human-evaluation/key.json and is excluded from GitHub. Reviewers should receive only their HTML packet and the rubric; do not send them model comparisons or the key. For a new study, use new output/key paths so an active review cannot be overwritten.

~~~powershell
.\.venv\Scripts\python.exe -m academic_rewriter.human_eval export --sample 24 --raters 2 --output output/new_human_pilot --key .cache/new_human_pilot/key.json
.\.venv\Scripts\python.exe -m academic_rewriter.human_eval summarize --ratings path/to/rater_1_ratings.jsonl path/to/rater_2_ratings.jsonl
~~~

The second command uses the default private key for the supplied default packets; pass --key for another study. It checks packet identity and text integrity, requires explicit ordinal/boolean values, reports pairwise quadratic weighted kappa, and computes matched group-bootstrap model differences when complete. Undefined kappa remains null. Distinct people and independent work are study requirements that software cannot prove from reviewer IDs.

The draft success rule requires every rater to assign meaning, academic quality and fluency at least 4/5, academic improvement at least +1, no major factual error, and a changed output. It is intentionally labeled provisional until thresholds are calibrated. Missing or partial ratings never become a success or a fabricated zero-percent rate. The overall rate remains unavailable until all candidates for that model have at least two completed reviews. LLM-as-judge is not silently substituted for human reviewers.

## Reproduce automatic scoring

~~~powershell
.\.venv\Scripts\python.exe -m pip install -e '.[evaluation,analysis]'
.\.venv\Scripts\python.exe scripts/download_evaluation_models.py
.\.venv\Scripts\python.exe -m academic_rewriter.assess --neural
.\.venv\Scripts\python.exe scripts/report_multidimensional.py
.\.venv\Scripts\python.exe scripts/report_stage_two.py
~~~

Public model dependencies are cached locally and excluded from GitHub. After downloading, scoring uses local snapshots only and does not transmit project text. The original machine stores additional evaluation packages under .venv/evaluation-packages, preserving the shared Conda training environment. A new installation can use the standard optional-dependency command above. For a separate lexical-only check, use --output output/lexical_check to preserve the published neural assessment.

## Research basis

- [Babakov et al. (ACL SRW 2022)](https://aclanthology.org/2022.acl-srw.23/) compares content-preservation metrics and introduces the specialized MIS method.
- [Lai et al. (HumEval 2022)](https://aclanthology.org/2022.humeval-1.9/) evaluates formality transfer with human judgments across style, content and fluency.
- [Pauli et al. (Findings EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.1175/) examines limitations of metric meta-evaluation for style/attribute transfer.
- Implementation details: [BERTScore authors' repository](https://github.com/Tiiiger/bert_score), [DistilRoBERTa model card](https://huggingface.co/distilbert/distilroberta-base), and [NLI model card](https://huggingface.co/cross-encoder/nli-deberta-v3-small).
