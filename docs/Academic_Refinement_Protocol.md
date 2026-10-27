# Academic Sentence Refinement

The task is **Academic Style Enhancement with Meaning Preservation**: improve the precision, concision and naturalness of a scientific sentence while retaining its facts, scope, terminology and uncertainty. Keep BART and the full BART + Mamba hybrid. Continue to optimize token cross-entropy.

The [interactive data audit](../output/academic-refinement/index.html) presents the completed preparation work. It does not claim a new trained model or improved rewriting quality.

## Recorded data audit

A seeded simple random sample (`random.Random(42)`) selects 100 of the current 676 pairs, preserving their original train/validation/test membership. The per-pair source, target, task-specific judgment and reason are in [the audit JSONL](../data/curation/iterater_audit_100.jsonl).

| Preliminary task category | Pairs |
| --- | ---: |
| Academic Style | 3 |
| Grammar / Fluency | 30 |
| Content Change / Uncertain | 50 |
| No Meaningful Improvement | 17 |

These are AI-assisted preliminary judgments. They are neither independent human labels nor an estimate of model accuracy. The narrow inclusion rule is intentionally conservative. “Uncertain” includes revisions requiring document/domain context; it does not establish that the original human revision is wrong. For example, changing *almost doubled* to *doubled* changes a qualifier, while *which is focused on* to *which focuses on* can reduce wordiness without adding a claim. Deleting an optional *both* alone is not considered sufficient academic enhancement here.

The existing corpus and all original checkpoints remain intact. Existing training pairs still supply stage-one general revision learning. The old 87 test examples remain inspected diagnostics and cannot become a fresh confirmatory test set. The three preliminary academic candidates are not silently promoted into stage-two training.

## arXivEdits source and candidate screening

[Jiang, Xu and Stevens (EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.641/) describe 751 aligned full papers and fine-grained intention annotations for 1,000 sentence pairs. These are different dataset units. The downloaded annotated-edit subset spans **363** papers; it is not 751 independently style-labeled papers.

The [upstream repository](https://github.com/chaojiang06/arXivEdits) is pinned to `113913d68522c1c8428d8e191d5c9bed633da6ce`. The preparation script downloads all three edit subsets and sentence-alignment metadata, records byte hashes and preserves paper-version license URLs. Raw paper text stays under ignored `.cache/arxivedits`; the public manifest contains references, hashes, versions, intentions and partition decisions. The upstream source collection has mixed paper-specific licenses, so the project does not apply a blanket new text-redistribution license.

Reproduce the frozen candidate pool:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_refinement.py --download
```

The original arXivEdits train/dev/test edit partitions have zero document intersections. Keep them, mapping dev to validation. Exclude any previous-experiment document or normalized exact text, previously previewed papers, duplicate pairs and cross-partition document/text bridges. Freeze this assignment before reviewing task quality or seeing model predictions. Re-running against changed source/data fingerprints fails instead of silently overwriting the frozen study.

Candidate intent signals are `Lang-professional-Improve-style`, `Lang-improve-readability-Simplify` and `Lang-accurate-spefific` (the upstream spelling is preserved). Exclude a pair if **any** annotated combination contains `Content`. The intent filter leaves 271 pairs. Conservative screens compare numeric multisets, selected negation/hedge/quantifier words, acronyms and equation/citation placeholders, and reject unchanged-word pairs and obvious extraction artifacts. They leave **199** candidates: **118 train, 43 validation, 38 test**. These guards cannot prove equivalence or detect all technical changes.

All 161 train/validation candidates were inspected in AI-assisted preliminary triage. Only **21** were shortlisted (19 train, 2 validation), all still awaiting humans. The other candidates are *not confidently qualified*, rather than human-rejected. New test text was not used in this triage and is absent from the public explorer. The goal of 200–500 reviewed target pairs has not been reached. Additional annotation of alignment-only pairs or another suitable source may be needed; do not relabel hundreds of pairs as accepted merely to reach a quota.

## Independent data curation

Generate blank offline HTML/JSONL packets. Each curator receives their own files. The test packet goes only to independent data curators, who must not tune the models or thresholds using it.

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.refinement export
```

The current packets are in `.cache/academic-refinement/review-v2/{train,validation,test}/curator_{1,2}.html`. They are deliberately unfilled. Reviewers first assess source fluency, suitability and need for refinement, then classify the revision and score meaning preservation and target fluency. Meanings, fluency and factual judgments must consider numbers, hedges, negation, domain terms, comparison groups, attribution and scope. Choose “Content Change” when equivalence is uncertain rather than forcing an academic label.

After independent completion, discuss disagreement in a separate adjudication record. Do not replace the two original decisions. The current conservative admission rule requires both curators to select Academic Style, meaning ≥4/5, fluency ≥4/5, an original needing refinement, a suitable target and no major factual error. These are project criteria, not universal thresholds. Files alone cannot establish that the two named raters are truly independent people.

Build a separately versioned local target dataset from the six downloaded review files:

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.refinement build --reviews train_curator_1.jsonl train_curator_2.jsonl validation_curator_1.jsonl validation_curator_2.jsonl test_curator_1.jsonl test_curator_2.jsonl --output .cache/academic-refinement/approved-v1
```

Use actual saved review paths in that command. The builder binds each decision to the frozen source/target hashes and partition, rejects duplicate rater/item records, and writes `pending_curation` if any positive split is empty. Blank or AI reviews cannot approve a pair. Approved positives have `needs_revision=true`. Sources that both curators judge already suitable and fluent become separate `{split}_already_suitable.jsonl` identity-target controls; they are not counted as positive academic improvements or mixed into the positive training set.

## Controlled reverse augmentation

Synthetic data is optional and **training-only**. Begin with an accepted real training target. Produce a less polished source with the same propositions, quantities, uncertainty and technical terms. Do not treat a shorter word as inherently less academic or add inflated performance claims. Do not generate from validation/test papers or change their split.

The builder accepts `--augmentation PATH.jsonl` with `id`, `parent_id`, `doc_id`, `split: "train"`, `source`, `target`, `protected_terms` and two complete `independent_reviews`. The target must equal the approved real parent's target; document identity, numeric/hedge/acronym/placeholder guards and explicit protected-term counts must match. Each synthetic review also records `source_sha256` and `target_sha256`. Synthetic pairs cannot generate other synthetic parents. No synthetic pair has been approved or included in the current pool.

## Four matched training arms

[The experiment configuration](../configs/refinement_experiment.json) specifies BART/direct, full hybrid/direct, BART/two-stage and full hybrid/two-stage. The latter two initialize from `runs/bart_baseline` and `runs/bart_hybrid_unfrozen`. The frozen-backbone hybrid remains available in the stage-two historical report; this next experiment concerns fully trainable models.

Common stage-two settings: LR 2e-5, batch 2, seed 42, full approved training/validation sets, 256-token maximum with truncation rejection, 300 maximum updates, validation every 25 updates, best-validation checkpoint selection and early stopping after four unimproved checks. Each epoch is shuffled without replacement; the final batch may be smaller. Actual examples seen and dataset-equivalent passes are logged. A single seed is an exploratory run, not a robust architectural conclusion.

```powershell
.\.venv\Scripts\python.exe scripts/train_refinement.py --print-commands
# Run after actual independent curation produces approved nonempty partitions:
.\.venv\Scripts\python.exe scripts/train_refinement.py
```

The launcher verifies approved file hashes, independent-review metadata, document isolation and train-only synthetic provenance. It never evaluates test quality. Use a new output path for each study; existing weights are protected against accidental overwrite.

The training runner now supports:

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.run --model hybrid --init-checkpoint runs/bart_hybrid_unfrozen --data .cache/academic-refinement/approved-v1 --output runs/new_hybrid_specialization --steps 300 --train-limit 0 --val-limit 0 --batch-size 2 --lr 2e-5 --max-length 256 --checkpoint-selection best-validation --eval-every 25 --early-stopping-patience 4 --sampling shuffle-epochs --reject-truncation --local-files-only
```

Continuation restores the matching tokenizer, backbone config, architecture and weights, then starts a new optimizer. It is not an interrupted-optimizer resume. Best selection persists actual weights and can select step zero if specialization is harmful; `selected_step`, `final_step_validation_cross_entropy`, selected-weight validation loss, parent hash and data hashes distinguish the outcomes. Defaults preserve final-checkpoint selection for historical commands. Validation preview generation now uses the configured length rather than a fixed 48-token cap.

The old stage-one BART and hybrid used different learning rates. Continued-checkpoint differences alone do not isolate architecture. Direct controls, matched stage-one experiments and additional seeds are needed before stronger claims. Existing weights are unchanged. Four tiny-fixture paths and selected-weight restoration are tested; **no new real specialization training or quality result has been produced**.

## Task-specific output evaluation

Freeze source-only need judgments before model outputs exist. Compare the same cases and decoding conditions across all four arms. Evaluate the fresh frozen test only after data, selection rules and provisional thresholds are finalized. Collect at least two independent blinded output ratings per candidate.

| Outcome | Eligible denominator | Decision |
| --- | --- | --- |
| Academic tone improvement | Independently rated aligned cases | Mean −2…+2 improvement, plus absolute academic score |
| Meaning preservation | Independently rated aligned cases | Mean 1…5 and major-error counts; inspect numeric/hedge/term failures |
| Effective revision rate | Sources judged to need refinement | Changed output; every rater meaning, academic and fluency ≥4, improvement ≥1, no major factual error |
| Unnecessary edit rate | Already-suitable source controls | Changed output with at least one explicit unnecessary-edit judgment |

Editing an already-good input is not automatically an error. Preserve source controls separately; do not report an unnecessary-edit rate from positive-only inputs. A copy cannot be an effective revision. Missing ratings yield unavailable values, never a fabricated 0% rate. Current thresholds remain provisional until independently calibrated. Report document-group confidence intervals, evaluator agreement and paired model differences when sufficient data exists; inspect dozens of real outputs rather than selecting only attractive successes.

`academic_rewriter.refinement_metrics` summarizes completed ratings. Its case JSONL requires `id`, `model`, `doc_id`, `source`, `prediction`, the prior `needs_revision` boolean and `source_judgment_human_verified: true`. Each rating binds `text_digest = SHA256(source + "\u0000" + prediction)` and includes `reviewer_type: "human"`, `rater_id`, `meaning`, `academic`, `fluency`, `academic_improvement`, `major_fact_error` and, for suitable-source controls, `unnecessary_edit`.

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.refinement_metrics --cases PATH_TO_CASES.jsonl --ratings PATH_TO_RATER_1.jsonl PATH_TO_RATER_2.jsonl --output output/new_refinement_ratings.json
```

The existing BERTScore/NLI/classifier/human-evaluation machinery remains available as supporting analysis. An upstream edit-intent label is not an independent academic-style classifier. No external LLM judge, paid service or generated human score is introduced here.

## Rebuild the report

```powershell
.\.venv\Scripts\python.exe scripts/report_refinement.py
.\.venv\Scripts\python.exe scripts/report_stage_two.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The public explorer shows the existing 100-pair audit only. New arXivEdits text and unblinded review data remain local. Its readiness cards explicitly separate screened candidates, AI shortlists, human-verified pairs and actual model results.
