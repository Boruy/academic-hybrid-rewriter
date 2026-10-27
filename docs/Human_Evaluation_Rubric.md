# Blinded human evaluation rubric

Review independently. Each packet contains the source and one candidate, without system identity, automatic scores or another reviewer's decisions. At least two distinct people should rate the same candidates. File IDs cannot prove that the people were independent; the study organizer must document this.

The initial packet samples 24 of the 87 already inspected test cases, with three model outputs plus copy-source and human-reference controls: 120 candidates per reviewer. This is a diagnostic pilot, not the proposed fresh 100–200-example confirmatory study. Model comparisons and the private mapping must not be shown to reviewers. References are controls, not adjudicated guarantees of equivalent meaning.

## Meaning preservation (1–5)

| Score | Anchor |
| --- | --- |
| 5 | All material claims, relations, qualifications, numbers and entities are preserved. |
| 4 | Meaning is preserved, with only a minor nonmaterial nuance. |
| 3 | Noticeable omission, addition or changed qualification; some important meaning is retained. |
| 2 | A major claim, relation, condition or factual detail changes. |
| 1 | Opposite, unrelated or largely incorrect meaning. |

Check negation, modality (may vs does), certainty, statistical significance, comparisons, quantities, citations and technical terms. Similar words are not enough.

## Academic quality (1–5)

| Score | Anchor |
| --- | --- |
| 5 | Precise, objective, clear and appropriately qualified academic expression. |
| 4 | Suitable academic expression with small remaining weaknesses. |
| 3 | Mixed quality; vague, conversational, inflated or imprecise wording remains. |
| 2 | Multiple substantial issues with precision, objectivity or academic expression. |
| 1 | Clearly unsuitable expression for the intended academic context. |

Long words, passive voice and unsupported certainty do not automatically improve academic quality. Rate appropriateness to the claim and audience, rather than formality alone.

## Fluency (1–5)

| Score | Anchor |
| --- | --- |
| 5 | Natural, grammatical and easy to read, with appropriate punctuation. |
| 4 | Fluent overall, with a minor local issue. |
| 3 | Several awkward constructions or grammatical issues affect reading. |
| 2 | Major errors or fragments substantially impair readability. |
| 1 | Largely unintelligible or grammatically broken. |

## Academic improvement relative to the source (−2 to +2)

−2: clearly worse; −1: somewhat worse; 0: no academic improvement; +1: a small appropriate improvement; +2: a clear appropriate improvement. A source can already be academic, so high absolute quality does not necessarily mean improvement.

## Major factual error

Choose explicitly whether the candidate introduces a major factual error relative to the source. An unsupported certainty/significance claim, reversed negation, wrong quantity, altered entity, or substantive invented/omitted claim can qualify. If you cannot assess the technical claim, leave the item incomplete and explain the uncertainty; do not silently select No.

## Draft successful-rewrite rule

The configurable pilot rule requires every reviewer to give meaning, academic quality and fluency at least 4/5, academic improvement at least +1, and no major factual error. The candidate must differ from the source. Identical copying cannot become a successful rewrite through high absolute scores. An adequate unchanged source may be an acceptable no-op, but is reported separately from successful rewriting.

These thresholds are an experimental proposal, not a field standard. The software reports only a provisional rate until the study records that the rubric/thresholds were calibrated. Missing ratings are never imputed. Two reviewers must complete every candidate for a model before its overall rate is computed. Calibration should use a separate pilot set before confirmatory scoring.

## Operating the packet

Open your rater HTML file locally. Choose each score explicitly; no score is prefilled. Download the JSONL regularly and before closing the page. The resume input loads your saved JSONL only when packet identities and source/candidate text match. Ratings are not transmitted. The organizer should receive completed JSONL files through the agreed study process; this tool does not send them.

The organizer summarizes ratings using academic_rewriter.human_eval summarize and the private key under .cache. The key is excluded from GitHub. The summary reports pairwise quadratic weighted kappa (undefined constant marginals remain null), model means, provisional success, and paired document-group bootstrap intervals. With fewer than two completed independent reviewers, success remains unavailable.
