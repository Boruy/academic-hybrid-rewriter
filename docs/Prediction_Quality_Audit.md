# Prediction quality and missing-model audit

Measured on 27 October 2026 after inspection of the existing test examples. All checkpoint weights are unchanged. This is a decoding/quality diagnostic, not an independent confirmatory evaluation.

## What went wrong in the viewer

The initial viewer included BART and the frozen hybrid only. The full hybrid had a saved checkpoint and final validation CE, but its test generation had not been run. The report generator and template also assumed two prediction streams and two output cards. Both have now been expanded to all three models. Missing full-hybrid predictions were not evidence that the model was unavailable.

The old viewer showed edit markup by default. Removed source tokens appear in that mode but are not part of the model output. The corrected viewer defaults to verbatim predictions, with edit markup available as an explicit option.

## The 48-token budget hid part of the output

All 87 test sources fit inside the unchanged 128-token input limit (maximum 127 tokens). Reference lengths include tokenizer special tokens; 26/87 exceed 48 tokens, with a maximum of 108. In each historical BART/frozen-hybrid file, 37 predictions re-tokenize to exactly 48 tokens, and 31 lack final sentence punctuation. These are indicators of short-budget truncation; historical raw generation lengths were not recorded, so they are not reported as exact cap-hit counts.

All three models have now been evaluated with the same greedy, deterministic settings at 128 new tokens, and the new evaluator records actual generated lengths (excluding decoder start and batch padding), budget hits and input/target truncation. In this 128-token run, all three have zero cap hits and zero input/target truncations. The original two 48-token files were preserved; the full-hybrid 48-token run was added for a complete historical-protocol comparison.

## Removing the short budget does not fix rewriting

| Model / diagnostic | ROUGE-L at 48 | ROUGE-L at 128 | Exact copies at 128 | Same word sequence at 128 |
| --- | ---: | ---: | ---: | ---: |
| BART baseline | 0.7580 | 0.8146 | 32/87 | 83/87 |
| Frozen hybrid | 0.7566 | 0.8138 | 30/87 | 75/87 |
| Full hybrid | 0.7569 | 0.8138 | 32/87 | 77/87 |
| Return source unchanged | 0.8503 | 0.8503 | 87/87 | 87/87 |

Same word sequence means equality of lowercased Unicode word tokens (Python re.findall with the word-token pattern), ignoring punctuation and case. Only 4/87 human references meet that condition. The full hybrid therefore changes the word sequence in only 10 examples, compared with 83 human references. This is evidence that the current model is largely failing to perform the intended revision. Neither changed words nor number preservation certifies quality.

The full hybrid has lower teacher-forced test CE (0.7395) than BART (0.8213), but its free generation still underperforms the copy-source ROUGE-L diagnostic. CE measures the next reference token given preceding reference tokens; free generation follows its own previous tokens. A better CE value alone does not establish useful rewriting. These models also sometimes introduce small errors: example 031cc50894d24fba changes auto-encoders to auto-enoders rather than making the intended reference revision.

## Likely training limitations and next work

Training contains only 491 pairs, including 13 style-labeled pairs (24 across all splits). The 300 updates at batch size 2 draw 600 examples, about 1.22 training-set equivalents with random resampling. These facts are consistent with weak revision learning, but this audit does not isolate their causal contributions. Raising the token limit cannot supply missing supervision.

The next quality experiment should audit source/reference validity, expand revision supervision, record validation generation quality and the copy-source diagnostic, and save validation-selected checkpoints. Compare a longer training budget and multiple seeds under matched conditions. Keep test diagnostics separate from model selection, and use fresh held-out material for a confirmatory evaluation. Do not claim that longer training, a different decoder or the Mamba adapter has improved revision quality until measured.

## Reproduce

~~~powershell
# Use 48 explicitly to reproduce the historical protocol in a new folder.
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/bart_hybrid_unfrozen --max-new-tokens 48 --output runs/full_repeat/evaluation/test-48
# The evaluator and demo now default to 128 new tokens.
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/bart_hybrid_unfrozen --max-new-tokens 128 --output runs/full_repeat/evaluation/test-128
.\.venv\Scripts\python.exe scripts/report_stage_two.py
~~~

The three main checkpoints have evaluation/test (48-token protocol) and evaluation/test-128 (128-token diagnostic) folders. See output/stage-two/results-data.json for all model outputs, both protocols and source fingerprints. The web selector updates the tables and example explorer; exported figures use the explicitly labeled 128-token profile. The original submission PDF and presentation remain historical snapshots.
