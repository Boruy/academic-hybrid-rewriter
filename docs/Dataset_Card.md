# Academic revision datasets

The [word formality corpus](../data/word-formality-v1/README.md) supervises the word module during the same run as sentence-pair training. Its 76 synthetic examples are training only. Use separate, independently reviewed real word pairs for the optional word test.

All original dataset files, frozen splits, source hashes and curation records are retained. Dataset code is in `academic_rewriter.prepare` and `academic_rewriter.synthetic`. A changed study must use a new versioned path rather than replace frozen data.

## Active paired corpus

`data/academic-pairs-v1` contains:

| Split | Real pairs | Synthetic pairs | Total |
| --- | ---: | ---: | ---: |
| Training | 491 | 12,000 | 12,491 |
| Validation | 98 | 0 | 98 |
| Test | 87 | 0 | 87 |

The 12,000 synthetic pairs use 48 authored semantic frames across 24 technical domains. Source and target share semantic slots and describe hypothetical results. Records include provenance, hashes, protected terms and explicit synthetic/unverified training-only flags.

This is template-generated weak supervision. Shared slots and automatic guards do not establish independent semantic equivalence or academic quality. Variations in quantities and terms are not independent naturally occurring writing examples. The 491 real revisions are mixed-intent supervision rather than uniformly academic-style positives.

The corpus manifest records composition, the exact generator hash and both original and prepared split hashes. Validation and test files are byte-for-byte copies of `data/processed`. See the [corpus README](../data/academic-pairs-v1/README.md).

## Original IteraTeR data

Source: Wanyu Du et al., Understanding Iterative Revision from Human-Written Text (ACL 2022). The pinned `wanyu/IteraTeR_human_sent` release is revision `e22e0371dac444239b944f9293f5b491d62b73f0`. Source bytes and hashes are retained in `data/raw/` and `data/raw/manifest.json`; attribution and the upstream license are retained in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and `data/IteraTeR_LICENSE.txt`.

The downloaded human sentence release has 3,254 train, 400 dev and 364 test records (4,018 total). Processing selects arXiv documents and style, clarity or fluency intents; normalizes Unicode and whitespace; removes identities and duplicates; requires 5-100 words per side and a target/source ratio of 0.5-2.0; and rejects changed numeric multisets.

The retained 676 pairs are split by connected document groups with seed 42. Groups connect revisions of a document and documents sharing an exact normalized source or target. This differs from the upstream sentence splits.

| Split | Pairs | Documents | Clarity | Fluency | Style |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 491 | 76 | 355 | 123 | 13 |
| Validation | 98 | 9 | 69 | 26 | 3 |
| Test | 87 | 10 | 67 | 12 | 8 |

Recorded cross-split document and exact-text overlaps are zero. This does not exclude semantic near duplicates or paper-family overlap. Intent labels and unchanged numbers do not certify academic tone or factual equivalence. The already inspected validation/test sets are diagnostics, not unseen confirmatory evidence.

## Preserved curation records

`data/curation/iterater_audit_100.jsonl` retains the preliminary audit. `data/curation/arxivedits/` retains source references, hashes, the frozen partition, AI triage and a summary. The arXivEdits source revision is `113913d68522c1c8428d8e191d5c9bed633da6ce`.

These records do not add new paper text to the public repository. Original paper text and private reviewer mappings remain in ignored `.cache` locations. Paper-specific licenses still apply. The public records report zero independently human-verified new pairs; upstream intent signals and AI triage are not independent human approvals. The historical report and curation interfaces have been removed, but the frozen references and provenance remain intact.

## Loading and tokenization

Sentence training, validation and test records use `id`, `source` and `target`. The joint six-layer tokenizer is fitted to sentence and word training text only. Source and target receive BOS/EOS tokens, dynamic right padding uses attention masks, and target padding is ignored by cross-entropy. The versioned launcher verifies data hashes and rejects source/target truncation.

Use the included prepared files for normal training. Pinned raw downloading and data preparation are retained for provenance and creation of separately versioned studies; neither needs to run before using the existing corpus.
