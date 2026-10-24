# IteraTeR academic pilot data

Prepared 2026-10-24. Source: https://huggingface.co/datasets/wanyu/IteraTeR_human_sent . Author repository: https://github.com/vipulraheja/iterater . Paper: https://arxiv.org/abs/2203.03802 .

## Provenance

Revision: e22e0371dac444239b944f9293f5b491d62b73f0. Full file hashes and byte counts are in data/raw/manifest.json. The card lists Apache-2.0; source text originates from several domains and attribution should remain attached. Raw and processed corpora stay outside Git. The downloadable human sentence release contains 3,254 train, 400 dev and 364 test records (4,018 total), not the much larger FULL or v2 releases.

## Selection

Use doc_id patterns for arXiv identifiers, retain style / clarity / fluency, normalize Unicode NFKC and whitespace, remove identity or duplicate pairs, require 5–100 whitespace words on each side and a target/source word ratio of 0.5–2.0, and reject changed numeric multisets. Intent labels guide selection but do not certify academic tone or factual equivalence.

{
  "excluded_intent": 449,
  "length_ratio": 14,
  "number_change": 22,
  "word_length": 7,
  "empty_or_identity": 9,
  "non_arxiv_document": 2841
}

676 pairs remain, from 95 connected document groups. Combine the downloaded splits and rebuild 80/10/10 **by groups** with seed 42. This differs from the original proposal's nominal sentence proportions. Groups connect all revisions of a document and documents sharing an exact normalized source or target. Actual pair counts need not equal 80/10/10. Do not compare results directly with published official-split scores.

| Split | Pairs | Documents | Clarity | Fluency | Style |
|---|---:|---:|---:|---:|---:|
| Train | 491 | 76 | 355 | 123 | 13 |
| Validation | 98 | 9 | 69 | 26 | 3 |
| Test | 87 | 10 | 67 | 12 | 8 |

All cross-split document and exact-text overlaps are zero. This does not exclude semantic near duplicates, paper-family overlap, or undocumented provenance errors. Upstream document overlaps in the downloaded release are 13 for train/dev and 16 for train/test, which motivates re-splitting.

## Real example

Source: Discovering new intents is a crucial task in a dialogue system .

Reference: Discovering new intents is a crucial task in dialogue systems .

Intent: style. arXiv document 2012.08987. Upstream train.json, line 778. This is an upstream human edit, not a model prediction. Its style label does not establish that every reader will prefer the revised wording.

## Intended use and limits

This is a small pilot for scientific sentence revision. Only 24 pairs carry a style label, so most supervision concerns clarity or fluency. Number preservation cannot catch changed negation, hedging, entities or citation meaning. Data include imperfect edits and LaTeX artifacts. Manual review is pending. No synthetic training pairs have been added. The test split remains held out from model evaluation; counts and leakage checks inspect metadata/text only.

The tiny tokenizer learns its vocabulary on training text only. Source and target each receive BOS/EOS tokens. Dynamic right padding uses attention masks. Target padding becomes -100 for cross-entropy. Current smoke samples have zero truncation at 128 tokens. This word-level tokenizer is for pipeline verification; use the pretrained BART tokenizer for the main experiments.
