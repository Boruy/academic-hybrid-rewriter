# Academic expression pairs v1

**Training:** 12,000 generated pairs plus 491 existing real revision pairs, totaling
12,491. **Validation:** the original 98 real pairs. **Test:** the original 87 real pairs.
Validation/test files preserve their source bytes and contain no synthetic data.

Generated pairs describe the same hypothetical scientific content using less
academic and academic expressions. They use 48 model-authored semantic frames in
24 technical domains. The records include shared semantic slots, protected terms,
frame IDs, generator version, SHA-256 text hashes and explicit synthetic/unverified
provenance. Numbers are invented for the examples and are not real study results.

This is template-generated weak supervision, not an independently reviewed corpus.
Quantity, uncertainty, scope, negation and technical-term guards are automated checks.
They do not prove semantic equivalence or academic quality. Variation in numbers and
terms does not provide the diversity of independent naturally occurring examples.
All generated examples are training-only.

`manifest.json` records composition, frame usage, the generator code hash and exact
original/prepared split hashes. The existing real data retains the attribution in
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md). No new paper text is included.
See [the six-layer protocol](../../docs/Six_Layer_Model.md) and
[the generator](../../src/academic_rewriter/synthetic.py).
