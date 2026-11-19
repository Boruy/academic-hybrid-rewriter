# Single-word academic formalization v1

`train.jsonl` contains 76 controlled examples for the single-word module in
the joint training run. It has 56 one-word replacements and 20 retain examples,
covering 12 input word forms. Codex authored the sentences for this project using
hypothetical research settings. The examples contain no extracted paper text
and have no independent human annotations. All 76 examples are synthetic
training data, marked `synthetic=true`, `human_verified=false` and `split=train`.

Each UTF-8 JSONL record has `source` and `target` strings. In a replacement
example, one English word changes and all surrounding text stays identical.
In a retain example, the strings match and `focus` identifies the occurrence
to preserve. The surrounding words matter. For example, `get access` can
become `gain access`, `get estimates` can become `obtain estimates`, and
`get stuck` can stay as written. Retain examples also cover terms such as
`hard constraint` and `cheap talk`.

This small corpus teaches a limited set of observed word forms and contexts.
It has no validation or test split. An independent evaluation needs real
examples with held-out source documents and judgments of both appropriate
replacements and edits that should be avoided. The prepared academic revision
corpus and its frozen splits are separate from this training file.
