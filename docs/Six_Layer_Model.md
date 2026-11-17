# Six-layer academic rewriter

This version implements the requested **six sequence layers in total**:

| Layer | Module | Function |
| --- | --- | --- |
| 1 | Unidirectional LSTM | Local sequential features and recurrent memory |
| 2 | Unidirectional LSTM | Further recurrent composition |
| 3 | Transformer block | Causal multi-head self-attention and feed-forward network |
| 4 | Transformer block | Further contextual interaction |
| 5 | Mamba block | Selective state-space sequence mixing |
| 6 | Mamba block | Final selective sequence mixing |

LSTM is the chosen RNN variant. Each block has normalization and a residual path.
Token, position and source/target segment embeddings precede the six blocks; output
normalization and a tied vocabulary projection follow them. These are not additional
sequence layers. **No BART encoder or decoder is appended.** The historical BART and
BART + Mamba models remain available as separate experiment types.

## Conditional generation

The model is trained as a causal conditional language model. A training sequence is
`[BOS] source [EOS] [BOS] target [EOS]`. Only target tokens contribute to the shifted
cross-entropy loss; source tokens provide context. The attention mask prevents any
position from seeing future target tokens. There is no bidirectional RNN.

Generation consumes the source prefix and then greedily predicts target tokens.
LSTM and Mamba states are cached, and Transformer blocks retain their past normalized
states. Padding does not update recurrent caches. Tests compare cached logits with
full-sequence logits, check future-token isolation, exercise different padding lengths,
verify gradients in every block, and train/save/reload/continue a tiny six-layer model.

The Mamba backend is the existing plain-PyTorch Mamba-1 reference recurrence, not an
optimized official CUDA kernel. No speed or numerical-parity claim is made about the
official implementation. The Transformer blocks retain quadratic attention.

## Generated training pairs

The official GYAFC and Doolittle repositories require data access requests; their
published counts have not been represented as downloaded training examples. Following
the user's authorization to generate data, `academic_rewriter.synthetic` constructs
**12,000 synthetic pairs** from **48 authored semantic frames** across **24 technical
domains**. Some examples combine two compatible descriptions. Facts and quantities
are hypothetical, rather than extracted claims about real studies.

Each pair renders the same semantic slots as less academic and academic expressions.
Records retain frame IDs, slots, protected technical terms, text hashes, generator
version and the explicit flags `synthetic=true`, `human_verified=false`, `split=train`.
Guards compare quantities, negation, scope and uncertainty markers, protected-term
counts, identities and normalized exact held-out text. These checks do not establish
independent semantic validity. The corpus is template-generated weak supervision;
12,000 variations do not represent 12,000 independent human writing situations.

The prepared dataset has **12,491 training pairs** after adding the original 491
real, mixed-intent revision pairs. The original 98 validation and 87 test pairs are
copied byte-for-byte. No synthetic example enters validation or test, and the frozen
arXivEdits study is unchanged. This synthetic study does not approve its candidate
pairs, fill its independent reviewer packets or bypass its human-curated launcher.

The new tokenizer is byte-level BPE, fitted to training text only with a full byte
alphabet and a 4,096-token vocabulary. It is not the pretrained BART tokenizer.
Consequently, its token CE values cannot be directly ranked against historical BART
CE. The model starts from random weights; retaining two Transformer blocks does not
mean retaining BART's pretrained six-layer encoder or decoder.

## Reproduction

Preserve existing data and weights. Use a new versioned output for a changed study.
Commands are provided for the user to execute. Project code, tests and report builders
must not be run by the agent without an explicit execution request. The prepared
`data/academic-pairs-v1` files already exist, so skip the first-creation command below.
The default training output is the fresh `runs/six_layer_manual_v1` directory.

```powershell
# First creation only; a nonempty destination is rejected.
.\.venv\Scripts\python.exe -m academic_rewriter.synthetic --output data/academic-pairs-v1 --count 12000
.\.venv\Scripts\python.exe scripts/train_six_layer.py --print-command
.\.venv\Scripts\python.exe scripts/train_six_layer.py
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_manual_v1 --data data/academic-pairs-v1 --split validation --batch-size 8 --max-new-tokens 192
.\.venv\Scripts\python.exe -m academic_rewriter.test --checkpoint runs/six_layer_manual_v1 --data data/academic-pairs-v1 --split test --batch-size 8 --max-new-tokens 192
.\.venv\Scripts\python.exe scripts/report_six_layer.py
.\.venv\Scripts\python.exe -m academic_rewriter.demo --checkpoint runs/six_layer_manual_v1 --text 'We tried out the classifier on 100 samples.' --force --device cuda --max-new-tokens 192
```

For another run, pass `--output runs/six_layer_manual_v2` to the launcher and
`--run runs/six_layer_manual_v2 --output output/six-layer-v2` to the report builder.
To run tests yourself:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The repository's historical-report checks are also user-run commands. If scoring
inputs or predictions change, run these in order, then inspect the generated reports:

```powershell
.\.venv\Scripts\python.exe -m academic_rewriter.assess --neural
.\.venv\Scripts\python.exe scripts/report_multidimensional.py
.\.venv\Scripts\python.exe scripts/report_stage_two.py
.\.venv\Scripts\python.exe scripts/report_refinement.py
```

The versioned configuration uses width 192, four attention heads, feed-forward width
768, dropout 0.1, batch 16, LR 3e-4, seed 42, maximum source/target length 192, up to
2,000 updates, validation every 250 updates, best-validation checkpoint selection and
early stopping after four unimproved checks. Truncation is rejected. Each epoch is
shuffled without replacement. The launcher validates split and generator fingerprints.

## Evidence boundaries

Running `scripts/report_six_layer.py` writes `output/six-layer-v1/index.html` and its
numerical sources. The builder reads actual training and prediction files, checks
alignment and recalculates saved overlap/copy/number metrics. Its chart separates
training CE from real validation CE. It includes actual outputs rather than presenting
training targets as model predictions. This new report has not yet been generated or
visually verified; its execution is left to the user.

The old real validation/test material has already been inspected and is diagnostic.
A changed output is not automatically an improvement, and number preservation cannot
prove factual equivalence. Independent academic-expression, fluency and meaning
assessments remain unavailable. No synthetic-held-out success, architectural advantage
or improved academic rewriting quality is inferred from a lower training loss.
