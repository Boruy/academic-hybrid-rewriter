# Milestone evidence

Recorded 2026-10-24. Evidence is produced by executable files in this repository. No final-quality score is claimed.

## Completed

- Pinned public IteraTeR-HUMAN download: data/raw/manifest.json.
- Data cleaning and document-disjoint splitting: data/processed/stats.json.
- 676 retained pairs, split 491 / 98 / 87.
- Train-only tokenizer, batching and label masks: src/academic_rewriter/run.py.
- Tiny Transformer and Hybrid forward/backward/generation runs on an RTX 4060 Laptop GPU.
- Reference selective SSM and gated/residual/replace fusion implementations.
- Four tests passed: normalization/number checks, document/text leakage grouping, reference-mixer masking and gradients, and all fusion paths with loss and generation. Log: runs/tests.txt.
- Local checkpoint inference and transparent heuristic detector.

## Actual smoke results

| Measure | Transformer | Hybrid reference / gated |
|---|---:|---:|
| Steps | 30 | 30 |
| Train pairs / validation pairs | 64 / 16 | 64 / 16 |
| Parameters | 254,720 | 292,672 |
| Fixed training-batch CE before | 7.7781 | 7.7781 |
| Fixed training-batch CE after | 6.5706 | 6.5703 |
| Validation token CE | 6.8413 | 6.8395 |
| Training time in this run, seconds | 0.780 | 6.976 |
| Peak allocated CUDA memory, MiB | 31.61 | 40.63 |

These randomly initialized models only verify pipeline connectivity. Saved validation outputs are empty and unsuitable for academic rewriting. Tiny loss differences do not show an advantage. The reference scan is slower in this run, and its timing cannot establish the efficiency of optimized Mamba. GPU peak memory includes measured training/validation/generation allocations for this process and is not total VRAM use. Training time excludes evaluation and setup.

Reproducible sources: runs/transformer_smoke/result.json and runs/hybrid_smoke/result.json; step logs and validation predictions sit beside them. No test predictions have been produced.

## Still pending

Pretrained model training, the official optimized Mamba backend, independent style-detector labels and training, Mamba-only generator, early stopping and checkpoint selection for the main study, multi-seed ablations, BLEU/ROUGE/BERTScore quality results, blind human review, and remote GitHub publication.
