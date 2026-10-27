# Academic Text Style Enhancement

Second Submission Milestone Report

27 October 2026

Borui Ning, Lichen Wang, Haoxiang Ning, Lingfeng Lu

GitHub: [https://github.com/Boruy/academic-hybrid-rewriter](https://github.com/Boruy/academic-hybrid-rewriter)

We have completed a reproducible BART + Mamba revision prototype, pretrained pilot runs and an initial test comparison. The pipeline works, but current results do not establish better academic tone or meaning preservation.

## 1 Dataset

We use the pinned IteraTeR-HUMAN sentence release [1]. Cleaning retains 676 arXiv revision pairs from 4,018 records across 95 documents. Document-group splitting with seed 42 gives 491 training, 98 validation and 87 test pairs. Cross-split document and exact-text overlap is zero, although semantic near duplicates may remain. Only 24 pairs carry a style label.

## 2 Model

The rewriter combines a pretrained BART encoder, a normalized Mamba adapter and a BART decoder. Gated fusion adds a learned adapter contribution to encoder states. Residual and replacement fusion are also implemented. Training minimizes target-token cross-entropy with padding ignored. The detector uses transparent heuristics, and the author reviews suggested revisions. The Mamba module currently uses a sequential PyTorch reference scan.

## 3 Current Progress

The runs below use facebook/bart-base, 300 updates, batch size 2, seed 42 and a 128-token input limit on an RTX 4060 Laptop GPU. Learning rates are 5e-5, 1e-4 and 2e-5 respectively.

| Model | Trainable params | Validation CE | Test CE | Test ROUGE-L F1 |
| --- | --- | --- | --- | --- |
| BART baseline | 139.42M | 1.2312 | 0.8213 | 0.7580 |
| Frozen hybrid | 4.92M | 1.1379 | 0.7802 | 0.7566 |
| Full hybrid | 144.34M | 1.0795 | 0.7395 | 0.7569 |

CE denotes token cross-entropy. Test scores use the same 87 pairs, deterministic decoding and at most 48 new tokens. ROUGE-L uses whitespace tokens. Learning rates and trainable capacities differ, so these runs do not isolate the effect of model architecture.

The frozen hybrid preserves detected number multisets in 85/87 cases, versus 87/87 for BART, including sentences without numbers. Both exactly copy the source in 24/87 cases. A separate validation-only learning-rate comparison favors 2e-5 among the tested full-hybrid rates. Eighteen automated tests pass. Nine complete checkpoints, predictions and reproducible code are available in the GitHub repository [2].

Subsequent audit (27 October 2026): full-hybrid test predictions have been added. At 128 new tokens, BART / frozen hybrid / full hybrid ROUGE-L is 0.8146 / 0.8138 / 0.8138, while returning the source unchanged scores 0.8503. Full hybrid retains the source word sequence in 77/87 cases versus 4/87 human references (case/punctuation ignored). The 48-token table above remains a historical-protocol comparison. This post-inspection decoding diagnostic does not establish effective rewriting; see [the quality audit](Prediction_Quality_Audit.md). The exported submission PDF and presentation are earlier snapshots.

## 4 Problems

Academic-tone supervision is limited, and overlap or number checks cannot establish semantic equivalence. The test set has now been inspected. Official Mamba kernel parity and speedups remain unverified. A trained style detector, Mamba-only baseline, fusion ablations and independent human review remain pending. BERTScore, bidirectional NLI and SacreBLEU have since been measured; independent academic-style and fluency ratings are still unavailable. See Multidimensional_Evaluation.md. The trainer saves final-step weights rather than the best validation checkpoint.

## 5 Next Plan

Audit revision pairs and collect independent tone and meaning labels. Add validation-based checkpoint selection and early stopping. Compare BART and hybrid models with matched optimization budgets, parameter controls and multiple seeds. Validate official Mamba before efficiency testing, then run fusion ablations and blinded human review. Use fresh held-out examples or a revised protocol for confirmatory evaluation.

## Sources

[1] IteraTeR-HUMAN, revision e22e0371; see data/raw/manifest.json. [2] Project code and run records at the GitHub repository above, commit 3b497fb.
