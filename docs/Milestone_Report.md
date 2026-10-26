# Academic Text Style Enhancement

Milestone report, 26 October 2026.

## Dataset

The project uses the author-released IteraTeR-HUMAN sentence data at revision e22e0371dac444239b944f9293f5b491d62b73f0. Cleaning retains 676 arXiv revision pairs from 4,018 records. Document-group splitting with seed 42 gives 491 training, 98 validation and 87 test pairs across 76, 9 and 10 documents. Cross-split document and exact-text overlap is zero. Only 24 retained pairs carry a style label, limiting direct academic-tone supervision. See Dataset_Card.md.

## Model

A pretrained BART encoder produces contextual source states. A normalized Mamba reference adapter refines them, and gated fusion adds a learned adapter contribution before the BART decoder generates the revision. Residual and replacement fusion are available for ablations. Training uses target-token cross-entropy with ignored padding labels. The detector remains a transparent heuristic, and the author reviews generated suggestions. The sequential PyTorch Mamba reference has not been validated against optimized official kernels.

## Current results

Pretrained runs now complete 300 updates on all 491 training pairs and evaluate 98 validation pairs. Validation token CE is 1.2312 for BART, 1.1379 for the frozen hybrid and 1.0795 for the fully trainable hybrid. The frozen hybrid trains 4.92 million parameters, compared with 139.42 million for BART. Learning rates and trainable capacity differ, so these single-seed measurements do not isolate architectural benefits.

The same 87 test examples yield whitespace ROUGE-L F1 of 0.7580 for BART and 0.7566 for the frozen hybrid. Their test CE values are 0.8213 and 0.7802. Both copy the source exactly in 24 cases. The number multiset check passes in 87 baseline cases and 85 hybrid cases, including cases without numbers. These diagnostics do not establish better tone or semantic equivalence. Predictions and numerical results are retained beside the checkpoints.

## Problems and next work

Reference overlap can remain high when the output misses the substantive human edit. Independent tone and meaning labels are still needed. The test predictions have now been inspected, so confirmatory work needs new held-out examples or a revised protocol. Matched multi-seed experiments, parameter controls, validation-based best-checkpoint saving and official Mamba validation remain priorities. Current weights represent the final training step.

The repository includes reproducible source, tests, data, complete retained checkpoints and English reporting materials. Sources and measured values are documented in Progress_Evidence.md, Architecture_and_Protocol.md and Learning_Rate_Analysis.md. Third-party attribution remains in ../THIRD_PARTY_NOTICES.md and the original data license.
