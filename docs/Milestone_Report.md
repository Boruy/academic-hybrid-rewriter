# Academic Text Style Enhancement

Milestone Report

## 1 Dataset

We downloaded the author-released IteraTeR-HUMAN sentence dataset [1] at revision e22e0371dac444239b944f9293f5b491d62b73f0 and recorded file checksums. From 4,018 records, we retained 676 arXiv pairs labeled clarity, fluency or style after normalization, deduplication, length filtering and numeric-consistency checks. Seed-42 document-group splitting gives 491 training, 98 validation and 87 test pairs across 76, 9 and 10 documents. Cross-split document and exact-text overlap is zero. A real human edit changes “in a dialogue system” to “in dialogue systems”.

## 2 Models

The rewriter combines a BART Transformer encoder, a Mamba reference adapter, gated or residual fusion, and a Transformer decoder. We also implement replacement fusion as a skip-path ablation. Training uses target-token cross-entropy with padding ignored. Our sequential PyTorch selective-state implementation follows the Mamba-1 structure [2]; official optimized-kernel parity is unverified. The current detector uses transparent heuristics. A supervised detector and Mamba-only baseline remain planned.

## 3 Current Progress

Cleaning, document-level splitting, train-only tokenization, batching, model training, generation and checkpoint reload are implemented. Four integrity tests passed. On an RTX 4060 Laptop GPU, both randomly initialized tiny BART models completed 30 updates on 64 real training pairs with 16 validation pairs. Fixed-batch training cross-entropy decreased from 7.778 to 6.571 for Transformer and 7.778 to 6.570 for Hybrid. Validation cross-entropy was 6.841 and 6.840 respectively. These are pipeline smoke checks, not evidence of improved rewriting quality. Test data have not been used for model evaluation.

## 4 Problems

Only 24 retained pairs carry a style label, limiting direct supervision for academic tone. Numeric checks cannot certify meaning preservation. Human review, pretrained fine-tuning and official Mamba integration remain incomplete. Saved tiny-model validation generations are empty and unsuitable for practical writing. The reference scan does not establish an efficiency advantage.

## 5 Next Plan

During milestone week, audit 50 pairs and prepare independent detector labels. Suggested team targets are a pretrained BART baseline for Nov 1–7, Hybrid integration and fusion ablations for Nov 8–14, and blind review plus efficiency analysis for Nov 15–21. Use matched splits and decoding, validation-based checkpoint selection, and a frozen protocol before final test evaluation. Assess reference overlap, semantic preservation, academic tone and resource use. Local source, README and experiment logs are ready for version control; remote GitHub publication is pending.