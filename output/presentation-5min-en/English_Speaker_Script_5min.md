# Five-minute English presentation

Target delivery: approximately 5 minutes at 130 words per minute. Spoken script: 648 words.

## 1. Academic Text Style Enhancement

0:00–0:20 (20 seconds)

Good morning. Our project explores academic sentence revision using a Transformer and Mamba hybrid. The goal is to improve academic expression while preserving the author's meaning. Today, I will explain our data and model, show the latest training and test results, and discuss what we still need to validate.

## 2. 676 academic revision pairs

0:20–1:00 (40 seconds)

We use the human revision release of IteraTeR and retain 676 pairs from arXiv documents after cleaning. The training set contains 491 pairs, validation contains 98, and the test set contains 87. We group records by document and linked text before splitting, so the recorded cross-split document and exact-text overlap is zero. However, only 24 pairs carry a style label. Most examples concern clarity or fluency. This means our current dataset supports revision experiments, but it provides limited direct supervision for academic tone. Near-duplicate meanings may also remain.

## 3. A gated Mamba adapter inside BART

1:00–1:50 (50 seconds)

Our backbone is pretrained BART, which supplies the encoder and decoder. The encoder produces contextual representations, H. We normalize these representations and pass them through a Mamba refinement module to obtain M. A learned gate controls how much of M we add back to H. The BART decoder then generates the revised sentence from the fused representation, Z. We train with target-token cross-entropy and ignore padding labels. Residual and replacement fusion are also available for future ablations. The current Mamba module uses a sequential PyTorch reference implementation. We have not verified parity with official optimized kernels, and this hybrid still contains Transformer attention.

## 4. Pretrained training runs end to end

1:50–2:40 (50 seconds)

We have progressed beyond tiny random models to pretrained BART experiments. These three runs use all 491 training pairs, 98 validation pairs, and 300 updates with seed 42. Validation cross-entropy is 1.231 for BART, 1.138 for the frozen-backbone hybrid, and 1.079 for the fully trainable hybrid. The frozen hybrid trains about 4.92 million parameters, compared with 139.42 million for BART. Its measured peak allocated GPU memory is also lower in these runs. These are preliminary observations. The learning rates and trainable parameter counts differ, so this comparison does not isolate an architectural advantage. We need matched controls and multiple seeds.

## 5. Test overlap scores are nearly equal

2:40–3:35 (55 seconds)

The test comparison covers BART and the frozen-backbone hybrid on the same 87 examples. Both use deterministic generation with the same token limits. Their whitespace ROUGE-L scores are almost identical, at approximately 0.758 and 0.757. The hybrid has lower test cross-entropy, but that does not translate into a clear overlap improvement. Exact copying occurs in 24 of 87 cases for each model. The hybrid preserves the detected number multiset in 85 cases, compared with all 87 for BART. Some examples contain no numbers, so this measure is especially limited. These results do not establish better academic tone or semantic equivalence. The test set has now been inspected, which must guide how we design future confirmatory evaluation.

## 6. A high score can still miss the revision

3:35–4:20 (45 seconds)

This real test example explains why we need qualitative review. The source says that the tails are closely related to positional information. The human reference changes this to information captured by positional embeddings, which is more specific. Our frozen hybrid mainly removes the space before the final period and leaves the substantive wording unchanged. It still receives a whitespace ROUGE-L score of 0.741 because much of the sentence overlaps with the reference. This example is diagnostic, rather than representative of every output. It shows that overlap alone cannot tell us whether the model made the intended revision.

## 7. The next milestone is reliable rewriting

4:20–5:00 (40 seconds)

Our next work addresses these gaps. First, we will audit revision pairs and collect independent academic-tone and meaning judgments. Second, we will run matched BART and hybrid experiments with multiple seeds and select checkpoints using validation data. We will also validate the official Mamba backend before making efficiency claims. Since we have already examined this test set, any confirmatory study should use fresh held-out examples or an explicitly revised protocol. We now have a reproducible training and evaluation pipeline. Our remaining research question is whether the hybrid improves useful revision while preserving meaning. Thank you.
