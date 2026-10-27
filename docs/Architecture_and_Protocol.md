# Architecture and experiment protocol

Status: 27 October 2026.

## Implemented model

A heuristic detector flags conversational or vague patterns. The author can request rewriting and reviews the suggestion. The rewriter uses a pretrained BART encoder, a normalized Mamba adapter, feature fusion and the BART decoder.

Let H denote the encoder states and M = Mamba(LayerNorm(H)). Implemented fusion modes are:

- residual: Z = H + M
- gated: G = sigmoid(W[H; M] + b), Z = H + G ⊙ M
- replace: Z = M, for a skip-path ablation

The decoder cross-attends to Z. Training minimizes target-token cross-entropy and ignores padding labels. Generation uses the same encoder/fusion path. Right padding prevents later padding tokens from changing earlier causal adapter states. The unidirectional adapter receives encoder states that already contain bidirectional source context.

The reference module follows the Mamba-1 pattern with projection, a gate, causal depthwise convolution, input-dependent selective state updates and an output projection. It uses a sequential PyTorch scan. Official-kernel numerical parity, gradient equivalence and optimized efficiency are unverified. The hybrid retains Transformer attention and does not make the full model linear-time. The optional official backend fails explicitly if unavailable.

## Completed experiments

Tiny Transformer and hybrid runs verify forward/backward, generation and checkpoint reload. Their random initialization does not produce useful rewrites.

Pretrained runs initialize from facebook/bart-base. BART baseline, frozen hybrid and fully trainable hybrid use 491 training pairs, 98 validation pairs, batch size 2, length 128, 300 updates and seed 42. The learning rates are 5e-5, 1e-4 and 2e-5 respectively. Parameter counts, optimizer behavior and trainable capacity differ, so these runs cannot isolate an architectural effect.

Controlled constant-rate comparisons for the full hybrid use 2e-5, 1e-5 and 5e-6 with matched data, initialization, seed, batches and update budget. Final validation CE selects 2e-5 within this limited pilot. Additional plateau scheduler trials remain available. See Learning_Rate_Analysis.md and runs/README.md.

## Training controls

The runner records per-step loss, a 20-step moving average, the applied learning rate and pre-clipping gradient norm. Optional periodic validation drives a ReduceLROnPlateau scheduler. It records validation CE, fixed-batch training loss and the next learning rate.

The saved weights are from the final update. A best observed validation score does not imply a saved best checkpoint. Best-checkpoint saving, early stopping, schedulers beyond the implemented plateau mode and main-study optimization controls remain future work.

## Evaluation boundaries

All three main checkpoints have been evaluated on all 87 test pairs with deterministic greedy decoding and a 128-token input limit. The historical protocol allows 48 new tokens; an additional post-inspection length diagnostic allows 128. The same checkpoints and examples are used for both. No model selection claim is made from this diagnostic. Test predictions have also been inspected in the visualization and presentation. This set is no longer unseen. It is separate from training and validation, but a confirmatory study needs new held-out data or a transparently revised protocol.

Whitespace ROUGE-L is an overlap diagnostic. At 128 new tokens, returning the input unchanged scores 0.8503, above the full hybrid at 0.8138. The full hybrid retains the source word sequence in 77/87 cases (case/punctuation ignored), versus 4/87 human references. See Prediction_Quality_Audit.md. Number multiset preservation cannot detect changes to negation, hedging, entities or citations. Exact copying can earn a high overlap score while missing the intended revision. Independent academic-tone ratings and blinded meaning review remain pending.

## Next study

Audit source/reference pairs and collect independent tone and meaning judgments. Compare BART and hybrid models with matched splits, decoding, optimization budgets and seed list [13, 42, 2026]. Include a parameter-matched Transformer control and residual/gated/replace ablations. Use validation-only checkpoint selection. Validate official Mamba kernels before efficiency comparisons.

Measure median/p95 latency after warm-up with CUDA synchronization, trainable/total parameters and peak allocated memory at matched batches and lengths. For human review, use a documented rubric for tone, fluency, meaning and factual changes, with blinded model identities and more than one reviewer. The current heuristic detector is not a trained academic-style classifier. A supervised detector and Mamba-only generator remain future work.
