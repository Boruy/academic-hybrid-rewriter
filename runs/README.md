# Recorded experiments

Each directory listed below contains a complete checkpoint, tokenizer, backbone configuration and result/training records. Model weights use Git LFS.

| Directory | Model / purpose | Steps | Train / validation pairs | Learning rate | Validation CE |
| --- | --- | ---: | ---: | ---: | ---: |
| runs/bart_adapter_visible | hybrid / frozen | 300 | 491 / 98 | 1e-4 | 1.137914 |
| runs/bart_baseline | transformer | 300 | 491 / 98 | 5e-5 | 1.231209 |
| runs/bart_hybrid_lr_adjusted | hybrid / full | 300 | 491 / 98 | 2e-5 | 1.079475 |
| runs/bart_hybrid_lr_adjusted_patience0 | hybrid / full | 300 | 491 / 98 | 2e-5 | 1.079941 |
| runs/bart_hybrid_unfrozen | hybrid / full | 300 | 491 / 98 | 2e-5 | 1.079475 |
| runs/hybrid_smoke | hybrid / full | 30 | 64 / 16 | 1e-3 | 6.839501 |
| runs/lr_tuning_offline/hybrid_lr_1em05 | hybrid / full | 300 | 491 / 98 | 1e-5 | 1.088660 |
| runs/lr_tuning_offline/hybrid_lr_5em06 | hybrid / full | 300 | 491 / 98 | 5e-6 | 1.107842 |
| runs/transformer_smoke | transformer | 30 | 64 / 16 | 1e-3 | 6.841281 |

The primary frozen hybrid for the presentation and test viewer is bart_adapter_visible. Its historical directory name refers to console progress logging, not a different architecture. The older repeated frozen run was superseded by this one. hybrid_smoke and transformer_smoke verify the pipeline with random tiny models.

The lr_tuning_offline comparison uses only validation scores. Its two additional constant-rate checkpoints and both plateau diagnostic checkpoints are retained. Result metadata preserves original paths and settings. The report generator resolves historical Windows run paths relative to the current checkout.

All three main checkpoints have test predictions in evaluation/test (48 new tokens) and evaluation/test-128 (128 new tokens). The latter is a post-inspection length diagnostic with raw generation-length and cap-hit metadata. The full-hybrid test evaluation was added after the initial report; weights are unchanged. The complete three-model viewer is output/stage-two/index.html. The test set has been inspected and should not be treated as unseen confirmatory data. Current unit-test output is in tests.txt.
