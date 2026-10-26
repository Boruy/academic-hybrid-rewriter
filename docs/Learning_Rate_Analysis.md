# Learning rate and loss analysis

Measured on 26 October 2026. These are single-seed pilot experiments.

## Loss variation

The full hybrid at 2e-5 has a first-50-step mean loss of 0.8407 and a last-50-step mean of 0.5858. The maximum individual loss is 3.3770. Each update samples only two training pairs, so successive losses measure different batches.

The frozen and full hybrid losses have same-step correlation 0.9567. Their shared spikes support batch difficulty as an explanation. They do not, by themselves, establish an excessively high learning rate or an architectural advantage.

## Controlled learning-rate comparison

The runs use facebook/bart-base, hybrid/reference/gated, seed 42, 491 training pairs, 98 validation pairs, batch size 2, a 128-token input limit and 300 updates. Constant-rate trials change only the learning rate. New trials evaluate validation loss every 50 steps. The reference run has only a final validation value, which the plot marks explicitly.

| Initial LR | Schedule | Last applied LR | First 50 mean | Last 50 mean | Step loss SD | Final validation CE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2e-05 | constant | 2e-05 | 0.8407 | 0.5858 | 0.5769 | 1.0795 |
| 1e-05 | constant | 1e-05 | 0.8809 | 0.6183 | 0.5762 | 1.0887 |
| 5e-06 | constant | 5e-06 | 0.9407 | 0.6649 | 0.5864 | 1.1078 |
| 2e-05 | plateau | 1e-05 | 0.8407 | 0.5865 | 0.5762 | 1.0799 |

The lowest final validation loss among constant-rate trials selects 2e-05. Considering the additional scheduler trials, the current recommendation for this limited comparison is 2e-05 with a constant schedule. Smoother training loss is not the selection criterion. The comparison does not use test scores and does not demonstrate better revision quality.

## Training controls

- training_progress.jsonl records raw loss, a 20-step moving average, the actual learning rate and the gradient norm before clipping.
- --eval-every 50 records validation token CE and fixed-batch training loss at step 0 and every 50 steps.
- --lr-scheduler plateau responds to validation loss, with relative improvement threshold 0.1%, a configurable patience, drop factor and minimum LR. Individual training spikes do not trigger it.
- A scheduler change after the final validation has not trained any weights. The table therefore reports the last learning rate actually applied.
- weights.pt contains the final-step weights. best_recorded_validation_cross_entropy records an observation, not a saved best checkpoint. Validation-based checkpoint saving and early stopping remain future work.
- Default learning rates are 1e-3 for tiny smoke runs, 1e-4 for a frozen hybrid and 2e-5 for full fine-tuning. An explicit --lr overrides these defaults.

The patience-1 diagnostic changes the rate only after step 300. The patience-0 trial applies the decrease after step 250 and uses it for the last 50 updates. These trials remain in runs/bart_hybrid_lr_adjusted and runs/bart_hybrid_lr_adjusted_patience0.

## Reproduction

~~~powershell
.\.venv\Scripts\python.exe -u -m academic_rewriter.run --model hybrid --backend reference --fusion gated --output runs/hybrid_repeat --steps 300 --train-limit 0 --val-limit 0 --batch-size 2 --lr 2e-05 --eval-every 50 --lr-scheduler constant --log-every 10 --local-files-only
.\.venv\Scripts\python.exe scripts/report_learning_rate.py
~~~

The offline flag requires an existing cached pretrained backbone. Raw comparisons are in runs/lr_tuning_offline/comparison.json. Figures and their numerical source are in output/learning-rate-analysis.
