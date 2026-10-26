# Progress evidence

Updated 26 October 2026. Values below come from saved results and prediction files.

## Data

Source: data/raw/manifest.json. Cleaning and split statistics: data/processed/stats.json.

676 arXiv revision pairs across 95 documents remain from 4,018 original records. The split contains 491 training, 98 validation and 87 test pairs. Cross-split document and exact-text intersections are zero. Only 24 pairs carry a style label.

## Pretrained training

| Run directory under runs | Trainable parameters | Validation CE | Measured training seconds | Peak allocated CUDA MiB |
| --- | ---: | ---: | ---: | ---: |
| bart_baseline | 139,420,416 | 1.231209 | 30.758 | 2690.99 |
| bart_adapter_visible | 4,915,968 | 1.137914 | 46.050 | 834.07 |
| bart_hybrid_unfrozen | 144,336,384 | 1.079475 | 229.345 | 2782.95 |

All runs use 300 updates, seed 42, 491 training and 98 validation pairs. Learning rates differ. Time and memory values are observations from these processes, not controlled efficiency benchmarks. Peak allocated CUDA memory is not total VRAM use.

## Test evaluation

Source files: runs/bart_baseline/evaluation/test and runs/bart_adapter_visible/evaluation/test.

| Metric | BART | Frozen hybrid |
| --- | ---: | ---: |
| Whitespace ROUGE-L F1 | 0.757985 | 0.756592 |
| Token CE | 0.821325 | 0.780219 |
| Detected number multiset preservation | 87/87 | 85/87 |
| Exact source copies | 24/87 | 24/87 |

The same 87 test examples use deterministic decoding, max_length=128 and max_new_tokens=48. Number preservation includes examples without numbers and does not establish semantic equivalence. Academic-tone improvement is unverified. The test set has now been inspected.

## Reproduction records

Every retained checkpoint has weights.pt, a tokenizer, backbone configuration, result.json and a training log. The learning-rate comparison and scheduler diagnostics remain in runs/lr_tuning_offline/comparison.json and their referenced run directories. Current automated test output is recorded in runs/tests.txt.

The latest English presentation and timed script are in output/presentation-5min-en. The standalone test viewer and its numerical source are in output/visualizations. Older translated documents, superseded exports, duplicate reruns and incomplete/debug runs have been removed from the current tree.
