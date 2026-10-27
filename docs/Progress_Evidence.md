# Progress evidence

Updated 27 October 2026. Values below come from saved results and prediction files.

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

All three main checkpoints have predictions under evaluation/test (48-token historical protocol) and evaluation/test-128 (128-token output-length diagnostic). Checkpoints, input limit, greedy decoding and the 87 aligned test pairs are unchanged.

| Metric, max 128 new tokens | BART | Frozen hybrid | Full hybrid |
| --- | ---: | ---: | ---: |
| Whitespace ROUGE-L F1 | 0.814633 | 0.813772 | 0.813834 |
| Token CE | 0.821325 | 0.780219 | 0.739459 |
| Number multiset preservation | 87/87 | 85/87 | 87/87 |
| Exact source copies | 32/87 | 30/87 | 32/87 |
| Same word sequence, case/punctuation ignored | 83/87 | 75/87 | 77/87 |
| Generation cap reached | 0/87 | 0/87 | 0/87 |

Returning the input unchanged scores 0.850270 ROUGE-L. Only 4/87 human references keep the same word sequence. These observations expose a copying failure despite lower teacher-forced CE. See [the prediction-quality audit](Prediction_Quality_Audit.md).

The old 48-token ROUGE-L values remain 0.757985 / 0.756592 / 0.756902. Short-budget truncation was a visible evaluation problem; removing it does not make the models effective revisers. The additional length diagnostic followed test inspection and is not an independent confirmatory evaluation. Number preservation includes examples without numbers and does not establish semantic equivalence. Academic-tone improvement remains unverified.

## Reproduction records

Every retained checkpoint has weights.pt, a tokenizer, backbone configuration, result.json and a training log. The learning-rate comparison and scheduler diagnostics remain in runs/lr_tuning_offline/comparison.json and their referenced run directories. Current automated test output is recorded in runs/tests.txt.

The latest English presentation and timed script are in output/presentation-5min-en. The complete three-model test viewer, both protocols and numerical sources are in output/stage-two. The older output/visualizations/test-results.html entry redirects there; its old JSON is retained as historical two-model data. Eight automated tests now pass. Older translated documents, superseded exports, duplicate reruns and incomplete/debug runs have been removed from the current tree.
