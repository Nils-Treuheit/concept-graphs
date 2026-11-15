# YOLOE-26-M detection-only: COCO box benchmark

## Result

Completed real CUDA inference and standard COCO scoring on the fixed 500-image COCO val2017 subset. This arm uses a YOLOE-26-M detection-only head derived from the matching official segmentation checkpoint; it is not an official released detect-only weight. It reached **43.51 bbox AP**, **60.00 AP50**, **46.49 AP75**, and **59.39 AR100**.

| Metric | YOLOE-26-M detect-only |
|---|---:|
| Box AP@[.50:.95] | 0.4351 |
| Box AP50 | 0.6000 |
| Box AP75 | 0.4649 |
| Box AR100 | 0.5939 |

The run completed 500/500 images, produced 50,496 boxes, and had no failures. Its exact image IDs match all three other arms. Its box AP is effectively identical to M-seg (difference below 0.000001).

## Configuration and resources

- **Checkpoint:** `models/yoloe-26m-seg.pt`, 70,443,185 bytes, SHA-256 `585f5ec9028fd358035da8d860c27c56be285a795cba2076fba536a4391c2c83`.
- **Derived head:** logical `yoloe-26m.yaml`; installed unified file `ultralytics/cfg/models/26/yoloe-26.yaml`; inferred scale `m`. The runner transferred 882/882 items and loaded the matching 80-category prompt profile. Profile SHA-256 `b9401e4111b21b12091595f499c37d9753cfe076d726e105c0c9022a111d6492`.
- **Dataset/split:** COCO val2017, same deterministic seed-42 fixed subset; 500 images, 3,245 GT instances and 80 categories.
- **Inference:** 640 px, confidence 0.001, IoU 0.7, max detections 300, batch 1, CUDA:0; one unscored warmup image.
- **Software:** Python 3.12.13, PyTorch 2.14.1+cu130, CUDA 13.0, Ultralytics 8.4.183, NVIDIA RTX 5090, driver 595.99.02.
- **Runtime:** 5.435 s measured runner window; 91.9952 images/s; median 9.848 ms/image; p95 21.326 ms/image. Checkpoint load 0.065 s, prompt setup 2.599 s, profile save 0.0055 s, derived transfer 0.226 s, profile load 0.0015 s, warmup 0.466 s. Standard COCO scoring took 3.036 s.
- **GPU memory:** process peak allocated 0.297 GiB and reserved 0.635 GiB. Whole-device baseline 2.640 GiB, peak 2.908 GiB, increase 0.268 GiB over 84 one-second samples.
- **Failures:** none.

## Comparison to completed arms

The derived M head's box AP matches M-seg to below 0.000001. The L detection-only arm reached 0.4440 AP on the same IDs, 0.0090 higher. M detection-only was faster in these observed runs (92.00 versus 79.49 images/s for L detect-only) and had slightly lower process allocation. Thus this subset favors L when box AP is the priority and M when observed throughput is the priority. A production choice should account for the intended accuracy/latency tradeoff; no threshold tuning was performed.

Earlier provisional custom AP50 values use a different metric and are not numeric standard-COCO baselines.

## Artifacts

- Predictions: `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1950/yoloe-26m-detect-only.json`
- COCO metrics and scorer evidence: `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1950/yoloe-26m-detect-only-score/metrics.json`
- Exact ground-truth subset and evaluator inputs: adjacent files in the score directory.
- Harness and scorer: `conceptgraph/scripts/yoloe_component_benchmark_v2.py`, `conceptgraph/scripts/yoloe_coco_score.py`
