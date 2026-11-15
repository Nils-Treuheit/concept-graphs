# YOLOE-26-L detection-only: COCO box benchmark

## Result

Completed real CUDA inference and standard COCO scoring on the fixed 500-image COCO val2017 subset. This arm uses a YOLOE-26-L detection-only head **derived from** the matching official segmentation checkpoint. It is not an official released detect-only weight. It reached **44.40 bbox AP**, **60.45 AP50**, **47.74 AP75**, and **60.40 AR100**.

| Metric | YOLOE-26-L detect-only |
|---|---:|
| Box AP@[.50:.95] | 0.4440 |
| Box AP50 | 0.6045 |
| Box AP75 | 0.4774 |
| Box AR100 | 0.6040 |

The run completed 500/500 images with 49,819 predictions and no failures. Its sample IDs exactly match the completed YOLOE-26-L-seg arm. Box AP is effectively identical between those L heads (difference below 0.000001). Detection-only inference reached 79.49 images/s versus 9.64 images/s for L-seg in these runs, which also serialized full-resolution masks. Do not read that throughput difference as a fully controlled model-kernel speedup; mask output and serialization differ by design.

## Configuration and resources

- **Checkpoint:** `models/yoloe-26l-seg.pt`, 79,417,413 bytes, SHA-256 `a612d2d505f24e14d87ec82d688b823b6cb600646664f16125ce6c84ce360da9`.
- **Derived head:** logical `yoloe-26l.yaml`; installed unified file `ultralytics/cfg/models/26/yoloe-26.yaml`; inferred scale `l`. The runner transferred 1,206/1,206 items and loaded the matching 80-class prompt profile. The generated profile SHA-256 is `26d6df0ea4949d9ada71cd09292eb38c8d4b85b3102adcce4ef06566cb244290`.
- **Dataset/split:** COCO val2017, same deterministic seed-42 500-image subset as L-seg; 3,245 GT instances across 80 categories.
- **Inference:** 640 px, confidence 0.001, IoU 0.7, max detections 300, batch 1, CUDA:0; one unscored warmup image.
- **Software:** Python 3.12.13, PyTorch 2.14.1+cu130, CUDA 13.0, Ultralytics 8.4.183, NVIDIA RTX 5090, driver 595.99.02.
- **Runtime:** 6.290 s timed runner window; 79.4944 images/s; median 11.44 ms/image; p95 19.978 ms/image. Checkpoint load 0.100 s, prompt setup 2.649 s, profile save 0.045 s, derived transfer 0.722 s, profile load 0.0015 s, warmup 0.492 s. COCO scoring took 3.077 s.
- **GPU memory:** process peak allocated 0.315 GiB and reserved 0.648 GiB. Whole-device baseline 2.640 GiB, peak 2.888 GiB, over 90 one-second samples.
- **Failures:** the first attempt stopped before model construction because preflight expected a nonexistent per-scale YAML. The accepted loader-mirroring correction fixed that issue. The corrected run completed with no failures.

## Comparison to prior evidence

The earlier provisional custom box AP50 values used a different metric implementation and were not retained per-instance predictions, so they are not numeric standard-COCO baselines. On this fixed standard-COCO subset, the derived L detection-only head and official L segmentation head have effectively equal box metrics. The remaining M arms are required before choosing a size or variant.

## Artifacts

- Predictions: `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1935/yoloe-26l-detect-only.json`
- COCO metrics and scorer evidence: `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1935/yoloe-26l-detect-only-score/metrics.json`
- Initial preserved preflight failure: `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1912/yoloe-26l-detect-only.json.failure.json`
- Harness and scorer: `conceptgraph/scripts/yoloe_component_benchmark_v2.py`, `conceptgraph/scripts/yoloe_coco_score.py`
- L-seg comparison report: `scripts/pipeline_model_reports/yoloe26_l_seg_coco_report.md`
