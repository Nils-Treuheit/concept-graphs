# YOLOE-26-M segmentation: COCO box and mask benchmark

## Result

Completed real CUDA inference and standard COCO scoring on the fixed 500-image COCO val2017 subset. YOLOE-26-M-seg reached **43.51 bbox AP** and **36.55 mask AP**.

| Metric | Box | Mask |
|---|---:|---:|
| AP@[.50:.95] | 0.4351 | 0.3655 |
| AP50 | 0.6000 | 0.5851 |
| AP75 | 0.4649 | 0.3782 |
| AR100 | 0.5939 | 0.4851 |

This arm completed 500/500 images, produced 50,488 box/mask pairs, and had no inference or scoring failures. Its image IDs exactly match YOLOE-26-L-seg.

## Configuration and resources

- **Checkpoint:** `models/yoloe-26m-seg.pt`, 70,443,185 bytes, SHA-256 `585f5ec9028fd358035da8d860c27c56be285a795cba2076fba536a4391c2c83`.
- **Prompt encoder:** `mobileclip2_b.ts`, SHA-256 `35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f`.
- **Dataset/split:** COCO val2017; same deterministic seed-42 fixed subset as the other completed arms; 500 images, 3,245 GT instances and 80 categories.
- **Inference:** 640 px, confidence 0.001, IoU 0.7, max detections 300, batch 1, CUDA:0, `retina_masks=True`; one unscored warmup image.
- **Software:** Python 3.12.13, PyTorch 2.14.1+cu130, CUDA 13.0, Ultralytics 8.4.183, NVIDIA RTX 5090, driver 595.99.02.
- **Runtime:** 39.274 s measured runner window; 12.7312 images/s; median 54.573 ms/image; p95 219.871 ms/image. Model load 0.116 s, prompt setup 2.779 s, warmup 0.555 s. Standard COCO scoring took 26.238 s.
- **GPU memory:** process peak allocated 1.438 GiB and reserved 2.594 GiB. Whole-device baseline 2.625 GiB, peak 4.856 GiB, increase 2.231 GiB over 107 one-second samples.
- **Failures:** none. PyTorch emitted a `torch.jit.load` deprecation warning; the run completed and scored successfully.

## Comparison to completed L-seg

On this fixed subset, L-seg scored 0.4440 box AP and 0.3678 mask AP. M-seg scored 0.4351 box AP and 0.3655 mask AP. L leads by 0.0090 box AP and 0.0023 mask AP. M-seg was faster by this run's throughput measure (12.73 versus 9.64 images/s), although runtime comparisons remain sensitive to system load. The earlier custom-polygon AP values are not standard COCO metrics and are not numeric baselines.

No overall variant is selected yet; the M detection-only arm remains to run.

## Artifacts

- Predictions: `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1940/yoloe-26m-seg.json`
- COCO metrics and scorer evidence: `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1940/yoloe-26m-seg-score/metrics.json`
- Exact ground-truth subset and evaluator inputs: adjacent files in the score directory.
- Harness: `conceptgraph/scripts/yoloe_component_benchmark_v2.py`
- L-seg report: `scripts/pipeline_model_reports/yoloe26_l_seg_coco_report.md`
