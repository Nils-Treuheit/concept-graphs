# YOLOE-26-L segmentation: COCO box and mask benchmark

## Result

Completed real CUDA inference and standard COCO scoring on a fixed 500-image COCO val2017 subset. YOLOE-26-L-seg reached **44.40 bbox AP** and **36.78 mask AP**. It is the first completed arm of the four-way L/M segmentation versus derived detection-only comparison.

| Metric | Box | Mask |
|---|---:|---:|
| AP@[.50:.95] | 0.4440 | 0.3678 |
| AP50 | 0.6045 | 0.5869 |
| AP75 | 0.4774 | 0.3891 |
| AR100 | 0.6040 | 0.4942 |

The complete subset contains 500 images and 3,245 ground-truth instances across 80 COCO categories. The run produced 49,784 predicted boxes and masks. Every segmentation RLE round-trip and binary-raster hash check passed. No inference or scoring failures occurred.

## Configuration and resources

- **Checkpoint:** `models/yoloe-26l-seg.pt`, 79,417,413 bytes, SHA-256 `a612d2d505f24e14d87ec82d688b823b6cb600646664f16125ce6c84ce360da9`.
- **Prompt encoder:** `mobileclip2_b.ts`, SHA-256 `35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f`.
- **Dataset/split:** COCO val2017; `random.Random(42).sample` from sorted locally available image IDs, then sorted; 500 images. Exact IDs are in the prediction and scoring artifacts.
- **Inference:** 640 px, confidence 0.001, IoU 0.7, max detections 300, batch 1, CUDA:0, `retina_masks=True`; one unscored warmup image.
- **Software:** Python 3.12.13, PyTorch 2.14.1+cu130, CUDA 13.0, Ultralytics 8.4.183, NVIDIA RTX 5090, driver 595.99.02.
- **Runtime:** 51.862 s measured runner window; 9.6409 images/s; median 80.279 ms/image; p95 244.77 ms/image. Model load 0.161 s, prompt setup 5.015 s, warmup 3.778 s. Standard COCO scoring took 26.126 s.
- **GPU memory:** process peak allocated 1.458 GiB and reserved 2.671 GiB. Whole-device baseline 2.646 GiB, peak 5.352 GiB, increase 2.706 GiB over 119 one-second samples.
- **Failures:** none. A PyTorch `torch.jit.load` deprecation warning was emitted; it did not affect completion or scoring.

## Comparison to prior evidence

The earlier provisional custom-polygon benchmark reported L box AP50 0.5972 and mask AP50 0.5881. Those values use a different, non-COCO metric and do not retain per-instance predictions, so they are contextual only and cannot be treated as a numeric baseline. The standard COCO scores above replace the earlier provisional values for this 500-image protocol. No L-versus-M or segmentation-versus-detection-only winner is selected until all four arms share the same image IDs and have complete standard scores.

## Artifacts

- Predictions: `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1912/yoloe-26l-seg.json`
- COCO metrics and scorer evidence: `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1912/yoloe-26l-seg-score/metrics.json`
- Exact ground-truth subset and evaluator inputs: adjacent files in the score directory.
- Harness: `conceptgraph/scripts/yoloe_component_benchmark_v2.py`
- Earlier provisional result: `scripts/pipeline_model_reports/native_mask_yoloe26l.md`
