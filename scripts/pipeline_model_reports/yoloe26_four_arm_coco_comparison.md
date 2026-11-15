# YOLOE-26-L/M four-arm COCO comparison

## Summary

All four arms completed CUDA inference and standard COCO evaluation on the same fixed 500-image COCO val2017 subset. The L scale led M by about **0.90 percentage points bbox AP**. Segmentation and detection-only heads at each scale had effectively identical box metrics. The derived L detection-only head is the accuracy-oriented choice for the FG-CLIP2 padded proposal path; the derived M head is a faster, lower-accuracy option in these observed runs.

These are component results on a sampled COCO validation subset. They do not measure map-level association or establish a full-scene pipeline acceptance result.

## Shared protocol

- **Dataset/split:** COCO val2017. The subset was sampled from sorted locally available image IDs with `random.Random(42).sample`, then sorted. It contains 500 images, 3,245 ground-truth instances, and 80 COCO categories. The four prediction artifacts contain identical image IDs.
- **Inference:** 640-pixel input, confidence 0.001, IoU 0.7, maximum 300 detections, batch size 1, CUDA:0, and one unscored warmup image.
- **Scoring:** standard COCO bbox and segmentation AP/AR through pycocotools 2.0.11. Detection-only heads are evaluated on boxes only. Segmentation RLE round-trip and binary-mask raster hash checks passed.
- **Software/hardware:** Python 3.12.13, PyTorch 2.14.1+cu130, CUDA 13.0, Ultralytics 8.4.183, NVIDIA RTX 5090, driver 595.99.02.
- **Head provenance:** L/M segmentation checkpoints are official YOLOE-26 segmentation weights. The detection-only heads are derived from the matching segmentation weights by loading the scale-specific model YAML and transferring compatible parameters. They are not separate official detect-only checkpoints.
- **Prompt encoder:** `mobileclip2_b.ts`, SHA-256 `35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f`.

## Accuracy

Values are standard COCO metrics. `—` means the detection-only arm did not produce masks.

| Arm | Box AP@[.50:.95] | Box AP50 | Box AP75 | Box AR100 | Mask AP@[.50:.95] | Mask AP50 | Mask AP75 | Mask AR100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOE-26-L-seg | **0.444025** | **0.604451** | **0.477425** | **0.603962** | **0.367780** | **0.586911** | **0.389059** | **0.494192** |
| YOLOE-26-L detect-only, derived | 0.444025 | 0.604450 | 0.477425 | 0.603962 | — | — | — | — |
| YOLOE-26-M-seg | 0.435065 | 0.600023 | 0.464947 | 0.593889 | 0.365515 | 0.585065 | 0.378199 | 0.485125 |
| YOLOE-26-M detect-only, derived | 0.435064 | 0.600023 | 0.464945 | 0.593889 | — | — | — | — |

The box AP differences between segmentation and detection-only heads are below 0.000001 at each scale. L exceeds M by approximately 0.00896 AP. Among mask-producing arms, L exceeds M by approximately 0.00226 mask AP.

## Runtime and device memory

| Arm | Inference window | Images/s | Median / p95 image time | Process peak allocated / reserved | Device baseline / peak |
|---|---:|---:|---:|---:|---:|
| YOLOE-26-L-seg | 51.862 s | 9.64 | 80.279 / 244.770 ms | 1.458 / 2.671 GiB | 2.646 / 5.352 GiB |
| YOLOE-26-L detect-only | 6.290 s | 79.49 | 11.440 / 19.978 ms | 0.315 / 0.648 GiB | 2.640 / 2.888 GiB |
| YOLOE-26-M-seg | 39.274 s | 12.73 | 54.573 / 219.871 ms | 1.438 / 2.594 GiB | 2.625 / 4.856 GiB |
| YOLOE-26-M detect-only | 5.435 s | 92.00 | 9.848 / 21.326 ms | 0.297 / 0.635 GiB | 2.640 / 2.908 GiB |

The timed windows are descriptive, not a controlled kernel-speed comparison. Segmentation runs generate and serialize full-resolution masks while detection-only runs serialize boxes. The observed throughput gap therefore includes different output work. Runtime also varies with system load. The scorer took 26.126 s for L-seg, 3.077 s for L detect-only, 26.238 s for M-seg, and 3.036 s for M detect-only. Whole-device peaks were sampled approximately once per second and include background allocation.

## Decision for proposal boxes

For the padded proposal path used by FG-CLIP2, use YOLOE-26-L detect-only when box AP is the priority: it matches L-seg's box scores to below 0.000001 and avoids mask production. The M detect-only arm is the throughput-oriented option, with about 92 images/s in this run versus 79.5 for L detect-only, and about 0.90 percentage points lower bbox AP. This is a component tradeoff only; it does not reopen the fixed segmentation detector choice elsewhere in the pipeline. No thresholds were tuned.

## Failures and limitations

- The first L-seg pass completed 500 CUDA images but failed while writing JSON because a CUDA device UUID was not JSON serializable. Its sidecar was preserved. The corrected strict-JSON run completed and scored.
- The first L detection-only attempt stopped before model construction because its preflight expected a per-scale YAML file absent from the installed package. The runner was corrected to resolve the installed unified scale YAML; the rerun completed. The failure artifact remains alongside the run evidence.
- Corrected L/M segmentation and detection-only runs completed 500/500 with no inference or scoring failures. PyTorch emitted a `torch.jit.load` deprecation warning on segmentation runs; it did not affect results.
- Earlier custom polygon mask metrics are not directly comparable to standard COCO AP. They are not used as a baseline here.
- This 500-image validation subset does not establish performance on the complete COCO split, other datasets, or ConceptGraphs map-level association. There is no instance-tracking ground truth in this evaluation.

## Per-arm reports and artifacts

- [YOLOE-26-L segmentation report](yoloe26_l_seg_coco_report.md): predictions and COCO evidence under `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1912/`.
- [YOLOE-26-L detection-only report](yoloe26_l_detect_only_coco_report.md): predictions, scoring evidence, and preserved preflight failure under `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1935/` and the initial `rerun_20261115_1912/` directory.
- [YOLOE-26-M segmentation report](yoloe26_m_seg_coco_report.md): predictions and COCO evidence under `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1940/`.
- [YOLOE-26-M detection-only report](yoloe26_m_detect_only_coco_report.md): predictions and COCO evidence under `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1950/`.
- Shared harness and scorer: `conceptgraph/scripts/yoloe_component_benchmark_v2.py` and `conceptgraph/scripts/yoloe_coco_score.py`.
