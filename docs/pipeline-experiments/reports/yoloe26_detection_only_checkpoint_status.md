# YOLOE-26-L/M detection-only benchmark

**Status:** completed. Both detection-only heads were derived from official segmentation checkpoints and scored on the shared 500-image COCO val2017 subset. No official unsuffixed YOLOE-26-L/M detector checkpoint is released.

| Model | Box AP@[.50:.95] | AP50 | AP75 | AR100 | Images/s |
|---|---:|---:|---:|---:|---:|
| YOLOE-26-L detect-only, derived | 0.444025 | 0.604450 | 0.477425 | 0.603962 | 79.49 |
| YOLOE-26-M detect-only, derived | 0.435064 | 0.600023 | 0.464945 | 0.593889 | 92.00 |

The L head leads by about 0.90 percentage points in box AP. The M head was faster in these runs. Each derived head reproduced its matching segmentation model's box scores to below 0.000001. Both used 640-pixel input, confidence 0.001, IoU 0.7, maximum 300 detections, batch one, and CUDA:0.

See the [full four-arm comparison](../../../scripts/pipeline_model_reports/yoloe26_four_arm_coco_comparison.md), [L per-model report](../../../scripts/pipeline_model_reports/yoloe26_l_detect_only_coco_report.md), and [M per-model report](../../../scripts/pipeline_model_reports/yoloe26_m_detect_only_coco_report.md) for checkpoints, hashes, runtime, resources, failures, and artifacts. These are component detector metrics, not map-level association results.
