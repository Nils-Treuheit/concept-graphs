# YOLOE-26-L/M detection-only benchmark status

**Status:** completed for the derived L and M detection-only heads on the shared 500-image COCO val2017 subset.

No official unsuffixed YOLOE-26-L/M detection-only checkpoints are released. These two heads were constructed from the matching official segmentation checkpoints using the documented YOLOE architecture path. They are derived heads, not separate downloadable checkpoint variants.

| Model | Source checkpoint | Box AP@[.50:.95] | AP50 | AP75 | AR100 | Inference window | Images/s |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOE-26-L detect-only, derived | `models/yoloe-26l-seg.pt` | 0.444025 | 0.604450 | 0.477425 | 0.603962 | 6.290 s | 79.49 |
| YOLOE-26-M detect-only, derived | `models/yoloe-26m-seg.pt` | 0.435064 | 0.600023 | 0.464945 | 0.593889 | 5.435 s | 92.00 |

Both heads used 640-pixel inputs, confidence 0.001, IoU 0.7, maximum 300 detections, batch one, and CUDA:0. The shared COCO subset contains 500 images, 3,245 ground-truth instances, and 80 categories. Exact checkpoint hashes, derived prompt profile hashes, software versions, memory, failures, and prediction artifacts appear in the [L report](yoloe26_l_detect_only_coco_report.md), [M report](yoloe26_m_detect_only_coco_report.md), and [four-arm comparison](yoloe26_four_arm_coco_comparison.md).

The L head leads by about 0.90 percentage points in box AP; the M head was faster in these runs. Each derived head reproduced its matching scale's segmentation-head box AP to below 0.000001. This component comparison does not establish map-level association or full-scene acceptance.

## Checkpoint provenance

The official L/M segmentation weights are also present under `weights/`. Repository copies under `models/` are local working assets. Their SHA-256 values are `a612d2d505f24e14d87ec82d688b823b6cb600646664f16125ce6c84ce360da9` for L and `585f5ec9028fd358035da8d860c27c56be285a795cba2076fba536a4391c2c83` for M.
