# YOLOE-26 standard COCO box and mask benchmark status

**Status:** all four CUDA benchmark arms and standard COCO scoring completed on the same fixed 500-image COCO val2017 subset.

The initial YOLOE-26-L-seg attempt inferred all 500 images but failed to serialize a CUDA UUID. The preserved failure is documented in [`yoloe26_l_seg_run1_failure.md`](yoloe26_l_seg_run1_failure.md). A corrected strict-JSON run completed. The L and M detection-only heads were derived from their matching official segmentation checkpoints and evaluated on boxes only.

| Arm | Box AP@[.50:.95] | Mask AP@[.50:.95] | Inference seconds | Images/s |
|---|---:|---:|---:|---:|
| YOLOE-26-L-seg | 0.444025 | 0.367780 | 51.862 | 9.64 |
| YOLOE-26-L detect-only, derived | 0.444025 | — | 6.290 | 79.49 |
| YOLOE-26-M-seg | 0.435065 | 0.365515 | 39.274 | 12.73 |
| YOLOE-26-M detect-only, derived | 0.435064 | — | 5.435 | 92.00 |

The complete configurations, per-arm software and checkpoint provenance, resource use, error history, and raw artifact paths are in [`yoloe26_four_arm_coco_comparison.md`](yoloe26_four_arm_coco_comparison.md) and its linked per-arm reports. All four prediction artifacts contain the identical image IDs. The subset contains 3,245 ground-truth instances across 80 categories.
