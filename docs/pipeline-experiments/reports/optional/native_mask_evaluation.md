# YOLOE-26 native box and mask evaluation

**Status:** completed on the shared 500-image COCO val2017 subset. The official COCO scorer evaluated boxes and segmentation masks for the L/M segmentation heads. It evaluated boxes for the matching derived detection-only heads.

| Arm | Box AP@[.50:.95] | Mask AP@[.50:.95] |
|---|---:|---:|
| YOLOE-26-L-seg | 0.444025 | 0.367780 |
| YOLOE-26-L detect-only, derived | 0.444025 | — |
| YOLOE-26-M-seg | 0.435065 | 0.365515 |
| YOLOE-26-M detect-only, derived | 0.435064 | — |

See the [full four-arm report](../../../../scripts/pipeline_model_reports/yoloe26_four_arm_coco_comparison.md) for protocol, configurations, runtime, resource use, failures, and artifact paths. These component metrics do not establish map-level object association or full-scene acceptance. Earlier custom polygon AP values are not standard COCO metrics.
