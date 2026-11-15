# Native YOLOE mask evaluation

**Status:** completed on the shared 500-image COCO val2017 subset. Standard COCO box and mask metrics are available for the L/M segmentation models; derived detection-only heads have standard box metrics.

The canonical result and full protocol are in [`yoloe26_four_arm_coco_comparison.md`](yoloe26_four_arm_coco_comparison.md). All four arms share the same image IDs. YOLOE-26-L-seg achieved box AP 0.444025 and mask AP 0.367780. YOLOE-26-M-seg achieved box AP 0.435065 and mask AP 0.365515. The L/M derived detection-only heads reproduced their matching scale's box AP to below 0.000001.

The models used 640-pixel inputs, confidence 0.001, IoU 0.7, and a 300-detection cap on CUDA:0. Full-resolution masks were retained as COCO RLE for official `pycocotools` scoring. Checkpoint hashes, runtimes, memory use, scorer evidence, and per-arm artifacts are documented in the canonical report and its linked per-model reports.

These sampled COCO results validate component box/mask behavior. They do not measure map-level association or full-scene pipeline acceptance. Earlier custom polygon AP values remain a separate provisional metric and are not compared numerically with standard COCO AP.
