# YOLOE-26-L-seg native-mask evaluation

## Result

Official COCO instance-segmentation metrics were not measured. The prior run reported custom polygon mask AP50 of **0.5881**, which is not COCO mask AP.

## Prior benchmark configuration

- Dataset: COCO val2017, 500 images sampled with seed 42; sorted image IDs are in [the raw benchmark JSON](data/modernized-yoloe-seg-component.json).
- Checkpoint: `weights/yoloe-26l-seg.pt`.
- Prompts: all 80 COCO category names.
- Inference: CUDA device 0, image size 640, confidence 0.001, NMS IoU 0.7, maximum 300 detections per image, `retina_masks=True`.
- Prior custom metrics: polygon mask AP50 **0.5880563**; box AP50 **0.5972366**.
- Prior runtime/resources: checkpoint load and prompt setup 1.732 s; inference 6.296 s total (79.42 images/s); median 10.67 ms/image; P95 14.96 ms/image; peak allocated CUDA memory 1.607 GiB.
- Prior scoring: polygon rasterization, greedy one-to-one matching, 101-point interpolated AP at IoU 0.5.

## Blocker and missing metrics

The raw result retains aggregate and per-category summaries, but no per-instance predicted masks or boxes. The configured checkpoint is absent from the checkout and was not found by exact-name search under `/mnt/HDD1` or `/home/nils`. Therefore COCOeval segmentation AP@[0.50:0.95], AP50, AP75, AR, and per-category COCO metrics were not measured. No rerun runtime or resource usage is available.

Follow the shared [native-mask evaluation protocol](native_mask_evaluation.md) after restoring the checkpoint and rerunning inference with original-resolution mask RLE predictions.
