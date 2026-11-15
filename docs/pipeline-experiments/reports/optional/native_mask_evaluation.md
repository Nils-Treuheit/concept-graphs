# Native YOLOE mask evaluation

## Status

Standard COCO instance-segmentation evaluation was not run. The retained benchmark contains summary metrics but no per-instance predicted masks, and the two YOLOE checkpoint files are absent from the searched local roots. Existing polygon AP values therefore remain provisional and must not be reported as COCO mask AP.

## Existing run

The benchmark used COCO val2017 at `/mnt/HDD1/Project_Data/public_datasets/coco`, with annotations at `annotations/instances_val2017.json` and images under `val2017/`. It selected 500 images using Python `random.Random(42).sample(...)`, then sorted the selected IDs. The IDs are retained in [modernized_yoloe_seg_component.json](../../data/modernized-yoloe-seg-component.json).

Both models used the 80 COCO category prompts, `imgsz=640`, confidence `0.001`, NMS IoU `0.7`, `max_det=300`, and `retina_masks=True` on CUDA device 0. The benchmark implementation is [yoloe_component_benchmark.py](../../data/yoloe_component_benchmark.py).

| Model | Custom polygon mask AP50 | Box AP50 | Inference time, 500 images | Median latency | P95 latency | Peak allocated CUDA memory |
|---|---:|---:|---:|---:|---:|---:|
| YOLOE-26-L-seg | 0.5881 | 0.5972 | 6.296 s | 10.67 ms/image | 14.96 ms/image | 1.607 GiB |
| YOLOE-26-M-seg | 0.5805 | 0.5931 | 4.972 s | 9.20 ms/image | 11.63 ms/image | 1.571 GiB |

The reported segmentation score used polygon rasterization, greedy one-to-one matching, and 101-point interpolated AP at IoU 0.5. It is not the official COCO mask metric. The raw JSON retains sample IDs, aggregate scores, per-category summaries, timing, and memory, but no prediction records. In the script, predictions are held in memory only for custom scoring and omitted from the output JSON.

## Blocker

The configured checkpoint paths are `weights/yoloe-26l-seg.pt` and `weights/yoloe-26m-seg.pt`. Neither file exists in the checkout. An exact-name search under `/mnt/HDD1` and `/home/nils` found no local copies. The COCO annotations, validation images, and `mobileclip2_b.ts` prompt encoder are present.

Because neither raw predictions nor checkpoints are available, official segmentation AP, AP50, AP75, AR, per-category COCO metrics, and a native-mask quality comparison are unavailable. No GPU rerun, runtime, or resource measurement was performed for this evaluation.

## Rerun protocol

1. Restore both configured segmentation checkpoints and verify their provenance and file hashes.
2. Run the existing benchmark on the same image root and annotation JSON with `--models yoloe-26l yoloe-26m --max-images 500 --seed 42 --image-size 640 --confidence 0.001`.
3. Preserve each predicted instance at original image resolution as binary mask RLE, plus `image_id`, `category_id`, and confidence score. Encode masks using `pycocotools.mask.encode` and convert byte counts to ASCII for JSON serialization. Retain original boxes if box metrics are also required.
4. Use the exact retained `sample_image_ids` and unmodified COCO ground-truth annotations to run `pycocotools.cocoeval.COCOeval` with `iouType='segm'`. Report AP@[0.50:0.95], AP50, AP75, AR, and per-category metrics separately from the existing custom polygon AP50. Evaluate box AP with `iouType='bbox'` if desired.
5. Record checkpoint hashes, software versions, exact configuration, elapsed load and inference time, throughput, latency distribution, peak CUDA allocation and total GPU use, errors, and artifact paths in a new raw results JSON and this report.

Before any GPU probe or model run, release this workflow's managed jobs with `python3 src/executor.py release-local` from `.agentic_workflow`, then verify `/status`, `/models`, and `/sessions` at the model-control app. Execute the run in host context and use only the configured GPU.
