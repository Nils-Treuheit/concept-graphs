"""Compare YOLOE-26 segmentation checkpoints on a deterministic COCO-val subset.

The benchmark uses real CUDA inference and COCO instance-segmentation
annotations. It reports class-macro mask AP@0.5, box AP@0.5, latency, and peak
CUDA allocation.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import cv2

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from ultralytics import YOLOE, __version__ as ultralytics_version

from conceptgraph.utils.gpu_memory_budget import NvmlMemoryBudget


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COCO_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/coco")
CHECKPOINTS = {
    "yoloe-26l": "yoloe-26l-seg.pt",
    "yoloe-26m": "yoloe-26m-seg.pt",
}
MIN_CHECKPOINT_BYTES = 50 * 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_COCO_ROOT / "val2017")
    parser.add_argument(
        "--annotations", type=Path,
        default=DEFAULT_COCO_ROOT / "annotations" / "instances_val2017.json",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "weights")
    parser.add_argument("--models", nargs="+", choices=tuple(CHECKPOINTS),
                        default=["yoloe-26l", "yoloe-26m"])
    parser.add_argument("--max-images", type=int, default=500,
                        help="Deterministically sampled images; use -1 for all COCO val images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / ".agentic_workflow/workspace/modernized_yoloe_component.json")
    return parser.parse_args()


def _prompts(names: list[str]) -> list[str]:
    prompts = [re.sub(r"\s+", "-", name.strip()) for name in names]
    if any(not name for name in prompts) or len(set(prompts)) != len(prompts):
        raise ValueError("COCO class names must produce unique non-empty YOLOE prompts")
    return prompts


def _load_subset(args: argparse.Namespace):
    annotations_path = args.annotations.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    categories = sorted(payload["categories"], key=lambda row: int(row["id"]))
    image_rows = {int(row["id"]): row for row in payload["images"]}
    valid_image_ids = sorted(
        image_id for image_id, row in image_rows.items()
        if (image_root / row["file_name"]).is_file()
    )
    if args.max_images == 0 or args.max_images < -1:
        raise ValueError("--max-images must be positive or -1 for every image")
    count = len(valid_image_ids) if args.max_images == -1 else min(args.max_images, len(valid_image_ids))
    rng = random.Random(args.seed)
    image_ids = sorted(rng.sample(valid_image_ids, count))
    selected = set(image_ids)
    ground_truth: dict[int, dict[int, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for annotation in payload["annotations"]:
        if annotation.get("iscrowd", 0) or int(annotation["image_id"]) not in selected:
            continue
        x, y, width, height = (float(value) for value in annotation["bbox"])
        if width <= 0 or height <= 0:
            continue
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list) or not segmentation:
            raise ValueError(
                "Every selected COCO instance needs polygon masks; this dependency-free "
                "segmentation scorer does not accept RLE or empty masks."
            )
        ground_truth[int(annotation["image_id"])][int(annotation["category_id"])].append({
            "box": [x, y, x + width, y + height],
            "segmentation": segmentation,
        })
    selected_categories = {category_id for row in ground_truth.values() for category_id in row}
    if not selected_categories:
        raise ValueError("The selected COCO subset has no non-crowd ground-truth boxes")
    for image_id in image_ids:
        image_rows[image_id]["path"] = image_root / image_rows[image_id]["file_name"]
    return categories, image_ids, image_rows, ground_truth, annotations_path, image_root


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(right_bottom - left_top, 0).prod(axis=1)
    box_area = np.maximum(box[2:] - box[:2], 0).prod()
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(box_area + areas - intersection, 1e-12)


def _polygon_mask(polygons, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    contours = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32)
        if points.size >= 6 and points.size % 2 == 0:
            contours.append(np.rint(points.reshape(-1, 2)).astype(np.int32))
    if contours:
        cv2.fillPoly(mask, contours, 1)
    return mask.astype(bool)


def _average_precision(detections, ground_truth, image_ids, category_ids, *, use_masks, image_rows):
    per_category = {}
    total_tp = total_fp = total_gt = 0
    aps = []
    for category_id in category_ids:
        gt_by_image = {image_id: ground_truth.get(image_id, {}).get(category_id, []) for image_id in image_ids}
        gt_masks = {}
        if use_masks:
            for image_id, items in gt_by_image.items():
                height, width = int(image_rows[image_id]["height"]), int(image_rows[image_id]["width"])
                gt_masks[image_id] = [_polygon_mask(item["segmentation"], width, height) for item in items]
        positives = sum(len(boxes) for boxes in gt_by_image.values())
        if not positives:
            continue
        ranked = sorted(
            (row for row in detections if row["category_id"] == category_id),
            key=lambda row: row["score"], reverse=True,
        )
        matched = {image_id: np.zeros(len(items), dtype=bool) for image_id, items in gt_by_image.items()}
        tp = np.zeros(len(ranked), dtype=np.float64)
        fp = np.zeros(len(ranked), dtype=np.float64)
        for index, detection in enumerate(ranked):
            image_id = detection["image_id"]
            items = gt_by_image[image_id]
            if use_masks:
                height, width = int(image_rows[image_id]["height"]), int(image_rows[image_id]["width"])
                predicted_mask = _polygon_mask(detection["segmentation"], width, height)
                overlaps = np.asarray([
                    np.logical_and(predicted_mask, target_mask).sum()
                    / max(np.logical_or(predicted_mask, target_mask).sum(), 1)
                    for target_mask in gt_masks[image_id]
                ], dtype=np.float32)
            else:
                boxes = np.asarray([item["box"] for item in items], dtype=np.float32).reshape(-1, 4)
                overlaps = _iou(np.asarray(detection["box"], dtype=np.float32), boxes)
            if overlaps.size:
                order = np.argsort(overlaps)[::-1]
                best = next((int(candidate) for candidate in order if overlaps[candidate] >= 0.5 and not matched[image_id][candidate]), None)
            else:
                best = None
            if best is None:
                fp[index] = 1
            else:
                tp[index] = 1
                matched[image_id][best] = True
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        recall = cumulative_tp / positives
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
        interpolated = [float(precision[recall >= threshold].max()) if np.any(recall >= threshold) else 0.0
                        for threshold in np.linspace(0, 1, 101)]
        ap = float(np.mean(interpolated))
        aps.append(ap)
        total_tp += int(tp.sum())
        total_fp += int(fp.sum())
        total_gt += positives
        per_category[str(category_id)] = {
            "ap50": ap,
            "ground_truth_count": positives,
            "detections": len(ranked),
        }
    return {
        "map50_macro_category": float(np.mean(aps)) if aps else 0.0,
        "precision50_micro": total_tp / max(total_tp + total_fp, 1),
        "recall50_micro": total_tp / max(total_gt, 1),
        "ground_truth_count": total_gt,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "categories_evaluated": len(aps),
        "per_category": per_category,
    }


def _run_model(name, checkpoint, categories, image_ids, image_rows, ground_truth, args, device, budget):
    if not checkpoint.is_file() or checkpoint.stat().st_size < MIN_CHECKPOINT_BYTES:
        raise FileNotFoundError(f"Missing or incomplete YOLOE checkpoint: {checkpoint}")
    prompt_names = _prompts([str(row["name"]) for row in categories])
    category_by_model_index = {index: int(row["id"]) for index, row in enumerate(categories)}
    category_lookup = {int(row["id"]): row["name"] for row in categories}
    if importlib.util.find_spec("clip") is None:
        raise RuntimeError("YOLOE prompting requires the declared `clip` dependency")
    prompt_encoder = REPO_ROOT / "mobileclip2_b.ts"
    if not prompt_encoder.is_file() or prompt_encoder.stat().st_size < 100 * 1024**2:
        raise FileNotFoundError(f"Missing local YOLOE prompt encoder: {prompt_encoder}")
    print(f"[yoloe-component] loading {name}: {checkpoint}", flush=True)
    load_started = time.perf_counter()
    model = YOLOE(str(checkpoint))
    model.model.to(device).eval()
    predictions = []
    latencies = []
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.set_classes(prompt_names)
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - load_started
        budget.check(f"{name} prompt setup")
        for image_index, image_id in enumerate(image_ids):
            image_path = image_rows[image_id]["path"]
            started = time.perf_counter()
            with torch.inference_mode():
                result = model.predict(
                    source=str(image_path), device=str(device), imgsz=args.image_size,
                    conf=args.confidence, iou=0.7, max_det=300,
                    retina_masks=True, verbose=False,
                )[0]
            torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - started)
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.detach().float().cpu().numpy()
                scores = result.boxes.conf.detach().float().cpu().numpy()
                class_ids = result.boxes.cls.detach().long().cpu().numpy()
                if result.masks is None or len(result.masks.xy) != len(boxes):
                    raise RuntimeError(f"{name} returned boxes without aligned instance masks")
                for box, score, class_id, polygon in zip(boxes, scores, class_ids, result.masks.xy):
                    category_id = category_by_model_index.get(int(class_id))
                    if category_id is None:
                        continue
                    polygon = np.asarray(polygon, dtype=np.float32)
                    segmentation = [polygon.reshape(-1).tolist()] if len(polygon) >= 3 else []
                    predictions.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "box": box.tolist(),
                        "segmentation": segmentation,
                        "score": float(score),
                    })
            del result
            if (image_index + 1) % 100 == 0:
                budget.check(f"{name} image {image_index + 1}/{len(image_ids)}")
                print(f"[yoloe-component] {name}: {image_index + 1}/{len(image_ids)} images", flush=True)
        metrics = {
            "mask": _average_precision(
                predictions, ground_truth, image_ids, category_by_model_index.values(),
                use_masks=True, image_rows=image_rows,
            ),
            "box": _average_precision(
                predictions, ground_truth, image_ids, category_by_model_index.values(),
                use_masks=False, image_rows=image_rows,
            ),
        }
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
        total_seconds = float(sum(latencies))
        result = {
            "name": name,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "prompt_names": prompt_names,
            "image_count": len(image_ids),
            "ground_truth_classes": sorted(
                category_lookup[category_id]
                for category_id in {
                    category_id for rows in ground_truth.values() for category_id in rows
                }
            ),
            "load_seconds": load_seconds,
            "inference_seconds": total_seconds,
            "images_per_second": len(image_ids) / max(total_seconds, 1e-12),
            "latency_ms_median": float(np.median(latencies) * 1000),
            "latency_ms_p95": float(np.percentile(latencies, 95) * 1000),
            "peak_cuda_allocated_gib": peak_memory,
            "scoring": "COCO polygon mask and box IoU; greedy one-to-one matching; 101-point interpolated AP@0.5",
            **metrics,
        }
        return result
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        budget.check(f"{name} release")


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    if args.image_size <= 0 or not 0 <= args.confidence <= 1:
        raise ValueError("image size must be positive and confidence must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("YOLOE component benchmarking requires CUDA; CPU fallback is disabled")
    device = torch.device("cuda:0")
    budget = NvmlMemoryBudget(device)
    categories, image_ids, image_rows, ground_truth, annotations_path, image_root = _load_subset(args)
    output = []
    for name in args.models:
        checkpoint = args.checkpoint_dir.expanduser().resolve() / CHECKPOINTS[name]
        output.append(_run_model(
            name, checkpoint, categories, image_ids, image_rows, ground_truth,
            args, device, budget,
        ))
        metrics = output[-1]
        print(f"[yoloe-component] {name}: mask mAP50={metrics['mask']['map50_macro_category']:.4f}, "
              f"box mAP50={metrics['box']['map50_macro_category']:.4f}, "
              f"images/s={metrics['images_per_second']:.2f}", flush=True)
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "dataset": "COCO val2017",
        "annotations": str(annotations_path),
        "image_root": str(image_root),
        "sample_seed": args.seed,
        "sample_image_ids": image_ids,
        "ultralytics_version": ultralytics_version,
        "models": output,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[yoloe-component] wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
