#!/usr/bin/env python3
"""Freeze paired YOLOE-mask/SAM3.1 COCO val inputs without loading a model.

The selected prompt is a YOLOE-26-L-seg predicted box matched to one COCO
instance of the same category at bbox IoU >= 0.5. The base mask remains the
original YOLOE prediction. One object is sampled per image, without looking at
mask IoU, using a separate deterministic seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PREDICTIONS = ROOT / "scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1912/yoloe-26l-seg.json"
DEFAULT_GT = ROOT / "scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1912/yoloe-26l-seg-score/gt_subset_instances.json"
DEFAULT_IMAGE_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/coco/val2017")
SAM_COMMIT = "eef6839bcfc3458dd52ea80535dc6bd62aea86b1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def box_iou_xywh(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def decode_rle(segmentation: Any, height: int, width: int):
    import numpy as np
    from pycocotools import mask as mask_api

    if isinstance(segmentation, list):
        rles = mask_api.frPyObjects(segmentation, height, width)
        rle = mask_api.merge(rles) if isinstance(rles, list) else rles
    else:
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_api.frPyObjects(rle, height, width)
    decoded = mask_api.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    value = np.asarray(decoded, dtype=bool)
    if value.shape != (height, width):
        raise ValueError(f"decoded mask shape {value.shape}, expected {(height, width)}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "models/experiments/sam3/data/coco-yoloe-l-seg-matched100-v1")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--min-box-iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = args.predictions.resolve(strict=True)
    gt_path = args.ground_truth.resolve(strict=True)
    image_root = args.image_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen data: {output}")
    if args.count < 1 or not 0.0 <= args.min_box_iou <= 1.0:
        raise ValueError("count must be positive and min-box-iou must be in [0,1]")
    pred_doc = json.loads(predictions_path.read_text(encoding="utf-8"))
    gt_doc = json.loads(gt_path.read_text(encoding="utf-8"))
    if pred_doc.get("status") != "complete" or pred_doc.get("arm") != "yoloe-26l-seg":
        raise ValueError("predictions must be a complete YOLOE-26-L-seg artifact")
    if not isinstance(pred_doc.get("segm_results"), list) or not pred_doc["segm_results"]:
        raise ValueError("prediction artifact has no segmentation results")
    images = {int(row["id"]): row for row in gt_doc.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in gt_doc.get("annotations", []):
        if not int(ann.get("iscrowd", 0)):
            annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    predictions_by_image: dict[int, list[dict[str, Any]]] = {}
    if len(pred_doc["bbox_results"]) != len(pred_doc["segm_results"]):
        raise ValueError("box and mask prediction rows are not aligned")
    for index, (box_row, mask_row) in enumerate(zip(pred_doc["bbox_results"], pred_doc["segm_results"])):
        if any(box_row.get(key) != mask_row.get(key) for key in ("image_id", "category_id", "score")):
            raise ValueError(f"box/mask prediction row mismatch at index {index}")
        row = {**mask_row, "bbox": box_row["bbox"]}
        row["_prediction_index"] = index
        predictions_by_image.setdefault(int(row["image_id"]), []).append(row)

    candidates: dict[int, list[tuple[float, dict[str, Any], dict[str, Any]]]] = {}
    for image_id in sorted(set(images) & set(predictions_by_image)):
        pairs = []
        preds = predictions_by_image[image_id]
        gts = annotations_by_image.get(image_id, [])
        for pred in preds:
            for gt in gts:
                if int(pred["category_id"]) != int(gt["category_id"]):
                    continue
                overlap = box_iou_xywh(pred["bbox"], gt["bbox"])
                if overlap >= args.min_box_iou:
                    pairs.append((overlap, pred, gt))
        used_predictions: set[int] = set()
        used_annotations: set[int] = set()
        matched = []
        for overlap, pred, gt in sorted(
            pairs,
            key=lambda item: (-item[0], -float(item[1]["score"]),
                              int(item[1]["_prediction_index"]), int(item[2]["id"])),
        ):
            pred_id = int(pred["_prediction_index"])
            gt_id = int(gt["id"])
            if pred_id in used_predictions or gt_id in used_annotations:
                continue
            used_predictions.add(pred_id)
            used_annotations.add(gt_id)
            matched.append((overlap, pred, gt))
        if matched:
            candidates[image_id] = matched

    rng = random.Random(args.seed)
    eligible_ids = sorted(candidates)
    rng.shuffle(eligible_ids)
    selected = []
    for image_id in eligible_ids[:args.count]:
        image_matches = sorted(candidates[image_id], key=lambda item: int(item[1]["_prediction_index"]))
        selected.append((image_id, image_matches[rng.randrange(len(image_matches))]))
    if len(selected) < args.count:
        raise ValueError(f"only {len(selected)} unique images have eligible matches; need {args.count}")

    from PIL import Image

    (output / "images").mkdir(parents=True)
    (output / "base_masks").mkdir()
    (output / "ground_truth_masks").mkdir()
    items = []
    for image_id, (match_iou, pred, gt) in selected:
        image_info = images[image_id]
        source = (image_root / image_info["file_name"]).resolve(strict=True)
        sample_id = f"coco-{image_id}-pred-{int(pred['_prediction_index'])}"
        image_rel = Path("images") / f"{sample_id}{source.suffix.lower()}"
        shutil.copyfile(source, output / image_rel)
        width, height = int(image_info["width"]), int(image_info["height"])
        if pred["segmentation"].get("size") != [height, width]:
            raise ValueError(f"prediction mask dimensions mismatch for {sample_id}")
        base_mask = decode_rle(pred["segmentation"], height, width)
        gt_mask = decode_rle(gt["segmentation"], height, width)
        base_rel = Path("base_masks") / f"{sample_id}.png"
        gt_rel = Path("ground_truth_masks") / f"{sample_id}.png"
        Image.fromarray(base_mask.astype("uint8") * 255, mode="L").save(output / base_rel)
        Image.fromarray(gt_mask.astype("uint8") * 255, mode="L").save(output / gt_rel)
        x, y, box_w, box_h = map(float, pred["bbox"])
        if box_w <= 0 or box_h <= 0:
            raise ValueError(f"invalid prompt box for {sample_id}")
        item = {
            "sample_id": sample_id,
            "image": image_rel.as_posix(),
            "bbox_xyxy": [x, y, x + box_w, y + box_h],
            "base_mask": base_rel.as_posix(),
            "ground_truth_mask": gt_rel.as_posix(),
            "sha256": {
                "image": sha256(output / image_rel),
                "base_mask": sha256(output / base_rel),
                "ground_truth_mask": sha256(output / gt_rel),
            },
            "prompt_source": "YOLOE-26-L-seg predicted box; no ground-truth box used for prompt",
            "category_id": int(pred["category_id"]),
            "prediction_index": int(pred["_prediction_index"]),
            "prediction_score": float(pred["score"]),
            "matched_ground_truth_annotation_id": int(gt["id"]),
            "same_category_box_iou": float(match_iou),
        }
        items.append(item)

    manifest = {
        "schema": "sam31-template-refinement-v1",
        "dataset": "COCO val2017 paired detector-mask refinement proxy",
        "split": "fixed YOLOE-26-L-seg COCO val2017 subset; one-to-one same-category bbox matches at IoU >= threshold; one object per image",
        "sam3_repo_commit": SAM_COMMIT,
        "sampling": {"seed": args.seed, "count": args.count,
                     "seed42_detector_subset_count": len(pred_doc["sample_image_ids"]),
                     "minimum_box_iou": args.min_box_iou,
                     "one_to_one_greedy_match": True,
                     "selection_uses_mask_iou": False},
        "sources": {
            "predictions": str(predictions_path), "predictions_sha256": sha256(predictions_path),
            "ground_truth_subset": str(gt_path), "ground_truth_subset_sha256": sha256(gt_path),
        },
        "items": items,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "output_dir": str(output),
                      "manifest": str(output / "manifest.json"), "items": len(items),
                      "eligible_images": len(candidates), "seed": args.seed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
