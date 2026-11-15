#!/usr/bin/env python3
"""CPU-only COCO scorer for the YOLOE-26 four-arm benchmark.

Invoked exactly as (no GPU, no model inference, no project dependencies):

  /usr/bin/python3 yoloe_coco_score.py \
      --predictions <arm>_predictions.json \
      --annotations /mnt/HDD1/Project_Data/public_datasets/coco/annotations/instances_val2017.json \
      --image-root /mnt/HDD1/Project_Data/public_datasets/coco/val2017 \
      --output-dir <dir>

The preflight hard-fails unless the interpreter, OS user, Python version,
pycocotools version and resolved module path match the values pinned by
accepted plan v5; the resolved values are recorded in the metrics output.
Scoring is refused for inference outputs whose status is not "complete"
(interrupted or failed runs are never silently scored) and for segmentation
arms whose "segm_results" key is missing or not a list (an explicit empty
list is valid).

Scoring rules (pycocotools 2.0.11, verified against the installed source):
  * Ground truth keeps the original annotation rows for the selected image
    IDs, untouched: crowd flags and areas are preserved and only image
    records are subset. Categories keep all 80 rows.
  * Bbox loadRes rows: image_id, actual category_id, COCO [x,y,width,height],
    score, box area, iscrowd=0.
  * Segmentation loadRes rows: image_id, actual category_id, segmentation,
    score. bbox is omitted so pycocotools derives the area from the RLE
    (maskUtils.area) instead of replacing it with the box area.
  * The uncompressed Fortran-order RLE counts persisted by inference are
    converted to compressed RLE with mask.frPyObjects before loadRes; the
    compressed and uncompressed forms are round-trip verified with
    maskUtils.decode, every RLE canvas is shape-checked against the
    annotation image size, and each row's persisted SHA-256 digest of the
    source binary mask raster (canvas dimensions + Fortran-order
    declaration + exact binary bytes) is recomputed from the decoded
    compressed RLE of the exact JSON artifact and must match, proving
    orientation and content survive the conversion. Benchmark metadata
    (the digest) is stripped from the loadRes rows.
  * Each evaluator uses the exact selected imgIds, the exact annotation
    category IDs, and maxDets [1, 10, 100] with the default IoU thresholds.
  * Empty result lists are supported without fabricating detections:
    pycocotools 2.0.11 loadRes cannot accept an empty list, so a valid empty
    COCO detections object is constructed directly (mirroring loadRes).

Outputs written to --output-dir:
  gt_subset_instances.json       the exact subset GT passed to COCO()
  bbox_loadres_input.json        the exact bbox rows passed to loadRes
  segm_loadres_input.json        the exact segm rows (compressed RLE), seg arms only
  bbox_evaluator_params.json     the exact params of the bbox COCOeval
  segm_evaluator_params.json     the exact params of the segm COCOeval, seg arms only
  metrics.json                   resolved scorer environment, inputs, GT info,
                                 and AP/AP50/AP75/AR100 per metric
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import pwd
import re
import sys
import time
from pathlib import Path

EXPECTED_INTERPRETER = "/usr/bin/python3"
EXPECTED_USER = "nils"
EXPECTED_PYTHON_VERSION = "3.10.12"
EXPECTED_PYCOCOTOOLS_VERSION = "2.0.11"
EXPECTED_MODULE_PATH = "/home/nils/.local/lib/python3.10/site-packages/pycocotools/__init__.py"

DEFAULT_COCO_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/coco")
DEFAULT_ANNOTATIONS = DEFAULT_COCO_ROOT / "annotations" / "instances_val2017.json"
DEFAULT_IMAGE_ROOT = DEFAULT_COCO_ROOT / "val2017"

MAX_DETS = [1, 10, 100]

# stats layout of COCOeval.summarize() for bbox/segm (_summarizeDets):
# [0] AP, [1] AP50, [2] AP75, [3..5] AP small/medium/large,
# [6] AR@1, [7] AR@10, [8] AR@100, [9..11] AR small/medium/large.
STAT_INDEX = {"AP": 0, "AP50": 1, "AP75": 2, "AR100": 8}

# 64 lowercase hex characters: the SHA-256 source-raster digest persisted
# per segmentation row by inference ('raster_sha256').
_RASTER_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight_scorer_environment() -> dict:
    """Validate the exact interpreter/user/version/module-path pin, then import."""
    resolved = {
        "sys_executable": sys.executable,
        "user": None,
        "python_version": platform.python_version(),
        "pycocotools_version": None,
        "module_path": None,
    }
    try:
        resolved["user"] = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        pass
    try:
        import pycocotools
        resolved["module_path"] = getattr(pycocotools, "__file__", None)
    except ImportError:
        pass
    try:
        resolved["pycocotools_version"] = importlib.metadata.version("pycocotools")
    except importlib.metadata.PackageNotFoundError:
        pass

    expected = {
        "sys_executable": EXPECTED_INTERPRETER,
        "user": EXPECTED_USER,
        "python_version": EXPECTED_PYTHON_VERSION,
        "pycocotools_version": EXPECTED_PYCOCOTOOLS_VERSION,
        "module_path": EXPECTED_MODULE_PATH,
    }
    problems = [key for key in expected if resolved[key] != expected[key]]
    if problems:
        print("[yoloe-coco-score] preflight FAILURE - pinned scorer environment mismatch:",
              file=sys.stderr)
        for key in expected:
            marker = "MISMATCH" if key in problems else "ok"
            print(f"  {marker:8s} {key}: expected {expected[key]!r}, resolved {resolved[key]!r}",
                  file=sys.stderr)
        print("[yoloe-coco-score] re-run with: "
              f"{EXPECTED_INTERPRETER} {Path(__file__).resolve()} ... as user {EXPECTED_USER}",
              file=sys.stderr)
        sys.exit(2)

    # The exact imports named by the accepted contract (pycocotools 2.0.11's
    # top-level package exports nothing; COCO/COCOeval live in submodules).
    from pycocotools.coco import COCO  # noqa: F401  (import verified)
    from pycocotools.cocoeval import COCOeval  # noqa: F401  (import verified)
    import pycocotools.mask as mask_utils  # noqa: F401  (import verified)
    return resolved


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"[yoloe-coco-score] {message}", file=sys.stderr)
    sys.exit(1)


def load_predictions(path: Path) -> dict:
    if not path.is_file():
        fail(f"predictions file not found: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"predictions file is not valid JSON ({path}): {exc}")
    status = record.get("status")
    if status != "complete":
        fail(
            f"refusing to score an interrupted, failed, or partial inference output "
            f"(status={status!r}) at {path}; finish the arm run first so the "
            "output is not silently treated as complete"
        )
    for key in ("arm", "arm_spec", "parameters", "class_map", "sample_image_ids", "bbox_results"):
        if key not in record:
            fail(f"predictions file is missing required key {key!r}: {path}")
    if not isinstance(record["arm_spec"], dict):
        fail(f"predictions file 'arm_spec' must be an object: {path}")
    if not isinstance(record["bbox_results"], list):
        fail(f"predictions file 'bbox_results' must be a list: {path}")
    if record["arm_spec"].get("segmentation"):
        if "segm_results" not in record:
            fail(
                f"predictions file for segmentation arm {record['arm']!r} is missing "
                "the 'segm_results' key; a missing mask list is malformed and is "
                f"refused instead of being scored as empty: {path}"
            )
        if not isinstance(record["segm_results"], list):
            fail(
                "predictions file 'segm_results' must be a list (an explicit empty "
                f"list is valid), got {type(record['segm_results']).__name__}: {path}"
            )
    return record


def build_gt_subset(record: dict, annotations_path: Path) -> tuple[dict, dict, list[int]]:
    """Subset GT: all original annotation rows for the selected image IDs.

    Crowd flags and areas are preserved verbatim; only image records are
    subset. The full 80-category list is kept and must equal the category
    IDs of the persisted class map.
    """
    if not annotations_path.is_file():
        fail(f"annotation file not found: {annotations_path}")
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    selected = [int(image_id) for image_id in record["sample_image_ids"]]
    annotation_image_ids = {int(row["id"]) for row in payload["images"]}
    missing = [image_id for image_id in selected if image_id not in annotation_image_ids]
    if missing:
        fail(f"{len(missing)} selected image IDs are absent from the annotations "
             f"(first: {missing[:5]})")
    selected_set = set(selected)
    images = [row for row in payload["images"] if int(row["id"]) in selected_set]
    annotations = [row for row in payload["annotations"] if int(row["image_id"]) in selected_set]
    categories = list(payload["categories"])
    if len(images) != len(selected):
        fail(f"subset lost images: {len(selected)} selected, {len(images)} kept")

    mapped_ids = {int(row["category_id"]) for row in record["class_map"]}
    category_ids = sorted(int(row["id"]) for row in categories)
    if mapped_ids != set(category_ids):
        fail("class map category IDs differ from the annotation category IDs; "
             "the category mapping is inconsistent between inference and annotations")

    info = {
        "image_count": len(images),
        "annotation_count": len(annotations),
        "crowd_annotation_count": sum(1 for row in annotations if row.get("iscrowd")),
        "area_preserved": True,
        "crowd_and_area_metadata_dropped": False,
        "category_count": len(categories),
        "category_ids": category_ids,
    }
    return {"images": images, "annotations": annotations, "categories": categories}, info, selected


# ---------------------------------------------------------------------------
# Result rows
# ---------------------------------------------------------------------------

def check_rows(rows: list[dict], selected: set, mapped_ids: set, label: str) -> None:
    seen_ids = set()
    for position, row in enumerate(rows):
        image_id = int(row["image_id"])
        category_id = int(row["category_id"])
        if image_id not in selected:
            fail(f"{label} row {position}: image_id {image_id} is outside the selected subset")
        if category_id not in mapped_ids:
            fail(f"{label} row {position}: category_id {category_id} is not one of the "
                 f"{len(mapped_ids)} mapped COCO category IDs")
        seen_ids.add(image_id)
    _ = seen_ids


def bbox_rows_for_loadres(record: dict) -> list[dict]:
    """Exact bbox rows: COCO xywh, box area, iscrowd=0."""
    rows = []
    for row in record["bbox_results"]:
        x, y, width, height = (float(value) for value in row["bbox"])
        rows.append({
            "image_id": int(row["image_id"]),
            "category_id": int(row["category_id"]),
            "bbox": [x, y, width, height],
            "score": float(row["score"]),
            "area": float(row["area"]),
            "iscrowd": 0,
        })
    return rows


def pythonify_rle(rle) -> dict:
    """Return a JSON-serializable RLE dict (compressed counts as a str)."""
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("latin-1")
    elif not isinstance(counts, str):
        counts = [int(value) for value in counts]
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def canonical_raster_bytes(raster) -> bytes:
    """Canonical serialization of a binary mask raster.

    Must stay byte-identical to ``mask_raster_digest`` in
    yoloe_component_benchmark.py: the ASCII header
    ``"<height>x<width>:F\\n"`` declares the canvas dimensions and the
    Fortran (column-major) raster order, followed by one byte per raster
    cell in Fortran order (0x00 background, 0x01 foreground). The SHA-256
    of this serialization is the source-raster digest persisted per
    segmentation row ('raster_sha256').
    """
    import numpy as np

    array = (np.asarray(raster) != 0).astype(np.uint8)
    if array.ndim != 2:
        raise ValueError(f"mask rasters must be 2-D, got shape {array.shape}")
    height, width = array.shape
    return f"{height}x{width}:F\n".encode("ascii") + array.tobytes(order="F")


def compress_and_verify(record: dict, image_dimensions: dict, mask_utils) -> list[dict]:
    """Convert uncompressed inference RLEs to compressed RLE via frPyObjects.

    mask.frPyObjects takes one canvas (h, w) per call, so rows are grouped by
    canvas; every row's RLE canvas must equal the annotation image size. Each
    row is round-trip verified with real pycocotools: the uncompressed
    list-count RLE is converted with mask.frPyObjects first (mask.decode
    cannot consume a raw uncompressed counts list), and the mask decoded from
    the converted form must equal the mask decoded from the exact persisted
    compressed row. Persisted rows keep the compressed form. Each row must
    also carry the SHA-256 source-raster digest ('raster_sha256'); the
    decoded persisted compressed RLE must reproduce the decoded mask's
    canonical serialization digest, proving the conversion preserved the
    source raster's orientation and content.
    """
    rows = record["segm_results"]
    by_canvas: dict[tuple[int, int], list[int]] = {}
    for position, row in enumerate(rows):
        rle = row["segmentation"]
        if not isinstance(rle, dict) or "size" not in rle or "counts" not in rle:
            fail(f"segm row {position}: segmentation is not an RLE dict")
        if not isinstance(rle["counts"], list):
            fail(f"segm row {position}: uncompressed inference RLE counts must be a "
                 f"list of integer run lengths, got {type(rle['counts']).__name__}")
        height, width = int(rle["size"][0]), int(rle["size"][1])
        expected = image_dimensions[int(row["image_id"])]
        if (height, width) != expected:
            fail(f"segm row {position} (image {row['image_id']}): RLE canvas "
                 f"{(height, width)} does not match the annotation canvas {expected}; "
                 "masks are never resized")
        raster_digest = row.get("raster_sha256")
        if not isinstance(raster_digest, str) or not _RASTER_DIGEST_RE.fullmatch(raster_digest):
            fail(f"segm row {position} (image {row['image_id']}): source-raster digest "
                 "'raster_sha256' is missing or malformed (expected a 64-hex-char "
                 f"SHA-256 string), got {raster_digest!r}")
        by_canvas.setdefault((height, width), []).append(position)

    compressed = [None] * len(rows)
    for (height, width), positions in sorted(by_canvas.items()):
        rles = [rows[position]["segmentation"] for position in positions]
        converted = mask_utils.frPyObjects(rles, height, width)
        if len(converted) != len(positions):
            fail(f"mask.frPyObjects returned {len(converted)} RLEs for {len(positions)} inputs")
        for position, converted_rle in zip(positions, converted):
            row_image_id = rows[position]["image_id"]
            compressed[position] = pythonify_rle(converted_rle)
            # mask.decode cannot consume a raw uncompressed list-count RLE:
            # decode the frPyObjects-converted form and the exact persisted
            # (str-count) row, then compare the decoded masks.
            decoded_converted = mask_utils.decode(converted_rle)
            decoded_persisted = mask_utils.decode(
                {
                    "size": compressed[position]["size"],
                    "counts": compressed[position]["counts"].encode("latin-1"),
                }
            )
            if decoded_converted.shape != decoded_persisted.shape or \
                    not _array_equal(decoded_converted, decoded_persisted):
                fail(f"segm row {position} (image {row_image_id}): compressed and "
                     "uncompressed RLE round-trip mismatch; the inference RLE encoder output "
                     "does not survive pycocotools 2.0.11 conversion")
            # Shape validation: the decoded canvas must equal the RLE's
            # declared size (the per-row check above already ties that
            # size to the annotation image size).
            if decoded_persisted.shape != (height, width) or \
                    decoded_converted.shape != (height, width):
                fail(f"segm row {position} (image {row_image_id}): decoded RLE shape "
                     f"{decoded_persisted.shape} does not match the declared canvas "
                     f"size {(height, width)}")
            # Source-raster digest: decode the exact JSON artifact (the
            # persisted compressed RLE), reconstruct the canonical binary
            # raster serialization, and require the SHA-256 to equal the
            # digest inference computed while the source raster was still
            # in memory. A mismatch means orientation or content did not
            # survive the uncompressed-to-compressed conversion.
            reconstructed_digest = hashlib.sha256(
                canonical_raster_bytes(decoded_persisted)
            ).hexdigest()
            persisted_digest = rows[position]["raster_sha256"]
            if reconstructed_digest != persisted_digest:
                fail(f"segm row {position} (image {row_image_id}): decoded compressed-RLE "
                     f"raster digest {reconstructed_digest} does not match the persisted "
                     f"source-raster digest {persisted_digest}; the RLE conversion does "
                     "not reproduce the source mask raster")
    return compressed


def _array_equal(left, right) -> bool:
    import numpy as np

    return bool(np.array_equal(left, right))


def segm_rows_for_loadres(record: dict, compressed: list[dict]) -> list[dict]:
    """Exact segm rows: RLE + score, no bbox (area is derived from the RLE).

    Rows are rebuilt from named keys only, so benchmark metadata persisted
    by inference (the 'raster_sha256' source-raster digest) is stripped and
    never reaches COCO loadRes.
    """
    rows = []
    for row, rle in zip(record["segm_results"], compressed):
        rows.append({
            "image_id": int(row["image_id"]),
            "category_id": int(row["category_id"]),
            "segmentation": rle,
            "score": float(row["score"]),
        })
    return rows


def empty_coco_dt(coco_gt):
    """Build a valid empty COCO detections object.

    pycocotools 2.0.11 loadRes IndexErrors on an empty list (it branches on
    anns[0]); this mirrors loadRes's own construction with zero annotations
    so the evaluator runs on a real empty result set instead of fabricated
    detections.
    """
    from pycocotools.coco import COCO

    dt = COCO()
    dt.dataset["info"] = copy.deepcopy(coco_gt.dataset.get("info", {}))
    dt.dataset["images"] = [img for img in coco_gt.dataset["images"]]
    dt.dataset["categories"] = copy.deepcopy(coco_gt.dataset.get("categories", {}))
    dt.dataset["annotations"] = []
    dt.createIndex()
    return dt


def write_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def params_to_dict(params) -> dict:
    return {
        "iouType": params.iouType,
        "imgIds": [int(value) for value in params.imgIds],
        "catIds": [int(value) for value in params.catIds],
        "iouThrs": [float(value) for value in params.iouThrs],
        "recThrs": [float(value) for value in params.recThrs],
        "maxDets": [int(value) for value in params.maxDets],
        "areaRng": [[float(low), float(high)] for low, high in params.areaRng],
        "areaRngLbl": list(params.areaRngLbl),
        "useCats": int(params.useCats),
    }


def score_one(coco_gt, dt, iou_type: str, selected: list[int], category_ids: list[int]):
    """Run one COCOeval (bbox or segm) and return metrics + params + stdout."""
    from pycocotools.cocoeval import COCOeval

    evaluator = COCOeval(coco_gt, dt, iou_type)
    evaluator.params.imgIds = list(selected)
    evaluator.params.catIds = list(category_ids)
    evaluator.params.maxDets = list(MAX_DETS)
    params_before = params_to_dict(evaluator.params)

    buffer = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(buffer):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    seconds = time.perf_counter() - started

    stats = evaluator.stats
    metrics = {key: float(stats[index]) for key, index in STAT_INDEX.items()}
    return {
        "detection_count": len(dt.dataset.get("annotations", [])),
        "AP": metrics["AP"],
        "AP50": metrics["AP50"],
        "AP75": metrics["AP75"],
        "AR100": metrics["AR100"],
        "stats": [float(value) for value in stats],
        "maxDets": list(MAX_DETS),
        "evaluator_params": params_before,
        "evaluator_params_after_run": params_to_dict(evaluator.params),
        "evaluator_stdout": buffer.getvalue(),
        "seconds": round(seconds, 3),
        "note": "AP/AP50/AP75 use the standard top-100 cutoff (maxDets [1,10,100]); "
                "a value of -1.0 means pycocotools found no detections for that slice",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COCO scorer for the YOLOE-26 four-arm benchmark (CPU, pycocotools 2.0.11).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", type=Path, required=True,
                        help="self-contained inference output JSON (status must be 'complete')")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <predictions dir>/score")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolved = preflight_scorer_environment()

    predictions_path = args.predictions.expanduser().resolve()
    output_dir = (args.output_dir or predictions_path.parent / "score").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.image_root.is_dir():
        fail(f"image root is not a directory: {args.image_root}")

    started = time.perf_counter()
    record = load_predictions(predictions_path)
    gt_subset, gt_info, selected = build_gt_subset(record, args.annotations)
    category_ids = gt_info["category_ids"]
    mapped_ids = {int(row["category_id"]) for row in record["class_map"]}
    selected_set = set(selected)

    gt_path = output_dir / "gt_subset_instances.json"
    write_json(gt_path, gt_subset)
    from pycocotools.coco import COCO
    coco_gt = COCO(str(gt_path))
    image_dimensions = {int(row["id"]): (int(row["height"]), int(row["width"]))
                        for row in gt_subset["images"]}

    # --- bbox -------------------------------------------------------------
    check_rows(record["bbox_results"], selected_set, mapped_ids, "bbox")
    bbox_rows = bbox_rows_for_loadres(record)
    bbox_input_path = output_dir / "bbox_loadres_input.json"
    write_json(bbox_input_path, bbox_rows)
    if bbox_rows:
        dt_bbox = coco_gt.loadRes(bbox_rows)
    else:
        dt_bbox = empty_coco_dt(coco_gt)
    bbox_metrics = score_one(coco_gt, dt_bbox, "bbox", selected, category_ids)
    write_json(output_dir / "bbox_evaluator_params.json", bbox_metrics["evaluator_params_after_run"])

    # --- segmentation (segmentation arms only) ----------------------------
    segm_metrics = None
    if record["arm_spec"].get("segmentation"):
        import pycocotools.mask as mask_utils
        check_rows(record["segm_results"], selected_set, mapped_ids, "segm")
        compressed = compress_and_verify(record, image_dimensions, mask_utils)
        segm_rows = segm_rows_for_loadres(record, compressed)
        segm_input_path = output_dir / "segm_loadres_input.json"
        write_json(segm_input_path, segm_rows)
        if segm_rows:
            dt_segm = coco_gt.loadRes(segm_rows)
        else:
            dt_segm = empty_coco_dt(coco_gt)
        segm_metrics = score_one(coco_gt, dt_segm, "segm", selected, category_ids)
        write_json(output_dir / "segm_evaluator_params.json", segm_metrics["evaluator_params_after_run"])

    metrics = {
        "schema": "yoloe26-benchmark-score/v1",
        "arm": record["arm"],
        "arm_spec": record["arm_spec"],
        "subset": {
            "label": f"COCO val2017 fixed {len(selected)}-image subset",
            "description": (
                "sorted random.Random(seed).sample of sorted annotation image IDs whose "
                "files exist, then sorted; identical across all four arms"
            ),
            "seed": record["parameters"]["seed"],
            "image_count": len(selected),
            "image_ids": selected,
        },
        "scorer_environment": {
            "sys_executable": resolved["sys_executable"],
            "user": resolved["user"],
            "python_version": resolved["python_version"],
            "pycocotools": {
                "version": resolved["pycocotools_version"],
                "module_path": resolved["module_path"],
            },
        },
        "inputs": {
            "predictions": str(predictions_path),
            "predictions_sha256": sha256(predictions_path),
            "annotations": str(args.annotations.resolve()),
            "annotations_sha256": sha256(args.annotations),
            "image_root": str(args.image_root.resolve()),
            "gt_subset_file": str(gt_path),
            "bbox_loadres_input_file": str(bbox_input_path),
            "segm_loadres_input_file": str(output_dir / "segm_loadres_input.json")
            if segm_metrics is not None else None,
            # The exact prompt-profile artifact used by detector arms (path,
            # size, SHA-256 as persisted in the inference JSON's provenance);
            # None for segmentation arms, which generate no profile.
            "prompt_profile": (record.get("provenance") or {}).get("prompt_profile"),
        },
        "ground_truth": gt_info,
        "maxDets": list(MAX_DETS),
        "bbox": bbox_metrics,
        "segm": segm_metrics,
        "interpretation": [
            "Scores are for the fixed 500-image COCO val2017 subset, not full val2017.",
            "Arms use the 80 hyphenated COCO category names as YOLOE prompts; AP values are "
            "not directly comparable to official prompt-tuned (-seg-pf) release numbers.",
            "The two detection-only arms are derived from the matching official segmentation "
            "checkpoints via the documented YOLOE YAML + saved prompt-profile route; they are "
            "not official detect-only releases.",
            "A metric value of -1.0 means pycocotools found no detections for that slice; "
            "no detections were fabricated for empty result lists.",
        ],
        "scored_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output_dir / "metrics.json", metrics)

    bbox = metrics["bbox"]
    print(f"[yoloe-coco-score] arm={record['arm']} bbox: AP={bbox['AP']:.3f} "
          f"AP50={bbox['AP50']:.3f} AP75={bbox['AP75']:.3f} AR100={bbox['AR100']:.3f} "
          f"(n={bbox['detection_count']})", flush=True)
    if segm_metrics is not None:
        segm = segm_metrics
        print(f"[yoloe-coco-score] arm={record['arm']} segm: AP={segm['AP']:.3f} "
              f"AP50={segm['AP50']:.3f} AP75={segm['AP75']:.3f} AR100={segm['AR100']:.3f} "
              f"(n={segm['detection_count']})", flush=True)
    print(f"[yoloe-coco-score] metrics written to {output_dir / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
