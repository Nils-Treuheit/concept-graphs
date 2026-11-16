#!/usr/bin/env python3
"""EOVSAM mask-fusion benchmark scorer for the frozen COCO500 contract v7.

Compares the baseline YOLOE-26-L-seg instance masks with deterministic
matched-replacement and matched-union arms on the fixed 500-image COCO
val2017 subset, exactly as frozen by ``eovsam-fusion-contract-v7.md``:

  * YOLOE boxes, classes, confidence scores, and the proposal count are
    unchanged in every arm; one row per original YOLOE detection;
  * same-class thing-only matching: candidate edges below a predicted-mask
    IoU of 0.50 are removed, then the feasible one-to-one assignment with
    maximum total predicted-mask IoU is chosen with SciPy 1.15.3
    ``linear_sum_assignment`` in deterministic order (detections by original
    detection index, segments by segment ID); unmatched detections keep
    their baseline masks; unmatched EOVSAM segments are never added;
  * COCO mask AP plus a separately reported box AP per arm on exactly the
    frozen 500 image IDs (pycocotools 2.0.11, maxDets [1, 10, 100]);
  * the baseline must reproduce mask AP 0.367780 and box AP 0.444025 within
    1e-6 before any final output is materialized; otherwise the run aborts
    fail-closed without writing arm result files or metrics;
  * every pinned input hash and the 500 image IDs are verified; malformed
    YOLOE masks (``yoloe_detections_without_valid_mask``) and invalid
    EOVSAM image outputs (``eovsam_image_output_errors``) block scoring;
  * final outputs are staged in a system temporary directory and only
    materialized under the frozen ``fusion/`` destination after all arms
    pass, with destinations required to remain absent (never overwritten).

Invoked exactly as (CPU only; no model inference, no GPU, no project
dependencies beyond the pinned scorer stack):

  /usr/bin/python3 fuse_masks.py

Every option defaults to its pinned contract value; an explicit override is
accepted only if it resolves exactly to the pinned path.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import pwd
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Pinned contract values (eovsam-fusion-contract-v7.md)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[4]

PINNED_YOLOE_JSON = (
    ROOT / "scripts" / "pipeline_model_reports" / "data" / "yoloe26"
    / "rerun_20261115_1912" / "yoloe-26l-seg.json"
)
PINNED_INSTANCE_GT = Path(
    "/mnt/HDD1/Project_Data/public_datasets/coco/annotations/instances_val2017.json"
)
PINNED_PANOPTIC_GT = Path(
    "/mnt/HDD1/Project_Data/public_datasets/coco/annotations/panoptic_val2017.json"
)
PINNED_SAMPLE_MANIFEST = (
    ROOT / "scripts" / "pipeline_model_reports" / "data" / "yoloe26"
    / "coco-val2017-500-manifest.json"
)
PINNED_INFERENCE_MANIFEST = (
    ROOT / "models" / "experiments" / "eovsam" / "output"
    / "coco500-mask-fusion-run2" / "inference" / "retained-output-manifest.json"
)
PINNED_RUN_MANIFEST = ROOT / "models" / "experiments" / "eovsam" / "adapter" / "run-manifest.json"
PINNED_RUN_ROOT = ROOT / "models" / "experiments" / "eovsam" / "output" / "coco500-mask-fusion-run2"
PINNED_PREDICTIONS_JSON = PINNED_RUN_ROOT / "inference" / "predictions.json"
PINNED_PNG_DIR = PINNED_RUN_ROOT / "inference" / "panoptic_predictions"
PINNED_FUSION_DIR = (
    ROOT / "models" / "experiments" / "eovsam" / "output"
    / "coco500-mask-fusion-score-run4" / "fusion"
)
PINNED_LOG_FILE = (
    ROOT / "models" / "experiments" / "eovsam" / "output"
    / "coco500-mask-fusion-score-run4.log"
)
PINNED_INFERENCE_LOG = (
    ROOT / "models" / "experiments" / "eovsam" / "output"
    / "coco500-mask-fusion-run2.log"
)
PINNED_FUSION_FILES = (
    "baseline_instances.json",
    "replace_instances.json",
    "union_instances.json",
    "metrics.json",
)
PINNED_PATHS = {
    "yoloe_json": PINNED_YOLOE_JSON,
    "instance_gt": PINNED_INSTANCE_GT,
    "panoptic_gt": PINNED_PANOPTIC_GT,
    "sample_manifest": PINNED_SAMPLE_MANIFEST,
    "inference_manifest": PINNED_INFERENCE_MANIFEST,
    "run_manifest": PINNED_RUN_MANIFEST,
    "run_root": PINNED_RUN_ROOT,
    "predictions_json": PINNED_PREDICTIONS_JSON,
    "png_dir": PINNED_PNG_DIR,
    "fusion_dir": PINNED_FUSION_DIR,
    "log_file": PINNED_LOG_FILE,
}
PINNED_SHA256 = {
    "yoloe_json": "8abc7db9e57f9629aef5a6ebe9945a091c324d01572d99fe89b0bcc0406d6f26",
    "instance_gt": "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f",
    "panoptic_gt": "454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a",
    "sample_manifest": "512abb4eabfcadf8ee4818a1dfb2beaef4763679bbf127119ddbdc7b4348af62",
    "inference_manifest": "a5bfd8914cd01ae544f40db473618f34024a64348c0592bd4919a591fac8789a",
}
EXPECTED_EVALUATOR_SHA256 = (
    "a58388d39118d3473016bca333c63a605cbca2fcf920a71accf154a5c29cc52b"
)

EXPECTED_MASK_AP = 0.367780
EXPECTED_BOX_AP = 0.444025
BASELINE_TOLERANCE = 1e-6
IOU_GATE = 0.50
MAX_DETS = [1, 10, 100]
DISALLOWED_COST = 1e6
ARM_NAMES = ("baseline", "replace", "union")
STAT_NAMES = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")

# Provenance references (read-only; not overwritten by this scorer).
EXISTING_SCORER_METRICS = (
    ROOT / "scripts" / "pipeline_model_reports" / "data" / "yoloe26"
    / "rerun_20261115_1912" / "yoloe-26l-seg-score" / "metrics.json"
)
FORK_EVALUATOR = (
    ROOT / "models" / "experiments" / "eovsam" / "detectron2-repo" / "detectron2"
    / "evaluation" / "panoptic_evaluation.py"
)

# Pinned scorer environment (mirrors the accepted yoloe_coco_score.py preflight).
EXPECTED_INTERPRETER = "/usr/bin/python3"
EXPECTED_USER = "nils"
EXPECTED_PYTHON_VERSION = "3.10.12"
EXPECTED_PYCOCOTOOLS_VERSION = "2.0.11"
EXPECTED_PYCOCOTOOLS_MODULE_PATH = "/home/nils/.local/lib/python3.10/site-packages/pycocotools/__init__.py"
EXPECTED_SCIPY_VERSION = "1.15.3"

_RASTER_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_DIGITS = frozenset("0123456789abcdef")


# ---------------------------------------------------------------------------
# Logging and failure handling
# ---------------------------------------------------------------------------

class FusionLog:
    """Concise scoring log; the file is the preserved failure record on abort."""

    def __init__(self, path: Path):
        if path.exists():
            raise SystemExit(
                "[fuse-masks] scoring log already exists; a prior attempt's log is never "
                f"overwritten: {path}"
            )
        self._path = path
        self._fh = path.open("w", encoding="utf-8")

    def line(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._fh.write(f"[{stamp}] {message}\n")
        self._fh.flush()
        print(f"[fuse-masks] {message}", flush=True)

    def close(self) -> None:
        self._fh.close()


_LOG: FusionLog | None = None


def log(message: str) -> None:
    if _LOG is not None:
        _LOG.line(message)
    else:
        print(f"[fuse-masks] {message}", flush=True)


def fail(message: str, code: int = 1) -> None:
    log(f"FAIL: {message}")
    sys.exit(code)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        not (isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))))


def require_int(value, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value}")
    return value


def pythonify_rle(rle) -> dict:
    """JSON-serializable RLE dict (compressed counts as a latin-1 str)."""
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("latin-1")
    elif not isinstance(counts, str):
        counts = [int(value) for value in counts]
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


# ---------------------------------------------------------------------------
# Pinned environment preflight
# ---------------------------------------------------------------------------

def preflight_environment() -> dict:
    """Validate the pinned interpreter/user/version pins, then import the stack."""
    resolved = {
        "sys_executable": sys.executable,
        "user": None,
        "python_version": platform.python_version(),
        "pycocotools_version": None,
        "pycocotools_module_path": None,
        "scipy_version": None,
        "numpy_version": None,
        "pil_version": None,
    }
    try:
        resolved["user"] = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        pass
    for key, package in (
        ("pycocotools_version", "pycocotools"),
        ("scipy_version", "scipy"),
        ("numpy_version", "numpy"),
        ("pil_version", "PIL"),
    ):
        try:
            resolved[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        import pycocotools

        resolved["pycocotools_module_path"] = getattr(pycocotools, "__file__", None)
    except ImportError:
        pass

    expected = {
        "sys_executable": EXPECTED_INTERPRETER,
        "user": EXPECTED_USER,
        "python_version": EXPECTED_PYTHON_VERSION,
        "pycocotools_version": EXPECTED_PYCOCOTOOLS_VERSION,
        "pycocotools_module_path": EXPECTED_PYCOCOTOOLS_MODULE_PATH,
        "scipy_version": EXPECTED_SCIPY_VERSION,
    }
    problems = [key for key in expected if resolved[key] != expected[key]]
    if problems:
        print("[fuse-masks] preflight FAILURE - pinned scorer environment mismatch:", file=sys.stderr)
        for key in expected:
            marker = "MISMATCH" if key in problems else "ok"
            print(f"  {marker:8s} {key}: expected {expected[key]!r}, resolved {resolved[key]!r}",
                  file=sys.stderr)
        print("[fuse-masks] re-run as user nils with the accepted pinned stack "
              "(/usr/bin/python3, pycocotools 2.0.11, SciPy 1.15.3)", file=sys.stderr)
        sys.exit(2)

    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from scipy.optimize import linear_sum_assignment
    import pycocotools.mask as mask_utils

    resolved["np"] = np
    resolved["Image"] = Image
    resolved["COCO"] = COCO
    resolved["COCOeval"] = COCOeval
    resolved["linear_sum_assignment"] = linear_sum_assignment
    resolved["mask_utils"] = mask_utils
    return resolved


# ---------------------------------------------------------------------------
# CLI: pinned inputs only
# ---------------------------------------------------------------------------

def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="EOVSAM mask-fusion scorer (frozen COCO500 contract v7). "
        "All options default to the pinned contract values; an override is "
        "accepted only if it resolves exactly to the pinned path.",
    )
    for option in ("yoloe_json", "instance_gt", "panoptic_gt", "sample_manifest",
                   "inference_manifest",
                   "run_manifest", "run_root", "predictions_json", "png_dir",
                   "fusion_dir", "log_file"):
        parser.add_argument(f"--{option.replace('_', '-')}", type=Path, default=None,
                            help=f"override for the pinned {option} (must equal the pinned value)")
    args = parser.parse_args()
    resolved = {}
    for option, pinned in PINNED_PATHS.items():
        value = getattr(args, option)
        if value is None:
            resolved[option] = pinned
        else:
            candidate = value.expanduser().resolve()
            if candidate != pinned:
                fail(f"override for --{option.replace('_', '-')} does not exactly match the "
                     f"pinned contract value: got {candidate}, pinned {pinned}", code=2)
            resolved[option] = pinned
    return resolved


# ---------------------------------------------------------------------------
# Pinned input verification (all hashes, counts, and IDs)
# ---------------------------------------------------------------------------

def verify_pinned_inputs(paths: dict, started: float) -> dict:
    """Verify every pinned input hash, the 500-image IDs, and destination absence."""
    hashes = {}
    for key, pinned_hash in PINNED_SHA256.items():
        path = paths[key]
        if not path.is_file():
            fail(f"pinned input missing: {path}")
        digest = sha256_file(path)
        if digest != pinned_hash:
            fail(f"pinned input hash mismatch for {path}: got {digest}, "
                 f"pinned {pinned_hash}")
        hashes[key] = digest
        log(f"verified {key}: {path} sha256={digest}")

    sample_manifest = json.loads(paths["sample_manifest"].read_text(encoding="utf-8"))
    image_ids = [require_int(value, "sample image id") for value in sample_manifest["sample_image_ids"]]
    if len(image_ids) != 500 or len(set(image_ids)) != 500 or image_ids != sorted(image_ids):
        fail(f"sample manifest must contain 500 unique, sorted image IDs; got {len(image_ids)}")
    if sample_manifest.get("sample_count") != 500 or sample_manifest.get("sample_seed") != 42:
        fail("sample manifest count/seed is not the frozen 500-image seed-42 sample")

    inference_manifest = json.loads(
        paths["inference_manifest"].read_text(encoding="utf-8")
    )
    if inference_manifest.get("schema") != "eovsam-run2-inference-artifact-manifest/v1":
        fail("run2 retained-output manifest has an unexpected schema")
    if inference_manifest.get("sample_manifest_sha256") != PINNED_SHA256["sample_manifest"]:
        fail("run2 retained-output manifest references a different sample manifest")
    if inference_manifest.get("evaluator_source_sha256") != EXPECTED_EVALUATOR_SHA256:
        fail("run2 retained-output manifest references a different evaluator source")
    manifest_image_ids = [
        require_int(value, "run2 retained-output image id")
        for value in inference_manifest.get("image_ids", [])
    ]
    if manifest_image_ids != image_ids:
        fail("run2 retained-output image IDs differ from the pinned sample manifest")
    if inference_manifest.get("panoptic_png_count") != 500:
        fail("run2 retained-output manifest must list exactly 500 panoptic PNGs")
    if not paths["predictions_json"].is_file():
        fail(f"run2 predictions JSON is missing: {paths['predictions_json']}")
    if not paths["png_dir"].is_dir():
        fail(f"run2 retained PNG directory is missing: {paths['png_dir']}")
    if sha256_file(paths["predictions_json"]) != inference_manifest.get(
        "predictions_json_sha256"
    ):
        fail("run2 predictions.json differs from its retained-output manifest hash")
    predictions_payload = json.loads(paths["predictions_json"].read_text(encoding="utf-8"))
    prediction_annotations = predictions_payload.get("annotations")
    if not isinstance(prediction_annotations, list) or len(prediction_annotations) != 500:
        fail("run2 predictions.json must contain exactly 500 annotations")
    prediction_names = [row.get("file_name") for row in prediction_annotations
                        if isinstance(row, dict)]
    if len(prediction_names) != 500 or any(
        not isinstance(name, str) or not name.endswith(".png") or
        Path(name).name != name or "/" in name or "\\" in name or name in (".", "..")
        for name in prediction_names
    ):
        fail("run2 predictions.json contains an unsafe or invalid PNG filename")
    if len(set(prediction_names)) != 500:
        fail("run2 predictions.json has duplicate PNG filenames")
    png_hashes = inference_manifest.get("panoptic_png_sha256")
    if not isinstance(png_hashes, dict) or set(png_hashes) != set(prediction_names):
        fail("run2 retained-output PNG filename set differs from predictions.json")
    for file_name, expected_hash in sorted(png_hashes.items()):
        png_path = paths["png_dir"] / file_name
        if not png_path.is_file():
            fail(f"run2 retained PNG missing: {png_path}")
        actual_hash = sha256_file(png_path)
        if actual_hash != expected_hash:
            fail(f"run2 retained PNG hash mismatch for {png_path}: got {actual_hash}, "
                 f"expected {expected_hash}")
    run_root = paths["run_root"]
    sidecars = {
        "runner_metadata_sha256": run_root / "runner-metadata.txt",
        "config_sha256": run_root / "config.yaml",
        "console_log_sha256": PINNED_INFERENCE_LOG,
        "time_file_sha256": run_root / "time.txt",
    }
    for key, path in sidecars.items():
        if not path.is_file() or sha256_file(path) != inference_manifest.get(key):
            fail(f"run2 retained-output sidecar hash mismatch or missing file: {path}")
    log("run2 retained-output manifest verified: predictions JSON, 500 PNGs, "
        "runner metadata, config, console log, and timing file")

    if not paths["run_manifest"].is_file():
        fail(f"generated adapter run-manifest missing: {paths['run_manifest']}")
    if not FORK_EVALUATOR.is_file():
        fail(f"accepted fork evaluator source missing: {FORK_EVALUATOR}")
    evaluator_hash = sha256_file(FORK_EVALUATOR)
    if evaluator_hash != EXPECTED_EVALUATOR_SHA256:
        fail("fork evaluator source hash differs from the v4 accepted source: "
             f"got {evaluator_hash}, expected {EXPECTED_EVALUATOR_SHA256}")
    run_manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    if run_manifest.get("sample_manifest_sha256") != PINNED_SHA256["sample_manifest"]:
        fail("adapter run-manifest does not reference the pinned sample manifest hash")
    run_manifest_ids = [require_int(value, "run-manifest image id") for value in run_manifest["image_ids"]]
    if run_manifest_ids != image_ids:
        fail("adapter run-manifest image IDs differ from the pinned sample manifest")
    source_value = run_manifest.get("evaluator_source")
    if not isinstance(source_value, str) or not source_value:
        fail("adapter run-manifest has no evaluator_source path")
    try:
        source_path = Path(source_value).expanduser().resolve(strict=True)
        expected_source_path = FORK_EVALUATOR.resolve(strict=True)
    except OSError as exc:
        fail(f"adapter run-manifest evaluator_source cannot be resolved: {exc}")
    if source_path != expected_source_path:
        fail("adapter run-manifest evaluator_source differs from the accepted fork: "
             f"got {source_path}, expected {expected_source_path}")
    if run_manifest.get("evaluator_source_sha256") != EXPECTED_EVALUATOR_SHA256:
        fail("adapter run-manifest evaluator_source_sha256 differs from the v4 accepted source")
    hashes["run_manifest"] = sha256_file(paths["run_manifest"])
    log(f"verified generated run_manifest: {paths['run_manifest']} "
        f"sha256={hashes['run_manifest']} evaluator_sha256={evaluator_hash}")
    log(f"verified run2 output manifest: {paths['inference_manifest']} "
        f"sha256={hashes['inference_manifest']}")

    instance_gt = json.loads(paths["instance_gt"].read_text(encoding="utf-8"))
    gt_images = {int(row["id"]): (int(row["width"]), int(row["height"])) for row in instance_gt["images"]}
    missing_gt = [image_id for image_id in image_ids if image_id not in gt_images]
    if missing_gt:
        fail(f"{len(missing_gt)} sample image IDs are absent from the instance GT "
             f"(first: {missing_gt[:5]})")
    gt_dims = {image_id: gt_images[image_id] for image_id in image_ids}
    gt_category_ids = sorted(int(row["id"]) for row in instance_gt["categories"])
    if len(gt_category_ids) != 80 or len(set(gt_category_ids)) != 80:
        fail(f"instance GT must contain the 80 COCO thing categories; got {len(gt_category_ids)}")
    gt_ann_count = sum(1 for row in instance_gt["annotations"] if int(row["image_id"]) in set(image_ids))
    log(f"instance GT: {gt_ann_count} annotation rows over the 500 frozen images")

    panoptic_gt = json.loads(paths["panoptic_gt"].read_text(encoding="utf-8"))
    panoptic_categories = {}
    for row in panoptic_gt["categories"]:
        category_id = int(row["id"])
        if category_id in panoptic_categories:
            fail(f"panoptic category table has duplicate category ID {category_id}")
        panoptic_categories[category_id] = bool(row["isthing"])
    if len(panoptic_categories) != 133:
        fail(f"panoptic category table must contain 133 categories; got {len(panoptic_categories)}")
    panoptic_thing_ids = sorted(cid for cid, isthing in panoptic_categories.items() if isthing)
    if panoptic_thing_ids != gt_category_ids:
        fail("panoptic thing category IDs differ from the instance GT category IDs")
    panoptic_dims = {int(row["id"]): (int(row["width"]), int(row["height"])) for row in panoptic_gt["images"]}
    dim_mismatch = [image_id for image_id in image_ids
                    if image_id in panoptic_dims and panoptic_dims[image_id] != gt_dims[image_id]]
    if dim_mismatch:
        fail(f"panoptic and instance GT dimensions disagree for {len(dim_mismatch)} images "
             f"(first: {dim_mismatch[:5]})")
    panoptic_ann_count = sum(1 for row in panoptic_gt["annotations"] if int(row["image_id"]) in set(image_ids))
    if panoptic_ann_count != 500:
        fail(f"panoptic GT must hold exactly one annotation row per frozen image; got {panoptic_ann_count}")
    log("panoptic GT category table verified (133 categories; 80 things == instance GT IDs)")

    for key in ("predictions_json",):
        path = paths[key]
        if not path.is_file():
            fail(f"EOVSAM prediction JSON is missing (inference not complete?): {path}")
    if not paths["png_dir"].is_dir():
        fail(f"EOVSAM retained PNG directory is missing (evaluator persistence not active?): "
             f"{paths['png_dir']}")
    if paths["run_root"].exists() and not paths["run_root"].is_dir():
        fail(f"run root exists but is not a directory: {paths['run_root']}")

    for option in ("fusion_dir",):
        if paths[option].exists():
            fail(f"frozen fusion destination already exists; never overwritten: {paths[option]}")
        if paths[option].parent.exists():
            fail(f"fresh score-run parent already exists; never merged or overwritten: "
                 f"{paths[option].parent}")
    for name in PINNED_FUSION_FILES:
        if (paths["fusion_dir"] / name).exists():
            fail(f"frozen fusion destination file already exists; never overwritten: "
                 f"{paths['fusion_dir'] / name}")

    ctx = {
        "hashes": hashes,
        "image_ids": image_ids,
        "image_id_set": set(image_ids),
        "gt_dims": gt_dims,
        "gt_category_ids": gt_category_ids,
        "gt_category_set": set(gt_category_ids),
        "instance_gt": instance_gt,
        "panoptic_categories": panoptic_categories,
        "panoptic_thing_set": set(panoptic_thing_ids),
        "gt_annotation_rows": [row for row in instance_gt["annotations"]
                               if int(row["image_id"]) in set(image_ids)],
    }
    log(f"input verification complete in {time.perf_counter() - started:.1f}s "
        "(all pinned hashes, 500 image IDs, category tables, and absent destinations verified)")
    return ctx


# ---------------------------------------------------------------------------
# YOLOE frozen predictions
# ---------------------------------------------------------------------------

def load_and_verify_yoloe(paths: dict, ctx: dict) -> dict:
    """Verify the frozen YOLOE record and its per-row consistency."""
    record = json.loads(paths["yoloe_json"].read_text(encoding="utf-8"))
    if record.get("status") != "complete":
        fail(f"YOLOE record status is {record.get('status')!r}; only a complete run may be fused")
    if record.get("arm") != "yoloe-26l-seg":
        fail(f"YOLOE record arm is {record.get('arm')!r}; expected 'yoloe-26l-seg'")
    arm_spec = record.get("arm_spec")
    if not isinstance(arm_spec, dict) or arm_spec.get("segmentation") is not True:
        fail("YOLOE record is not the segmentation arm")
    sample_ids = [require_int(value, "yoloe sample image id") for value in record["sample_image_ids"]]
    if sample_ids != ctx["image_ids"]:
        fail("YOLOE sample image IDs differ from the pinned sample manifest")
    mapped_ids = sorted(int(row["category_id"]) for row in record["class_map"])
    if mapped_ids != ctx["gt_category_ids"]:
        fail("YOLOE class map category IDs differ from the instance GT category IDs")

    bbox_rows = record["bbox_results"]
    segm_rows = record["segm_results"]
    if not isinstance(bbox_rows, list) or not isinstance(segm_rows, list):
        fail("YOLOE bbox_results/segm_results must be lists")
    if len(bbox_rows) != len(segm_rows):
        fail(f"YOLOE bbox/segm row counts differ: {len(bbox_rows)} vs {len(segm_rows)}")
    total = len(bbox_rows)

    det_indices_by_image = {image_id: [] for image_id in ctx["image_ids"]}
    for index, (bbox_row, segm_row) in enumerate(zip(bbox_rows, segm_rows)):
        image_id = require_int(bbox_row["image_id"], f"bbox row {index} image_id")
        if image_id not in ctx["image_id_set"]:
            fail(f"bbox row {index}: image_id {image_id} is outside the frozen subset")
        if require_int(segm_row["image_id"], f"segm row {index} image_id") != image_id:
            fail(f"segm row {index} is not aligned with bbox row {index} (image_id)")
        category_id = require_int(bbox_row["category_id"], f"bbox row {index} category_id")
        if category_id not in ctx["gt_category_set"]:
            fail(f"bbox row {index}: category_id {category_id} is not a mapped COCO thing ID")
        if require_int(segm_row["category_id"], f"segm row {index} category_id") != category_id:
            fail(f"segm row {index} is not aligned with bbox row {index} (category_id)")
        score = bbox_row["score"]
        if not is_finite_number(score):
            fail(f"bbox row {index}: score is not a finite number: {score!r}")
        segm_score = segm_row["score"]
        if not is_finite_number(segm_score) or float(segm_score) != float(score):
            fail(f"segm row {index} is not aligned with bbox row {index} (score)")
        bbox = bbox_row["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4 and
                all(is_finite_number(value) for value in bbox)):
            fail(f"bbox row {index}: bbox must be a list of 4 finite numbers: {bbox!r}")
        x, y, width, height = (float(value) for value in bbox)
        if width <= 0 or height <= 0:
            fail(f"bbox row {index}: bbox width/height must be positive")
        area = bbox_row["area"]
        if not is_finite_number(area) or float(area) < 0:
            fail(f"bbox row {index}: bbox area must be a finite non-negative number")
        if bbox_row.get("iscrowd") != 0:
            fail(f"bbox row {index}: iscrowd must be 0 for detection rows")
        rle = segm_row["segmentation"]
        if not (isinstance(rle, dict) and isinstance(rle.get("size"), list)
                and len(rle["size"]) == 2):
            fail(f"segm row {index}: segmentation is not an RLE dict with a size")
        height_px, width_px = (require_int(value, f"segm row {index} RLE size")
                               for value in rle["size"])
        if (height_px, width_px) != (ctx["gt_dims"][image_id][1], ctx["gt_dims"][image_id][0]):
            fail(f"segm row {index} (image {image_id}): RLE canvas "
                 f"{(height_px, width_px)} does not match the annotation canvas "
                 f"{ctx['gt_dims'][image_id][::-1]}; masks are never resized")
        counts = rle.get("counts")
        if not (isinstance(counts, list) and
                all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in counts)):
            fail(f"segm row {index}: RLE counts must be a list of non-negative integers")
        digest = segm_row.get("raster_sha256")
        if digest is not None and not isinstance(digest, str):
            fail(f"segm row {index}: raster_sha256 must be a string when present")
        det_indices_by_image[image_id].append(index)

    ctx["yoloe_record"] = record
    ctx["bbox_rows"] = bbox_rows
    ctx["segm_rows"] = segm_rows
    ctx["det_indices_by_image"] = det_indices_by_image
    log(f"YOLOE record verified: {total} detection rows over 500 images "
        f"(status=complete, arm={record['arm']})")
    return ctx


# ---------------------------------------------------------------------------
# EOVSAM prediction metadata (structure only; PNGs are validated per image)
# ---------------------------------------------------------------------------

def load_and_verify_predictions(paths: dict, ctx: dict) -> dict:
    payload = json.loads(paths["predictions_json"].read_text(encoding="utf-8"))
    ctx["predictions_sha256"] = sha256_file(paths["predictions_json"])
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        fail("EOVSAM predictions.json must contain an 'annotations' list")
    if len(annotations) != 500:
        fail(f"EOVSAM predictions.json must hold exactly 500 annotation rows; got {len(annotations)}")

    # Annotation-row identity is validated unconditionally, also when an
    # 'images' array is present: exactly one annotation row per frozen image
    # ID (length 500, no duplicates, set-equal to the frozen IDs), failing
    # closed before any per-image mapping is constructed.
    annotation_image_ids = [require_int(row["image_id"], f"annotation row {position} image_id")
                            for position, row in enumerate(annotations)]
    annotation_id_set = set(annotation_image_ids)
    if len(annotation_id_set) != 500 or annotation_id_set != ctx["image_id_set"]:
        fail(f"EOVSAM prediction annotations must hold exactly one row per frozen image ID "
             f"(length 500, no duplicates, set-equal to the frozen IDs); got "
             f"{len(annotations)} rows covering {len(annotation_id_set)} unique IDs")

    images = payload.get("images")
    if isinstance(images, list):
        image_ids = [require_int(row["id"], "predictions image id") for row in images]
        if len(image_ids) != 500 or sorted(image_ids) != ctx["image_ids"]:
            fail("EOVSAM predictions.json image rows do not cover exactly the frozen 500 IDs")
        if set(image_ids) != annotation_id_set:
            fail("EOVSAM predictions.json image row IDs do not match the annotation "
                 "image_id set")
        for row in images:
            dims = (int(row["width"]), int(row["height"]))
            if dims != ctx["gt_dims"][int(row["id"])]:
                fail(f"EOVSAM predictions.json image {row['id']} dimensions {dims} "
                     f"disagree with the instance GT")

    file_name_by_image = {}
    file_names_seen = set()
    segments_by_image = {}
    for position, row in enumerate(annotations):
        image_id = require_int(row["image_id"], f"annotation row {position} image_id")
        file_name = row.get("file_name")
        if (not isinstance(file_name, str) or file_name == "" or file_name in (".", "..")
                or os.path.isabs(file_name) or os.sep in file_name or "/" in file_name
                or file_name != os.path.basename(file_name)):
            fail(f"annotation row {position}: file_name is not a single safe basename: {file_name!r}")
        if file_name in file_names_seen:
            fail(f"annotation row {position}: duplicate file_name {file_name!r}")
        file_names_seen.add(file_name)
        file_name_by_image[image_id] = file_name
        segments = row.get("segments_info")
        if not isinstance(segments, list):
            fail(f"annotation row {position} (image {image_id}): segments_info must be a list")
        seen_segment_ids = set()
        for seg_position, segment in enumerate(segments):
            segment_id = segment.get("id")
            if isinstance(segment_id, bool) or not isinstance(segment_id, int) or segment_id <= 0:
                fail(f"image {image_id} segment {seg_position}: id must be a positive integer, "
                     f"got {segment_id!r}")
            if segment_id in seen_segment_ids:
                fail(f"image {image_id}: duplicate segment id {segment_id} in segments_info")
            seen_segment_ids.add(segment_id)
            category_id = segment.get("category_id")
            if isinstance(category_id, bool) or not isinstance(category_id, int):
                fail(f"image {image_id} segment {segment_id}: category_id must be an integer, "
                     f"got {category_id!r}")
            if category_id not in ctx["panoptic_categories"]:
                fail(f"image {image_id} segment {segment_id}: category_id {category_id} does not "
                     "resolve in the frozen panoptic category table")
        segments_by_image[image_id] = segments
    ctx["file_name_by_image"] = file_name_by_image
    ctx["segments_by_image"] = segments_by_image
    log(f"EOVSAM predictions.json verified: 500 annotation rows, 500 unique safe file names, "
        f"all segment category IDs resolvable in the panoptic table")
    return ctx

# ---------------------------------------------------------------------------
# Per-image fusion: baseline mask decode, EOVSAM validation, matching, arms
# ---------------------------------------------------------------------------

def canonical_raster_bytes(mask, height: int, width: int, np) -> bytes:
    """Byte-identical serialization to the accepted scorer's source-raster digest.

    ASCII header ``"<height>x<width>:F\\n"`` plus the Fortran-order binary
    raster bytes (0x00 background, 0x01 foreground).
    """
    array = (np.asarray(mask) != 0).astype(np.uint8)
    return f"{height}x{width}:F\n".encode("ascii") + array.tobytes(order="F")


def decode_yoloe_baseline(segm_row: dict, h: int, w: int, env: dict):
    """Decode one frozen YOLOE mask.

    Returns ``(binary_mask, compressed_rle)``; raises ValueError when the mask
    is malformed or undecodable. The compressed RLE is exactly what the
    accepted scorer persists for loadRes (``mask.frPyObjects``), so the
    baseline arm reproduces the accepted scorer bit-for-bit.
    """
    mask_utils = env["mask_utils"]
    rle = segm_row["segmentation"]
    converted = mask_utils.frPyObjects([rle], h, w)[0]
    decoded = mask_utils.decode(converted)
    if decoded.shape != (h, w):
        raise ValueError(f"decoded RLE shape {decoded.shape} does not match {(h, w)}")
    compressed = pythonify_rle(converted)
    mask = decoded != 0
    digest = segm_row.get("raster_sha256")
    if digest is not None:
        if not _RASTER_DIGEST_RE.fullmatch(digest):
            raise ValueError(f"raster_sha256 is malformed: {digest!r}")
        reconstructed = hashlib.sha256(canonical_raster_bytes(mask, h, w, env["np"])).hexdigest()
        if reconstructed != digest:
            raise ValueError(f"decoded RLE raster digest {reconstructed} does not match the "
                             f"persisted source-raster digest {digest}; the frozen mask is "
                             "undecodable as recorded")
    return mask, compressed


def mask_to_rle(mask, env: dict) -> dict:
    """Compressed RLE (JSON-serializable) for a 2-D binary mask."""
    mask_utils = env["mask_utils"]
    array = env["np"].asfortranarray(env["np"].asarray(mask, dtype=bool).astype(env["np"].uint8))
    rle = mask_utils.encode(array)
    return pythonify_rle(rle)


def bbox_of(mask, np) -> tuple | None:
    """Inclusive (row0, col0, row1, col1) of the foreground; None if empty."""
    rows = np.any(mask, axis=1)
    if not rows.any():
        return None
    cols = np.any(mask, axis=0)
    row0, row1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    col0, col1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return (row0, col0, row1, col1)


def bboxes_overlap(a, b) -> bool:
    """Inclusive overlap of two (row0, col0, row1, col1) boxes.

    Row intervals use indices 0/2 and column intervals use indices 1/3;
    both axes must overlap for the candidate edge to survive pruning.
    """
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def match_image_detections(np, linear_sum_assignment, det_entries, seg_masks):
    """Maximum-total-IoU one-to-one matching for one image/class.

    ``det_entries`` is a list of ``(row_index, mask)`` in original detection
    index order; ``seg_masks`` is a dict ``segment_id -> mask``. Candidate
    edges with predicted-mask IoU < IOU_GATE are removed before the
    assignment; ``linear_sum_assignment`` (deterministic) on the cost matrix
    (rows in detection order, columns in ascending segment ID order) then
    yields the feasible one-to-one matching with maximum total IoU. Returns
    a list of ``(row_index, segment_id, iou)``.
    """
    if not det_entries or not seg_masks:
        return []
    segment_ids = sorted(seg_masks)
    det_masks = [entry[1] for entry in det_entries]
    n_det = len(det_masks)
    n_seg = len(segment_ids)
    det_bboxes = [bbox_of(mask, np) for mask in det_masks]
    seg_bboxes = {sid: bbox_of(seg_masks[sid], np) for sid in segment_ids}
    iou = np.full((n_det, n_seg), -1.0, dtype=np.float64)
    for i in range(n_det):
        det_box = det_bboxes[i]
        if det_box is None:
            continue  # empty baseline mask: no candidate edge, keeps baseline
        candidates = [j for j, sid in enumerate(segment_ids)
                      if seg_bboxes[sid] is not None and bboxes_overlap(det_box, seg_bboxes[sid])]
        if not candidates:
            continue
        det_mask = det_masks[i][None]
        stacked = np.stack([seg_masks[segment_ids[j]] for j in candidates])
        inter = (stacked & det_mask).sum(axis=(1, 2)).astype(np.float64)
        union = (stacked | det_mask).sum(axis=(1, 2)).astype(np.float64)
        valid = union > 0
        for position, j in enumerate(candidates):
            if valid[position]:
                iou[i, j] = inter[position] / union[position]
    cost = np.full((n_det, n_seg), DISALLOWED_COST, dtype=np.float64)
    allowed = iou >= IOU_GATE
    cost[allowed] = -iou[allowed]
    row_index, col_index = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row_index, col_index):
        if allowed[i, j]:
            matches.append((det_entries[i][0], segment_ids[j], float(iou[i, j])))
    return matches


def process_all_images(paths: dict, ctx: dict, env: dict) -> dict:
    """Stream the 500 images: decode, validate, match, and build all three arms.

    Memory is bounded per image (baseline masks and segment masks for one
    image only). Fail-closed counters are accumulated across all images; a
    nonzero counter blocks scoring before any output is written.
    """
    np = env["np"]
    Image = env["Image"]
    linear_sum_assignment = env["linear_sum_assignment"]
    total = len(ctx["bbox_rows"])
    arm_rows = {name: [None] * total for name in ARM_NAMES}
    png_sha256 = {}
    eovsam_image_errors = {}
    invalid_mask_rows = []
    matched_pairs = []  # (iou, category_id, image_id)
    matched_by_category = {cid: 0 for cid in ctx["gt_category_ids"]}
    thing_segment_counts = {cid: 0 for cid in ctx["gt_category_ids"]}
    stuff_segment_count = 0

    for image_id in ctx["image_ids"]:
        width, height = ctx["gt_dims"][image_id]
        rows = ctx["det_indices_by_image"][image_id]

        # --- baseline YOLOE masks for this image --------------------------
        baseline_masks = {}
        baseline_rles = {}
        for row in rows:
            segm_row = ctx["segm_rows"][row]
            try:
                mask, compressed = decode_yoloe_baseline(segm_row, height, width, env)
            except Exception as exc:  # fail closed on any malformed/undecodable mask
                invalid_mask_rows.append((row, str(exc)))
                mask, compressed = None, None
            if mask is not None:
                baseline_masks[row] = mask
                baseline_rles[row] = compressed
            else:
                # Frozen row preserved verbatim (uncompressed RLE) in the arm
                # rows; scoring is blocked by the nonzero failure counter.
                rle = segm_row["segmentation"]
                baseline_rles[row] = {"size": [height, width],
                                      "counts": [int(value) for value in rle["counts"]]}

        # --- EOVSAM retained PNG for this image ---------------------------
        file_name = ctx["file_name_by_image"][image_id]
        png_path = paths["png_dir"] / file_name
        image_errors = []
        id_array = None
        if not png_path.is_file():
            image_errors.append(f"missing PNG {png_path}")
        else:
            try:
                png_bytes = png_path.read_bytes()
                png_sha256[file_name] = hashlib.sha256(png_bytes).hexdigest()
                image = Image.open(io.BytesIO(png_bytes))
                if image.format != "PNG":
                    image_errors.append(f"not a PNG (format={image.format!r})")
                if image.mode != "RGB":
                    image_errors.append(f"not an RGB image (mode={image.mode!r})")
                if tuple(image.size) != (width, height):
                    image_errors.append(f"wrong size {tuple(image.size)}; expected {(width, height)}")
                if not image_errors:
                    array = np.asarray(image)
                    rgb = array.astype(np.uint32, copy=False)
                    id_array = (rgb[..., 0]
                                + (rgb[..., 1] << 8)
                                + (rgb[..., 2] << 16))
            except Exception as exc:
                image_errors.append(f"unreadable PNG: {exc}")
        segments = ctx["segments_by_image"][image_id]
        if image_errors:
            # Every nonempty per-image PNG validation error is stored before
            # the branch that skips matching, so image-level failures can
            # never escape the eovsam_image_output_errors counter.
            eovsam_image_errors[image_id] = image_errors
        if not image_errors and id_array is not None:
            unique_ids, counts = np.unique(id_array, return_counts=True)
            present = {int(sid): int(count) for sid, count in zip(unique_ids, counts) if sid != 0}
            for segment in segments:
                segment_id = int(segment["id"])
                if segment_id not in present:
                    image_errors.append(f"segment {segment_id} has no pixels in the PNG "
                                        "(metadata/mask inconsistency)")
            metadata_ids = {int(segment["id"]) for segment in segments}
            nonzero_missing = sorted(sid for sid in present if sid not in metadata_ids)
            if nonzero_missing:
                image_errors.append(f"PNG segment IDs without a metadata row: {nonzero_missing[:10]}")
            if image_errors:
                eovsam_image_errors[image_id] = image_errors
            else:
                # Per-segment binary masks for this image.
                segment_masks = {int(segment["id"]): (id_array == int(segment["id"]))
                                 for segment in segments}
                for segment in segments:
                    category_id = int(segment["category_id"])
                    if category_id in ctx["panoptic_thing_set"]:
                        thing_segment_counts[category_id] += 1
                    else:
                        stuff_segment_count += 1

                # --- same-class thing matching -----------------------------
                image_matches = {}  # row_index -> (segment_id, iou)
                rows_by_category = {}
                for row in rows:
                    if row not in baseline_masks:
                        continue
                    category_id = int(ctx["bbox_rows"][row]["category_id"])
                    rows_by_category.setdefault(category_id, []).append(row)
                for category_id in sorted(rows_by_category):
                    if category_id not in ctx["panoptic_thing_set"]:
                        continue
                    class_segments = [segment for segment in segments
                                      if int(segment["category_id"]) == category_id]
                    if not class_segments:
                        continue
                    det_entries = [(row, baseline_masks[row])
                                   for row in rows_by_category[category_id]]
                    seg_masks = {int(segment["id"]): segment_masks[int(segment["id"])]
                                 for segment in class_segments}
                    for row_index, segment_id, pair_iou in match_image_detections(
                            np, linear_sum_assignment, det_entries, seg_masks):
                        if row_index in image_matches:
                            fail(f"internal error: detection row {row_index} matched twice "
                                 f"for image {image_id}")
                        image_matches[row_index] = (segment_id, pair_iou)
                        matched_pairs.append((pair_iou, category_id, image_id))
                        matched_by_category[category_id] += 1

                # --- arm rows for this image (original row order) ----------
                for row in rows:
                    bbox_row = ctx["bbox_rows"][row]
                    base_rle = baseline_rles[row]
                    replace_rle = base_rle
                    union_rle = base_rle
                    match = image_matches.get(row)
                    if match is not None:
                        segment_mask = segment_masks[match[0]]
                        replace_rle = mask_to_rle(segment_mask, env)
                        union_rle = mask_to_rle(baseline_masks[row] | segment_mask, env)
                    for name in ARM_NAMES:
                        arm_rows[name][row] = {
                            "image_id": int(bbox_row["image_id"]),
                            "category_id": int(bbox_row["category_id"]),
                            "bbox": [float(value) for value in bbox_row["bbox"]],
                            "score": float(bbox_row["score"]),
                            "segmentation": (replace_rle if name == "replace"
                                             else union_rle if name == "union"
                                             else base_rle),
                        }
    stats = {
        "arm_rows": arm_rows,
        "png_sha256": png_sha256,
        "eovsam_image_errors": eovsam_image_errors,
        "invalid_mask_rows": invalid_mask_rows,
        "matched_pairs": matched_pairs,
        "matched_by_category": matched_by_category,
        "thing_segment_counts": thing_segment_counts,
        "stuff_segment_count": stuff_segment_count,
    }
    return stats

# ---------------------------------------------------------------------------
# COCO scoring (exact mirror of the accepted yoloe_coco_score.py rules)
# ---------------------------------------------------------------------------

def write_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def build_gt_subset(ctx: dict, staging_dir: Path) -> dict:
    """Exact subset GT: all original annotation rows for the frozen image IDs."""
    gt = ctx["instance_gt"]
    selected = ctx["image_id_set"]
    images = [row for row in gt["images"] if int(row["id"]) in selected]
    if len(images) != len(ctx["image_ids"]):
        fail("GT subset lost images")
    annotations = [row for row in ctx["gt_annotation_rows"]]
    categories = list(gt["categories"])
    subset = {"images": images, "annotations": annotations, "categories": categories}
    path = staging_dir / "gt_subset_instances.json"
    write_json(path, subset)
    return {
        "image_count": len(images),
        "annotation_count": len(annotations),
        "category_count": len(categories),
        "gt_subset_file": str(path),
    }


def empty_coco_dt(coco_gt):
    """Valid empty COCO detections object (mirrors the accepted scorer)."""
    dt = type(coco_gt)()
    dt.dataset["info"] = dict(coco_gt.dataset.get("info", {}))
    dt.dataset["images"] = [img for img in coco_gt.dataset["images"]]
    dt.dataset["categories"] = dict(coco_gt.dataset.get("categories", {}))
    dt.dataset["annotations"] = []
    dt.createIndex()
    return dt


def run_cocoeval(coco_gt, dt, iou_type: str, ctx: dict, env: dict) -> dict:
    """One COCOeval (mask or box) on exactly the frozen 500 image IDs."""
    COCOeval = env["COCOeval"]
    evaluator = COCOeval(coco_gt, dt, iou_type)
    evaluator.params.imgIds = list(ctx["image_ids"])
    evaluator.params.catIds = list(ctx["gt_category_ids"])
    evaluator.params.maxDets = list(MAX_DETS)
    buffer = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(buffer):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = [float(value) for value in evaluator.stats]
    metrics = {name: stats[index] for index, name in enumerate(STAT_NAMES)}
    return {
        "detection_count": len(dt.dataset.get("annotations", [])),
        "AP": metrics["AP"],
        "AP50": metrics["AP50"],
        "AP75": metrics["AP75"],
        "APs": metrics["APs"],
        "APm": metrics["APm"],
        "APl": metrics["APl"],
        "AR100": metrics["AR100"],
        "stats": stats,
        "maxDets": list(MAX_DETS),
        "evaluator_params_imgIds": [int(value) for value in evaluator.params.imgIds],
        "evaluator_params_catIds": [int(value) for value in evaluator.params.catIds],
        "evaluator_params_maxDets": [int(value) for value in evaluator.params.maxDets],
        "evaluator_stdout": buffer.getvalue(),
        "seconds": round(time.perf_counter() - started, 3),
        "note": "AP/AP50/AP75 use the standard top-100 cutoff (maxDets [1,10,100]); "
                "a value of -1.0 means pycocotools found no detections for that slice",
    }


def score_all_arms(ctx: dict, stats: dict, paths: dict, env: dict, staging_dir: Path) -> dict:
    """Score mask and box AP for all three arms on exactly the frozen IDs."""
    COCO = env["COCO"]
    gt_info = build_gt_subset(ctx, staging_dir)
    coco_gt = COCO(gt_info["gt_subset_file"])
    bbox_rows = []
    for row in ctx["bbox_rows"]:
        bbox_rows.append({
            "image_id": int(row["image_id"]),
            "category_id": int(row["category_id"]),
            "bbox": [float(value) for value in row["bbox"]],
            "score": float(row["score"]),
            "area": float(row["area"]),
            "iscrowd": 0,
        })
    write_json(staging_dir / "bbox_loadres_input.json", bbox_rows)

    arm_results = {}
    for name in ARM_NAMES:
        rows = stats["arm_rows"][name]
        if any(value is None for value in rows):
            fail(f"internal error: arm {name} has missing rows")
        segm_rows = [{
            "image_id": int(row["image_id"]),
            "category_id": int(row["category_id"]),
            "segmentation": row["segmentation"],
            "score": float(row["score"]),
        } for row in rows]
        write_json(staging_dir / f"{name}_loadres_input.json", segm_rows)
        log(f"scoring arm {name}: mask AP (segm) ...")
        dt_segm = coco_gt.loadRes(segm_rows) if segm_rows else empty_coco_dt(coco_gt)
        mask_metrics = run_cocoeval(coco_gt, dt_segm, "segm", ctx, env)
        log(f"scoring arm {name}: box AP (bbox) ...")
        dt_bbox = coco_gt.loadRes(bbox_rows) if bbox_rows else empty_coco_dt(coco_gt)
        box_metrics = run_cocoeval(coco_gt, dt_bbox, "bbox", ctx, env)
        arm_results[name] = {"segm": mask_metrics, "bbox": box_metrics}
        log(f"arm {name}: mask AP={mask_metrics['AP']:.6f} AP50={mask_metrics['AP50']:.6f} "
            f"AP75={mask_metrics['AP75']:.6f} APs/APm/APl="
            f"{mask_metrics['APs']:.6f}/{mask_metrics['APm']:.6f}/{mask_metrics['APl']:.6f}; "
            f"box AP={box_metrics['AP']:.6f}")

    # Baseline reproduction gate: must pass before any final materialization.
    baseline = arm_results["baseline"]
    mask_delta = abs(baseline["segm"]["AP"] - EXPECTED_MASK_AP)
    box_delta = abs(baseline["bbox"]["AP"] - EXPECTED_BOX_AP)
    log(f"baseline gate: mask AP {baseline['segm']['AP']:.9f} vs expected {EXPECTED_MASK_AP} "
        f"(delta {mask_delta:.3e}); box AP {baseline['bbox']['AP']:.9f} vs expected "
        f"{EXPECTED_BOX_AP} (delta {box_delta:.3e}); tolerance {BASELINE_TOLERANCE}")
    if mask_delta > BASELINE_TOLERANCE or box_delta > BASELINE_TOLERANCE:
        fail(f"baseline reproduction failed beyond {BASELINE_TOLERANCE} tolerance; aborting "
             "without writing arm result files or metrics (mask delta "
             f"{mask_delta:.3e}, box delta {box_delta:.3e})")

    # Provenance check: the box rows are identical across arms, so the box AP
    # must be bit-identical across all three COCOeval runs.
    box_aps = [arm_results[name]["bbox"]["AP"] for name in ARM_NAMES]
    if len({value for value in box_aps}) != 1:
        fail(f"box AP differs across arms on identical box rows: {box_aps}; aborting")

    return {
        "arms": arm_results,
        "baseline_mask_ap": baseline["segm"]["AP"],
        "baseline_box_ap": baseline["bbox"]["AP"],
        "baseline_mask_delta": mask_delta,
        "baseline_box_delta": box_delta,
        "gt_info": gt_info,
    }


# ---------------------------------------------------------------------------
# Metrics, staging, and atomic promotion
# ---------------------------------------------------------------------------

def build_metrics(ctx: dict, stats: dict, scoring: dict, paths: dict, env: dict,
                  started: float, stage_seconds: dict, staging_dir: Path) -> dict:
    matched_pairs = stats["matched_pairs"]
    total_detections = len(ctx["bbox_rows"])
    matched_count = len(matched_pairs)
    invalid_mask_row_set = {row for row, _reason in stats["invalid_mask_rows"]}
    valid_detection_rows_by_category = {cid: 0 for cid in ctx["gt_category_ids"]}
    for index, bbox_row in enumerate(ctx["bbox_rows"]):
        if index in invalid_mask_row_set:
            continue
        category_id = int(bbox_row["category_id"])
        valid_detection_rows_by_category[category_id] += 1
    per_category = []
    for category_id in ctx["gt_category_ids"]:
        name = next(row["name"] for row in ctx["instance_gt"]["categories"]
                    if int(row["id"]) == category_id)
        per_category.append({
            "category_id": category_id,
            "name": name,
            "matched_pairs": stats["matched_by_category"][category_id],
            "unmatched_detections": (valid_detection_rows_by_category[category_id]
                                     - stats["matched_by_category"][category_id]),
            "eovsam_thing_segments": stats["thing_segment_counts"][category_id],
            "unmatched_thing_segments": (stats["thing_segment_counts"][category_id]
                                         - stats["matched_by_category"][category_id]),
        })
    existing_scorer = None
    if EXISTING_SCORER_METRICS.is_file():
        existing_scorer = {
            "path": str(EXISTING_SCORER_METRICS),
            "sha256": sha256_file(EXISTING_SCORER_METRICS),
        }
        try:
            payload = json.loads(EXISTING_SCORER_METRICS.read_text(encoding="utf-8"))
            existing_scorer["reference_mask_ap"] = payload["segm"]["AP"]
            existing_scorer["reference_box_ap"] = payload["bbox"]["AP"]
        except (OSError, ValueError, KeyError):
            pass
    self_hash = sha256_file(Path(__file__).resolve())
    fork_hash = sha256_file(FORK_EVALUATOR) if FORK_EVALUATOR.is_file() else None
    return {
        "schema": "eovsam-mask-fusion/v7",
        "contract": "eovsam-fusion-contract-v7",
        "scope": "eovsam-fusion-implementation-scope-v7",
        "started_utc": stage_seconds["started_utc"],
        # The metrics are built before the final output promotion: this file
        # therefore carries a clearly named pre-promotion total and the
        # timestamp at which the metrics were written, and no finished or
        # complete-run total.
        "metrics_written_utc": utc_now(),
        "pre_promotion_seconds": round(time.perf_counter() - started, 3),
        "timing_scope": ("All timing fields in this file (started_utc, metrics_written_utc, "
                         "pre_promotion_seconds, stage_seconds) are measured before the final "
                         "output promotion and exclude the final output copy/rename into the "
                         "frozen fusion/ destination; the promotion duration is unavailable in "
                         "this file and is recorded in the scoring log after promotion "
                         "completes."),
        "stage_seconds": {
            "input_verification": stage_seconds["input_verification"],
            "fusion_build": stage_seconds["fusion_build"],
            "scoring": stage_seconds["scoring"],
            # Measured only after the metrics are written; unavailable here.
            "promotion_seconds": None,
        },
        "scoring_environment": {
            "sys_executable": env["sys_executable"],
            "user": env["user"],
            "python_version": env["python_version"],
            "pycocotools": {
                "version": env["pycocotools_version"],
                "module_path": env["pycocotools_module_path"],
            },
            "scipy_version": env["scipy_version"],
            "numpy_version": env["numpy_version"],
            "pil_version": env["pil_version"],
            "platform": platform.platform(),
        },
        "inputs": {
            "yoloe_json": {"path": str(paths["yoloe_json"]), "sha256": ctx["hashes"]["yoloe_json"]},
            "instance_gt": {"path": str(paths["instance_gt"]), "sha256": ctx["hashes"]["instance_gt"]},
            "panoptic_gt": {"path": str(paths["panoptic_gt"]), "sha256": ctx["hashes"]["panoptic_gt"]},
            "sample_manifest": {"path": str(paths["sample_manifest"]),
                                "sha256": ctx["hashes"]["sample_manifest"]},
            "inference_artifact_manifest": {
                "path": str(paths["inference_manifest"]),
                "sha256": ctx["hashes"]["inference_manifest"],
            },
            "run_manifest": {"path": str(paths["run_manifest"]), "sha256": ctx["hashes"]["run_manifest"]},
            "predictions_json": {"path": str(paths["predictions_json"]),
                                 "sha256": ctx["predictions_sha256"]},
            "panoptic_predictions_dir": str(paths["png_dir"]),
            "panoptic_png_sha256": stats["png_sha256"],
            "existing_scorer_metrics": existing_scorer,
            "fork_evaluator": {"path": str(FORK_EVALUATOR), "sha256": fork_hash},
            "fuse_masks": {"path": str(Path(__file__).resolve()), "sha256": self_hash},
        },
        "subset": {
            "label": "COCO val2017 fixed 500-image subset",
            "seed": 42,
            "image_count": len(ctx["image_ids"]),
            "image_ids": ctx["image_ids"],
        },
        "ground_truth": scoring["gt_info"],
        "matching": {
            "rule": "same-class (COCO thing category ID) predicted-mask IoU >= 0.50 edges, "
                    "then maximum-total-IoU one-to-one assignment (scipy.optimize."
                    "linear_sum_assignment, rows in original detection index order, columns "
                    "in ascending segment ID order); unmatched detections keep baseline masks; "
                    "unmatched EOVSAM segments are never added as proposals",
            "iou_gate": IOU_GATE,
            "matched_pairs": matched_count,
            "unmatched_yoloe_detections": total_detections - matched_count,
            "eovsam_thing_segments_total": sum(stats["thing_segment_counts"].values()),
            "unmatched_eovsam_thing_segments": (sum(stats["thing_segment_counts"].values())
                                                - matched_count),
            "eovsam_stuff_segments_total": stats["stuff_segment_count"],
            "average_matched_predicted_mask_iou": (
                round(sum(pair[0] for pair in matched_pairs) / matched_count, 6)
                if matched_pairs else None),
            "per_category": per_category,
        },
        "failure_counters": {
            "yoloe_detections_without_valid_mask": len(stats["invalid_mask_rows"]),
            "eovsam_image_output_errors": len(stats["eovsam_image_errors"]),
            "yoloe_invalid_mask_detail": stats["invalid_mask_rows"][:20],
            "eovsam_image_error_detail": {str(image_id): reasons
                                          for image_id, reasons in
                                          sorted(stats["eovsam_image_errors"].items())[:20]},
        },
        "baseline_reproduction": {
            "expected_mask_ap": EXPECTED_MASK_AP,
            "expected_box_ap": EXPECTED_BOX_AP,
            "tolerance": BASELINE_TOLERANCE,
            "baseline_mask_ap": scoring["baseline_mask_ap"],
            "baseline_box_ap": scoring["baseline_box_ap"],
            "mask_ap_delta": scoring["baseline_mask_delta"],
            "box_ap_delta": scoring["baseline_box_delta"],
        },
        "arms": {
            name: {
                "detection_count": scoring["arms"][name]["segm"]["detection_count"],
                "mask": scoring["arms"][name]["segm"],
                "box": scoring["arms"][name]["bbox"],
            } for name in ARM_NAMES
        },
        "paths": {
            "run_root": str(paths["run_root"]),
            "predictions_json": str(paths["predictions_json"]),
            "panoptic_predictions_dir": str(paths["png_dir"]),
            "fusion_dir": str(paths["fusion_dir"]),
            "fusion_files": {name: str(paths["fusion_dir"] / name) for name in PINNED_FUSION_FILES},
            "staging_dir": str(staging_dir),
            "scoring_log": str(paths["log_file"]),
        },
        "interpretation": [
            "Scores are for the fixed 500-image COCO val2017 subset, not full val2017.",
            "All three arms retain identical image_id/category_id/bbox/score rows per original "
            "YOLOE detection; only the segmentation differs (baseline vs matched EOVSAM "
            "replacement vs union).",
            "This is a detector mask-fusion experiment, not map acceptance, panoptic-model "
            "replacement, or an end-to-end association score.",
            "A metric value of -1.0 means pycocotools found no detections for that slice.",
        ],
    }


def promote_outputs(paths: dict, staging_dir: Path, stage_files: dict) -> dict:
    """Atomically materialize the final files with one same-filesystem rename.

    Model and scoring work stays staged in the system temporary directory
    (``staging_dir``). For promotion, the final files are copied into a
    unique sibling staging directory on the same filesystem as ``run_root``
    (``run_root.parent``). The absent score-run parent is created only after
    all scores have passed; the completed directory is then renamed to the
    absent ``fusion/`` destination in one atomic directory rename. If the
    rename fails, remove the new empty parent and staged directory.
    """
    fusion_dir = paths["fusion_dir"]
    if fusion_dir.exists() or any((fusion_dir / name).exists() for name in PINNED_FUSION_FILES):
        fail("frozen fusion destination is no longer absent; refusing to write")
    run_root = paths["run_root"]
    if not run_root.is_dir():
        fail(f"run root is not a directory; cannot stage final files on its "
             f"filesystem: {run_root}")
    staging_parent = run_root.parent
    if not staging_parent.is_dir():
        fail(f"run root parent is not a directory; cannot stage final files: "
             f"{staging_parent}")
    fusion_parent = fusion_dir.parent
    if fusion_parent.exists():
        fail(f"fresh score-run parent already exists; refusing to merge or overwrite: "
             f"{fusion_parent}")
    final_stage = Path(tempfile.mkdtemp(prefix=f"{run_root.name}.fusion-staging-",
                                        dir=staging_parent))
    renamed = False
    parent_created = False
    try:
        for name in PINNED_FUSION_FILES:
            shutil.copyfile(stage_files[name], final_stage / name)
        fusion_parent.mkdir()
        parent_created = True
        os.replace(final_stage, fusion_dir)
        renamed = True
    except Exception as exc:
        if not renamed:
            shutil.rmtree(final_stage, ignore_errors=True)
            if parent_created:
                try:
                    fusion_parent.rmdir()
                except OSError:
                    pass
        fail(f"final output promotion failed ({exc}); stopping without creating "
             "fusion/ or any final files")
    final_hashes = {}
    for name in PINNED_FUSION_FILES:
        digest = sha256_file(fusion_dir / name)
        if digest != sha256_file(stage_files[name]):
            fail(f"promoted file {fusion_dir / name} hash mismatch; aborting")
        final_hashes[name] = digest
    return {"promoted_as": "atomic directory rename", "final_hashes": final_hashes}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    started = time.perf_counter()
    stage_started_utc = utc_now()
    paths = parse_args()
    env = preflight_environment()
    global _LOG
    _LOG = FusionLog(paths["log_file"])
    stage_seconds = {
        "started_utc": stage_started_utc,
        "input_verification": 0.0,
        "fusion_build": 0.0,
        "scoring": 0.0,
        # Measured only after the final promotion; the metrics JSON is built
        # before that, so no placeholder value for the promotion duration is
        # ever emitted into it (it is recorded in the scoring log instead).
        "promotion": None,
    }
    log(f"EOVSAM mask-fusion scorer starting (contract v7); started {stage_started_utc}")
    log(f"environment: {env['sys_executable']} as {env['user']} python "
        f"{env['python_version']} pycocotools {env['pycocotools_version']} "
        f"scipy {env['scipy_version']} numpy {env['numpy_version']}")

    try:
        t0 = time.perf_counter()
        ctx = verify_pinned_inputs(paths, t0)
        stage_seconds["input_verification"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        load_and_verify_yoloe(paths, ctx)
        load_and_verify_predictions(paths, ctx)
        t0 = time.perf_counter()
        stats = process_all_images(paths, ctx, env)
        stage_seconds["fusion_build"] = round(time.perf_counter() - t0, 3)
        log(f"fusion build: {len(stats['matched_pairs'])} matched pairs, "
            f"{len(ctx['bbox_rows']) - len(stats['matched_pairs'])} unmatched detections, "
            f"yoloe_detections_without_valid_mask={len(stats['invalid_mask_rows'])}, "
            f"eovsam_image_output_errors={len(stats['eovsam_image_errors'])}")

        # A scored result requires both failure counters to be zero.
        if stats["invalid_mask_rows"]:
            fail(f"yoloe_detections_without_valid_mask={len(stats['invalid_mask_rows'])} "
                 "(first: "
                 f"{[(row, reason) for row, reason in stats['invalid_mask_rows'][:5]]}); "
                 "input failure blocks scoring; no fusion outputs written")
        if stats["eovsam_image_errors"]:
            fail(f"eovsam_image_output_errors={len(stats['eovsam_image_errors'])} "
                 "(first: "
                 f"{dict(list(sorted(stats['eovsam_image_errors'].items()))[:5])}); "
                 "contract failure blocks scoring; no fusion outputs written")

        staging_dir = Path(tempfile.mkdtemp(prefix="eovsam-fusion-"))
        try:
            t0 = time.perf_counter()
            scoring = score_all_arms(ctx, stats, paths, env, staging_dir)
            stage_seconds["scoring"] = round(time.perf_counter() - t0, 3)
            metrics = build_metrics(ctx, stats, scoring, paths, env, started, stage_seconds,
                                    staging_dir)
            stage_files = {}
            for name in ARM_NAMES:
                path = staging_dir / f"{name}_instances.json"
                write_json(path, stats["arm_rows"][name])
                stage_files[f"{name}_instances.json"] = path
            metrics_path = staging_dir / "metrics.json"
            write_json(metrics_path, metrics)
            stage_files["metrics.json"] = metrics_path

            t0 = time.perf_counter()
            promotion = promote_outputs(paths, staging_dir, stage_files)
            stage_seconds["promotion"] = round(time.perf_counter() - t0, 3)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

        log(f"promotion complete ({promotion['promoted_as']}) in "
            f"{stage_seconds['promotion']:.1f}s; final files under {paths['fusion_dir']}:")
        for name in PINNED_FUSION_FILES:
            log(f"  {paths['fusion_dir'] / name} sha256={promotion['final_hashes'][name]}")
        baseline = scoring["arms"]["baseline"]
        replace = scoring["arms"]["replace"]
        union = scoring["arms"]["union"]
        log(f"RESULT baseline mask AP={baseline['segm']['AP']:.6f} | "
            f"replace mask AP={replace['segm']['AP']:.6f} | "
            f"union mask AP={union['segm']['AP']:.6f} (box AP "
            f"{baseline['bbox']['AP']:.6f} unchanged in all arms)")
        total_seconds = stage_seconds["total_seconds"] if "total_seconds" in stage_seconds \
            else round(time.perf_counter() - started, 3)
        log(f"SUCCESS: all arms scored on the frozen 500 image IDs; total {total_seconds}s")
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"unexpected error; failing closed without final fusion outputs: {exc}")
    finally:
        _LOG.close()
        _LOG = None


if __name__ == "__main__":
    raise SystemExit(main())
