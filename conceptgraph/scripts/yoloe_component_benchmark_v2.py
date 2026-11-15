"""Run exactly one YOLOE-26 arm on the fixed COCO val2017 subset on host CUDA.

Accepted contract: ``yoloe-implementation-contract.md`` (plan v5,
``yoloe-detect-only-plan-v5.md``, adjudication
``yoloe-plan-v5-independent-review.md``).

This script performs real CUDA inference only. It never imports pycocotools
(the repository ``.venv`` does not ship it) and never scores. COCO scoring is
a separate CPU step: ``yoloe_coco_score.py`` under ``/usr/bin/python3`` as
user ``nils`` with pycocotools 2.0.11.

Arms (exactly one per invocation):
  yoloe-26l-seg          official models/yoloe-26l-seg.pt, boxes + raster masks
  yoloe-26m-seg          official models/yoloe-26m-seg.pt, boxes + raster masks
  yoloe-26l-detect-only  YOLOE('yoloe-26l.yaml').load('models/yoloe-26l-seg.pt')
                         plus the saved prompt profile generated from the
                         L segmentation checkpoint, boxes only
  yoloe-26m-detect-only  same construction with yoloe-26m.yaml and
                         models/yoloe-26m-seg.pt, boxes only

The two detection-only arms are derived from the matching official
segmentation checkpoints through the documented YOLOE construction route
(same-scale YAML, checkpoint transfer, saved NPZ prompt profile). They are
not official detect-only release checkpoints; read their box AP with that
disclosure. The scale-suffixed YAML names are logical names only: the
installed ultralytics package ships the unified
``cfg/models/26/yoloe-26.yaml`` whose ``scales`` table declares the ``l``
and ``m`` compound constants, and its loader resolves the logical
``yoloe-26l.yaml`` to that unified file while inferring ``l`` from the
name. Preflight reproduces that exact resolution and fails closed when the
unified YAML is absent, when the arm scale is not declared, or when the
construction would not preserve the requested scale.

Fixed protocol: sorted annotation image IDs whose files exist,
``random.Random(42).sample(..., 500)``, sorted and persisted. 80 categories
sorted by actual ``category_id``; names hyphenated into prompts; output class
index ``i`` maps to ``categories[i].id`` (never ``i + 1``; category ID 68 is
absent). ``imgsz=640``, ``conf=0.001``, ``iou=0.7``, ``max_det=300``,
batch one, ``cuda:0``, ``retina_masks=True`` only for segmentation arms. One
unscored warmup on ``sample[0]`` (persisted by image ID, excluded from
predictions, scoring and latency summaries). Per-image time begins before
the image path load and ends after the synchronized CPU extraction of boxes
and, for segmentation arms, raster masks and their RLE rows; each
segmentation row also persists a SHA-256 digest of the source binary
mask raster (canonical serialization of canvas dimensions + Fortran-
order declaration + exact binary bytes, see ``mask_raster_digest``)
which the scorer re-verifies against the decoded compressed RLE. No
thresholds are tuned and there is no CPU fallback.

Artifacts (default ``<script dir>/yoloe26/``):
  <output>.json                     self-contained output; the status field is
                                    "complete" only when all images finished
  <output>.rows.jsonl               durable per-image rows, retained on
                                    failure so interrupted progress survives
  <output>.status.json              live progress sidecar, retained on failure
  <output>.failure.json             traceback and stage, failure only
  <output>.tmp                      atomic-write temp file, consumed by
                                    os.replace
  <output-stem>_prompt_profile.npz  detector arms only; the NPZ profile
                                    filename is derived from the output JSON
                                    stem so each run gets its own unique
                                    per-run profile artifact

The run refuses to start when ANY artifact this run would write already
exists (the output JSON, its .rows.jsonl / .status.json / .failure.json /
.tmp sidecars, and the per-run prompt profile NPZ for detector arms):
prior-run artifacts (complete, partial, or failed) are preserved and
never unlinked, overwritten, or truncated, so an interrupted rerun can
never delete the only durable partial rows or leave a stale "complete"
artifact. Every run must use a unique --output path. A fresh running
status sidecar is persisted before any model work. An interrupted or
failed run never leaves a "complete" output: the JSON carries status
"partial"/"failed" and the scorer refuses to score it.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import io
import json
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Pin the no-download policy and make the repository importable before the
# project imports below. The system Python (/usr/bin/python3) is the pinned
# *scorer* interpreter and must never run this GPU script.
os.environ["YOLO_AUTOINSTALL"] = "false"

REPO_ROOT = Path("/mnt/HDD1/Project_Code/concept-graphs")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import ultralytics
from ultralytics import YOLOE
# The derived-arm preflight reproduces the installed package's own YAML
# resolution: yaml_model_load is the loader YOLOE(...) applies to the
# scale-suffixed logical name, and check_yaml is the file search it runs.
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils.checks import check_yaml
from conceptgraph.utils.gpu_memory_budget import NvmlMemoryBudget

DEFAULT_COCO_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/coco")
DEFAULT_ANNOTATIONS = DEFAULT_COCO_ROOT / "annotations" / "instances_val2017.json"
DEFAULT_IMAGE_ROOT = DEFAULT_COCO_ROOT / "val2017"
PROMPT_ENCODER_NAME = "mobileclip2_b.ts"
DEVICE = "cuda:0"
MIN_CHECKPOINT_BYTES = 50 * 1024**2
MIN_ENCODER_BYTES = 100 * 1024**2
NVML_SAMPLING_INTERVAL_SECONDS = 1.0

# Pinned scoring environment, recorded for the report; yoloe_coco_score.py
# validates every value at runtime before it scores anything.
PINNED_SCORER = {
    "interpreter": "/usr/bin/python3",
    "user": "nils",
    "python_version": "3.10.12",
    "pycocotools_version": "2.0.11",
    "module_path": "/home/nils/.local/lib/python3.10/site-packages/pycocotools/__init__.py",
}

_TRANSFER_COUNT_RE = re.compile(r"(\d+)\s+of\s+(\d+)\s+layers", re.IGNORECASE)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    scale: str
    checkpoint: str  # relative to REPO_ROOT
    yaml_config: str | None  # detection-only YAML for derived detector arms
    segmentation: bool  # True -> boxes + raster masks, retina_masks=True


ARMS: dict[str, ArmSpec] = {
    "yoloe-26l-seg": ArmSpec("yoloe-26l-seg", "l", "models/yoloe-26l-seg.pt", None, True),
    "yoloe-26m-seg": ArmSpec("yoloe-26m-seg", "m", "models/yoloe-26m-seg.pt", None, True),
    "yoloe-26l-detect-only": ArmSpec("yoloe-26l-detect-only", "l", "models/yoloe-26l-seg.pt", "yoloe-26l.yaml", False),
    "yoloe-26m-detect-only": ArmSpec("yoloe-26m-detect-only", "m", "models/yoloe-26m-seg.pt", "yoloe-26m.yaml", False),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-arm YOLOE-26 COCO benchmark inference (host CUDA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", nargs="+", choices=tuple(ARMS), required=True,
                        help="exactly one YOLOE-26 arm per invocation")
    parser.add_argument("--output", type=Path, default=None,
                        help="self-contained JSON output (default <script dir>/yoloe26/<arm>_predictions.json)")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--max-images", type=int, default=500,
                        help="deterministic sample size; -1 runs every present image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    if len(args.model) != 1:
        parser.error("exactly one --model choice is required per invocation")
    if args.max_images != -1 and args.max_images <= 0:
        parser.error("--max-images must be positive or -1 for every image")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.image_size <= 0:
        parser.error("--image-size must be positive")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if not 0.0 < args.iou < 1.0:
        parser.error("--iou must be in (0, 1)")
    if args.max_det <= 0:
        parser.error("--max-det must be positive")
    return args


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload) -> None:
    """Atomically write JSON so a crash never leaves a torn output file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def git_repo_head(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def gpu_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        lines = [line for line in completed.stdout.strip().splitlines() if line]
        return lines[0] if lines else None
    except Exception:
        return None


def append_rows_durable(rows_jsonl: Path, rows: list[tuple[str, dict]]) -> None:
    """Append one JSON line per row, fsynced, so SIGKILL still keeps progress."""
    if not rows:
        return
    with open(rows_jsonl, "a", encoding="utf-8") as handle:
        for kind, row in rows:
            handle.write(json.dumps({"type": kind, "row": row}, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def capture(callable):
    """Run ``callable()`` with stdout/stderr/warnings captured.

    Returns ``(result, seconds, combined_log, warning_lines)``. Best effort:
    loggers that hold a pre-redirect stream reference may still print.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = callable()
    seconds = time.perf_counter() - started
    log = (stdout.getvalue() + stderr.getvalue()).strip()
    warning_lines = [f"{item.category.__name__}: {item.message}" for item in caught]
    return result, seconds, log, warning_lines


def parse_transfer_layer_counts(log: str) -> dict | None:
    """Extract matched/total tensor counts from YOLOE.load() output, if exposed."""
    match = _TRANSFER_COUNT_RE.search(log)
    if match is None:
        return None
    return {"matched": int(match.group(1)), "total": int(match.group(2))}


class NvmlWindowSampler:
    """Whole-device NVML memory sampling on a fixed 1-second interval.

    Complements ``NvmlMemoryBudget`` (which enforces the 28 GiB ceiling on
    demand): this sampler records the baseline and peak of the measured
    window for the resource report. A sampling hiccup never kills the run;
    the budget check remains the guard.
    """

    def __init__(self, device_index: int, interval_seconds: float = NVML_SAMPLING_INTERVAL_SECONDS):
        import pynvml

        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval_seconds = interval_seconds
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.samples.append(int(info.used))
            except Exception:
                pass
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._loop, name="nvml-window-sampler", daemon=True)
        self._thread.start()
        # Immediate baseline sample so the window has a defined start point.
        info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        self.samples.append(int(info.used))

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        return {
            "source": "pynvml.nvmlDeviceGetMemoryInfo (whole device, includes all processes)",
            "interval_seconds": self.interval_seconds,
            "window": (
                "begins with the immediate baseline sample taken after the unscored warmup and "
                "torch.cuda.reset_peak_memory_stats, and ends when the sampler stops after the "
                "final timed image's CPU extraction; it covers every timed image"
            ),
            "baseline_bytes": int(self.samples[0]) if self.samples else None,
            "peak_bytes": int(max(self.samples)) if self.samples else None,
            "sample_count": len(self.samples),
        }


def mask_to_uncompressed_rle(mask: np.ndarray) -> dict:
    """Encode a 2-D raster into official COCO *uncompressed* RLE.

    Counts are in Fortran (column-major) raster order; the first run always
    counts background zeros, so a mask starting with a foreground pixel is
    stored with a leading 0 run. This is exactly the format produced by
    ``pycocotools.mask.encode`` and consumed by
    ``pycocotools.mask.frPyObjects`` in the scorer. Pure numpy: the venv has
    no pycocotools.
    """
    if mask.ndim != 2:
        raise ValueError(f"raster masks must be 2-D before RLE encoding, got {mask.shape}")
    height, width = mask.shape
    flat = np.asarray(mask, dtype=np.uint8, order="F").ravel(order="F")
    flips = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], flips, [flat.size]))
    counts = np.diff(bounds).astype(np.int64)
    if flat[0] != 0:
        counts = np.concatenate((np.zeros(1, dtype=np.int64), counts))
    if int(counts.sum()) != flat.size:
        raise RuntimeError("uncompressed RLE run lengths do not cover the whole mask")
    zeros = int(counts[0::2].sum())
    ones = int(counts[1::2].sum())
    if zeros != int((flat == 0).sum()) or ones != int((flat == 1).sum()):
        raise RuntimeError("uncompressed RLE runs do not match the mask's zero/one counts")
    return {"size": [int(height), int(width)], "counts": [int(value) for value in counts]}


def mask_raster_digest(binary_mask: np.ndarray) -> str:
    """SHA-256 over the canonical serialization of a binary mask raster.

    Canonical serialization (the documented reference the scorer must
    reconstruct from the decoded compressed RLE in
    yoloe_coco_score.py::canonical_raster_bytes):

      header   "<height>x<width>:F\\n"  (ASCII; declares the canvas
               dimensions and the Fortran/column-major raster order)
      payload  one byte per raster cell in Fortran (column-major) order,
               0x00 for background, 0x01 for foreground

    The digest covers the dimensions, the order declaration, and the
    exact binary bytes, so any orientation or content change in the
    uncompressed-to-compressed RLE conversion breaks it.
    """
    raster = np.asarray(binary_mask)
    if raster.ndim != 2:
        raise ValueError(f"mask rasters must be 2-D, got shape {raster.shape}")
    height, width = raster.shape
    binary = (raster != 0).astype(np.uint8)
    payload = f"{height}x{width}:F\n".encode("ascii") + binary.tobytes(order="F")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Data protocol
# ---------------------------------------------------------------------------

def load_subset(args: argparse.Namespace):
    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    categories = sorted(payload["categories"], key=lambda row: int(row["id"]))
    image_rows = {int(row["id"]): row for row in payload["images"]}
    present = sorted(
        image_id for image_id, row in image_rows.items()
        if (args.image_root / row["file_name"]).is_file()
    )
    if not present:
        raise ValueError(f"no COCO image files found under {args.image_root}")
    count = len(present) if args.max_images == -1 else min(args.max_images, len(present))
    image_ids = sorted(random.Random(args.seed).sample(present, count))
    return categories, image_ids, image_rows


def hyphenated_prompts(categories: list[dict]) -> list[str]:
    prompts = [re.sub(r"\s+", "-", name.strip()) for name in (str(row["name"]) for row in categories)]
    if any(not prompt for prompt in prompts) or len(set(prompts)) != len(prompts):
        raise ValueError("COCO class names must produce unique non-empty YOLOE prompts")
    return prompts


def build_class_map(categories: list[dict], prompts: list[str]) -> list[dict]:
    """Persist (index, actual category_id, original name, hyphenated prompt).

    The model output index ``i`` maps to ``class_map[i]["category_id"]``;
    the mapping never uses ``i + 1`` (COCO category ID 68 is absent).
    """
    return [
        {"index": index, "category_id": int(row["id"]), "name": str(row["name"]), "prompt": prompts[index]}
        for index, row in enumerate(categories)
    ]


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(args: argparse.Namespace, arm: ArmSpec) -> tuple[dict, NvmlMemoryBudget]:
    """Fail closed on any missing asset before any model work. No downloads."""
    if sys.executable.startswith("/usr/bin/python"):
        raise RuntimeError(
            "system Python detected; GPU inference must run under the repository "
            ".venv Python, while /usr/bin/python3 is reserved for yoloe_coco_score.py"
        )
    os.chdir(REPO_ROOT)  # mobileclip2_b.ts and models/ resolve from the repo root
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA:0 is required for this benchmark; CPU inference fallback is disabled")

    budget = NvmlMemoryBudget(torch.device(DEVICE))

    checkpoint = REPO_ROOT / arm.checkpoint
    if not checkpoint.is_file() or checkpoint.stat().st_size < MIN_CHECKPOINT_BYTES:
        raise FileNotFoundError(
            f"missing or incomplete YOLOE checkpoint: {checkpoint} "
            f"(downloads are disabled; expected >= {MIN_CHECKPOINT_BYTES // (1024**2)} MiB)"
        )
    encoder = REPO_ROOT / PROMPT_ENCODER_NAME
    if not encoder.is_file() or encoder.stat().st_size < MIN_ENCODER_BYTES:
        raise FileNotFoundError(
            f"missing local YOLOE prompt encoder: {encoder} "
            f"(downloads are disabled; expected >= {MIN_ENCODER_BYTES // (1024**2)} MiB)"
        )
    if importlib.util.find_spec("clip") is None:
        raise RuntimeError("YOLOE prompting requires the declared `clip` dependency")
    detection_yaml = None
    if arm.yaml_config is not None:
        # The installed ultralytics package ships the unified
        # cfg/models/26/yoloe-26.yaml, whose ``scales`` table declares the
        # per-scale compound constants; the scale-suffixed files
        # (yoloe-26l.yaml / yoloe-26m.yaml) do not exist on disk. The
        # installed loader (nn.tasks.yaml_model_load) unifies the logical
        # name with the regex below and infers the requested scale from it
        # (guess_model_scale), so YOLOE(arm.yaml_config) must keep the
        # scale-suffixed logical name. Preflight that exact construction
        # here, before any model work: no downloads, no fallback to the
        # segmentation head.
        model_dir = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "26"
        unified_name = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", arm.yaml_config)
        unified_yaml = model_dir / unified_name
        if not unified_yaml.is_file():
            raise FileNotFoundError(
                f"missing unified detection-only YAML in the installed ultralytics package: "
                f"{unified_yaml} (downloads are disabled)"
            )
        try:
            # Same working directory (REPO_ROOT) and same logical name as
            # the YOLOE(arm.yaml_config) construction below: the resolved
            # expression mirrors yaml_model_load's own file search, and the
            # returned dict carries the inferred scale plus the loaded
            # ``scales`` table of the unified YAML.
            resolved = check_yaml(arm.yaml_config, hard=False) or check_yaml(unified_name)
            loaded = yaml_model_load(arm.yaml_config)
        except Exception as exc:
            raise RuntimeError(
                f"installed ultralytics loader could not resolve the detection-only "
                f"YAML {arm.yaml_config!r} to the unified file {unified_yaml} "
                f"(downloads are disabled): {exc}"
            ) from exc
        if not resolved or Path(resolved).resolve() != unified_yaml.resolve():
            raise RuntimeError(
                f"installed ultralytics loader resolved {arm.yaml_config!r} to {resolved!r} "
                f"instead of the unified package file {unified_yaml}; refusing to build "
                "the derived detector from a different YAML"
            )
        if loaded.get("scale") != arm.scale:
            raise RuntimeError(
                f"YOLOE({arm.yaml_config!r}) would not preserve the arm scale {arm.scale!r} "
                f"(installed loader infers {loaded.get('scale')!r}); refusing the derived "
                "detection-only construction"
            )
        scales = loaded.get("scales")
        if not isinstance(scales, dict) or arm.scale not in scales:
            declared = sorted(scales) if isinstance(scales, dict) else None
            raise RuntimeError(
                f"unified YAML {unified_yaml} does not declare scale {arm.scale!r} "
                f"(declared scales: {declared}); refusing the derived detection-only "
                "construction"
            )
        detection_yaml = {
            "logical_name": arm.yaml_config,
            "logical_path": str(model_dir / arm.yaml_config),
            "resolved_path": str(unified_yaml),
            "inferred_scale": loaded["scale"],
        }
    if not args.annotations.is_file():
        raise FileNotFoundError(f"missing COCO annotation file: {args.annotations}")
    if not args.image_root.is_dir():
        raise NotADirectoryError(f"missing COCO image root: {args.image_root}")

    provenance = {
        "repo_root": str(REPO_ROOT),
        "repo_head": git_repo_head(REPO_ROOT),
        "script_path": str(Path(__file__).resolve()),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            # PyTorch may expose its CUDA UUID as a private _CUuuid object,
            # which the strict JSON encoder cannot serialize directly.
            "uuid": str(torch.cuda.get_device_properties(0).uuid),
            "driver_version": gpu_driver_version(),
        },
        "scorer_pinned": dict(PINNED_SCORER),
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
        },
        "prompt_encoder": {
            "path": str(encoder),
            "bytes": encoder.stat().st_size,
            "sha256": sha256(encoder),
        },
        "annotations": {
            "path": str(args.annotations.resolve()),
            "sha256": sha256(args.annotations),
        },
        "detection_yaml": detection_yaml,
        "prompt_profile": None,
    }
    try:
        # Fail during preflight, before model construction or inference, if
        # provenance contains a value unsupported by strict JSON encoding.
        json.dumps(provenance, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"provenance is not strict-JSON serializable before inference: {exc}"
        ) from exc
    if provenance["repo_head"] is None:
        print("[yoloe-component] warning: could not resolve the repository HEAD", flush=True)
    return provenance, budget


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def assert_prompt_names(model, expected_prompts: list[str], stage: str) -> None:
    names = getattr(model, "names", None)
    if not names:
        raise RuntimeError(f"{stage}: model exposes no class names after prompt profile load")
    ordered = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    if ordered != expected_prompts:
        raise RuntimeError(
            f"{stage}: loaded profile names do not equal the expected prompt order; "
            f"got {len(ordered)} names starting {ordered[:5]!r}, expected {expected_prompts[:5]!r}"
        )


def _checkpoint_and_prompts(checkpoint: Path, prompts: list[str], record: dict):
    """Load the checkpoint and set the ordered prompts (shared first stages).

    The caller owns the returned model binding and is responsible for
    dropping it before any other model is constructed.
    """
    setup = record["setup"]

    record["stage"] = "checkpoint-load"
    model, load_seconds, load_log, load_warnings = capture(lambda: YOLOE(str(checkpoint)))
    torch.cuda.synchronize(DEVICE)
    setup["checkpoint_load_seconds"] = round(load_seconds, 6)
    setup["checkpoint_load_log"] = load_log or None
    record["warnings"].extend(load_warnings)

    record["stage"] = "prompt-setup"
    _, prompt_seconds, prompt_log, prompt_warnings = capture(lambda: model.set_classes(prompts))
    torch.cuda.synchronize(DEVICE)
    setup["prompt_setup_seconds"] = round(prompt_seconds, 6)
    setup["prompt_setup_log"] = prompt_log or None
    record["warnings"].extend(prompt_warnings)
    return model


def _generate_prompt_profile(checkpoint: Path, prompts: list[str],
                             profile_path: Path, record: dict) -> None:
    """Generate the NPZ prompt profile on the same-scale segmentation checkpoint.

    The segmentation model is a local of this function only: its binding dies
    when the function returns, so the model is fully released before the
    caller constructs the derived detection-only model.
    """
    model = _checkpoint_and_prompts(checkpoint, prompts, record)
    setup = record["setup"]

    record["stage"] = "profile-save"
    if profile_path.exists():
        raise RuntimeError(
            f"prompt profile already exists: {profile_path}; a prior-run "
            "artifact is never overwritten (the startup collision preflight "
            "refuses such runs before any model work)"
        )
    _, save_seconds, save_log, save_warnings = capture(lambda: model.save_prompt_embeddings(str(profile_path)))
    torch.cuda.synchronize(DEVICE)
    setup["profile_save_seconds"] = round(save_seconds, 6)
    record["warnings"].extend(save_warnings)
    if save_log:
        record["warnings"].append(f"profile-save log: {save_log}")
    if not profile_path.is_file():
        raise FileNotFoundError(f"save_prompt_embeddings did not produce {profile_path}")
    npz = np.load(profile_path)
    record["provenance"]["prompt_profile"] = {
        "path": str(profile_path.resolve()),
        "bytes": profile_path.stat().st_size,
        "sha256": sha256(profile_path),
        "npz_entries": {key: list(npz[key].shape) for key in npz.files},
    }
    npz.close()
    del model  # last local binding; the return below releases the model


def build_model(arm: ArmSpec, args: argparse.Namespace, prompts: list[str],
                profile_path: Path | None, record: dict) -> YOLOE:
    """Build the arm's production model and record setup timings separately.

    Segmentation arms load the official checkpoint and set the ordered
    prompts. Detector arms first generate the NPZ prompt profile inside
    ``_generate_prompt_profile`` on the same-scale segmentation checkpoint
    (``set_classes`` + ``save_prompt_embeddings``); that helper's scope ends
    before the detector is built, the model is then released (``gc.collect``
    + ``torch.cuda.empty_cache``), and only then the documented
    ``YOLOE(yaml).load(checkpoint)`` transfer runs, the saved profile is
    loaded, and the loaded names are asserted to equal the expected prompt
    order.
    """
    checkpoint = REPO_ROOT / arm.checkpoint

    if arm.segmentation:
        return _checkpoint_and_prompts(checkpoint, prompts, record)

    # --- derived detection-only arm -------------------------------------
    record["stage"] = "profile-generate"
    _generate_prompt_profile(checkpoint, prompts, profile_path, record)

    record["stage"] = "profile-model-release"
    # _generate_prompt_profile's local model binding died on its return;
    # reclaim the memory before the detection-only model is constructed.
    gc.collect()
    torch.cuda.empty_cache()
    record["warnings"].append(
        "prompt profile generated on the same-scale segmentation checkpoint, which was "
        "released (helper scope ended, del, gc.collect, torch.cuda.empty_cache) before "
        "the detection-only model was built"
    )

    record["stage"] = "derived-transfer"
    model, transfer_seconds, transfer_log, transfer_warnings = capture(
        lambda: YOLOE(arm.yaml_config).load(str(checkpoint))
    )
    torch.cuda.synchronize(DEVICE)
    setup = record["setup"]
    setup["derived_transfer_seconds"] = round(transfer_seconds, 6)
    setup["transfer_log"] = transfer_log or None
    counts = parse_transfer_layer_counts(transfer_log)
    setup["transfer_layer_counts"] = counts
    if counts is None:
        record["warnings"].append(
            "YOLOE.load() transfer tensor counts are not exposed by the installed "
            "ultralytics log format; recorded as null instead of guessing"
        )
    record["warnings"].extend(transfer_warnings)

    record["stage"] = "profile-load"
    _, profile_load_seconds, profile_load_log, profile_load_warnings = capture(
        lambda: model.load_prompt_embeddings(str(profile_path))
    )
    torch.cuda.synchronize(DEVICE)
    setup["profile_load_seconds"] = round(profile_load_seconds, 6)
    setup["profile_load_log"] = profile_load_log or None
    record["warnings"].extend(profile_load_warnings)

    record["stage"] = "profile-name-assert"
    assert_prompt_names(model, prompts, "profile-load")
    setup["profile_names_match"] = True
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_one(model, args: argparse.Namespace, arm: ArmSpec, image_id: int,
                image_rows: dict, category_ids_by_index: list[int]):
    """One production-path prediction; returns (seconds, bbox_rows, segm_rows).

    The timer starts before the image path load (predict reads the path) and
    ends after torch.cuda.synchronize plus the CPU extraction of boxes and,
    for segmentation arms, raster masks and their uncompressed RLE rows.
    """
    image_row = image_rows[image_id]
    started = time.perf_counter()
    with torch.inference_mode():
        result = model.predict(
            source=str(args.image_root / image_row["file_name"]),
            device=DEVICE,
            imgsz=args.image_size,
            conf=args.confidence,
            iou=args.iou,
            max_det=args.max_det,
            batch=1,
            retina_masks=arm.segmentation,
            verbose=False,
        )[0]
    torch.cuda.synchronize(DEVICE)

    bbox_rows: list[dict] = []
    segm_rows: list[dict] = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.detach().float().cpu().numpy()
        scores = result.boxes.conf.detach().float().cpu().numpy()
        class_ids = result.boxes.cls.detach().long().cpu().numpy()
        mask_canvas = None
        if arm.segmentation:
            if result.masks is None or getattr(result.masks, "data", None) is None:
                raise RuntimeError(f"image {image_id}: segmentation arm returned boxes without a mask tensor")
            mask_canvas = result.masks.data.detach().cpu().numpy()
            if mask_canvas.ndim != 3 or mask_canvas.shape[0] != boxes_xyxy.shape[0]:
                raise RuntimeError(
                    f"image {image_id}: mask tensor {mask_canvas.shape} does not align "
                    f"with {boxes_xyxy.shape[0]} boxes"
                )
            expected_hw = (int(image_row["height"]), int(image_row["width"]))
            if mask_canvas.shape[1:] != expected_hw:
                raise RuntimeError(
                    f"image {image_id}: mask canvas {mask_canvas.shape[1:]} does not match "
                    f"the COCO image canvas {expected_hw}; refusing to resize or reconstruct"
                )
        for box, score, class_id in zip(boxes_xyxy, scores, class_ids):
            index = int(class_id)
            if index >= len(category_ids_by_index):
                raise RuntimeError(
                    f"image {image_id}: class index {index} is outside the mapped "
                    f"COCO category range [0, {len(category_ids_by_index)})"
                )
            category_id = category_ids_by_index[index]
            x1, y1, x2, y2 = (float(value) for value in box)
            width = max(x2 - x1, 0.0)
            height = max(y2 - y1, 0.0)
            bbox_rows.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "score": float(score),
                "area": width * height,
                "iscrowd": 0,
            })
        if arm.segmentation:
            for position, score in enumerate(scores):
                # While the source binary mask raster is still in memory,
                # hash its canonical serialization; the scorer re-derives
                # the same digest from the decoded compressed RLE and
                # refuses the row on a mismatch (conversion fidelity).
                binary_mask = mask_canvas[position] > 0
                segm_rows.append({
                    "image_id": image_id,
                    "category_id": category_ids_by_index[int(class_ids[position])],
                    "segmentation": mask_to_uncompressed_rle(binary_mask),
                    "raster_sha256": mask_raster_digest(binary_mask),
                    "score": float(score),
                })
    seconds = time.perf_counter() - started
    return seconds, bbox_rows, segm_rows


def run_inference(arm: ArmSpec, args: argparse.Namespace, model, image_ids: list[int],
                  image_rows: dict, record: dict, rows_jsonl: Path, status_jsonl: Path,
                  budget: NvmlMemoryBudget) -> None:
    category_ids_by_index = [row["category_id"] for row in record["class_map"]]
    timed: list[tuple[int, float, list[dict], list[dict]]] = []

    record["stage"] = "warmup"
    warmup_image_id = image_ids[0]
    warmup_seconds, _, _ = predict_one(model, args, arm, warmup_image_id, image_rows, category_ids_by_index)
    record["timings"]["warmup"] = {"image_id": warmup_image_id, "seconds": round(warmup_seconds, 6)}
    record["warmup_image_id"] = warmup_image_id
    torch.cuda.reset_peak_memory_stats(DEVICE)

    sampler = NvmlWindowSampler(device_index=0)
    sampler.start()
    try:
        total = len(image_ids)
        for position, image_id in enumerate(image_ids, start=1):
            record["stage"] = f"inference image {position}/{total}"
            seconds, bbox_rows, segm_rows = predict_one(
                model, args, arm, image_id, image_rows, category_ids_by_index
            )
            append_rows_durable(rows_jsonl, [("bbox", row) for row in bbox_rows]
                                + [("segm", row) for row in segm_rows])
            timed.append((image_id, seconds, bbox_rows, segm_rows))
            record["timings"]["per_image"].append({"image_id": image_id, "seconds": round(seconds, 6)})
            record["bbox_results"].extend(bbox_rows)
            if not arm.segmentation:
                pass
            else:
                record["segm_results"].extend(segm_rows)
            record["completed_image_count"] = len(timed)
            write_json_atomic(status_jsonl, {
                "arm": arm.name,
                "status": "running",
                "completed_image_count": len(timed),
                "total_image_count": total,
                "last_image_id": image_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "output": str(args.output),
            })
            if position % 100 == 0:
                budget.check(f"{arm.name} image {position}/{total}")
    finally:
        nvml_summary = sampler.stop()
    budget.check(f"{arm.name} inference complete")

    record["resources"] = {
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(DEVICE)),
        "cuda_peak_note": "torch.cuda peak counters reset after the unscored warmup, so they cover the timed images only",
        "nvml": nvml_summary,
    }
    latencies = [entry[1] for entry in timed]
    record["timings"]["total_seconds"] = round(float(sum(latencies)), 6)
    record["timings"]["median_ms"] = round(float(np.median(latencies)) * 1000.0, 3) if latencies else None
    record["timings"]["p95_ms"] = round(float(np.percentile(latencies, 95)) * 1000.0, 3) if latencies else None
    record["timings"]["images_per_second"] = (
        round(len(latencies) / float(sum(latencies)), 4) if latencies and sum(latencies) > 0 else None
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def make_record(arm: ArmSpec, args: argparse.Namespace, image_ids: list[int],
                class_map: list[dict], provenance: dict) -> dict:
    return {
        "schema": "yoloe26-benchmark-predictions/v1",
        "status": "partial",
        "arm": arm.name,
        "arm_spec": {
            "scale": arm.scale,
            "checkpoint": arm.checkpoint,
            "yaml_config": arm.yaml_config,
            "segmentation": arm.segmentation,
            "retina_masks": arm.segmentation,
            "derived_from_segmentation_checkpoint": arm.yaml_config is not None,
        },
        "parameters": {
            "model": arm.name,
            "annotations": str(args.annotations.resolve()),
            "image_root": str(args.image_root.resolve()),
            "max_images": args.max_images,
            "seed": args.seed,
            "image_size": args.image_size,
            "confidence": args.confidence,
            "iou": args.iou,
            "max_det": args.max_det,
            "batch": 1,
            "device": DEVICE,
        },
        "class_map": class_map,
        "sample_image_ids": image_ids,
        "warmup_image_id": None,
        "provenance": provenance,
        "setup": {
            "checkpoint_load_seconds": None,
            "checkpoint_load_log": None,
            "prompt_setup_seconds": None,
            "prompt_setup_log": None,
            "profile_save_seconds": None,
            "derived_transfer_seconds": None,
            "transfer_log": None,
            "transfer_layer_counts": None,
            "profile_load_seconds": None,
            "profile_load_log": None,
            "profile_names_match": None,
        },
        "resources": None,
        "timings": {
            "per_image": [],
            "warmup": None,
            "total_seconds": None,
            "median_ms": None,
            "p95_ms": None,
            "images_per_second": None,
        },
        "bbox_results": [],
        "segm_results": [] if arm.segmentation else None,
        "completed_image_count": 0,
        "total_image_count": len(image_ids),
        "failure": None,
        "warnings": [],
    }


def main() -> int:
    args = parse_args()
    arm = ARMS[args.model[0]]
    args.output = (args.output or Path(__file__).resolve().parent / "yoloe26"
                   / f"{arm.name}_predictions.json").expanduser().resolve()

    # Collision preflight: derive every artifact this run could write and
    # refuse before any of them is opened, truncated, or unlinked.
    # Prior-run artifacts (including interrupted partial sidecars) are
    # never deleted; the operator must choose a new unique --output path.
    profile_path = None
    if not arm.segmentation:
        # Detector arms: the profile filename is derived from the output
        # JSON stem so two runs with distinct --output paths never
        # collide on the same per-arm profile.
        profile_path = args.output.parent / f"{args.output.stem}_prompt_profile.npz"
    candidate_artifacts = {
        "final output JSON": args.output,
        "per-image rows sidecar": args.output.with_name(args.output.name + ".rows.jsonl"),
        "progress status sidecar": args.output.with_name(args.output.name + ".status.json"),
        "failure sidecar": args.output.with_name(args.output.name + ".failure.json"),
        "atomic-write temp file": args.output.with_name(args.output.name + ".tmp"),
    }
    if profile_path is not None:
        candidate_artifacts["detector prompt profile"] = profile_path
    collisions = [f"{label}: {path}" for label, path in candidate_artifacts.items()
                  if path.exists()]
    if collisions:
        print(
            "[yoloe-component] refusing to start: prior-run artifacts already "
            "exist at this output path; they are preserved and never deleted, "
            "overwritten, or truncated:\n  "
            + "\n  ".join(collisions)
            + "\n  choose a new unique --output path for this run",
            file=sys.stderr,
        )
        return 2

    rows_jsonl = args.output.with_name(args.output.name + ".rows.jsonl")
    status_jsonl = args.output.with_name(args.output.name + ".status.json")
    failure_jsonl = args.output.with_name(args.output.name + ".failure.json")

    record = None
    stage = "preflight"  # tracked separately while the record does not exist yet
    try:
        provenance, budget = preflight(args, arm)
        stage = "load-subset"
        categories, image_ids, image_rows = load_subset(args)
        stage = "make-record"
        prompts = hyphenated_prompts(categories)
        class_map = build_class_map(categories, prompts)
        record = make_record(arm, args, image_ids, class_map, provenance)
        record["stage"] = stage
        write_json_atomic(status_jsonl, {
            "arm": arm.name,
            "status": "running",
            "completed_image_count": 0,
            "total_image_count": len(image_ids),
            "last_image_id": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "output": str(args.output),
        })

        print(f"[yoloe-component] arm={arm.name} images={len(image_ids)} "
              f"device={DEVICE} output={args.output}", flush=True)
        model = build_model(arm, args, prompts, profile_path, record)
        run_inference(arm, args, model, image_ids, image_rows, record,
                      rows_jsonl, status_jsonl, budget)
        record["status"] = "complete"
        record["stage"] = "complete"
    except Exception as exc:
        if record is None:
            # Failure before the record exists (preflight / subset load):
            # build the minimal failure artifact from the tracked stage so
            # the run still leaves a non-complete, scorer-refused output.
            record = {
                "schema": "yoloe26-benchmark-predictions/v1",
                "status": "failed",
                "arm": arm.name,
                "stage": stage,
                "failure": {
                    "stage": stage,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                "completed_image_count": 0,
            }
        else:
            record["status"] = "failed"
            record["stage"] = record.get("stage", "unknown")
            record["failure"] = {
                "stage": record.get("stage", "unknown"),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        record["metrics_available"] = False
        record["interruption_note"] = (
            "this output is NOT complete; partial rows are retained in "
            f"{rows_jsonl.name} and the scorer refuses to score it"
        )
        try:
            write_json_atomic(args.output, record)
            write_json_atomic(failure_jsonl, record["failure"])
            print(f"[yoloe-component] arm={arm.name} FAILED at stage "
                  f"{record['failure']['stage']}: {record['failure']['error']} "
                  f"(partial predictions retained in {args.output} and {rows_jsonl})", flush=True)
        except Exception:
            print(f"[yoloe-component] arm={arm.name} FAILED at stage "
                  f"{record.get('stage', 'unknown')}: {exc!r}; writing the failure record also failed",
                  flush=True)
        traceback.print_exc()
        return 1

    write_json_atomic(args.output, record)
    # The scratch sidecars were only for interruption survival; the final
    # self-contained JSON supersedes them on success.
    for scratch in (rows_jsonl, status_jsonl):
        if scratch.exists():
            scratch.unlink()
    timings = record["timings"]
    print(f"[yoloe-component] arm={arm.name} complete: {len(record['bbox_results'])} bbox rows, "
          f"{len(record['segm_results'] or [])} segm rows, "
          f"{timings['images_per_second']} img/s, p95={timings['p95_ms']} ms -> {args.output}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
