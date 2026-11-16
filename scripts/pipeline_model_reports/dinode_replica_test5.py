#!/usr/bin/env python3
"""DINOde mask-isolated crop retrieval — single-template runner (test5).

Runs on ReplicaSSG GT object crops from five scenes.  This script implements
the frozen protocol described in ``dinode-crop-protocol-v2.md`` and
``dinode-api-input-brief.md``.

Output (future raw path)::

    models/experiments/dinode/data/replica_test5/dinode-replica-test5-mask-isolated-single-template.json

This runner performs a complete benchmark: it validates inputs, runs DINOde
dense-logit inference on every 4,408 crop file, computes retrieval metrics
(macro AP, top-1, top-5, per-category and per-scene), and writes the full
result JSON.  If preflight fails it records the full report and exits 1.

No network access is attempted after offline flags are set.
"""

from __future__ import annotations

# ── Bytecode suppression (before any third-party import) ────────────────────
import sys as _sys

_B = True
if _B:
    _sys.dont_write_bytecode = True

# ── Offline flags (before any torch / HF import) ───────────────────────────

import os as _os

_os.environ["HF_HUB_OFFLINE"] = "1"
_os.environ["HF_DATASETS_OFFLINE"] = "1"
_os.environ["TRANSFORMERS_OFFLINE"] = "1"
_os.environ["TORCH_HUB_OFFLINE"] = "1"
_os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ── Standard library (no cache-side-effects) ───────────────────────────────

import argparse
import copy
import datetime
import hashlib
import json
import platform
import re
import resource
import subprocess
import textwrap
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
import string
from typing import Any, Dict, List, Optional, Tuple

# ── HDD-backed TMPDIR / MPLCONFIGDIR / library caches ──────────────────────
# Must be set before NumPy, Torch, or upstream model code (they may read
# these envs at import time for temporary-file placement and cache rooting).

_CACHE_BASE: str = "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/dinode/data/replica_test5/.cache"
_os.makedirs(_os.path.join(_CACHE_BASE, "tmp"), exist_ok=True)
_os.makedirs(_os.path.join(_CACHE_BASE, "mpl"), exist_ok=True)
_os.makedirs(_os.path.join(_CACHE_BASE, "hf"), exist_ok=True)
_os.makedirs(_os.path.join(_CACHE_BASE, "torch"), exist_ok=True)
_os.environ["TMPDIR"] = _os.path.join(_CACHE_BASE, "tmp")
_os.environ["MPLCONFIGDIR"] = _os.path.join(_CACHE_BASE, "mpl")
_os.environ["HF_HOME"] = _os.path.join(_CACHE_BASE, "hf")
_os.environ["TORCH_HOME"] = _os.path.join(_CACHE_BASE, "torch")
_os.environ["TORCH_EXTRA_MODULES_PATH"] = _os.path.join(_CACHE_BASE, "torch")

# ── Third-party (imported lazily, only when needed) ────────────────────────

import numpy as np
import torch

# ── Constants / frozen paths ───────────────────────────────────────────────

CROP_ROOT: str = (
    "/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/"
    "results/dinov3-test5-crops-fp32"
)
MANIFEST_PATH: str = (
    "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/t_ren/"
    "data/dinotxt_test5/t_ren_test5_crop_manifest_20261116.json"
)
LABELS_PATH: str = (
    "/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/"
    "ReplicaSSG/files/objects.json"
)

DINODE_SRC: str = "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/dinode/repo"
DINODE_CONFIG: str = _os.path.join(DINODE_SRC, "configs", "dinode_eval.json")
DINODE_CHECKPOINT: str = (
    "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/dinode/"
    "checkpoints/eccv26_dinode_coco_stuff.pth"
)

DINOV3_SNAPSHOT: str = (
    "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/"
    "models--facebook--dinov3-vitl16-pretrain-lvd1689m/"
    "snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51"
)
OPENCLIP_SNAPSHOT: str = (
    "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/"
    "models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/"
    "snapshots/ffcbdd7da3d52f9fd7bb15e78ab6815e1a87ef60"
)
OPENCLIP_WEIGHTS: str = _os.path.join(OPENCLIP_SNAPSHOT, "open_clip_pytorch_model.bin")
PREPROCESSOR_CONFIG: str = _os.path.join(DINOV3_SNAPSHOT, "preprocessor_config.json")

OUTPUT_DIR: str = (
    "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/dinode/"
    "data/replica_test5"
)
OUTPUT_FILE: str = _os.path.join(
    OUTPUT_DIR, "dinode-replica-test5-mask-isolated-single-template.json"
)
CONTROL_JSON: str = (
    "/mnt/HDD1/Project_Code/concept-graphs/scripts/pipeline_model_reports/"
    "data/replica-test-retrieval-controls.json"
)

EXPECTED_LABELS_SHA256: str = (
    "3bceb3d838c7992ec7099dc4b46b49346b4961154376bff5748b1d1b7c5e96d5"
)
MANIFEST_EXPECTED_COUNT: int = 4408
MANIFEST_EXPECTED_BYTES: int = 341988119
MANIFEST_EXPECTED_SHA256: str = (
    "f59384e7d3a1f4bdb60a79caf238aa965b2c45b611b9d4aa7467077c4d9c0b11"
)
EXPECTED_SCENES: Dict[str, int] = {
    "office_1": 338,
    "office_3": 553,
    "office_4": 402,
    "room_1": 505,
    "room_2": 406,
}
EXPECTED_SCENE_TOTAL: int = 2204
EXPECTED_CATEGORY_COUNT: int = 49
EXPECTED_CATEGORIES: List[str] = sorted([
    "anonymize_picture", "anonymize_text", "basket", "bed", "bench",
    "bin", "blanket", "blinds", "book", "bottle", "bowl", "box",
    "cabinet", "camera", "ceiling", "chair", "clock", "cloth",
    "comforter", "desk", "desk-organizer", "door", "floor",
    "indoor-plant", "lamp", "monitor", "nightstand", "non-plane",
    "other-leaf", "panel", "picture", "pillar", "pillow", "pipe",
    "plate", "rug", "sculpture", "shelf", "switch", "table",
    "tablet", "tissue-paper", "tv-screen", "tv-stand", "vase",
    "vent", "wall", "wall-plug", "window",
])
TEMPLATE: str = "a photo of a {}"
BOOTSTRAP_SEED: int = 20261116
BOOTSTRAP_RESAMPLES: int = 10000
RESAMPLE_SCENES: int = 5
RESAMPLE_BOUNDS: Tuple[float, float] = (2.5, 97.5)

MASK_FILL_RGB = [128, 128, 128]

# ── Utility helpers ─────────────────────────────────────────────────────────


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def host_info() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "os_release": platform.release(),
    }


def host_rss_bytes() -> int:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return ru.ru_maxrss * 1024
    except Exception:
        return 0


def host_rss_human() -> str:
    b = host_rss_bytes()
    for unit in ("B", "KiB", "MiB", "GiB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TiB"


def _cuda_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    if not torch.cuda.is_available():
        return info
    info["available"] = True
    info["device_count"] = torch.cuda.device_count()
    info["current_device"] = torch.cuda.current_device()
    info["name"] = torch.cuda.get_device_name(info["current_device"])
    try:
        info["total_memory"] = torch.cuda.get_device_properties(
            info["current_device"]
        ).total_memory
        info["allocated"] = torch.cuda.memory_allocated(
            info["current_device"]
        )
        info["reserved"] = torch.cuda.memory_reserved(
            info["current_device"]
        )
    except Exception:
        pass
    return info


def _cuda_peak() -> Dict[str, int]:
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0}
    dev = torch.cuda.current_device()
    return {
        "allocated": torch.cuda.memory_allocated(dev),
        "reserved": torch.cuda.memory_reserved(dev),
    }


def _fmt(v: Any) -> str:
    """Format a None or numeric value for safe display.

    Distinguishes None from non-finite floats so control comparison
    reporting cannot fail after a completed benchmark.
    """
    if v is None:
        return "None"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(fv):
        return "NaN"
    return f"{fv:.6f}"


# ── Result data model ───────────────────────────────────────────────────────

@dataclass
class PreflightStatus:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class PreflightReport:
    timestamp: str = ""
    gates: List[PreflightStatus] = field(default_factory=list)
    hashes: Dict[str, str] = field(default_factory=dict)
    labels_info: Dict[str, Any] = field(default_factory=dict)
    manifest_info: Dict[str, Any] = field(default_factory=dict)
    cuda_info: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    failed_gates: List[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return len(self.failed_gates) == 0

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        g = PreflightStatus(name=name, ok=ok, detail=detail)
        self.gates.append(g)
        if not ok:
            self.failed_gates.append(name)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "preflight_version": "1.0",
            "run_timestamp": self.timestamp,
            "environment": self.environment,
            "cuda": self.cuda_info,
            "host_rss": host_rss_human(),
            "gates": [asdict(g) for g in self.gates],
            "failed_gates": self.failed_gates,
            "all_ok": self.all_ok,
            "hashes": self.hashes,
            "manifest_info": self.manifest_info,
            "labels_info": self.labels_info,
        }
        if not self.all_ok:
            result["blocked_reason"] = (
                "Preflight failed — one or more gates did not pass. "
                "Inference is blocked until all gates pass."
            )
        return result


# ── Gate functions (preflight) ─────────────────────────────────────────────

def gate_cuda(report: PreflightReport) -> None:
    ci = _cuda_info()
    report.cuda_info = ci
    if not ci.get("available"):
        report.add("cuda_available", False, "torch.cuda.is_available() == False")
    else:
        report.add("cuda_available", True, f"{ci['device_count']} device(s); {ci['name']}")


def gate_python_bytecode(report: PreflightReport) -> None:
    ok = getattr(_sys, "dont_write_bytecode", True)
    report.add("bytecode_suppressed", ok, f"sys.dont_write_bytecode = {ok}")


def gate_hf_offline(report: PreflightReport) -> None:
    hf_offline = _os.environ.get("HF_HUB_OFFLINE", _os.environ.get("HF_OFFLINE", ""))
    torch_hub_offline = _os.environ.get("TORCH_HUB_OFFLINE", "")
    detail = f"HF_HUB_OFFLINE={hf_offline!r}; TORCH_HUB_OFFLINE={torch_hub_offline!r}"
    report.add("hf_offline",
               hf_offline in ("1", "true", "True") or torch_hub_offline in ("1", "true", "True"),
               detail)


def gate_hdd_dirs(report: PreflightReport) -> None:
    dir_paths = [
        ("crop_root", CROP_ROOT),
        ("dinode_src", DINODE_SRC),
        ("dinov3_snapshot", DINOV3_SNAPSHOT),
        ("openclip_snapshot", OPENCLIP_SNAPSHOT),
    ]
    for label, p in dir_paths:
        if not _os.path.isdir(p):
            report.add(f"path_exists_{label}", False, f"{p} not found")
        elif not _os.access(p, _os.R_OK):
            report.add(f"path_exists_{label}", False, f"{p} not readable")
        else:
            report.add(f"path_exists_{label}", True, f"{p} (directory)")

    file_paths = [
        ("manifest", MANIFEST_PATH),
        ("labels", LABELS_PATH),
        ("dinode_config", DINODE_CONFIG),
        ("dinode_checkpoint", DINODE_CHECKPOINT),
        ("openclip_weights", OPENCLIP_WEIGHTS),
    ]
    for label, p in file_paths:
        if not _os.path.isfile(p):
            report.add(f"path_exists_{label}", False, f"{p} not found")
        elif not _os.access(p, _os.R_OK):
            report.add(f"path_exists_{label}", False, f"{p} not readable")
        else:
            sz = _os.path.getsize(p)
            report.add(f"path_exists_{label}", True, f"{p} ({sz} bytes)")


def gate_preprocessor_config(report: PreflightReport) -> None:
    if not _os.path.isfile(PREPROCESSOR_CONFIG):
        report.add("preprocessor_config_present", False,
                    f"MISSING: {PREPROCESSOR_CONFIG}")
    else:
        digest = sha256_file(PREPROCESSOR_CONFIG)
        report.hashes["preprocessor_config"] = digest
        report.add("preprocessor_config_present", True,
                    f"SHA-256={digest}; {_os.path.getsize(PREPROCESSOR_CONFIG)} bytes")


def gate_checkpoint_hash(report: PreflightReport) -> None:
    digest = sha256_file(DINODE_CHECKPOINT)
    report.hashes["checkpoint"] = digest
    expected = "494354fe148a278a99c8186a59d9fa5ef14cf6254e312195eb08cac69cac1377"
    report.add("checkpoint_hash", digest == expected,
                f"SHA-256={digest} (expected {expected})")


def gate_openclip_hash(report: PreflightReport) -> None:
    digest = sha256_file(OPENCLIP_WEIGHTS)
    report.hashes["openclip_weights"] = digest
    expected = "5ddb47339f44e4fd9cace3d3960d38af1b51a25857440cfae90afc44706d7e2b"
    report.add("openclip_weights_hash", digest == expected,
                f"SHA-256={digest} (expected {expected})")


def gate_dinov3_snapshot_hash(report: PreflightReport) -> None:
    model_path = _os.path.join(DINOV3_SNAPSHOT, "model.safetensors")
    if _os.path.isfile(model_path):
        digest = sha256_file(model_path)
        report.hashes["dinov3_model"] = digest
        expected = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
        report.add("dinov3_model_hash", digest == expected,
                    f"SHA-256={digest} (expected {expected})")
    else:
        report.add("dinov3_model_hash", False, f"model.safetensors not found in {DINOV3_SNAPSHOT}")


def gate_source_commit(report: PreflightReport) -> None:
    git_dir = _os.path.join(DINODE_SRC, ".git")
    if _os.path.isdir(git_dir):
        try:
            out = subprocess.check_output(
                ["git", "-C", DINODE_SRC, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL, text=True).strip()
        except subprocess.CalledProcessError:
            head_file = _os.path.join(git_dir, "HEAD")
            if _os.path.isfile(head_file):
                head_content = open(head_file).read().strip()
                if head_content.startswith("ref: "):
                    ref = head_content[5:]
                    ref_file = _os.path.join(git_dir, ref)
                    if _os.path.isfile(ref_file):
                        out = open(ref_file).read().strip()
                    else:
                        out = "unknown"
                else:
                    out = head_content
            else:
                out = "unknown"
        except Exception as e:
            out = f"error: {e}"
        report.hashes["source_commit"] = out
        expected = "0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1"
        report.add("source_commit", out == expected,
                    f"commit={out} (expected 0a2c518)")
    else:
        report.add("source_commit", False, f"Not a git repo at {DINODE_SRC}")


def gate_config_keys(report: PreflightReport) -> None:
    with open(DINODE_CONFIG, "r") as f:
        cfg = json.load(f)
    required = ["backbone", "text_encoder", "head", "flow", "training"]
    missing = [k for k in required if k not in cfg]
    if missing:
        report.add("config_keys", False, f"Missing keys: {missing}")
    else:
        report.add("config_keys", True,
                    f"Keys: {list(cfg.keys())}; "
                    f"backbone.model_id={cfg['backbone']['model_id']}; "
                    f"head.tau={cfg['head']['tau']}")


def gate_manifest(report: PreflightReport) -> None:
    with open(MANIFEST_PATH, "r") as f:
        mf = json.load(f)
    entries = mf.get("entries", [])
    total_files = mf.get("files", 0)
    total_bytes = mf.get("bytes", 0)
    actual_sha = mf.get("manifest_sha256", "")

    report.manifest_info = {
        "total_entries": len(entries),
        "total_files_key": total_files,
        "total_bytes_key": total_bytes,
        "actual_sha256": actual_sha,
    }

    report.add("manifest_sha256", actual_sha == MANIFEST_EXPECTED_SHA256,
                f"SHA-256={actual_sha} (expected {MANIFEST_EXPECTED_SHA256})")
    report.add("manifest_count", len(entries) == MANIFEST_EXPECTED_COUNT,
                f"entries={len(entries)} (expected {MANIFEST_EXPECTED_COUNT})")
    report.add("manifest_bytes", total_bytes == MANIFEST_EXPECTED_BYTES,
                f"bytes={total_bytes} (expected {MANIFEST_EXPECTED_BYTES})")


def gate_labels(report: PreflightReport) -> None:
    digest = sha256_file(LABELS_PATH)
    report.hashes["labels"] = digest
    report.add("labels_sha256", digest == EXPECTED_LABELS_SHA256,
                f"SHA-256={digest} (expected {EXPECTED_LABELS_SHA256})")

    with open(LABELS_PATH, "r") as f:
        labels_data = json.load(f)

    scans = labels_data.get("scans", [])
    scene_map: Dict[str, Dict[int, str]] = {}
    dup_errors: List[str] = []
    for scan in scans:
        scene_map[scan["scan"]] = {}
        for obj in scan.get("objects", []):
            obj_id = int(obj["id"])
            label = obj["label"].strip().lower()
            if obj_id in scene_map[scan["scan"]]:
                dup_errors.append(f"dup id={obj_id} in {scan['scan']}")
            else:
                scene_map[scan["scan"]][obj_id] = label

    test_scenes_set = set(EXPECTED_SCENES.keys())
    all_test_cats: set = set()
    for scan in scans:
        if scan["scan"] in test_scenes_set:
            for obj in scan.get("objects", []):
                all_test_cats.add(obj["label"].strip().lower())

    all_categories: set = set()
    for sm in scene_map.values():
        all_categories.update(sm.values())
    sorted_all = sorted(all_categories)

    report.labels_info = {
        "num_scans": len(scans),
        "scene_map_keys": list(scene_map.keys()),
        "all_categories_count": len(sorted_all),
        "test_scenes_categories_count": len(all_test_cats),
        "test_scenes_categories": sorted(all_test_cats),
    }

    if dup_errors:
        report.add("labels_no_duplicates", False, dup_errors[0])
    else:
        report.add("labels_no_duplicates", True, "No duplicates")
    report.add("labels_parsed", True,
                f"{len(scans)} scans; {len(all_test_cats)} categories across 5 test scenes")


def gate_crop_inventory(report: PreflightReport) -> None:
    scene_counts: Dict[str, int] = {s: 0 for s in EXPECTED_SCENES}
    observation_ids: set = set()
    box_pattern = re.compile(
        r"^(?P<frame>frame-\d+)-object-(?P<obj_id>\d+)-box\.png$"
    )
    for scene_name in _os.listdir(CROP_ROOT):
        if scene_name not in scene_counts:
            continue
        scene_path = _os.path.join(CROP_ROOT, scene_name)
        if not _os.path.isdir(scene_path):
            continue
        for fname in _os.listdir(scene_path):
            m = box_pattern.match(fname)
            if not m:
                continue
            frame_str = m.group("frame")
            obj_id = int(m.group("obj_id"))
            obs_key = (scene_name, frame_str, obj_id)
            if obs_key in observation_ids:
                report.add("no_duplicate_observations", False, f"Duplicate: {obs_key}")
                return
            observation_ids.add(obs_key)
            scene_counts[scene_name] = scene_counts.get(scene_name, 0) + 1

    total_obs = len(observation_ids)
    report.manifest_info["unique_observations"] = total_obs

    for scene_name, expected_count in EXPECTED_SCENES.items():
        actual = scene_counts.get(scene_name, 0)
        report.add(f"scene_count_{scene_name}", actual == expected_count,
                    f"{scene_name}: {actual} (expected {expected_count})")

    report.add("scene_total", total_obs == EXPECTED_SCENE_TOTAL,
                f"total unique (scene,frame,object_id) = {total_obs} (expected {EXPECTED_SCENE_TOTAL})")


def gate_labels_join(report: PreflightReport) -> None:
    with open(LABELS_PATH, "r") as f:
        labels_data = json.load(f)

    scans = labels_data.get("scans", [])
    scene_obj_labels: Dict[str, Dict[int, str]] = {}
    for scan in scans:
        scene_obj_labels[scan["scan"]] = {}
        for obj in scan.get("objects", []):
            scene_obj_labels[scan["scan"]][int(obj["id"])] = obj["label"].strip().lower()

    with open(MANIFEST_PATH, "r") as f:
        mf = json.load(f)
    entries = mf.get("entries", [])

    scene_obj_ids: Dict[str, Dict[int, bool]] = {}
    for e in entries:
        path_str = e.get("path", "")
        parts = path_str.split("/")
        if len(parts) < 2:
            continue
        scene = parts[0]
        fname = parts[1]
        m = re.search(r"object-(\d+)", fname)
        if not m:
            continue
        obj_id = int(m.group(1))
        if scene not in scene_obj_ids:
            scene_obj_ids[scene] = {}
        scene_obj_ids[scene][obj_id] = True

    joined_categories: set = set()
    missing_labels: List[str] = []
    malformed_ids: List[str] = []

    for scene in sorted(scene_obj_ids.keys()):
        if scene not in scene_obj_labels:
            continue
        for obj_id in sorted(scene_obj_ids[scene].keys()):
            label = scene_obj_labels[scene].get(obj_id)
            if label is None:
                missing_labels.append(f"{scene}/obj-{obj_id}")
            elif not label or not isinstance(label, str):
                malformed_ids.append(f"{scene}/obj-{obj_id} -> {label!r}")
            else:
                joined_categories.add(label)

    sorted_categories = sorted(joined_categories)

    report.manifest_info["joined_categories"] = sorted_categories
    report.manifest_info["joined_category_count"] = len(sorted_categories)
    if missing_labels:
        report.manifest_info["missing_labels"] = missing_labels[:10]
    if malformed_ids:
        report.manifest_info["malformed_ids"] = malformed_ids[:10]

    report.add("labels_join_categories",
                len(sorted_categories) == EXPECTED_CATEGORY_COUNT,
                f"joined categories={len(sorted_categories)} (expected {EXPECTED_CATEGORY_COUNT}); "
                f"missing={len(missing_labels)}, malformed={len(malformed_ids)}")
    report.add("labels_join_exact_set",
                sorted_categories == EXPECTED_CATEGORIES,
                f"set matches: {sorted_categories == EXPECTED_CATEGORIES}")


def gate_dinode_imports(report: PreflightReport) -> None:
    repo_parent = _os.path.dirname(_os.path.dirname(DINODE_SRC))
    extra = [DINODE_SRC, _os.path.join(DINODE_SRC, "model"), repo_parent]
    for p in extra:
        if p not in _sys.path:
            _sys.path.insert(0, p)
    try:
        from eval import EvalPipeline
        from model import TextCondHead, FlowTrainer
        report.add("dinode_imports", True,
                    "EvalPipeline, TextCondHead, FlowTrainer importable")
    except ImportError as e:
        report.add("dinode_imports", False, f"Import error: {e}")
    except Exception as e:
        report.add("dinode_imports", False,
                    f"Unexpected import error: {e}\n{traceback.format_exc()}")


# ── Preflight runner ───────────────────────────────────────────────────────

def run_preflight() -> PreflightReport:
    report = PreflightReport()
    report.timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    report.environment = {
        **host_info(),
        "HF_HOME": _os.environ.get("HF_HOME", ""),
        "HF_HUB_OFFLINE": _os.environ.get("HF_HUB_OFFLINE", ""),
        "TORCH_HUB_OFFLINE": _os.environ.get("TORCH_HUB_OFFLINE", ""),
        "CUDA_VISIBLE_DEVICES": _os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "TMPDIR": _os.environ.get("TMPDIR", ""),
    }

    gate_python_bytecode(report)
    gate_cuda(report)

    if not report.cuda_info.get("available"):
        report.add("hardware_ready", False, "CUDA unavailable")

    gate_hf_offline(report)
    gate_hdd_dirs(report)
    gate_preprocessor_config(report)
    gate_checkpoint_hash(report)
    gate_openclip_hash(report)
    gate_dinov3_snapshot_hash(report)
    gate_source_commit(report)
    gate_config_keys(report)
    gate_manifest(report)
    gate_labels(report)
    gate_crop_inventory(report)
    gate_labels_join(report)
    gate_dinode_imports(report)

    return report


# ── Metrics helpers ─────────────────────────────────────────────────────────


def _binary_true(label_matrix: np.ndarray, cats: List[str]) -> np.ndarray:
    """Return a (n_obs x n_cats) bool matrix; category column True where label == cat."""
    # labels: array of length n_obs; cats sorted
    cat_index = {c: i for i, c in enumerate(cats)}
    out = np.zeros((label_matrix.shape[0], len(cats)), dtype=bool)
    for i, lbl in enumerate(label_matrix):
        idx = cat_index.get(lbl)
        if idx is not None:
            out[i, idx] = True
    return out


def _ap_one_vs_rest(scores: np.ndarray, binary: np.ndarray) -> float:
    """Compute average precision (one-vs-rest) for a single category.

    scores: 1-D array of length n_obs (higher = more confident)
    binary: 1-D bool array of length n_obs
    Stable descending-score argsort; observations pre-sorted by (scene,frame,object_id).
    """
    pos = binary.sum()
    if pos == 0:
        return float("nan")
    n = len(scores)

    # stable descending-sort indices
    order = np.argsort(-scores, kind="stable")
    ranked_binary = binary[order]

    cum_pos = 0.0
    ap = 0.0
    for rank, is_pos in enumerate(ranked_binary, 1):
        if is_pos:
            cum_pos += 1.0
            ap += cum_pos / rank
    return ap / pos


def _macro_ap(scores_matrix: np.ndarray, binary_matrix: np.ndarray,
              cats: List[str]) -> Tuple[float, Dict[str, float], np.ndarray]:
    """Compute macro AP over categories with per-category detail.

    Returns (macro_ap, per_cat_ap_dict, per_cat_ap_array).
    NaN categories (no positives) are omitted from the macro average.
    """
    per_cat: Dict[str, float] = {}
    per_cat_arr = np.empty(len(cats))
    for ci, cat in enumerate(cats):
        ap_val = _ap_one_vs_rest(scores_matrix[:, ci], binary_matrix[:, ci])
        per_cat[cat] = ap_val
        per_cat_arr[ci] = ap_val

    valid = ~np.isnan(per_cat_arr)
    macro = float(np.nanmean(per_cat_arr[valid])) if valid.any() else float("nan")
    return macro, per_cat, per_cat_arr


def _top_k_accuracy(scores_matrix: np.ndarray, binary_matrix: np.ndarray,
                    k: int) -> Tuple[float, Dict[str, float]]:
    """Top-k accuracy: fraction of observations where the true category
    appears among the top-k highest-scoring categories."""
    n = len(binary_matrix)
    correct = 0
    cat_correct: Dict[str, int] = {}
    cat_total: Dict[str, int] = {}
    for i in range(n):
        # find categories with True in this row
        true_cats = binary_matrix[i].nonzero()[0]
        if len(true_cats) == 0:
            continue
        # top-k indices
        topk = np.argsort(-scores_matrix[i], kind="stable")[:k]
        if len(true_cats) <= k:
            hit = any(tc in topk for tc in true_cats)
        else:
            hit = any(tc in topk for tc in true_cats)
        if hit:
            correct += 1
        for tc in true_cats:
            cat_total[tc] = cat_total.get(tc, 0) + 1
            if hit:
                cat_correct[tc] = cat_correct.get(tc, 0) + 1

    overall = correct / n if n > 0 else 0.0
    per_cat = {}
    for cat in sorted(cat_total):
        per_cat[cat] = cat_correct.get(cat, 0) / cat_total[cat]
    return overall, per_cat


# ── Bootstrap ────────────────────────────────────────────────────────────────


def _bootstrap_macro_ap(
    scene_scores: Dict[str, np.ndarray],
    scene_binary: Dict[str, np.ndarray],
    cats: List[str],
    rng: np.random.Generator,
    n_resamples: int = 10000,
) -> Dict[str, Any]:
    """Bootstrap macro AP over scene-cluster resamples.

    Each resample draws ``RESAMPLE_SCENES`` scenes with replacement from the
    five test scenes, concatenates the observations, and computes macro AP.
    Returns percentile interval [2.5, 97.5] and per-resample included-category
    coverage statistics (compact frequency count of categories present across
    valid resamples, alongside the existing interval).
    """
    scene_names = sorted(scene_scores.keys())
    values: List[float] = []
    # Track included-category coverage per resample.
    # cat_coverage[c] = number of resamples where category c had ≥1 observation.
    cat_coverage: Dict[str, int] = {c: 0 for c in cats}

    for _ in range(n_resamples):
        chosen = rng.choice(scene_names, size=RESAMPLE_SCENES, replace=True)
        cat_scores_list: List[np.ndarray] = []
        cat_bin_list: List[np.ndarray] = []
        for sn in chosen:
            cat_scores_list.append(scene_scores[sn])
            cat_bin_list.append(scene_binary[sn])
        merged_scores = np.concatenate(cat_scores_list, axis=0)
        merged_binary = np.concatenate(cat_bin_list, axis=0)

        macro, per_cat_ap, _ = _macro_ap(
            merged_scores, merged_binary, cats
        )
        if np.isnan(macro):
            continue
        values.append(macro)
        # Count categories with ≥1 positive observation in this resample
        for ci, c in enumerate(cats):
            if merged_binary[:, ci].sum() > 0:
                cat_coverage[c] += 1

    values_arr = np.array(values) if values else np.array([])
    result: Dict[str, Any] = {
        "n_resamples": n_resamples,
        "resample_scenes": RESAMPLE_SCENES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "exploratory": True,
        "disclosure": (
            "Five-cluster bootstrap intervals are exploratory; "
            "they do not guarantee coverage of the full population."
        ),
    }
    if len(values_arr) > 0:
        lo, hi = np.percentile(values_arr, [2.5, 97.5])
        result["macro_ap_mean"] = float(np.mean(values_arr))
        result["macro_ap_median"] = float(np.median(values_arr))
        result["macro_ap_ci_lower"] = float(lo)
        result["macro_ap_ci_upper"] = float(hi)
        result["macro_ap_values_count"] = int(len(values_arr))
        # Compact included-category coverage: fraction of valid resamples
        # in which each category appeared at least once.
        n_valid = len(values_arr)
        result["category_coverage"] = {
            c: round(cat_coverage[c] / n_valid, 4)
            for c in cats
        }
    else:
        result["macro_ap_ci_lower"] = None
        result["macro_ap_ci_upper"] = None
        result["macro_ap_mean"] = None
        result["macro_ap_median"] = None
        result["category_coverage"] = {
            c: None for c in cats
        }
    return result


# ── Threshold-grouped AP (tie-robust sensitivity) ──────────────────────────


def _threshold_ap(
    scores: np.ndarray, binary: np.ndarray, thresholds: np.ndarray
) -> Dict[str, float]:
    """Compute AP at each threshold by rounding scores to the nearest threshold."""
    result = {}
    for th in thresholds:
        rounded = np.round(scores / th) * th
        ap = _ap_one_vs_rest(rounded, binary)
        result[f"th_{th:.4f}"] = float(ap) if not np.isnan(ap) else None
    return result


# ── Control comparison ──────────────────────────────────────────────────────


def _load_control(path: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load control JSON; return (data, reason).

    reason is "" on success, or one of:
      "missing_file", "read_error", "json_parse_error", "unexpected_error"
    """
    if not _os.path.isfile(path):
        return None, "missing_file"
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data, ""
    except json.JSONDecodeError:
        return None, "json_parse_error"
    except OSError:
        return None, "read_error"
    except Exception:
        return None, "unexpected_error"


def _compare_to_controls(
    control_data: Dict[str, Any],
    macro_ap: float,
    cats_ap: Dict[str, float],
    top1: float,
    top5: float,
    control_sha256: str = "",
    load_reason: str = "",
) -> Dict[str, Any]:
    """Descriptive comparison with SigLIP2 / FG-CLIP2 controls.

    Validates shared metadata from actual root paths of the control JSON,
    then reads each model's own input_conditions["ground-truth-mask-isolated"]
    block for per-category metrics.  Never reuses the first model's scores for
    other models.

    Root-level validations (dataset, settings, measurement):
      - dataset.object_observations == 2204
      - dataset.object_categories == 49
      - dataset.scenes[scene].visible_object_observations matches expected
      - settings.text_prompt_template normalised match to TEMPLATE
      - measurement.primary semantics (per-category macro AP for category
        text queries retrieving matching GT object views)

    Per-model validations (results[*]):
      - mask_background_rgb_8bit == MASK_FILL_RGB
      - per_category key set has 49 labels matching EXPECTED_CATEGORIES

    Returns named model results for SigLIP2 and FG-CLIP2, every validation
    boolean and reason (including scene_reason), and an aggregate comparable
    boolean that is true only when all required checks pass for all models.
    Missing fields produce explicit reasons and never suppress DINOde metrics.
    """
    # ══════════════════════════════════════════════════════════════════════
    # 0.  Root and results validation (fail closed, no crash)
    # ══════════════════════════════════════════════════════════════════════
    root_data: Optional[Dict[str, Any]] = (
        control_data if isinstance(control_data, dict) else None
    )
    results_is_list: bool = False
    results_nonempty: bool = False
    results_entry_validations: List[Dict[str, Any]] = []
    results_reason: Optional[str] = None
    raw_results: Any = []

    if root_data is None:
        results_reason = "Control root is not a dict"
    else:
        raw_results = root_data.get("results")
        if isinstance(raw_results, list) and len(raw_results) > 0:
            results_is_list = True
            results_nonempty = True
        else:
            results_reason = (
                "results key is missing or not a nonempty list"
            )

        # Validate each result entry is a dict with a nonempty string name
        all_entry_names: List[str] = []
        for idx, entry in enumerate(
            raw_results if results_is_list else []
        ):
            if not isinstance(entry, dict):
                results_entry_validations.append({
                    "index": idx,
                    "valid": False,
                    "reason": f"results[{idx}] is not a dict",
                })
                continue
            name = entry.get("name", "")
            if not isinstance(name, str) or not name.strip():
                results_entry_validations.append({
                    "index": idx,
                    "name": name,
                    "valid": False,
                    "reason": f"results[{idx}].name is missing or not a nonempty string",
                })
            else:
                all_entry_names.append(name.strip())
                results_entry_validations.append({
                    "index": idx,
                    "name": name,
                    "valid": True,
                })

    # ══════════════════════════════════════════════════════════════════════
    # 1.  Root-level shared metadata validations
    # ══════════════════════════════════════════════════════════════════════
    same_cohort: bool = False
    same_49_vocab: bool = False
    same_literal_template: bool = False
    same_ap_semantics: bool = False
    control_scene_match: bool = False

    cohort_reason: Optional[str] = None
    vocab_reason: Optional[str] = None
    scene_reason: Optional[str] = None
    template_reason: Optional[str] = None
    ap_semantics_reason: Optional[str] = None

    control_obs_count: Optional[int] = None
    control_template_used: Optional[str] = None

    # Derived results list for downstream iteration
    results_list: List[Dict[str, Any]] = (
        [e for e in raw_results if isinstance(e, dict)]
        if results_is_list
        else []
    )

    if root_data is None:
        cohort_reason = "Control data is not a dict"
        vocab_reason = "Control data is not a dict"
        scene_reason = "Control data is not a dict"
        template_reason = "Control data is not a dict"
        ap_semantics_reason = "Control data is not a dict"
    else:
        ds = root_data.get("dataset")
        if isinstance(ds, dict):
            obs_count = ds.get("object_observations")
            if obs_count == 2204:
                same_cohort = True
                control_obs_count = 2204
            elif obs_count is not None:
                cohort_reason = (
                    f"dataset.object_observations={obs_count}, expected 2204"
                )
            else:
                cohort_reason = "dataset.object_observations missing or not numeric"

            cats_count = ds.get("object_categories")
            if cats_count == 49:
                same_49_vocab = True
            elif cats_count is not None:
                vocab_reason = (
                    f"dataset.object_categories={cats_count}, expected 49"
                )
            else:
                vocab_reason = "dataset.object_categories missing or not numeric"

            scenes = ds.get("scenes")
            if isinstance(scenes, dict):
                expected_scene_keys = set(EXPECTED_SCENES.keys())
                actual_scene_keys = set(scenes.keys())
                missing_scenes = expected_scene_keys - actual_scene_keys
                extra_scenes = actual_scene_keys - expected_scene_keys
                scene_reason_parts: List[str] = []
                scene_ok = True
                for scene_name, expected_count in sorted(EXPECTED_SCENES.items()):
                    if scene_name not in scenes:
                        scene_ok = False
                        scene_reason_parts.append(
                            f"missing scene: {scene_name}"
                        )
                        continue
                    scene_obj = scenes[scene_name]
                    if not isinstance(scene_obj, dict):
                        scene_ok = False
                        scene_reason_parts.append(
                            f"dataset.scenes.{scene_name} is not a dict"
                        )
                        continue
                    vis = scene_obj.get("visible_object_observations")
                    if vis != expected_count:
                        scene_ok = False
                        scene_reason_parts.append(
                            f"dataset.scenes.{scene_name}"
                            f".visible_object_observations={vis}, "
                            f"expected {expected_count}"
                        )
                if missing_scenes:
                    scene_ok = False
                    scene_reason_parts.append(
                        f"missing scene keys: {sorted(missing_scenes)}"
                    )
                if extra_scenes:
                    scene_ok = False
                    scene_reason_parts.append(
                        f"unexpected scene keys: {sorted(extra_scenes)}"
                    )
                if scene_ok:
                    control_scene_match = True
                if scene_reason_parts:
                    scene_reason = "; ".join(scene_reason_parts)
            else:
                scene_reason = "dataset.scenes is not a dict"
        else:
            cohort_reason = "dataset is not a dict"
            vocab_reason = "dataset is not a dict"
            scene_reason = "dataset is not a dict"

        settings = root_data.get("settings")
        if isinstance(settings, dict):
            ctrl_tpl = settings.get("text_prompt_template")
            if isinstance(ctrl_tpl, str):
                control_template_used = ctrl_tpl
                # Normalise: {category} and {} are equivalent one-slot forms
                norm_ctrl = ctrl_tpl.replace("{category}", "{}").strip()
                norm_exp = TEMPLATE.replace("{category}", "{}").strip()
                if norm_ctrl == norm_exp:
                    same_literal_template = True
                else:
                    template_reason = (
                        f"normalised control template {norm_ctrl!r} != "
                        f"{norm_exp!r}"
                    )
            else:
                template_reason = "settings.text_prompt_template is not a string"
        else:
            template_reason = "settings is not a dict"

        measurement = root_data.get("measurement")
        if isinstance(measurement, dict):
            ctrl_ap_def = measurement.get("primary")
            if isinstance(ctrl_ap_def, str):
                # Normalize lower-case, punctuation, and whitespace
                stripped = ctrl_ap_def.lower()
                stripped = stripped.translate(
                    str.maketrans("", "", string.punctuation)
                )
                normalized_def = ' '.join(stripped.split())
                # Require exact phrase/concept matches from the frozen text.
                # Do not accept broad isolated tokens (e.g. only "gt" or only
                # "query"). The one-vs-rest wording is a caveat, not a
                # requirement.
                essential_phrases = [
                    "macro ap",
                    "category text query",
                    "retrieving matching gt object views",
                    "full observation bank",
                ]
                matched = [ph in normalized_def for ph in essential_phrases]
                if all(matched):
                    same_ap_semantics = True
                else:
                    missing = [
                        essential_phrases[i] for i in range(len(matched))
                        if not matched[i]
                    ]
                    ap_semantics_reason = (
                        f"control measurement.primary is missing essential "
                        f"concepts: {', '.join(missing)}"
                    )
            else:
                ap_semantics_reason = "measurement.primary is not a string"
        else:
            ap_semantics_reason = "measurement is not a dict"

    # ══════════════════════════════════════════════════════════════════════
    # 2.  Per-model ground-truth-mask-isolated block reads and validations
    # ══════════════════════════════════════════════════════════════════════
    models_compared: List[Dict[str, Any]] = []
    per_model_validations: List[Dict[str, Any]] = []

    gt_mask_isolated: Optional[Dict[str, Any]] = None

    for r in results_list:
        name = r.get("name", "")
        if not name:
            continue

        ic = r.get("input_conditions")
        if not isinstance(ic, dict):
            per_model_validations.append({
                "model": name,
                "has_gt_mask_isolated": False,
                "reason": "input_conditions is not a dict",
            })
            models_compared.append({
                "model": name,
                "macro_ap": None,
                "top1_micro": None,
                "top5_micro": None,
            })
            continue

        gtm = ic.get("ground-truth-mask-isolated")
        if gtm is None or not isinstance(gtm, dict):
            per_model_validations.append({
                "model": name,
                "has_gt_mask_isolated": False,
                "reason": "ground-truth-mask-isolated missing or not a dict",
            })
            models_compared.append({
                "model": name,
                "macro_ap": None,
                "top1_micro": None,
                "top5_micro": None,
            })
            continue

        # Save the first model's block for top-level DINOde comparison
        if gt_mask_isolated is None:
            gt_mask_isolated = gtm

        # Extract metrics from THIS model's own ground-truth-mask-isolated
        ctrl_macro = gtm.get("object_query_retrieval_map_macro_category")
        ctrl_top1 = gtm.get("zero_shot_top1_micro")
        ctrl_top5 = gtm.get("zero_shot_top5_micro")
        ctrl_per_cat = gtm.get("per_category", {})

        # Validate mask_background_rgb_8bit == MASK_FILL_RGB
        mask_ok: bool = False
        mask_reason: Optional[str] = None
        mb = r.get("mask_background_rgb_8bit")
        if isinstance(mb, list) and mb == MASK_FILL_RGB:
            mask_ok = True
        elif mb is not None:
            mask_reason = (
                f"mask_background_rgb_8bit={mb}, expected {MASK_FILL_RGB}"
            )
        else:
            mask_reason = "mask_background_rgb_8bit key missing from result"

        # Validate per_category key set: 49 labels matching EXPECTED_CATEGORIES
        cats_ok: bool = False
        cats_reason: Optional[str] = None
        if isinstance(ctrl_per_cat, dict):
            cat_keys = sorted(ctrl_per_cat.keys())
            if len(cat_keys) != 49:
                cats_reason = (
                    f"per_category has {len(cat_keys)} labels, expected 49"
                )
            elif cat_keys != EXPECTED_CATEGORIES:
                cats_reason = "per_category label set mismatch with EXPECTED_CATEGORIES"
            else:
                cats_ok = True
        else:
            cats_reason = "per_category is not a dict"

        per_model_validations.append({
            "model": name,
            "has_gt_mask_isolated": True,
            "mask_ok": mask_ok,
            "mask_reason": mask_reason,
            "cats_ok": cats_ok,
            "cats_reason": cats_reason,
            "macro_ap": ctrl_macro,
            "top1_micro": ctrl_top1,
            "top5_micro": ctrl_top5,
        })

        models_compared.append({
            "model": name,
            "macro_ap": ctrl_macro,
            "top1_micro": ctrl_top1,
            "top5_micro": ctrl_top5,
            "per_category": ctrl_per_cat,
        })

    # ══════════════════════════════════════════════════════════════════════
    # 2b. Stricter per-model validation: fail closed on missing, malformed,
    #     duplicate, or unexpected controls.
    # ══════════════════════════════════════════════════════════════════════
    per_model_strict_ok: bool = True
    strict_reasons: List[str] = []

    # Require settings.models is a nonempty list of unique nonempty strings
    settings_models_valid: bool = True
    settings_models_reason: Optional[str] = None
    expected_model_set: set = set()
    sm = root_data.get("settings") if isinstance(root_data, dict) else None
    if isinstance(sm, dict):
        sm_models = sm.get("models")
        if sm_models is not None:
            if not isinstance(sm_models, list):
                settings_models_valid = False
                settings_models_reason = "settings.models is not a list"
            elif len(sm_models) == 0:
                settings_models_valid = False
                settings_models_reason = "settings.models is empty"
            else:
                valid_models: List[str] = []
                seen_models: set = set()
                for mi, m in enumerate(sm_models):
                    if not isinstance(m, str) or not m.strip():
                        settings_models_valid = False
                        settings_models_reason = (
                            f"settings.models[{mi}] is not a nonempty string"
                        )
                        break
                    if m.strip() in seen_models:
                        settings_models_valid = False
                        settings_models_reason = (
                            f"settings.models contains duplicate: {m.strip()!r}"
                        )
                        break
                    seen_models.add(m.strip())
                    valid_models.append(m.strip())
                expected_model_set = set(valid_models)
        else:
            settings_models_valid = False
            settings_models_reason = "settings.models key is missing"
    else:
        settings_models_valid = False
        settings_models_reason = "settings is not a dict"

    # Require unique result names, matching settings.models as a set
    result_name_set: set = set()
    for r2 in results_list:
        n2 = r2.get("name", "")
        if not isinstance(n2, str) or not n2.strip():
            continue
        stripped = n2.strip()
        if stripped in result_name_set:
            per_model_strict_ok = False
            strict_reasons.append(f"Duplicate model name: {stripped!r}")
        result_name_set.add(stripped)

    if expected_model_set and result_name_set:
        if result_name_set != expected_model_set:
            per_model_strict_ok = False
            missing = expected_model_set - result_name_set
            extra = result_name_set - expected_model_set
            parts = []
            if missing:
                parts.append(f"missing from results: {sorted(missing)}")
            if extra:
                parts.append(f"unexpected in results: {sorted(extra)}")
            strict_reasons.append("; ".join(parts))

    # Record settings.models validation status
    if not settings_models_valid:
        strict_reasons.append(f"settings.models invalid: {settings_models_reason}")

    # Validate each model's own metrics and per_category fields
    for vm2 in per_model_validations:
        model_name = vm2.get("model", "unknown")
        if not vm2.get("has_gt_mask_isolated"):
            per_model_strict_ok = False
            strict_reasons.append(
                f"{model_name}: missing ground-truth-mask-isolated"
            )
            continue
        if not vm2.get("mask_ok", False):
            per_model_strict_ok = False
            strict_reasons.append(
                f"{model_name}: {vm2.get('mask_reason', 'mask validation failed')}"
            )
        if not vm2.get("cats_ok", False):
            per_model_strict_ok = False
            strict_reasons.append(
                f"{model_name}: {vm2.get('cats_reason', 'category validation failed')}"
            )
        # Validate macro AP is actual finite number (int/float, not bool, not string)
        v_macro = vm2.get("macro_ap")
        if isinstance(v_macro, bool):
            per_model_strict_ok = False
            strict_reasons.append(
                f"{model_name}: macro AP is a bool, not a number"
            )
        elif isinstance(v_macro, (int, float)):
            if not np.isfinite(float(v_macro)):
                per_model_strict_ok = False
                strict_reasons.append(
                    f"{model_name}: macro AP is not finite ({v_macro!r})"
                )
        else:
            per_model_strict_ok = False
            strict_reasons.append(
                f"{model_name}: macro AP is not a number ({v_macro!r})"
            )
        # Validate top-1 is actual finite number (int/float, not bool, not string)
        v_top1 = vm2.get("top1_micro")
        if isinstance(v_top1, bool):
            per_model_strict_ok = False
            strict_reasons.append(
                f"{model_name}: top-1 is a bool, not a number"
            )
        elif isinstance(v_top1, (int, float)):
            if not np.isfinite(float(v_top1)):
                per_model_strict_ok = False
                strict_reasons.append(
                    f"{model_name}: top-1 is not finite ({v_top1!r})"
                )
        else:
            per_model_strict_ok = False
            strict_reasons.append(
                f"{model_name}: top-1 is not a number ({v_top1!r})"
            )
        # Validate top-5 is actual finite number (int/float, not bool, not string)
        v_top5 = vm2.get("top5_micro")
        if isinstance(v_top5, bool):
            per_model_strict_ok = False
            strict_reasons.append(
                f"{model_name}: top-5 is a bool, not a number"
            )
        elif isinstance(v_top5, (int, float)):
            if not np.isfinite(float(v_top5)):
                per_model_strict_ok = False
                strict_reasons.append(
                    f"{model_name}: top-5 is not finite ({v_top5!r})"
                )
        else:
            per_model_strict_ok = False
            strict_reasons.append(
                f"{model_name}: top-5 is not a number ({v_top5!r})"
            )
        # Validate per_category field types
        v_pc = vm2.get("per_category", {})
        if not isinstance(v_pc, dict):
            per_model_strict_ok = False
            strict_reasons.append(
                f"{model_name}: per_category is not a dict"
            )
            continue
        cat_keys = set(v_pc.keys())
        for cat_name, cat_vals in v_pc.items():
            if not isinstance(cat_vals, dict):
                per_model_strict_ok = False
                strict_reasons.append(
                    f"{model_name}/{cat_name}: per_category entry is not a dict"
                )
                continue
            required_fields = {
                "query_average_precision",
                "zero_shot_top1_accuracy",
                "ground_truth_objects",
            }
            missing_fields = required_fields - set(cat_vals.keys())
            if missing_fields:
                per_model_strict_ok = False
                strict_reasons.append(
                    f"{model_name}/{cat_name}: "
                    f"missing per_category fields: {sorted(missing_fields)}"
                )
                continue
            # Validate query_average_precision is actual finite number (not bool, not string)
            qap_val = cat_vals.get("query_average_precision")
            if isinstance(qap_val, bool):
                per_model_strict_ok = False
                strict_reasons.append(
                    f"{model_name}/{cat_name}: "
                    f"query_average_precision is a bool, not a number"
                )
            elif isinstance(qap_val, (int, float)):
                if not np.isfinite(float(qap_val)):
                    per_model_strict_ok = False
                    strict_reasons.append(
                        f"{model_name}/{cat_name}: "
                        f"query_average_precision not finite ({qap_val!r})"
                    )
            else:
                per_model_strict_ok = False
                strict_reasons.append(
                    f"{model_name}/{cat_name}: "
                    f"query_average_precision is not a number ({qap_val!r})"
                )
            # Validate zero_shot_top1_accuracy is actual finite number (not bool, not string)
            zt1_val = cat_vals.get("zero_shot_top1_accuracy")
            if isinstance(zt1_val, bool):
                per_model_strict_ok = False
                strict_reasons.append(
                    f"{model_name}/{cat_name}: "
                    f"zero_shot_top1_accuracy is a bool, not a number"
                )
            elif isinstance(zt1_val, (int, float)):
                if not np.isfinite(float(zt1_val)):
                    per_model_strict_ok = False
                    strict_reasons.append(
                        f"{model_name}/{cat_name}: "
                        f"zero_shot_top1_accuracy not finite ({zt1_val!r})"
                    )
            else:
                per_model_strict_ok = False
                strict_reasons.append(
                    f"{model_name}/{cat_name}: "
                    f"zero_shot_top1_accuracy is not a number ({zt1_val!r})"
                )
            # Validate ground_truth_objects is actual non-negative integer (not bool, not float)
            gto = cat_vals.get("ground_truth_objects")
            if isinstance(gto, bool):
                per_model_strict_ok = False
                strict_reasons.append(
                    f"{model_name}/{cat_name}: "
                    f"ground_truth_objects is a bool, not an integer"
                )
            elif isinstance(gto, int):
                if gto < 0:
                    per_model_strict_ok = False
                    strict_reasons.append(
                        f"{model_name}/{cat_name}: "
                        f"ground_truth_objects negative ({gto})"
                    )
            else:
                per_model_strict_ok = False
                strict_reasons.append(
                    f"{model_name}/{cat_name}: "
                    f"ground_truth_objects is not an integer ({gto!r})"
                )

    # ══════════════════════════════════════════════════════════════════════
    # 3.  DINOde values and assembled comparison
    # ══════════════════════════════════════════════════════════════════════
    dinode_values: Dict[str, str] = {
        "dinode_macro_ap": _fmt(macro_ap),
        "dinode_top1": _fmt(top1),
        "dinode_top5": _fmt(top5),
    }

    # Pull top-level control values from first model's ground-truth-mask-isolated
    ctrl_per_cat_top: Dict[str, Any] = {}
    control_macro_val: Any = None
    control_top1_val: Any = None
    control_top5_val: Any = None

    if gt_mask_isolated is not None:
        control_macro_val = gt_mask_isolated.get(
            "object_query_retrieval_map_macro_category"
        )
        control_top1_val = gt_mask_isolated.get("zero_shot_top1_micro")
        control_top5_val = gt_mask_isolated.get("zero_shot_top5_micro")
        ctrl_per_cat_top = gt_mask_isolated.get("per_category", {})

    # Aggregate comparable: true only when all checks pass; no vacuous all([])
    results_valid: bool = (
        results_nonempty
        and all(v.get("valid", False) for v in results_entry_validations)
    )
    results_valid_reason: Optional[str] = None
    if not results_valid:
        parts: List[str] = []
        if load_reason:
            parts.append(f"control load failed: {load_reason}")
        if results_reason:
            parts.append(results_reason)
        invalid_entries = [
            v for v in results_entry_validations if not v.get("valid", False)
        ]
        for ie in invalid_entries:
            if ie.get("reason"):
                parts.append(f"invalid entry: {ie['reason']}")
        results_valid_reason = "; ".join(parts) if parts else "results validation failed"

    all_mask_ok = (
        len(per_model_validations) > 0
        and all(vm.get("mask_ok", False) for vm in per_model_validations)
    )
    all_cats_ok = (
        len(per_model_validations) > 0
        and all(vm.get("cats_ok", False) for vm in per_model_validations)
    )
    comparable = (
        results_valid
        and same_cohort
        and same_49_vocab
        and same_literal_template
        and same_ap_semantics
        and control_scene_match
        and all_mask_ok
        and all_cats_ok
        and settings_models_valid
        and per_model_strict_ok
    )

    # ── Assemble ─────────────────────────────────────────────────────────
    comparison: Dict[str, Any] = {
        **dinode_values,
        "control_sha256": control_sha256,
        "control_load_reason": load_reason,
        "comparability": {
            "comparable": comparable,
            "load_reason": load_reason,
            "results_valid": results_valid,
            "results_valid_reason": results_valid_reason,
            "same_cohort_observations": same_cohort,
            "cohort_reason": cohort_reason,
            "same_49_label_vocabulary": same_49_vocab,
            "vocab_reason": vocab_reason,
            "same_literal_template": same_literal_template,
            "template_reason": template_reason,
            "same_primary_ap_semantics": same_ap_semantics,
            "ap_semantics_reason": ap_semantics_reason,
            "control_observation_count": control_obs_count,
            "five_scene_counts_match": control_scene_match,
            "scene_reason": scene_reason,
            "control_template_used": control_template_used,
            "settings_models_valid": settings_models_valid,
            "settings_models_reason": settings_models_reason,
            "per_model": per_model_validations,
            "strict_reasons": strict_reasons,
        },
        "limitations": (
            "Control JSON has no per-view scores; this comparison is "
            "descriptive only. Pixel-for-pixel equivalence is not proven. "
            "No paired bootstrap or paired significance is reported. "
            "The control measurement.primary wording does not literally "
            "contain 'one-vs-rest'; the check reports available semantics "
            "and discloses this wording limit."
        ),
        "models_compared": models_compared,
    }

    if gt_mask_isolated is not None:
        comparison["control_macro_ap"] = control_macro_val
        comparison["control_top1_micro"] = control_top1_val
        comparison["control_top5_micro"] = control_top5_val
        comparison["control_per_category"] = ctrl_per_cat_top

    return comparison


# ── Inference pipeline ─────────────────────────────────────────────────────


def _collect_crops() -> List[Dict[str, Any]]:
    """Build a sorted list of crop observations from the manifest.

    Returns list of dicts with keys: scene, frame, obj_id, box_path, mask_path, label.
    Sorted by (scene, frame, obj_id).
    """
    with open(MANIFEST_PATH, "r") as f:
        mf = json.load(f)

    # Build scene → {obj_id: label} from labels file
    with open(LABELS_PATH, "r") as f:
        labels_data = json.load(f)
    scene_labels: Dict[str, Dict[int, str]] = {}
    for scan in labels_data.get("scans", []):
        scene_labels[scan["scan"]] = {}
        for obj in scan.get("objects", []):
            scene_labels[scan["scan"]][int(obj["id"])] = obj["label"].strip().lower()

    # Map box paths to mask paths from manifest
    manifest_map: Dict[str, str] = {}
    for e in mf.get("entries", []):
        p = e.get("path", "")
        if p.endswith("-box.png"):
            mask_p = p.replace("-box.png", "-mask.png")
            manifest_map[p] = mask_p

    observations: List[Dict[str, Any]] = []
    for e in mf.get("entries", []):
        p = e.get("path", "")
        if not p.endswith("-box.png"):
            continue
        parts = p.split("/")
        if len(parts) < 2:
            continue
        scene = parts[0]
        fname = parts[1]
        m = re.search(r"frame-(\d+)-object-(\d+)-box\.png$", fname)
        if not m:
            continue
        frame_str = m.group(1)
        obj_id = int(m.group(2))
        label = scene_labels.get(scene, {}).get(obj_id)
        if label is None:
            continue
        observations.append({
            "scene": scene,
            "frame": f"frame-{frame_str}",
            "obj_id": obj_id,
            "box_path": _os.path.join(CROP_ROOT, p),
            "mask_path": _os.path.join(CROP_ROOT, manifest_map.get(p, "")),
            "label": label,
        })

    observations.sort(key=lambda x: (x["scene"], x["frame"], x["obj_id"]))
    return observations


def _fill_masked_crop(
    rgb: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Fill pixels outside the positive mask with [128,128,128].

    rgb: HWC uint8, mask: HWC uint8 (grayscale, positive > 0).
    Returns new RGB array.
    """
    out = rgb.copy()
    mask_positive = mask > 0
    if not np.any(mask_positive):
        # Empty mask: fill entire crop
        out[:] = MASK_FILL_RGB
        return out
    # Fill outside mask
    out[~mask_positive] = MASK_FILL_RGB
    return out


def _verify_manifest_entries(phase: str) -> Dict[str, Any]:
    """Verify every baseline-manifest entry against on-disk path, byte size, and SHA-256.

    Requires exactly 4,408 files and 2,204 complete box/mask pairs.
    Verifies matching image dimensions and nonempty foreground masks.
    Called before and after inference to detect drift.
    """
    with open(MANIFEST_PATH, "r") as f:
        mf = json.load(f)

    entries = mf.get("entries", [])
    total_files = len(entries)
    n_box = 0
    n_mask = 0
    errors: List[str] = []
    verified_box = 0
    verified_mask = 0
    dim_checks = 0
    mask_checks = 0

    # Require exactly 4,408 total files
    if total_files != 4408:
        errors.append(
            f"Expected 4,408 total manifest entries, got {total_files}"
        )

    # Track box and mask stems for pairing verification
    box_stems: set = set()
    mask_stems: set = set()
    # Track per-scene counts
    scene_counts: Dict[str, int] = {}

    for e in entries:
        # Require {path, bytes, sha256} schema; fail closed if any required field missing
        path_str = e.get("path")
        byte_size = e.get("bytes")
        file_sha = e.get("sha256")
        if path_str is None or byte_size is None or file_sha is None:
            errors.append(
                f"Missing required field(s) in manifest entry: {e}"
            )
            continue

        # Require safe relative paths beneath the crop root
        rel_path = path_str.replace("\\", "/")
        if not rel_path or rel_path.startswith("/") or ".." in rel_path.split("/"):
            errors.append(f"Unsafe path in manifest: {path_str}")
            continue

        disk_path = _os.path.join(CROP_ROOT, rel_path)

        # Track per-scene counts
        parts = rel_path.split("/")
        if len(parts) >= 2:
            scene = parts[0]
            scene_counts[scene] = scene_counts.get(scene, 0) + 1

        if path_str.endswith("-box.png"):
            # Track box stem
            box_stems.add(rel_path[:-9])  # Remove "-box.png"
            n_box += 1

            if not _os.path.isfile(disk_path):
                errors.append(f"Missing box: {disk_path}")
                continue

            actual_size = _os.path.getsize(disk_path)
            if actual_size != byte_size:
                errors.append(
                    f"Box size mismatch {path_str}: "
                    f"expected {byte_size}, got {actual_size}"
                )
                continue

            actual_sha = sha256_file(disk_path)
            if actual_sha != file_sha:
                errors.append(
                    f"Box SHA mismatch {path_str}: "
                    f"expected {file_sha}, got {actual_sha}"
                )

            # Load paired mask for dimension and foreground checks
            mask_rel = path_str.replace("-box.png", "-mask.png")
            mask_disk = _os.path.join(CROP_ROOT, mask_rel)

            if _os.path.isfile(disk_path) and _os.path.isfile(mask_disk):
                pil_img = __import__("PIL").Image.open(disk_path).convert("RGB")
                pil_mask = __import__("PIL").Image.open(mask_disk).convert("L")
                w, h = pil_img.size
                mw, mh = pil_mask.size
                if w != mw or h != mh:
                    errors.append(
                        f"Dimension mismatch {path_str}: "
                        f"box({w}x{h}) mask({mw}x{mh})"
                    )

                mask_data = np.array(pil_mask)
                if mask_data.sum() == 0:
                    errors.append(f"Empty foreground mask {path_str}")

                dim_checks += 1
                mask_checks += 1

            verified_box += 1

        elif path_str.endswith("-mask.png"):
            # Track mask stem
            mask_stems.add(rel_path[:-9])  # Remove "-mask.png"
            n_mask += 1

            if not _os.path.isfile(disk_path):
                errors.append(f"Missing mask: {disk_path}")
                continue

            actual_size = _os.path.getsize(disk_path)
            if actual_size != byte_size:
                errors.append(
                    f"Mask size mismatch {path_str}: "
                    f"expected {byte_size}, got {actual_size}"
                )
                continue

            actual_sha = sha256_file(disk_path)
            if actual_sha != file_sha:
                errors.append(
                    f"Mask SHA mismatch {path_str}: "
                    f"expected {file_sha}, got {actual_sha}"
                )

            verified_mask += 1
        else:
            errors.append(f"Unexpected file type in manifest: {path_str}")

    # Verify equal paired stems
    if box_stems != mask_stems:
        missing_masks = sorted(box_stems - mask_stems)
        missing_boxes = sorted(mask_stems - box_stems)
        if missing_masks:
            errors.append(
                f"Box files without matching masks: {missing_masks[:5]}"
            )
        if missing_boxes:
            errors.append(
                f"Mask files without matching boxes: {missing_boxes[:5]}"
            )

    # Verify exact per-scene counts
    for scene_name, expected_count in EXPECTED_SCENES.items():
        actual = scene_counts.get(scene_name, 0)
        if actual != expected_count * 2:
            errors.append(
                f"Scene {scene_name}: {actual} files (expected "
                f"{expected_count * 2}, {expected_count} box + "
                f"{expected_count} mask)"
            )

    # --- Exact directory-set comparison ---
    # Construct the normalized set of manifest paths
    manifest_path_set: set = set()
    for e in entries:
        p = e.get("path", "")
        if p:
            manifest_path_set.add(p.replace("\\", "/").rstrip("/"))

    # Independently enumerate every file below CROP_ROOT
    disk_path_set: set = set()
    for _root, _dirs, files in _os.walk(CROP_ROOT):
        for _fname in files:
            full = _os.path.join(_root, _fname)
            rel = _os.path.relpath(full, CROP_ROOT).replace("\\", "/")
            disk_path_set.add(rel)

    # Require exact set equality; report bounded examples for extras/missing
    extra_on_disk = sorted(disk_path_set - manifest_path_set)
    missing_on_disk = sorted(manifest_path_set - disk_path_set)
    if extra_on_disk:
        errors.append(f"Extra files on disk: {extra_on_disk[:10]}")
    if missing_on_disk:
        errors.append(f"Files in manifest missing from disk: {missing_on_disk[:10]}")

    return {
        "phase": phase,
        "total_entries": total_files,
        "n_box": n_box,
        "n_mask": n_mask,
        "verified_box": verified_box,
        "verified_mask": verified_mask,
        "dim_checks": dim_checks,
        "mask_checks": mask_checks,
        "manifest_path_count": len(manifest_path_set),
        "disk_path_count": len(disk_path_set),
        "extra_files_count": len(extra_on_disk),
        "missing_files_count": len(missing_on_disk),
        "extra_files_examples": extra_on_disk[:10],
        "missing_files_examples": missing_on_disk[:10],
        "error_count": len(errors),
        "errors": errors[:20],
    }


def _run_inference() -> Dict[str, Any]:
    """Execute the full DINOde benchmark inference and metric pipeline."""
    import importlib

    # -- Import DINOde modules --
    repo_parent = _os.path.dirname(_os.path.dirname(DINODE_SRC))
    for p in [DINODE_SRC, _os.path.join(DINODE_SRC, "model"), repo_parent]:
        if p not in _sys.path:
            _sys.path.insert(0, p)

    eval_mod = importlib.import_module("eval")
    model_mod = importlib.import_module("model")
    EvalPipeline = eval_mod.EvalPipeline
    TextCondHead = model_mod.TextCondHead
    FlowTrainer = model_mod.FlowTrainer
    l2norm = model_mod.l2norm
    CLIPTextEncoder = model_mod.CLIPTextEncoder
    DinoV3HFBackbone = model_mod.DinoV3HFBackbone

    # -- Load config --
    with open(DINODE_CONFIG, "r") as f:
        config = json.load(f)

    # -- Offline: point backbone to local snapshot --
    # Set config_copy["backbone"]["model_id"] to DINOV3_SNAPSHOT and
    # config_copy["text_encoder"]["pretrained"] to OPENCLIP_WEIGHTS (the exact
    # open_clip_pytorch_model.bin file) before constructing EvalPipeline.
    config_copy = copy.deepcopy(config)
    config_copy["backbone"]["model_id"] = DINOV3_SNAPSHOT
    config_copy["text_encoder"]["pretrained"] = OPENCLIP_WEIGHTS

    # -- Output dir (with cache dirs inside ignored area) --
    _os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- CUDA check --
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; cannot run GPU inference")
    torch.cuda.empty_cache()

    t_load_start = time.perf_counter()

    # -- Build EvalPipeline --
    pipeline = EvalPipeline(
        config=config_copy,
        output_dir=OUTPUT_DIR,
        num_classes=EXPECTED_CATEGORY_COUNT,
        no_cache=True,
    )

    # -- Load checkpoint with official strict method --
    pipeline.load_checkpoint_for_eval(DINODE_CHECKPOINT)

    # -- Strict checkpoint key verification (no silent acceptance) --
    _ckpt_raw = torch.load(
        DINODE_CHECKPOINT, map_location="cpu", weights_only=False,
    )
    _ckpt_keys = set(_ckpt_raw.get("model_state_dict", {}).keys())
    _head_keys = set(pipeline.head.state_dict().keys())
    _missing_from_ckpt = _head_keys - _ckpt_keys
    _unexpected_from_ckpt = _ckpt_keys - _head_keys
    if _missing_from_ckpt:
        raise RuntimeError(
            f"Checkpoint missing keys that head expects: "
            f"{sorted(_missing_from_ckpt)[:10]}"
        )
    if _unexpected_from_ckpt:
        raise RuntimeError(
            f"Checkpoint has unexpected keys not in head: "
            f"{sorted(_unexpected_from_ckpt)[:10]}"
        )

    # -- Release raw checkpoint object immediately (strict validation done) --
    del _ckpt_raw

    # -- Call .eval() on backbone, text_encoder, and head --
    pipeline.backbone.eval()
    pipeline.text_encoder.eval()
    pipeline.head.eval()

    # -- Record effective tau --
    effective_tau = pipeline.head.tau

    t_load_end = time.perf_counter()

    # -- Encode all category prompts --
    # text_encoder.encode([literal_prompt], aggregate="mean") → [1, 768]
    t_encode_start = time.perf_counter()
    category_prompts = [TEMPLATE.format(cat) for cat in EXPECTED_CATEGORIES]
    with torch.inference_mode():
        clip_embeddings = torch.stack([
            pipeline.text_encoder.encode([p], aggregate="mean").squeeze(0)
            for p in category_prompts
        ], dim=0)  # [49, 768]
    t_encode_end = time.perf_counter()

    # -- Project through text_proj and L2-normalize --
    with torch.inference_mode():
        projected = pipeline.head.text_proj(clip_embeddings)  # [49, 1024]
        text_features = l2norm(projected)  # [49, 1024]

    # -- Collect crops --
    observations = _collect_crops()
    n_obs = len(observations)
    if n_obs != EXPECTED_SCENE_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_SCENE_TOTAL} observations after join, got {n_obs}"
        )

    labels_list = [obs["label"] for obs in observations]
    scene_list = [obs["scene"] for obs in observations]

    # -- Pre-inference manifest verification --
    pre_verify = _verify_manifest_entries("pre_inference")
    if pre_verify["error_count"] > 0:
        raise RuntimeError(
            f"Pre-inference manifest verification failed ({pre_verify['error_count']} errors): "
            f"{pre_verify['errors'][:5]}"
        )
    if pre_verify["n_box"] != 2204 or pre_verify["n_mask"] != 2204:
        raise RuntimeError(
            f"Expected 2,204 box + 2,204 mask pairs; got {pre_verify['n_box']} box, "
            f"{pre_verify['n_mask']} mask"
        )

    # -- Reset CUDA peak memory stats immediately before inference --
    torch.cuda.reset_peak_memory_stats()

    # -- Score matrix: [n_obs x n_cats] --
    t_infer_start = time.perf_counter()
    scores_matrix = np.zeros((n_obs, EXPECTED_CATEGORY_COUNT), dtype=np.float32)

    for idx, obs in enumerate(observations):
        # Load RGB crop
        pil_img = __import__("PIL").Image.open(obs["box_path"]).convert("RGB")
        rgb_np = np.array(pil_img, dtype=np.uint8)

        # Load mask
        pil_mask = __import__("PIL").Image.open(obs["mask_path"]).convert("L")
        mask_np = np.array(pil_mask, dtype=np.uint8)

        # Fill pixels outside mask with 128
        rgb_masked = _fill_masked_crop(rgb_np, mask_np)

        # Re-check that crops haven't changed (inventory drift detection)
        if rgb_np.shape[:2] != rgb_masked.shape[:2]:
            raise RuntimeError(f"Dimension error at observation {idx}")

        # -- Per-crop forward + pooling in inference_mode --
        with torch.inference_mode():
            # -- Slide-window inference via official EvalPipeline path --
            head_fn = lambda A: pipeline.head(A, text_features)[0]  # [1, 49, h_p, w_p]

            logits_map = pipeline._sliding_window_inference(
                __import__("PIL").Image.fromarray(rgb_masked),
                head_fn,
                crop_size=448,
                stride=224,
                max_long_side=2048,
            )  # [49, H_resized, W_resized]

            # Resize logits to native crop dimensions
            native_h, native_w = pil_img.size[1], pil_img.size[0]  # PIL: (W, H)
            logits_native = torch.nn.functional.interpolate(
                logits_map.unsqueeze(0),
                size=(native_h, native_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # [49, H, W]

            # Load mask again for pooling (binary: positive > 0)
            mask_native = __import__("PIL").Image.open(obs["mask_path"]).convert("L")
            mask_native_np = np.array(mask_native, dtype=np.float32)
            mask_positive = (mask_native_np > 0).astype(np.float32)

            total_mask_pixels = mask_positive.sum()
            if total_mask_pixels == 0:
                raise RuntimeError(f"Empty mask at observation {idx}")

            # Pool each category's logits over positive mask pixels
            # logits_native: [49, H, W]; mask_positive: [H, W]
            pooled = (logits_native * mask_positive).sum(dim=(1, 2)) / total_mask_pixels
            scores_matrix[idx] = pooled.numpy()

    t_infer_end = time.perf_counter()

    # -- Post-inference manifest verification (detect drift) --
    post_verify = _verify_manifest_entries("post_inference")
    if post_verify["error_count"] > 0:
        raise RuntimeError(
            f"Post-inference manifest verification failed ({post_verify['error_count']} errors): "
            f"{post_verify['errors'][:5]}"
        )

    # -- Compute metrics --
    binary_matrix = _binary_true(labels_list, EXPECTED_CATEGORIES)

    # Check full score matrix for finite values before metrics
    if not np.all(np.isfinite(scores_matrix)):
        raise RuntimeError("Score matrix contains non-finite values")

    # Per-category counts
    cat_counts: Dict[str, int] = {}
    for lbl in labels_list:
        cat_counts[lbl] = cat_counts.get(lbl, 0) + 1

    # Macro AP
    macro_ap, per_cat_ap, per_cat_ap_arr = _macro_ap(
        scores_matrix, binary_matrix, EXPECTED_CATEGORIES
    )

    # Top-1 / Top-5
    top1, top1_per_cat = _top_k_accuracy(scores_matrix, binary_matrix, 1)
    top5, top5_per_cat = _top_k_accuracy(scores_matrix, binary_matrix, 5)

    # Threshold-grouped AP (tie-robust sensitivity)
    thresh_vals = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
    threshold_ap_results: Dict[str, Any] = {}
    for ci, cat in enumerate(EXPECTED_CATEGORIES):
        th_ap = _threshold_ap(scores_matrix[:, ci], binary_matrix[:, ci], thresh_vals)
        threshold_ap_results[cat] = th_ap

    # Per-scene scores and binary
    scene_scores: Dict[str, np.ndarray] = {}
    scene_binary: Dict[str, np.ndarray] = {}
    scene_labels_map: Dict[str, List[str]] = {}
    for sn in EXPECTED_SCENES:
        mask = np.array([scene_list[i] == sn for i in range(n_obs)])
        if mask.sum() > 0:
            scene_scores[sn] = scores_matrix[mask]
            scene_binary[sn] = binary_matrix[mask]
            scene_labels_map[sn] = [labels_list[i] for i in range(n_obs) if mask[i]]

    # Per-scene macro AP
    scene_macro_ap: Dict[str, Any] = {}
    for sn in sorted(scene_scores):
        s_macro, s_per_cat, _ = _macro_ap(
            scene_scores[sn], scene_binary[sn], EXPECTED_CATEGORIES
        )
        # included categories (those with at least one positive observation in this scene)
        # use category-name keys from _macro_ap for robustness
        included = sorted(c for c in s_per_cat if not np.isnan(s_per_cat[c]))
        scene_macro_ap[sn] = {
            "macro_ap": s_macro,
            "included_categories": included,
            "included_count": len(included),
            "per_category_ap": {c: s_per_cat[c] for c in included},
        }

    # Bootstrap
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    bootstrap_result = _bootstrap_macro_ap(
        scene_scores, scene_binary, EXPECTED_CATEGORIES, rng
    )

    # CUDA peaks (reset before inference; report max over the whole run)
    peak_cuda = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }

    # -- Control comparison --
    control_data, control_load_reason = _load_control(CONTROL_JSON)
    # Always invoke comparison; pass a safe invalid sentinel when loading fails.
    if control_data is None:
        control_data = {"__invalid_json_root": True}
    control_sha256 = ""
    if isinstance(control_data, dict) and "__invalid_json_root" not in control_data:
        control_sha256 = sha256_file(CONTROL_JSON)
    control_comparison = _compare_to_controls(
        control_data, macro_ap, per_cat_ap, top1, top5,
        control_sha256=control_sha256,
        load_reason=control_load_reason
    )

    # -- Timing summary --
    timings = {
        "model_load_seconds": round(t_load_end - t_load_start, 4),
        "text_encode_seconds": round(t_encode_end - t_encode_start, 4),
        "inference_seconds": round(t_infer_end - t_infer_start, 4),
        "total_seconds": round(t_infer_end - t_load_start, 4),
    }

    # -- Cache/output sizes --
    cache_size = 0
    for root_dir, dirs, files in _os.walk(OUTPUT_DIR):
        for fn in files:
            fp = _os.path.join(root_dir, fn)
            if _os.path.isfile(fp):
                cache_size += _os.path.getsize(fp)

    # -- Assemble result --
    result: Dict[str, Any] = {
        "benchmark": "DINOde mask-isolated crop retrieval — single-template (test5)",
        "protocol_version": "dinode-crop-protocol-v2.md",
        "run_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "source": {
            "repo": DINODE_SRC,
            "checkpoint": DINODE_CHECKPOINT,
            "config": DINODE_CONFIG,
            "dinov3_snapshot": DINOV3_SNAPSHOT,
            "openclip_snapshot": OPENCLIP_SNAPSHOT,
            "openclip_weights": OPENCLIP_WEIGHTS,
            "preprocessor_config": PREPROCESSOR_CONFIG,
        },
        "hashes": {
            "labels": sha256_file(LABELS_PATH),
            "checkpoint": sha256_file(DINODE_CHECKPOINT),
            "config": sha256_file(DINODE_CONFIG),
            "openclip_weights": sha256_file(OPENCLIP_WEIGHTS),
            "dinov3_model": sha256_file(_os.path.join(DINOV3_SNAPSHOT, "model.safetensors")),
        },
        "preprocessor_config_hash": sha256_file(PREPROCESSOR_CONFIG) if _os.path.isfile(PREPROCESSOR_CONFIG) else None,
        "prompts": {
            "template": TEMPLATE,
            "literal": True,
            "categories": EXPECTED_CATEGORIES,
            "category_count": EXPECTED_CATEGORY_COUNT,
        },
        "environment": {
            **host_info(),
            "HF_HUB_OFFLINE": _os.environ.get("HF_HUB_OFFLINE", ""),
            "CUDA_VISIBLE_DEVICES": _os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "cuda": {
            **_cuda_info(),
            "peak_allocated": peak_cuda["allocated"],
            "peak_reserved": peak_cuda["reserved"],
        },
        "host_rss": host_rss_human(),
        "cache_paths": {
            "cache_base": _CACHE_BASE,
            "TMPDIR": _os.environ.get("TMPDIR", ""),
            "MPLCONFIGDIR": _os.environ.get("MPLCONFIGDIR", ""),
            "HF_HOME": _os.environ.get("HF_HOME", ""),
            "TORCH_HOME": _os.environ.get("TORCH_HOME", ""),
        },
        "resolved_cache_paths": {
            "cache_base": _CACHE_BASE,
            "TMPDIR": _os.path.join(_CACHE_BASE, "tmp"),
            "MPLCONFIGDIR": _os.path.join(_CACHE_BASE, "mpl"),
            "HF_HOME": _os.path.join(_CACHE_BASE, "hf"),
            "TORCH_HOME": _os.path.join(_CACHE_BASE, "torch"),
            "output_dir": OUTPUT_DIR,
        },
        "cache_output_dir": OUTPUT_DIR,
        "cache_output_bytes": cache_size,
        "timings": timings,
        "checkpoint_effective_tau": effective_tau,
        "observations": {
            "total": n_obs,
            "scenes": dict(sorted(EXPECTED_SCENES.items())),
            "scene_observation_counts": {sn: len(scene_scores[sn]) for sn in scene_scores},
            "per_category_counts": cat_counts,
        },
        "metrics": {
            "primary_ap_definition": (
                "Macro one-vs-rest AP over 49 categories. "
                "Stable descending-score argsort over observations "
                "pre-sorted by (scene, frame, object_id). "
                "NaN categories (no positives) omitted."
            ),
            "macro_ap": macro_ap,
            "top1": top1,
            "top5": top5,
            "per_category_ap": {c: float(per_cat_ap[c]) for c in EXPECTED_CATEGORIES},
            "per_category_top1": {c: float(top1_per_cat.get(c, 0.0)) for c in EXPECTED_CATEGORIES},
            "per_category_top5": {c: float(top5_per_cat.get(c, 0.0)) for c in EXPECTED_CATEGORIES},
            "per_category_counts": cat_counts,
        },
        "threshold_grouped_ap": threshold_ap_results,
        "per_scene": scene_macro_ap,
        "bootstrap": bootstrap_result,
        "per_view_scores": True,
        "scores_matrix": scores_matrix.tolist(),
        "observation_ids": [
            f"{obs['scene']}/{obs['frame']}/object-{obs['obj_id']}"
            for obs in observations
        ],
        "observation_labels": labels_list,
        "control_comparison": control_comparison,
        "limitations": (
            "Control comparison is descriptive only. Control JSON has no per-view "
            "scores. Pixel-for-pixel equivalence is not proven. No paired bootstrap "
            "or paired differences are reported."
        ),
    }

    return result


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DINOde test5 mask-isolated crop retrieval runner. "
            "Runs preflight, then inference + metrics, then writes result JSON."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run only the preflight gate; do not execute inference.",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("DINOde test5 — mask-isolated crop retrieval runner")
    print("=" * 78)
    print()

    # ── Preflight ────────────────────────────────────────────────────────
    report = run_preflight()

    print(f"Timestamp      : {report.timestamp}")
    print(f"Platform       : {report.environment.get('python', '?')} on {report.environment.get('platform', '?')}")
    print(f"Host RSS peak  : {host_rss_human()}")
    print(f"CUDA available : {report.cuda_info.get('available', False)}")
    if report.cuda_info.get("available"):
        print(f"  Device       : {report.cuda_info.get('name', '?')}")
    print()

    print("─" * 78)
    print("Preflight gates")
    print("─" * 78)
    for g in report.gates:
        status = "PASS" if g.ok else "FAIL"
        print(f"  [{status:>4}] {g.name}")
        if g.detail:
            for line in textwrap.wrap(g.detail, width=72,
                                      initial_indent="           ",
                                      subsequent_indent="           "):
                print(f"           {line}")
    print()

    if report.hashes:
        print("─" * 78)
        print("Input hashes")
        print("─" * 78)
        for k, v in sorted(report.hashes.items()):
            print(f"  {k:30s}  {v}")
        print()

    if report.manifest_info.get("total_entries"):
        print("─" * 78)
        print("Manifest summary")
        print("─" * 78)
        for k, v in report.manifest_info.items():
            print(f"  {k:35s}  {v}")
        print()

    if report.all_ok:
        print("PREFLIGHT ALL PASSED")
    else:
        print(f"PREFLIGHT  {len(report.failed_gates)} GATE(S) FAILED")
        print(f"          Failed: {', '.join(report.failed_gates)}")
    print("=" * 78)

    # Write preflight report
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    preflight_path = _os.path.join(OUTPUT_DIR, "dinode-replica-test5-preflight.json")
    if not _os.path.exists(preflight_path):
        with open(preflight_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
    else:
        print(f"\nExisting preflight report; skipping overwrite: {preflight_path}")

    if not report.all_ok:
        print("\nInference BLOCKED. Fix preflight failures and re-run.")
        _sys.exit(1)

    # ── Inference (if not preflight-only) ────────────────────────────────
    if args.preflight_only:
        print("\n--preflight-only: stopping here.")
        _sys.exit(0)

    print("\nStarting inference …")
    try:
        result = _run_inference()
    except Exception as e:
        # Capture failure evidence
        result = {
            "error": True,
            "exception": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "preflight_gates_passed": report.all_ok,
            "preflight_report": report.to_dict(),
        }
        print(f"\nInference FAILED: {e}")
        import sys as _sys2
        print(traceback.format_exc())
        _os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = OUTPUT_FILE
        if not _os.path.exists(out_path):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Error report written to: {out_path}")
        _sys.exit(2)

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("RESULTS SUMMARY")
    print("=" * 78)
    print(f"  Observations : {result['observations']['total']}")
    print(f"  Categories   : {EXPECTED_CATEGORY_COUNT}")
    print(f"  Macro AP     : {result['metrics']['macro_ap']:.6f}")
    print(f"  Top-1        : {result['metrics']['top1']:.6f}")
    print(f"  Top-5        : {result['metrics']['top5']:.6f}")
    print(f"  Load time    : {result['timings']['model_load_seconds']:.2f}s")
    print(f"  Encode time  : {result['timings']['text_encode_seconds']:.2f}s")
    print(f"  Inference    : {result['timings']['inference_seconds']:.2f}s")
    print(f"  Effective τ  : {result['checkpoint_effective_tau']:.6f}")
    boot = result.get("bootstrap", {})
    if boot.get("macro_ap_ci_lower") is not None:
        print(f"  Bootstrap CI : [{boot['macro_ap_ci_lower']:.6f}, {boot['macro_ap_ci_upper']:.6f}]")
        print(f"  Bootstrap mean: {boot['macro_ap_mean']:.6f}")
    print()

    # Per-scene
    print("  Per-scene macro AP:")
    for sn, si in sorted(result["per_scene"].items()):
        print(f"    {sn:12s}  macro_ap={si['macro_ap']:.6f}  "
              f"included={si['included_count']}")
    print()

    # Control comparison
    ctrl = result.get("control_comparison")
    if ctrl:
        print("  Control comparison:")
        print(f"    DINOde       macro_ap={_fmt(ctrl.get('dinode_macro_ap'))}  "
              f"top1={_fmt(ctrl.get('dinode_top1'))}  "
              f"top5={_fmt(ctrl.get('dinode_top5'))}")
        for m in ctrl.get("models_compared", []):
            _ma = m.get("macro_ap")
            _t1 = m.get("top1_micro")
            print(f"    {m['model']:20s}  macro_ap={_fmt(_ma):>10s}  "
                  f"top1={_fmt(_t1):>10s}")
        print()

    # ── Write result ─────────────────────────────────────────────────────
    out_path = OUTPUT_FILE
    if _os.path.exists(out_path):
        print(f"\nExisting output detected; skipping overwrite: {out_path}")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResult written to: {out_path}")

    _sys.exit(0)


if __name__ == "__main__":
    main()
