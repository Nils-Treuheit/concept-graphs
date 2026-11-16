"""T-REN ReplicaSSG test-5 GPU benchmark runner (official region-text path).

Measures T-REN's official region-text path on the frozen 2,204 ReplicaSSG
object views (5 held-out test scenes) with the official ViT-L/16 DINOv3
backbone, DINOtxt vision/text components and the T-REN RegionEncoder loaded
from the official region checkpoint.

Constraints honored:
- CUDA + FP32 only; fails clearly when CUDA is unavailable.
- Offline: HF/Torch-Hub network flags are set before any torch import;
  nothing is ever downloaded at runtime.
- No edits to the external T-REN or DINOv3 checkouts. The upstream
  hardcoded ``torch.hub.load('facebookresearch/dinov3', ...)`` calls are
  intercepted by a runner-owned wrapper installed right after
  ``import torch``: the single DINOv3 entry call is routed to the
  official factory function
  ``dinov3.hub.dinotxt.dinov3_vitl16_dinotxt_tet1280d20h24l`` of the
  pinned local DINOv3 checkout and called directly, with the imported
  ``dinov3`` package verified to resolve from that checkout, so the
  broad ``hubconf.py`` (whose segmentation modules import the absent
  ``torchmetrics``) is never loaded. The factory's ``(model,
  tokenizer)`` return is passed through unchanged; the wrapper
  ``setdefault``-injects only the local ``backbone_weights`` /
  ``weights`` / ``bpe_path_or_url`` assets (upstream never passes the
  BPE path) and ``pretrained=True``, strips the Torch Hub routing
  options the keyword-only factory does not accept, keeps every
  legitimate factory keyword, and fails loudly on unsupported inputs
  (positional arguments or unknown keywords). Every other
  ``torch.hub.load`` call passes through with its original arguments.
  The checkouts themselves stay byte-identical and unmodified.
- ``TORCH_HOME`` points at a unique run-owned temporary directory created
  inside the validated output parent (``output_path.parent``) after the
  argument and output safety checks pass and before the first torch
  import, and removed when the run exits. torch stages even file://
  weights under ``<TORCH_HOME>/hub/checkpoints``, so the large weight
  cache copies stay on the project HDD and never touch the user cache.
- The default evidence output is a unique UTC-stamped/PID JSON under the
  Git-ignored project data directory
  (``<result_dir>/t_ren_test5_region_retrieval_test5_<UTC stamp>_<pid>.json``);
  the resolved path is checked to be within the resolved result directory
  and the resolved result directory is verified inside the ConceptGraphs
  worktree root before any model work, and a rejected or unresolved
  target is never written to for success or failure evidence. An existing
  file is never overwritten unless ``--overwrite`` is given, and the
  no-overwrite rule is re-enforced atomically at write time so a
  concurrently created target is preserved.
- The GT object mask only selects one deterministic interior point per
  view (max Euclidean distance to the mask boundary, row-major tie-break),
  mapped to 512-space and passed to the official RegionEncoder as a point
  prompt. Official token aggregation is used with
  ``remove_singleton_groups=True``; resulting token counts are recorded.
- Source provenance is privacy-safe: the ``git remote -v`` origin URL is
  sanitized before recording (for network-style remotes the scheme/host
  and repository path are kept; userinfo, query parameters and fragments
  stripped); ``file://`` origins and plain local-path origins are
  recorded only as a kind plus ``path_redacted: true`` (no filesystem
  path, username or filename is ever serialized); a remote that cannot
  be safely normalized is recorded only as a redaction marker, never as
  the raw value. Commit hash, date and subject are kept.
- Per-view score = max cosine over the view's aggregated T-REN
  region-text tokens vs the 49 ``a photo of a {category}`` DINOtxt text
  queries. The joined label set must equal the exact frozen 49-category
  set (a count match alone is not sufficient), and missing, null,
  empty, ``unknown`` or ``undefined`` labels for any crop-referenced
  object ID fail the run. Metric conventions mirror
  conceptgraph/scripts/clip_component.py::_score / _average_precision.
- Comparison vs replica-test-retrieval-controls.json is descriptive only
  (aggregate controls; no paired per-view scores, no significance claims).
- A strict recursive crop-cache audit rejects symlinks, non-regular
  files (FIFOs, sockets, device nodes), nested directories, extra or
  unrecognized files, invalid names, duplicate view keys, unpaired
  box/mask files and any mismatch against the frozen per-scene view
  counts or the 4,408 direct crop files. A content-sensitive manifest
  (sorted relative paths + byte sizes + SHA-256) over every crop-cache
  file is computed before inference, compared entry-wise against
  models/experiments/t_ren/data/dinotxt_test5/t_ren_test5_crop_manifest_20261116.json
  when present (baseline
  entries are strictly validated; a malformed or duplicate-path baseline
  can never yield a match), and the same routine is recomputed after
  inference to detect any file change.

This runner does not probe GPU/model state and does not manage model
processes; GPU scheduling is owned by the root workflow.
"""
from __future__ import annotations

import os

# Offline guarantees are established before any torch/HF import.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# TORCH_HOME is deliberately NOT set at import time. main() creates a
# unique run-owned temporary directory on the project HDD (inside the
# validated output parent) and points TORCH_HOME at it only after the
# argument and output safety checks pass and before the first
# torch import; the directory is cleaned in main's finally block. An
# early --help or preflight failure therefore never triggers deletion.

import argparse
import hashlib
import json
import platform
import re
import resource
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_CROP_CACHE = (
    "/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/results/dinov3-test5-crops-fp32"
)
DEFAULT_ASSETS_DIR = (
    "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/t_ren/weights/dinov3"
)
DEFAULT_T_REN_REPO = (
    "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/t_ren/repo"
)
DEFAULT_DINOV3_REPO = (
    "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/t_ren/dinov3-repo"
)
OBJECTS_JSON = (
    "/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/ReplicaSSG/files/objects.json"
)
CONTROL_JSON = (
    "/mnt/HDD1/Project_Code/concept-graphs/scripts/pipeline_model_reports/data/"
    "replica-test-retrieval-controls.json"
)
BASELINE_MANIFEST = (
    "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/t_ren/"
    "data/dinotxt_test5/t_ren_test5_crop_manifest_20261116.json"
)
DINOTXT_ENTRY = "dinov3_vitl16_dinotxt_tet1280d20h24l"
# Torch Hub routing options accepted by torch.hub.load() itself (and by
# the hubconf entry resolution) but unknown to the official DINOv3
# factory: the hub wrapper strips exactly these for the target entry
# call, keeps every legitimate factory keyword, and fails loudly on any
# other keyword the factory does not accept.
HUB_ROUTING_OPTIONS = frozenset((
    "source", "force_reload", "trust_repo", "verbose", "skip_validation",
))
# Default evidence output directory under the Git-ignored T-REN experiment
# data folder. Benchmark results and scratch stay outside Git. The resolved
# evidence output must be inside this resolved directory, and this
# resolved directory must be inside the resolved ConceptGraphs worktree
# root (do not accept arbitrary ignored directories).
CONCEPT_GRAPHS_ROOT = "/mnt/HDD1/Project_Code/concept-graphs"
RESULT_DIR = (
    "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/t_ren/"
    "data/dinotxt_test5/t_ren_region_retrieval_test5_20261116"
)


def _default_output_path() -> str:
    """Unique default evidence output under the ignored project data directory.

    Computed once at ``parse_args`` time from the current UTC stamp and PID
    so concurrent runs can never target the same evidence file.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{RESULT_DIR}/t_ren_test5_region_retrieval_test5_{stamp}_"
        f"{os.getpid()}.json"
    )


TEXT_PROMPT_TEMPLATE = "a photo of a {category}"
TEXT_ENCODER_PROMPT = "a photo of a "
# Labels that never identify a category for a crop-referenced object.
INVALID_LABELS = frozenset({"unknown", "undefined"})
# The exact frozen 49-category query set for the held-out test scenes
# (verified against objects.json joins on 2026-11-16). The scene catalog
# carries 51 valid labels; the two without crops (cushion, sofa) are not
# part of this set.
FROZEN_CATEGORIES = frozenset((
    "anonymize_picture", "anonymize_text", "basket", "bed", "bench", "bin",
    "blanket", "blinds", "book", "bottle", "bowl", "box", "cabinet",
    "camera", "ceiling", "chair", "clock", "cloth", "comforter", "desk",
    "desk-organizer", "door", "floor", "indoor-plant", "lamp", "monitor",
    "nightstand", "non-plane", "other-leaf", "panel", "picture", "pillar",
    "pillow", "pipe", "plate", "rug", "sculpture", "shelf", "switch",
    "table", "tablet", "tissue-paper", "tv-screen", "tv-stand", "vase",
    "vent", "wall", "wall-plug", "window",
))
ASSET_FILES = {
    "backbone_weights": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinotxt_head_and_text_encoder": (
        "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
    ),
    "bpe_vocab": "bpe_simple_vocab_16e6.txt.gz",
    "tren_region_encoder_checkpoint": "tren_region_encoder.pth",
}
REQUIRED_SCENE_COUNTS = {
    "office_1": 338,
    "office_3": 553,
    "office_4": 402,
    "room_1": 505,
    "room_2": 406,
}
TOTAL_VIEWS_EXPECTED = 2204
NUM_CATEGORIES_EXPECTED = 49
TEXT_EMBED_DIM_EXPECTED = 1024
IMAGE_RESOLUTION = 512
PATCH_SIZE = 16
CROP_NAME_RE = re.compile(r"^frame-(\d+)-object-(\d+)-(box|mask)\.png$")
# 2,204 frozen views, each a direct (box, mask) pair = 4,408 crop files.
TOTAL_CROP_FILES_EXPECTED = 2 * TOTAL_VIEWS_EXPECTED


class CropCacheAuditError(RuntimeError):
    """Raised when the frozen crop cache deviates from the frozen layout."""


class OutputPathError(RuntimeError):
    """Raised when the resolved evidence output path is unsafe (collides
    with a run input) or would overwrite an existing file."""


# SHA-256 values verified by the root workflow on 2026-11-16 (BRAIN-INDEX).
# Used only as cross-checks; the runner always recomputes fresh hashes.
PRIOR_ASSET_SHA256 = {
    "backbone_weights": "e73f0550f00e5eafe9712177da027e5f34df3de5250b56370bcb9acaa23dab9c",
    "dinotxt_head_and_text_encoder": (
        "a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0"
    ),
    "bpe_vocab": "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
    "tren_region_encoder_checkpoint": (
        "7f85132237001d52176226b3fd6ebde8ecfc08569ae196d56deeb5b052bf9b1f"
    ),
}
PRIOR_ASSET_BYTES = {
    "backbone_weights": 1213021279,
    "dinotxt_head_and_text_encoder": 2253936683,
    "bpe_vocab": 1356917,
    "tren_region_encoder_checkpoint": 1455980684,
}


def validate_output_path(output_path: Path,
                         result_dir: Path,
                         protected: list[tuple[str, Path]],
                         allow_overwrite: bool) -> Path:
    """Resolve the evidence output path and reject unsafe or colliding paths.

    First verifies that ``result_dir`` resolves inside the resolved
    ConceptGraphs worktree root, and that the output path resolves inside
    the resolved result directory. Then checks ``protected``: a list of
    ``(label, path)`` pairs covering every run input (the crop cache, the
    assets directory and the four asset files (symlinks resolved at their
    targets), objects.json, the control JSON, the baseline manifest, and
    the T-REN/DINOv3 checkouts). The output path is resolved with symlinks
    followed and rejected if it equals any resolved protected path or lies
    inside a protected directory, so symlinked inputs are protected at
    their targets too. A resolved existing file or directory is rejected
    unless ``allow_overwrite`` is set (the default is to never overwrite),
    and an existing non-directory parent is rejected. Returns the resolved
    output path on success.
    """
    try:
        resolved = output_path.resolve()
    except OSError as exc:
        raise OutputPathError(
            f"Cannot resolve output path {output_path}: {exc}") from exc

    # Containment checks: result_dir must be inside the ConceptGraphs
    # worktree root; the output must be inside result_dir.
    try:
        result_dir_resolved = result_dir.resolve()
    except OSError as exc:
        raise OutputPathError(
            f"Cannot resolve result directory {result_dir}: {exc}") from exc
    try:
        cg_root_resolved = Path(CONCEPT_GRAPHS_ROOT).resolve()
    except OSError as exc:
        raise OutputPathError(
            f"Cannot resolve ConceptGraphs root {CONCEPT_GRAPHS_ROOT}: {exc}") from exc
    if not result_dir_resolved.is_relative_to(cg_root_resolved):
        raise OutputPathError(
            f"Result directory {result_dir} resolves to {result_dir_resolved}, "
            f"which is not inside the ConceptGraphs root ({cg_root_resolved})")
    if not resolved.is_relative_to(result_dir_resolved):
        raise OutputPathError(
            f"Output path {output_path} resolves to {resolved}, which is not "
            f"inside the result directory ({result_dir_resolved})")

    for label, base in protected:
        try:
            base_resolved = base.resolve()
        except OSError:
            continue  # existence of the input is checked elsewhere
        if resolved == base_resolved:
            raise OutputPathError(
                f"Output path {output_path} resolves to the {label} itself "
                f"({resolved}); refusing to write")
        if base_resolved.is_dir() and resolved.is_relative_to(base_resolved):
            raise OutputPathError(
                f"Output path {output_path} resolves inside the {label} "
                f"({base_resolved}); the evidence output must not write "
                f"into any run input")
    if resolved.exists():
        if resolved.is_dir():
            raise OutputPathError(
                f"Output path {output_path} resolves to an existing "
                f"directory: {resolved}")
        if not allow_overwrite:
            raise OutputPathError(
                f"Output path {output_path} already exists ({resolved}); "
                "refusing to overwrite an existing file (pass --overwrite "
                "to allow this)")
    parent = resolved.parent
    if parent.exists() and not parent.is_dir():
        raise OutputPathError(
            f"Output parent path {parent} exists and is not a directory")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T-REN ReplicaSSG test-5 GPU benchmark (region-text path)."
    )
    parser.add_argument("--crop-cache", type=str, default=DEFAULT_CROP_CACHE,
                        help="Frozen 512-input crop cache (box/mask PNG pairs).")
    parser.add_argument("--assets-dir", type=str, default=DEFAULT_ASSETS_DIR,
                        help="Directory holding the four local asset files.")
    parser.add_argument("--output", type=str, default=_default_output_path(),
                        help="Evidence JSON output path (default: unique "
                             "path under the result directory per run). "
                             "Existing files are refused unless --overwrite "
                             "is given.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting an existing evidence output "
                             "file (default: reject).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Inference batch size (T-REN eval.py default is 8).")
    parser.add_argument("--t-ren-repo", type=str, default=DEFAULT_T_REN_REPO,
                        help="Local T-REN checkout (read-only).")
    parser.add_argument("--dinov3-repo", type=str, default=DEFAULT_DINOV3_REPO,
                        help="Local DINOv3 checkout (read-only).")
    parser.add_argument("--objects-json", type=str, default=OBJECTS_JSON,
                        help="ReplicaSSG objects.json for the category join.")
    parser.add_argument("--control-json", type=str, default=CONTROL_JSON,
                        help="Retrieval-control aggregates for descriptive comparison.")
    parser.add_argument("--baseline-manifest", type=str, default=BASELINE_MANIFEST,
                        help="Pre-run per-file SHA-256 baseline manifest (optional).")
    return parser.parse_args()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


# Remote-origin sanitization (evidence privacy): the raw ``git remote -v``
# value is never serialized. Network-style remotes record a normalized
# identity (scheme/host/repo path) with userinfo, query parameters and
# fragments stripped; ``file://`` and plain local-path origins record
# only a kind plus a ``path_redacted: true`` marker, never a filesystem
# path, username or filename.
_REMOTE_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_REMOTE_HOST_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*$")
_REMOTE_SCP_RE = re.compile(r"^([A-Za-z0-9._-]+)@([A-Za-z0-9.-]+):(.+)$")
_REMOTE_PLAIN_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")


def _redacted_remote(note: str) -> dict[str, Any]:
    """Redaction marker for a remote origin that cannot be safely normalized.

    The raw value is deliberately omitted; only this marker is recorded so
    reviewers know a remote existed but was withheld.
    """
    return {
        "kind": "redacted",
        "note": (note + "; the raw remote value is not recorded because "
                 "safe normalization is ambiguous"),
    }


def _is_local_path(text: str) -> bool:
    """True for plain filesystem path references (local-path origins).

    These name local filesystem locations (``~`` may embed a username);
    they are classified as local-path origins and redacted rather than
    serialized.
    """
    if text.startswith("/"):
        return True
    if text == "~" or text.startswith("~/"):
        return True
    if text.startswith("./") or text.startswith("../"):
        return True
    return _REMOTE_PLAIN_NAME_RE.fullmatch(text) is not None


def _sanitize_remote_url(raw_url: str) -> dict[str, Any]:
    """Sanitize a git remote origin URL for evidence output.

    Preserves the repository identity (host and repository path where
    available) while stripping username/password (userinfo), query
    parameters and fragments from HTTPS/HTTP/SSH-style remotes and the
    userinfo from SCP-style ``user@host:path`` remotes. The SCP form
    has no syntax separating a path from a query or fragment, so an SCP
    path containing ``?`` or ``#`` is redacted wholesale: only the
    redaction marker is returned and no part of the raw path is
    persisted. ``file://`` origins and plain local-path origins are
    redacted wholesale as well: only the kind and a
    ``path_redacted: true`` marker are returned, so no filesystem path,
    username or filename is ever serialized. When the remote cannot be
    safely normalized, the raw URL is omitted entirely and only a
    redaction marker is returned. The raw value is never returned,
    printed or persisted.
    """
    text = raw_url.strip()
    if not text:
        return _redacted_remote("remote origin is empty")
    if _REMOTE_SCHEME_RE.match(text):
        try:
            parts = urlsplit(text)
        except ValueError:
            return _redacted_remote("remote origin could not be parsed")
        scheme = parts.scheme.lower()
        if scheme == "file":
            # A file:// origin names a local filesystem location (and its
            # authority component may embed a hostname or username).
            # Nothing of it is serialized: only the kind and a
            # redaction marker.
            return {"kind": "file_url", "path_redacted": True}
        host = parts.hostname or ""
        if not host or not _REMOTE_HOST_RE.fullmatch(host):
            return _redacted_remote("remote origin has no safe host")
        try:
            port = parts.port
        except ValueError:
            return _redacted_remote("remote origin has an invalid port")
        if not parts.path:
            return _redacted_remote("remote origin has no repository path")
        return {
            "kind": scheme,
            "host": f"{host}:{port}" if port is not None else host,
            "path": parts.path,
            "userinfo_redacted": (parts.username is not None
                                  or parts.password is not None),
            "query_or_fragment_dropped": bool(parts.query or parts.fragment),
        }
    match = _REMOTE_SCP_RE.match(text)
    if match:
        host = match.group(2)
        if not _REMOTE_HOST_RE.fullmatch(host):
            return _redacted_remote(
                "SCP-style remote host could not be safely normalized")
        # The SCP form has no delimiter separating a path from a query or
        # fragment: a ``?`` or ``#`` in the captured path can carry query
        # or fragment syntax (possibly including credentials), so the
        # whole remote is redacted and no part of the raw path is
        # persisted. Ordinary ``git@host:org/repo.git`` paths are
        # unaffected.
        path = match.group(3)
        if "?" in path or "#" in path:
            return _redacted_remote(
                "SCP-style remote path contains query or fragment syntax")
        return {
            "kind": "scp",
            "host": host,
            "path": path,
            "userinfo_redacted": True,
        }
    if _is_local_path(text):
        # A plain local path names a local filesystem location; nothing
        # of it is serialized: only the kind and a redaction marker.
        return {"kind": "local_path", "path_redacted": True}
    return _redacted_remote("remote origin is not a recognized form")


def git_metadata(repo_path: str) -> dict[str, Any]:
    """Read-only git revision metadata via a safe.directory override.

    The remote origin is sanitized before recording (see
    ``_sanitize_remote_url``): network-style remotes keep only the
    normalized repository identity, and ``file://`` / local-path origins
    keep only a kind plus a ``path_redacted`` marker, so the evidence can
    never expose credentials or local filesystem locations. The raw
    remote value is neither printed nor persisted.
    """
    out: dict[str, Any] = {"path": repo_path, "available": False}
    base = ["git", "-c", f"safe.directory={repo_path}", "-C", repo_path]
    try:
        head = subprocess.run(base + ["log", "-1", "--format=%H|%ci|%s"],
                              capture_output=True, text=True, timeout=60).stdout.strip()
        if head:
            commit, when, subject = head.split("|", 2)
            out.update({"head": commit, "head_date": when, "head_subject": subject})
        remotes = subprocess.run(base + ["remote", "-v"],
                                 capture_output=True, text=True, timeout=60).stdout
        origin = next((line for line in remotes.splitlines()
                       if line.startswith("origin\t")), None)
        if origin:
            # ``git remote -v`` appends " (fetch)"/" (push)" to each line;
            # keep only the URL, then sanitize it (never record raw).
            rest = origin.split("\t", 1)[1].strip()
            tail = re.match(r"^(?P<url>.+?)\s+\((?:fetch|push)\)$", rest)
            out["remote_origin"] = _sanitize_remote_url(
                tail.group("url") if tail else rest)
        dirty_lines = subprocess.run(base + ["status", "--porcelain"],
                                     capture_output=True, text=True,
                                     timeout=60).stdout.splitlines()
        out["working_tree_dirty"] = bool(dirty_lines)
        out["dirty_files_sample"] = dirty_lines[:10]
        out["available"] = True
    except Exception as exc:  # revision evidence is non-fatal
        out["error"] = str(exc)
    return out


def _manifest_digest(entries: list[dict[str, Any]], files: int, total_bytes: int) -> str:
    """Aggregate digest: SHA-256 over the canonical JSON of the manifest body.

    The root baseline manifest's aggregate digest algorithm is not documented;
    entry-wise content equality is therefore the authoritative match check and
    is recorded separately.
    """
    body = {"files": files, "bytes": total_bytes, "entries": entries}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def audit_crop_cache(crop_cache: Path) -> dict[str, Any]:
    """Recursively enumerate and validate the frozen crop cache.

    The cache must contain exactly the five configured scene directories
    at its top level and 4,408 direct crop files (2,204 box/mask pairs),
    every one a regular file. Symlinks, non-regular files (FIFOs,
    sockets, device nodes), nested directories, extra or unrecognized
    files, invalid names, duplicate view keys, missing box/mask partners
    and any per-scene or total count mismatch are collected and raised
    together as CropCacheAuditError. The cache is only read, never
    modified.

    On success returns a dict with the absolute ``root``, ``files``,
    ``views``, ``views_per_scene`` and ``entries`` sorted by relative
    path; each entry carries its relative ``path``, absolute ``full``
    path and the parsed (scene, frame, object_id, kind) view key.
    """
    crop_cache = Path(crop_cache)
    if not crop_cache.is_dir():
        raise FileNotFoundError(f"Missing crop-cache directory: {crop_cache}")
    if crop_cache.is_symlink():
        raise CropCacheAuditError(f"Crop-cache root is a symlink: {crop_cache}")
    if not crop_cache.is_absolute():
        crop_cache = crop_cache.absolute()

    violations: list[str] = []
    scene_dirs: dict[str, Path] = {}
    with os.scandir(crop_cache) as iterator:
        top_entries = sorted(iterator, key=lambda item: item.name)
    for entry in top_entries:
        if entry.is_symlink():
            violations.append(f"symlink at crop-cache top level: {entry.name}")
        elif entry.is_dir(follow_symlinks=False):
            if entry.name in REQUIRED_SCENE_COUNTS:
                scene_dirs[entry.name] = crop_cache / entry.name
            else:
                violations.append(f"unexpected top-level directory: {entry.name}")
        elif entry.is_file(follow_symlinks=False):
            violations.append(f"unexpected top-level file: {entry.name}")
        else:
            violations.append(
                f"non-regular file (FIFO/socket/device node) at crop-cache "
                f"top level: {entry.name}")
    for scene in sorted(REQUIRED_SCENE_COUNTS):
        if scene not in scene_dirs:
            violations.append(f"missing scene directory: {scene}")

    view_files: dict[tuple[str, int, int, str], str] = {}
    for scene in sorted(set(scene_dirs) & set(REQUIRED_SCENE_COUNTS)):
        scene_dir = scene_dirs[scene]
        for dirpath, dirnames, filenames in os.walk(scene_dir, followlinks=False):
            rel_dir = Path(dirpath).relative_to(scene_dir)
            for name in sorted(dirnames):
                rel = f"{scene}/{rel_dir / name}"
                if (Path(dirpath) / name).is_symlink():
                    violations.append(f"symlinked directory in {scene}: {rel}")
                else:
                    violations.append(f"nested directory in {scene}: {rel}")
            for name in sorted(filenames):
                rel = f"{scene}/{rel_dir / name}"
                full = Path(dirpath) / name
                if full.is_symlink():
                    violations.append(f"symlinked file in {scene}: {rel}")
                    continue
                try:
                    file_mode = os.lstat(full).st_mode
                except OSError as exc:
                    violations.append(f"unreadable crop file in {scene}: {rel} ({exc})")
                    continue
                if not stat.S_ISREG(file_mode):
                    violations.append(
                        f"non-regular file (FIFO/socket/device node) in "
                        f"{scene}: {rel}")
                    continue
                if rel_dir != Path("."):
                    violations.append(f"file inside nested directory in {scene}: {rel}")
                match = CROP_NAME_RE.match(name)
                if match is None:
                    violations.append(f"unrecognized crop file name in {scene}: {rel}")
                    continue
                frame = int(match.group(1))
                object_id = int(match.group(2))
                kind = match.group(3)
                key = (scene, frame, object_id, kind)
                if key in view_files:
                    violations.append(
                        f"duplicate crop view key "
                        f"{scene}/frame-{frame:04d}/object-{object_id}-{kind}: "
                        f"{view_files[key]} and {rel}")
                    continue
                view_files[key] = rel

    views: dict[tuple[str, int, int], dict[str, str]] = {}
    for (scene, frame, object_id, kind), rel in view_files.items():
        views.setdefault((scene, frame, object_id), {})[kind] = rel
    for scene, frame, object_id in sorted(views):
        for kind in ("box", "mask"):
            if kind not in views[(scene, frame, object_id)]:
                violations.append(
                    f"missing {kind} partner in {scene}: "
                    f"frame-{frame:04d}-object-{object_id}")

    views_per_scene = {
        scene: sum(1 for view in views if view[0] == scene)
        for scene in REQUIRED_SCENE_COUNTS
    }
    for scene, expected in REQUIRED_SCENE_COUNTS.items():
        if views_per_scene[scene] != expected:
            violations.append(
                f"scene count mismatch for {scene}: expected {expected} views, "
                f"found {views_per_scene[scene]}")
    if len(views) != TOTAL_VIEWS_EXPECTED:
        violations.append(
            f"view count mismatch: expected {TOTAL_VIEWS_EXPECTED}, "
            f"found {len(views)}")
    if len(view_files) != TOTAL_CROP_FILES_EXPECTED:
        violations.append(
            f"crop file count mismatch: expected {TOTAL_CROP_FILES_EXPECTED}, "
            f"found {len(view_files)}")

    if violations:
        sample = violations[:20]
        raise CropCacheAuditError(
            f"Crop-cache audit failed with {len(violations)} violation(s); "
            f"first {len(sample)}:\n" + "\n".join(f"  - {v}" for v in sample))

    entries = [
        {"path": rel, "full": str(crop_cache / rel), "scene": scene,
         "frame": frame, "object_id": object_id, "kind": kind}
        for (scene, frame, object_id, kind), rel in sorted(
            view_files.items(), key=lambda item: item[1])
    ]
    return {
        "root": str(crop_cache),
        "files": len(entries),
        "views": len(views),
        "views_per_scene": views_per_scene,
        "entries": entries,
    }


def compute_crop_manifest(crop_cache: Path) -> dict[str, Any]:
    """Content-sensitive manifest over every file in the crop cache.

    Enumeration and layout validation are delegated to audit_crop_cache,
    so the identical routine used before inference can be recomputed
    after inference; the entry-wise (relative path, byte size, SHA-256)
    comparison then detects any file add, removal or replacement.
    """
    started = time.perf_counter()
    audit = audit_crop_cache(crop_cache)
    entries: list[dict[str, Any]] = []
    for item in audit["entries"]:
        digest, size = sha256_file(item["full"])
        entries.append({"path": item["path"], "bytes": size, "sha256": digest})
    total_bytes = sum(entry["bytes"] for entry in entries)
    manifest = {
        "schema_version": 1,
        "root": str(Path(crop_cache).resolve()),
        "files": len(entries),
        "bytes": total_bytes,
        "manifest_sha256": _manifest_digest(entries, len(entries), total_bytes),
        "manifest_digest_algorithm": (
            "sha256 of json.dumps({files, bytes, entries}, sort_keys=True, "
            "separators=(',',':')) over entries sorted by relative path"
        ),
        "views": audit["views"],
        "views_per_scene": audit["views_per_scene"],
        "entries": entries,
    }
    manifest["seconds"] = time.perf_counter() - started
    return manifest


def compare_with_baseline_manifest(manifest: dict[str, Any],
                                   baseline_path: Path) -> dict[str, Any]:
    """Entry-wise comparison against the pre-run baseline manifest.

    ``match_status`` is one of:
      baseline_absent   - no baseline file; nothing to compare
      baseline_unreadable - file unreadable or not valid JSON
      baseline_malformed  - valid JSON but structurally invalid (non-object
                            root, non-list entries, non-object entry,
                            invalid/empty path or sha256, non-integer or
                            negative bytes, or a duplicate relative path);
                            a malformed baseline can never be a match
      match / mismatch  - entry-wise (path, bytes, sha256) comparison
    """
    result: dict[str, Any] = {"baseline_manifest_path": str(baseline_path)}
    if not baseline_path.is_file():
        result["present"] = False
        result["match_status"] = "baseline_absent"
        return result
    result["present"] = True
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["match_status"] = "baseline_unreadable"
        result["error"] = f"Baseline manifest is unreadable: {exc}"
        return result
    if not isinstance(baseline, dict):
        result["match_status"] = "baseline_malformed"
        result["error"] = "Baseline manifest root is not a JSON object"
        return result
    result["baseline_manifest_sha256_field"] = baseline.get("manifest_sha256")
    result["baseline_files"] = baseline.get("files")
    result["baseline_bytes"] = baseline.get("bytes")
    base_entries = baseline.get("entries")
    if not isinstance(base_entries, list):
        result["match_status"] = "baseline_malformed"
        result["error"] = "Baseline manifest 'entries' is not a list"
        return result
    base_by_path: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(base_entries):
        if not isinstance(entry, dict):
            result["match_status"] = "baseline_malformed"
            result["error"] = f"Baseline manifest entry {index} is not an object"
            return result
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            result["match_status"] = "baseline_malformed"
            result["error"] = f"Baseline manifest entry {index} has an invalid 'path'"
            return result
        if path in base_by_path:
            result["match_status"] = "baseline_malformed"
            result["error"] = (
                f"Baseline manifest has a duplicate relative path: {path}")
            return result
        entry_bytes = entry.get("bytes")
        if (isinstance(entry_bytes, bool) or not isinstance(entry_bytes, int)
                or entry_bytes < 0):
            result["match_status"] = "baseline_malformed"
            result["error"] = (
                f"Baseline manifest entry {index} has an invalid 'bytes'")
            return result
        entry_sha = entry.get("sha256")
        if not isinstance(entry_sha, str) or not entry_sha:
            result["match_status"] = "baseline_malformed"
            result["error"] = (
                f"Baseline manifest entry {index} has an invalid 'sha256'")
            return result
        base_by_path[path] = entry
    mine_by_path = {entry["path"]: entry for entry in manifest["entries"]}
    missing = sorted(set(base_by_path) - set(mine_by_path))
    extra = sorted(set(mine_by_path) - set(base_by_path))
    content_mismatches = []
    for path in sorted(set(base_by_path) & set(mine_by_path)):
        if (base_by_path[path]["bytes"], base_by_path[path]["sha256"]) != (
                mine_by_path[path]["bytes"], mine_by_path[path]["sha256"]):
            content_mismatches.append(path)
    entrywise_match = not (missing or extra or content_mismatches)
    result.update({
        "baseline_entry_count": len(base_by_path),
        "computed_entry_count": len(mine_by_path),
        "missing_paths": missing[:20],
        "extra_paths": extra[:20],
        "content_mismatch_paths": content_mismatches[:20],
        "baseline_entrywise_match": entrywise_match,
        "file_count_match": baseline.get("files") == manifest["files"],
        "total_bytes_match": baseline.get("bytes") == manifest["bytes"],
        "aggregate_digest_equal_to_baseline_field": (
            manifest["manifest_sha256"] == baseline.get("manifest_sha256")
        ),
        "note": (
            "The root baseline manifest's aggregate-digest algorithm is not "
            "documented, so its manifest_sha256 field is recorded but not the "
            "match criterion; entry-wise equality over (path, bytes, sha256) "
            "is the authoritative content comparison."
        ),
        "match_status": ("match" if entrywise_match else "mismatch"),
    })
    return result


def _coerce_object_id(raw_id: Any) -> int:
    """Coerce an objects.json object ID to int; raise ValueError if malformed.

    Accepts ints, integral floats and integer digit strings (the reference
    ``int(row["id"])`` semantics); rejects booleans, non-integral floats
    and anything else.
    """
    if isinstance(raw_id, bool):
        raise ValueError(f"boolean object ID {raw_id!r}")
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, float):
        if not raw_id.is_integer():
            raise ValueError(f"non-integral float object ID {raw_id!r}")
        return int(raw_id)
    if isinstance(raw_id, str):
        text = raw_id.strip()
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
    raise ValueError(f"object ID {raw_id!r} is not an integer")


def _scene_object_rows(objects: Any) -> dict[str, list[Any]]:
    """Per-scene object rows for the five relevant scenes, strictly checked.

    Rejects a malformed top-level shape, a duplicate scan entry for a
    relevant scene, and a missing scan entry for a relevant scene.
    Scans outside the five benchmark scenes are ignored (ReplicaSSG
    annotates more scenes than this benchmark uses).
    """
    if not isinstance(objects, dict) or not isinstance(objects.get("scans"), list):
        raise RuntimeError("objects.json must be a JSON object with a 'scans' list")
    rows: dict[str, list[Any]] = {}
    for index, scan in enumerate(objects["scans"]):
        if not isinstance(scan, dict) or "scan" not in scan:
            raise RuntimeError(f"objects.json scans[{index}] has no scan name")
        name = str(scan["scan"])
        if name not in REQUIRED_SCENE_COUNTS:
            continue
        if name in rows:
            raise RuntimeError(f"objects.json has duplicate scene entries for {name}")
        rows[name] = scan.get("objects", [])
    for scene in REQUIRED_SCENE_COUNTS:
        if scene not in rows:
            raise RuntimeError(f"objects.json is missing a scene entry for {scene}")
    return rows


def _validate_label(raw_label: Any) -> str | None:
    """Normalize an objects.json label for the category join.

    Returns the stripped lower-case label, or None when the label is
    invalid for any crop referencing the object: the ``label`` key is
    missing, null, empty after stripping, or (case-insensitively)
    ``unknown`` or ``undefined``.
    """
    if raw_label is None:
        return None
    text = str(raw_label).strip().lower()
    if not text or text in INVALID_LABELS:
        return None
    return text


def _object_metadata(scene: str, scene_objects: Any) -> dict[int, str | None]:
    """Strict per-scene object ID -> label map.

    Joins labels with the same semantics as
    replica_clip_benchmark._object_metadata (integer id, stripped
    lower-case label, negative ids excluded) but fails loudly instead of
    silently overwriting or skipping rows: duplicate or malformed object
    IDs within the scene raise RuntimeError. Invalid labels (missing,
    null, empty, ``unknown`` or ``undefined``) are recorded as None and
    only fail when a crop references the affected object; unreferenced
    scene rows with such labels are tolerated.
    """
    if not isinstance(scene_objects, list):
        raise RuntimeError(f"objects.json scene {scene}: 'objects' is not a list")
    raw_labels: dict[int, tuple[str, int]] = {}
    for index, row in enumerate(scene_objects):
        if not isinstance(row, dict) or "id" not in row:
            raise RuntimeError(
                f"objects.json scene {scene}: object row {index} has no object ID")
        try:
            object_id = _coerce_object_id(row["id"])
        except ValueError:
            raise RuntimeError(
                f"objects.json scene {scene}: object row {index} has malformed "
                f"object ID {row['id']!r}") from None
        if object_id < 0:
            raise RuntimeError(
                f"objects.json scene {scene}: object row {index} has negative "
                f"object ID {object_id}")
        if object_id in raw_labels:
            raise RuntimeError(
                f"objects.json scene {scene}: duplicate object ID {object_id} "
                f"at rows {raw_labels[object_id][1]} and {index}")
        raw_labels[object_id] = (_validate_label(row.get("label")), index)
    return {object_id: label for object_id, (label, _) in raw_labels.items()}


def _select_interior_point(mask_foreground: Any) -> tuple[int, int, float]:
    """Deterministic interior point: max Euclidean distance to the boundary.

    ``scipy.ndimage.distance_transform_edt`` on the native-resolution
    foreground mask; ties resolved by row-major ``np.argmax`` (first max).
    Returns (y, x) in native pixels plus the distance.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    distances = distance_transform_edt(mask_foreground)
    flat_index = int(np.argmax(distances))
    height, width = distances.shape
    y = flat_index // width
    x = flat_index % width
    return y, x, float(distances.flat[flat_index])


def _map_point_to_512(y: int, x: int, height: int, width: int) -> list[int]:
    py = min(IMAGE_RESOLUTION - 1, max(0, int(round(y * IMAGE_RESOLUTION / height))))
    px = min(IMAGE_RESOLUTION - 1, max(0, int(round(x * IMAGE_RESOLUTION / width))))
    return [py, px]


def build_dataset(crop_cache: Path, objects_json: Path,
                  warnings: list[str]) -> tuple[list[dict[str, Any]], list[str],
                                                dict[str, int], int, int]:
    """Discover the frozen crop pairs, join categories, validate, select points.

    Returns the five-value contract:
      rows: one dict per GT object view, sorted by (scene, frame, object_id)
      categories: sorted joined labels (frozen 49-category check applies)
      joined_labels: label -> number of views joined to that label
      min_mask_pixels: smallest GT-mask foreground pixel count (int)
      max_mask_pixels: largest GT-mask foreground pixel count (int)

    Crop-cache layout validation (five scene directories, 4,408 direct
    crop files, names, box/mask pairs, counts, symlinks, nested
    directories) is delegated to audit_crop_cache and fails before any
    image is opened. Rejected here as well: duplicate/missing scene
    entries, duplicate or malformed object IDs, referenced IDs with
    missing, null, empty, ``unknown`` or ``undefined`` labels, unreadable
    images, empty masks, box/mask dimension mismatches, non-RGB box crops
    and any deviation from the frozen per-scene counts, 2,204 total views
    and the exact frozen 49-category set.
    """
    import numpy as np
    from PIL import Image

    crop_cache = Path(crop_cache)
    objects_json = Path(objects_json)
    if not objects_json.is_file():
        raise FileNotFoundError(f"Missing objects.json: {objects_json}")

    audit = audit_crop_cache(crop_cache)
    views: dict[tuple[str, int, int], dict[str, str]] = {}
    for entry in audit["entries"]:
        views.setdefault((entry["scene"], entry["frame"], entry["object_id"]),
                         {})[entry["kind"]] = entry["full"]

    objects = json.loads(objects_json.read_text(encoding="utf-8"))
    scene_object_rows = _scene_object_rows(objects)

    metadata_cache: dict[str, dict[int, str | None]] = {}
    rows: list[dict[str, Any]] = []
    joined_labels: dict[str, int] = {}
    min_mask_pixels: int | None = None
    max_mask_pixels: int | None = None
    for scene, frame, object_id in sorted(views):
        box_path = views[(scene, frame, object_id)]["box"]
        mask_path = views[(scene, frame, object_id)]["mask"]
        if scene not in metadata_cache:
            metadata_cache[scene] = _object_metadata(scene, scene_object_rows[scene])
        metadata = metadata_cache[scene]
        if object_id not in metadata:
            raise RuntimeError(
                f"Category join failure: {scene}/frame-{frame:04d}/object-"
                f"{object_id} is referenced by the crop cache but missing "
                f"from objects.json")
        label = metadata[object_id]
        if label is None:
            raise RuntimeError(
                f"Category join failure: {scene}/frame-{frame:04d}/object-"
                f"{object_id} has a missing, null, empty, 'unknown' or "
                f"'undefined' label in objects.json")
        joined_labels[label] = joined_labels.get(label, 0) + 1

        try:
            with Image.open(mask_path) as mask_image:
                mask_array = np.asarray(mask_image.convert("L"))
        except Exception as exc:
            raise RuntimeError(f"Unreadable GT mask image {mask_path}: {exc}") from exc
        if mask_array.ndim != 2:
            raise RuntimeError(
                f"GT mask is not a 2D image for {scene}/frame-{frame:04d}/object-"
                f"{object_id}: shape {mask_array.shape}")
        foreground = mask_array > 0
        mask_pixels = int(foreground.sum())
        if mask_pixels == 0:
            raise RuntimeError(
                f"Empty GT mask for {scene}/frame-{frame:04d}/object-{object_id}")
        min_mask_pixels = (mask_pixels if min_mask_pixels is None
                           else min(min_mask_pixels, mask_pixels))
        max_mask_pixels = (mask_pixels if max_mask_pixels is None
                           else max(max_mask_pixels, mask_pixels))

        try:
            with Image.open(box_path) as box_image:
                box_mode = box_image.mode
                box_size = box_image.size
        except Exception as exc:
            raise RuntimeError(f"Unreadable box crop image {box_path}: {exc}") from exc
        if box_mode != "RGB":
            raise RuntimeError(
                f"Box crop is not RGB (mode {box_mode!r}) for "
                f"{scene}/frame-{frame:04d}/object-{object_id}")
        expected_size = (int(mask_array.shape[1]), int(mask_array.shape[0]))
        if tuple(box_size) != expected_size:
            raise RuntimeError(
                f"Box/mask size mismatch for {scene}/frame-{frame:04d}/object-"
                f"{object_id}: box {tuple(box_size)} vs mask {expected_size}")

        y, x, distance = _select_interior_point(foreground)
        if distance < 0.5:
            warnings.append(
                f"degenerate interior point (distance {distance:.3f}) for "
                f"{scene}/frame-{frame:04d}/object-{object_id}: mask has no "
                f"interior margin; first max-distance pixel used")
        rows.append({
            "scene": scene, "frame": frame, "object_id": object_id, "label": label,
            "box_path": box_path, "mask_path": mask_path,
            "native_size": [int(mask_array.shape[1]), int(mask_array.shape[0])],
            "point_native": [int(y), int(x)],
            "point_distance": distance,
            "point_512": _map_point_to_512(y, x, mask_array.shape[0],
                                           mask_array.shape[1]),
            "mask_foreground_pixels": mask_pixels,
        })

    categories = sorted(joined_labels)
    if len(categories) != NUM_CATEGORIES_EXPECTED:
        raise RuntimeError(
            f"Category count mismatch: expected {NUM_CATEGORIES_EXPECTED}, "
            f"found {len(categories)}: {categories}")
    missing = sorted(FROZEN_CATEGORIES - set(categories))
    extra = sorted(set(categories) - FROZEN_CATEGORIES)
    if missing or extra:
        raise RuntimeError(
            "Category set mismatch against the frozen 49-category set: "
            f"missing={missing} extra={extra}")
    return rows, categories, joined_labels, min_mask_pixels, max_mask_pixels


class _SklearnStubLoader:
    """Loader metadata for the synthetic sklearn stub modules.

    The stub modules are placed directly into ``sys.modules``; nothing is
    ever loaded from disk. The loader exists so the stub modules carry a
    well-formed import spec: ``importlib.util.find_spec`` on a
    ``sys.modules`` entry returns that entry's ``__spec__`` and raises
    ``ValueError: <name>.__spec__ is None`` when the spec is ``None`` or
    missing. That is the crash mode the stub must not leave behind —
    torch's dependency discovery (Dynamo) issues ``find_spec("sklearn")``
    while torch/torchvision are imported. ``create_module`` returning
    ``None`` selects the default module creation; ``exec_module`` is a
    no-op because the stub modules are pre-populated and never executed.
    """

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        return None


def install_sklearn_pca_stub() -> dict[str, Any]:
    """Fail-loud stub for sklearn.decomposition.PCA when sklearn is absent.

    T-REN task_utils.py imports PCA at module level but only the unused
    visualize_features helper uses it; the region-text inference path never
    touches it.

    Activation is strict: the stub installs only when the import probe
    fails with a ``ModuleNotFoundError`` whose ``exc.name`` is a
    nonempty string whose first dotted component is exactly
    ``sklearn`` (i.e. sklearn is genuinely unavailable). A missing or
    empty ``exc.name`` does not establish that sklearn is absent, and
    any other import failure — an installed-but-broken sklearn, or a
    missing dependency of an installed sklearn — propagates loudly
    instead of being masked by the stub. When an installed sklearn is
    already present in ``sys.modules`` with a real spec, the stub
    refuses to overlay it.

    The synthetic modules carry valid import metadata (a real
    ``importlib.machinery.ModuleSpec`` on ``__spec__``, plus ``__loader__``,
    ``__package__`` and the package ``__path__``), so normal dependency
    discovery — ``importlib.util.find_spec("sklearn")`` and
    ``find_spec("sklearn.decomposition")`` — returns a valid
    package/module spec. Constructing or invoking the stubbed ``PCA``
    still raises a clear ``RuntimeError``.
    """
    try:
        import sklearn.decomposition  # noqa: F401
        return {"stub_used": False,
                "note": "sklearn importable; no stub installed"}
    except ModuleNotFoundError as exc:
        # Only a missing sklearn (or one of its submodules) may be
        # stubbed. The gate is strict: ``exc.name`` must be a nonempty
        # string whose first dotted component is exactly ``sklearn``.
        # A missing or empty ``exc.name`` does not prove that sklearn
        # itself is absent, and a name belonging to any other module
        # means an installed sklearn is present but cannot be imported
        # (broken install, or a missing dependency of sklearn itself);
        # in every such case the error is re-raised so the real problem
        # surfaces instead of being masked by the stub.
        missing_name = getattr(exc, "name", None)
        if not isinstance(missing_name, str) or not missing_name:
            raise
        if missing_name.split(".")[0] != "sklearn":
            raise

        import importlib.machinery
        import types

        class _StubPCA:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "sklearn.decomposition.PCA stub: sklearn is not installed "
                    "in the runner environment and PCA must never be used by "
                    "the T-REN region-text inference path")

        def _make_stub_module(name: str, is_package: bool) -> types.ModuleType:
            """Create a stub module with valid import metadata.

            ``importlib.util.find_spec`` on a ``sys.modules`` entry
            returns the entry's ``__spec__`` and raises
            ``ValueError: <name>.__spec__ is None`` when the spec is
            ``None`` or missing, so every synthetic module needs a real
            ``ModuleSpec`` (with a loader) and, for the package, a
            search path (``__path__``; the empty path marks it closed —
            nothing is discovered beyond the pre-registered submodules).
            """
            loader = _SklearnStubLoader()
            spec = importlib.machinery.ModuleSpec(
                name, loader,
                origin="t-ren-replica-runner-sklearn-stub",
                is_package=is_package)
            if is_package:
                spec.submodule_search_locations = []
            module = types.ModuleType(name)
            module.__spec__ = spec
            module.__loader__ = loader
            if is_package:
                module.__package__ = name
                module.__path__ = []
            else:
                module.__package__ = name.rsplit(".", 1)[0]
            return module

        sklearn_mod = sys.modules.get("sklearn")
        if sklearn_mod is not None and getattr(
                sklearn_mod, "__spec__", None) is not None:
            # An installed sklearn package is present in sys.modules
            # while its decomposition submodule failed to import: a
            # broken install. Refuse to mask it with the stub.
            raise RuntimeError(
                "sklearn is present in sys.modules with a real import "
                "spec but 'import sklearn.decomposition' failed; refusing "
                "to install the fail-loud PCA stub over an installed "
                "sklearn (repair or remove the sklearn installation)")
        if sklearn_mod is None:
            sklearn_mod = _make_stub_module("sklearn", is_package=True)
            sys.modules["sklearn"] = sklearn_mod
        decomp_mod = _make_stub_module("sklearn.decomposition",
                                       is_package=False)
        decomp_mod.PCA = _StubPCA
        sys.modules["sklearn.decomposition"] = decomp_mod
        sklearn_mod.decomposition = decomp_mod
        return {
            "stub_used": True,
            "reason": "sklearn is not installed in the runner venv",
            "note": ("PCA is imported at module level by T-REN "
                     "task_utils.py but only used by the unused "
                     "visualize_features helper; the stub raises loudly "
                     "if it is ever instantiated. The stub modules carry "
                     "valid import specs so "
                     "importlib.util.find_spec('sklearn') and "
                     "find_spec('sklearn.decomposition') (issued by "
                     "torch dependency discovery during torchvision "
                     "imports) return a spec instead of raising "
                     "ValueError: sklearn.__spec__ is None"),
        }


# ---------------------------------------------------------------------------
# Model-framework setup (torch is imported lazily, only in this section)
# ---------------------------------------------------------------------------

def _library_versions() -> dict[str, str]:
    """Version strings for every library the run depends on (evidence)."""
    import importlib

    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("torch", "numpy", "scipy", "PIL", "kornia", "torchvision",
                 "yaml", "matplotlib", "ftfy", "regex"):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    return versions


def resolve_asset_paths(assets_dir: Path) -> dict[str, Path]:
    """Strictly resolve the four asset files from --assets-dir only."""
    assets_dir = Path(assets_dir)
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"Missing assets directory: {assets_dir}")
    resolved: dict[str, Path] = {}
    for role, filename in ASSET_FILES.items():
        path = assets_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing asset file ({assets_dir} is the only source "
                f"consulted; no fallback to torch hub caches): {path}")
        resolved[role] = path.resolve()
    return resolved


def install_hub_routing(dinov3_repo: Path,
                        assets: dict[str, Path]) -> dict[str, Any]:
    """Route the DINOv3 entry to the official local factory, wrapping
    torch.hub.load.

    T-REN's model.py hardcodes
    ``torch.hub.load('facebookresearch/dinov3', DINOTXT_ENTRY, ...)``.
    Loading the pinned checkout through its ``hubconf.py`` fails in the
    runner environment: the hubconf re-exports every DINOv3 hub module
    (backbones, classifiers, depthers, detectors, dinotxt, segmentors)
    at import time, and the segmentation modules import
    ``torchmetrics``, which is not installed. None of those sibling
    modules are needed for the single entry, so instead of executing
    ``hubconf.py`` the wrapper imports the ``dinov3`` package from the
    pinned local checkout and calls the exact official factory function
    ``dinov3.hub.dinotxt.dinov3_vitl16_dinotxt_tet1280d20h24l``
    directly. That module imports only torch plus the local
    ``dinov3.hub.backbones`` / ``dinov3.hub.utils`` helpers at module
    level (the ``dinov3.eval.text`` components are imported lazily
    inside the factory body), so no hubconf sibling module is ever
    imported.

    The factory is keyword-only (``pretrained``, ``weights``,
    ``backbone_weights``, ``bpe_path_or_url``, ``check_hash``) and
    returns the official ``(model, tokenizer)`` pair, which the wrapper
    returns unchanged. For a call to the target entry the wrapper
    (identified by repo and entry name) then:
      - rejects positional arguments (the factory takes none);
      - strips the Torch Hub routing options in
        ``HUB_ROUTING_OPTIONS`` that the factory cannot accept and
        records them;
      - keeps every legitimate factory keyword the caller passed (e.g.
        T-REN's ``backbone_weights`` / ``weights``);
      - setdefault-injects only the verified local
        ``backbone_weights`` / ``weights`` / ``bpe_path_or_url`` assets
        (T-REN never passes the BPE path, and the factory defaults for
        the backbone weights and BPE point at network URLs) plus
        ``pretrained=True``;
      - fails loudly on any other keyword the factory does not accept.
    Every non-target ``torch.hub.load`` call (any other repo or entry)
    passes through unchanged to the original ``torch.hub.load`` with
    its original arguments.

    The imported ``dinov3`` package and the factory module are verified
    to resolve inside the pinned checkout (symlinks followed); a
    ``dinov3`` package already imported from any other installation is
    refused with a loud error, so no other installation is ever used.
    The checkouts themselves stay byte-identical and unmodified.
    """
    import importlib
    import inspect

    import torch

    if getattr(torch.hub, "_t_ren_replica_hub_wrapped", False):
        return {"installed": False, "note": "hub routing already installed"}

    original_load = torch.hub.load
    repo_resolved = Path(dinov3_repo).resolve()
    if not repo_resolved.is_dir():
        raise RuntimeError(
            f"Missing pinned DINOv3 checkout directory: {repo_resolved}")
    if not (repo_resolved / "dinov3" / "hub" / "dinotxt.py").is_file():
        raise RuntimeError(
            "Pinned DINOv3 checkout is missing dinov3/hub/dinotxt.py: "
            f"{repo_resolved / 'dinov3' / 'hub' / 'dinotxt.py'}")

    def _resolved_location(module) -> str | None:
        """Resolved filesystem location of an imported module or package.

        Uses ``__file__`` for modules and the first ``__path__`` entry
        for packages; symlinks are followed via ``Path.resolve``.
        """
        file_attr = getattr(module, "__file__", None)
        if file_attr:
            return str(Path(file_attr).resolve())
        path_attr = getattr(module, "__path__", None)
        if path_attr:
            first = next(iter(path_attr), None)
            if first:
                return str(Path(first).resolve())
        return None

    def _require_inside_checkout(location: str | None, what: str) -> None:
        if location is None:
            raise RuntimeError(
                f"could not resolve the filesystem location of {what}; "
                f"refusing to route the DINOv3 entry without verifying "
                f"the pinned checkout {repo_resolved}")
        if not Path(location).is_relative_to(repo_resolved):
            raise RuntimeError(
                f"{what} resolved to {location!r}, which is outside the "
                f"pinned local DINOv3 checkout {repo_resolved!r}; another "
                f"installation is in use and will not be used")

    # --- resolve the dinov3 package from the pinned checkout only -----
    existing = sys.modules.get("dinov3")
    if existing is not None:
        _require_inside_checkout(
            _resolved_location(existing),
            "the already-imported 'dinov3' package")
        dinov3_package = existing
    else:
        repo_str = str(repo_resolved)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        try:
            dinov3_package = importlib.import_module("dinov3")
        except Exception as exc:
            raise RuntimeError(
                f"failed to import the 'dinov3' package from the pinned "
                f"local checkout {repo_resolved}: {exc}") from exc
        _require_inside_checkout(
            _resolved_location(dinov3_package),
            "the imported 'dinov3' package")

    try:
        factory_module = importlib.import_module("dinov3.hub.dinotxt")
    except Exception as exc:
        raise RuntimeError(
            f"failed to import dinov3.hub.dinotxt from the pinned local "
            f"checkout {repo_resolved}: {exc}") from exc
    _require_inside_checkout(
        _resolved_location(factory_module), "the dinov3.hub.dinotxt module")
    factory = getattr(factory_module, DINOTXT_ENTRY, None)
    if not callable(factory):
        raise RuntimeError(
            f"dinov3.hub.dinotxt in the pinned checkout {repo_resolved} "
            f"does not define a callable {DINOTXT_ENTRY!r} factory")
    factory_params = inspect.signature(factory).parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD
           for param in factory_params.values()):
        raise RuntimeError(
            f"the {DINOTXT_ENTRY!r} factory in {repo_resolved} accepts "
            f"**kwargs; the unsupported-keyword gate would be defeated")
    accepted_kwargs = set(factory_params)
    injected_assets = {
        "backbone_weights": str(assets["backbone_weights"]),
        "weights": str(assets["dinotxt_head_and_text_encoder"]),
        "bpe_path_or_url": str(assets["bpe_vocab"]),
    }
    for role in injected_assets:
        if role not in accepted_kwargs:
            raise RuntimeError(
                f"the {DINOTXT_ENTRY!r} factory in {repo_resolved} no "
                f"longer accepts the {role!r} keyword; local asset "
                f"injection would be impossible")

    target_calls: list[dict[str, Any]] = []

    def _direct_factory_call(args, kwargs):
        if args:
            raise RuntimeError(
                f"torch.hub.load call for the {DINOTXT_ENTRY!r} entry "
                f"passed positional argument(s) {args!r}; the direct "
                f"factory is keyword-only and accepts no positional "
                f"arguments")
        stripped = {key: kwargs[key] for key in sorted(kwargs)
                    if key in HUB_ROUTING_OPTIONS}
        factory_kwargs = {key: value for key, value in kwargs.items()
                          if key not in HUB_ROUTING_OPTIONS}
        unsupported = sorted(set(factory_kwargs) - accepted_kwargs)
        if unsupported:
            raise RuntimeError(
                f"torch.hub.load call for the {DINOTXT_ENTRY!r} entry "
                f"passed unsupported keyword argument(s) {unsupported}; "
                f"the official factory accepts only "
                f"{sorted(accepted_kwargs)} (the Torch Hub routing "
                f"options {sorted(HUB_ROUTING_OPTIONS)} are stripped)")
        factory_kwargs.setdefault("pretrained", True)
        for role, value in injected_assets.items():
            factory_kwargs.setdefault(role, value)
        result = factory(**factory_kwargs)
        if not (isinstance(result, tuple) and len(result) == 2):
            raise RuntimeError(
                f"the {DINOTXT_ENTRY!r} factory did not return the "
                f"official (model, tokenizer) pair; got {type(result)!r}")
        model, tokenizer = result
        target_calls.append({
            "stripped_hub_routing_options": stripped,
            "factory_kwargs": factory_kwargs,
            "returned": ["model", "tokenizer"],
            "model_type": type(model).__name__,
            "tokenizer_type": type(tokenizer).__name__,
        })
        return model, tokenizer

    def _hub_load_wrapper(repo_or_dir, model, *args, **kwargs):
        if (repo_or_dir == "facebookresearch/dinov3"
                and model == DINOTXT_ENTRY):
            return _direct_factory_call(args, kwargs)
        return original_load(repo_or_dir, model, *args, **kwargs)

    _hub_load_wrapper._t_ren_replica_hub_wrapped = True
    torch.hub.load = _hub_load_wrapper
    return {
        "installed": True,
        "routing_mode": "direct_local_factory (hubconf.py never loaded)",
        "wrapped_repo": "facebookresearch/dinov3",
        "wrapped_entry": DINOTXT_ENTRY,
        "local_dinov3_repo": str(repo_resolved),
        "dinov3_package_path": _resolved_location(dinov3_package),
        "factory_module": factory_module.__name__,
        "factory_module_path": _resolved_location(factory_module),
        "factory_function": DINOTXT_ENTRY,
        "factory_accepted_kwargs": sorted(accepted_kwargs),
        "hub_routing_options_stripped": sorted(HUB_ROUTING_OPTIONS),
        "injected_assets": injected_assets,
        "target_calls": target_calls,
        "note": (
            "Only torch.hub.load calls for the entry (repo "
            "facebookresearch/dinov3, entry "
            f"{DINOTXT_ENTRY}) are routed: the official factory "
            f"dinov3.hub.dinotxt.{DINOTXT_ENTRY} of the pinned local "
            "checkout is imported and called directly (the broad "
            "hubconf.py, whose segmentation modules require the absent "
            "torchmetrics, is never loaded); the factory's "
            "(model, tokenizer) return is passed through unchanged and "
            "the local backbone/head/BPE assets are setdefault-injected. "
            "Every other torch.hub.load call passes through unchanged "
            "to the original torch.hub.load with its original "
            "arguments."),
    }


# Top-level modules the T-REN region-text path requires at import time,
# with their distribution names for clear error messages. sklearn is
# deliberately absent: its missing case is covered by the fail-loud PCA
# stub installed before importing T-REN model.py.
_RUNTIME_DEPENDENCIES = (
    ("torch", "torch"),
    ("kornia", "kornia"),
    ("torchvision", "torchvision"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("PIL", "pillow"),
    ("yaml", "pyyaml"),
)


def check_runtime_dependencies() -> dict[str, Any]:
    """Hard preflight dependency gate (imports nothing).

    Uses ``importlib.util.find_spec`` per required top-level module, so a
    missing dependency (e.g. kornia, which T-REN's model.py requires at
    import time) fails at preflight with a clear message instead of deep
    inside the model load. Raises RuntimeError listing every missing
    module when any is absent; returns a per-module status report for the
    evidence on success.
    """
    # ``import importlib`` alone does not bind the ``importlib.util``
    # submodule, so it must be imported explicitly; a bare ``import
    # importlib`` made ``importlib.util.find_spec`` raise AttributeError
    # for every module and produced a false missing-dependency report.
    import importlib.util

    report: dict[str, Any] = {}
    missing: list[str] = []
    for import_name, distribution in _RUNTIME_DEPENDENCIES:
        try:
            spec = importlib.util.find_spec(import_name)
        except Exception as exc:
            report[import_name] = f"find_spec raised: {exc}"
            missing.append(distribution)
            continue
        if spec is None:
            report[import_name] = "not found"
            missing.append(distribution)
        else:
            report[import_name] = "ok"
    if missing:
        raise RuntimeError(
            "Missing runtime dependency(ies) for the T-REN region-text "
            f"path: {', '.join(missing)}; install them in the runner "
            "environment before running the benchmark (sklearn is "
            "intentionally not gated: the fail-loud PCA stub covers its "
            "absence)")
    report["sklearn_note"] = (
        "sklearn is not gated: its absence is covered by the fail-loud "
        "PCA stub installed before importing T-REN model.py")
    return report


def _load_tren_config(t_ren_repo: Path,
                      assets_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the frozen T-REN yaml and aim its logging paths at the assets dir.

    T-REN's model.py builds asset paths as
    ``os.path.join(config['logging']['save_dir'], config['logging']['exp_name'],
    <filename>)``; overriding save_dir/exp_name to (assets_dir.parent + sep,
    assets_dir.name) makes every one of those paths hit the assets
    directory. Frozen architecture values are asserted so a drifted yaml
    fails loudly instead of silently changing the model.
    """
    import yaml

    config_path = Path(t_ren_repo) / "configs" / "train_dinov3_vitl16.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing T-REN config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("logging"), dict):
        raise RuntimeError(f"T-REN config must be a YAML mapping: {config_path}")
    assets_resolved = Path(assets_dir).resolve()
    config["logging"]["save_dir"] = str(assets_resolved.parent) + os.sep
    config["logging"]["exp_name"] = assets_resolved.name
    frozen_checks = {
        ("pretrained", "feature_extractor"): "dinov3_vitl16",
        ("architecture", "patch_size"): PATCH_SIZE,
        ("architecture", "text_embed_dim"): TEXT_EMBED_DIM_EXPECTED,
        ("parameters", "image_resolution"): IMAGE_RESOLUTION,
    }
    for (section, key), expected in frozen_checks.items():
        try:
            actual = config[section][key]
        except (KeyError, TypeError):
            raise RuntimeError(f"T-REN config drift: missing {section}.{key}") from None
        if actual != expected:
            raise RuntimeError(
                f"T-REN config drift: {section}.{key}={actual!r}, "
                f"expected {expected!r}")
    identity = {
        "config_path": str(config_path),
        "frozen_values": {f"{section}.{key}": expected
                          for (section, key), expected in frozen_checks.items()},
        "logging_save_dir": config["logging"]["save_dir"],
        "logging_exp_name": config["logging"]["exp_name"],
        "num_multiscale_regions": config["parameters"]["num_multiscale_regions"],
        "merging_iou_threshold": config["parameters"]["merging_iou_threshold"],
        "merging_similarity_threshold": (
            config["parameters"]["merging_similarity_threshold"]),
        "num_decoder_layers": config["architecture"]["num_decoder_layers"],
        "num_attention_heads": config["architecture"]["num_attention_heads"],
        "hidden_dim": config["architecture"]["hidden_dim"],
    }
    return config, identity


def _require_cuda() -> "torch.device":
    """CUDA + FP32 gate; fails loudly with no CPU fallback."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this benchmark and no CPU fallback is "
            "allowed; torch.cuda.is_available() returned False")
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError(
            f"FP32 default dtype is required, got {torch.get_default_dtype()}")
    return torch.device("cuda", 0)


def _check_fp32_parameters(module: Any, name: str) -> None:
    """Assert every parameter of an official component is float32."""
    import torch

    non_fp32 = [
        f"{name}.{param_name}"
        for param_name, param in module.named_parameters()
        if param.dtype != torch.float32
    ]
    if non_fp32:
        raise RuntimeError(
            f"{name} has non-FP32 parameters (CUDA FP32 is required): "
            f"{non_fp32[:5]}")


def load_tren_components(config: dict[str, Any], device: Any,
                         t_ren_repo: Path, dinov3_repo: Path,
                         assets: dict[str, Path],
                         warnings: list[str]) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Instantiate the official T-REN components on CUDA in FP32.

    Composes the three official components from T-REN's model.py exactly as
    tren.py does (FeatureExtractor, RegionEncoder, TextEncoder) rather than
    importing tren.py, which would additionally pull in its 128x128 grid and
    extra top-level dependencies. The hub-routing wrapper is installed
    before the component constructors run (the constructors issue the
    intercepted torch.hub.load calls; the wrapper routes the DINOv3 entry
    to the pinned checkout's official dinov3.hub.dinotxt factory, whose
    own weight loader is used unmodified). The RegionEncoder is loaded from
    the official checkpoint's ``ren_region_encoder_state`` with
    ``load_state_dict_resolution_agnostic`` at strict=True so any missing or
    unexpected key fails loudly; the exact state key list and the
    strict-load missing/unexpected key lists are captured for the
    evidence.
    """
    import torch

    t_ren_repo = Path(t_ren_repo)
    if not (t_ren_repo / "model.py").is_file():
        raise FileNotFoundError(f"T-REN checkout missing model.py: {t_ren_repo}")
    if not (t_ren_repo / "task_utils.py").is_file():
        raise FileNotFoundError(
            f"T-REN checkout missing task_utils.py: {t_ren_repo}")

    repo_dir = str(t_ren_repo)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    # T-REN model.py imports task_utils, which imports sklearn at module
    # level; install the fail-loud PCA stub first when sklearn is absent.
    pca_stub = install_sklearn_pca_stub()
    if pca_stub.get("stub_used"):
        warnings.append(
            "sklearn is absent in the runner venv; the fail-loud PCA stub "
            "was installed before importing T-REN model.py")

    import model as t_ren_model  # T-REN flat top-level module

    load_info: dict[str, Any] = {
        "pca_stub": pca_stub,
        # Must be installed before any component constructor runs, since
        # the constructors issue the intercepted torch.hub.load calls.
        # (The wrapper imports dinov3 from the pinned checkout before any
        # model construction; no hubconf.py is ever loaded.)
        "hub_routing": install_hub_routing(dinov3_repo, assets),
    }

    init_started = time.perf_counter()
    torch.cuda.synchronize(device)
    feature_extractor = t_ren_model.FeatureExtractor(config, device).eval()
    region_encoder = t_ren_model.RegionEncoder(config).to(device).eval()
    text_encoder = t_ren_model.TextEncoder(
        config, device=device, prompt=TEXT_ENCODER_PROMPT).eval()
    checkpoint_path = assets["tren_region_encoder_checkpoint"]
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or (
            "ren_region_encoder_state" not in checkpoint):
        raise RuntimeError(
            f"T-REN checkpoint {checkpoint_path} does not contain "
            f"'ren_region_encoder_state'; keys: "
            f"{sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)!r}")
    ren_state = checkpoint["ren_region_encoder_state"]
    if not isinstance(ren_state, dict) or not ren_state:
        raise RuntimeError(
            f"T-REN checkpoint {checkpoint_path}: 'ren_region_encoder_state' "
            f"must be a non-empty state dict, got {type(ren_state)!r}")
    load_result = region_encoder.load_state_dict_resolution_agnostic(
        ren_state, strict=True)
    if (not hasattr(load_result, "missing_keys")
            or not hasattr(load_result, "unexpected_keys")):
        raise RuntimeError(
            f"Unexpected result from load_state_dict_resolution_agnostic: "
            f"{load_result!r}")
    missing_keys = sorted(getattr(load_result, "missing_keys") or [])
    unexpected_keys = sorted(getattr(load_result, "unexpected_keys") or [])
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Strict T-REN RegionEncoder checkpoint load was not clean "
            f"(strict=True): missing_keys={missing_keys} "
            f"unexpected_keys={unexpected_keys}")
    torch.cuda.synchronize(device)
    load_info["init_seconds"] = time.perf_counter() - init_started
    load_info["checkpoint_path"] = str(checkpoint_path)
    load_info["checkpoint_state_key"] = "ren_region_encoder_state"
    load_info["checkpoint_state_key_count"] = len(ren_state)
    load_info["checkpoint_state_keys"] = sorted(str(key) for key in ren_state)
    load_info["checkpoint_missing_keys"] = missing_keys
    load_info["checkpoint_unexpected_keys"] = unexpected_keys
    load_info["checkpoint_load_strict"] = True
    load_info["checkpoint_other_state_keys_present"] = sorted(
        key for key in checkpoint if key != "ren_region_encoder_state")

    for module, name in ((feature_extractor, "FeatureExtractor"),
                         (region_encoder, "RegionEncoder"),
                         (text_encoder, "TextEncoder")):
        _check_fp32_parameters(module, name)

    device_properties = torch.cuda.get_device_properties(device)
    load_info["device_name"] = device_properties.name
    load_info["device_total_gib"] = round(
        device_properties.total_memory / (1024 ** 3), 3)
    load_info["torch_cuda_build"] = torch.version.cuda
    return feature_extractor, region_encoder, text_encoder, load_info


def run_region_inference(feature_extractor: Any, region_encoder: Any,
                         rows: list[dict[str, Any]], text_features: Any,
                         device: Any, batch_size: int) -> tuple[Any, dict[str, Any]]:
    """Score every view with the official region-text path (CUDA, FP32).

    Per view: the box crop is opened (PIL, RGB), bilinearly resized to
    512x512, scaled to float32 [0, 1] and fed to the official
    FeatureExtractor (which applies the official ImageNet normalization and
    CenterPadding itself); the deterministic GT point (512-space) is passed
    to the official RegionEncoder with ``aggregate_tokens=True`` and
    ``remove_singleton_groups=True``; the per-view score is the max cosine
    between the view's aggregated text-aligned region tokens and the 49
    normalized DINOtxt query vectors. No autocast is used: the run is FP32.
    Returns (scores np.ndarray of shape (views, categories), inference_info).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    num_categories = int(text_features.shape[0])
    scores = np.zeros((len(rows), num_categories), dtype=np.float64)
    text_features_norm = F.normalize(text_features.float(), dim=-1)

    cold_seconds = 0.0
    warm_seconds = 0.0
    feature_seconds = 0.0
    region_seconds = 0.0
    cold_views = 0
    warm_views = 0
    token_counts: list[int] = []

    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        is_cold = start == 0
        images = torch.zeros(
            (len(chunk), 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION),
            dtype=torch.float32)
        for i, row in enumerate(chunk):
            try:
                with Image.open(row["box_path"]) as image:
                    image = image.convert("RGB").resize(
                        (IMAGE_RESOLUTION, IMAGE_RESOLUTION),
                        Image.BILINEAR)
                    array = np.asarray(image, dtype=np.float32) / 255.0
            except Exception as exc:
                raise RuntimeError(
                    f"Unreadable box crop during inference "
                    f"{row['box_path']}: {exc}") from exc
            images[i] = torch.from_numpy(array.transpose(2, 0, 1))
        points = torch.tensor(
            [row["point_512"] for row in chunk],
            dtype=torch.long).unsqueeze(1).to(device)

        with torch.inference_mode():
            torch.cuda.synchronize(device)
            t_feature_start = time.perf_counter()
            feature_maps = feature_extractor(images)["feature_maps"]
            torch.cuda.synchronize(device)
            t_feature_end = time.perf_counter()
            ren = region_encoder(
                feature_maps, points, aggregate_tokens=True,
                remove_singleton_groups=True)
            torch.cuda.synchronize(device)
            t_region_end = time.perf_counter()

        feature_seconds += t_feature_end - t_feature_start
        region_seconds += t_region_end - t_feature_end
        chunk_elapsed = t_region_end - t_feature_start
        if is_cold:
            cold_seconds += chunk_elapsed
            cold_views += len(chunk)
        else:
            warm_seconds += chunk_elapsed
            warm_views += len(chunk)

        for i, row in enumerate(chunk):
            tokens = ren["text_aligned_tokens"][i]
            if tokens.shape[0] < 1 or tokens.shape[-1] != TEXT_EMBED_DIM_EXPECTED:
                raise RuntimeError(
                    f"Unexpected text-aligned token shape {tuple(tokens.shape)} "
                    f"for {row['scene']}/frame-{row['frame']:04d}/object-"
                    f"{row['object_id']} (expected (n>=1, "
                    f"{TEXT_EMBED_DIM_EXPECTED}))")
            token_counts.append(int(tokens.shape[0]))
            # tokens (n_tokens, 1024) @ text_features_norm.T (1024, C) gives
            # (n_tokens, C); max over the token dim returns a
            # (values (C,), indices (C,)) namedtuple, so .values is the
            # per-view score row over the C categories.
            view_scores = (F.normalize(tokens.float(), dim=-1)
                           @ text_features_norm.t()).max(dim=0).values
            scores[start + i] = view_scores.cpu().numpy()

    total_seconds = feature_seconds + region_seconds
    info: dict[str, Any] = {
        "views": len(rows),
        "batch_size": batch_size,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "cold_views": cold_views,
        "warm_views": warm_views,
        "feature_seconds": feature_seconds,
        "region_seconds": region_seconds,
        "total_inference_seconds": total_seconds,
        "cold_views_per_second": (cold_views / cold_seconds
                                  if cold_seconds > 0 else float("nan")),
        "warm_views_per_second": (warm_views / warm_seconds
                                  if warm_seconds > 0 else float("nan")),
        "overall_views_per_second": (len(rows) / total_seconds
                                     if total_seconds > 0 else float("nan")),
        "token_counts_min": min(token_counts),
        "token_counts_max": max(token_counts),
        "token_counts_mean": sum(token_counts) / len(token_counts),
        "note": ("cold = first inference batch (includes CUDA warm-up); "
                 "warm = remaining batches; feature/region are the "
                 "synchronized wall-clock seconds inside "
                 "torch.inference_mode over all batches; no autocast "
                 "(FP32 run)"),
    }
    return scores, info


def _average_precision(scores: np.ndarray, relevant: np.ndarray) -> float:
    import numpy as np
    # Mirrors conceptgraph/scripts/clip_component.py::_average_precision
    # (stable argsort ranking; AP undefined -> NaN for empty relevant sets).
    order = np.argsort(-scores, kind="stable")
    ranked = relevant[order].astype(np.float64)
    cumulative = np.cumsum(ranked)
    precision_at_relevant = cumulative[ranked.astype(bool)] / (np.flatnonzero(ranked) + 1)
    return float(precision_at_relevant.sum() / relevant.sum()) if relevant.any() else float("nan")


def _score_matrix(similarities: np.ndarray, targets: np.ndarray,
                  category_names: list[str]) -> dict[str, Any]:
    """Retrieval metrics over a (views, categories) similarity matrix.

    Metric bodies mirror conceptgraph/scripts/clip_component.py::_score
    (top-1/top-5 via argpartition kth=4, per-category stable-AP and
    recall@1/5/10/50, macro over the categories present in the views).
    """
    import numpy as np
    predictions = similarities.argmax(axis=1)
    topk_count = min(5, similarities.shape[1])
    topk = np.argpartition(-similarities, kth=topk_count - 1, axis=1)[:, :topk_count]
    top1 = predictions == targets
    top5 = np.any(topk == targets[:, None], axis=1)
    present = np.unique(targets)
    per_category: dict[str, Any] = {}
    aps: list[float] = []
    retrieval_r1, retrieval_r5, retrieval_r10, retrieval_r50 = [], [], [], []
    for category_index in present:
        relevant = targets == category_index
        scores = similarities[:, category_index]
        ap = _average_precision(scores, relevant)
        aps.append(ap)
        order = np.argsort(-scores, kind="stable")
        hits = relevant[order]
        per_category[category_names[int(category_index)]] = {
            "ground_truth_objects": int(relevant.sum()),
            "zero_shot_top1_accuracy": float(top1[relevant].mean()),
            "query_average_precision": ap,
            "query_recall_at_1": float(hits[:1].any()),
            "query_recall_at_5": float(hits[:5].any()),
            "query_recall_at_10": float(hits[:10].any()),
            "query_recall_at_50": float(hits[:50].any()),
        }
        retrieval_r1.append(float(hits[:1].any()))
        retrieval_r5.append(float(hits[:5].any()))
        retrieval_r10.append(float(hits[:10].any()))
        retrieval_r50.append(float(hits[:50].any()))
    return {
        "zero_shot_top1_micro": float(top1.mean()),
        "zero_shot_top1_macro_category": float(np.mean([top1[targets == c].mean() for c in present])),
        "zero_shot_top5_micro": float(top5.mean()),
        "object_query_retrieval_map_macro_category": float(np.mean(aps)),
        "object_query_retrieval_recall_at_1_macro_query": float(np.mean(retrieval_r1)),
        "object_query_retrieval_recall_at_5_macro_query": float(np.mean(retrieval_r5)),
        "object_query_retrieval_recall_at_10_macro_query": float(np.mean(retrieval_r10)),
        "object_query_retrieval_recall_at_50_macro_query": float(np.mean(retrieval_r50)),
        "per_category": per_category,
    }


def compute_retrieval_metrics(scores: np.ndarray,
                              rows: list[dict[str, Any]],
                              categories: list[str]) -> dict[str, Any]:
    """Overall + per-scene retrieval metrics from the score matrix."""
    import numpy as np

    label_to_index = {name: i for i, name in enumerate(categories)}
    targets = np.asarray(
        [label_to_index[row["label"]] for row in rows], dtype=np.int64)
    overall = _score_matrix(scores, targets, categories)
    per_scene: dict[str, Any] = {}
    scene_indexes: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        scene_indexes.setdefault(row["scene"], []).append(index)
    for scene in sorted(scene_indexes):
        idx = np.asarray(scene_indexes[scene], dtype=np.int64)
        per_scene[scene] = {
            "views": int(idx.size),
            **_score_matrix(scores[idx], targets[idx], categories),
        }
    return {"overall": overall, "per_scene": per_scene}


# ---------------------------------------------------------------------------
# Descriptive controls
# ---------------------------------------------------------------------------

_CONTROL_AGGREGATE_KEYS = (
    "zero_shot_top1_micro",
    "zero_shot_top1_macro_category",
    "zero_shot_top5_micro",
    "object_query_retrieval_map_macro_category",
    "object_query_retrieval_recall_at_1_macro_query",
    "object_query_retrieval_recall_at_5_macro_query",
    "object_query_retrieval_recall_at_10_macro_query",
    "object_query_retrieval_recall_at_50_macro_query",
)


def load_descriptive_controls(control_json: Path,
                              warnings: list[str]) -> dict[str, Any]:
    """Extract per-condition aggregate control metrics (descriptive only).

    The frozen control JSON was produced by a separate pipeline without
    paired per-view scores, so only its per-condition aggregate scalars
    (``results[*].input_conditions[*]``) are extracted for side-by-side
    reporting; per-category details are skipped and no paired,
    significance or improvement claims are made.
    """
    if not control_json.is_file():
        warnings.append(
            f"Control JSON missing; descriptive comparison skipped: {control_json}")
        return {"available": False, "source": str(control_json)}
    try:
        payload = json.loads(control_json.read_text(encoding="utf-8"))
        control_sha, control_bytes = sha256_file(control_json)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(
            f"Control JSON unreadable; descriptive comparison skipped: {exc}")
        return {"available": False, "source": str(control_json)}
    if not isinstance(payload, dict):
        warnings.append("Control JSON root is not an object; comparison skipped")
        return {"available": False, "source": str(control_json)}
    conditions: dict[str, Any] = {}
    for result in payload.get("results", []):
        if not isinstance(result, dict):
            continue
        name = result.get("name") or result.get("model_id") or "unknown"
        per_model: dict[str, Any] = {}
        input_conditions = result.get("input_conditions")
        if isinstance(input_conditions, dict):
            for cond_name, cond in input_conditions.items():
                if not isinstance(cond, dict):
                    continue
                per_model[cond_name] = {
                    key: cond.get(key)
                    for key in _CONTROL_AGGREGATE_KEYS if key in cond
                }
        conditions[str(name)] = per_model
    return {
        "available": True,
        "source": str(control_json),
        "source_sha256": control_sha,
        "source_bytes": control_bytes,
        "benchmark": payload.get("benchmark"),
        "dataset": payload.get("dataset"),
        "created_at_utc": payload.get("created_at_utc"),
        "text_prompt_template": (
            (payload.get("settings") or {}).get("text_prompt_template")
        ),
        "conditions": conditions,
        "note": (
            "Descriptive comparison only: control aggregates come from a "
            "separate pipeline run (no paired per-view scores); values are "
            "reported side by side, not compared with significance claims."),
    }


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Canonical evidence serialization (the scope of the output digest)."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def write_evidence_atomic(evidence: dict[str, Any], output_path: Path,
                          allow_overwrite: bool = False) -> dict[str, Any]:
    """Write the evidence JSON atomically (tmp + atomic promotion) with a
    self digest.

    ``output_sha256`` / ``output_bytes`` describe the evidence payload as
    canonically serialized WITHOUT those two fields (verifiable: strip the
    two keys, re-serialize with the same canonical settings, compare).
    NaN/Inf would raise (allow_nan=False), which doubles as a last-line
    finiteness guard.

    The no-overwrite rule is re-enforced at write time, not only by the
    preflight validation (which can race with a file created while
    inference runs): the tmp file is created with O_EXCL, and with
    ``allow_overwrite=False`` the promotion uses ``os.link`` (link(2)),
    which fails atomically with EEXIST when the target already exists — a
    concurrently created target is therefore preserved and the write
    fails loudly. With ``allow_overwrite=True`` the promotion uses
    ``os.replace`` (rename(2)) and an existing target is replaced.
    """
    core_bytes = _canonical_json_bytes(evidence)
    evidence = {
        **evidence,
        "output_sha256": hashlib.sha256(core_bytes).hexdigest(),
        "output_bytes": len(core_bytes),
    }
    final_text = json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_created = False
    try:
        try:
            tmp_fd = os.open(tmp_path,
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            raise OutputPathError(
                f"Temporary evidence file already exists at {tmp_path}; "
                "another run may be writing the same output path") from exc
        tmp_created = True
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as stream:
            stream.write(final_text)
            stream.write("\n")
        if allow_overwrite:
            os.replace(tmp_path, output_path)
        else:
            # link(2) fails atomically with EEXIST if output_path was
            # created after the preflight validation; the existing target
            # is preserved and the failure is raised below.
            try:
                os.link(tmp_path, output_path)
            except FileExistsError as exc:
                raise OutputPathError(
                    f"Output path {output_path} exists after the "
                    "preflight validation (created concurrently); "
                    "refusing to overwrite it (pass --overwrite to allow "
                    "replacement)") from exc
            os.unlink(tmp_path)
    except BaseException:
        if tmp_created and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    written = Path(output_path).stat().st_size
    reloaded = json.loads(output_path.read_text(encoding="utf-8"))
    reloaded_core = {
        key: value
        for key, value in reloaded.items()
        if key not in ("output_sha256", "output_bytes")
    }
    if hashlib.sha256(_canonical_json_bytes(reloaded_core)).hexdigest() != \
            evidence["output_sha256"]:
        raise RuntimeError("Evidence self-digest verification failed after write")
    return {"output_path": str(output_path), "file_bytes": written}


def _write_failure_evidence_best_effort(exc: BaseException,
                                        output_path: Path | None,
                                        phase: str,
                                        established: dict[str, Any],
                                        warnings: list[str],
                                        started_at: str,
                                        wall_started: float,
                                        args: argparse.Namespace) -> None:
    """Record failure evidence at the validated output path (best effort).

    Only attempted when the output path has been resolved and validated
    (``output_path`` is not None); an unvalidated or input-overlapping
    path is never written to. Any error while composing or writing the
    failure evidence is printed and ignored so the original exception is
    never masked. The payload records the timestamp, the inputs, the
    phase reached, the exception type/message, the warnings so far and
    the manifests/hashes established before the failure.
    """
    if output_path is None:
        return
    try:
        failure_evidence = {
            "schema": "t-ren-replica-test5-region-retrieval-failure",
            "schema_version": 1,
            "status": "failed",
            "failures": [
                {
                    "phase": phase,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            ],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started_at,
            "wall_seconds_at_failure": time.perf_counter() - wall_started,
            "phase_reached": phase,
            "inputs": {
                "crop_cache": args.crop_cache,
                "objects_json": args.objects_json,
                "assets_dir": args.assets_dir,
                "control_json": args.control_json,
                "baseline_manifest": args.baseline_manifest,
                "t_ren_repo": args.t_ren_repo,
                "dinov3_repo": args.dinov3_repo,
                "output": str(output_path),
                "overwrite": args.overwrite,
                "batch_size": args.batch_size,
            },
            "established": established,
            "warnings": list(warnings),
        }
        result = write_evidence_atomic(
            failure_evidence, output_path, allow_overwrite=args.overwrite)
        print(
            f"[t-ren-replica-test5] failure evidence written to "
            f"{result['output_path']} ({result['file_bytes']} bytes)",
            file=sys.stderr, flush=True,
        )
    except Exception as failure_exc:
        print(
            f"[t-ren-replica-test5] WARNING: could not write failure "
            f"evidence: {type(failure_exc).__name__}: {failure_exc}",
            file=sys.stderr, flush=True,
        )


def main() -> int:
    args = parse_args()
    warnings: list[str] = []
    wall_started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    # Failure-evidence bookkeeping: the phase reached, the output path
    # (set only once resolved and validated; failure evidence is written
    # only to a validated path) and the manifests/hashes established so
    # far. The run-owned TORCH_HOME temporary directory is created on the
    # project HDD (inside the validated output parent) after the
    # argument and output safety checks and cleaned in the finally.
    phase = "startup"
    established: dict[str, Any] = {}
    output_path: Path | None = None
    torch_home_dir: tempfile.TemporaryDirectory | None = None

    try:
        # ------------------------------------------------------------------
        # Pre-model-work validation: locations, assets, crop manifest.
        # ------------------------------------------------------------------
        phase = "preflight:input-existence"
        crop_cache = Path(args.crop_cache)
        objects_json = Path(args.objects_json)
        assets_dir = Path(args.assets_dir)
        control_json = Path(args.control_json)
        baseline_manifest_path = Path(args.baseline_manifest)
        t_ren_repo = Path(args.t_ren_repo)
        dinov3_repo = Path(args.dinov3_repo)
        # Keep the requested path separate: ``output_path`` stays None
        # until validation succeeds, so a rejected or unresolved target
        # is never used for success or failure output.
        requested_output_path = Path(args.output)

        for path, label in (
            (crop_cache, "crop cache directory"),
            (objects_json, "objects.json"),
            (assets_dir, "assets directory"),
            (t_ren_repo, "T-REN checkout"),
            (dinov3_repo, "DINOv3 checkout"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing {label}: {path}")
        if not (dinov3_repo / "hubconf.py").is_file():
            raise FileNotFoundError(f"Missing DINOv3 hubconf.py: {dinov3_repo / 'hubconf.py'}")
        if not (t_ren_repo / "configs" / "train_dinov3_vitl16.yaml").is_file():
            raise FileNotFoundError(
                f"Missing T-REN config: {t_ren_repo / 'configs' / 'train_dinov3_vitl16.yaml'}")

        # Output path safety: resolve the requested path (symlinks
        # followed) and reject if the result directory does not resolve
        # inside the ConceptGraphs worktree root, if the output does not
        # resolve inside the result directory, or if it collides with a
        # run input or an existing file without --overwrite.
        # Must pass before any model work; on success ``output_path``
        # holds the resolved path used for the evidence and (on failure)
        # the failure evidence. If validation rejects the path,
        # ``output_path`` stays None and no success or failure output is
        # ever written to the rejected target.
        phase = "preflight:output-validation"
        protected_inputs: list[tuple[str, Path]] = [
            ("crop cache", crop_cache),
            ("assets directory", assets_dir),
            *((f"asset {role}", assets_dir / filename)
              for role, filename in ASSET_FILES.items()),
            ("objects.json", objects_json),
            ("control JSON", control_json),
            ("baseline manifest", baseline_manifest_path),
            ("T-REN checkout", t_ren_repo),
            ("DINOv3 checkout", dinov3_repo),
        ]
        output_path = validate_output_path(
            requested_output_path, Path(RESULT_DIR),
            protected_inputs, args.overwrite)

        # Hard dependency gate: a missing importable module (e.g. kornia,
        # required by T-REN model.py) fails here with a clear message
        # instead of deep inside the model load.
        phase = "preflight:dependencies"
        runtime_dependencies = check_runtime_dependencies()

        # Asset identity: strict resolution from --assets-dir plus a fatal
        # cross-check against the root-verified 2026-11-16 hashes/sizes.
        phase = "preflight:asset-identity"
        assets = resolve_asset_paths(assets_dir)
        asset_report: dict[str, Any] = {}
        for role, path in assets.items():
            sha, size = sha256_file(path)
            prior_sha = PRIOR_ASSET_SHA256[role]
            prior_bytes = PRIOR_ASSET_BYTES[role]
            entry: dict[str, Any] = {
                "filename": ASSET_FILES[role],
                "resolved_path": str(path),
                "bytes": size,
                "sha256": sha,
                "prior_bytes_20261116": prior_bytes,
                "prior_sha256_20261116": prior_sha,
                "bytes_match": size == prior_bytes,
                "sha256_match": sha == prior_sha,
            }
            if not (entry["bytes_match"] and entry["sha256_match"]):
                raise RuntimeError(
                    f"Asset identity mismatch for {role} "
                    f"(expected sha256 {prior_sha}, {prior_bytes} bytes; "
                    f"found sha256 {sha}, {size} bytes): {path}")
            asset_report[role] = entry
        established["assets"] = {
            role: {"resolved_path": entry["resolved_path"],
                   "bytes": entry["bytes"], "sha256": entry["sha256"]}
            for role, entry in asset_report.items()
        }

        # Full crop manifest before inference + baseline comparison.
        phase = "preflight:crop-manifest"
        manifest_pre = compute_crop_manifest(crop_cache)
        baseline_comparison = compare_with_baseline_manifest(
            manifest_pre, baseline_manifest_path)
        baseline_status = baseline_comparison["match_status"]
        established["crop_manifest_pre"] = {
            "files": manifest_pre["files"],
            "bytes": manifest_pre["bytes"],
            "manifest_sha256": manifest_pre["manifest_sha256"],
        }
        established["baseline_match_status"] = baseline_status
        if baseline_status == "match":
            pass
        elif baseline_status == "baseline_absent":
            warnings.append(
                "Baseline manifest absent; proceeding without the pre-run "
                "baseline comparison (content is still fully hashed)")
        else:  # mismatch, baseline_unreadable, baseline_malformed
            raise RuntimeError(
                "Crop-cache baseline validation failed "
                f"(status={baseline_status}: "
                f"{baseline_comparison.get('error', '')}); the frozen cache "
                "must match the root-verified baseline manifest (or be "
                "absent, recorded as a warning) before any model work")

        phase = "preflight:dataset"
        rows, categories, joined_labels, min_mask_pixels, max_mask_pixels = \
            build_dataset(crop_cache, objects_json, warnings)
        if len(rows) != TOTAL_VIEWS_EXPECTED:
            raise RuntimeError(
                f"Row count mismatch: expected {TOTAL_VIEWS_EXPECTED}, "
                f"found {len(rows)}")
        queries = [TEXT_PROMPT_TEMPLATE.format(category=name)
                   for name in categories]
        objects_sha, objects_bytes = sha256_file(objects_json)
        established["objects_json"] = {
            "sha256": objects_sha, "bytes": objects_bytes}

        # ------------------------------------------------------------------
        # Model work (CUDA + FP32, lazy torch imports).
        # ------------------------------------------------------------------
        # Run-owned TORCH_HOME: unique per run, created only now — after
        # all argument and output safety checks — and removed in the
        # finally below. torch.hub.load_state_dict_from_url stages even
        # file:// weights under <TORCH_HOME>/hub/checkpoints, so the
        # multi-GB weight cache copies stay on the project HDD.
        # Create the temp directory inside the already validated output
        # parent so the path is guaranteed to exist.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch_home_dir = tempfile.TemporaryDirectory(
            prefix=f"t_ren_test5_torch_home_{os.getpid()}_",
            dir=str(output_path.parent))
        os.environ["TORCH_HOME"] = torch_home_dir.name

        import torch

        phase = "model:init"

        device = _require_cuda()
        config, config_identity = _load_tren_config(t_ren_repo, assets_dir)
        (feature_extractor, region_encoder, text_encoder,
         load_info) = load_tren_components(
            config, device, t_ren_repo, dinov3_repo, assets, warnings)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        phase = "model:text"
        text_started = time.perf_counter()
        with torch.inference_mode():
            text_features = text_encoder(categories)
        torch.cuda.synchronize(device)
        text_seconds = time.perf_counter() - text_started
        if tuple(text_features.shape) != (len(categories), TEXT_EMBED_DIM_EXPECTED):
            raise RuntimeError(
                f"Text feature shape mismatch: expected "
                f"({len(categories)}, {TEXT_EMBED_DIM_EXPECTED}), got "
                f"{tuple(text_features.shape)}")
        if text_features.dtype != torch.float32:
            raise RuntimeError(
                f"Text features must be float32, got {text_features.dtype}")
        if not bool(torch.isfinite(text_features).all()):
            raise RuntimeError("Non-finite text features detected")

        phase = "model:inference"
        scores, inference_info = run_region_inference(
            feature_extractor, region_encoder, rows, text_features,
            device, args.batch_size)
        if (tuple(scores.shape)
                != (TOTAL_VIEWS_EXPECTED, NUM_CATEGORIES_EXPECTED)):
            raise RuntimeError(
                f"Score matrix shape mismatch: expected "
                f"({TOTAL_VIEWS_EXPECTED}, {NUM_CATEGORIES_EXPECTED}), "
                f"got {tuple(scores.shape)}")
        import numpy as np
        if not np.isfinite(scores).all():
            bad = int((~np.isfinite(scores)).sum())
            raise RuntimeError(
                f"Non-finite scores detected in {bad} of "
                f"{scores.size} entries; aborting (finite scores required)")

        torch.cuda.synchronize(device)
        peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

        # ------------------------------------------------------------------
        # Post-inference manifest recompute: any path/size/hash change fails.
        # ------------------------------------------------------------------
        phase = "post:crop-manifest"
        manifest_post = compute_crop_manifest(crop_cache)
        pre_by_path = {
            entry["path"]: (entry["bytes"], entry["sha256"])
            for entry in manifest_pre["entries"]
        }
        post_by_path = {
            entry["path"]: (entry["bytes"], entry["sha256"])
            for entry in manifest_post["entries"]
        }
        changed = sorted(
            path for path in set(pre_by_path) | set(post_by_path)
            if pre_by_path.get(path) != post_by_path.get(path))
        if changed:
            raise RuntimeError(
                f"Crop cache changed during the run; {len(changed)} file(s) "
                f"with changed path/size/hash, first 10: {changed[:10]}")
        drift_report = {
            "entrywise_equal": True,
            "pre_files": manifest_pre["files"],
            "post_files": manifest_post["files"],
            "pre_bytes": manifest_pre["bytes"],
            "post_bytes": manifest_post["bytes"],
            "pre_manifest_sha256": manifest_pre["manifest_sha256"],
            "post_manifest_sha256": manifest_post["manifest_sha256"],
        }

        # ------------------------------------------------------------------
        # Metrics, controls, evidence, write.
        # ------------------------------------------------------------------
        phase = "evidence:write"
        metrics = compute_retrieval_metrics(scores, rows, categories)
        controls = load_descriptive_controls(control_json, warnings)
        host_rss_peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Run-owned TORCH_HOME stats before cleanup (the finally block
        # removes the directory after the evidence write).
        torch_home_path = Path(torch_home_dir.name)
        torch_home_present = torch_home_path.exists()
        torch_home_bytes = sum(
            (path.stat().st_size if path.is_file() else 0)
            for path in torch_home_path.rglob("*")
        ) if torch_home_present else 0

        evidence: dict[str, Any] = {
            "schema": "t-ren-replica-test5-region-retrieval",
            "schema_version": 1,
            "status": "completed",
            "created_at_utc": started_at,
            "inputs": {
                "crop_cache": str(crop_cache),
                "objects_json": str(objects_json),
                "assets_dir": str(assets_dir),
                "control_json": str(control_json),
                "baseline_manifest": str(baseline_manifest_path),
                "t_ren_repo": str(t_ren_repo),
                "dinov3_repo": str(dinov3_repo),
                "output": str(output_path),
                "output_resolved": True,
                "overwrite": args.overwrite,
                "batch_size": args.batch_size,
                "cuda_device": str(device),
                "objects_json_bytes": objects_bytes,
                "objects_json_sha256": objects_sha,
            },
            "environment": {
                "platform": platform.platform(),
                "python": sys.version,
                "offline_env_flags": {
                    "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
                    "HF_DATASETS_OFFLINE": os.environ["HF_DATASETS_OFFLINE"],
                    "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
                    "TORCH_HUB_OFFLINE": os.environ["TORCH_HUB_OFFLINE"],
                    "HF_HUB_DISABLE_TELEMETRY": os.environ["HF_HUB_DISABLE_TELEMETRY"],
                    "TORCH_HOME": os.environ["TORCH_HOME"],
                },
                "library_versions": _library_versions(),
                "torch_cuda_build": getattr(torch.version, "cuda", None),
                "device_name": load_info.get("device_name"),
                "device_total_gib": load_info.get("device_total_gib"),
                "runtime_dependencies": runtime_dependencies,
                "torch_home": {
                    "path": str(torch_home_dir.name),
                    "ownership": (
                        "run-owned temporary directory (tempfile."
                        "TemporaryDirectory), created on the project HDD "
                        "inside the validated output parent after the "
                        "argument and output safety checks and before the "
                        "torch import; removed in main's finally block"),
                    "purpose": (
                        "TORCH_HOME for torch weight cache copies: "
                        "torch.hub.load_state_dict_from_url stages even "
                        "file:// weights under <TORCH_HOME>/hub/"
                        "checkpoints, so the multi-GB copies stay on the "
                        "project HDD and never under /tmp"),
                    "present_at_evidence_write": torch_home_present,
                    "bytes_at_evidence_write": torch_home_bytes,
                },
            },
            "assets": {
                "source": "assets_dir only (no torch hub cache fallback)",
                "files": asset_report,
            },
            "source_identity": {
                "t_ren": git_metadata(str(t_ren_repo)),
                "dinov3": git_metadata(str(dinov3_repo)),
                "checkpoint": {
                    "path": load_info.get("checkpoint_path"),
                    "bytes": asset_report["tren_region_encoder_checkpoint"]["bytes"],
                    "sha256": asset_report["tren_region_encoder_checkpoint"]["sha256"],
                    "state_key": load_info.get("checkpoint_state_key"),
                    "state_key_count": load_info.get(
                        "checkpoint_state_key_count"),
                    "state_keys": load_info.get("checkpoint_state_keys"),
                    "missing_keys": load_info.get("checkpoint_missing_keys"),
                    "unexpected_keys": load_info.get(
                        "checkpoint_unexpected_keys"),
                    "load_strict": load_info.get("checkpoint_load_strict"),
                    "other_state_keys_present": load_info.get(
                        "checkpoint_other_state_keys_present"),
                    "note": ("RegionEncoder loaded strictly from the "
                             "official checkpoint's "
                             "ren_region_encoder_state via "
                             "load_state_dict_resolution_agnostic(strict="
                             "True); the exact state key list and the "
                             "strict-load missing/unexpected key lists "
                             "(both empty on success) are recorded above. "
                             "Image and text encoders use the official "
                             "DINOv3/DINOtxt weights, mirroring tren.py's "
                             "load_checkpoint"),
                },
            },
            "config": config_identity,
            "model": {
                "hub_routing": load_info.get("hub_routing"),
                "pca_stub": load_info.get("pca_stub"),
                "components": (
                    "official T-REN FeatureExtractor, RegionEncoder and "
                    "TextEncoder (model.py), composed as in tren.py; "
                    "TREN class not imported (avoids the 128x128 grid and "
                    "extra top-level imports)"),
                "dtype": "float32 (no autocast)",
                "device": str(device),
            },
            "queries": {
                "text_prompt_template": TEXT_PROMPT_TEMPLATE,
                "text_encoder_prompt": TEXT_ENCODER_PROMPT,
                "categories": categories,
                "queries": queries,
                "num_categories": len(categories),
                "text_embed_dim": TEXT_EMBED_DIM_EXPECTED,
                "frozen_set_match": sorted(categories) == sorted(FROZEN_CATEGORIES),
            },
            "counts": {
                "views": len(rows),
                "views_per_scene": manifest_pre["views_per_scene"],
                "categories": len(categories),
                "joined_label_counts": dict(sorted(joined_labels.items())),
                "min_mask_pixels": min_mask_pixels,
                "max_mask_pixels": max_mask_pixels,
                "crop_files": manifest_pre["files"],
                "crop_bytes": manifest_pre["bytes"],
                "token_counts": {
                    "min": inference_info["token_counts_min"],
                    "max": inference_info["token_counts_max"],
                    "mean": inference_info["token_counts_mean"],
                },
            },
            "crop_manifest": {
                "pre": {
                    "files": manifest_pre["files"],
                    "bytes": manifest_pre["bytes"],
                    "manifest_sha256": manifest_pre["manifest_sha256"],
                    "seconds": manifest_pre["seconds"],
                    "note": ("full per-file entries (4,408) are hashed but "
                             "not embedded; aggregate digest + entry-wise "
                             "checks below are the recorded evidence"),
                },
                "post": {
                    "files": manifest_post["files"],
                    "bytes": manifest_post["bytes"],
                    "manifest_sha256": manifest_post["manifest_sha256"],
                    "seconds": manifest_post["seconds"],
                },
                "baseline_comparison": baseline_comparison,
                "post_inference_drift": drift_report,
            },
            "timings": {
                "init_seconds": load_info.get("init_seconds"),
                "text_seconds": text_seconds,
                "feature_seconds": inference_info["feature_seconds"],
                "region_seconds": inference_info["region_seconds"],
                "cold_seconds": inference_info["cold_seconds"],
                "warm_seconds": inference_info["warm_seconds"],
                "total_inference_seconds": inference_info[
                    "total_inference_seconds"],
                "total_wall_seconds": time.perf_counter() - wall_started,
                "note": (
                    "init = component construction + strict checkpoint load "
                    "on CUDA; text = 49 category encodings; feature/region = "
                    "synchronized wall-clock seconds of the two official "
                    "stages over all batches; cold = first inference batch "
                    "(CUDA warm-up), warm = the rest"),
            },
            "throughput": {
                "cold_views_per_second": inference_info[
                    "cold_views_per_second"],
                "warm_views_per_second": inference_info[
                    "warm_views_per_second"],
                "overall_views_per_second": inference_info[
                    "overall_views_per_second"],
                "batch_size": inference_info["batch_size"],
            },
            "memory": {
                "peak_cuda_allocated_gib": peak_allocated_gib,
                "peak_cuda_reserved_gib": peak_reserved_gib,
                "host_rss_peak_kib": host_rss_peak_kib,
                "note": (
                    "peak CUDA stats cover text encoding + inference "
                    "(reset after component load); host RSS peak is "
                    "RUSAGE_SELF.ru_maxrss"),
            },
            "io_sizes": {
                "crop_cache_files": manifest_pre["files"],
                "crop_cache_bytes": manifest_pre["bytes"],
                "objects_json_bytes": objects_bytes,
                "control_bytes": controls.get("source_bytes"),
                "asset_bytes": {
                    role: entry["bytes"] for role, entry in asset_report.items()
                },
                "torch_home_bytes_at_evidence_write": torch_home_bytes,
            },
            "score_matrix": {
                "shape": [int(scores.shape[0]), int(scores.shape[1])],
                "score_row_order": "dataset row order (scene, frame, object_id)",
                "columns": categories,
                "dtype": "float64 (from float32 GPU scores)",
                "finite": True,
                "scores": [[float(value) for value in row]
                           for row in scores.tolist()],
            },
            "metrics": {
                "convention": (
                    "mirrors conceptgraph/scripts/clip_component.py "
                    "(_score/_average_precision): stable-argsort AP, "
                    "argpartition top-5, macro over present categories; "
                    "per-scene blocks use the same metric bodies"),
                "overall": metrics["overall"],
                "per_scene": metrics["per_scene"],
            },
            "controls": controls,
            "warnings": warnings,
            "failures": [],
            "notes": [
                "Outputs and scratch live outside version control; the "
                "runner never copies or stages model weights.",
                "Control comparison is descriptive only (no paired "
                "per-view scores, no significance claims).",
                "GT mask contributes one deterministic interior point per "
                "view (max boundary distance, row-major tie-break); "
                "official token aggregation with remove_singleton_groups.",
            ],
        }

        phase = "completed"
        write_result = write_evidence_atomic(
            evidence, output_path, allow_overwrite=args.overwrite)
        print(
            f"[t-ren-replica-test5] OK: {len(rows)} views x {len(categories)} "
            f"categories; top1_micro="
            f"{metrics['overall']['zero_shot_top1_micro']:.4f}; evidence at "
            f"{write_result['output_path']} ({write_result['file_bytes']} "
            f"bytes)",
            flush=True,
        )
        return 0
    except Exception as exc:
        print(f"[t-ren-replica-test5] FAILED: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        # Best-effort failure evidence at the validated output path (it is
        # never written for an unvalidated or overlapping path); any error
        # here is printed and does not mask the original exception.
        _write_failure_evidence_best_effort(
            exc, output_path, phase, established, warnings, started_at,
            wall_started, args)
        raise
    finally:
        # Clean up the run-owned TORCH_HOME directory on both success and
        # failure. It is only created after the argument and output safety
        # checks pass, so --help or an early preflight failure never
        # triggers a deletion.
        if torch_home_dir is not None:
            try:
                torch_home_dir.cleanup()
            except OSError as cleanup_exc:
                print(
                    f"[t-ren-replica-test5] WARNING: could not clean the "
                    f"run-owned TORCH_HOME directory "
                    f"{torch_home_dir.name}: {cleanup_exc}",
                    file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    # The run-owned TORCH_HOME directory is cleaned in main's finally
    # block (created only after the argument and output safety checks
    # pass); nothing to clean here on the failure path either.
