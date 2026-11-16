#!/usr/bin/env python3
"""DINOv3 + dino.txt global shared-space crop retrieval runner (frozen test-5).

Stage 1 of docs/pipeline-experiments/reports/optional/dinotxt_clip_substitution.md:
zero-shot 49-category retrieval over the frozen 2,204 GT-mask-isolated
ReplicaSSG object views (five held-out scenes, 4,408 box/mask crop files),
scored in dino.txt's shared 2048-dimensional image/text space.

All persistent artifacts (evidence output, embedding cache, metadata sidecar)
live under ``models/experiments/t_ren/data/dinotxt_test5/`` inside the
repository checkout; nothing is placed on ``/tmp``.

Constraints honored:
- CUDA + FP32 only; fails clearly when CUDA is unavailable, before any model
  load. No CPU inference, no mocks, no network: HF/Torch-Hub offline flags
  are set before any torch/HF import and every weight comes from a locally
  verified asset file.
- The DINOv3 checkout is pinned to 6876159a11b4df116f30f667f8c9888617df0751;
  the dino.txt head, ViT-L/16 LVD-1689M backbone and BPE vocabulary are
  validated by filename, official hash prefix, byte size and full SHA-256
  before the model is loaded.
- Image preprocessing is the official dino.txt notebook transform,
  ``dinov3.data.transforms.make_classification_eval_transform()`` with its
  defaults (the first preprocessing cell of the pinned
  notebooks/dinotxt_inference.ipynb); prompts are exactly
  ``a photo of a {category}`` for the frozen 49 categories. Nothing is
  tuned on the held-out set.
- Crop-cache audit, baseline manifest comparison, label join and metric
  conventions are reused from the accepted T-REN runner
  (scripts/pipeline_model_reports/benchmarks/t_ren_replica_test5.py); only
  the named crop-cache-audit and label-join helpers and their metric
  companions are imported.
- Every input crop/mask file is hashed before and after inference; any
  change fails the run. All outputs and caches live under the configured
   artifact directory; the raw
  result JSON is unique and never overwritten.
- Descriptive control comparison only: the frozen control JSON carries no
  paired per-view scores, so aggregate values are reported side by side
  without significance claims.
- This stage measures crop category retrieval only. It does not measure
  identity, mapping, graph quality or VQA.
"""
from __future__ import annotations

import os

# Offline guarantees are established before any torch/HF import.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import platform
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CROP_CACHE = (
    "/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/results/"
    "dinov3-test5-crops-fp32"
)
DEFAULT_OBJECTS_JSON = (
    "/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/ReplicaSSG/"
    "files/objects.json"
)
DEFAULT_CONTROL_JSON = (
    "/mnt/HDD1/Project_Code/concept-graphs/scripts/pipeline_model_reports/"
    "data/replica-test-retrieval-controls.json"
)
DEFAULT_ASSETS_DIR = (
    "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/t_ren/"
    "weights/dinov3"
)
DEFAULT_DINOV3_REPO = (
    "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/t_ren/"
    "dinov3-repo"
)
DEFAULT_BASELINE_MANIFEST = (
    "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/t_ren/"
    "data/dinotxt_test5/t_ren_test5_crop_manifest_20261116.json"
)
DEFAULT_ARTIFACTS_DIR = (
    "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/t_ren/"
    "data/dinotxt_test5"
)
DEFAULT_EMBEDDING_CACHE = (
    "/mnt/HDD1/Project_Code/concept-graphs/models/experiments/t_ren/"
    "data/dinotxt_test5/dinotxt_test5_global_embeddings_cache.npz"
)
TREN_RUNNER_PATH = (
    "/mnt/HDD1/Project_Code/concept-graphs/scripts/pipeline_model_reports/"
    "benchmarks/t_ren_replica_test5.py"
)
CONCEPT_GRAPHS_ROOT = Path("/mnt/HDD1/Project_Code/concept-graphs")

EXPECTED_REPO_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
DINOTXT_ENTRY = "dinov3_vitl16_dinotxt_tet1280d20h24l"
TEXT_PROMPT_TEMPLATE = "a photo of a {category}"
EMBED_DIM_EXPECTED = 2048
TOTAL_VIEWS_EXPECTED = 2204

# Verified by the root workflow on 2026-11-16 (BRAIN-INDEX). The runner
# recomputes every size and hash itself before the model is loaded.
EXPECTED_ASSETS: dict[str, dict[str, Any]] = {
    "backbone_weights": {
        "filename": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "prefix": "8aa4cbdd",
        "sha256": (
            "e73f0550f00e5eafe9712177da027e5f34df3de5250b56370bcb9acaa"
            "23dab9c"
        ),
        "bytes": 1213021279,
    },
    "dinotxt_head_and_text_encoder": {
        "filename": (
            "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
        ),
        "prefix": "a442d8f5",
        "sha256": (
            "a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d"
            "2eca4f0"
        ),
        "bytes": 2253936683,
    },
    "bpe_vocab": {
        "filename": "bpe_simple_vocab_16e6.txt.gz",
        "prefix": None,
        "sha256": (
            "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48"
            "d6804a"
        ),
        "bytes": 1356917,
    },
}

LIMITATIONS = (
    "Stage 1 crop category retrieval comparison only: does not measure "
    "identity, mapping, graph quality or VQA.",
    "Control comparison is descriptive only: the frozen control JSON has "
    "no paired per-view scores, so no paired or significance claims are "
    "made.",
    "Five of the test scenes are locally complete (office_1, office_3, "
    "office_4, room_1, room_2); six test scenes lack local RGB/poses and "
    "are excluded.",
    "Scores come from the GT-mask-isolated crop mode; they must not be "
    "compared against other crop modes as if paired.",
    "The five scenes are the independent sampling units; object views "
    "within a scene are correlated, so uncertainty is reported "
    "descriptively.",
    "The official frozen dino.txt transform (default "
    "make_classification_eval_transform) and 'a photo of a {category}' "
    "prompts are used as-is; nothing was tuned on this held-out set.",
    "Single-run GPU timings; no multi-run variance estimates.",
)


def check_paths_collide(path_a: Path, path_b: Path) -> bool:
    """True when *path_a* equals or is an ancestor/descendant of *path_b*."""
    if path_a == path_b:
        return True
    try:
        if path_a in path_b.parents:
            return True
    except ValueError:
        pass
    try:
        if path_b in path_a.parents:
            return True
    except ValueError:
        pass
    return False


class RunnerError(RuntimeError):
    """Raised when a pre-inference gate or a run invariant fails."""


def _load_tren_helpers() -> Any:
    """Import the accepted T-REN runner by path for its reusable helpers.

    Only the named crop-cache-audit and label-join helpers (plus their
    metric/convention companions) are used: sha256_file, audit_crop_cache,
    compute_crop_manifest, compare_with_baseline_manifest, build_dataset,
    compute_retrieval_metrics, load_descriptive_controls,
    _CONTROL_AGGREGATE_KEYS and REQUIRED_SCENE_COUNTS.
    """
    spec = importlib.util.spec_from_file_location("t_ren_replica_test5",
                                                 TREN_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RunnerError(
            f"Cannot load T-REN helper module: {TREN_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t_ren = _load_tren_helpers()


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_output_path() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{DEFAULT_ARTIFACTS_DIR}/dinotxt_test5_global_retrieval_"
        f"{stamp}_{os.getpid()}.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DINOv3 + dino.txt global crop retrieval on the frozen "
                    "ReplicaSSG test-5 split (2,204 views, 49 categories).")
    parser.add_argument("--crops", default=DEFAULT_CROP_CACHE)
    parser.add_argument("--objects", default=DEFAULT_OBJECTS_JSON)
    parser.add_argument("--controls", default=DEFAULT_CONTROL_JSON)
    parser.add_argument("--assets-dir", default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--repo", default=DEFAULT_DINOV3_REPO)
    parser.add_argument("--baseline-manifest",
                        default=DEFAULT_BASELINE_MANIFEST)
    parser.add_argument("--embedding-cache",
                        default=DEFAULT_EMBEDDING_CACHE)
    parser.add_argument("--output", default=None,
                        help="evidence JSON path (default: unique path "
                             "under the artifact directory)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    args.output = args.output or _default_output_path()
    return args


def resolve_output_path(
        output: str,
        protected: list[tuple[str, Path]]) -> Path:
    """Resolve the evidence output; reject protected-input targets.

    The resolved path must be strictly beneath the default artifact
    directory after non-strict resolution, enforced by
    ``require_artifact_path``.  A duplicate target is never overwritten:
    that is re-checked atomically at write time by
    write_json_no_overwrite.
    """
    path = Path(output).expanduser().resolve()
    for label, protected_path in protected:
        protected_resolved = protected_path.expanduser().resolve()
        if protected_resolved == path or protected_resolved in path.parents:
            raise RunnerError(
                f"Output path is inside run input {label}: "
                f"{protected_resolved}")
    require_artifact_path(path)
    return path


def require_artifact_path(path: Path, role: str = "Output") -> None:
    """Validate that *path* is a file path strictly beneath the default
    artifact directory, and that the artifact directory itself is a strict
    descendant of the resolved ConceptGraphs project root.

    Rejects the artifact directory itself, any path whose resolved
    parent or symlink chain escapes the artifact directory, and any
    configuration where the resolved artifact directory is not a strict
    descendant of the resolved project root.

    *role* identifies the destination role in error messages.
    """
    project_root = Path(CONCEPT_GRAPHS_ROOT).expanduser().resolve()
    artifact_dir = Path(DEFAULT_ARTIFACTS_DIR).expanduser().resolve()
    if artifact_dir == project_root:
        raise RunnerError(
            f"Artifact directory cannot be the project root itself; "
            f"benchmark artifacts must remain beneath "
            f"{project_root}: {artifact_dir}")
    try:
        artifact_dir.relative_to(project_root)
    except ValueError:
        raise RunnerError(
            f"Artifact directory resolves outside the ConceptGraphs project "
            f"root: artifact={artifact_dir}, project root={project_root}")
    if path == artifact_dir:
        raise RunnerError(
            f"{role} path cannot be the artifact directory itself; "
            f"benchmark output/cache files must remain beneath "
            f"{artifact_dir}: {path}")
    if not path.is_relative_to(artifact_dir):
        raise RunnerError(
            f"{role} path is not beneath the artifact directory; "
            f"benchmark output/cache files must remain beneath "
            f"{artifact_dir}: {path}")


def write_json_no_overwrite(payload: dict[str, Any], path: Path) -> None:
    """Atomically write JSON to a path that must not already exist.

    Uses ``tempfile.mkstemp`` for exclusive random-temp creation: concurrent
    calls in the same process never collide on the temp path.  The temp file
    is fsync'd before the atomic ``os.link`` promotion, and the temp is
    cleaned up on both success and failure.
    """
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.",
        dir=str(path.parent),
    )
    f = None
    try:
        try:
            f = os.fdopen(fd, "wb")
            fd = -1
            data = json.dumps(payload, indent=1).encode("utf-8")
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            raise
        finally:
            if f is not None:
                try:
                    f.close()
                except OSError:
                    pass
        try:
            os.link(tmp_path, str(path))
        except FileExistsError:
            raise RunnerError(f"Output appeared during write; "
                              f"not overwriting: {path}")
    except BaseException:
        raise
    finally:
        try:
            if fd >= 0:
                os.close(fd)
        except OSError:
            pass
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass



def validate_repo_commit(repo: Path) -> dict[str, Any]:
    """Record and verify the pinned DINOv3 checkout commit."""
    repo = repo.expanduser().resolve()
    if not (repo / "dinov3" / "hub" / "dinotxt.py").is_file():
        raise RunnerError(
            f"DINOv3 checkout missing dinov3/hub/dinotxt.py: {repo}")
    proc = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo),
         "rev-parse", "HEAD"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RunnerError(
            f"Cannot resolve the DINOv3 commit at {repo}: "
            f"{proc.stderr.strip()}")
    commit = proc.stdout.strip()
    if commit != EXPECTED_REPO_COMMIT:
        raise RunnerError(
            f"DINOv3 commit mismatch: expected {EXPECTED_REPO_COMMIT}, "
            f"found {commit}")
    return {
        "path": str(repo),
        "commit": commit,
        "commit_expected": EXPECTED_REPO_COMMIT,
        "commit_match": True,
    }


def validate_assets(assets_dir: Path) -> dict[str, Any]:
    """Validate filename, prefix, byte size and SHA-256 for every asset.

    All three assets are checked before the model is loaded; any deviation
    from the root-verified values fails the run.
    """
    assets_dir = assets_dir.expanduser().resolve()
    if not assets_dir.is_dir():
        raise RunnerError(f"Missing assets directory: {assets_dir}")
    report: dict[str, Any] = {}
    for role, spec in EXPECTED_ASSETS.items():
        path = assets_dir / spec["filename"]
        if path.is_symlink() or not path.is_file():
            raise RunnerError(
                f"Asset missing or not a regular file: {path}")
        digest, size = t_ren.sha256_file(path)
        prefix = spec["prefix"]
        entry = {
            "path": str(path),
            "filename": spec["filename"],
            "bytes": size,
            "sha256": digest,
            "size_expected": spec["bytes"],
            "size_match": size == spec["bytes"],
            "sha_expected": spec["sha256"],
            "sha_match": digest == spec["sha256"],
            "prefix_expected": prefix,
            "prefix_match": prefix is None or prefix in path.name,
        }
        if not (entry["size_match"] and entry["sha_match"]
                and entry["prefix_match"]):
            raise RunnerError(
                f"Asset validation failed for {role}: {entry}")
        report[role] = entry
    return report


def _require_cuda(device: str) -> tuple[Any, dict[str, Any]]:
    """Import torch and fail before any model load when CUDA is missing."""
    import torch
    if not torch.cuda.is_available():
        raise RunnerError(
            "CUDA is unavailable; this benchmark requires the configured "
            "GPU and refuses to fall back to CPU")
    dev = torch.device(device)
    if dev.type != "cuda":
        raise RunnerError(
            f"Requested device {device} is not a CUDA device")
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if idx < 0:
        raise RunnerError(
            f"Requested device index {dev.index} must be nonnegative")
    if idx >= torch.cuda.device_count():
        raise RunnerError(
            f"Requested device {device} is not available "
            f"(device_count={torch.cuda.device_count()})")
    return torch, {
        "device": device,
        "device_name": torch.cuda.get_device_name(dev),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "offline_env": {
            key: os.environ.get(key)
            for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE",
                        "TRANSFORMERS_OFFLINE", "TORCH_HUB_OFFLINE",
                        "HF_HUB_DISABLE_TELEMETRY")
        },
    }


def build_view_rows(crop_cache: Path, objects_json: Path,
                    warnings: list[str]) -> tuple[list[dict[str, Any]],
                                                  list[str], dict[str, int],
                                                  dict[str, int]]:
    """Label join plus frozen-set validation via the T-REN helper.

    Reuses t_ren.build_dataset: strict crop-cache audit, objects.json
    join, per-scene counts (2,204 views), the exact frozen 49-category
    set, RGB box crops and non-empty paired masks.
    """
    rows, categories, joined_labels, min_px, max_px = t_ren.build_dataset(
        crop_cache, objects_json, warnings)
    if len(rows) != TOTAL_VIEWS_EXPECTED:
        raise RunnerError(
            f"View count mismatch: expected {TOTAL_VIEWS_EXPECTED}, "
            f"found {len(rows)}")
    if len(categories) != 49:
        raise RunnerError(
            f"Category count mismatch: expected 49, found "
            f"{len(categories)}")
    return rows, categories, joined_labels, {
        "min_mask_foreground_pixels": min_px,
        "max_mask_foreground_pixels": max_px,
    }


def capture_manifest(crop_cache: Path,
                     baseline_manifest: Path) -> tuple[dict[str, Any],
                                                       dict[str, Any]]:
    """Content-sensitive manifest over all crop files plus baseline check."""
    manifest = t_ren.compute_crop_manifest(crop_cache)
    baseline = t_ren.compare_with_baseline_manifest(
        manifest, baseline_manifest)
    return manifest, baseline


def load_dinotxt_model(repo: Path, assets: dict[str, Any],
                       device: str) -> tuple[Any, Any, float, float]:
    """Load the pinned dino.txt model from the validated local assets.

    Uses the verified official factory
    ``dinov3.hub.dinotxt.dinov3_vitl16_dinotxt_tet1280d20h24l`` with
    keyword-only arguments, ``check_hash=True``, and local weights only.
    All weights are frozen (requires_grad off, eval mode).
    """
    import torch
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    module = importlib.import_module("dinov3.hub.dinotxt")
    factory = getattr(module, DINOTXT_ENTRY)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model, tokenizer = factory(
        pretrained=True,
        weights=str(assets["dinotxt_head_and_text_encoder"]["path"]),
        backbone_weights=str(assets["backbone_weights"]["path"]),
        bpe_path_or_url=str(assets["bpe_vocab"]["path"]),
        check_hash=True,
    )
    load_seconds = time.perf_counter() - started
    model.requires_grad_(False)
    model.eval()
    model = model.to(device)
    torch.cuda.synchronize(device)
    logit_scale = float(model.logit_scale.exp().item())
    if not np.isfinite(logit_scale) or logit_scale <= 0.0:
        raise RunnerError(f"Invalid dino.txt logit scale: {logit_scale}")
    return model, tokenizer, load_seconds, logit_scale


def build_official_transform(repo: Path) -> tuple[Any, str, str]:
    """The official dino.txt notebook image transform, defaults only.

    Mirrors the first preprocessing cell of the pinned
    notebooks/dinotxt_inference.ipynb:
    ``make_classification_eval_transform()`` (ToImage, resize,
    CenterCrop, RGB normalization with ImageNet mean/std).
    """
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dinov3.data.transforms import make_classification_eval_transform
    transform = make_classification_eval_transform()
    signature = str(inspect.signature(make_classification_eval_transform))
    return transform, signature, str(transform)


def encode_text_queries(model: Any, tokenizer: Any, categories: list[str],
                        device: str) -> tuple[np.ndarray, dict[str, Any],
                                              float]:
    """Encode the 49 frozen category prompts with the official tokenizer."""
    import torch
    prompts = [TEXT_PROMPT_TEMPLATE.format(category=name)
               for name in categories]
    prompt_sha = hashlib.sha256(
        json.dumps(prompts, sort_keys=True).encode("utf-8")).hexdigest()
    started = time.perf_counter()
    with torch.no_grad():
        tokens = tokenizer.tokenize(prompts).to(device)
        text_features = model.encode_text(tokens, normalize=True)
        text_features = (text_features.float().cpu().numpy()
                         .astype(np.float32))
    seconds = time.perf_counter() - started
    if text_features.ndim != 2 or text_features.shape[0] != len(prompts):
        raise RunnerError(
            f"Unexpected text feature shape: {text_features.shape} for "
            f"{len(prompts)} prompts")
    if text_features.shape[1] != EMBED_DIM_EXPECTED:
        raise RunnerError(
            f"Unexpected text embedding dimension: expected "
            f"{EMBED_DIM_EXPECTED}, found {text_features.shape[1]}")
    if not np.isfinite(text_features).all():
        raise RunnerError("Non-finite text features")
    norms = np.linalg.norm(text_features, axis=1)
    return text_features, {
        "prompts": prompts,
        "prompt_sha256": prompt_sha,
        "dimension": int(text_features.shape[1]),
        "feature_sha256": hashlib.sha256(
            text_features.tobytes()).hexdigest(),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }, seconds


def _cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".json")


def load_embedding_cache(cache_path: Path,
                         meta: dict[str, Any],
                         sidecar_path: Path) -> dict[str, np.ndarray]:
    """Load the dino.txt embedding cache; stale or corrupt caches are empty.

    The cache lives in the dino.txt artifact directory and maps
    input-crop SHA-256 to the normalized 2048-d FP32 embedding.  Any meta
    mismatch (transform, prompts, weights) or structural defect rebuilds
    it from scratch; a corrupt cache is never trusted.
    """
    if not (cache_path.is_file() and sidecar_path.is_file()):
        return {}
    try:
        saved_meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if saved_meta != meta:
            return {}
        npz = np.load(cache_path, allow_pickle=False)
        entries = {key: np.asarray(npz[key], dtype=np.float32)
                   for key in npz.files}
        if any(value.shape != (EMBED_DIM_EXPECTED,)
               for value in entries.values()):
            return {}
        if not all(np.isfinite(value).all() for value in entries.values()):
            return {}
        return entries
    except Exception:
        return {}


def save_embedding_cache(cache_path: Path,
                         entries: dict[str, np.ndarray],
                         meta: dict[str, Any],
                         sidecar_path: Path) -> None:
    """Atomically persist the embedding cache and its meta sidecar.

    Uses ``tempfile.mkstemp`` for exclusive random-temp creation on both
    the NPZ cache and the JSON metadata sidecar, preventing collisions
    when concurrent calls target the same cache path.  Both artifacts are
    fsync'd before ``os.replace`` promotion; temp paths are removed in
    ``finally`` blocks.
    """
    # --- NPZ cache ---
    npz_fd, npz_tmp = tempfile.mkstemp(
        prefix=".cache_tmp.",
        dir=str(cache_path.parent),
    )
    npz_f = None
    try:
        try:
            npz_f = os.fdopen(npz_fd, "wb")
            npz_fd = -1
            np.savez(npz_f, **entries)
            npz_f.flush()
            os.fsync(npz_f.fileno())
        except BaseException:
            raise
        finally:
            if npz_f is not None:
                try:
                    npz_f.close()
                except OSError:
                    pass
        os.replace(npz_tmp, cache_path)
    except BaseException:
        raise
    finally:
        try:
            if npz_fd >= 0:
                os.close(npz_fd)
        except OSError:
            pass
        try:
            Path(npz_tmp).unlink(missing_ok=True)
        except OSError:
            pass

    # --- metadata sidecar ---
    meta_fd, meta_tmp = tempfile.mkstemp(
        prefix=".meta_tmp.",
        dir=str(sidecar_path.parent),
    )
    meta_f = None
    try:
        try:
            meta_f = os.fdopen(meta_fd, "wb")
            meta_fd = -1
            data = json.dumps(meta, indent=1, sort_keys=True).encode("utf-8")
            meta_f.write(data)
            meta_f.flush()
            os.fsync(meta_f.fileno())
        except BaseException:
            raise
        finally:
            if meta_f is not None:
                try:
                    meta_f.close()
                except OSError:
                    pass
        os.replace(meta_tmp, sidecar_path)
    except BaseException:
        raise
    finally:
        try:
            if meta_fd >= 0:
                os.close(meta_fd)
        except OSError:
            pass
        try:
            Path(meta_tmp).unlink(missing_ok=True)
        except OSError:
            pass


def encode_images(rows: list[dict[str, Any]], crops_root: Path,
                  manifest: dict[str, Any], transform: Any, model: Any,
                  device: str, batch_size: int, cache_path: Path,
                  cache_meta: dict[str, Any],
                  sidecar_path: Path) -> tuple[np.ndarray,
                                                       dict[str, Any],
                                                       list[float]]:
    """Encode the mask-isolated box crops in batches, cache-aware.

    Cache keys are the input-crop SHA-256 values from the pre-inference
    manifest, so the cache is content-addressed and transform-bound via
    its meta sidecar. Non-finite features and unexpected embedding
    dimensions are rejected.
    """
    import torch
    from PIL import Image

    sha_by_path = {entry["path"]: entry["sha256"]
                   for entry in manifest["entries"]}
    keys: list[str] = []
    for row in rows:
        rel = str(Path(row["box_path"]).relative_to(crops_root))
        if rel not in sha_by_path:
            raise RunnerError(f"View missing from input manifest: {rel}")
        keys.append(sha_by_path[rel])

    cached_entries = load_embedding_cache(
        cache_path, cache_meta, sidecar_path)
    features = np.empty((len(rows), EMBED_DIM_EXPECTED), dtype=np.float32)
    missing_indexes: list[int] = []
    for index, key in enumerate(keys):
        cached = cached_entries.get(key)
        if cached is not None:
            features[index] = cached
        else:
            missing_indexes.append(index)
    cache_hits = len(rows) - len(missing_indexes)
    batch_seconds: list[float] = []
    paths = [Path(row["box_path"]) for row in rows]
    for start in range(0, len(missing_indexes), batch_size):
        batch_indexes = missing_indexes[start:start + batch_size]
        images = torch.stack([
            transform(Image.open(paths[i]).convert("RGB"))
            for i in batch_indexes
        ]).to(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            batch_features = model.encode_image(images, normalize=True)
            batch_features = (batch_features.float().cpu().numpy()
                              .astype(np.float32))
        torch.cuda.synchronize(device)
        batch_seconds.append(time.perf_counter() - started)
        if (batch_features.ndim != 2
                or batch_features.shape[0] != len(batch_indexes)
                or batch_features.shape[1] != EMBED_DIM_EXPECTED):
            raise RunnerError(
                f"Unexpected image feature shape: {batch_features.shape}")
        if not np.isfinite(batch_features).all():
            raise RunnerError(
                "Non-finite image features for views: "
                + ", ".join(str(int(i)) for i in batch_indexes[:8]))
        for i, feature in zip(batch_indexes, batch_features):
            features[i] = feature
    if missing_indexes:
        merged = {**cached_entries,
                  **{keys[i]: features[i] for i in missing_indexes}}
        save_embedding_cache(
            cache_path, merged, cache_meta, sidecar_path)
        stored_entries = len(merged)
    else:
        stored_entries = len(cached_entries)
    stats = {
        "path": str(cache_path),
        "meta": cache_meta,
        "entries": stored_entries,
        "bytes": (cache_path.stat().st_size
                  if cache_path.is_file() else 0),
        "hits": cache_hits,
        "misses": len(missing_indexes),
        "rebuilt": bool(missing_indexes),
    }
    return features, stats, batch_seconds


def build_control_comparison(controls_json: Path, overall: dict[str, Any],
                             warnings: list[str]) -> dict[str, Any]:
    """Descriptive-only aggregate comparison against the frozen controls.

    The frozen control JSON has no paired per-view scores, so only
    aggregate scalars from the ground-truth-mask-isolated condition are
    placed side by side with the dino.txt overall metrics; deltas are
    descriptive and carry no significance claims.
    """
    controls = t_ren.load_descriptive_controls(controls_json, warnings)
    block: dict[str, Any] = {
        "descriptive_only": True,
        "paired_per_view_scores_available": False,
        "note": ("Descriptive side-by-side comparison only: the frozen "
                 "control JSON has no paired per-view scores, so no "
                 "paired deltas or significance claims are made."),
        "dinotxt_overall": {
            key: overall[key] for key in t_ren._CONTROL_AGGREGATE_KEYS
            if key in overall
        },
    }
    if not controls.get("available"):
        block["controls_available"] = False
        block["control_source"] = controls.get("source")
        return block
    block["controls_available"] = True
    block["control_source"] = controls["source"]
    block["control_source_sha256"] = controls.get("source_sha256")
    block["control_benchmark"] = controls.get("benchmark")
    models: dict[str, Any] = {}
    for name, conditions in (controls.get("conditions") or {}).items():
        condition = (conditions or {}).get("ground-truth-mask-isolated")
        condition = condition if isinstance(condition, dict) else {}
        row = {key: condition.get(key)
               for key in t_ren._CONTROL_AGGREGATE_KEYS}
        row["delta_control_minus_dinotxt"] = {
            key: (value - overall[key]
                  if isinstance(value, (int, float))
                  and not isinstance(value, bool) and key in overall
                  else None)
            for key, value in row.items()
        }
        models[str(name)] = row
    block["models"] = models
    return block


def main() -> int:
    args = parse_args()
    output_path = resolve_output_path(
        args.output,
        [("crop cache", Path(args.crops)),
         ("assets directory", Path(args.assets_dir)),
         ("objects.json", Path(args.objects)),
         ("control JSON", Path(args.controls)),
         ("embedding cache", Path(args.embedding_cache)),
         ("baseline manifest", Path(args.baseline_manifest))])
    # Resolve and validate cache destination before model initialization.
    cache_path = Path(args.embedding_cache).expanduser().resolve()
    sidecar_path = cache_path.with_suffix(".json").resolve()
    require_artifact_path(output_path)
    require_artifact_path(cache_path, role="Cache")
    require_artifact_path(sidecar_path, role="Sidecar")
    if check_paths_collide(output_path, cache_path):
        raise RunnerError(
            "Output and cache destination collide; "
            "benchmark output/cache files must be pairwise disjoint")
    if check_paths_collide(output_path, sidecar_path):
        raise RunnerError(
            "Output and sidecar destination collide; "
            "benchmark output/cache files must be pairwise disjoint")
    if check_paths_collide(cache_path, sidecar_path):
        raise RunnerError(
            "Cache and sidecar destination collide; "
            "benchmark output/cache files must be pairwise disjoint")
    # Create parent directories only after all destinations validated and
    # collision-checked; before cache I/O/model inference.
    artifact_dir = Path(DEFAULT_ARTIFACTS_DIR).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    crops_root = Path(args.crops).expanduser().resolve()
    warnings: list[str] = []
    config: dict[str, Any] = {
        "crops": args.crops,
        "objects_json": args.objects,
        "controls": args.controls,
        "assets_dir": args.assets_dir,
        "repo": args.repo,
        "baseline_manifest": args.baseline_manifest,
        "embedding_cache": str(cache_path),
        "artifact_directory": str(artifact_dir),
        "output": str(output_path),
        "sidecar": str(sidecar_path),
        "device": args.device,
        "batch_size": args.batch_size,
        "prompt_template": TEXT_PROMPT_TEMPLATE,
        "embed_dim_expected": EMBED_DIM_EXPECTED,
        "view_count_expected": TOTAL_VIEWS_EXPECTED,
    }
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": _utcnow(),
        "status": "running",
        "benchmark": ("DINOv3 + dino.txt global shared-space crop retrieval "
                      "on the frozen ReplicaSSG test-5 split"),
        "protocol": ("docs/pipeline-experiments/reports/optional/"
                     "dinotxt_clip_substitution.md"),
        "runner_file": __file__,
        "config": config,
        "failures": [],
        "warnings": warnings,
        "limitations": list(LIMITATIONS),
    }
    started = time.perf_counter()
    try:
        torch, environment = _require_cuda(args.device)
        evidence["environment"] = environment
        evidence["source_hashes"] = validate_repo_commit(Path(args.repo))
        assets = validate_assets(Path(args.assets_dir))
        evidence["assets"] = assets

        rows, categories, joined_labels, mask_stats = build_view_rows(
            crops_root, Path(args.objects), warnings)
        config["categories"] = categories
        config["prompts"] = [
            TEXT_PROMPT_TEMPLATE.format(category=name)
            for name in categories
        ]
        config["prompt_sha256"] = hashlib.sha256(
            json.dumps(config["prompts"], sort_keys=True)
            .encode("utf-8")).hexdigest()
        objects_sha, objects_bytes = t_ren.sha256_file(Path(args.objects))
        evidence["dataset"] = {
            "views": len(rows),
            "crop_files": 2 * len(rows),
            "scenes": dict(t_ren.REQUIRED_SCENE_COUNTS),
            "categories": categories,
            "num_categories": len(categories),
            "joined_labels": joined_labels,
            "mask_foreground_pixels": mask_stats,
            "objects_json": {
                "path": args.objects,
                "sha256": objects_sha,
                "bytes": objects_bytes,
            },
        }

        manifest_pre, baseline_comparison = capture_manifest(
            crops_root, Path(args.baseline_manifest))
        evidence["input_manifest"] = manifest_pre
        evidence["baseline_manifest_comparison"] = baseline_comparison

        model, tokenizer, load_seconds, logit_scale = load_dinotxt_model(
            Path(args.repo), assets, args.device)
        config["logit_scale_exp"] = logit_scale
        resident_after_load_gib = (
            torch.cuda.memory_allocated(args.device) / 2**30)
        transform, transform_signature, transform_pipeline = (
            build_official_transform(Path(args.repo)))
        config["transform_signature"] = transform_signature
        config["transform_pipeline"] = transform_pipeline
        text_features, text_info, text_seconds = encode_text_queries(
            model, tokenizer, categories, args.device)
        evidence["text_queries"] = text_info
        cache_meta = {
            "transform_pipeline": transform_pipeline,
            "logit_scale_exp": logit_scale,
            "prompt_sha256": text_info["prompt_sha256"],
            "backbone_sha256": assets["backbone_weights"]["sha256"],
            "head_sha256": (
                assets["dinotxt_head_and_text_encoder"]["sha256"]
            ),
            "bpe_sha256": assets["bpe_vocab"]["sha256"],
            "dim": EMBED_DIM_EXPECTED,
        }
        features, cache_stats, batch_seconds = encode_images(
            rows, crops_root, manifest_pre, transform, model, args.device,
            args.batch_size, cache_path, cache_meta, sidecar_path)
        evidence["embedding_cache"] = cache_stats
        torch.cuda.synchronize(args.device)
        scores = (features @ text_features.T).astype(np.float32)
        scores *= logit_scale
        metric_started = time.perf_counter()
        metrics = t_ren.compute_retrieval_metrics(scores, rows, categories)
        metric_seconds = time.perf_counter() - metric_started
        evidence["metrics"] = metrics

        per_view_scores: dict[str, Any] = {}
        for index, row in enumerate(rows):
            stem = Path(row["box_path"]).name[: -len("-box.png")]
            per_view_scores[f"{row['scene']}/{stem}"] = {
                "category": row["label"],
                "scores": scores[index].tolist(),
            }
        evidence["per_view_scores"] = per_view_scores
        evidence["control_comparison"] = build_control_comparison(
            Path(args.controls), metrics["overall"], warnings)

        manifest_post = t_ren.compute_crop_manifest(crops_root)
        unchanged = (manifest_post["manifest_sha256"]
                     == manifest_pre["manifest_sha256"])
        if not unchanged:
            raise RunnerError(
                "Input crop/mask files changed during inference; the "
                "pre- and post-inference manifests differ")
        evidence["post_inference_manifest"] = {
            "files": manifest_post["files"],
            "bytes": manifest_post["bytes"],
            "manifest_sha256": manifest_post["manifest_sha256"],
            "seconds": round(float(manifest_post["seconds"]), 3),
        }
        evidence["input_manifest_unchanged_after_inference"] = unchanged

        encode_seconds = float(sum(batch_seconds))
        warm_batches = batch_seconds[1:]
        encoded_views = int(cache_stats["misses"])
        evidence["timings"] = {
            "total_seconds": round(time.perf_counter() - started, 3),
            "model_load_seconds": round(load_seconds, 3),
            "text_encode_seconds": round(text_seconds, 3),
            "image_encode_seconds": round(encode_seconds, 3),
            "image_encode_first_batch_seconds": (
                round(batch_seconds[0], 3) if batch_seconds else None
            ),
            "image_encode_warm_batch_mean_seconds": (
                round(float(sum(warm_batches) / len(warm_batches)), 3)
                if warm_batches else None
            ),
            "image_encode_batches": len(batch_seconds),
            "manifest_pre_seconds": round(float(manifest_pre["seconds"]), 3),
            "manifest_post_seconds": (
                round(float(manifest_post["seconds"]), 3)
            ),
            "retrieval_metrics_seconds": round(metric_seconds, 3),
        }
        evidence["throughput"] = {
            "views_from_cache": int(cache_stats["hits"]),
            "views_encoded": encoded_views,
            "encoded_views_per_second": (
                round(encoded_views / encode_seconds, 3)
                if encode_seconds > 0 else None
            ),
        }
        evidence["memory"] = {
            "resident_cuda_allocated_gib_after_model_load": (
                round(resident_after_load_gib, 6)
            ),
            "peak_cuda_allocated_gib": round(
                torch.cuda.max_memory_allocated(args.device) / 2**30, 6
            ),
            "peak_cuda_reserved_gib": round(
                torch.cuda.max_memory_reserved(args.device) / 2**30, 6
            ),
            "host_rss_peak_gib": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 * 1024), 6
            ),
            "embedding_cache_bytes": int(cache_stats["bytes"]),
        }
        evidence["status"] = "success"
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        evidence["failures"].append(evidence["error"])
        evidence["traceback"] = traceback.format_exc(limit=24)
        try:
            write_json_no_overwrite(evidence, output_path)
        except Exception:
            pass
        print(f"DINOtxt test-5 run FAILED: {evidence['error']}",
              file=sys.stderr)
        return 2
    evidence["output"] = str(output_path)
    write_json_no_overwrite(evidence, output_path)
    overall = evidence["metrics"]["overall"]
    print(
        "DINOtxt test-5 global retrieval: status=success views=%d "
        "map=%.4f top1_macro=%.4f top5_micro=%.4f recall@1=%.4f "
        "output=%s"
        % (len(rows),
           overall["object_query_retrieval_map_macro_category"],
           overall["zero_shot_top1_macro_category"],
           overall["zero_shot_top5_micro"],
           overall["object_query_retrieval_recall_at_1_macro_query"],
           output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
