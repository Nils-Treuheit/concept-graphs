#!/usr/bin/env python3
"""Benchmark SAM 3.1 box-prompt template-mask refinement.

This script deliberately imports SAM/PyTorch only after validating the local
manifest and checkpoint. Run it only after the project GPU release gate passes.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import io
import json
import logging
import re
import subprocess
import sys
import threading
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={str(path)}", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != "sam31-template-refinement-v1":
        raise ValueError("manifest schema must be sam31-template-refinement-v1")
    for key in ("dataset", "split"):
        if not isinstance(doc.get(key), str) or not doc[key].strip():
            raise ValueError(f"manifest {key} must be a non-empty string")
    if doc.get("sam3_repo_commit") != "eef6839bcfc3458dd52ea80535dc6bd62aea86b1":
        raise ValueError("manifest must pin SAM repository commit eef6839bcfc3458dd52ea80535dc6bd62aea86b1")
    items = doc.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("manifest items must be a non-empty list")
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"items[{index}] must be an object")
        required = ("sample_id", "image", "bbox_xyxy", "base_mask", "ground_truth_mask")
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"items[{index}] missing keys: {', '.join(missing)}")
        sample_id = str(item["sample_id"])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sample_id) or sample_id in {".", ".."}:
            raise ValueError(f"sample_id is not a safe artifact name: {sample_id!r}")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id!r}")
        hashes = item.get("sha256")
        if not isinstance(hashes, dict) or any(name not in hashes for name in ("image", "base_mask", "ground_truth_mask")):
            raise ValueError(f"{sample_id}: sha256 must map image, base_mask and ground_truth_mask to digests")
        for name in ("image", "base_mask", "ground_truth_mask"):
            digest = hashes[name]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{sample_id}: sha256.{name} must be a lowercase SHA-256 digest")
        seen.add(sample_id)
        box = item["bbox_xyxy"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"{sample_id}: bbox_xyxy must be [x0,y0,x1,y1] in source pixels")
        try:
            x0, y0, x1, y1 = (float(value) for value in box)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{sample_id}: bbox_xyxy must contain finite numbers") from exc
        if not all(map(lambda v: v == v and abs(v) != float("inf"), (x0, y0, x1, y1))):
            raise ValueError(f"{sample_id}: bbox_xyxy contains a non-finite number")
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError(f"{sample_id}: bbox_xyxy must have positive area and nonnegative origin")
    return doc, items


def validate_assets(manifest_path: Path, items: list[dict[str, Any]], checkpoint: Path) -> list[dict[str, Path]]:
    base = manifest_path.parent
    if checkpoint.name != "sam3.1_multiplex.pt":
        raise ValueError(f"expected official SAM 3.1 multiplex checkpoint filename, got {checkpoint.name!r}")
    checkpoint_target = checkpoint.resolve()
    if not checkpoint_target.is_file() or checkpoint_target.stat().st_size == 0:
        raise FileNotFoundError(f"SAM 3.1 checkpoint missing/empty: {checkpoint_target}")
    resolved = []
    for item in items:
        paths: dict[str, Path] = {}
        for name in ("image", "base_mask", "ground_truth_mask"):
            raw = Path(str(item[name])).expanduser()
            if raw.is_absolute() or ".." in raw.parts or any(part in ("", ".") for part in raw.parts):
                raise ValueError(f"{item['sample_id']}: {name} must be an unambiguous relative path under the manifest directory")
            unresolved = base / raw
            cursor = base
            for part in raw.parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    raise ValueError(f"{item['sample_id']}: {name} path must not traverse symlinks: {raw}")
            path = unresolved.resolve()
            if not path.is_relative_to(base.resolve()):
                raise ValueError(f"{item['sample_id']}: {name} resolves outside the manifest directory")
            paths[name] = path
        for name, path in paths.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"{item['sample_id']}: {name} missing/empty: {path}")
            if sha256(path) != item["sha256"][name]:
                raise ValueError(f"{item['sample_id']}: {name} SHA-256 does not match the frozen manifest")
        resolved.append(paths)
    return resolved


def binary_mask(path: Path):
    import numpy as np
    from PIL import Image

    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        value = np.asarray(Image.open(path).convert("L"))
    if value.ndim == 3:
        if value.shape[2] not in (1, 3, 4):
            raise ValueError(f"unsupported mask array shape {value.shape}: {path}")
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"mask must be HxW, got {value.shape}: {path}")
    return value > 0


def iou(a, b) -> float:
    import numpy as np

    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().to("cpu").numpy()
    return value


def box_iou_xywh(a, b) -> float:
    ax, ay, aw, ah = (float(value) for value in a)
    bx, by, bw, bh = (float(value) for value in b)
    if not all(map(lambda v: v == v and abs(v) != float("inf"), (ax, ay, aw, ah, bx, by, bw, bh))):
        raise ValueError("SAM or prompt box contains non-finite coordinates")
    if min(aw, ah, bw, bh) <= 0 or min(ax, ay, bx, by) < 0 or max(ax + aw, ay + ah, bx + bw, by + bh) > 1.00001:
        raise ValueError("SAM or prompt box is outside normalized XYWH range")
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = ix * iy
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def extract_prediction(outputs, height: int, width: int, prompt_box_xywh):
    import numpy as np

    if not isinstance(outputs, dict):
        raise TypeError("SAM add_prompt outputs must be a mapping")
    for key in ("out_binary_masks", "out_probs", "out_obj_ids", "out_boxes_xywh"):
        if key not in outputs:
            raise KeyError(f"SAM outputs missing {key}")
    masks = np.asarray(as_numpy(outputs["out_binary_masks"]))
    scores = np.asarray(as_numpy(outputs["out_probs"])).reshape(-1)
    obj_ids = np.asarray(as_numpy(outputs["out_obj_ids"])).reshape(-1)
    boxes = np.asarray(as_numpy(outputs["out_boxes_xywh"]))
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    if boxes.ndim == 1:
        boxes = boxes[None, ...]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"SAM output boxes must be Nx4 normalized XYWH, got {boxes.shape}")
    if masks.ndim != 3 or masks.shape[-2:] != (height, width):
        raise ValueError(f"SAM output mask dimensions unexpected: {masks.shape}, expected Nx{height}x{width}")
    if len(masks) != len(scores) or len(masks) != len(obj_ids) or len(masks) != len(boxes) or len(masks) == 0:
        raise ValueError("SAM masks, scores, object IDs and boxes must be non-empty and aligned")
    if not np.isfinite(scores).all():
        raise ValueError("SAM output scores contain non-finite values")
    overlaps = np.asarray([box_iou_xywh(box, prompt_box_xywh) for box in boxes], dtype=float)
    eligible = overlaps > 0
    if not eligible.any():
        raise ValueError("no SAM output box overlaps the requested template box")
    eligible_scores = np.where(eligible, scores, -np.inf)
    best = int(np.argmax(eligible_scores))
    return masks[best].astype(bool), float(scores[best]), int(obj_ids[best]), float(overlaps[best]), len(masks)


def derived_rope_allowed_keys() -> list[str]:
    """Exact derived-RoPE buffer keys the official SAM source reconstructs at init.

    These are rotary-frequency buffers built deterministically from freqs_cis
    during module initialization, not learned checkpoint weights.
    Blocks 0..31 x {freqs_cis_real, freqs_cis_imag} = 64 unique keys.
    """
    keys = [
        f"detector.backbone.vision_backbone.trunk.blocks.{block}.attn.freqs_cis_{suffix}"
        for block in range(32)
        for suffix in ("real", "imag")
    ]
    if len(keys) != 64 or len(set(keys)) != 64:
        raise RuntimeError("derived-RoPE allowlist must be exactly 64 unique keys")
    return keys


def checkpoint_key_gate(composite: list[dict[str, Any]], allowed_rope_keys: list[str]) -> list[str]:
    """Fail unless the single composite load misses exactly the 64 derived-RoPE keys."""
    problems: list[str] = []
    missing: list[str] = []
    unexpected: list[str] = []
    if len(composite) != 1:
        problems.append(
            "expected exactly one complete Sam3MultiplexTrackingWithInteractivity checkpoint load, "
            f"observed {len(composite)}"
        )
    for entry in composite:
        missing.extend(entry["missing_keys"])
        unexpected.extend(entry["unexpected_keys"])
    allowed = set(allowed_rope_keys)
    if len(missing) != len(allowed):
        problems.append(f"missing-key count is {len(missing)}, expected exactly {len(allowed)}")
    duplicates = sorted({key for key in missing if missing.count(key) > 1})
    if duplicates:
        problems.append(f"duplicate missing keys: {duplicates}")
    off_pattern = sorted({key for key in missing if key not in allowed})
    if off_pattern:
        problems.append(f"missing keys outside the derived-RoPE allowlist: {off_pattern}")
    if unexpected:
        problems.append(f"unexpected keys: {sorted(set(unexpected))}")
    return problems


def build_predictor_with_checkpoint_audit(builder, torch, checkpoint: Path, output_dir: Path):
    """Call the real official builder while preserving exact loader diagnostics."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    log_buffer = io.StringIO()
    log_handler = logging.StreamHandler(log_buffer)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    original_load = torch.nn.Module.load_state_dict
    diagnostics: list[dict[str, Any]] = []

    def audited_load(module, state_dict, *args, **kwargs):
        result = original_load(module, state_dict, *args, **kwargs)
        diagnostics.append({
            "module_class": type(module).__name__,
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        })
        return result

    torch.nn.Module.load_state_dict = audited_load
    predictor = None
    build_error = None
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            predictor = builder(
                checkpoint_path=str(checkpoint),
                compile=False,
                warm_up=False,
                async_loading_frames=False,
            )
    except Exception as exc:
        build_error = f"{type(exc).__name__}: {exc}"
    finally:
        torch.nn.Module.load_state_dict = original_load
        root_logger.removeHandler(log_handler)
        log_handler.close()

    log_text = "".join((stdout_buffer.getvalue(), stderr_buffer.getvalue(), log_buffer.getvalue()))
    (output_dir / "predictor_builder.log").write_text(log_text, encoding="utf-8")
    composite = [
        entry for entry in diagnostics
        if entry["module_class"] == "Sam3MultiplexTrackingWithInteractivity"
    ]
    allowed_rope_keys = derived_rope_allowed_keys()
    key_gate_problems = checkpoint_key_gate(composite, allowed_rope_keys)
    diagnostic_report = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_target_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "builder": "sam3.model_builder.build_sam3_multiplex_video_predictor",
        "load_state_dict_calls": diagnostics,
        "composite_model_load_count": len(composite),
        "composite_model_missing_keys": [key for entry in composite for key in entry["missing_keys"]],
        "composite_model_unexpected_keys": [key for entry in composite for key in entry["unexpected_keys"]],
        "derived_rope_allowed_key_count": len(allowed_rope_keys),
        "derived_rope_allowed_keys": allowed_rope_keys,
        "key_gate_disposition": "passed" if not key_gate_problems else "failed: " + "; ".join(key_gate_problems),
        "builder_error": build_error,
    }
    (output_dir / "checkpoint_load_diagnostics.json").write_text(
        json.dumps(diagnostic_report, indent=2) + "\n", encoding="utf-8"
    )
    if build_error:
        raise RuntimeError(f"SAM predictor builder failed; see predictor_builder.log: {build_error}")
    if key_gate_problems:
        raise RuntimeError(
            "checkpoint key gate failed; see checkpoint_load_diagnostics.json: " + "; ".join(key_gate_problems)
        )
    return predictor, diagnostic_report


def start_multiplex_session(predictor, resource_path: str) -> dict[str, Any]:
    """Benchmark-local session bootstrap for the pinned SAM 3.1 multiplex model.

    The pinned ``Sam3BasePredictor.start_session()`` always forwards
    ``offload_state_to_cpu`` to ``model.init_state()``, but the pinned
    ``Sam3MultiplexTrackingWithInteractivity.init_state()`` does not accept
    that argument, so every per-item session failed before prompt inference.
    This adapter calls the model's actual ``init_state()`` with only the
    settings it supports (the frozen one-frame resource directory,
    ``offload_video_to_cpu=False`` and the predictor's configured
    ``async_loading_frames`` value) and registers the returned state in the
    predictor's existing session table using the same fields as
    ``Sam3BasePredictor.start_session()``. Box prompts and session close
    still use the predictor's existing ``handle_request`` path.
    """
    model = getattr(predictor, "model", None)
    if model is None or not hasattr(model, "init_state"):
        raise RuntimeError("predictor model does not expose init_state(); cannot bootstrap a session")
    session_table = getattr(predictor, "_all_inference_states", None)
    if not isinstance(session_table, dict):
        raise RuntimeError("predictor has no _all_inference_states session table; cannot bootstrap a session")
    init_kwargs = dict(
        resource_path=resource_path,
        offload_video_to_cpu=False,
    )
    if hasattr(predictor, "async_loading_frames"):
        init_kwargs["async_loading_frames"] = predictor.async_loading_frames
    inference_state = model.init_state(**init_kwargs)
    if not isinstance(inference_state, dict):
        raise TypeError(f"init_state() returned {type(inference_state).__name__}, expected the inference state dict")
    session_id = str(uuid.uuid4())
    session_table[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": time.time(),
        "last_use_time": time.time(),
    }
    return {"session_id": session_id}


def nvidia_smi_memory_mib(device_index: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", f"--id={device_index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def nvidia_smi_device_metadata(device_index: int) -> dict[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parts = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
    if len(parts) != 4:
        raise RuntimeError(f"unexpected nvidia-smi metadata row: {result.stdout!r}")
    return {"name": parts[0], "uuid": parts[1], "driver_version": parts[2], "total_memory_mib": parts[3]}


class DeviceMemorySampler:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.baseline_mib = nvidia_smi_memory_mib(device_index)
        self.peak_mib = self.baseline_mib
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, name="sam31-nvidia-smi-sampler", daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.5):
            try:
                self.peak_mib = max(self.peak_mib, nvidia_smi_memory_mib(self.device_index))
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self._stop.set()

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mib = max(self.peak_mib, nvidia_smi_memory_mib(self.device_index))
        if self.error:
            raise RuntimeError(f"whole-device memory sampling failed: {self.error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="local sam3.1_multiplex.pt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1] / "repo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-items", type=int, default=0, help="0 processes every manifest item")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()
    manifest_path = args.manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser()
    repo = args.repo.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.device != "cuda" and not re.fullmatch(r"cuda:[0-9]+", args.device):
        raise ValueError("this experiment requires a CUDA device; use --device cuda or cuda:<index>")
    if output_dir.exists():
        raise FileExistsError(f"use a new output directory; already exists: {output_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not repo.is_dir() or not (repo / "sam3" / "model_builder.py").is_file():
        raise FileNotFoundError(f"official SAM3 repo not found at {repo}")
    sam_repo_commit = git_revision(repo)
    expected_repo_commit = "eef6839bcfc3458dd52ea80535dc6bd62aea86b1"
    if sam_repo_commit != expected_repo_commit:
        raise ValueError(f"SAM repository commit mismatch: expected {expected_repo_commit}, got {sam_repo_commit}")
    manifest, items = load_manifest(manifest_path)
    if args.max_items > 0:
        items = items[: args.max_items]
    assets = validate_assets(manifest_path, items, checkpoint)
    manifest_sha = sha256(manifest_path)
    checkpoint_sha = sha256(checkpoint)
    checkpoint_target = checkpoint.resolve()
    # No Torch/SAM import occurs until all manifest, image, mask, repo, and
    # checkpoint paths have passed validation above.
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable in this host process")
    device_index = 0 if args.device == "cuda" else int(args.device.split(":", 1)[1])
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f"requested CUDA device index {device_index} is unavailable")
    device_name = torch.cuda.get_device_name(device_index)
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    torch.cuda.synchronize(device_index)
    memory_sampler = DeviceMemorySampler(device_index)
    gpu_metadata = nvidia_smi_device_metadata(device_index)
    output_dir.mkdir(parents=True, exist_ok=False)
    memory_sampler.start()

    load_start = time.perf_counter()
    try:
        predictor, checkpoint_diagnostics = build_predictor_with_checkpoint_audit(
            build_sam3_multiplex_video_predictor, torch, checkpoint, output_dir
        )
        torch.cuda.synchronize(device_index)
    except Exception:
        memory_sampler.close()
        raise
    model_load_seconds = time.perf_counter() - load_start
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for item, paths in zip(items, assets):
        sample_id = str(item["sample_id"])
        item_start = time.perf_counter()
        session_id = None
        temp = None
        try:
            with Image.open(paths["image"]) as im:
                width, height = im.size
                rgb = im.convert("RGB")
                temp = tempfile.TemporaryDirectory(prefix="sam31-one-frame-")
                frame_path = Path(temp.name) / "00000.png"
                rgb.save(frame_path, format="PNG")
            gt_mask = binary_mask(paths["ground_truth_mask"])
            base_mask = binary_mask(paths["base_mask"])
            if gt_mask.shape != (height, width) or base_mask.shape != (height, width):
                raise ValueError(f"base/GT masks must match RGB dimensions {(height, width)}")
            x0, y0, x1, y1 = (float(value) for value in item["bbox_xyxy"])
            if x1 > width or y1 > height:
                raise ValueError(f"bbox exceeds source dimensions {(width, height)}")
            box_xywh = [[x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]]
            start = start_multiplex_session(predictor, temp.name)
            session_id = start.get("session_id")
            if not session_id:
                raise RuntimeError(f"SAM start_session returned no session_id: {start}")
            prompt_start = time.perf_counter()
            response = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "bounding_boxes": box_xywh,
                "bounding_box_labels": [1],
            })
            torch.cuda.synchronize(device_index)
            prompt_seconds = time.perf_counter() - prompt_start
            if not isinstance(response, dict) or "outputs" not in response:
                raise TypeError(f"SAM add_prompt response missing outputs: {type(response).__name__}")
            refined_mask, score, output_object_id, selected_box_iou, candidate_count = extract_prediction(
                response["outputs"], height, width, box_xywh[0]
            )
            base_iou = iou(base_mask, gt_mask)
            refined_iou = iou(refined_mask, gt_mask)
            mask_out = output_dir / "masks" / f"{sample_id}.npz"
            mask_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_out, refined_mask=refined_mask.astype(np.uint8))
            rows.append({
                "sample_id": sample_id,
                "prompt_source": str(item.get("prompt_source", "manifest_bbox_xyxy")),
                "width": width,
                "height": height,
                "sam_output_object_id": output_object_id,
                "sam_score": score,
                "candidate_count": candidate_count,
                "selected_box_iou": selected_box_iou,
                "box_selection_rule": "max IoU to requested normalized prompt box; zero-overlap outputs fail",
                "base_iou": base_iou,
                "refined_iou": refined_iou,
                "iou_delta": refined_iou - base_iou,
                "prompt_seconds": prompt_seconds,
                "item_wall_seconds": time.perf_counter() - item_start,
                "mask_artifact": str(mask_out.relative_to(output_dir)),
            })
        except Exception as exc:  # retain per-item failure evidence and continue
            failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
            (output_dir / "failures").mkdir(exist_ok=True)
            (output_dir / "failures" / f"{sample_id}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            if session_id:
                try:
                    predictor.handle_request({"type": "close_session", "session_id": session_id})
                except Exception as exc:
                    failures.append({"sample_id": sample_id, "error": f"close_session: {type(exc).__name__}: {exc}"})
            if temp is not None:
                temp.cleanup()

    torch.cuda.synchronize(device_index)
    peak_allocated = int(torch.cuda.max_memory_allocated(device_index))
    peak_reserved = int(torch.cuda.max_memory_reserved(device_index))
    memory_sampler.close()
    if rows:
        means = {
            "base_iou_mean": float(np.mean([r["base_iou"] for r in rows])),
            "refined_iou_mean": float(np.mean([r["refined_iou"] for r in rows])),
            "mean_iou_delta": float(np.mean([r["iou_delta"] for r in rows])),
            "median_prompt_seconds": float(np.median([r["prompt_seconds"] for r in rows])),
        }
    else:
        means = {"base_iou_mean": None, "refined_iou_mean": None, "mean_iou_delta": None, "median_prompt_seconds": None}
    csv_path = output_dir / "per_object.csv"
    fields = list(rows[0]) if rows else ["sample_id", "error"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema": "sam31-template-refinement-results-v1",
        "started_utc": run_started_at,
        "total_wall_seconds": time.perf_counter() - run_start,
        "dataset": manifest.get("dataset"),
        "split": manifest.get("split"),
        "protocol": "SAM 3.1 multiplex predictor; one-frame session; positive visual XYWH box prompt at frame 0; masks compared against fixed GT",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "checkpoint_path": str(checkpoint),
        "checkpoint_target_path": str(checkpoint_target),
        "checkpoint_sha256": checkpoint_sha,
        "repo_path": str(repo),
        "repo_commit": sam_repo_commit,
        "device": args.device,
        "device_name": device_name,
        "device_index": device_index,
        "gpu_metadata": gpu_metadata,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "model_load_seconds": model_load_seconds,
        "processed_count": len(rows),
        "failed_count": len(failures),
        "failures": failures,
        "metrics": means,
        "peak_gpu_allocated_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "whole_device_gpu_memory_baseline_mib": memory_sampler.baseline_mib,
        "whole_device_gpu_memory_peak_mib": memory_sampler.peak_mib,
        "whole_device_gpu_memory_delta_mib": memory_sampler.peak_mib - memory_sampler.baseline_mib,
        "results_csv": str(csv_path),
        "mask_dir": str(output_dir / "masks"),
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if rows and not failures else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SAM3.1 benchmark preflight/runtime failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
