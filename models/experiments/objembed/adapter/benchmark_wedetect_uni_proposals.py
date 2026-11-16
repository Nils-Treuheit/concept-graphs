#!/usr/bin/env python3
"""GPU-only WeDetect-Uni proposal benchmark on frozen ReplicaSSG frames.

Loads SimpleYOLOWorldDetector from the installed official
models/experiments/objembed/wedetect-repo/generate_proposal.py and runs the
official base-Uni proposal path (FP32, CUDA 0, 300 proposals per frame,
official letterbox preprocessing and official NMS) on the frozen
five-scene/250-frame/2204-object/49-category ReplicaSSG manifest. It reports
proposal recall against GT boxes at IoU 0.5/0.75 for K=100/300, mean best
IoU, average proposal count, and per-scene values. A separate
proposal-conditioned feature diagnostic assigns each GT object view the
embedding of its highest-IoU proposal among the 300 (even when IoU < 0.5) and
scores it with the existing _association_and_graph_metrics helper on fixed
ReplicaSSG predicates; this is a GT-matched proposal embedding diagnostic,
not autonomous map association and not predicate prediction.

Separate from the completed ObjEmbed-2B/4B oracle-box runs. Run only after
the workflow release/controller gate passes in host context. No CPU fallback.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
WEDETECT_REPO = ROOT / "models/experiments/objembed/wedetect-repo"
OFFICIAL_GENERATE_PROPOSAL = WEDETECT_REPO / "generate_proposal.py"
ASSOCIATION_HELPER = ROOT / "conceptgraph/scripts/replica_clip_benchmark.py"
DEFAULT_HF_HOME = Path("/mnt/HDD1/unsloth_and_hugging_face_models/huggingface")
DEFAULT_MANIFEST = ROOT / "models/experiments/objembed/data/replicassg-five-scene-v1.json"
DEFAULT_OUT_ROOT = ROOT / "models/experiments/objembed/results"
WEDETECT_REPO_ID = "fushh7/WeDetect"
WEDETECT_REVISION = "3b9d0670cbf300d0b48f900ae87a6c4785d88939"
CHECKPOINT_NAME = "wedetect_base_uni.pth"
EXPECTED_MANIFEST_SHA256 = "b04b2e90e15a96d5a1e04154c7fbe1935135336b1aee5e8903c1e5aab77ab277"
EXPECTED = {"scene_count": 5, "frame_view_count": 250,
            "visible_object_view_count": 2204, "category_count": 49}
NUM_PROPOSALS = 300
PROPOSAL_KS = (100, 300)
IOU_THRESHOLDS = (0.5, 0.75)
CSV_HEADER = [
    "view_id", "scene", "frame_index", "object_id", "category", "n_proposals",
    "best_rank_k100", "best_iou_k100", "hit_iou_0.5_k100", "hit_iou_0.75_k100",
    "best_rank_k300", "best_iou_k300", "hit_iou_0.5_k300", "hit_iou_0.75_k300",
    "assigned_score", "assigned_x1", "assigned_y1", "assigned_x2", "assigned_y2",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path.resolve(strict=True)


def round6(value: Any) -> float:
    return float(f"{float(value):.6g}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--hf-home", type=Path,
                   default=Path(os.getenv("HF_HOME", str(DEFAULT_HF_HOME))))
    p.add_argument("--output-dir", type=Path,
                   default=DEFAULT_OUT_ROOT /
                   f"wedetect-uni-proposal-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    return p.parse_args()


def validate_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Verify schema, required SHA-256, exact counts, and every source RGB file.

    Runs before any model import so a manifest or image mismatch fails closed
    without touching the model stack.
    """
    path = require(path, "frozen ReplicaSSG manifest")
    manifest_sha = sha256_file(path)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"manifest SHA-256 mismatch: {manifest_sha}, "
            f"expected frozen value {EXPECTED_MANIFEST_SHA256}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "objembed-replicassg-gt-manifest/v1" or data.get("status") != "frozen_before_model_inference":
        raise ValueError("manifest schema/status is not the accepted frozen ReplicaSSG input")
    counts = data.get("counts", {})
    for key, expected in EXPECTED.items():
        if counts.get(key) != expected:
            raise ValueError(f"manifest count {key}={counts.get(key)!r}, expected {expected}")
    scenes = data.get("split", {}).get("evaluated_scenes", [])
    if len(scenes) != 5 or len(set(scenes)) != 5:
        raise ValueError(f"expected five unique frozen scenes, got {scenes}")
    rows = data.get("object_views", [])
    if len(rows) != EXPECTED["visible_object_view_count"]:
        raise ValueError(f"expected 2204 object views, got {len(rows)}")
    view_ids = [row.get("view_id") for row in rows]
    if len(set(view_ids)) != len(rows):
        raise ValueError("manifest contains duplicate view IDs")
    categories = sorted({str(row["category"]) for row in rows})
    if len(categories) != EXPECTED["category_count"]:
        raise ValueError(f"expected 49 categories, found {len(categories)}")
    verified_files: dict[str, tuple[int, str]] = {}
    for row in rows:
        item = row.get("source_rgb", {})
        source = require(Path(item.get("path", "")), f"source RGB for {row.get('view_id')}")
        identity = str(source)
        if identity not in verified_files:
            verified_files[identity] = (source.stat().st_size, sha256_file(source))
        if verified_files[identity] != (item.get("bytes"), item.get("sha256")):
            raise ValueError(f"source RGB byte size/SHA-256 mismatch: {source}")
        box = row.get("box_crop", {}).get("region_box_xyxy")
        if not isinstance(box, list) or len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
            raise ValueError(f"invalid source-frame GT box for {row.get('view_id')}: {box}")
    return data, rows, categories


def group_frames(rows: list[dict[str, Any]]) -> list[tuple[str, str, int, list[dict[str, Any]]]]:
    """Group GT object views by unique source RGB image (manifest order)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["source_rgb"]["path"], []).append(row)
    ordered: list[tuple[str, str, int, list[dict[str, Any]]]] = []
    for path, group_rows in groups.items():
        scene_ids = {row["scene"] for row in group_rows}
        frame_ids = {int(row["frame_index"]) for row in group_rows}
        if len(scene_ids) != 1 or len(frame_ids) != 1:
            raise ValueError(
                f"source RGB {path} maps to multiple scenes/frames: {sorted(map(str, scene_ids))} {sorted(frame_ids)}")
        ordered.append((path, scene_ids.pop(), frame_ids.pop(), group_rows))
    if len(ordered) != EXPECTED["frame_view_count"]:
        raise ValueError(f"expected 250 unique source frames, got {len(ordered)}")
    return ordered


def resolve_checkpoint(hf_home: Path) -> tuple[Path, str, int]:
    snapshot = hf_home / "hub" / "models--fushh7--WeDetect" / "snapshots" / WEDETECT_REVISION
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing local HF snapshot {WEDETECT_REPO_ID}@{WEDETECT_REVISION}: {snapshot}")
    checkpoint = require(snapshot / CHECKPOINT_NAME, f"WeDetect checkpoint {CHECKPOINT_NAME}")
    return checkpoint, sha256_file(checkpoint), checkpoint.stat().st_size


def repo_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-c", f"safe.directory={str(repo)}",
                              "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def remap_official_checkpoint_keys(state: dict[str, Any]) -> None:
    """Reproduce the official generate_proposal.py checkpoint key remap in place."""
    keys = list(state.keys())
    for key in keys:
        if "backbone" in key:
            new_key = key.replace("backbone.image_model.model.", "backbone.")
            state[new_key] = state.pop(key)
    keys = list(state.keys())
    for key in keys:
        if "bbox_head" in key:
            new_key = key.replace("bbox_head.head_module.", "bbox_head.")
            new_key = new_key.replace("0.2.", "0.6.")
            new_key = new_key.replace("1.2.", "1.6.")
            new_key = new_key.replace("2.2.", "2.6.")
            new_key = new_key.replace("1.bn", "4")
            new_key = new_key.replace("1.conv", "3")
            new_key = new_key.replace("0.bn", "1")
            new_key = new_key.replace("0.conv", "0")
            state[new_key] = state.pop(key)


def load_official_detector(checkpoint: Path, device: torch.device) -> tuple[Any, dict[str, Any], float]:
    """Build the official base-Uni detector in FP32 and load the checkpoint."""
    sys.path.insert(0, str(WEDETECT_REPO))
    from generate_proposal import SimpleYOLOWorldDetector

    started = time.perf_counter()
    model = SimpleYOLOWorldDetector(backbone_size="base", prompt_dim=768,
                                    num_prompts=256, num_proposals=NUM_PROPOSALS)
    try:
        state = torch.load(str(checkpoint), map_location="cpu")
    except Exception as exc:
        raise ValueError(f"cannot load checkpoint {checkpoint} via the official torch.load path: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint {checkpoint} is not a plain state-dict mapping: {type(state)}")
    remap_official_checkpoint_keys(state)
    # Official generate_proposal.py uses strict=False and runs with the
    # returned incompatibility report; key-only mismatches are therefore
    # recorded here, not rejected. load_state_dict still raises on tensor
    # shape mismatches, which propagates through the failure path unchanged.
    msg = model.load_state_dict(state, strict=False)
    missing_keys = list(msg.missing_keys)
    unexpected_keys = list(msg.unexpected_keys)
    model = model.to(device).eval()
    torch.cuda.synchronize(device)
    report = {
        "remap_source": "official generate_proposal.py backbone/bbox_head key remap, reproduced verbatim",
        "strict": False,
        "missing_keys": missing_keys,
        "missing_key_count": len(missing_keys),
        "unexpected_keys": unexpected_keys,
        "unexpected_key_count": len(unexpected_keys),
    }
    return model, report, time.perf_counter() - started


def iou_matrix(gt_boxes: np.ndarray, proposal_boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (M, 4) GT boxes against (N, 4) proposal boxes (xyxy)."""
    if proposal_boxes.shape[0] == 0:
        return np.zeros((gt_boxes.shape[0], 0), dtype=np.float64)
    gx1, gy1, gx2, gy2 = (gt_boxes[:, i].astype(np.float64) for i in range(4))
    px1, py1, px2, py2 = (proposal_boxes[:, i].astype(np.float64) for i in range(4))
    ix1 = np.maximum(gx1[:, None], px1[None, :])
    ix2 = np.minimum(gx2[:, None], px2[None, :])
    iy1 = np.maximum(gy1[:, None], py1[None, :])
    iy2 = np.minimum(gy2[:, None], py2[None, :])
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
    area_g = (gx2 - gx1) * (gy2 - gy1)
    area_p = (px2 - px1) * (py2 - py1)
    union = area_g[:, None] + area_p[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    return np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)


def latency_summary(seconds: list[float]) -> dict[str, float]:
    array = np.asarray(seconds, dtype=np.float64)
    return {
        "mean": round6(array.mean()), "median": round6(np.median(array)),
        "p95": round6(np.percentile(array, 95)), "min": round6(array.min()),
        "max": round6(array.max()), "std": round6(array.std()),
    }


def score_group(records: list[dict[str, Any]], frame_seconds: list[float]) -> dict[str, Any]:
    hits = {f"hit_iou_{t:g}_k{k}": np.asarray([r[f"hit_iou_{t:g}_k{k}"] for r in records], dtype=np.float64)
            for t in IOU_THRESHOLDS for k in PROPOSAL_KS}
    bests = {f"best_iou_k{k}": np.asarray([r[f"best_iou_k{k}"] for r in records], dtype=np.float64)
             for k in PROPOSAL_KS}
    return {
        "object_view_count": len(records),
        "frame_count": len(frame_seconds),
        "average_proposal_count": round6(float(np.mean([r["n_proposals"] for r in records]))),
        "proposal_recall_iou_0.5_k100": round6(float(hits["hit_iou_0.5_k100"].mean())),
        "proposal_recall_iou_0.75_k100": round6(float(hits["hit_iou_0.75_k100"].mean())),
        "proposal_recall_iou_0.5_k300": round6(float(hits["hit_iou_0.5_k300"].mean())),
        "proposal_recall_iou_0.75_k300": round6(float(hits["hit_iou_0.75_k300"].mean())),
        "mean_best_iou_k100": round6(float(bests["best_iou_k100"].mean())),
        "mean_best_iou_k300": round6(float(bests["best_iou_k300"].mean())),
    }


def main() -> int:
    args = parse_args()
    args.manifest = args.manifest.expanduser().resolve(strict=True)
    started_utc = datetime.now(timezone.utc)
    total_started = time.perf_counter()
    failures: list[str] = []

    # Validate manifest, source images and checkpoint before any model import.
    data, rows, categories = validate_manifest(args.manifest)
    frame_groups = group_frames(rows)
    hf_home = args.hf_home.expanduser().resolve()
    checkpoint, checkpoint_sha, checkpoint_bytes = resolve_checkpoint(hf_home)
    require(OFFICIAL_GENERATE_PROPOSAL, "official WeDetect generate_proposal.py")
    require(ASSOCIATION_HELPER, "_association_and_graph_metrics helper module")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable; this benchmark has no CPU fallback")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    commit = repo_commit(WEDETECT_REPO)
    if commit is None:
        failures.append(f"could not read git HEAD of {WEDETECT_REPO}; repo commit recorded as null")
    try:
        import torchvision
        torchvision_version: str | None = torchvision.__version__
    except Exception:
        torchvision_version = None

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)  # unique new directory; refuse overwrite
    (output_dir / "run-input.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "expected_counts": EXPECTED,
        "model_repo_id": WEDETECT_REPO_ID,
        "model_revision": WEDETECT_REVISION,
        "checkpoint": CHECKPOINT_NAME,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_bytes,
        "wedetect_repo_path": str(WEDETECT_REPO),
        "wedetect_repo_commit": commit,
        "num_proposals": NUM_PROPOSALS,
        "proposal_ks": list(PROPOSAL_KS),
        "iou_thresholds": list(IOU_THRESHOLDS),
        "dtype": "float32",
        "device": "cuda:0",
    }, indent=2) + "\n", encoding="utf-8")

    per_frame_path = output_dir / "per_frame_proposals.json"
    csv_path = output_dir / "per_object.csv"
    assigned_path = output_dir / "assigned_proposal_embeddings.npy"
    completed_frames = 0
    completed_views = 0
    try:
        torch.cuda.reset_peak_memory_stats(device)
        model, state_report, model_load_seconds = load_official_detector(checkpoint, device)
        resident_allocated_gib = torch.cuda.memory_allocated(device) / 1024**3
        load_peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        load_peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
        torch.cuda.reset_peak_memory_stats(device)

        object_records: list[dict[str, Any]] = []
        assigned_embeddings: list[np.ndarray] = []
        frame_seconds: list[float] = []
        proposal_counts: list[int] = []
        embedding_dim: int | None = None

        with per_frame_path.open("w", encoding="utf-8") as per_frame_file, \
                csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            per_frame_file.write('{\n  "schema": "wedetect-uni-replicassg-per-frame-proposals/v1",\n')
            per_frame_file.write(f'  "created_utc": {json.dumps(started_utc.isoformat())},\n')
            per_frame_file.write(f'  "source_frame_count": {len(frame_groups)},\n')
            per_frame_file.write(f'  "num_proposals_requested": {NUM_PROPOSALS},\n')
            per_frame_file.write('  "frames": [\n')
            writer = csv.writer(csv_file)
            writer.writerow(CSV_HEADER)

            loop_started = time.perf_counter()
            for frame_idx, (image_path, scene, frame_index, group_rows) in enumerate(frame_groups):
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                torch.cuda.synchronize(device)
                frame_started = time.perf_counter()
                with torch.inference_mode():
                    outputs = model([image])
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - frame_started
                image.close()
                record = outputs[0]
                bboxes = np.asarray(record["bboxes"].detach().float().cpu().numpy(), dtype=np.float64)
                scores = np.asarray(record["scores"].detach().float().cpu().numpy(), dtype=np.float64)
                embeddings = np.asarray(record["embeddings"].detach().float().cpu().numpy(), dtype=np.float64)
                if bboxes.ndim != 2 or bboxes.shape[1] != 4 or scores.ndim != 1 or embeddings.ndim != 2:
                    raise ValueError(f"unexpected WeDetect-Uni output shapes for {image_path}: "
                                     f"{bboxes.shape} {scores.shape} {embeddings.shape}")
                if bboxes.shape[0] != scores.shape[0] or bboxes.shape[0] != embeddings.shape[0]:
                    raise ValueError(f"inconsistent WeDetect-Uni proposal counts for {image_path}")
                if bboxes.shape[0] == 0:
                    raise ValueError(f"WeDetect-Uni produced zero proposals for {image_path}")
                if not (np.isfinite(bboxes).all() and np.isfinite(scores).all() and np.isfinite(embeddings).all()):
                    raise ValueError(f"non-finite WeDetect-Uni outputs for {image_path}")
                if embeddings.shape[1] != 768:
                    if embedding_dim is None:
                        raise ValueError(f"unexpected WeDetect-Uni proposal embedding dimension {embeddings.shape[1]} for {image_path}")
                    raise ValueError(f"inconsistent WeDetect-Uni proposal embedding dimension {embeddings.shape[1]} for {image_path}")
                embedding_dim = int(embeddings.shape[1])

                order = np.argsort(-scores, kind="stable")  # official output is already score-ranked; stable no-op otherwise
                ranked_scores = scores[order]
                ranked_boxes = bboxes[order]
                ranked_embeddings = embeddings[order]
                gt_boxes = np.asarray([row["box_crop"]["region_box_xyxy"] for row in group_rows],
                                      dtype=np.float64)
                iou_full = iou_matrix(gt_boxes, ranked_boxes)
                per_k: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                for k in PROPOSAL_KS:
                    iou_k = iou_full[:, :k]
                    best_idx_k = iou_k.argmax(axis=1)  # first max wins score-order ties
                    best_iou_k = iou_k[np.arange(len(group_rows)), best_idx_k]
                    per_k[k] = (best_idx_k, best_iou_k)

                for row_idx, row in enumerate(group_rows):
                    record_entry: dict[str, Any] = {
                        "view_id": row["view_id"], "scene": scene, "frame_index": frame_index,
                        "object_id": row["object_id"], "category": row["category"],
                        "n_proposals": int(bboxes.shape[0]),
                    }
                    for k in PROPOSAL_KS:
                        best_idx_k, best_iou_k = per_k[k]
                        record_entry[f"best_rank_k{k}"] = int(best_idx_k[row_idx]) + 1
                        record_entry[f"best_iou_k{k}"] = float(best_iou_k[row_idx])
                        for t in IOU_THRESHOLDS:
                            record_entry[f"hit_iou_{t:g}_k{k}"] = int(best_iou_k[row_idx] >= t)
                    assigned_idx = int(per_k[300][0][row_idx])
                    record_entry["assigned_score"] = float(ranked_scores[assigned_idx])
                    record_entry["assigned_box"] = [float(v) for v in ranked_boxes[assigned_idx]]
                    object_records.append(record_entry)
                    assigned_embeddings.append(ranked_embeddings[assigned_idx].astype(np.float32))
                    writer.writerow([
                        row["view_id"], scene, frame_index, row["object_id"], row["category"],
                        int(bboxes.shape[0]),
                        record_entry["best_rank_k100"], f"{record_entry['best_iou_k100']:.6f}",
                        record_entry["hit_iou_0.5_k100"], record_entry["hit_iou_0.75_k100"],
                        record_entry["best_rank_k300"], f"{record_entry['best_iou_k300']:.6f}",
                        record_entry["hit_iou_0.5_k300"], record_entry["hit_iou_0.75_k300"],
                        f"{record_entry['assigned_score']:.6f}",
                    ] + [f"{v:.4f}" for v in record_entry["assigned_box"]])

                frame_record = {
                    "source_rgb": str(image_path), "scene": scene, "frame_index": frame_index,
                    "object_view_count": len(group_rows), "inference_seconds": round6(seconds),
                    "proposal_count": int(bboxes.shape[0]),
                    "proposals": [
                        {"rank": rank + 1, "score": round6(ranked_scores[rank]),
                         "bbox_xyxy": [round6(v) for v in ranked_boxes[rank]]}
                        for rank in range(len(ranked_scores))
                    ],
                }
                separator = "," if frame_idx < len(frame_groups) - 1 else ""
                per_frame_file.write(json.dumps(frame_record, allow_nan=False) + separator + "\n")
                frame_seconds.append(seconds)
                proposal_counts.append(int(bboxes.shape[0]))
                completed_frames += 1
                completed_views += len(group_rows)
                if (frame_idx + 1) % 25 == 0 or frame_idx == len(frame_groups) - 1:
                    print(f"[wedetect-uni] {frame_idx + 1}/{len(frame_groups)} frames", flush=True)
            per_frame_file.write("  ]\n}\n")
            inference_loop_seconds = time.perf_counter() - loop_started

        torch.cuda.synchronize(device)
        inference_peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        inference_peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3

        assigned = np.stack(assigned_embeddings)  # (2204, 768), manifest order
        np.save(assigned_path, assigned)
        feature_tensor = torch.from_numpy(assigned).float()
        association = __import__("conceptgraph.scripts.replica_clip_benchmark",
                                 fromlist=["_association_and_graph_metrics"])._association_and_graph_metrics(
            feature_tensor, [{**row, "frame": row["frame_index"]} for row in rows],
            data["ground_truth_relationships"], data["split"]["evaluated_scenes"],
        )
        del model, assigned, feature_tensor
        torch.cuda.empty_cache()

        scenes = data["split"]["evaluated_scenes"]
        metrics = {
            "overall": {
                **score_group(object_records, frame_seconds),
                "total_proposal_count": int(sum(proposal_counts)),
                "proposal_count": latency_summary(proposal_counts),
            },
            "per_scene": {scene: score_group([r for r in object_records if r["scene"] == scene],
                                             [s for s, (_, sc, _, g) in zip(frame_seconds, frame_groups) if sc == scene])
                          for scene in scenes},
        }
        results = {
            "schema": "wedetect-uni-replicassg-proposal-run/v1",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "started_utc": started_utc.isoformat(),
            "configuration": {
                "manifest": str(args.manifest),
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "expected_counts": EXPECTED,
                "counts": data["counts"],
                "split": data["split"],
                "categories": categories,
                "num_proposals": NUM_PROPOSALS,
                "proposal_ks": list(PROPOSAL_KS),
                "iou_thresholds": list(IOU_THRESHOLDS),
                "dtype": "float32",
                "device": "cuda:0",
                "preprocessing": "official generate_proposal.py letterbox (640x640, RGB/255) and head_predict NMS (score threshold 0.0, topk 30000, batched NMS IoU 0.7, top 300)",
                "gt_box_source": "manifest source-frame region_box_xyxy on the full RGB frame",
            },
            "provenance": {
                "model_repo_id": WEDETECT_REPO_ID,
                "model_revision": WEDETECT_REVISION,
                "snapshot_path": str(checkpoint.parent),
                "checkpoint": CHECKPOINT_NAME,
                "checkpoint_bytes": checkpoint_bytes,
                "checkpoint_sha256": checkpoint_sha,
                "wedetect_repo_path": str(WEDETECT_REPO),
                "wedetect_repo_commit": commit,
                "hf_home": str(hf_home),
                "torch_version": torch.__version__,
                "torchvision_version": torchvision_version,
                "cuda_version": torch.version.cuda,
            },
            "state_dict": state_report,
            "device": {
                "name": properties.name, "uuid": str(properties.uuid),
                "total_memory_gib": properties.total_memory / 1024**3,
                "major": properties.major, "minor": properties.minor,
            },
            "counts": {
                "source_frame_count": len(frame_groups),
                "object_view_count": len(object_records),
                "category_count": len(categories),
                "scene_count": len(scenes),
                "total_proposal_count": int(sum(proposal_counts)),
                "average_proposal_count": round6(float(np.mean(proposal_counts))) if proposal_counts else None,
                "proposal_embedding_dimension": embedding_dim,
            },
            "latency": {
                "per_frame_inference_seconds": latency_summary(frame_seconds),
                "frame_count": len(frame_seconds),
            },
            "metrics": metrics,
            "proposal_conditioned_feature_diagnostic": {
                "label": (
                    "GT-matched proposal embedding diagnostic: every GT object view is assigned the "
                    "embedding of its highest-IoU WeDetect-Uni proposal among the frame's score-ranked "
                    "300, even when IoU is below 0.5. Association and scene-graph metrics reuse the fixed "
                    "ReplicaSSG predicates via _association_and_graph_metrics. This is not autonomous map "
                    "association and not predicate prediction."
                ),
                "assignment_rule": "argmax bbox IoU over the score-ranked proposals; ties go to the higher-scored proposal",
                "coverage_iou_0.5": round6(float(np.mean([r["hit_iou_0.5_k300"] for r in object_records]))),
                "association_and_graph_metrics": association,
            },
            "timing_seconds": {
                "model_load": round6(model_load_seconds),
                "inference_loop": round6(inference_loop_seconds),
                "total_wall": round6(time.perf_counter() - total_started),
            },
            "resources": {
                "model_resident_allocated_gib": round6(resident_allocated_gib),
                "load_peak_allocated_gib": round6(load_peak_allocated_gib),
                "load_peak_reserved_gib": round6(load_peak_reserved_gib),
                "inference_peak_allocated_gib": round6(inference_peak_allocated_gib),
                "inference_peak_reserved_gib": round6(inference_peak_reserved_gib),
                "peak_allocated_gib": round6(max(load_peak_allocated_gib, inference_peak_allocated_gib)),
                "peak_reserved_gib": round6(max(load_peak_reserved_gib, inference_peak_reserved_gib)),
            },
            "failures": failures,
            "output_files": {
                "results": str(output_dir / "results.json"),
                "per_object_csv": str(csv_path),
                "per_frame_proposals": str(per_frame_path),
                "assigned_embeddings": str(assigned_path),
                "run_input": str(output_dir / "run-input.json"),
            },
        }
        results_path = output_dir / "results.json"
        if results_path.exists():
            raise FileExistsError(f"refusing to overwrite {results_path}")
        results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "complete", "output_dir": str(output_dir),
                          "average_proposal_count": metrics["overall"]["average_proposal_count"],
                          "proposal_recall_iou_0.5_k300": metrics["overall"]["proposal_recall_iou_0.5_k300"],
                          "proposal_recall_iou_0.75_k300": metrics["overall"]["proposal_recall_iou_0.75_k300"]},
                         indent=2))
        return 0
    except BaseException as exc:
        (output_dir / "failure.json").write_text(json.dumps({
            "status": "failed",
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "completed_frames": completed_frames,
            "completed_object_views": completed_views,
            "partial_artifacts": [str(p) for p in (per_frame_path, csv_path, assigned_path) if p.exists()],
        }, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    raise SystemExit(main())
