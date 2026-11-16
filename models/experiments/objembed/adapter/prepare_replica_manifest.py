#!/usr/bin/env python3
"""Freeze the ReplicaSSG test views for ObjEmbed/WeDetect without model work.

This uses the project's existing CPU Open3D raycast and crop helpers. It never
loads an encoder, calls CUDA, or scores a model. Existing crop/manifest outputs
are preserved; use unique output paths for a new preparation attempt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
from conceptgraph.scripts import replica_clip_benchmark as shared  # noqa: E402

EXPECTED_SCENES = ["office_1", "office_3", "office_4", "room_1", "room_2"]
EXPECTED_FRAMES_PER_SCENE = 50
EXPECTED_TOTAL_FRAMES = 250
EXPECTED_OBJECT_VIEWS = 2204
EXPECTED_CATEGORIES = 49


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _unique_output(path: Path, label: str) -> None:
    if path.exists():
        raise FileExistsError(
            f"{label} already exists; preserving prior artifacts and refusing to overwrite: {path}"
        )


def _available_frames(image_root: Path, scene: str) -> tuple[list[int], int]:
    scene_root = image_root / scene.replace("_", "")
    paths = sorted(
        (scene_root / "results").glob("frame*.jpg"),
        key=lambda path: int(shared.re.search(r"frame(\d+)", path.stem).group(1)),
    )
    if not paths:
        raise FileNotFoundError(f"no frame*.jpg images found for {scene}: {scene_root}")
    count = len(paths)
    selected = np.arange(count)
    if count > EXPECTED_FRAMES_PER_SCENE:
        selected = np.linspace(0, count - 1, EXPECTED_FRAMES_PER_SCENE, dtype=int)
    return [int(value) for value in selected], count


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    image_root = args.image_root.expanduser().resolve(strict=True)
    semantic_root = args.semantic_root.expanduser().resolve(strict=True)
    annotation_root = args.annotations.expanduser().resolve(strict=True)
    split_path = annotation_root / "test_scans.txt"
    requested_test_scenes = [
        line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unavailable = [scene for scene in EXPECTED_SCENES if scene not in requested_test_scenes]
    if unavailable:
        raise ValueError(f"fixed scenes are absent from ReplicaSSG test_scans.txt: {unavailable}")

    args.scenes = list(EXPECTED_SCENES)
    args.max_frames_per_scene = EXPECTED_FRAMES_PER_SCENE
    objects, relationships, scenes = shared._require_dataset(args)
    if scenes != EXPECTED_SCENES:
        raise ValueError(f"scene order differs from frozen split: {scenes}")

    selected_frames = {}
    scene_sources = {}
    for scene in scenes:
        frames, input_count = _available_frames(image_root, scene)
        if len(frames) != EXPECTED_FRAMES_PER_SCENE:
            raise ValueError(f"{scene} produced {len(frames)} frames, expected {EXPECTED_FRAMES_PER_SCENE}")
        selected_frames[scene] = frames
        scene_sources[scene] = {
            "input_frame_count": input_count,
            "selected_frame_indices": frames,
            "semantic_mesh": file_record(semantic_root / scene / "habitat/mesh_semantic.ply"),
            "semantic_info": file_record(semantic_root / scene / "habitat/info_semantic.json"),
        }

    if args.crop_cache.exists() and any(args.crop_cache.iterdir()):
        raise FileExistsError(
            f"crop cache is nonempty; prior artifacts are preserved: {args.crop_cache}"
        )
    args.crop_cache.mkdir(parents=True, exist_ok=True)
    rows, graph_data = shared._render_crop_rows(args, objects, relationships, scenes)
    unique_categories = sorted({str(row["label"]) for row in rows})
    sampled_frame_rows = [
        {"scene": scene, "frame_index": frame}
        for scene in scenes for frame in selected_frames[scene]
    ]
    actual_counts = {
        "scene_count": len(scenes),
        "frames_per_scene": {scene: len(selected_frames[scene]) for scene in scenes},
        "frame_view_count": len(sampled_frame_rows),
        "visible_object_view_count": len(rows),
        "category_count": len(unique_categories),
    }
    expected_counts = {
        "scene_count": 5,
        "frame_view_count": EXPECTED_TOTAL_FRAMES,
        "visible_object_view_count": EXPECTED_OBJECT_VIEWS,
        "category_count": EXPECTED_CATEGORIES,
    }
    if any(actual_counts[key] != value for key, value in expected_counts.items()):
        raise ValueError(
            "frozen dataset differs from the prior comparison; no manifest was written: "
            f"expected={expected_counts}, actual={actual_counts}"
        )

    object_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[str]] = {}
    for row in rows:
        scene = str(row["scene"])
        frame = int(row["frame"])
        object_id = int(row["object_id"])
        view_id = f"{scene}/frame-{frame:04d}/object-{object_id}"
        label = str(row["label"])
        box_path = Path(row["box_path"]).resolve(strict=True)
        mask_path = Path(row["mask_path"]).resolve(strict=True)
        rgb_path = Path(row["rgb_path"]).resolve(strict=True)
        object_rows.append({
            "view_id": view_id,
            "scene": scene,
            "frame_index": frame,
            "object_id": object_id,
            "track_id": f"{scene}/object-{object_id}",
            "category": label,
            "query_text": f"a photo of a {label}",
            "box_crop": {**file_record(box_path), "region_box_xyxy": row["region_box"]},
            "mask_crop": file_record(mask_path),
            "source_rgb": file_record(rgb_path),
            "gallery_id": "replicassg-five-scene-all-visible-gt-views",
        })
        grouped.setdefault(label, []).append(view_id)

    queries = [
        {
            "query_id": f"category:{label}",
            "category": label,
            "text": f"a photo of a {label}",
            "positive_view_ids": sorted(view_ids),
            "gallery_id": "replicassg-five-scene-all-visible-gt-views",
        }
        for label, view_ids in sorted(grouped.items())
    ]
    tracks: dict[str, dict[str, Any]] = {}
    for row in object_rows:
        tracks.setdefault(row["track_id"], {
            "track_id": row["track_id"],
            "scene": row["scene"],
            "object_id": row["object_id"],
            "category": row["category"],
            "view_ids": [],
        })["view_ids"].append(row["view_id"])

    helper_path = REPO_ROOT / "conceptgraph/scripts/replica_clip_benchmark.py"
    camera_path = image_root / "cam_params.json"
    manifest = {
        "schema": "objembed-replicassg-gt-manifest/v1",
        "status": "frozen_before_model_inference",
        "split": {
            "name": "ReplicaSSG test_scans.txt, local-complete five-scene subset",
            "path": str(split_path.resolve()),
            "sha256": sha256_file(split_path),
            "requested_test_scenes": requested_test_scenes,
            "evaluated_scenes": scenes,
            "excluded_test_scenes": [scene for scene in requested_test_scenes if scene not in scenes],
        },
        "sampling": {
            "algorithm": "numpy.linspace(0, frame_count - 1, 50, dtype=int), matching replica_clip_benchmark._render_crop_rows",
            "frames_per_scene": EXPECTED_FRAMES_PER_SCENE,
            "selected_frame_indices": selected_frames,
            "frame_views": sampled_frame_rows,
        },
        "counts": actual_counts,
        "sources": {
            "preparation_helper": file_record(helper_path),
            "camera_parameters": file_record(camera_path),
            "objects_json": file_record(annotation_root / "objects.json"),
            "relationships_json": file_record(annotation_root / "relationships.json"),
            "scene_meshes_and_metadata": scene_sources,
        },
        "scene_info": graph_data["scene_info"],
        "ground_truth_relationships": graph_data["relationship_rows"],
        "object_views": object_rows,
        "queries": queries,
        "tracks": list(tracks.values()),
        "gallery": {
            "gallery_id": "replicassg-five-scene-all-visible-gt-views",
            "view_ids": [row["view_id"] for row in object_rows],
            "view_count": len(object_rows),
            "track_count": len(tracks),
        },
        "crop_cache": str(args.crop_cache.resolve()),
        "limits": {
            "proposal_source": "Replica semantic-mesh instance IDs raycast at corresponding local RGB camera poses",
            "text_queries": "category labels only; ReplicaSSG provides no natural-language captions or VQA answers",
            "no_model_inference": True,
        },
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=shared.DEFAULT_IMAGE_ROOT)
    parser.add_argument("--semantic-root", type=Path, default=shared.DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--annotations", type=Path, default=shared.DEFAULT_ANNOTATION_ROOT)
    parser.add_argument(
        "--crop-cache",
        type=Path,
        default=REPO_ROOT / "models/experiments/objembed/data/gt-crops-v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models/experiments/objembed/data/replicassg-five-scene-v1.json",
    )
    args = parser.parse_args()
    args.crop_cache = args.crop_cache.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    _unique_output(args.output, "manifest output")
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(args.output),
        "manifest_sha256": sha256_file(args.output),
        "counts": manifest["counts"],
        "crop_cache": manifest["crop_cache"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
