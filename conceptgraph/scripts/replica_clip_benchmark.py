"""GPU benchmark of CLIP-family embeddings on ReplicaSSG ground truth.

Uses the existing Replica RGB sequences and poses plus the official semantic
instance meshes and ReplicaSSG graph annotations. Open3D raycasts the GT mesh
from those camera poses to create oracle instance masks; no detector or
segmenter is used. It measures category-text retrieval, cross-view object
association, and resulting loss in annotated scene-graph edge recovery when
only object identity association is predicted and relation labels are held fixed.

Run after the Replica download is complete:

    python3 /home/nils/Project_Code/codex_opencode_pipeline/src/executor.py release-local
    .venv/bin/python -m conceptgraph.scripts.replica_clip_benchmark

This is intentionally a separate benchmark from the COCO component run. It does
not treat CLIP as a caption generator; ReplicaSSG supplies object labels and
relations, but no human-written object captions.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import struct
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from conceptgraph.scripts.clip_component import (
    Encoder, MODEL_IDS, _check_local_model_cache, _score,
)


DEFAULT_IMAGE_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/Replica")
DEFAULT_SEMANTIC_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/replica_v1")
DEFAULT_ANNOTATION_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/ReplicaSSG/files")
DEFAULT_HF_HOME = Path("/mnt/HDD1/unsloth_and_hugging_face_models/huggingface")
DEFAULT_OUTPUT = Path("/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/results")
MIN_MASK_PIXELS = 32
PAD_PIXELS = 20
PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
PLY_STRUCT_FORMATS = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT,
                        help="Existing Replica RGB-D sequence root (office0, room0, and similar)")
    parser.add_argument("--semantic-root", type=Path, default=DEFAULT_SEMANTIC_ROOT,
                        help="Official Replica release root containing semantic instance meshes")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--hf-home", type=Path, default=Path(os.getenv("HF_HOME", DEFAULT_HF_HOME)))
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_IDS), default=list(MODEL_IDS))
    parser.add_argument("--scenes", nargs="+", help="Default: scenes available in the existing local RGB sequences")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames-per-scene", type=int, default=50,
                        help="Evenly sampled local frames per scene; default is 50")
    parser.add_argument("--check-data-only", action="store_true",
                        help="Check annotation and semantic mesh completeness without rendering or using the GPU")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "replica-clip-benchmark.json")
    parser.add_argument("--crop-cache", type=Path, default=DEFAULT_OUTPUT / "rendered-crops")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required Replica benchmark file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_dataset(args: argparse.Namespace) -> tuple[dict, dict, list[str]]:
    image_root = args.image_root.expanduser().resolve()
    semantic_root = args.semantic_root.expanduser().resolve()
    annotation_root = args.annotations.expanduser().resolve()
    objects = _load_json(annotation_root / "objects.json")
    relations = _load_json(annotation_root / "relationships.json")
    object_rows = _scene_records(objects, "objects")
    available_scenes = sorted(
        scene for scene in object_rows
        if (image_root / scene.replace("_", "") / "results").is_dir()
    )
    scenes = args.scenes or available_scenes
    if not scenes:
        raise FileNotFoundError(f"No local Replica RGB sequences found under {image_root}")
    for scene in scenes:
        local_scene = image_root / scene.replace("_", "")
        if not (local_scene / "traj.txt").is_file() or not (local_scene / "results").is_dir():
            raise FileNotFoundError(f"Missing local Replica sequence or pose file for {scene}: {local_scene}")
        for rel in ("habitat/mesh_semantic.ply", "habitat/info_semantic.json"):
            path = semantic_root / scene / rel
            if not path.is_file():
                raise FileNotFoundError(
                    f"Replica download is incomplete: missing {path}. "
                    "Resume download_replica_benchmark_data.py and rerun after extraction completes."
                )
    if not (image_root / "cam_params.json").is_file():
        raise FileNotFoundError(f"Replica camera intrinsics missing: {image_root / 'cam_params.json'}")
    return objects, relations, scenes


def _scene_records(payload: dict, key: str) -> dict[str, Any]:
    return {str(row["scan"]): row[key] for row in payload["scans"]}


def _object_metadata(scene_objects: list[dict]) -> dict[int, str]:
    result = {}
    for row in scene_objects:
        try:
            object_id = int(row["id"])
        except (TypeError, ValueError, KeyError):
            continue
        label = str(row.get("label", "unknown")).strip().lower()
        if label and label not in {"unknown", "undefined"} and object_id >= 0:
            result[object_id] = label
    return result


def _read_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read Replica PLY vertex positions, triangulated faces, and face object IDs."""
    properties: dict[str, list[tuple[str, str, str | None]]] = {}
    elements: list[tuple[str, int]] = []
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        encoding = None
        current = None
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"PLY header is incomplete: {path}")
            line = raw.decode("ascii").strip().split()
            if not line:
                continue
            if line[0] == "format":
                encoding = line[1]
            elif line[0] == "element":
                current = line[1]
                elements.append((current, int(line[2])))
                properties[current] = []
            elif line[0] == "property" and current:
                if line[1] == "list":
                    properties[current].append(("list", line[2], line[3], line[4]))
                else:
                    properties[current].append(("scalar", line[1], line[2], None))
            elif line[0] == "end_header":
                break
        if encoding not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise ValueError(f"Unsupported PLY encoding {encoding!r} in {path}")
        endian = "<" if encoding == "binary_little_endian" else ">"
        counts = dict(elements)
        if "vertex" not in counts or "face" not in counts:
            raise ValueError(f"Replica semantic PLY must contain vertex and face elements: {path}")
        vertex_properties = properties["vertex"]
        if any(kind != "scalar" for kind, *_ in vertex_properties):
            raise ValueError(f"Unsupported list-valued vertex attributes in {path}")
        vertex_names = [prop[2] for prop in vertex_properties]
        if not all(name in vertex_names for name in ("x", "y", "z")):
            raise ValueError(f"Replica semantic PLY has no x/y/z vertex coordinates: {path}")

        vertices = None
        faces_raw = None
        object_ids = None
        for element, count in elements:
            props = properties[element]
            if element == "vertex":
                if encoding == "ascii":
                    rows = [stream.readline().split() for _ in range(count)]
                    data = np.asarray(rows, dtype=np.float64)
                    vertices = data[:, [vertex_names.index(axis) for axis in ("x", "y", "z")]].astype(np.float32)
                else:
                    dtype = np.dtype([(name, endian + PLY_TYPES[scalar_type])
                                      for _, scalar_type, name, _ in props])
                    data = np.fromfile(stream, dtype=dtype, count=count)
                    vertices = np.column_stack([data[name] for name in ("x", "y", "z")]).astype(np.float32)
            elif element == "face":
                list_prop = next((prop for prop in props if prop[0] == "list" and prop[3] == "vertex_indices"), None)
                object_prop = next((prop for prop in props if prop[0] == "scalar" and prop[2] == "object_id"), None)
                if list_prop is None or object_prop is None:
                    raise ValueError(f"Replica semantic PLY face element lacks vertex_indices/object_id: {path}")
                polygons: list[np.ndarray] = []
                ids: list[int] = []
                if encoding == "ascii":
                    for _ in range(count):
                        values = stream.readline().split()
                        cursor = 0
                        polygon = None
                        object_id = None
                        for kind, first_type, name, maybe_name in props:
                            if kind == "scalar":
                                value = values[cursor]
                                cursor += 1
                                if name == "object_id":
                                    object_id = int(float(value))
                            else:
                                length = int(values[cursor])
                                cursor += 1
                                value = np.asarray(values[cursor:cursor + length], dtype=np.int64)
                                cursor += length
                                if maybe_name == "vertex_indices":
                                    polygon = value
                        if polygon is None or object_id is None:
                            raise ValueError(f"Malformed face data in {path}")
                        polygons.append(polygon)
                        ids.append(object_id)
                else:
                    face_start = stream.tell()
                    count_type = list_prop[1]
                    count_format = PLY_STRUCT_FORMATS[count_type]
                    count_size = struct.calcsize(endian + count_format)

                    def parse_one_face() -> tuple[list[int], int, list[int]]:
                        polygon = None
                        object_id = None
                        list_lengths = []
                        for kind, first_type, name, maybe_name in props:
                            if kind == "scalar":
                                scalar_format = PLY_STRUCT_FORMATS[first_type]
                                value = struct.unpack(endian + scalar_format, stream.read(struct.calcsize(endian + scalar_format)))[0]
                                if name == "object_id":
                                    object_id = int(value)
                            else:
                                count_bytes = stream.read(count_size)
                                if len(count_bytes) != count_size:
                                    raise ValueError(f"Unexpected end of PLY face data in {path}")
                                length = int(struct.unpack(endian + count_format, count_bytes)[0])
                                item_format = PLY_STRUCT_FORMATS[name]
                                item_size = struct.calcsize(endian + item_format)
                                payload = stream.read(length * item_size)
                                if len(payload) != length * item_size:
                                    raise ValueError(f"Unexpected end of PLY face list in {path}")
                                value = list(struct.unpack(endian + item_format * length, payload))
                                list_lengths.append(length)
                                if maybe_name == "vertex_indices":
                                    polygon = value
                        if polygon is None or object_id is None:
                            raise ValueError(f"Malformed binary face data in {path}")
                        return polygon, object_id, list_lengths

                    first_polygon, first_id, first_lengths = parse_one_face()
                    stream.seek(face_start)
                    # Replica meshes use a fixed polygon width, so a structured
                    # NumPy read keeps multi-million-face meshes practical.
                    dtype_fields = []
                    fixed = True
                    for kind, first_type, name, maybe_name in props:
                        if kind == "scalar":
                            dtype_fields.append((name, endian + PLY_TYPES[first_type]))
                        else:
                            length = first_lengths.pop(0)
                            dtype_fields.append((f"{maybe_name}_count", endian + PLY_TYPES[first_type]))
                            dtype_fields.append((maybe_name, endian + PLY_TYPES[name], (length,)))
                    dtype = np.dtype(dtype_fields)
                    binary_faces = np.fromfile(stream, dtype=dtype, count=count)
                    list_props = [prop for prop in props if prop[0] == "list"]
                    for prop in list_props:
                        if not np.all(binary_faces[f"{prop[3]}_count"] == binary_faces[prop[3]].shape[1]):
                            fixed = False
                            break
                    if fixed:
                        faces_raw = [binary_faces[prop[3]] for prop in list_props if prop[3] == "vertex_indices"][0]
                        object_ids = binary_faces["object_id"].astype(np.int64)
                    else:
                        stream.seek(face_start)
                        polygons = []
                        ids = []
                        for _ in range(count):
                            polygon, object_id, _ = parse_one_face()
                            polygons.append(np.asarray(polygon, dtype=np.int64))
                            ids.append(object_id)
                        polygons = polygons
                        object_ids = np.asarray(ids, dtype=np.int64)
                        faces_raw = polygons
            else:
                # Known Replica semantic files contain only vertices and faces.
                # Refuse unknown interleaved records rather than misalign reads.
                if count:
                    raise ValueError(f"Unsupported PLY element {element!r} in {path}")
    if vertices is None or faces_raw is None or object_ids is None:
        raise ValueError(f"Replica semantic PLY has no usable faces: {path}")

    triangles = []
    triangle_object_ids = []
    for index, polygon in enumerate(faces_raw):
        polygon = np.asarray(polygon, dtype=np.int64)
        for corner in range(1, len(polygon) - 1):
            triangles.append((polygon[0], polygon[corner], polygon[corner + 1]))
            triangle_object_ids.append(object_ids[index])
    return vertices, np.asarray(triangles, dtype=np.uint32), np.asarray(triangle_object_ids, dtype=np.int64)


def _make_observation_crop(rgb: np.ndarray, instance_map: np.ndarray, object_id: int) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]] | None:
    mask = instance_map == object_id
    ys, xs = np.where(mask)
    if len(xs) < MIN_MASK_PIXELS:
        return None
    height, width = mask.shape
    left = max(0, int(xs.min()) - PAD_PIXELS)
    top = max(0, int(ys.min()) - PAD_PIXELS)
    right = min(width, int(xs.max()) + 1 + PAD_PIXELS)
    bottom = min(height, int(ys.max()) + 1 + PAD_PIXELS)
    box_crop = Image.fromarray(rgb[top:bottom, left:right, :3].astype(np.uint8), "RGB")
    local_mask = Image.fromarray((mask[top:bottom, left:right] * 255).astype(np.uint8), "L")
    return box_crop, local_mask, (left, top, right, bottom)


def _render_crop_rows(args: argparse.Namespace, objects: dict, relations: dict,
                      scenes: list[str]) -> tuple[list[dict], dict]:
    image_root = args.image_root.expanduser().resolve()
    semantic_root = args.semantic_root.expanduser().resolve()
    cache_root = args.crop_cache.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    object_rows = _scene_records(objects, "objects")
    relation_rows = _scene_records(relations, "relationships")
    camera_data = _load_json(image_root / "cam_params.json")["camera"]

    observations = []
    scene_info = {}
    for scene in scenes:
        id_to_label = _object_metadata(object_rows[scene])
        local_scene = image_root / scene.replace("_", "")
        rgb_paths = sorted(
            (local_scene / "results").glob("frame*.jpg"),
            key=lambda path: int(re.search(r"frame(\d+)", path.stem).group(1)),
        )
        poses = np.loadtxt(local_scene / "traj.txt", dtype=np.float64).reshape(-1, 4, 4)
        if len(rgb_paths) != len(poses):
            raise ValueError(f"RGB/pose count mismatch in {local_scene}: {len(rgb_paths)} images, {len(poses)} poses")
        frame_count = len(rgb_paths)
        selected_indices = np.arange(frame_count)
        if args.max_frames_per_scene > 0 and frame_count > args.max_frames_per_scene:
            selected_indices = np.linspace(0, frame_count - 1, args.max_frames_per_scene, dtype=int)

        mesh_path = semantic_root / scene / "habitat" / "mesh_semantic.ply"
        vertices, triangles, triangle_object_ids = _read_ply_mesh(mesh_path)
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("Replica mask rendering requires the project's Open3D installation") from exc
        mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32),
        )
        ray_scene = o3d.t.geometry.RaycastingScene()
        ray_scene.add_triangles(mesh)

        scene_cache = cache_root / scene
        scene_cache.mkdir(parents=True, exist_ok=True)
        visible_by_frame = defaultdict(list)
        for frame_idx in selected_indices:
            image_path = rgb_paths[int(frame_idx)]
            with Image.open(image_path) as source:
                rgb = np.asarray(source.convert("RGB"))
            height, width = rgb.shape[:2]
            scale_x = width / float(camera_data["w"])
            scale_y = height / float(camera_data["h"])
            intrinsic = np.array([
                [camera_data["fx"] * scale_x, 0, camera_data["cx"] * scale_x],
                [0, camera_data["fy"] * scale_y, camera_data["cy"] * scale_y],
                [0, 0, 1],
            ], dtype=np.float64)
            c2w = poses[int(frame_idx)].copy()
            # Match ReplicaDataset.load_poses: convert Replica's camera basis to OpenCV.
            c2w[:3, 1:3] *= -1
            world_to_camera = np.linalg.inv(c2w)
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                o3d.core.Tensor(intrinsic, dtype=o3d.core.Dtype.Float64),
                o3d.core.Tensor(world_to_camera, dtype=o3d.core.Dtype.Float64),
                width_px=width,
                height_px=height,
            )
            primitive_ids = ray_scene.cast_rays(rays)["primitive_ids"].numpy().astype(np.int64)
            instance_map = np.full((height, width), -1, dtype=np.int64)
            valid = (primitive_ids >= 0) & (primitive_ids < len(triangle_object_ids))
            instance_map[valid] = triangle_object_ids[primitive_ids[valid]]
            for object_id, label in id_to_label.items():
                crop_data = _make_observation_crop(rgb, instance_map, object_id)
                if crop_data is None:
                    continue
                box_crop, local_mask, region_box = crop_data
                box_path = scene_cache / f"frame-{int(frame_idx):04d}-object-{object_id}-box.png"
                mask_path = scene_cache / f"frame-{int(frame_idx):04d}-object-{object_id}-mask.png"
                box_crop.save(box_path)
                local_mask.save(mask_path)
                observations.append({
                    "scene": scene,
                    "frame": int(frame_idx),
                    "object_id": object_id,
                    "label": label,
                    "box_path": str(box_path),
                    "mask_path": str(mask_path),
                    "masked_path": str(mask_path),
                    "rgb_path": str(image_path),
                    "region_box": region_box,
                })
                visible_by_frame[int(frame_idx)].append(object_id)
        scene_info[scene] = {
            "input_frames": frame_count,
            "frames_sampled": len(selected_indices),
            "annotated_objects": len(id_to_label),
            "visible_object_observations": sum(map(len, visible_by_frame.values())),
            "gt_relationship_edges": len(relation_rows[scene]),
            "frames_with_visible_objects": len(visible_by_frame),
        }
        print(f"[replica-clip] raycast {scene}: {scene_info[scene]['visible_object_observations']} oracle object views", flush=True)
    return observations, {
        "scene_info": scene_info,
        "relationship_rows": relation_rows,
        "object_labels": object_rows,
        "scenes": scenes,
    }


def _encode_condition(encoder: Encoder, rows: list[dict], key: str, batch_size: int,
                      device: torch.device) -> tuple[torch.Tensor, float]:
    encoded = []
    started = time.perf_counter()
    for begin in range(0, len(rows), batch_size):
        images = []
        for row in rows[begin:begin + batch_size]:
            if key == "masked_path":
                with Image.open(row["box_path"]) as source:
                    crop = source.convert("RGB")
                with Image.open(row["mask_path"]) as source:
                    mask = source.convert("L")
                isolated = Image.new("RGB", crop.size, encoder.masked_background_rgb)
                isolated.paste(crop, mask=mask)
                crop.close()
                mask.close()
                images.append(isolated)
            else:
                with Image.open(row[key]) as source:
                    images.append(source.convert("RGB"))
        with torch.inference_mode():
            encoded.append(encoder.encode_images(images).cpu())
        torch.cuda.synchronize(device)
        for image in images:
            image.close()
    elapsed = time.perf_counter() - started
    return torch.cat(encoded), elapsed


def _encode_native_regions(encoder: Encoder, rows: list[dict], batch_size: int,
                           device: torch.device) -> tuple[torch.Tensor, float]:
    """Encode full Replica frames and their oracle boxes with FG-CLIP2 RoIAlign."""
    by_image: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_image.setdefault(row["rgb_path"], []).append(index)
    groups = list(by_image.items())
    output = [None] * len(rows)
    started = time.perf_counter()
    for begin in range(0, len(groups), batch_size):
        group = groups[begin:begin + batch_size]
        images, region_boxes = [], []
        for path, indices in group:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
            region_boxes.append([rows[index]["region_box"] for index in indices])
        with torch.inference_mode():
            encoded = encoder.encode_regions(images, region_boxes).cpu()
        torch.cuda.synchronize(device)
        cursor = 0
        for _, indices in group:
            for index in indices:
                output[index] = encoded[cursor]
                cursor += 1
        for image in images:
            image.close()
    return torch.stack(output), time.perf_counter() - started


def _association_and_graph_metrics(features: torch.Tensor, rows: list[dict],
                                   relation_rows: dict, scenes: list[str]) -> dict:
    features = F.normalize(features.float(), dim=-1).cpu()
    by_scene_object: dict[tuple[str, int], list[int]] = defaultdict(list)
    by_scene_frame_object: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        oid = int(row["object_id"])
        by_scene_object[(row["scene"], oid)].append(index)
        by_scene_frame_object[(row["scene"], int(row["frame"]))][oid] = index

    identity_correct = []
    reciprocal_ranks = []
    recalls = {1: [], 5: [], 10: []}
    graph_edge_total = 0
    graph_edge_correct = 0
    graph_node_total = 0
    graph_node_correct = 0

    for scene in scenes:
        scene_object_ids = sorted(oid for sc, oid in by_scene_object if sc == scene)
        frame_ids = sorted(frame for sc, frame in by_scene_frame_object if sc == scene)
        for frame in frame_ids:
            observations_this_frame = by_scene_frame_object[(scene, frame)]
            prototypes = {}
            for oid in scene_object_ids:
                other = [i for i in by_scene_object[(scene, oid)] if int(rows[i]["frame"]) != frame]
                if not other:
                    continue
                prototypes[oid] = F.normalize(features[other].mean(dim=0), dim=0)
            candidate_ids = sorted(prototypes)
            if len(candidate_ids) < 2:
                continue
            candidate_matrix = torch.stack([prototypes[oid] for oid in candidate_ids])
            for oid, observation_idx in observations_this_frame.items():
                if oid not in prototypes:
                    continue
                scores = features[observation_idx] @ candidate_matrix.T
                order = torch.argsort(scores, descending=True).tolist()
                rank = order.index(candidate_ids.index(oid)) + 1
                identity_correct.append(float(rank == 1))
                reciprocal_ranks.append(1.0 / rank)
                graph_node_total += 1
                graph_node_correct += int(rank == 1)
                for cutoff in recalls:
                    recalls[cutoff].append(float(rank <= cutoff))

        # Oracle predicates are retained. Only the visual mapping of observed
        # object nodes is replaced by each embedding model's nearest map node.
        gt_edges = []
        for rel in relation_rows[scene]:
            try:
                gt_edges.append((int(rel[0]), str(rel[3]), int(rel[2])))
            except (IndexError, TypeError, ValueError):
                continue
        edge_set = set(gt_edges)
        for frame in frame_ids:
            visible = by_scene_frame_object[(scene, frame)]
            prototypes = {}
            for oid in scene_object_ids:
                other = [i for i in by_scene_object[(scene, oid)] if int(rows[i]["frame"]) != frame]
                if other:
                    prototypes[oid] = F.normalize(features[other].mean(dim=0), dim=0)
            candidate_ids = sorted(prototypes)
            if len(candidate_ids) < 2:
                continue
            matrix = torch.stack([prototypes[oid] for oid in candidate_ids])
            predicted_ids = {}
            for oid, observation_idx in visible.items():
                if oid in prototypes:
                    predicted_ids[oid] = candidate_ids[int(torch.argmax(features[observation_idx] @ matrix.T))]
            for subject, predicate, obj in edge_set:
                if subject not in predicted_ids or obj not in predicted_ids:
                    continue
                graph_edge_total += 1
                predicted = (predicted_ids[subject], predicate, predicted_ids[obj])
                graph_edge_correct += int(predicted == (subject, predicate, obj))

    def mean_or_nan(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "cross_view_instance_retrieval_top1": mean_or_nan(identity_correct),
        "cross_view_instance_retrieval_mrr": mean_or_nan(reciprocal_ranks),
        "cross_view_instance_retrieval_recall_at_1": mean_or_nan(recalls[1]),
        "cross_view_instance_retrieval_recall_at_5": mean_or_nan(recalls[5]),
        "cross_view_instance_retrieval_recall_at_10": mean_or_nan(recalls[10]),
        "scene_graph_node_identity_accuracy": graph_node_correct / graph_node_total if graph_node_total else float("nan"),
        "scene_graph_oracle_predicate_triplet_recall": graph_edge_correct / graph_edge_total if graph_edge_total else float("nan"),
        "scene_graph_node_observations_scored": graph_node_total,
        "scene_graph_edge_observations_scored": graph_edge_total,
        "scene_graph_metric_scope": (
            "ReplicaSSG predicate labels are held fixed; this measures how object identity errors from visual matching propagate to graph nodes and relation triplets. It does not evaluate predicate generation."
        ),
    }


def _run_model(name: str, args: argparse.Namespace, rows: list[dict],
               labels: list[str], graph_data: dict, device: torch.device) -> dict:
    print(f"[replica-clip] loading {name}: {MODEL_IDS[name]}", flush=True)
    encoder = Encoder(name, args.hf_home.expanduser().resolve(), device)
    model_resident_cuda_gib = torch.cuda.memory_allocated(device) / (1024**3)
    torch.cuda.reset_peak_memory_stats(device)
    text_started = time.perf_counter()
    with torch.inference_mode():
        text_features = encoder.encode_text(labels).cpu()
    torch.cuda.synchronize(device)
    text_encode_seconds = time.perf_counter() - text_started
    conditions = {}
    for condition, path_key in (("ground-truth-padded-box", "box_path"),
                                ("ground-truth-mask-isolated", "masked_path")):
        features, elapsed = _encode_condition(encoder, rows, path_key, args.batch_size, device)
        category_index = {label: index for index, label in enumerate(labels)}
        scored_rows = [{**row, "category_index": category_index[row["label"]]} for row in rows]
        metrics = _score(features, text_features, scored_rows, labels, device)
        metrics.update(_association_and_graph_metrics(
            features, rows, graph_data["relationship_rows"], graph_data["scenes"]
        ))
        metrics["image_encode_seconds"] = elapsed
        metrics["category_text_encode_seconds"] = text_encode_seconds
        metrics["scene_graph_identity_inference_seconds"] = elapsed + text_encode_seconds
        metrics["object_views_per_second"] = len(rows) / elapsed
        conditions[condition] = metrics
        del features
    if name.startswith("fg-clip2-"):
        region_text_started = time.perf_counter()
        region_text_features = encoder.encode_text(labels, fg_walk_type="box").cpu()
        torch.cuda.synchronize(device)
        region_text_encode_seconds = time.perf_counter() - region_text_started
        features, elapsed = _encode_native_regions(encoder, rows, args.batch_size, device)
        category_index = {label: index for index, label in enumerate(labels)}
        scored_rows = [{**row, "category_index": category_index[row["label"]]} for row in rows]
        metrics = _score(features, region_text_features, scored_rows, labels, device)
        metrics.update(_association_and_graph_metrics(
            features, rows, graph_data["relationship_rows"], graph_data["scenes"]
        ))
        metrics["image_encode_seconds"] = elapsed
        metrics["category_text_encode_seconds"] = region_text_encode_seconds
        metrics["scene_graph_identity_inference_seconds"] = elapsed + region_text_encode_seconds
        metrics["object_views_per_second"] = len(rows) / elapsed
        metrics["text_space"] = "FG-CLIP2 region-caption alignment: RoIAlign region features against walk_type=box text features"
        conditions["ground-truth-native-roi"] = metrics
        del features, region_text_features
    result = {
        "name": name,
        "model_id": encoder.model_id,
        "model_revision": encoder.revision,
        "processor_image_mean_rgb_0_1": list(encoder.image_mean),
        "mask_background_rgb_8bit": list(encoder.masked_background_rgb),
        "model_load_seconds": encoder.load_seconds,
        "model_resident_cuda_allocated_gib": model_resident_cuda_gib,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        "vram_measurement_scope": "per-process PyTorch CUDA allocated/reserved bytes; excludes VRAM owned by concurrent processes",
        "category_text_encode_seconds": text_encode_seconds,
        "feature_dimension": int(text_features.shape[1]),
        "input_conditions": conditions,
    }
    encoder.close()
    del text_features
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.max_frames_per_scene == 0 or args.max_frames_per_scene < -1:
        raise ValueError("--batch-size must be positive; --max-frames-per-scene must be positive or -1")
    args.output = args.output.expanduser().resolve()
    args.hf_home = args.hf_home.expanduser().resolve()
    os.environ["HF_HOME"] = str(args.hf_home)
    objects, relations, scenes = _require_dataset(args)
    if args.check_data_only:
        for name in args.models:
            checkpoint = _check_local_model_cache(name, args.hf_home)
            print(f"[replica-clip] local checkpoint ready: {name} ({checkpoint})")
        print(f"Replica inputs ready for {len(scenes)} scene(s): {', '.join(scenes)}")
        return 0
    observations, graph_data = _render_crop_rows(args, objects, relations, scenes)
    if not observations:
        raise RuntimeError("No ground-truth object masks were visible in the selected ReplicaSSG frames")

    labels = sorted({row["label"] for row in observations})
    from conceptgraph.utils.model_utils import require_cuda_device
    device = require_cuda_device("cuda:0")
    results = []
    for name in args.models:
        results.append(_run_model(name, args, observations, labels, graph_data, device))

    baseline = next((row for row in results if row["name"] == "clip"), None)
    if baseline:
        for row in results:
            for condition, metrics in row["input_conditions"].items():
                if condition not in baseline["input_conditions"]:
                    continue
                base = baseline["input_conditions"][condition]
                for metric in (
                    "object_query_retrieval_map_macro_category",
                    "zero_shot_top1_macro_category",
                    "cross_view_instance_retrieval_top1",
                    "scene_graph_node_identity_accuracy",
                    "scene_graph_oracle_predicate_triplet_recall",
                ):
                    metrics[f"{metric}_delta_vs_clip"] = metrics[metric] - base[metric]
                metrics["scene_graph_identity_inference_seconds_delta_vs_clip"] = (
                    metrics["scene_graph_identity_inference_seconds"]
                    - base["scene_graph_identity_inference_seconds"]
                )
                metrics["scene_graph_identity_inference_seconds_ratio_vs_clip"] = (
                    metrics["scene_graph_identity_inference_seconds"]
                    / base["scene_graph_identity_inference_seconds"]
                )
            for key in ("model_load_seconds", "model_resident_cuda_allocated_gib", "peak_cuda_allocated_gib", "peak_cuda_reserved_gib"):
                row[f"{key}_delta_vs_clip"] = row[key] - baseline[key]
                row[f"{key}_ratio_vs_clip"] = row[key] / baseline[key] if baseline[key] else float("nan")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "ReplicaSSG ground-truth object crop, cross-view association, and scene-graph node impact",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "rgb_sequence_root": str(args.image_root.expanduser().resolve()),
            "semantic_mesh_root": str(args.semantic_root.expanduser().resolve()),
            "replicassg_annotations": str(args.annotations.expanduser().resolve()),
            "proposal_source": "Replica semantic mesh face object IDs raycast at local Replica RGB sequence camera poses",
            "label_source": "ReplicaSSG objects.json",
            "relationship_source": "ReplicaSSG relationships.json",
            "frames_used": len({(row["scene"], row["frame"]) for row in observations}),
            "object_observations": len(observations),
            "object_categories": len(labels),
            "scenes": graph_data["scene_info"],
            "caption_note": "ReplicaSSG supplies object categories and scene relationships, not reference object captions. CLIP-family models rank label text; they do not generate captions.",
        },
        "measurement": {
            "primary": "macro AP for each observed Replica category text query retrieving matching GT object views from the full observation bank",
            "secondary": "zero-shot label accuracy; leave-one-view-out cross-view instance retrieval; scene-graph node identity and oracle-predicate triplet recall",
            "graph_interpretation": "The relation labels are ground truth and held fixed. The graph score isolates propagation of CLIP-based object association errors into scene-graph nodes and edge triplets; it is not a predicate-generation score.",
            "inputs": {
                "ground-truth-padded-box": "GT semantic instance mask-derived box with 20-pixel context",
                "ground-truth-mask-isolated": "same box and pixels outside its GT instance mask filled with this model's processor mean RGB, which normalizes to approximately zero",
                "ground-truth-native-roi": "FG-CLIP2-only supplemental input: the full source RGB frame and oracle GT box are passed through FG-CLIP2's native RoIAlign feature API, paired with walk_type=box text embeddings; this is separate from the direct shared-crop pipeline comparison",
            },
        },
        "settings": {"models": args.models, "batch_size": args.batch_size,
                     "device": str(device), "text_prompt_template": "a photo of a {category}"},
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[replica-clip] wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[replica-clip] ERROR: {exc}", file=sys.stderr)
        raise
