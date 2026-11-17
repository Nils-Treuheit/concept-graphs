#!/usr/bin/env python3
"""CPU-only evaluator: saved room0 map track-to-Replica semantic identity matching.

Offline evaluation of the existing 200-frame stride-10 room0 map. Measures how
consistently the saved PCD associates its observations to ReplicaSSG semantic
instance IDs using Open3D CPU raycasting.

Inputs (all frozen):
  - PCD  : .agentic_workflow/workspace/room0-relation-propagation-test/data/Replica/room0/exps/r_room0_stride10_siglip2so400m_qwen38_pcd/pcd_r_room0_stride10_siglip2so400m_qwen38_pcd.pkl.gz
  - PLY  : /mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/replica_v1/room_0/habitat/mesh_semantic.ply
  - Poses: /mnt/HDD1/Project_Data/public_datasets/Replica/room0/traj.txt
  - Intrinsics: /mnt/HDD1/Project_Data/public_datasets/Replica/cam_params.json
  - Annotations: /mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/ReplicaSSG/files/objects.json

Outputs:
  - Result JSON : models/experiments/t_ren/data/map_assoc_room0/room0_identity_result.json
  - Overlays    : outputs/overlays/room0_identity_{frame}.png (frames 0, 40, 530, 1060, 1550)

No model, GPU, network, or /tmp usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import psutil
import struct
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import scipy
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Paths (relative to this packet or absolute)
# ---------------------------------------------------------------------------

# PACKET resolves to the task directory:
#   .../.agentic_workflow/workspace/pipeline-experiments/map-association-eval-YYYYMMDD
PACKET = Path(__file__).resolve().parents[4]

# Map PCD is under the workspace sibling:
#   .../.agentic_workflow/workspace/room0-relation-propagation-test/data/...
PCD_PATH = (
    PACKET.parents[1]
    / "room0-relation-propagation-test"
    / "data"
    / "Replica"
    / "room0"
    / "exps"
    / "r_room0_stride10_siglip2so400m_qwen38_pcd"
    / "pcd_r_room0_stride10_siglip2so400m_qwen38_pcd.pkl.gz"
)

PLY_PATH = Path("/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/replica_v1/room_0/habitat/mesh_semantic.ply")
TRAJ_PATH = Path("/mnt/HDD1/Project_Data/public_datasets/Replica/room0/traj.txt")
INTRINSICS_PATH = Path("/mnt/HDD1/Project_Data/public_datasets/Replica/cam_params.json")
ANNOTATIONS_PATH = Path(
    "/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/ReplicaSSG/files/objects.json"
)
RGB_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/Replica/room0/results")

# Durable result directory: repository root / models/experiments/t_ren/data/map_assoc_room0/
RESULT_DIR = PACKET.parents[3] / "models" / "experiments" / "t_ren" / "data" / "map_assoc_room0"
OVERLAY_DIR = PACKET / "outputs" / "overlays"

OVERLAY_FRAMES = [0, 40, 530, 1060, 1550]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# PLY mesh loading (from replica_clip_benchmark.py)
# ---------------------------------------------------------------------------

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
                if encoding == "ascii":
                    triangles = []
                    triangle_object_ids = []
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
                        for corner in range(1, len(polygon) - 1):
                            triangles.append((polygon[0], polygon[corner], polygon[corner + 1]))
                            triangle_object_ids.append(object_id)
                    triangles = np.asarray(triangles, dtype=np.uint32)
                    triangle_object_ids = np.asarray(triangle_object_ids, dtype=np.int64)
                    faces_raw = triangles
                    object_ids = triangle_object_ids
                else:
                    count_type = list_prop[1]
                    count_format = PLY_STRUCT_FORMATS[count_type]
                    count_size = struct.calcsize(endian + count_format)

                    def parse_one_face(stream_obj, path_obj):
                        polygon = None
                        object_id = None
                        list_lengths = []
                        for kind, first_type, name, maybe_name in props:
                            if kind == "scalar":
                                scalar_format = PLY_STRUCT_FORMATS[first_type]
                                value = struct.unpack(endian + scalar_format, stream_obj.read(struct.calcsize(endian + scalar_format)))[0]
                                if name == "object_id":
                                    object_id = int(value)
                            else:
                                count_bytes = stream_obj.read(count_size)
                                if len(count_bytes) != count_size:
                                    raise ValueError(f"Unexpected end of PLY face data in {path_obj}")
                                length = int(struct.unpack(endian + count_format, count_bytes)[0])
                                item_format = PLY_STRUCT_FORMATS[name]
                                item_size = struct.calcsize(endian + item_format)
                                payload = stream_obj.read(length * item_size)
                                if len(payload) != length * item_size:
                                    raise ValueError(f"Unexpected end of PLY face list in {path_obj}")
                                value = list(struct.unpack(endian + item_format * length, payload))
                                list_lengths.append(length)
                                if maybe_name == "vertex_indices":
                                    polygon = value
                        if polygon is None or object_id is None:
                            raise ValueError(f"Malformed binary face data in {path_obj}")
                        return polygon, object_id, list_lengths

                    face_start = stream.tell()
                    first_polygon, first_id, first_lengths = parse_one_face(stream, path)
                    stream.seek(face_start)
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
                    list_props_filtered = [prop for prop in props if prop[0] == "list"]
                    for prop in list_props_filtered:
                        if not np.all(binary_faces[f"{prop[3]}_count"] == binary_faces[prop[3]].shape[1]):
                            fixed = False
                            break
                    if fixed:
                        faces_raw_list = [binary_faces[prop[3]] for prop in list_props_filtered if prop[3] == "vertex_indices"]
                        faces_raw = faces_raw_list[0] if faces_raw_list else None
                        object_ids = binary_faces["object_id"].astype(np.int64)
                    else:
                        stream.seek(face_start)
                        triangles = []
                        triangle_object_ids = []
                        for _ in range(count):
                            polygon, object_id, _ = parse_one_face(stream, path)
                            polygon = np.asarray(polygon, dtype=np.int64)
                            for corner in range(1, len(polygon) - 1):
                                triangles.append([polygon[0], polygon[corner], polygon[corner + 1]])
                                triangle_object_ids.append(object_id)
                        faces_raw = np.asarray(triangles, dtype=np.uint32)
                        object_ids = np.asarray(triangle_object_ids, dtype=np.int64)
            else:
                if count:
                    raise ValueError(f"Unsupported PLY element {element!r} in {path}")
        if vertices is None or faces_raw is None or object_ids is None:
            raise ValueError(f"Replica semantic PLY has no usable faces: {path}")

    # Convert to triangles
    triangles = []
    triangle_object_ids = []
    for index in range(len(faces_raw)):
        polygon = faces_raw[index]
        polygon = np.asarray(polygon, dtype=np.int64)
        for corner in range(1, len(polygon) - 1):
            triangles.append((int(polygon[0]), int(polygon[corner]), int(polygon[corner + 1])))
            triangle_object_ids.append(int(object_ids[index]))
    return (
        vertices,
        np.asarray(triangles, dtype=np.uint32),
        np.asarray(triangle_object_ids, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Global mesh and raycasting scene (built once, reused per frame)
# ---------------------------------------------------------------------------

_global_mesh: Any = None
_global_ray_scene: Any = None


def _build_global_mesh(vertices: np.ndarray, triangles: np.ndarray) -> None:
    """Build Open3D mesh and raycasting scene once for all frames. CPU-only."""
    global _global_mesh, _global_ray_scene
    _global_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32),
    )
    _global_ray_scene = o3d.t.geometry.RaycastingScene()
    _global_ray_scene.add_triangles(_global_mesh)


def _render_semantic_mask_frame(
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    triangle_object_ids: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Render semantic instance mask for a single frame using the global raycasting scene.

    triangle_object_ids: (N,) int64 array mapping triangle IDs to semantic IDs.
    width, height: image dimensions.
    Returns: (height, width) int64 array of instance IDs, -1 for misses.
    """
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3d.core.Tensor(intrinsic, dtype=o3d.core.Dtype.Float64),
        o3d.core.Tensor(world_to_camera, dtype=o3d.core.Dtype.Float64),
        width_px=width,
        height_px=height,
    )
    results = _global_ray_scene.cast_rays(rays)
    primitive_ids = results["primitive_ids"].numpy().astype(np.int64)

    instance_map = np.full((height, width), -1, dtype=np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(triangle_object_ids))
    instance_map[valid] = triangle_object_ids[primitive_ids[valid]]

    del rays, results, primitive_ids
    return instance_map


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------


def make_overlay(
    rgb: np.ndarray,
    gt_instance_map: np.ndarray,
    track_label_map: np.ndarray,
    label_to_uuid: dict[int, str],
    frame: int,
    frame_matched: list[tuple[str, int]],
    frame_unmatched: list[str],
) -> Image.Image:
    """Build an RGB overlay with GT instance boundaries and track labels.

    Track IDs are UUID strings.  track_label_map uses deterministic integer
    labels (no hashing) mapped via label_to_uuid for display.

    frame_matched: list of (track_id_str, gt_id_int) matched for this frame.
    frame_unmatched: list of track_id_str not matched for this frame.
    """
    h, w = rgb.shape[:2]
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Assign distinct colors to GT instances (deterministic, seeded by ID)
    gt_colors: dict[int, tuple[int, int, int]] = {}
    color_seed = 1
    for y in range(h):
        for x in range(w):
            oid = int(gt_instance_map[y, x])
            if oid >= 0 and oid not in gt_colors:
                gt_colors[oid] = (
                    (color_seed * 137) % 256,
                    (color_seed * 211) % 256,
                    (color_seed * 173) % 256,
                )
                color_seed += 1

    # Draw GT boundaries (1px where adjacent pixels differ in ID)
    boundary_mask = np.zeros((h, w), dtype=bool)
    for y in range(h - 1):
        for x in range(w - 1):
            c0 = gt_instance_map[y, x]
            c1 = gt_instance_map[y, x + 1]
            c2 = gt_instance_map[y + 1, x]
            if c0 != c1 or c0 != c2:
                boundary_mask[y, x] = True
    boundary_mask[h - 1, :] = True
    boundary_mask[:, w - 1] = True

    for y in range(h):
        for x in range(w):
            if boundary_mask[y, x]:
                c0 = int(gt_instance_map[y, x])
                if c0 >= 0:
                    color = gt_colors.get(c0, (255, 255, 0))
                    draw.point([(x, y)], fill=(0, 0, 0, 180))
                    draw.point([(x, y)], fill=color + (200,))

    # Color map for track labels – deterministic from label index
    track_color_map: dict[int, tuple[int, int, int]] = {}
    unique_labels = sorted(
        set(l for l in track_label_map.flat if l >= 0)
    )
    for idx, label in enumerate(unique_labels):
        track_color_map[label] = (
            (idx * 113 + 50) % 256,
            (idx * 179 + 80) % 256,
            (idx * 211 + 20) % 256,
        )

    for y in range(h):
        for x in range(w):
            lbl = int(track_label_map[y, x])
            if lbl >= 0 and lbl in track_color_map:
                c = track_color_map[lbl]
                draw.point([(x, y)], fill=c + (120,))

    # Text labels at centroid of each track, showing short UUID prefix
    all_uuids = sorted(set(t for t, _ in frame_matched) | set(frame_unmatched))
    for tid in all_uuids:
        # Find the label for this UUID
        lbl = None
        for label, uuid in label_to_uuid.items():
            if uuid == tid:
                lbl = label
                break
        if lbl is None:
            continue
        mask_y, mask_x = np.where(track_label_map == lbl)
        if len(mask_y) == 0:
            continue
        cy, cx = int(mask_y.mean()), int(mask_x.mean())
        matched_gt = None
        for mt, gt in frame_matched:
            if mt == tid:
                matched_gt = gt
                break
        short_uuid = tid[:8]
        text = f"T{short_uuid}"
        if matched_gt is not None:
            text = f"T{short_uuid}->G{matched_gt}"
        elif tid in frame_unmatched:
            text = f"T{short_uuid}X"
        draw.text((min(cx, w - 80), min(cy, h - 15)), text, fill=(0, 0, 0), font=None)

    # Composite
    rgb_img = Image.fromarray(rgb)
    result = Image.alpha_composite(rgb_img.convert("RGBA"), overlay)
    return result


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def load_pcd(path: Path) -> dict:
    import pickle
    import gzip

    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_pcd_objects(pcd: dict) -> list[dict]:
    """Extract the objects list, handling both list and dict formats."""
    if isinstance(pcd["objects"], list):
        return pcd["objects"]
    if isinstance(pcd["objects"], dict):
        return list(pcd["objects"].values())
    raise ValueError(f"Unexpected PCD objects type: {type(pcd['objects'])}")


def validate_pcd(objects: list[dict], pcd: dict) -> dict:
    """Validate PCD structure and report findings. Returns a findings dict.

    Track IDs are UUID strings and must be preserved as such.
    Reports duplicate (track_id, source_frame) groups as diagnostics only;
    no longer fails closed on them.  Adds field-alignment validation so that
    every per-observation list needed for union aggregation has matching length.
    """
    findings: dict[str, Any] = {
        "total_tracks": 0,
        "total_observations": 0,
        "recoverable_raw_rows": 0,
        "malformed_frame_values": 0,
        "duplicate_joins": 0,
        "duplicate_join_groups": 0,
        "duplicate_join_groups_detail": [],
        "in_scene_paths": True,
        "mask_shape_ok": True,
        "mask_type_ok": True,
        "path_type_ok": True,
        "frame_alignment_ok": True,
        "field_alignment_ok": True,
        "errors": [],
    }

    track_ids: set[str] = set()
    obs_keys: list[tuple[str, int]] = []  # (track_id_str, source_frame)
    in_scene_ok = True
    mask_shape_ok = True
    mask_type_ok = True
    path_type_ok = True
    frame_alignment_ok = True
    field_alignment_ok = True
    total_recoverable_raw_rows = 0
    total_malformed_frame_values = 0

    def _is_sequence(obj):
        """Return True if *obj* is a sequence-like container (list/tuple/np.ndarray).

        Strings, mappings, scalars, and None return False.
        """
        if isinstance(obj, (str, bytes, bytearray)):
            return False
        if isinstance(obj, dict):
            return False
        if isinstance(obj, (list, tuple)):
            return True
        try:
            if hasattr(obj, "__len__") and hasattr(obj, "__iter__"):
                if hasattr(obj, "__getitem__"):
                    return True
        except (TypeError, AttributeError):
            pass
        return False

    def _try_len(value):
        """Return the length of a sequence-like object, or None on failure.

        Guards zero-dimensional arrays and any container that exposes
        __len__ but raises TypeError / ValueError / AttributeError when len()
        is called.
        """
        try:
            return len(value)
        except (TypeError, ValueError, AttributeError):
            return None

    def _safe_mask_array(raw_mask):
        """Convert a raw mask value to a numpy array, preserving its original dtype.

        Returns the array on success, or None on failure.
        The caller must validate dtype separately (e.g. mask.dtype != bool).
        """
        if raw_mask is None:
            return None
        try:
            arr = np.asarray(raw_mask)
            # Ensure it has a shape attribute with at least 2 dims
            if not hasattr(arr, "shape") or arr.ndim < 2:
                return None
            return arr
        except (ValueError, TypeError, AttributeError):
            return None

    for obj in objects:
        # Validate that obj is a usable mapping before accessing keys.
        if not isinstance(obj, dict):
            field_alignment_ok = False
            findings["errors"].append(
                f"Object is not a mapping (type {type(obj).__name__})"
            )
            continue

        if "id" not in obj:
            field_alignment_ok = False
            findings["errors"].append(
                f"Object is missing required 'id' field"
            )
            continue

        # Track IDs are UUID strings – preserve them as strings.
        track_id = str(obj["id"])
        track_ids.add(track_id)

        # Validate image_idx exists and is a usable sequence before len/index.
        img_idx_raw = obj.get("image_idx")
        if img_idx_raw is None or not _is_sequence(img_idx_raw):
            field_alignment_ok = False
            findings["errors"].append(
                f"image_idx missing or not a usable sequence for track {track_id}"
            )
            continue  # skip object-level observation validation, continue next object

        n = _try_len(img_idx_raw)
        if n is None:
            field_alignment_ok = False
            findings["errors"].append(
                f"image_idx is a sequence-like object but len() failed for track {track_id}"
            )
            continue

        # Count recoverable raw rows from every valid image_idx sequence.
        total_recoverable_raw_rows += n

        # Validate that every per-observation field has aligned length n.
        # Track per-field safety so the observation loop can skip unsafe access.
        per_field_safe: dict[str, bool] = {}
        for field_name in ("mask_idx", "color_path", "mask", "xyxy"):
            field_val = obj.get(field_name)
            if field_val is None:
                field_alignment_ok = False
                per_field_safe[field_name] = False
                findings["errors"].append(
                    f"Missing field {field_name!r} for track {track_id}"
                )
                continue
            if not _is_sequence(field_val):
                field_alignment_ok = False
                per_field_safe[field_name] = False
                findings["errors"].append(
                    f"Field {field_name!r} is not a sequence for track {track_id}"
                )
                continue
            field_len = _try_len(field_val)
            if field_len is None:
                field_alignment_ok = False
                per_field_safe[field_name] = False
                findings["errors"].append(
                    f"Field {field_name!r} length check failed for track {track_id}"
                )
                continue
            if field_len != n:
                field_alignment_ok = False
                per_field_safe[field_name] = False
                findings["errors"].append(
                    f"Field {field_name!r} length ({field_len}) != "
                    f"image_idx length ({n}) for track {track_id}"
                )
            else:
                per_field_safe[field_name] = True

        # Do not break at the first field mismatch — continue validating the
        # remaining objects so raw observation counts are preserved.

        for i in range(n):
            # Convert image_idx safely; record error and skip only this index.
            raw_val = img_idx_raw[i]
            try:
                img_idx = int(raw_val)
            except (TypeError, ValueError, OverflowError):
                field_alignment_ok = False
                total_malformed_frame_values += 1
                findings["errors"].append(
                    f"Cannot convert image_idx[{i}] = {raw_val!r} to int "
                    f"for track {track_id}"
                )
                continue

            source_frame = img_idx * 10
            obs_keys.append((track_id, source_frame))

            # Check color_path type and in-scene — only if color_path was validated safe.
            if per_field_safe.get("color_path", False):
                try:
                    color_path = obj["color_path"][i]
                except (IndexError, KeyError, TypeError):
                    color_path = None
                if color_path is None:
                    path_type_ok = False
                    findings["errors"].append(
                        f"color_path value is None for track {track_id} frame {source_frame}"
                    )
                elif not isinstance(color_path, (str, os.PathLike)):
                    path_type_ok = False
                    findings["errors"].append(
                        f"color_path value is not str or PathLike "
                        f"(type {type(color_path).__name__}) for track {track_id} frame {source_frame}"
                    )
                else:
                    cp_for_scene = Path(color_path) if isinstance(color_path, str) else Path(color_path)
                    if "room0" not in str(cp_for_scene).lower():
                        in_scene_ok = False
                        findings["errors"].append(f"Out-of-scene path for track {track_id}: {cp_for_scene}")

            # Check mask shape and dtype — only if mask was validated safe.
            if per_field_safe.get("mask", False):
                try:
                    raw_mask = obj["mask"][i]
                except (IndexError, KeyError, TypeError):
                    raw_mask = None
                if raw_mask is not None:
                    mask = _safe_mask_array(raw_mask)
                    if mask is None:
                        mask_shape_ok = False
                        findings["errors"].append(
                            f"Mask is not a valid array for track {track_id} frame {source_frame}"
                        )
                    elif mask.shape != (680, 1200):
                        mask_shape_ok = False
                        findings["errors"].append(
                            f"Mask shape mismatch for track {track_id} frame {source_frame}: {mask.shape}"
                        )
                    elif mask.dtype != bool:
                        mask_type_ok = False
                        findings["errors"].append(
                            f"Mask dtype mismatch for track {track_id} frame {source_frame}: {mask.dtype}"
                        )
                else:
                    mask_shape_ok = False
                    findings["errors"].append(
                        f"Mask value is None for track {track_id} frame {source_frame}"
                    )

            # Frame alignment check
            if source_frame != img_idx * 10:
                frame_alignment_ok = False

    findings["total_tracks"] = len(track_ids)
    findings["total_observations"] = len(obs_keys)
    findings["recoverable_raw_rows"] = total_recoverable_raw_rows
    findings["malformed_frame_values"] = total_malformed_frame_values

    # Check for duplicate (track_id, source_frame) joins
    key_counts = Counter(obs_keys)
    dup_groups = {k: v for k, v in key_counts.items() if v > 1}
    findings["duplicate_join_groups"] = len(dup_groups)
    findings["duplicate_joins"] = sum(v - 1 for v in dup_groups.values())
    findings["duplicate_join_groups_detail"] = [
        {"track_id": k[0], "source_frame": k[1], "count": v}
        for k, v in sorted(dup_groups.items())
    ]

    findings["in_scene_paths"] = in_scene_ok
    findings["mask_shape_ok"] = mask_shape_ok
    findings["mask_type_ok"] = mask_type_ok
    findings["path_type_ok"] = path_type_ok
    findings["frame_alignment_ok"] = frame_alignment_ok
    findings["field_alignment_ok"] = field_alignment_ok

    return findings


def flatten_observations(objects: list[dict]) -> list[dict]:
    """Flatten objects with multiple detections into individual observation records.

    Track IDs are preserved as UUID strings.  Skips observations whose
    color_path is not str or PathLike (already caught by validate_pcd).
    """
    observations = []
    for obj in objects:
        # Track ID is a UUID string.
        track_id = str(obj["id"])
        n = len(obj["image_idx"])
        for i in range(n):
            raw_val = obj["image_idx"][i]
            try:
                img_idx = int(raw_val)
            except (TypeError, ValueError, OverflowError):
                continue
            source_frame = img_idx * 10
            color_path_raw = obj["color_path"][i]
            if not isinstance(color_path_raw, (str, os.PathLike)):
                # Non-path value already flagged by validate_pcd; skip.
                continue
            color_path = Path(color_path_raw)
            mask = obj["mask"][i]
            xyxy = obj["xyxy"][i] if isinstance(obj["xyxy"], list) else obj["xyxy"]
            observations.append({
                "track_id": track_id,
                "image_idx": img_idx,
                "source_frame": source_frame,
                "color_path": color_path,
                "mask": mask,
                "xyxy": xyxy,
            })
    return observations


def aggregate_observations(
    observations: list[dict],
) -> tuple[list[dict], dict]:
    """Group every observation by (track_id, source_frame) and union all masks.

    Returns:
        aggregated: list of prediction dicts, one per unique (track_id, source_frame).
                    Each carries:
                      - mask       : boolean (680, 1200) union of all component masks
                      - components : int  (number of raw rows merged)
                      - comp_idx   : list of int (stable indices into the input list)
        diagnostics: {
              "raw_observation_count": int,
              "unique_track_frame_count": int,
              "duplicate_track_frame_groups": int,
              "extra_observation_components": int,
          }
    """
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for idx, obs in enumerate(observations):
        key = (obs["track_id"], obs["source_frame"])
        groups[key].append({"obs": obs, "idx": idx})

    aggregated: list[dict] = []
    for key in sorted(groups.keys()):
        members = groups[key]
        comp_count = len(members)
        comp_indices = [m["idx"] for m in members]
        # Logical-OR union of all masks, preserving dtype=bool.
        union_mask: np.ndarray | None = None
        for m in members:
            mask = m["obs"]["mask"].astype(bool)
            if union_mask is None:
                union_mask = mask.copy()
            else:
                np.logical_or(union_mask, mask, out=union_mask)
        # Carry forward a single copy of the observation dict (first member).
        base = members[0]["obs"].copy()
        base["mask"] = union_mask  # type: ignore[assignment]
        base["components"] = comp_count
        base["comp_idx"] = comp_indices
        aggregated.append(base)

    raw_count = len(observations)
    unique_count = len(aggregated)
    dup_groups = sum(1 for g in groups.values() if len(g) > 1)
    extra_components = sum(len(g) - 1 for g in groups.values() if len(g) > 1)

    diagnostics = {
        "raw_observation_count": raw_count,
        "unique_track_frame_count": unique_count,
        "duplicate_track_frame_groups": dup_groups,
        "extra_observation_components": extra_components,
    }
    return aggregated, diagnostics


def load_pose_file(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    return poses.reshape(-1, 4, 4)


def load_intrinsics(path: Path) -> dict:
    data = _load_json(path)
    return data["camera"]


def load_annotations(path: Path, scene_key: str) -> dict[int, str]:
    """Load annotations and return {annotation_id: label} for the given scene."""
    data = _load_json(path)
    scene_objects = None
    for scan in data["scans"]:
        if scan["scan"] == scene_key:
            scene_objects = scan["objects"]
            break
    if scene_objects is None:
        raise FileNotFoundError(f"No annotations for scene {scene_key}")

    annotation_ids: set[int] = set()
    id_to_label: dict[int, str] = {}
    for row in scene_objects:
        try:
            oid = int(row["id"])
        except (TypeError, ValueError, KeyError):
            continue
        if oid in annotation_ids:
            raise ValueError(f"Duplicate annotation ID {oid}")
        annotation_ids.add(oid)
        label = str(row.get("label", "unknown")).strip().lower()
        if label and label not in {"unknown", "undefined"}:
            id_to_label[oid] = label
    return id_to_label


def compute_iou_matrix(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> np.ndarray:
    """Compute IoU matrix between prediction and ground-truth masks.

    pred_masks: list of prediction masks (boolean, (H, W))
    gt_masks: list of GT masks (boolean, (H, W))
    Returns: (len(pred), len(gt)) IoU matrix
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)

    for i in range(n_pred):
        for j in range(n_gt):
            intersection = np.logical_and(pred_masks[i], gt_masks[j]).sum()
            union = np.logical_or(pred_masks[i], gt_masks[j]).sum()
            if union > 0:
                iou_matrix[i, j] = intersection / union

    return iou_matrix


def match_masks(
    iou_matrix: np.ndarray, threshold: float = 0.5
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Perform one-to-one matching using linear_sum_assignment.

    Returns: (matched_pairs, unmatched_pred_indices, unmatched_gt_indices).
    Exactly three values. No scipy_version return.
    """
    # Negate IoU for minimization
    cost_matrix = 1.0 - iou_matrix
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # Filter by IoU threshold
    matched_pairs = []
    for pi, gi in zip(pred_indices, gt_indices):
        if iou_matrix[pi, gi] >= threshold:
            matched_pairs.append((int(pi), int(gi)))

    all_pred = set(range(iou_matrix.shape[0]))
    all_gt = set(range(iou_matrix.shape[1]))
    matched_pred = set(p for p, _ in matched_pairs)
    matched_gt = set(g for _, g in matched_pairs)

    unmatched_pred = sorted(all_pred - matched_pred)
    unmatched_gt = sorted(all_gt - matched_gt)

    return matched_pairs, unmatched_pred, unmatched_gt


def compute_precision_recall_f1(matched: int, pred_total: int, gt_total: int) -> dict[str, float]:
    """Compute precision, recall, F1 from match counts."""
    precision = matched / pred_total if pred_total > 0 else float("nan")
    recall = matched / gt_total if gt_total > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else (
        0.0 if (pred_total > 0 and gt_total > 0) else float("nan")
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def predict_identity(
    track_gt_id_map: dict[str, dict[int, int]],
    frame_id: int,
    track_id: str,
) -> tuple[int | None, bool]:
    """Predict GT ID for an observation using majority vote from other frames.

    track_gt_id_map: {track_id_str: {frame: gt_id}}

    Excludes all observations from the query frame.
    Returns: (predicted_gt_id, is_tie)
    is_tie indicates a tie in majority vote.
    """
    if track_id not in track_gt_id_map:
        return None, False

    frame_gts = track_gt_id_map[track_id]
    # Exclude every observation from the query frame
    other_gts = [gt for f, gt in frame_gts.items() if f != frame_id]

    if not other_gts:
        return None, False

    counter = Counter(other_gts)
    max_count = counter.most_common(1)[0][1]
    candidates = [gt for gt, cnt in counter.items() if cnt == max_count]

    # Tie-break: smallest GT ID
    predicted = min(candidates)
    is_tie = len(candidates) > 1

    return predicted, is_tie


def _build_intrinsics_matrix(intrinsics_data: dict, width: int, height: int) -> np.ndarray:
    """Build scaled 3x3 intrinsic matrix from base intrinsics and image dimensions."""
    base_w = intrinsics_data["w"]
    base_h = intrinsics_data["h"]
    scale_x = width / float(base_w)
    scale_y = height / float(base_h)
    return np.array([
        [intrinsics_data["fx"] * scale_x, 0, intrinsics_data["cx"] * scale_x],
        [0, intrinsics_data["fy"] * scale_y, intrinsics_data["cy"] * scale_y],
        [0, 0, 1],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def main() -> int:
    started_at = time.perf_counter()
    process = psutil.Process()

    # Track failures for fail-closed
    failures: list[str] = []

    try:
        # ================================================================
        # 1. Load and validate PCD
        # ================================================================
        if not PCD_PATH.is_file():
            raise FileNotFoundError(f"PCD path not found: {PCD_PATH}")

        print("[room0-identity] Loading PCD...", flush=True)
        pcd = load_pcd(PCD_PATH)
        objects = load_pcd_objects(pcd)

        findings = validate_pcd(objects, pcd)
        print(
            f"  Tracks: {findings['total_tracks']}, "
            f"Recoverable rows: {findings['recoverable_raw_rows']}, "
            f"Malformed frame values: {findings['malformed_frame_values']}, "
            f"Observations (converted): {findings['total_observations']}",
            flush=True,
        )
        print(f"  Duplicate join groups: {findings['duplicate_join_groups']}", flush=True)

        if not findings["in_scene_paths"]:
            failures.append("Out-of-scene color paths detected")
        if not findings["mask_shape_ok"]:
            failures.append("Mask shape validation failed")
        if not findings["mask_type_ok"]:
            failures.append("Mask type validation failed")
        if not findings["path_type_ok"]:
            failures.append("color_path type validation failed")
        if not findings["frame_alignment_ok"]:
            failures.append("Frame alignment validation failed")
        if not findings["field_alignment_ok"]:
            failures.append("Per-observation field alignment validation failed")

        # Structural validation (PCD structural, frame, path, or mask-shape/type)
        # must fail-closed before flatten_observations.
        if (
            not findings["field_alignment_ok"]
            or not findings["in_scene_paths"]
            or not findings["mask_shape_ok"]
            or not findings["mask_type_ok"]
            or not findings["path_type_ok"]
            or not findings["frame_alignment_ok"]
        ):
            failure_record = {
                "evaluation": {
                    "name": "room0_saved_map_track_identity",
                    "metrics_produced": False,
                    "scope_statement": "PCD field-alignment validation failed; no metrics produced.",
                },
                "config": {
                    "pcd_path": str(PCD_PATH),
                    "pcd_sha256": sha256_file(PCD_PATH),
                    "num_pcd_tracks": findings["total_tracks"],
                    "num_pcd_observations": findings["total_observations"],
                    "recoverable_raw_rows": findings["recoverable_raw_rows"],
                    "malformed_frame_values": findings["malformed_frame_values"],
                    "duplicate_join_groups": findings["duplicate_join_groups"],
                    "duplicate_joins": findings["duplicate_joins"],
                    "duplicate_join_groups_detail": findings["duplicate_join_groups_detail"],
                },
                "errors": findings["errors"],
                "failures": failures,
                "runtime": {
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "peak_rss_bytes": process.memory_info().rss,
                },
            }
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            result_path = RESULT_DIR / "room0_identity_result.json"
            tmp_path = RESULT_DIR / "room0_identity_result.json.tmp"
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(failure_record, f, indent=2, ensure_ascii=False, default=str)
                f.write("\n")
            tmp_path.rename(result_path)
            result_sha256 = sha256_file(result_path)
            print(
                f"Result written (fail-closed, field alignment): "
                f"{result_path} (SHA-256: {result_sha256})",
                flush=True,
            )
            return 1

        # Fail-closed on duplicate observation joins (diagnostic only; plan records duplicates
        # as integrity counts, not as scoring blockers).
        if findings["duplicate_join_groups"] > 0:
            print(
                f"  Duplicate join groups (diagnostic): "
                f"{findings['duplicate_join_groups']} groups, "
                f"{findings['duplicate_joins']} extra keys",
                flush=True,
            )

        # Flatten observations
        all_obs = flatten_observations(objects)
        for obs in all_obs:
            cp = obs["color_path"]
            if not isinstance(cp, (str, os.PathLike)):
                failures.append(f"Non-path color_path in flattened observations: {cp!r}")
                continue
            if isinstance(cp, str):
                cp = Path(cp)
            if not cp.is_file():
                failures.append(f"Missing RGB path: {cp}")

        # Fail-closed: any validation error stops before mesh loading or scoring
        if failures:
            failure_record = {
                "evaluation": {
                    "name": "room0_saved_map_track_identity",
                    "metrics_produced": False,
                    "scope_statement": "PCD validation failed; no metrics produced.",
                },
                "config": {
                    "pcd_path": str(PCD_PATH),
                    "pcd_sha256": sha256_file(PCD_PATH),
                    "num_pcd_tracks": findings["total_tracks"],
                    "num_pcd_observations": findings["total_observations"],
                    "recoverable_raw_rows": findings["recoverable_raw_rows"],
                    "malformed_frame_values": findings["malformed_frame_values"],
                    "duplicate_join_groups": findings["duplicate_join_groups"],
                    "duplicate_joins": findings["duplicate_joins"],
                    "duplicate_join_groups_detail": findings["duplicate_join_groups_detail"],
                },
                "failures": failures,
                "runtime": {
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "peak_rss_bytes": process.memory_info().rss,
                },
            }
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            result_path = RESULT_DIR / "room0_identity_result.json"
            tmp_path = RESULT_DIR / "room0_identity_result.json.tmp"
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(failure_record, f, indent=2, ensure_ascii=False, default=str)
                f.write("\n")
            tmp_path.rename(result_path)
            result_sha256 = sha256_file(result_path)
            print(f"Result written (fail-closed): {result_path} (SHA-256: {result_sha256})", flush=True)
            return 1

        # Aggregate observations: group by (track_id, source_frame) and union masks.
        # Union aggregation reduces raw rows to unique track-frame prediction units.
        aggregated_preds, agg_diagnostics = aggregate_observations(all_obs)
        raw_observation_count = agg_diagnostics["raw_observation_count"]
        unique_track_frame_count = agg_diagnostics["unique_track_frame_count"]
        duplicate_track_frame_groups = agg_diagnostics["duplicate_track_frame_groups"]
        extra_observation_components = agg_diagnostics["extra_observation_components"]
        print(
            f"  Raw rows: {raw_observation_count}, "
            f"unique track-frame units: {unique_track_frame_count}, "
            f"dup groups: {duplicate_track_frame_groups}, "
            f"extra components: {extra_observation_components}",
            flush=True,
        )

        # Validate counts
        if findings["total_tracks"] != 2794:
            print(f"  WARNING: Expected 2794 tracks, got {findings['total_tracks']}", flush=True)
        print(
            f"  Union aggregation reduces {raw_observation_count} raw rows to "
            f"{unique_track_frame_count} unique track-frame units without discarding mask pixels.",
            flush=True,
        )

        # Collect unique source frames from aggregated predictions
        pcd_source_frames: set[int] = set()
        for pred in aggregated_preds:
            pcd_source_frames.add(pred["source_frame"])
        print(f"  Unique source frames: {len(pcd_source_frames)}", flush=True)

        # Validate RGB paths exist for every aggregated prediction
        for pred in aggregated_preds:
            cp = pred["color_path"]
            if not isinstance(cp, (str, os.PathLike)):
                failures.append(f"Non-path color_path in aggregated predictions: {cp!r}")
                continue
            if isinstance(cp, str):
                cp = Path(cp)
            if not cp.is_file():
                failures.append(f"Missing RGB path: {cp}")

        # ================================================================
        # 2. Load GT data
        # ================================================================
        print("[room0-identity] Loading GT data...", flush=True)

        # PLY mesh
        if not PLY_PATH.is_file():
            raise FileNotFoundError(f"PLY not found: {PLY_PATH}")
        ply_sha = sha256_file(PLY_PATH)
        print(f"  PLY SHA-256: {ply_sha}", flush=True)
        vertices, triangles, triangle_object_ids = _read_ply_mesh(PLY_PATH)
        print(
            f"  Mesh: {vertices.shape[0]} vertices, {triangles.shape[0]} triangles, "
            f"{len(set(triangle_object_ids))} unique object IDs (range {triangle_object_ids.min()}-{triangle_object_ids.max()})",
            flush=True,
        )

        # Validate GT IDs are non-negative
        if (triangle_object_ids < 0).any():
            failures.append("Negative semantic IDs in PLY")

        # Trajectory poses
        if not TRAJ_PATH.is_file():
            raise FileNotFoundError(f"Trajectory not found: {TRAJ_PATH}")
        poses = load_pose_file(TRAJ_PATH)
        print(f"  Trajectory poses: {len(poses)}", flush=True)

        # Intrinsics
        if not INTRINSICS_PATH.is_file():
            raise FileNotFoundError(f"Intrinsics not found: {INTRINSICS_PATH}")
        intrinsics_data = load_intrinsics(INTRINSICS_PATH)
        print(f"  Intrinsics: {intrinsics_data}", flush=True)

        # Annotations
        if not ANNOTATIONS_PATH.is_file():
            raise FileNotFoundError(f"Annotations not found: {ANNOTATIONS_PATH}")
        id_to_label = load_annotations(ANNOTATIONS_PATH, "room_0")
        # Collect all annotation IDs (including unknowns/undefined) for the full set.
        all_annotation_ids: set[int] = set()
        for scan in _load_json(ANNOTATIONS_PATH)["scans"]:
            if scan["scan"] == "room_0":
                for row in scan["objects"]:
                    try:
                        all_annotation_ids.add(int(row["id"]))
                    except (TypeError, ValueError, KeyError):
                        pass
        # valid_annotation_ids is the full ReplicaSSG annotation set; IDs outside
        # this set are invalid-ID failures.  IDs with unknown/undefined labels are
        # still valid GT mask inclusion candidates.
        valid_annotation_ids: set[int] = all_annotation_ids
        print(f"  GT object labels: {len(id_to_label)} annotated ({len(all_annotation_ids)} total IDs)", flush=True)

        # ================================================================
        # 3. Build global mesh and raycasting scene once (CPU only)
        # ================================================================
        print("[room0-identity] Building global mesh/raycasting scene...", flush=True)
        _build_global_mesh(vertices, triangles)
        print("  Mesh and raycasting scene built.", flush=True)

        # ================================================================
        # 4. Render semantic masks and compute matches
        # ================================================================
        print("[room0-identity] Rendering and matching...", flush=True)

        # Index aggregated predictions by source frame
        frame_to_obs: dict[int, list[dict]] = defaultdict(list)
        for pred in aggregated_preds:
            frame_to_obs[pred["source_frame"]].append(pred)

        # Sorted unique source frames
        sorted_frames = sorted(pcd_source_frames)

        # Per-frame results
        frame_results: list[dict] = []
        # Cached per-frame data for overlays: {frame: {"matched": [(tid, gt),...], ...}}
        frame_overlay_cache: dict[int, dict] = {}
        all_matched_pairs: list[tuple[str, int, int]] = []  # (track_id_str, gt_id, frame)
        all_matched_ious: list[float] = []
        all_empty_denominator_frames: list[int] = []
        total_matched_count = 0
        total_unmatched_pred = 0
        total_unmatched_gt = 0
        total_unmatched_gt_ids: set[int] = set()
        total_unmatched_gt_occurrences = 0  # aggregate count across all frames

        # Track-to-GT ID mapping for identity prediction
        # {track_id_str: {frame: gt_id_int}} for matched observations
        track_gt_id_map: dict[str, dict[int, int]] = defaultdict(dict)

        for frame_idx in sorted_frames:
            frame_num = int(frame_idx)

            # Get observations for this frame
            frame_obs_list = frame_to_obs.get(frame_num, [])
            if not frame_obs_list:
                continue

            # Load RGB for camera check and overlays
            rgb_path = RGB_ROOT / f"frame{frame_num:06d}.jpg"
            if not rgb_path.is_file():
                failures.append(f"Missing RGB frame: {rgb_path}")
                continue
            rgb = load_image(rgb_path)
            img_h, img_w = rgb.shape[:2]

            # Camera check: 680x1200
            if img_h != 680 or img_w != 1200:
                failures.append(f"RGB dimension mismatch at frame {frame_num}: {rgb.shape}")
                continue

            # Get pose for this frame
            if frame_num >= len(poses):
                failures.append(f"Frame {frame_num} out of pose range ({len(poses)})")
                continue
            pose = poses[frame_num]

            # Build scaled intrinsics
            intrinsic = _build_intrinsics_matrix(intrinsics_data, img_w, img_h)

            # Convert Replica camera basis to OpenCV, then to world_to_camera
            c2w = pose.copy()
            c2w[:3, 1:3] *= -1
            world_to_camera = np.linalg.inv(c2w)

            # Render GT semantic mask using global raycasting scene
            gt_instance_map = _render_semantic_mask_frame(
                intrinsic, world_to_camera, triangle_object_ids, img_w, img_h
            )

            # Validate: only allow IDs that are in the ReplicaSSG annotation set.
            # Exclude ray misses (-1) and unannotated IDs.
            gt_ids_in_frame = set(np.unique(gt_instance_map)) - {-1}
            for gid in gt_ids_in_frame:
                if gid < 0:
                    failures.append(f"Invalid GT ID {gid} in frame {frame_num}")
                if gid not in all_annotation_ids:
                    failures.append(f"GT ID {gid} in frame {frame_num} not in ReplicaSSG annotations")

            # Build GT masks using only annotated IDs (ReplicaSSG set)
            # First, intersect rendered IDs with the annotation set.
            gt_ids_present = sorted(gid for gid in gt_ids_in_frame if gid in valid_annotation_ids)

            # Build GT mask list for present annotated instances
            gt_mask_list: list[tuple[int, np.ndarray]] = []
            for gid in gt_ids_present:
                gt_mask_list.append((gid, (gt_instance_map == gid).astype(bool)))

            # Build predicted track masks from PCD observations.
            # Track IDs are UUID strings.
            track_masks: dict[str, np.ndarray] = {}
            for obs in frame_obs_list:
                tid = obs["track_id"]  # UUID string
                if tid not in track_masks:
                    track_masks[tid] = obs["mask"].copy().astype(bool)
                else:
                    track_masks[tid] = np.logical_or(track_masks[tid], obs["mask"].astype(bool))

            # Check for empty masks and fail closed
            empty_tracks: list[str] = []
            for tid, mask in track_masks.items():
                if mask.sum() == 0:
                    empty_tracks.append(tid)
                    failures.append(f"Empty mask for track {tid} at frame {frame_num}")

            valid_track_ids = [tid for tid in track_masks if tid not in empty_tracks]

            # Build prediction mask list sorted by track_id string for determinism
            pred_mask_list: list[tuple[str, np.ndarray]] = []
            for tid in sorted(valid_track_ids):
                pred_mask_list.append((tid, track_masks[tid].copy()))

            # Compute IoU matrix
            n_pred = len(pred_mask_list)
            n_gt = len(gt_mask_list)

            if n_pred == 0 or n_gt == 0:
                empty_reasons: list[str] = []
                if n_gt == 0:
                    all_empty_denominator_frames.append(frame_num)
                    empty_reasons.append("GT denominator is zero")
                if n_pred == 0:
                    all_empty_denominator_frames.append(frame_num)
                    empty_reasons.append("prediction denominator is zero")
                for tid, _ in pred_mask_list:
                    total_unmatched_pred += 1

                frame_results.append({
                    "frame": frame_num,
                    "n_pred": n_pred,
                    "n_gt": n_gt,
                    "n_matched": 0,
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "matched_ious": [],
                    "unmatched_pred_count": n_pred,
                    "unmatched_gt_count": n_gt,
                    "empty_denominator": True,
                    "empty_denominator_reason": "; ".join(empty_reasons),
                })
                continue

            iou_matrix = compute_iou_matrix(
                [m for _, m in pred_mask_list],
                [m for _, m in gt_mask_list],
            )

            # Match using linear_sum_assignment
            matched_pairs, unmatched_pred, unmatched_gt = match_masks(iou_matrix, threshold=0.50)

            # Record matched pairs with track ID strings
            for pi, gi in matched_pairs:
                tid_str, gt_id = pred_mask_list[pi][0], gt_mask_list[gi][0]
                iou_val = float(iou_matrix[pi, gi])
                all_matched_pairs.append((tid_str, int(gt_id), frame_num))
                all_matched_ious.append(iou_val)
                total_matched_count += 1
                track_gt_id_map[tid_str][frame_num] = int(gt_id)

            for pi in unmatched_pred:
                tid_str = pred_mask_list[pi][0]
                total_unmatched_pred += 1

            for gi in unmatched_gt:
                gid = gt_mask_list[gi][0]
                total_unmatched_gt_ids.add(gid)
            total_unmatched_gt_occurrences += len(unmatched_gt)

            # Compute per-frame metrics
            n_matched = len(matched_pairs)
            frame_metrics = compute_precision_recall_f1(n_matched, n_pred, n_gt)

            matched_ious = [float(iou_matrix[pi, gi]) for pi, gi in matched_pairs]

            frame_results.append({
                "frame": frame_num,
                "n_pred": n_pred,
                "n_gt": n_gt,
                "n_matched": n_matched,
                "precision": frame_metrics["precision"],
                "recall": frame_metrics["recall"],
                "f1": frame_metrics["f1"],
                "matched_ious": matched_ious,
                "unmatched_pred_count": len(unmatched_pred),
                "unmatched_gt_count": len(unmatched_gt),
                "empty_denominator": False,
            })

            # Cache per-frame match data for overlay generation
            frame_matched_list: list[tuple[str, int]] = []
            for pi, gi in matched_pairs:
                tid_str, gt_id = pred_mask_list[pi][0], gt_mask_list[gi][0]
                frame_matched_list.append((tid_str, int(gt_id)))
            frame_unmatched_list: list[str] = [pred_mask_list[pi][0] for pi in unmatched_pred]

            frame_overlay_cache[frame_num] = {
                "matched": frame_matched_list,
                "unmatched": frame_unmatched_list,
            }

        # ================================================================
        # 5. Aggregate metrics
        # ================================================================
        print("[room0-identity] Computing aggregate metrics...", flush=True)

        zero_denom_frames = [fr["frame"] for fr in frame_results if fr["empty_denominator"]]
        if zero_denom_frames:
            reasons = [
                f"frame {fr['frame']}: {fr.get('empty_denominator_reason', 'unknown')}"
                for fr in frame_results
                if fr["empty_denominator"]
            ]
            failure_record = {
                "evaluation": {
                    "name": "room0_saved_map_track_identity",
                    "metrics_produced": False,
                    "scope_statement": "Zero-denominator frames detected after rendering; no metrics produced.",
                },
                "config": {
                    "pcd_path": str(PCD_PATH),
                    "num_pcd_tracks": findings["total_tracks"],
                    "raw_observation_count": raw_observation_count,
                    "unique_track_frame_count": unique_track_frame_count,
                    "num_source_frames": len(sorted_frames),
                    "zero_denominator_frames": zero_denom_frames,
                },
                "failures": [
                    f"Empty denominator frames: {zero_denom_frames}; reasons: {'; '.join(reasons)}"
                ],
                "zero_denominator_details": [
                    {"frame": fr["frame"], "reason": fr.get("empty_denominator_reason", "unknown")}
                    for fr in frame_results
                    if fr["empty_denominator"]
                ],
                "runtime": {
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "peak_rss_bytes": process.memory_info().rss,
                },
            }
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            result_path = RESULT_DIR / "room0_identity_result.json"
            tmp_path = RESULT_DIR / "room0_identity_result.json.tmp"
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(failure_record, f, indent=2, ensure_ascii=False, default=str)
                f.write("\n")
            tmp_path.rename(result_path)
            result_sha256 = sha256_file(result_path)
            print(f"Result written (fail-closed, zero-denominator): {result_path} (SHA-256: {result_sha256})", flush=True)
            return 1

        # Aggregate IoU statistics
        if all_matched_ious:
            mean_iou = float(np.mean(all_matched_ious))
            median_iou = float(np.median(all_matched_ious))
            std_iou = float(np.std(all_matched_ious))
            min_iou = float(np.min(all_matched_ious))
            max_iou = float(np.max(all_matched_ious))
        else:
            mean_iou = median_iou = std_iou = min_iou = max_iou = float("nan")

        # Per-frame macro precision/recall/F1
        all_precisions = [fr["precision"] for fr in frame_results if not np.isnan(fr["precision"])]
        all_recalls = [fr["recall"] for fr in frame_results if not np.isnan(fr["recall"])]
        all_f1s = [fr["f1"] for fr in frame_results if not np.isnan(fr["f1"])]

        # Micro counts: sum matched/predicted/GT across all frames
        micro_matched = sum(fr["n_matched"] for fr in frame_results)
        micro_pred_total = sum(fr["n_pred"] for fr in frame_results)
        micro_gt_total = sum(fr["n_gt"] for fr in frame_results)

        # ================================================================
        # 6. Track identity prediction
        # ================================================================
        print("[room0-identity] Computing track identity...", flush=True)

        identity_eligible = 0
        identity_correct = 0
        identity_ties = 0
        identity_frame_results: list[dict] = []

        # Build a lookup: {(track_id_str, frame)} -> actual_gt_id from matched pairs
        matched_lookup: dict[tuple[str, int], int] = {}
        for tid_str, gt_id, frame in all_matched_pairs:
            matched_lookup[(tid_str, frame)] = gt_id

        for pred in aggregated_preds:
            tid = pred["track_id"]  # UUID string
            frame = pred["source_frame"]

            # Only predict for matched observations
            if (tid, frame) not in matched_lookup:
                continue

            actual_gt = matched_lookup[(tid, frame)]
            predicted, is_tie = predict_identity(track_gt_id_map, frame, tid)
            if is_tie:
                identity_ties += 1

            if predicted is not None:
                identity_eligible += 1
                if predicted == actual_gt:
                    identity_correct += 1

                identity_frame_results.append({
                    "track_id": tid,
                    "frame": frame,
                    "predicted_gt_id": predicted,
                    "actual_gt_id": actual_gt,
                    "correct": 1 if predicted == actual_gt else 0,
                })

        # Coverage-adjusted top-1 and coverage use total unique track-frame
        # prediction count as denominator; raw-row counts are descriptive only.
        raw_total_observations = findings["recoverable_raw_rows"]
        malformed_frame_values = findings["malformed_frame_values"]
        total_observations = unique_track_frame_count
        coverage_adjusted_top1 = identity_correct / total_observations if total_observations > 0 else float("nan")
        if identity_eligible > 0:
            top1 = identity_correct / identity_eligible
            identity_status = "computed"
        else:
            top1 = None
            identity_status = "not_estimable_zero_eligible"
        coverage = identity_eligible / total_observations if total_observations > 0 else float("nan")

        # GT-object fragmentation: distinct map tracks per matched GT ID
        gt_to_tracks: dict[int, set[str]] = defaultdict(set)
        for tid_str, gt_id, _ in all_matched_pairs:
            gt_to_tracks[int(gt_id)].add(tid_str)
        fragmentation = {
            str(gt_id): len(tracks) for gt_id, tracks in gt_to_tracks.items()
        }

        # ================================================================
        # 7. Hash input source frames
        # ================================================================
        source_frame_hashes: dict[str, str] = {}
        for frame_num in sorted_frames:
            rgb_path = RGB_ROOT / f"frame{frame_num:06d}.jpg"
            if rgb_path.is_file():
                source_frame_hashes[str(rgb_path)] = sha256_file(rgb_path)

        # ================================================================
        # 8. Produce overlays for requested frames
        # ================================================================
        print("[room0-identity] Producing overlays...", flush=True)

        OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
        overlay_hashes: dict[str, str] = {}

        for overlay_frame in OVERLAY_FRAMES:
            if overlay_frame not in pcd_source_frames:
                print(f"  Skipping overlay frame {overlay_frame} (not in PCD source frames)", flush=True)
                continue
            if overlay_frame not in frame_to_obs:
                print(f"  Skipping overlay frame {overlay_frame} (no PCD observations)", flush=True)
                continue
            if overlay_frame not in frame_overlay_cache:
                print(f"  Skipping overlay frame {overlay_frame} (no match data)", flush=True)
                continue

            # Load RGB
            rgb_path = RGB_ROOT / f"frame{overlay_frame:06d}.jpg"
            if not rgb_path.is_file():
                continue
            rgb = load_image(rgb_path)

            # Render GT mask for this frame
            pose = poses[overlay_frame]
            intrinsic = _build_intrinsics_matrix(intrinsics_data, 1200, 680)
            c2w = pose.copy()
            c2w[:3, 1:3] *= -1
            world_to_camera = np.linalg.inv(c2w)
            gt_instance_map = _render_semantic_mask_frame(
                intrinsic, world_to_camera, triangle_object_ids, 1200, 680
            )

            # Build track label map (integer labels) and label-to-UUID mapping.
            # label_to_uuid_map: int -> str for make_overlay display.
            # track_to_label: str -> int for UUID lookup when painting masks.
            track_label_map = np.full((680, 1200), -1, dtype=np.int64)
            label_to_uuid_map: dict[int, str] = {}
            track_to_label: dict[str, int] = {}
            next_label = 0
            for obs in frame_to_obs[overlay_frame]:
                tid = obs["track_id"]
                if tid not in track_to_label:
                    track_to_label[tid] = next_label
                    label_to_uuid_map[next_label] = tid
                    next_label += 1
                lbl = track_to_label[tid]
                mask = obs["mask"].astype(bool)
                ys, xs = np.where(mask)
                for y, x in zip(ys, xs):
                    track_label_map[y, x] = lbl

            # Get per-frame matched/unmatched from cache (not from outer loop)
            cache_data = frame_overlay_cache[overlay_frame]
            frame_matched = cache_data["matched"]
            frame_unmatched = cache_data["unmatched"]

            # Produce overlay
            overlay = make_overlay(
                rgb, gt_instance_map, track_label_map,
                label_to_uuid_map,
                overlay_frame,
                frame_matched,
                frame_unmatched,
            )
            overlay_path = OVERLAY_DIR / f"room0_identity_{overlay_frame:04d}.png"
            overlay.save(overlay_path, format="PNG")
            overlay_sha = sha256_file(overlay_path)
            print(f"  Saved overlay: {overlay_path} (SHA-256: {overlay_sha})", flush=True)
            overlay_hashes[str(overlay_path)] = overlay_sha

        # ================================================================
        # 9. Compile result JSON
        # ================================================================
        elapsed = time.perf_counter() - started_at
        peak_rss = process.memory_info().rss
        python_version = sys.version
        scipy_version_str = f"{scipy.__version__}"
        numpy_version_str = f"{np.__version__}"
        pillow_version_str = f"{Image.__version__}"
        psutil_version_str = f"{psutil.__version__}"

        # Build source frame path entries
        source_frame_entries = []
        for frame_num in sorted_frames:
            rgb_path = RGB_ROOT / f"frame{frame_num:06d}.jpg"
            if rgb_path.is_file():
                source_frame_entries.append({
                    "path": str(rgb_path),
                    "sha256": source_frame_hashes.get(str(rgb_path), "unknown"),
                })

        result = {
            "evaluation": {
                "name": "room0_saved_map_track_identity",
                "scene": "room0",
                "scope_statement": (
                    "This evaluation measures how consistently the existing saved room0 map associates "
                    "its observations to ReplicaSSG semantic instance IDs. It evaluates one saved map "
                    "and does not establish full-scene coverage or performance on other scenes."
                ),
            },
            "config": {
                "pcd_path": str(PCD_PATH),
                "pcd_sha256": sha256_file(PCD_PATH),
                "ply_path": str(PLY_PATH),
                "ply_sha256": ply_sha,
                "traj_path": str(TRAJ_PATH),
                "traj_sha256": sha256_file(TRAJ_PATH),
                "intrinsics_path": str(INTRINSICS_PATH),
                "intrinsics_sha256": sha256_file(INTRINSICS_PATH),
                "annotations_path": str(ANNOTATIONS_PATH),
                "annotations_sha256": sha256_file(ANNOTATIONS_PATH),
                "rgb_root": str(RGB_ROOT),
                "iou_threshold": 0.50,
                "overlay_frames": OVERLAY_FRAMES,
                "dataset": "Replica",
                "scene": "room0",
                "stride": 10,
                "num_source_frames": len(sorted_frames),
                "num_pcd_tracks": findings["total_tracks"],
                "raw_observation_count": raw_observation_count,
                "unique_track_frame_count": unique_track_frame_count,
                "duplicate_track_frame_groups": duplicate_track_frame_groups,
                "extra_observation_components": extra_observation_components,
                "duplicate_joins": findings["duplicate_joins"],
                "duplicate_join_groups_detail": findings["duplicate_join_groups_detail"],
                "recoverable_raw_rows": findings["recoverable_raw_rows"],
                "malformed_frame_values": findings["malformed_frame_values"],
            },
            "definitions": {
                "precision": "Matched predictions / all predictions (per-frame, then macro-averaged)",
                "recall": "Matched predictions / all GT instances (per-frame, then macro-averaged)",
                "f1": "2 * precision * recall / (precision + recall)",
                "matched_ious": "Per-match IoU between predicted track mask and GT instance mask",
                "top_1_identity": "Fraction of eligible matched unique track-frame predictions where predicted GT ID equals actual GT ID",
                "coverage_adjusted_top_1": "Correct predictions / total unique track-frame prediction count (includes zero-coverage units)",
                "coverage": "Eligible predictions / total unique track-frame prediction count",
            },
            "denominators": {
                "raw_total_observations": raw_observation_count,
                "unique_track_frame_count": unique_track_frame_count,
                "duplicate_track_frame_groups": duplicate_track_frame_groups,
                "extra_observation_components": extra_observation_components,
                "precision": "per-frame prediction counts (micro sum: %d)" % micro_pred_total,
                "recall": "per-frame GT counts (micro sum: %d)" % micro_gt_total,
                "top_1": identity_eligible,
                "coverage_adjusted_top_1": total_observations,
            },
            "metrics": {
                "aggregate": {
                    "mask_precision_macro": float(np.mean(all_precisions)) if all_precisions else float("nan"),
                    "mask_recall_macro": float(np.mean(all_recalls)) if all_recalls else float("nan"),
                    "mask_f1_macro": float(np.mean(all_f1s)) if all_f1s else float("nan"),
                    "mask_precision_micro": micro_matched / micro_pred_total if micro_pred_total > 0 else float("nan"),
                    "mask_recall_micro": micro_matched / micro_gt_total if micro_gt_total > 0 else float("nan"),
                    "mean_matched_iou": mean_iou,
                    "median_matched_iou": median_iou,
                    "std_matched_iou": std_iou,
                    "min_matched_iou": min_iou,
                    "max_matched_iou": max_iou,
                    "total_matched": total_matched_count,
                    "total_unmatched_predictions": total_unmatched_pred,
                    "total_unmatched_gt_unique_ids": len(total_unmatched_gt_ids),
                    "total_unmatched_gt_occurrences": total_unmatched_gt_occurrences,
                    "total_gt_denominator_frames": len(all_empty_denominator_frames),
                    "micro_matched": micro_matched,
                    "micro_pred_total": micro_pred_total,
                    "micro_gt_total": micro_gt_total,
                },
                "identity": {
                    "eligible": identity_eligible,
                    "correct": identity_correct,
                    "top_1": top1,
                    "coverage_adjusted_top_1": coverage_adjusted_top1,
                    "coverage": coverage,
                    "identity_status": identity_status,
                    "tie_count": identity_ties,
                    "total_unique_track_frame_predictions": total_observations,
                    "total_raw_observations": raw_observation_count,
                },
                "fragmentation": {
                    "gt_object_to_track_count": fragmentation,
                },
            },
            "per_frame": frame_results,
            "per_frame_identity": identity_frame_results,
            "input_hashes": {
                "pcd": sha256_file(PCD_PATH),
                "ply": ply_sha,
                "traj": sha256_file(TRAJ_PATH),
                "intrinsics": sha256_file(INTRINSICS_PATH),
                "annotations": sha256_file(ANNOTATIONS_PATH),
                "source_frames": source_frame_entries,
            },
            "versions": {
                "python": python_version,
                "numpy": numpy_version_str,
                "scipy": scipy_version_str,
                "open3d": str(o3d.__version__),
                "pillow": pillow_version_str,
                "psutil": psutil_version_str,
            },
            "runtime": {
                "elapsed_seconds": elapsed,
                "peak_rss_bytes": peak_rss,
                "peak_rss_mb": peak_rss / (1024 * 1024),
            },
            "artifacts": {
                "result_json": str(RESULT_DIR / "room0_identity_result.json"),
                "overlays": {
                    str(OVERLAY_DIR / f"room0_identity_{f:04d}.png"): overlay_hashes.get(
                        str(OVERLAY_DIR / f"room0_identity_{f:04d}.png"), "unknown"
                    )
                    for f in OVERLAY_FRAMES
                    if f in pcd_source_frames
                },
            },
            "failures": failures,
            "limitations": [
                "This evaluates one saved room0 map and does not establish full-scene coverage.",
                "Performance on other scenes is not assessed by this evaluation.",
                "Track IDs are UUID strings; they are not converted to integers.",
                f"Frame alignment validation: {'passed' if findings['frame_alignment_ok'] else 'failed'}.",
            ],
        }

        # ================================================================
        # 10. Write result JSON atomically
        # ================================================================
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        result_path = RESULT_DIR / "room0_identity_result.json"
        tmp_path = RESULT_DIR / "room0_identity_result.json.tmp"

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            f.write("\n")
        tmp_path.rename(result_path)
        result_sha256 = sha256_file(result_path)

        print(f"Result written: {result_path} (SHA-256: {result_sha256})", flush=True)

        print(f"Runtime: {elapsed:.2f}s, Peak RSS: {peak_rss / (1024*1024):.1f} MB", flush=True)

        if failures:
            print(f"Failures: {failures}", flush=True)
            return 1

        return 0

    except Exception as exc:
        print(f"[room0-identity] ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()

        # Write failure JSON to the specified ignored project result directory.
        try:
            elapsed = time.perf_counter() - started_at
            peak_rss = process.memory_info().rss
            failure_result = {
                "evaluation": {
                    "name": "room0_saved_map_track_identity",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "scope_statement": "Failed to complete evaluation.",
                },
                "config": {"pcd_path": str(PCD_PATH)},
                "failures": [str(exc)],
                "runtime": {
                    "elapsed_seconds": elapsed,
                    "peak_rss_bytes": peak_rss,
                },
            }
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            result_path = RESULT_DIR / "room0_identity_result.json"
            with result_path.open("w", encoding="utf-8") as f:
                json.dump(failure_result, f, indent=2, ensure_ascii=False, default=str)
            print(f"Error result written: {result_path}", flush=True)
        except Exception:
            print("Failed to write error result.", file=sys.stderr)

        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[room0-identity] UNEXPECTED ERROR: {exc}", file=sys.stderr)
        raise
