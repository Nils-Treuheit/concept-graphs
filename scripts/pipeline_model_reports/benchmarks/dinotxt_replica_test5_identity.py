"""
DINOtxt cross-view graph-node identity runner for ReplicaSSG test-5.

Leave-one-frame-out retrieval of identity from cached 2048-d FP32 normalized
DINOv3/dino.txt embeddings.  CPU only - no model load, no GPU, no inference.

Protocol
--------
1. Join each per_view_scores key (scene/frame-NNNN-object-ID) to exactly one
   -box.png crop-manifest entry and normalized 2048-d embedding by SHA-256.
2. Validate all 2,204 score rows, 49 categories, unique box-crop hashes,
   matching object node IDs against objects.json, exact frozen labels, and
   finite normalized vectors.  Fail closed on any mismatch.
3. Identity is (scene, int(object_id)).  For each query observation exclude
   every view that shares the same frame string.  Average remaining normalized
   embeddings per eligible node, L2-normalize each prototype, then rank eligible
   prototypes by FP32 cosine similarity.  Exclude nodes with no remaining
   observations.
4. Deterministic tie-break: descending unrounded FP32 cosine, scene name,
   then integer object ID.  Compute top-1 accuracy, MRR, Recall@1/5/10
   overall and per-scene.  Record candidate-count distributions, per-query
   ranks/scores, and node/view support counts.
5. Count exact frozen-scene relation tuples from relationships.json.  Do not
   infer visual relation-triplet recall - cached inputs carry only individual
   crop embeddings.
6. Write a unique result JSON under
   models/experiments/t_ren/data/dinotxt_test5/
   using atomic no-overwrite semantics.

Inputs (read-only)
------------------
* models/experiments/t_ren/data/dinotxt_test5/dinotxt_test5_global_retrieval_20261116T161741Z_2382729.json
  (the existing score JSON - used for per_view_scores + category map)
* models/experiments/t_ren/data/dinotxt_test5/dinotxt_test5_global_embeddings_cache.npz
* models/experiments/t_ren/data/dinotxt_test5/t_ren_test5_crop_manifest_20261116.json
* /mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/ReplicaSSG/files/objects.json
* /mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/ReplicaSSG/files/relationships.json

Output
------
* models/experiments/t_ren/data/dinotxt_test5/dinotxt_test5_identity_<UTC>_<pid>.json
"""

import hashlib
import json
import os
import pathlib
import platform
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths - canonical project layout described in the protocol.
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path("/mnt/HDD1/Project_Code/concept-graphs")

DATA_DIR = PROJECT_ROOT / "models" / "experiments" / "t_ren" / "data" / "dinotxt_test5"

SCORES_JSON = DATA_DIR / "dinotxt_test5_global_retrieval_20261116T161741Z_2382729.json"
EMBEDDINGS_NPZ = DATA_DIR / "dinotxt_test5_global_embeddings_cache.npz"
EMBEDDINGS_JSON = DATA_DIR / "dinotxt_test5_global_embeddings_cache.json"
MANIFEST_JSON = DATA_DIR / "t_ren_test5_crop_manifest_20261116.json"

REPLICA_ROOT = pathlib.Path(
    "/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/ReplicaSSG/files"
)
OBJECTS_JSON = REPLICA_ROOT / "objects.json"
RELATIONSHIPS_JSON = REPLICA_ROOT / "relationships.json"

VALID_SCENES = {"office_1", "office_3", "office_4", "room_1", "room_2"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def atomic_write_json(
    path: pathlib.Path,
    obj: Any,
    *,
    _data_bytes: bytes | None = None,
) -> pathlib.Path:
    """Write JSON atomically via ``tempfile.mkstemp`` in the output
    directory.  Serialize to bytes first so any JSON error leaves no temp
    file.  Then create the temp file, buffered ``fdopen`` write, flush +
    fsync, hard-link no-replace, parent-directory fsync, and a single
    outer ``finally`` that closes any still-open descriptor and unlinks
    the temp on every failure.  Never overwrites the destination.
    """
    dest_dir = path.parent
    # Serialize before creating the temp file.
    data_bytes: bytes
    if _data_bytes is not None:
        data_bytes = _data_bytes
    else:
        data_bytes = json.dumps(
            obj, indent=2, ensure_ascii=False, default=str
        ).encode("utf-8")
    fd: int | None = None
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=path.name + ".tmp.",
            dir=str(dest_dir),
        )
        with os.fdopen(fd, "wb") as fh:
            fd = None
            fh.write(data_bytes)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(str(tmp_path), str(path))
    except FileExistsError:
        raise FileExistsError(f"Result already exists: {path}")
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    # fsync the parent directory so the hard-link is durable.
    dir_fd = os.open(str(dest_dir), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return path


def load_score_json() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    """Load the existing scores JSON; return (full_data, config, per_view_scores)."""
    if not SCORES_JSON.exists():
        raise FileNotFoundError(f"Scores JSON not found: {SCORES_JSON}")
    with open(SCORES_JSON) as f:
        data = json.load(f)
    return data, data.get("config", {}), data.get("per_view_scores", {})


def load_manifest() -> Tuple[Dict[str, Any], List[Dict]]:
    with open(MANIFEST_JSON) as f:
        data = json.load(f)
    return data, data.get("entries", [])


def load_objects() -> Dict[str, Any]:
    with open(OBJECTS_JSON) as f:
        return json.load(f)


def load_relationships() -> Dict[str, Any]:
    with open(RELATIONSHIPS_JSON) as f:
        return json.load(f)


def load_embeddings() -> Any:
    return np.load(EMBEDDINGS_NPZ)


def find_embedding_sidecar_json() -> pathlib.Path:
    """Pin the exact sidecar path declared in the protocol."""
    p = DATA_DIR / "dinotxt_test5_global_embeddings_cache.json"
    if not p.exists():
        raise FileNotFoundError(f"Embedding sidecar JSON not found: {p}")
    return p


def compute_relation_tuples(
    relationships: Dict, test_scenes: set,
) -> Dict[str, Any]:
    """Count exact relation tuples from relationships.json for test scenes."""
    total = 0
    per_scene: Dict[str, int] = {}
    relation_names: Dict[str, int] = defaultdict(int)
    relation_ids: Dict[int, str] = {}

    for scan in relationships.get("scans", []):
        scene = scan["scan"]
        if scene not in test_scenes:
            continue
        tuples = scan.get("relationships", [])
        per_scene[scene] = len(tuples)
        total += len(tuples)
        for t in tuples:
            if len(t) >= 4:
                rel_id = t[2]
                rel_name = t[3]
                relation_ids[rel_id] = rel_name
                relation_names[rel_name] += 1

    return {
        "total_tuples": total,
        "per_scene": per_scene,
        "unique_relation_names": sorted(relation_names.keys()),
        "relation_name_counts": dict(sorted(relation_names.items())),
        "unique_relation_ids": sorted(relation_ids.keys()),
        "relation_id_to_name": {str(k): v for k, v in sorted(relation_ids.items())},
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_inputs(
    per_view_scores: Dict[str, Dict],
    manifest_entries: List[Dict],
    objects_data: Dict,
    embed_data: Any,
    relationships: Dict,
) -> Tuple[Dict[str, Dict], Dict[str, Dict], set]:
    """
    Build join tables and validate:
    - All 2,204 score rows present.
    - 49 unique categories.
    - Each box crop SHA-256 is unique and matches an embedding key.
    - Object IDs and labels match objects.json per scene.
    - All embeddings are finite, shape (2048,), L2-norm ~1.0.
    - Exactly five expected scenes present.
    - No duplicate box manifest paths.
    - Every box-index entry joins to a score row (1-to-1).
    - relationships.json has exactly one scan per expected scene.

    Returns (resolved, scene_object_map, validated_box_hashes).
    """
    # Index manifest by box path; fail on duplicates.
    box_index: Dict[str, Dict] = {}
    seen_paths: set = set()
    for entry in manifest_entries:
        p = entry["path"]
        if p.endswith("-box.png"):
            if p in seen_paths:
                raise ValueError(f"Duplicate manifest path: {p}")
            seen_paths.add(p)
            box_index[p] = entry

    # Build scene->object map from objects.json.
    # Reject duplicate scene scans and duplicate object IDs per scene.
    scene_object_map: Dict[str, Dict[str, str]] = {}
    seen_scene_scans: set[str] = set()
    for scan in objects_data.get("scans", []):
        scene = scan["scan"]
        if scene not in VALID_SCENES:
            continue
        if scene in seen_scene_scans:
            raise ValueError(
                f"Duplicate scan for expected scene '{scene}' in objects.json"
            )
        seen_scene_scans.add(scene)
        obj_map: Dict[str, str] = {}
        obj_int_seen: Dict[int, None] = {}
        for obj in scan["objects"]:
            raw_id = obj.get("id")
            if raw_id is None:
                raise ValueError(f"Missing 'id' in objects.json scan={scene}")
            try:
                obj_id_int = int(raw_id)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Non-integer object id '{raw_id}' in scene '{scene}' "
                    f"objects.json"
                )
            if obj_id_int in obj_int_seen:
                raise ValueError(
                    f"Duplicate integer object id '{obj_id_int}' "
                    f"(raw '{raw_id}') in scene '{scene}' objects.json"
                )
            obj_int_seen[obj_id_int] = None
            obj_map[str(obj_id_int)] = obj["label"]
        scene_object_map[scene] = obj_map

    # Verify every expected scene was scanned exactly once.
    for scene in sorted(VALID_SCENES):
        if scene not in scene_object_map:
            raise ValueError(
                f"Missing objects.json scan for expected scene '{scene}'"
            )

    # Validate relationships.json: exactly one scan per expected scene.
    rel_scene_scans: set[str] = set()
    rel_data = relationships
    for scan in rel_data.get("scans", []):
        rscene = scan.get("scan")
        if rscene in VALID_SCENES:
            if rscene in rel_scene_scans:
                raise ValueError(
                    f"Duplicate scan for expected scene '{rscene}' in relationships.json"
                )
            rel_scene_scans.add(rscene)
    for scene in sorted(VALID_SCENES):
        if scene not in rel_scene_scans:
            raise ValueError(
                f"Missing relationships.json scan for expected scene '{scene}'"
            )

    score_keys = set(per_view_scores.keys())
    actual_scenes: set = set()
    categories: set = set()
    box_hashes: set = set()
    join_errors: List[str] = []
    resolved: Dict[str, Dict] = {}

    for score_key in sorted(score_keys):
        parts = score_key.split("/", 1)
        if len(parts) != 2:
            join_errors.append(f"Malformed score key: {score_key}")
            continue
        scene, frame_obj = parts
        actual_scenes.add(scene)
        if scene not in VALID_SCENES:
            join_errors.append(f"Unexpected scene in score key: {scene}")
            continue

        frame_obj_parts = frame_obj.split("-object-", 1)
        if len(frame_obj_parts) != 2:
            join_errors.append(f"Cannot parse frame/object from {score_key}")
            continue
        frame_str, obj_id_str_raw = frame_obj_parts

        # Validate object ID is parseable as integer.
        try:
            obj_id_int = int(obj_id_str_raw)
        except (ValueError, TypeError):
            join_errors.append(
                f"Non-integer object id '{obj_id_str_raw}' in score key {score_key}"
            )
            continue

        obj_id_str = str(obj_id_int)

        box_path = f"{scene}/{frame_obj}-box.png"
        if box_path not in box_index:
            join_errors.append(f"No box manifest entry for {score_key}")
            continue
        manifest_entry = box_index[box_path]
        sha256_val = manifest_entry["sha256"]

        if sha256_val not in embed_data.files:
            join_errors.append(f"Embedding missing for {score_key} (sha256={sha256_val})")
            continue

        emb = embed_data[sha256_val]
        if emb.shape != (2048,):
            join_errors.append(f"Wrong embedding shape for {score_key}: {emb.shape}")
            continue
        if not np.all(np.isfinite(emb)):
            join_errors.append(f"Non-finite embedding for {score_key}")
            continue
        norm = float(np.linalg.norm(emb))
        if abs(norm - 1.0) > 1e-4:
            join_errors.append(f"Embedding norm {norm:.6f} != 1.0 for {score_key}")
            continue

        expected_label = scene_object_map.get(scene, {}).get(obj_id_str)
        cat = per_view_scores[score_key].get("category", "")
        if expected_label is None:
            join_errors.append(f"No objects.json label for {scene}/object-{obj_id_str}")
        elif cat != expected_label:
            join_errors.append(
                f"Label mismatch for {score_key}: scores says '{cat}', "
                f"objects.json says '{expected_label}'"
            )

        categories.add(cat)
        if sha256_val in box_hashes:
            join_errors.append(f"Duplicate box hash {sha256_val} in {score_key}")
        box_hashes.add(sha256_val)

        resolved[score_key] = {
            "scene": scene,
            "frame": frame_str,
            "object_id": obj_id_str_raw,
            "object_id_str": obj_id_str,
            "object_id_int": obj_id_int,
            "category": cat,
            "sha256": sha256_val,
            "manifest": manifest_entry,
            "embedding": emb,
        }

    if len(actual_scenes) != len(VALID_SCENES):
        join_errors.append(
            f"Scene count mismatch: expected {len(VALID_SCENES)}, "
            f"got {len(actual_scenes)} ({actual_scenes})"
        )

    total_rows = len(resolved)
    if total_rows != 2204:
        join_errors.append(f"Expected 2204 score rows, got {total_rows}")

    if len(categories) != 49:
        join_errors.append(f"Expected 49 categories, got {len(categories)}")

    # Assert 1-to-1 join: every manifest box path matched exactly once.
    if len(box_index) != len(box_hashes):
        join_errors.append(
            f"Manifest join mismatch: box_index has {len(box_index)} entries "
            f"but only {len(box_hashes)} unique hashes"
        )

    if join_errors:
        raise RuntimeError(
            f"Validation failed with {len(join_errors)} error(s):\n"
            + "\n".join(join_errors[:50])
        )

    return resolved, scene_object_map, box_hashes


# ---------------------------------------------------------------------------
# Frame and node indexing
# ---------------------------------------------------------------------------


def build_frame_map(
    resolved: Dict[str, Dict],
) -> Dict[str, Dict[str, List[str]]]:
    """scene -> { frame_str: [score_keys] }."""
    frame_map: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sk, info in resolved.items():
        frame_map[info["scene"]][info["frame"]].append(sk)
    return frame_map


def build_node_views(
    resolved: Dict[str, Dict],
) -> Dict[str, Dict[str, List[str]]]:
    """scene -> { object_id_str: [score_keys] }."""
    node_views: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sk, info in resolved.items():
        node_views[info["scene"]][info["object_id_str"]].append(sk)
    return node_views


# ---------------------------------------------------------------------------
# Per-query leave-one-frame-out prototype computation
# ---------------------------------------------------------------------------


def compute_prototypes(
    resolved: Dict[str, Dict],
    frame_map: Dict[str, Dict[str, List[str]]],
    node_views: Dict[str, Dict[str, List[str]]],
) -> Tuple[
    Dict[str, Dict[int, np.ndarray]],  # query_key -> {obj_id_int: prototype}
    Dict[str, Dict[str, int]],  # (scene, obj_id_str) -> alternate-frame count
    Dict[str, Dict[str, bool]],  # query_key -> {obj_id_str: has_support}
    Dict[str, int],  # query_key -> candidate count (int)
]:
    """
    For each query observation (leave-one-frame-out):
    - Exclude all views from the same frame as the query.
    - For every object node in the same scene, collect embeddings whose
      frame differs from the query frame.
    - Average collected embeddings per node, L2-normalize the prototype.
    - Store per-query candidate map and support flags.

    Returns:
        query_prototypes:  query_sk -> {obj_id_int: L2-normalized prototype}
        node_alternate_counts: scene -> {obj_id_str: count of alternate-frame views}
        query_support:    query_sk -> {obj_id_str: has_alternate_support}
        query_candidate_counts: query_sk -> {obj_id_str: number of candidates}
    """
    node_alternate_counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    query_prototypes: Dict[str, Dict[int, np.ndarray]] = {}
    query_support: Dict[str, Dict[str, bool]] = {}
    query_candidate_counts: Dict[str, Dict[str, int]] = {}

    for scene in sorted(VALID_SCENES):
        scene_frames = frame_map.get(scene, {})
        scene_objects = node_views.get(scene, {})
        if not scene_frames or not scene_objects:
            continue

        # Collect per-object, per-frame embedding lists for this scene.
        # obj_id_str -> frame_str -> [embeddings]
        obj_frame_embs: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for frame_str, sk_list in scene_frames.items():
            for sk in sk_list:
                info = resolved[sk]
                obj_id = info["object_id_str"]
                obj_frame_embs[obj_id][frame_str].append(info["embedding"])

        # Compute alternate-frame counts per object (across all frames).
        for obj_id, frame_embs in obj_frame_embs.items():
            total = sum(len(embs) for embs in frame_embs.values())
            node_alternate_counts[scene][obj_id] = total

        # Build per-query prototypes: exclude query's own frame.
        for frame_str, sk_list in scene_frames.items():
            for sk in sk_list:
                info = resolved[sk]
                obj_id_str = info["object_id_str"]

                # Collect alternate embeddings per object (exclude frame_str).
                candidates: Dict[int, List[np.ndarray]] = {}
                for alt_obj_id, frame_embs in obj_frame_embs.items():
                    alt_embs: List[np.ndarray] = []
                    for f, embs in frame_embs.items():
                        if f != frame_str:
                            alt_embs.extend(embs)
                    if alt_embs:
                        avg = np.stack(alt_embs).mean(axis=0)
                        proto = l2_normalize(avg)
                        candidates[int(alt_obj_id)] = proto

                query_prototypes[sk] = candidates

                # Track per-object support (has >=1 alternate-frame view).
                support: Dict[str, bool] = {}
                for alt_obj_id, frame_embs in obj_frame_embs.items():
                    alt_count = sum(
                        len(embs) for f, embs in frame_embs.items() if f != frame_str
                    )
                    support[str(alt_obj_id)] = alt_count > 0

                query_support[sk] = support

                # Per-query candidate count — single integer equal to map size.
                query_candidate_counts[sk] = len(candidates)

    return (
        query_prototypes,
        dict(node_alternate_counts),
        query_support,
        query_candidate_counts,
    )


# ---------------------------------------------------------------------------
# Candidate ranking
# ---------------------------------------------------------------------------


def rank_candidates(
    query_sk: str,
    query_emb: np.ndarray,
    query_scene: str,
    query_obj_id_int: int,
    query_prototypes: Dict[str, Dict[int, np.ndarray]],
) -> List[Tuple[int, float]]:
    """
    Rank eligible prototypes for a single query.
    Returns list of (object_id_int, cosine_score) sorted by
    descending cosine (FP32), then scene name, then object_id_int.
    The correct node IS included when it has support.
    """
    protos = query_prototypes.get(query_sk, {})
    scores: List[Tuple[int, float]] = []
    for obj_id, proto in protos.items():
        cos = float(np.dot(query_emb, proto))
        scores.append((obj_id, cos))
    scores.sort(key=lambda x: (-x[1], query_scene, x[0]))
    return scores


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics_from_scores(
    scores_list: List[Tuple[str, int, int]],
) -> Dict[str, Any]:
    """Compute top-1 accuracy, MRR, Recall@1/5/10 overall and per-scene."""
    overall = {"top1": 0, "rr": 0.0, "r1": 0, "r5": 0, "r10": 0, "total": 0}
    scene_stats: Dict[str, Dict] = {}

    for scene, obj_id, rank in scores_list:
        overall["total"] += 1
        if rank == 1:
            overall["top1"] += 1
        if rank >= 1:
            overall["rr"] += 1.0 / rank
        if rank >= 1 and rank <= 1:
            overall["r1"] += 1
        if rank >= 1 and rank <= 5:
            overall["r5"] += 1
        if rank >= 1 and rank <= 10:
            overall["r10"] += 1

        if scene not in scene_stats:
            scene_stats[scene] = {
                "top1": 0, "rr": 0.0, "r1": 0, "r5": 0, "r10": 0, "total": 0
            }
        ss = scene_stats[scene]
        ss["total"] += 1
        if rank == 1:
            ss["top1"] += 1
        if rank >= 1:
            ss["rr"] += 1.0 / rank
        if rank >= 1 and rank <= 1:
            ss["r1"] += 1
        if rank >= 1 and rank <= 5:
            ss["r5"] += 1
        if rank >= 1 and rank <= 10:
            ss["r10"] += 1

    def _summary(s: Dict) -> Dict[str, float]:
        t = s["total"]
        if t == 0:
            return {
                "top1_accuracy": 0.0, "mrr": 0.0,
                "recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0,
            }
        return {
            "top1_accuracy": s["top1"] / t,
            "mrr": s["rr"] / t,
            "recall_at_1": s["r1"] / t,
            "recall_at_5": s["r5"] / t,
            "recall_at_10": s["r10"] / t,
        }

    return {
        "overall": _summary(overall),
        "per_scene": {s: _summary(scene_stats[s]) for s in sorted(scene_stats)},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()

    print("=== DINOtxt ReplicaSSG test-5 identity runner ===")
    print(f"PID={os.getpid()}  Python={platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    # 1. Load inputs.
    print("\n[1/6] Loading scores JSON ...")
    scores_data, scores_config, per_view_scores = load_score_json()
    print(f"  Loaded {len(per_view_scores)} score entries")

    print("[2/6] Loading manifest ...")
    manifest_data, manifest_entries = load_manifest()
    print(f"  {len(manifest_entries)} entries in manifest")

    print("[3/6] Loading objects.json ...")
    objects_data = load_objects()
    total_objects = sum(
        len(s["objects"]) for s in objects_data.get("scans", [])
    )
    print(f"  {total_objects} total objects across {len(objects_data.get('scans', []))} scenes")

    print("[4/6] Loading relationships.json ...")
    relationships = load_relationships()

    print("[5/6] Loading embedding cache ...")
    embed_data = load_embeddings()
    print(f"  {len(embed_data.files)} embedding keys in cache")

    # Verify embedding sidecar JSON exists.
    sidecar_path = find_embedding_sidecar_json()
    print(f"  Sidecar JSON: {sidecar_path.name}")

    # 6. Validate and join.
    print("\n[6/7] Validating inputs and building joins ...")
    resolved, scene_object_map, box_hashes = validate_inputs(
        per_view_scores, manifest_entries, objects_data, embed_data, relationships
    )
    print(f"  Validated {len(resolved)} entries")

    # 7. Count relations.
    print("\n[7/8] Counting relation tuples ...")
    rel_counts = compute_relation_tuples(relationships, VALID_SCENES)
    print(f"  {rel_counts['total_tuples']} tuples across {rel_counts['per_scene']}")

    # 8. Build frame map and node views.
    print("\n[8/9] Building frame map ...")
    frame_map = build_frame_map(resolved)
    node_views = build_node_views(resolved)

    # 9. Compute prototypes and rank per query.
    print("\n[9/10] Computing prototypes (leave-one-frame-out) ...")
    start_proto = time.time()
    (
        query_prototypes,
        node_alternate_counts,
        query_support,
        query_candidate_counts,
    ) = compute_prototypes(resolved, frame_map, node_views)
    proto_time = time.time() - start_proto

    # Internal consistency checks.
    print("\n[9.5/10] Running internal consistency checks ...")
    ic_checks: Dict[str, Any] = {
        "query_prototypes_keys_match": True,
        "query_support_keys_match": True,
        "query_candidate_counts_keys_match": True,
        "candidate_count_matches_map_size": True,
        "prototypes_finite_normalized": True,
        "all_candidates_scene_local": True,
        "true_candidate_present_iff_support": True,
        "failures": [],
    }
    total_score_keys = len(resolved)
    resolved_keys = set(resolved.keys())

    # (0) Direct key-set equality across all required per-query maps.
    if set(query_prototypes.keys()) != resolved_keys:
        ic_checks["query_prototypes_keys_match"] = False
        ic_checks["failures"].append(
            "query_prototypes keys differ from resolved keys"
        )
    if set(query_support.keys()) != resolved_keys:
        ic_checks["query_support_keys_match"] = False
        ic_checks["failures"].append(
            "query_support keys differ from resolved keys"
        )
    if set(query_candidate_counts.keys()) != resolved_keys:
        ic_checks["query_candidate_counts_keys_match"] = False
        ic_checks["failures"].append(
            "query_candidate_counts keys differ from resolved keys"
        )

    # Pre-compute per-scene object ID sets for efficient scene-local checks.
    scene_obj_ids: Dict[str, set[int]] = {}
    for _sk, _info in resolved.items():
        _scene = _info["scene"]
        if _scene not in scene_obj_ids:
            scene_obj_ids[_scene] = set()
        scene_obj_ids[_scene].add(_info["object_id_int"])

    for sk in sorted(resolved.keys()):
        info = resolved[sk]
        scene = info["scene"]
        obj_id_int = info["object_id_int"]
        obj_id_str = info["object_id_str"]

        # Check all score keys appear in query_prototypes.
        if sk not in query_prototypes:
            ic_checks["query_prototypes_keys_match"] = False
            ic_checks["failures"].append(f"{sk}: missing from query_prototypes")
            continue

        qprotos = query_prototypes[sk]

        # Verify every prototype is finite and L2-normalized.
        for oid, proto in qprotos.items():
            if not np.all(np.isfinite(proto)):
                ic_checks["prototypes_finite_normalized"] = False
                ic_checks["failures"].append(
                    f"{sk} object {oid}: non-finite prototype"
                )
            norm = float(np.linalg.norm(proto))
            if abs(norm - 1.0) > 1e-6:
                ic_checks["prototypes_finite_normalized"] = False
                ic_checks["failures"].append(
                    f"{sk} object {oid}: prototype norm {norm:.8f}"
                )

        # (a) Candidate count (scalar) must equal prototype map size.
        cand_count = len(qprotos)
        reported_count = query_candidate_counts.get(sk, -1)
        if reported_count != cand_count:
            ic_checks["candidate_count_matches_map_size"] = False
            ic_checks["failures"].append(
                f"{sk}: reported count={reported_count} != map size={cand_count}"
            )

        # (b) Verify per-query candidates are unique IDs.
        cand_ids = list(qprotos.keys())
        if len(cand_ids) != len(set(cand_ids)):
            ic_checks["all_candidates_scene_local"] = False
            ic_checks["failures"].append(f"{sk}: duplicate candidate IDs")

        # (c) Verify per-query candidates are scene-local.
        scene_ids = scene_obj_ids.get(scene, set())
        for oid in qprotos:
            if oid not in scene_ids:
                ic_checks["all_candidates_scene_local"] = False
                ic_checks["failures"].append(
                    f"{sk}: candidate object {oid} not in scene {scene}"
                )

        # (d) True candidate present iff it has alternate support.
        true_support = query_support.get(sk, {}).get(obj_id_str, False)
        true_in_map = obj_id_int in qprotos
        if true_support != true_in_map:
            ic_checks["true_candidate_present_iff_support"] = False
            ic_checks["failures"].append(
                f"{sk}: support={true_support} but in_map={true_in_map}"
            )

    # Fail-closed: abort before metrics/output if any check failed.
    failed_bools = [
        v for v in ic_checks.values()
        if isinstance(v, bool) and not v
    ]
    if failed_bools:
        raise RuntimeError(
            f"Internal consistency check failed ({len(failed_bools)} flag(s)). "
            f"Failures: {ic_checks['failures']}"
        )

    print(f"  All {sum(1 for v in ic_checks.values() if isinstance(v, bool) and v)} "
          f"internal consistency checks passed.")

    print("\n[10/10] Ranking candidates ...")
    start_rank = time.time()

    per_query_ranks: List[Dict] = []
    eligible_scores_list: List[Tuple[str, int, int]] = []  # (scene, obj_id, rank)
    all_scores_list: List[Tuple[str, int, int]] = []  # coverage-adjusted
    eligible_query_count = 0
    eligible_excluded_count = 0
    node_support_dist: Dict[str, Dict] = defaultdict(dict)
    candidate_count_dist: Dict[str, int] = defaultdict(int)
    node_view_support: Dict[str, Dict[str, int]] = defaultdict(dict)

    for sk in sorted(resolved.keys()):
        info = resolved[sk]
        scene = info["scene"]
        obj_id_str = info["object_id_str"]
        obj_id_int = info["object_id_int"]

        # Per-query candidate count from corrected prototype map.
        q_cands = query_prototypes.get(sk, {})
        per_query_cand_count = len(q_cands)

        # Eligibility: true node has alternate support AND >= 2 candidates.
        node_support = query_support.get(sk, {}).get(obj_id_str, False)

        # Coverage-adjusted metrics: rank for ALL 2,204 queries.
        if node_support and per_query_cand_count >= 2:
            # Eligible: compute prototypes and rank.
            eligible_query_count += 1
            scores = rank_candidates(
                sk, info["embedding"], scene, obj_id_int, query_prototypes
            )
            candidate_count_dist[len(scores)] += 1

            # Find rank of correct node.
            correct_rank = None
            for r, (cid, _) in enumerate(scores, 1):
                if cid == obj_id_int:
                    correct_rank = r
                    break
            if correct_rank is None:
                correct_rank = len(scores) + 1

            # Primary metrics.
            eligible_scores_list.append((scene, obj_id_int, correct_rank))

            # Coverage-adjusted metrics (same as primary for eligible).
            all_scores_list.append((scene, obj_id_int, correct_rank))

            per_query_ranks.append({
                "query": sk,
                "scene": scene,
                "object_id": obj_id_int,
                "category": info["category"],
                "rank": correct_rank,
                "eligible": True,
                "excluded_reason": None,
                "num_candidates": len(scores),
                "per_query_cand_count": per_query_cand_count,
                "top5": [{"object_id": cid, "score": round(cos, 6)}
                         for cid, cos in scores[:5]],
                "top10": [{"object_id": cid, "score": round(cos, 6)}
                          for cid, cos in scores[:10]],
            })

        else:
            # Ineligible.
            eligible_excluded_count += 1
            if not node_support:
                reason = "identity_unavailable"
            elif per_query_cand_count < 2:
                reason = "insufficient_candidates"
            else:
                reason = "other"

            # Coverage-adjusted: miss sentinel rank 0 so the metric
            # aggregation contributes zero to top-1, MRR, and every
            # Recall@k while still counting the view in the denominator.
            all_scores_list.append((scene, obj_id_int, 0))

            per_query_ranks.append({
                "query": sk,
                "scene": scene,
                "object_id": obj_id_int,
                "category": info["category"],
                "rank": None,
                "matched": False,
                "eligible": False,
                "excluded_reason": reason,
                "num_candidates": per_query_cand_count,
                "per_query_cand_count": per_query_cand_count,
                "top5": [],
                "top10": [],
            })

        node_support_dist[scene][obj_id_str] = node_alternate_counts.get(scene, {}).get(obj_id_str, 0)
        node_view_support[scene][obj_id_str] = node_alternate_counts.get(scene, {}).get(obj_id_str, 0)

    rank_time = time.time() - start_rank
    total_queries = len(per_query_ranks)

    # 10. Compute metrics.
    print("\nComputing metrics ...")
    primary_metrics = compute_metrics_from_scores(eligible_scores_list)
    coverage_metrics = compute_metrics_from_scores(all_scores_list)

    elapsed = time.time() - t0

    # Sample peak RSS after analysis.
    peak_rss_kb = 0
    try:
        import resource
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        pass

    # Build result JSON.
    print("\nBuilding result JSON ...")
    utc_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = DATA_DIR / f"dinotxt_test5_identity_{utc_now}_{os.getpid()}.json"

    # Input hashes.
    input_hashes: Dict[str, str] = {}
    for p_name, p_path in [
        ("scores", SCORES_JSON),
        ("manifest", MANIFEST_JSON),
        ("objects", OBJECTS_JSON),
        ("relationships", RELATIONSHIPS_JSON),
        ("embeddings_npz", EMBEDDINGS_NPZ),
        ("embeddings_sidecar_json", sidecar_path),
    ]:
        input_hashes[p_name] = sha256_of(pathlib.Path(p_path))

    # Scene view counts.
    scene_view_counts: Dict[str, int] = {}
    for scene in sorted(VALID_SCENES):
        scene_view_counts[scene] = sum(
            len(fv) for fv in frame_map.get(scene, {}).values()
        )

    # Candidate-count distribution.
    cand_dist: Dict[str, int] = defaultdict(int)
    for c in candidate_count_dist:
        cand_dist[str(c)] = candidate_count_dist[c]

    # Node/view support distribution (scene -> {obj_id_str: count}).
    nvs_dist: Dict[str, Dict] = {}
    for s in sorted(VALID_SCENES):
        nvs_dist[s] = dict(node_support_dist.get(s, {}))

    result = {
        "schema_version": 1,
        "created_at_utc": utc_now,
        "status": "completed",
        "benchmark": "DINOv3 + dino.txt cross-view graph-node identity on the frozen ReplicaSSG test-5 split",
        "protocol": "scripts/pipeline_model_reports/reports/optional/dinotxt_clip_substitution.md",
        "runner_file": "scripts/pipeline_model_reports/benchmarks/dinotxt_replica_test5_identity.py",
        "config": {
            "embeddings_npz": str(EMBEDDINGS_NPZ),
            "manifest": str(MANIFEST_JSON),
            "objects_json": str(OBJECTS_JSON),
            "relationships_json": str(RELATIONSHIPS_JSON),
            "scores_json": str(SCORES_JSON),
            "test_scenes": sorted(VALID_SCENES),
            "embed_dim": 2048,
            "analysis_type": "leave_one_frame_out_identity",
        },
        "input_hashes": input_hashes,
        "join_counts": {
            "total_validated": len(resolved),
            "expected": 2204,
            "scene_view_counts": scene_view_counts,
            "unique_category_count": 49,
            "unique_box_hashes": len(box_hashes),
            "unique_scene_object_ids": len({
                (info["scene"], info["object_id_int"])
                for info in resolved.values()
            }),
            "scenes_with_2plus_views": sum(
                1 for scene, obj_ids in node_views.items()
                if scene in VALID_SCENES and len(obj_ids) >= 2
            ),
            "scenes_with_3plus_views": sum(
                1 for scene, obj_ids in node_views.items()
                if scene in VALID_SCENES and len(obj_ids) >= 3
            ),
            "scenes_with_5plus_views": sum(
                1 for scene, obj_ids in node_views.items()
                if scene in VALID_SCENES and len(obj_ids) >= 5
            ),
        },
        "relation_counts": {
            "total_tuples": rel_counts["total_tuples"],
            "per_scene": rel_counts["per_scene"],
            "unique_relation_names": rel_counts["unique_relation_names"],
            "note": "Exact frozen-scene relation tuple count from relationships.json. Visual relation-triplet recall is unmeasured: cached inputs contain only individual crop embeddings, no relation-pair features or relation-text score matrix.",
        },
        "analysis_status": {
            "identity_retrieval": "completed",
            "relation_triplet_recall": "unmeasured",
        },
        "runtime": {
            "wall_seconds": round(elapsed, 3),
            "prototype_computation_seconds": round(proto_time, 3),
            "ranking_seconds": round(rank_time, 3),
        },
        "memory": {
            "peak_rss_kb": peak_rss_kb,
            "unit": "Linux ru_maxrss (KB)",
        },
        "output": {
            "path": str(result_path),
            "size_bytes": 0,
        },
        "metrics": {
            "overall": primary_metrics["overall"],
            "per_scene": primary_metrics["per_scene"],
        },
        "candidate_count_distribution": dict(sorted(cand_dist.items())),
        "node_view_support_distribution": nvs_dist,
        "coverage": {
            "total_validated_views": total_queries,
            "eligible_query_count": eligible_query_count,
            "excluded_count": eligible_excluded_count,
            "excluded_reasons": {
                "identity_unavailable": sum(
                    1 for q in per_query_ranks
                    if q["excluded_reason"] == "identity_unavailable"
                ),
                "insufficient_candidates": sum(
                    1 for q in per_query_ranks
                    if q["excluded_reason"] == "insufficient_candidates"
                ),
            },
        },
        "metrics_primary": {
            "description": "Top-1, MRR, Recall@1/5/10 over eligible queries only (supported true node + >= 2 candidates).",
            "eligible_query_count": eligible_query_count,
            "overall": primary_metrics["overall"],
            "per_scene": primary_metrics["per_scene"],
        },
        "metrics_coverage_adjusted": {
            "description": "Top-1, MRR, Recall@1/5/10 over all 2,204 views. Excluded queries counted as misses.",
            "total_views": total_queries,
            "overall": coverage_metrics["overall"],
            "per_scene": coverage_metrics["per_scene"],
        },
        "internal_consistency": ic_checks,
        "per_query_ranks": per_query_ranks,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pid": os.getpid(),
        },
        "limitations": [
            "Cached-feature CPU analysis only. No model load, GPU operation, or new input generation.",
            "Leave-one-frame-out retrieval of graph-node identity - not map association or pipeline-slot acceptance.",
            "Relation-triplet recall is unmeasured from these cached embeddings.",
        ],
        "failures": [],
        "warnings": [],
    }

    # Remove size_bytes from result before serialisation to avoid the
    # self-referential fixed-point chicken-and-egg problem.
    # The actual file size is attached after the atomic write completes.
    result["output"].pop("size_bytes", None)

    # Write result.
    print(f"\nWriting result to {result_path} ...")
    try:
        result_path_parent = result_path.parent
        result_path_parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(result_path, result)
        actual_size = result_path.stat().st_size
        result["output"]["size_bytes"] = actual_size
        print(f"  Written {actual_size:,} bytes")
    except FileExistsError as e:
        print(f"  SKIP: {e}")

    print(f"\nDone in {elapsed:.2f}s.  {total_queries} queries ranked.")
    print(f"  Eligible queries: {eligible_query_count}")
    print(f"  Excluded queries: {eligible_excluded_count}")
    print(f"  Primary overall MRR: {primary_metrics['overall']['mrr']:.6f}")
    print(f"  Primary overall top-1: {primary_metrics['overall']['top1_accuracy']:.6f}")
    print(f"  Primary overall R@1: {primary_metrics['overall']['recall_at_1']:.6f}")
    print(f"  Primary overall R@5: {primary_metrics['overall']['recall_at_5']:.6f}")
    print(f"  Primary overall R@10: {primary_metrics['overall']['recall_at_10']:.6f}")
    print(f"  Coverage-adjusted MRR: {coverage_metrics['overall']['mrr']:.6f}")
    print(f"  Coverage-adjusted top-1: {coverage_metrics['overall']['top1_accuracy']:.6f}")
    print(f"  Peak RSS: {peak_rss_kb:,} KB")


if __name__ == "__main__":
    main()
