"""Real-GPU acceptance for cached 2D relation propagation into a room0 map.

This integration test intentionally runs the repository mapper against the
saved Replica RGB-D sequence and detection caches. It does not mock the
association or map-edge code and has no CPU fallback.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import pickle
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET_ROOT = Path(
    os.environ.get(
        "CONCEPTGRAPHS_REPLICA_ROOT",
        "/mnt/HDD1/Project_Data/public_datasets/Replica",
    )
)
SCENE = "room0"
STRIDE = 50
DETECTION_SUFFIX = "s_modernized_yoloe26l_stride50"
MAPPING_SUFFIX = "r_room0_relation_propagation_acceptance"
EXPECTED_CACHE_COUNT = 40
EXPECTED_2D_RELATION_COUNT = 199


def _read_cached_relations(cache_paths: list[Path]) -> tuple[int, set[str]]:
    count = 0
    relation_types: set[str] = set()
    for path in cache_paths:
        with gzip.open(path, "rb") as stream:
            detections = pickle.load(stream)
        edges = detections.get("edges") or []
        count += len(edges)
        relation_types.update(
            str(edge[1]) for edge in edges if isinstance(edge, (list, tuple)) and len(edge) == 3
        )
    return count, relation_types


def _stage_dataset_view(workspace: Path, source_scene: Path) -> Path:
    """Link immutable sequence inputs while keeping mapping outputs writable."""
    staged_root = workspace / "data" / "Replica"
    staged_scene = staged_root / SCENE
    staged_experiments = staged_scene / "exps"
    source_experiments = source_scene / "exps"
    source_detection_experiment = source_experiments / DETECTION_SUFFIX

    if workspace.exists():
        pytest.fail(
            f"Refusing to overwrite existing relation acceptance workspace: {workspace}"
        )
    if not source_scene.is_dir():
        pytest.skip(f"Replica scene data is unavailable: {source_scene}")
    if not source_detection_experiment.is_dir():
        pytest.skip(
            "The saved modernized detection experiment is unavailable: "
            f"{source_detection_experiment}"
        )

    staged_experiments.mkdir(parents=True)
    for child in source_scene.iterdir():
        if child.name == "exps":
            continue
        (staged_scene / child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())
    (staged_experiments / DETECTION_SUFFIX).symlink_to(
        source_detection_experiment.resolve(), target_is_directory=True
    )
    return staged_root


def test_room0_cached_relations_survive_real_rgbd_mapping() -> None:
    """The 40-frame cache must produce valid 3D relations on the real GPU."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU acceptance unresolved: CUDA is unavailable; no CPU substitute is allowed")

    source_scene = SOURCE_DATASET_ROOT / SCENE
    source_detection_dir = (
        source_scene / "exps" / DETECTION_SUFFIX / "detections"
    )
    cache_paths = sorted(source_detection_dir.glob("frame*.pkl.gz"))
    if not source_scene.is_dir() or not source_detection_dir.is_dir():
        pytest.skip("GPU acceptance unresolved: Replica room0 inputs or caches are unavailable")
    if len(cache_paths) != EXPECTED_CACHE_COUNT:
        pytest.fail(
            "Expected exactly 40 cached sampled frames, found "
            f"{len(cache_paths)} under {source_detection_dir}"
        )
    relation_count, input_relation_types = _read_cached_relations(cache_paths)
    if relation_count != EXPECTED_2D_RELATION_COUNT:
        pytest.fail(
            "Expected 199 cached 2D relation tuples, found "
            f"{relation_count} in {source_detection_dir}"
        )

    workspace = REPO_ROOT / ".agentic_workflow" / "workspace" / "room0-relation-propagation-test"
    staged_dataset_root = _stage_dataset_view(workspace, source_scene)
    mapping_output = (
        staged_dataset_root / SCENE / "exps" / MAPPING_SUFFIX
    )
    if mapping_output.exists():
        pytest.fail(f"Refusing to overwrite existing map output: {mapping_output}")

    mapper_log = workspace / "mapper.log"
    dataset_config = (
        REPO_ROOT / "conceptgraph" / "dataset" / "dataconfigs" / "replica" / "replica.yaml"
    )
    python = REPO_ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        pytest.skip(f"GPU acceptance unresolved: project Python environment missing: {python}")
    if not dataset_config.is_file():
        pytest.skip(f"GPU acceptance unresolved: Replica dataset config missing: {dataset_config}")

    command = [
        str(python),
        "-m",
        "conceptgraph.slam.rerun_realtime_mapping",
        f"dataset_root={staged_dataset_root}",
        f"dataset_config={dataset_config}",
        f"scene_id={SCENE}",
        f"stride={STRIDE}",
        f"detections_exp_suffix={DETECTION_SUFFIX}",
        f"exp_suffix={MAPPING_SUFFIX}",
        "device=cuda",
        "make_edges=false",
        "use_rerun=false",
        "save_rerun=false",
        "consolidate_captions=false",
        "save_pcd=false",
        "save_json=true",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as exc:
        mapper_log.write_text(exc.stdout or "", encoding="utf-8")
        pytest.fail(f"GPU mapper did not finish within one hour; log: {mapper_log}")
    mapper_log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        pytest.fail(
            f"Real CUDA mapper failed with exit code {completed.returncode}; "
            f"log: {mapper_log}"
        )

    edge_path = mapping_output / f"edge_json_{MAPPING_SUFFIX}.json"
    object_path = mapping_output / f"obj_json_{MAPPING_SUFFIX}.json"
    if not edge_path.is_file() or not object_path.is_file():
        pytest.fail(
            "The mapper completed without both expected JSON outputs: "
            f"{edge_path}, {object_path}; log: {mapper_log}"
        )

    edges = json.loads(edge_path.read_text(encoding="utf-8"))
    objects = json.loads(object_path.read_text(encoding="utf-8"))
    object_tags = {
        record.get("id"): record.get("object_tag")
        for record in objects.values()
        if isinstance(record, dict)
    }
    if not edges:
        pytest.fail(
            "All 199 cached 2D relation tuples were lost before final 3D map "
            f"serialization; map output: {mapping_output}; log: {mapper_log}"
        )
    if len(edges) > relation_count:
        pytest.fail(
            f"Final 3D edge count {len(edges)} exceeds {relation_count} cached "
            "2D relation observations"
        )

    invalid_endpoints = []
    unknown_relation_types = []
    for edge_key, edge in edges.items():
        relation_type = edge.get("relationship")
        if relation_type not in input_relation_types:
            unknown_relation_types.append((edge_key, relation_type))
        for endpoint_id_key, endpoint_tag_key in (
            ("object_1_id", "object_1_tag"),
            ("object_2_id", "object_2_tag"),
        ):
            endpoint_id = edge.get(endpoint_id_key)
            if endpoint_id not in object_tags:
                invalid_endpoints.append((edge_key, endpoint_id_key, endpoint_id))
            elif object_tags[endpoint_id] != edge.get(endpoint_tag_key):
                invalid_endpoints.append(
                    (
                        edge_key,
                        endpoint_tag_key,
                        edge.get(endpoint_tag_key),
                        object_tags[endpoint_id],
                    )
                )
    if invalid_endpoints:
        pytest.fail(
            "Final relation endpoints do not resolve to matching saved object "
            f"IDs and tags: {invalid_endpoints[:10]}"
        )
    if unknown_relation_types:
        pytest.fail(
            "Final map contains relationship types absent from the cached 2D "
            f"inputs: {unknown_relation_types[:10]}"
        )

    print(
        "room0 relation propagation: "
        f"{relation_count} cached 2D tuples -> {len(edges)} valid 3D map edges; "
        f"output={mapping_output}; log={mapper_log}"
    )
