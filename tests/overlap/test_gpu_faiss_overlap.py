"""Overlap integration tests against real Open3D, CUDA and GPU Faiss."""

import uuid

import faiss
import numpy as np
import open3d as o3d
import pytest
import torch

from conceptgraph.slam import utils
from conceptgraph.slam.slam_classes import DetectionList, MapObjectList


def _require_gpu_faiss():
    assert torch.cuda.is_available(), "these overlap tests require CUDA"
    assert faiss.get_num_gpus() > 0, "these overlap tests require GPU Faiss"


def _object(points, bounds=([-1, -1, -1], [10, 2, 2])):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    box = o3d.geometry.AxisAlignedBoundingBox(*map(np.asarray, bounds))
    return {"id": uuid.uuid4(), "pcd": cloud, "bbox": box}


def _map_objects(*point_sets):
    return MapObjectList([_object(points) for points in point_sets], device="cuda:0")


def _detections(*point_sets):
    return DetectionList([_object(points) for points in point_sets], device="cuda:0")


def test_single_set_overlap_runs_gpu_search_and_reuses_unchanged_indexes():
    _require_gpu_faiss()
    utils.clear_gpu_faiss_index_cache()
    objects = [
        _object([[0, 0, 0], [1, 0, 0]]),
        _object([[0.05, 0, 0], [1.05, 0, 0], [4, 0, 0]]),
    ]

    first = utils.compute_overlap_matrix(objects, 0.1)
    cache = utils._gpu_faiss_index_cache
    first_indexes = [entry[0] for entry in cache._entries.values()]
    second = utils.compute_overlap_matrix(objects, 0.1)
    second_indexes = [entry[0] for entry in cache._entries.values()]

    assert first[0, 1] == pytest.approx(1.0)
    assert first[1, 0] == pytest.approx(2 / 3)
    np.testing.assert_allclose(second, first)
    assert len(first_indexes) == 2
    assert all(a is b for a, b in zip(first_indexes, second_indexes))
    assert cache.resident_bytes == sum(
        np.asarray(item["pcd"].points, dtype=np.float32).nbytes for item in objects
    )

    del first_indexes, second_indexes, cache
    utils.clear_gpu_faiss_index_cache()
    torch.cuda.synchronize()


def test_two_set_overlap_uses_real_gpu_faiss_with_query_normalization():
    _require_gpu_faiss()
    utils.clear_gpu_faiss_index_cache()
    mapped = _map_objects([[0, 0, 0], [2, 0, 0]])
    new = _detections([[0.01, 0, 0], [2.01, 0, 0], [9, 0, 0]])

    result = utils.compute_overlap_matrix_2set(mapped, new, 0.05)

    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(2 / 3)
    utils.clear_gpu_faiss_index_cache()
    torch.cuda.synchronize()


def test_general_overlap_returns_cuda_tensor_and_preserves_object_orientation():
    _require_gpu_faiss()
    utils.clear_gpu_faiss_index_cache()
    objects = _map_objects([[0, 0, 0], [2, 0, 0]])
    detections = _detections([[0.01, 0, 0], [2.01, 0, 0], [9, 0, 0]])

    result = utils.compute_overlap_matrix_general_torch(objects, detections, 0.05)

    assert result.device == torch.device("cuda:0")
    assert tuple(result.shape) == (1, 1)
    assert result.cpu().item() == pytest.approx(2 / 3)
    utils.clear_gpu_faiss_index_cache()
    torch.cuda.synchronize()
