"""Run the mapping overlap adapter through CUDA and the installed GPU Faiss."""

import uuid

import faiss
import numpy as np
import open3d as o3d
import pytest
import torch

from conceptgraph.slam import mapping, utils
from conceptgraph.slam.slam_classes import DetectionList, MapObjectList


def _require_gpu_faiss():
    assert torch.cuda.is_available(), "mapping overlap requires CUDA"
    assert faiss.get_num_gpus() > 0, "mapping overlap requires GPU Faiss"


def _object(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    box = o3d.geometry.AxisAlignedBoundingBox(
        np.asarray([-1.0, -1.0, -1.0]), np.asarray([10.0, 2.0, 2.0])
    )
    return {"id": uuid.uuid4(), "pcd": cloud, "bbox": box}


def test_mapping_overlap_adapter_returns_real_cuda_detection_by_object_matrix():
    _require_gpu_faiss()
    utils.clear_gpu_faiss_index_cache()
    objects = MapObjectList([_object([[0, 0, 0], [2, 0, 0]])], device="cuda:0")
    detections = DetectionList(
        [_object([[0.01, 0, 0], [2.01, 0, 0], [9, 0, 0]])],
        device="cuda:0",
    )

    result = mapping.compute_spatial_similarities(
        "overlap", detections, objects, 0.05
    )

    assert result.device == torch.device("cuda:0")
    assert tuple(result.shape) == (1, 1)
    assert result.cpu().item() == pytest.approx(2 / 3)
    utils.clear_gpu_faiss_index_cache()
    torch.cuda.synchronize()
