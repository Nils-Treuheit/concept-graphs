"""GPU-resident cache tests using the installed CUDA Faiss implementation."""

import numpy as np
import pytest
import torch
import faiss

from conceptgraph.slam import utils


def _require_gpu_faiss():
    assert torch.cuda.is_available(), "these Faiss tests require CUDA"
    assert faiss.get_num_gpus() > 0, "these Faiss tests require GPU Faiss"


def _make_gpu_index(points, resource):
    tensor = utils._point_arrays_to_cuda([points])[0]
    index = utils._make_gpu_flat_l2_index(3, resource)
    index.add(tensor)
    return index


def test_real_gpu_indexes_are_reused_and_geometry_changes_replace_entries():
    _require_gpu_faiss()
    utils.clear_gpu_faiss_index_cache()
    resource = utils._create_gpu_resources()
    cache = utils.GpuFaissIndexCache(
        max_bytes=4096,
        index_factory=lambda points: _make_gpu_index(points, resource),
    )
    points = np.arange(12, dtype=np.float32).reshape(4, 3)

    first = cache.get(points, cache_token="map-object-1")
    repeated = cache.get(points.copy(), cache_token="map-object-1")
    changed = cache.get(points + 0.5, cache_token="map-object-1")

    assert repeated is first
    assert changed is not first
    assert cache.resident_bytes == points.nbytes
    query = torch.from_numpy(points[:1].copy()).to("cuda:0")
    distances, _ = changed.search(query, 1)
    assert distances.device == torch.device("cuda:0")
    assert torch.isfinite(distances).all().item()

    del query, first, repeated, changed, cache, resource
    torch.cuda.synchronize()
    utils.clear_gpu_faiss_index_cache()


def test_real_gpu_cache_rejects_active_index_payload_over_limit_before_upload():
    _require_gpu_faiss()
    utils.clear_gpu_faiss_index_cache()
    resource = utils._create_gpu_resources()
    cache = utils.GpuFaissIndexCache(
        max_bytes=12,
        index_factory=lambda points: _make_gpu_index(points, resource),
    )
    points = np.zeros((2, 3), dtype=np.float32)
    smoke_index = _make_gpu_index(points[:1], resource)
    smoke_query = torch.zeros((1, 3), dtype=torch.float32, device="cuda:0")
    smoke_index.search(smoke_query, 1)

    with pytest.raises(MemoryError, match="active GPU Faiss point payload"):
        cache.get_many([points], max_active_bytes=points.nbytes - 1)

    assert cache.resident_bytes == 0
    del smoke_query, smoke_index, cache, resource
    torch.cuda.synchronize()
    utils.clear_gpu_faiss_index_cache()
