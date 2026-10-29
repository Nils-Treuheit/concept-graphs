"""D4 focused tests for CUDA point transfer and the GPU overlap matrix."""

import numpy as np
import faiss
import torch

from conceptgraph.slam import utils


def _require_gpu_faiss():
    assert torch.cuda.is_available(), "point-transfer tests require CUDA"
    assert faiss.get_num_gpus() > 0, "point-transfer tests require GPU Faiss"


def test_point_arrays_to_cuda_returns_exact_cuda0_views():
    _require_gpu_faiss()

    arrays = [
        np.arange(3, dtype=np.float32).reshape(1, 3),
        (np.arange(9, dtype=np.int64) + 10).astype(np.float32).reshape(3, 3),
        (np.arange(15, dtype=np.int64) + 100).astype(np.float32).reshape(5, 3),
    ]

    tensors = utils._point_arrays_to_cuda(arrays)

    assert len(tensors) == 3
    for source, tensor in zip(arrays, tensors):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == source.shape
        assert tensor.dtype == torch.float32
        assert tensor.device == torch.device("cuda:0")
        assert tensor.is_contiguous()
        assert torch.equal(tensor, torch.from_numpy(source).cuda())
