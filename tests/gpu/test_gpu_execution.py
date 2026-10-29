"""Exercise configured mapping tensor operations on the real CUDA device."""

import numpy as np
import torch

from conceptgraph.slam.slam_classes import DetectionList, MapObjectList
from conceptgraph.utils.model_utils import require_cuda_device


def _require_cuda():
    assert torch.cuda.is_available(), "GPU execution tests require CUDA"


def test_cuda_device_validation_and_mapping_features_use_real_cuda():
    _require_cuda()
    device = require_cuda_device(torch.device("cuda:0"))
    assert device == torch.device("cuda:0")

    for list_type in (DetectionList, MapObjectList):
        values = list_type(device="cuda:0")
        values.extend(
            [
                {"feature": np.asarray([1.0, 2.0], dtype=np.float32)},
                {"feature": torch.tensor([3.0, 4.0], dtype=torch.float32)},
            ]
        )
        stacked = values.get_stacked_values_torch("feature")
        assert stacked.device == device
        assert stacked.dtype == torch.float32
        torch.testing.assert_close(
            stacked,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device),
        )
