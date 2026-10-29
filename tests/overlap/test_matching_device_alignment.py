"""Contract tests for similarity aggregation device alignment (D6)."""

import pytest
import torch

from conceptgraph.slam.mapping import aggregate_similarities


def test_sim_sum_preserves_values_when_inputs_share_device():
    assert torch.cuda.is_available(), "matching tests require CUDA"
    spatial = torch.tensor([[0.2, 0.8], [0.4, 0.6]], dtype=torch.float32, device="cuda:0")
    visual = torch.tensor([[0.6, 0.4], [0.2, 0.8]], dtype=torch.float32, device="cuda:0")
    phys_bias = 0.25

    result = aggregate_similarities("sim_sum", phys_bias, spatial, visual)

    expected = (1 + phys_bias) * spatial + (1 - phys_bias) * visual
    assert result.device == spatial.device
    torch.testing.assert_close(result, expected)


def test_cuda_visual_similarity_is_aligned_with_cuda_spatial_similarity():
    assert torch.cuda.is_available(), "matching tests require CUDA"
    device = torch.device("cuda")
    spatial = torch.tensor([[0.2, 0.8], [0.4, 0.6]], device=device)
    visual_cuda = torch.tensor([[0.6, 0.4], [0.2, 0.8]], device=device)
    phys_bias = 0.25

    result = aggregate_similarities("sim_sum", phys_bias, spatial, visual_cuda)

    expected = (1 + phys_bias) * spatial + (1 - phys_bias) * visual_cuda
    assert result.device == spatial.device
    torch.testing.assert_close(result, expected)


def test_unsupported_matching_method_raises_value_error():
    assert torch.cuda.is_available(), "matching tests require CUDA"
    spatial = torch.tensor([[0.5]], device="cuda:0")
    visual = torch.tensor([[0.25]], device="cuda:0")

    with pytest.raises(ValueError, match="Unknown matching method: unsupported"):
        aggregate_similarities("unsupported", 0.5, spatial, visual)
