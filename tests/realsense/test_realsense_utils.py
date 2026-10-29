"""Focused contract tests for the public pure RealSense helper functions."""

import numpy as np
import pytest

from conceptgraph.utils.realsense_utils import (
    depth_to_meters,
    is_valid_pose,
    source_to_target_to_camera_to_world,
)


def test_depth_to_meters_scales_nonzero_depth_and_preserves_invalid_zero():
    raw = np.array([[0, 1, 250], [1000, 0, 42]], dtype=np.uint16)

    meters = depth_to_meters(raw, 0.001)

    assert meters.dtype == np.float32
    assert meters.flags.c_contiguous
    np.testing.assert_allclose(
        meters, [[0.0, 0.001, 0.25], [1.0, 0.0, 0.042]], rtol=1e-6, atol=0.0
    )


@pytest.mark.parametrize(
    "depth",
    [np.array(1, dtype=np.uint16), np.zeros((1, 1, 1), dtype=np.uint16)],
)
def test_depth_to_meters_rejects_non_2d_depth(depth):
    with pytest.raises(ValueError):
        depth_to_meters(depth, 0.001)


@pytest.mark.parametrize("scale", [0.0, -0.001, np.inf, -np.inf, np.nan])
def test_depth_to_meters_rejects_non_positive_or_non_finite_scale(scale):
    with pytest.raises(ValueError):
        depth_to_meters(np.ones((2, 2), dtype=np.uint16), scale)


def test_source_to_target_is_inverted_when_composing_camera_to_world():
    previous_camera_to_world = np.eye(4)
    previous_camera_to_world[:3, 3] = [3.0, -2.0, 0.5]
    source_to_target = np.eye(4)
    source_to_target[:3, 3] = [1.0, 2.0, -4.0]

    result = source_to_target_to_camera_to_world(
        previous_camera_to_world, source_to_target
    )

    expected = previous_camera_to_world @ np.linalg.inv(source_to_target)
    np.testing.assert_allclose(result, expected)
    assert result.shape == (4, 4)


@pytest.mark.parametrize(
    "previous, motion",
    [
        (np.eye(3), np.eye(4)),
        (np.eye(4), np.eye(3)),
        (np.full((4, 4), np.nan), np.eye(4)),
        (np.eye(4), np.full((4, 4), np.inf)),
    ],
)
def test_source_to_target_rejects_invalid_matrices(previous, motion):
    with pytest.raises(ValueError):
        source_to_target_to_camera_to_world(previous, motion)


def test_source_to_target_rejects_singular_motion_transform():
    singular = np.eye(4)
    singular[2, 2] = 0.0

    with pytest.raises(ValueError):
        source_to_target_to_camera_to_world(np.eye(4), singular)


def test_is_valid_pose_accepts_identity_and_rigid_transform():
    pose = np.eye(4)
    angle = np.deg2rad(30.0)
    pose[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    pose[:3, 3] = [1.0, -3.0, 2.5]

    assert is_valid_pose(np.eye(4))
    assert is_valid_pose(pose)


@pytest.mark.parametrize(
    "pose",
    [
        np.eye(3),
        np.full((4, 4), np.nan),
        np.diag([2.0, 1.0, 1.0, 1.0]),
        np.diag([-1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_is_valid_pose_rejects_malformed_or_non_rigid_pose(pose):
    assert not is_valid_pose(pose)


def test_is_valid_pose_rejects_invalid_homogeneous_last_row():
    pose = np.eye(4)
    pose[3, 0] = 0.01

    assert not is_valid_pose(pose)
