"""Regression checks for the camera basis used by Replica trajectory poses."""

import numpy as np
import pytest

from conceptgraph.dataset.datasets_common import ReplicaDataset


def test_replica_load_poses_applies_camera_yz_basis_flip_once(tmp_path):
    angle = 0.37
    cosine, sine = np.cos(angle), np.sin(angle)
    source_c2w = np.array(
        [
            [cosine, -sine, 0.0, 1.25],
            [sine, cosine, 0.0, -2.5],
            [0.0, 0.0, 1.0, 3.75],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pose_path = tmp_path / "traj.txt"
    pose_path.write_text(" ".join(map(str, source_c2w.ravel())) + "\n")

    dataset = ReplicaDataset.__new__(ReplicaDataset)
    dataset.pose_path = str(pose_path)
    dataset.num_imgs = 1
    loaded = dataset.load_poses()[0].cpu().numpy()

    camera_basis_flip = np.diag([1.0, -1.0, -1.0, 1.0])
    expected = source_c2w @ camera_basis_flip
    np.testing.assert_allclose(loaded, expected, atol=1e-6)
    np.testing.assert_allclose(loaded[:3, 3], source_c2w[:3, 3], atol=1e-6)
    np.testing.assert_allclose(loaded[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(loaded[:3, :3].T @ loaded[:3, :3], np.eye(3), atol=1e-6)
    assert np.linalg.det(loaded[:3, :3]) == pytest.approx(1.0, abs=1e-6)
