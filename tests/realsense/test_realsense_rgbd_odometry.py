"""Run Open3D's real CUDA RGB-D odometry on a small textured sequence."""

import numpy as np
import open3d as o3d

from conceptgraph.utils.realsense_utils import RGBDOdometry


def test_rgbd_odometry_uses_cuda_and_processes_a_real_rgbd_sequence():
    assert o3d.core.cuda.is_available(), "RGB-D odometry tests require Open3D CUDA"
    odometry = RGBDOdometry(
        intrinsics={"fx": 80.0, "fy": 80.0, "cx": 31.5, "cy": 31.5},
        device="CUDA:0",
        min_fitness=0.0,
    )
    assert str(odometry.device).upper() == "CUDA:0"

    y, x = np.mgrid[:64, :64]
    rgb = np.stack(
        ((x * 3) % 255, (y * 5) % 255, ((x + y) * 2) % 255), axis=-1
    ).astype(np.uint8)
    depth = (1.5 + 0.001 * x + 0.001 * y).astype(np.float32)

    initial_pose = odometry.estimate(rgb, depth)
    next_pose = odometry.estimate(rgb.copy(), depth.copy())

    np.testing.assert_allclose(initial_pose, np.eye(4), atol=1e-8)
    assert next_pose.shape == (4, 4)
    assert np.isfinite(next_pose).all()
    assert odometry._previous_rgbd is not None
