"""Resolution contract between full-resolution detections and dataset mapping frames."""

import numpy as np
import torch

from conceptgraph.slam.utils import batch_mask_depth_to_points_colors, resize_gobs


def test_full_resolution_sam_masks_and_boxes_align_with_mapping_rgbd_before_projection():
    """SAM runs at 1280x720 while mapping consumes 640x360 RGB-D frames."""
    full_h, full_w = 720, 1280
    map_h, map_w = 360, 640
    # One rectangular SAM mask and its full-resolution XYXY box.
    mask = np.zeros((1, full_h, full_w), dtype=bool)
    mask[0, 20:40, 40:80] = True
    gobs = {"mask": mask, "xyxy": np.array([[40, 20, 80, 40]], dtype=np.float32)}

    rgb = np.zeros((map_h, map_w, 3), dtype=np.uint8)
    rgb[12, 25] = [11, 22, 33]
    depth = np.ones((map_h, map_w), dtype=np.float32)

    resized = resize_gobs(gobs, rgb)
    resized_mask = resized["mask"][0]

    assert resized_mask.shape == depth.shape == rgb.shape[:2] == (map_h, map_w)
    np.testing.assert_array_equal(resized["xyxy"][0], [20, 10, 40, 20])
    assert resized_mask[12, 25]
    assert not resized_mask[12, 41]

    # Exercise the same batched mask/depth projection used before point-cloud
    # creation. The sampled point and color must refer to the same mapping pixel.
    points, colors = batch_mask_depth_to_points_colors(
        depth_tensor=torch.from_numpy(depth),
        masks_tensor=torch.from_numpy(resized_mask[None].astype(np.float32)),
        cam_K=torch.eye(3),
        image_rgb_tensor=torch.from_numpy(rgb.astype(np.float32) / 255.0),
        device="cpu",
    )
    np.testing.assert_allclose(points[0, 12, 25].numpy(), [25, 12, 1])
    np.testing.assert_allclose(colors[0, 12, 25].numpy(), np.array([11, 22, 33]) / 255.0)
