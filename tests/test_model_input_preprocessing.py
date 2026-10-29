"""Coordinate handling for model-specific resized and letterboxed inputs."""

import numpy as np

from conceptgraph.utils.model_utils import (
    boxes_from_letterbox,
    boxes_to_letterbox,
    masks_from_letterbox,
    resize_and_letterbox,
)


def test_letterbox_preserves_aspect_and_box_round_trip_for_hd_frame():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas, transform = resize_and_letterbox(image, 640)

    assert canvas.shape == (640, 640, 3)
    assert transform["resized_shape"] == (360, 640)
    assert transform["top"] == 140

    source_boxes = np.array([[100, 80, 400, 300], [0, 0, 1280, 720]], dtype=np.float32)
    restored = boxes_from_letterbox(boxes_to_letterbox(source_boxes, transform), transform)
    np.testing.assert_allclose(restored, source_boxes)


def test_mask_restore_removes_letterbox_padding_and_restores_source_size():
    source = np.zeros((360, 640), dtype=np.uint8)
    source[40:160, 100:300] = 1
    canvas, transform = resize_and_letterbox(source, 640)
    restored = masks_from_letterbox(canvas[None], transform)

    assert restored.shape == (1, 360, 640)
    np.testing.assert_array_equal(restored[0], source)


def test_empty_boxes_keep_xyxy_shape():
    _, transform = resize_and_letterbox(np.zeros((8, 12, 3), dtype=np.uint8), 16)
    assert boxes_from_letterbox(np.empty((0, 4)), transform).shape == (0, 4)
    assert boxes_to_letterbox(np.empty((0, 4)), transform).shape == (0, 4)
