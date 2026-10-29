"""Compatibility tests for detection records without optional VLM captions."""

import numpy as np

from conceptgraph.slam.utils import filter_gobs


def test_filter_gobs_accepts_detection_records_without_captions():
    gobs = {
        "xyxy": np.array([[0, 0, 4, 4]], dtype=np.float32),
        "confidence": np.array([0.9], dtype=np.float32),
        "class_id": np.array([0], dtype=np.int64),
        "mask": np.ones((1, 4, 4), dtype=bool),
        "classes": ["chair"],
        "detection_class_labels": ["chair 0"],
    }

    filtered = filter_gobs(
        gobs,
        np.zeros((4, 4, 3), dtype=np.uint8),
        skip_bg=False,
        BG_CLASSES=[],
        mask_area_threshold=1,
    )

    assert filtered["captions"] == [{"id": "0", "name": "chair", "caption": None}]
