"""Regression tests for empty results from ``filter_detections``."""

import numpy as np
import supervision as sv

from conceptgraph.utils.general_utils import filter_detections
from conceptgraph.utils.model_utils import compute_clip_features_batched


class _Classes:
    bg_classes = ["background"]

    @staticmethod
    def get_classes_arr():
        return ["background", "object"]


def _empty_detections(height=100, width=100):
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.int64),
        mask=np.empty((0, height, width), dtype=np.bool_),
    )


def _assert_empty_result(detections, labels, height=100, width=100):
    assert isinstance(detections, sv.Detections)
    assert labels == []
    assert detections.class_id.shape == (0,)
    assert detections.confidence.shape == (0,)
    assert detections.xyxy.shape == (0, 4)
    assert detections.mask.shape == (0, height, width)


def test_filter_detections_returns_empty_result_for_no_input_detections():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result, labels = filter_detections(
        image=image,
        detections=_empty_detections(),
        classes=_Classes(),
        given_labels=[],
    )

    _assert_empty_result(result, labels)


def test_filter_detections_returns_empty_result_when_all_candidates_are_background():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((1, 100, 100), dtype=np.bool_)
    mask[0, 10:20, 10:20] = True
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int64),
        mask=mask,
    )

    result, labels = filter_detections(
        image=image,
        detections=detections,
        classes=_Classes(),
        given_labels=["background"],
    )

    _assert_empty_result(result, labels)


def test_compute_clip_features_batched_skips_clip_for_no_detections():
    class _MustNotBeCalled:
        def __getattr__(self, name):
            raise AssertionError(f"CLIP model method {name} must not be called")

        def __call__(self, *args, **kwargs):
            raise AssertionError("CLIP preprocess/tokenizer must not be called")

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    crops, image_features, text_features = compute_clip_features_batched(
        image=image,
        detections=_empty_detections(),
        clip_model=_MustNotBeCalled(),
        clip_preprocess=_MustNotBeCalled(),
        clip_tokenizer=_MustNotBeCalled(),
        classes=[],
        device="cpu",
    )

    assert crops == []
    assert isinstance(image_features, np.ndarray)
    assert image_features.ndim == 2
    assert image_features.shape[0] == 0
    assert image_features.size == 0
    assert text_features == []
