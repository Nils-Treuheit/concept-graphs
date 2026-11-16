"""Real-GPU acceptance for the YOLOE-26-L modernized detector path.

Configure the local frame, detector checkpoint, both encoder checkpoints, and
Hydra config with the ``CG_MODERNIZED_TEST_*`` environment variables below.
This test intentionally runs real inference; it has no mocked model path.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf




def _required_local_path(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        pytest.skip(f"GPU acceptance unresolved: set {env_name} to a local path")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        pytest.skip(f"GPU acceptance unresolved: {env_name} does not exist: {path}")
    return path


def _required_output_path(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        pytest.skip(f"GPU acceptance unresolved: set {env_name} to a local path")
    return Path(value).expanduser().resolve()


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _assert_normalized_features(value, *, rows: int, label: str) -> np.ndarray:
    features = _as_numpy(value)
    assert features.ndim == 2, f"{label} must have shape (count, dimension)"
    assert features.shape[0] == rows, f"{label} is not aligned with its paired rows"
    assert features.shape[1] > 0, f"{label} has an empty embedding dimension"
    assert np.isfinite(features).all(), f"{label} contains non-finite values"
    np.testing.assert_allclose(
        np.linalg.norm(features, axis=1), 1.0, rtol=2e-3, atol=2e-3,
        err_msg=f"{label} rows must be L2-normalized",
    )
    return features


def test_real_yoloe_frame_emits_native_masks_two_feature_spaces_and_roundtrips():
    """Run one configured RGB frame through actual CUDA models and consumer IO."""
    if not torch.cuda.is_available():
        pytest.skip("GPU acceptance unresolved: CUDA is unavailable")

    frame_path = _required_local_path("CG_MODERNIZED_TEST_FRAME")
    config_path = _required_local_path("CG_MODERNIZED_TEST_CONFIG")
    if not config_path.is_file():
        pytest.skip(f"GPU acceptance unresolved: config path is not a file: {config_path}")
    yoloe_path = _required_local_path("CG_MODERNIZED_TEST_YOLOE_CHECKPOINT")
    siglip_path = _required_local_path("CG_MODERNIZED_TEST_SIGLIP2_CHECKPOINT")
    fgclip_path = _required_local_path("CG_MODERNIZED_TEST_FGCLIP2_CHECKPOINT")
    output_dir = _required_output_path("CG_MODERNIZED_TEST_OUTPUT_DIR")

    frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        pytest.skip(f"GPU acceptance unresolved: unreadable configured frame: {frame_path}")
    height, width = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    cfg = OmegaConf.load(config_path)
    # Pin local assets explicitly so the test cannot silently download or use
    # a different model selected by a user's default config.
    cfg.model.yoloe_checkpoint = str(yoloe_path)
    cfg.features.siglip2_checkpoint = str(siglip_path)
    cfg.features.fg_clip2_checkpoint = str(fgclip_path)
    cfg.device = "cuda:0"

    from conceptgraph.scripts import modernized_detections
    from conceptgraph.utils.gpu_memory_budget import NvmlMemoryBudget

    device = torch.device("cuda:0")
    budget = NvmlMemoryBudget(device)
    record = modernized_detections.run_frame(
        frame_rgb, cfg, device=device, budget=budget
    )

    required = {
        "xyxy", "confidence", "class_id", "mask", "classes", "image_crops",
        "image_feats", "text_feats", "image_feats_box", "text_feats_box",
    }
    assert required <= record.keys(), f"missing detection fields: {required - record.keys()}"
    classes = list(record["classes"])
    prompt_names = list(record["feature_metadata"]["yoloe_prompt_names"])
    assert prompt_names == [re.sub(r"\s+", "-", name.strip()) for name in classes]
    boxes = _as_numpy(record["xyxy"])
    scores = _as_numpy(record["confidence"]).reshape(-1)
    class_ids = _as_numpy(record["class_id"]).reshape(-1).astype(np.int64)
    masks = _as_numpy(record["mask"])
    count = len(boxes)

    assert classes, "configured YOLOE prompt list must not be empty"
    assert boxes.shape == (count, 4)
    assert scores.shape == class_ids.shape == (count,)
    assert masks.shape == (count, height, width), "YOLOE masks must retain source resolution"
    assert np.isfinite(boxes).all() and np.isfinite(scores).all()
    assert np.all((boxes[:, 0::2] >= 0) & (boxes[:, 0::2] <= width))
    assert np.all((boxes[:, 1::2] >= 0) & (boxes[:, 1::2] <= height))
    assert np.all((class_ids >= 0) & (class_ids < len(classes)))
    assert np.isin(masks, (False, True, 0, 1)).all(), "masks must be binary"

    image_feats = _assert_normalized_features(
        record["image_feats"], rows=count, label="SigLIP2 image_feats"
    )
    box_feats = _assert_normalized_features(
        record["image_feats_box"], rows=count, label="FG-CLIP2 image_feats_box"
    )
    # Text features are one category vector per configured prompt. Their
    # dimensions must match only the corresponding encoder's image vectors.
    text_feats = _assert_normalized_features(
        record["text_feats"], rows=len(classes), label="SigLIP2 text_feats"
    )
    box_text_feats = _assert_normalized_features(
        record["text_feats_box"], rows=len(classes), label="FG-CLIP2 text_feats_box"
    )
    assert text_feats.shape[1] == image_feats.shape[1]
    assert box_text_feats.shape[1] == box_feats.shape[1]

    metadata = record.get("feature_metadata")
    assert isinstance(metadata, dict), "cache must record feature encoder metadata"
    serialized_metadata = str(metadata).lower().replace("-", "")
    assert "siglip2" in serialized_metadata and "so400m" in serialized_metadata
    assert "fgclip2" in serialized_metadata and "so400m" in serialized_metadata
    assert "mask" in serialized_metadata and "padded" in serialized_metadata

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_path = output_dir / "modernized-yoloe-real-frame.pkl.gz"
    modernized_detections.save_gobs(saved_path, record)
    loaded = modernized_detections.load_gobs(saved_path)
    assert required <= loaded.keys()
    for key in (
        "xyxy", "confidence", "class_id", "mask", "image_feats", "text_feats",
        "image_feats_box", "text_feats_box",
    ):
        np.testing.assert_allclose(_as_numpy(loaded[key]), _as_numpy(record[key]))
    assert loaded["classes"] == classes
    assert loaded["feature_metadata"] == metadata
