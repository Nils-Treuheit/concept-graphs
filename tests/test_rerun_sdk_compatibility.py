"""Offline regression checks for the Rerun 0.40 logging call sites."""

import ast
from pathlib import Path

import numpy as np
import pytest

rr = pytest.importorskip("rerun")

from conceptgraph.utils import optional_rerun_wrapper as rerun_wrapper


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def rerun_without_recording(monkeypatch):
    """Use the installed archetypes but intercept logs before any recording starts."""
    logged = []
    monkeypatch.setattr(rerun_wrapper.orr, "_rerun", rr)
    monkeypatch.setattr(rerun_wrapper.orr, "_config_use_rerun", True)
    monkeypatch.setattr(rr, "log", lambda entity, *components, **kwargs: logged.append((entity, components)))
    return logged


def test_camera_pose_is_a_valid_camera_to_world_transform(rerun_without_recording):
    intrinsics = np.array([[100.0, 0.0, 20.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, 3] = [1.0, 2.0, 3.0]

    rerun_wrapper.orr_log_camera(intrinsics, camera_to_world, camera_to_world, 40, 20, 0)

    transforms = [component for _, components in rerun_without_recording for component in components
                  if isinstance(component, rr.Transform3D)]
    assert len(transforms) == 1
    transform = transforms[0]
    assert transform.quaternion is not None
    assert transform.relation is not None
    assert transform.relation.as_arrow_array().to_pylist() == [rr.TransformRelation.ParentFromChild.value]
    assert transform.translation.as_arrow_array().to_pylist() == [[1.0, 2.0, 3.0]]


def test_replica_declares_y_up_for_world_view(rerun_without_recording):
    rerun_wrapper.orr_log_world_view_coordinates("replica")

    assert len(rerun_without_recording) == 1
    entity, components = rerun_without_recording[0]
    assert entity == "world"
    assert len(components) == 1
    assert [direction.name for direction in components[0].coordinates] == [
        "Right", "Up", "Back"
    ]


def test_non_replica_keeps_default_world_view_coordinates(rerun_without_recording):
    rerun_wrapper.orr_log_world_view_coordinates("record3d")

    assert rerun_without_recording == []


def test_replica_world_coordinates_are_optional_when_rerun_is_disabled(
    monkeypatch, rerun_without_recording
):
    monkeypatch.setattr(rerun_wrapper.orr, "_config_use_rerun", False)

    rerun_wrapper.orr_log_world_view_coordinates("replica")

    assert rerun_without_recording == []


def test_rgb_annotated_and_vlm_images_use_encoded_image(tmp_path, rerun_without_recording):
    from PIL import Image

    rgb_path = tmp_path / "frame.png"
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(rgb_path)
    annotated_dir = tmp_path / "annotated"
    annotated_dir.mkdir()
    Image.new("RGB", (2, 2), color=(30, 20, 10)).save(annotated_dir / "frame.png")
    vlm_path = tmp_path / "vlm.png"
    Image.new("RGB", (2, 2), color=(50, 60, 70)).save(vlm_path)

    rerun_wrapper.orr_log_rgb_image(rgb_path)
    rerun_wrapper.orr_log_annotated_image(rgb_path, annotated_dir)
    rerun_wrapper.orr_log_vlm_image(vlm_path, label="test")

    images = [(entity, component) for entity, components in rerun_without_recording
              for component in components if component is not None]
    assert [entity for entity, _ in images] == [
        "world/camera/rgb_image_encoded",
        "world/camera/rgb_image_annotated",
        "world/camera/vlm_image_test",
    ]
    assert all(isinstance(component, rr.EncodedImage) for _, component in images)


@pytest.mark.parametrize(
    "module_path",
    [
        "conceptgraph/slam/rerun_realtime_mapping.py",
        "conceptgraph/slam/r3d_stream_rerun_realtime_mapping.py",
    ],
)
def test_mapping_uses_current_rerun_frame_timeline_api(module_path):
    source = (REPO_ROOT / module_path).read_text()
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]

    assert any(
        call.func.attr == "set_time"
        and len(call.args) >= 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "frame"
        and any(keyword.arg == "sequence" for keyword in call.keywords)
        for call in calls
    )
    assert not any(call.func.attr == "set_time_sequence" for call in calls)
