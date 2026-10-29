"""Contract tests for deterministic frame sampling and dataset writing."""

import json

import numpy as np
from PIL import Image

from conceptgraph.scripts.prepare_realsense_bag import (
    should_sample_frame,
    write_dataset_config,
    write_sampled_frame,
)


def test_sample_offset_and_stride_choose_deterministic_replay_indices():
    assert [i for i in range(30) if should_sample_frame(i, 10, 0)] == [0, 10, 20]
    assert [i for i in range(30) if should_sample_frame(i, 10, 4)] == [4, 14, 24]


def test_dataset_config_records_exported_dimensions_and_depth_scale(tmp_path):
    intrinsics = {"fx": 900.0, "fy": 901.0, "cx": 639.5, "cy": 359.5}

    config_path = write_dataset_config(tmp_path, 1280, 720, intrinsics)

    import yaml

    contents = yaml.safe_load(config_path.read_text())
    assert contents["dataset_name"] == "realsense"
    assert contents["camera_params"] == {
        **intrinsics,
        "image_height": 720,
        "image_width": 1280,
        "png_depth_scale": 1000.0,
    }


def test_writer_emits_realsense_dataset_layout_depth_mm_pose_and_manifest(tmp_path):
    root = tmp_path / "dataset"
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 10, 20]
    depth_mm = np.array([[0, 1, 1000], [65535, 2500, 42]], dtype=np.uint16)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [1.25, -2.0, 0.5]

    write_sampled_frame(root, "room", 17, 1234.5, rgb, depth_mm, pose)

    scene = root / "room"
    rgb_files = list((scene / "rgb").glob("*.jpg"))
    depth_files = list((scene / "depth").glob("*.png"))
    pose_files = list((scene / "poses").glob("*.npy"))
    assert len(rgb_files) == len(depth_files) == len(pose_files) == 1
    assert rgb_files[0].stem == depth_files[0].stem == pose_files[0].stem
    np.testing.assert_array_equal(np.asarray(Image.open(depth_files[0])), depth_mm)
    np.testing.assert_allclose(np.load(pose_files[0]), pose)

    manifests = list(scene.glob("*manifest*.jsonl"))
    assert len(manifests) == 1
    rows = [json.loads(line) for line in manifests[0].read_text().splitlines()]
    assert rows == [{"frame_index": 17, "timestamp_ms": 1234.5}]
