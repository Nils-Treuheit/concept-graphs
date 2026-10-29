"""Reusable contracts for archive-first RealSense preparation and capture.

These tests deliberately use synthetic frames and CDR payloads. They need no
camera, librealsense replay, or GPU and exercise the hand-off after recording.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import cv2


def _get(record, key):
    return record[key] if isinstance(record, dict) else getattr(record, key)


class _CdrWriter:
    """Small aligned CDR serializer for the CameraInfo fields under test."""

    def __init__(self):
        self.data = bytearray(b"\x00\x01\x00\x00")

    def _align(self, size):
        self.data.extend(b"\0" * (-len(self.data) % size))

    def scalar(self, fmt, value):
        size = struct.calcsize("<" + fmt)
        self._align(size)
        self.data.extend(struct.pack("<" + fmt, value))

    def string(self, value):
        encoded = value.encode() + b"\0"
        self.scalar("I", len(encoded))
        self.data.extend(encoded)

    def doubles(self, values):
        self.scalar("I", len(values))
        for value in values:
            self.scalar("d", value)

    def camera_info(self, *, width=2, height=2, frame_id="color_optical_frame"):
        self.scalar("i", 12)  # header stamp seconds
        self.scalar("I", 345_000_000)
        self.string(frame_id)
        self.scalar("I", height)
        self.scalar("I", width)
        self.string("plumb_bob")
        self.doubles([0.0] * 5)
        for value in (2.0, 0.0, 0.5, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0):
            self.scalar("d", value)
        for value in (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            self.scalar("d", value)
        for value in (2.0, 0.0, 0.5, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0):
            self.scalar("d", value)
        self.scalar("I", 0)  # binning_x
        self.scalar("I", 0)  # binning_y
        for value in (0, 0, 0, 0):
            self.scalar("I", value)  # ROI x/y/height/width
        self.scalar("B", 0)  # ROI do_rectify
        return bytes(self.data)

    def string_message(self, value):
        self.string(value)
        return bytes(self.data)


def test_camera_info_cdr_decoder_returns_intrinsics_and_rejects_truncation():
    from conceptgraph.utils.realsense_archive import decode_camera_info_cdr

    payload = _CdrWriter().camera_info(width=640, height=480)

    decoded = decode_camera_info_cdr(payload)

    assert decoded["width"] == 640
    assert decoded["height"] == 480
    assert decoded["frame_id"] == "color_optical_frame"
    assert decoded["distortion_model"] == "plumb_bob"
    np.testing.assert_allclose(decoded["k"], [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]])
    assert decoded["d"].shape == (5,)
    assert decoded["r"].shape == (3, 3)
    assert decoded["p"].shape == (3, 4)
    with pytest.raises(ValueError):
        decode_camera_info_cdr(payload[:-4])


def test_rgb_depth_pairing_uses_nearest_timestamp_and_enforces_tolerance():
    from conceptgraph.utils.realsense_archive import pair_rgb_depth_frames

    rgb = [
        {"frame_number": 2, "timestamp_ns": 100, "frame": "rgb-a"},
        {"frame_number": 3, "timestamp_ns": 250, "frame": "rgb-b"},
    ]
    depth = [
        {"frame_number": 8, "timestamp_ns": 90, "frame": "depth-a"},
        {"frame_number": 9, "timestamp_ns": 260, "frame": "depth-b"},
        {"frame_number": 10, "timestamp_ns": 900, "frame": "depth-outlier"},
    ]

    pairs = list(pair_rgb_depth_frames(rgb, depth, max_delta_ns=20))

    assert [(_get(color, "frame"), _get(z, "frame")) for color, z in pairs] == [
        ("rgb-a", "depth-a"),
        ("rgb-b", "depth-b"),
    ]
    with pytest.raises(ValueError):
        list(pair_rgb_depth_frames(rgb, depth, max_delta_ns=-1))


def test_rgb_depth_pairing_never_reuses_depth_and_skips_rgb_outside_tolerance():
    from conceptgraph.utils.realsense_archive import pair_rgb_depth_frames

    rgb = [
        {"frame_number": 1, "timestamp_ns": 100, "frame": "rgb-near"},
        {"frame_number": 2, "timestamp_ns": 101, "frame": "rgb-competing"},
        {"frame_number": 3, "timestamp_ns": 1000, "frame": "rgb-unmatched"},
    ]
    depth = [{"frame_number": 8, "timestamp_ns": 100, "frame": "depth-only-once"}]

    pairs = list(pair_rgb_depth_frames(rgb, depth, max_delta_ns=5))

    assert len(pairs) == 1
    assert _get(pairs[0][0], "frame") == "rgb-near"
    assert _get(pairs[0][1], "frame") == "depth-only-once"


def test_depth_alignment_projects_to_color_and_keeps_nearest_z_on_collision():
    from conceptgraph.utils.realsense_archive import (
        align_depth_to_color,
        decode_camera_info_cdr,
    )

    depth = np.array([[1000, 2000]], dtype=np.uint16)
    depth_intrinsics = decode_camera_info_cdr(_CdrWriter().camera_info(width=2, height=1))
    color_intrinsics = decode_camera_info_cdr(_CdrWriter().camera_info(width=3, height=1))
    for intrinsics in (depth_intrinsics, color_intrinsics):
        intrinsics["k"] = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        intrinsics["p"] = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
    depth_to_color = np.eye(4)
    # With fx=1, source points are x=0,z=1m and x=2,z=2m. A +2m
    # translation projects both to color pixel u=2, so z-buffering must
    # retain the nearer 1m sample (1000 raw units).
    depth_to_color[0, 3] = 2.0

    aligned = align_depth_to_color(
        depth, depth_intrinsics, color_intrinsics, depth_to_color,
        depth_scale_m=0.001,
    )

    assert aligned.shape == (1, 3)
    assert aligned.dtype == np.uint16
    assert aligned[0, 2] == 1000
    assert np.count_nonzero(aligned) == 1


@pytest.mark.parametrize(
    "depth_intrinsics,color_intrinsics,transform",
    [
        ({"k": np.diag([0.0, 1.0, 1.0]), "width": 1, "height": 1},
         {"k": np.eye(3), "width": 1, "height": 1}, np.eye(4)),
        ({"k": np.eye(3), "width": 1, "height": 1},
         {"k": np.eye(3), "width": 1, "height": 1}, np.eye(3)),
        ({"k": np.eye(3)}, {"k": np.eye(3), "width": 1, "height": 1}, np.eye(4)),
    ],
)
def test_depth_alignment_rejects_malformed_calibration(depth_intrinsics, color_intrinsics, transform):
    from conceptgraph.utils.realsense_archive import align_depth_to_color

    with pytest.raises(ValueError):
        align_depth_to_color(
            np.ones((1, 1), dtype=np.uint16),
            depth_intrinsics,
            color_intrinsics,
            transform,
        )


def test_preparation_cli_accepts_archive_directory(tmp_path):
    from conceptgraph.scripts.prepare_realsense_bag import build_parser

    archive = tmp_path / "my_corner_office_archive"
    args = build_parser().parse_args([
        "--archive", str(archive),
        "--dataset-root", str(tmp_path / "prepared"),
        "--scene-id", "office",
    ])

    assert Path(args.archive) == archive


def test_prepare_archive_uses_archived_calibration_without_realsense_dependency(
    tmp_path, monkeypatch
):
    """The archive path must run from recorded metadata, not SDK playback."""
    import builtins
    from argparse import Namespace

    from conceptgraph.scripts import prepare_realsense_bag as prepare_module

    from conceptgraph.utils.realsense_archive import NumpyArchiveWriter, NumpyArchiveReader

    archive = tmp_path / "recording_001_archive"
    capture_session_path = tmp_path / "capture_session.json"
    capture_session_path.write_text(json.dumps({
        "recordings": [{"recording_id": "office-recording", "archive": archive.name}],
        "still_images": [{
            "recording_id": "office-recording", "timestamp_ms": 100.0,
            "metadata": "still_000.json",
        }],
    }), encoding="utf-8")
    (tmp_path / "still_000.json").write_text("{}", encoding="utf-8")
    rgb = np.full((2, 2, 3), 80, dtype=np.uint8)
    depth = np.full((2, 2), 1000, dtype=np.uint16)
    calibration_payload = _CdrWriter().camera_info()
    writer = NumpyArchiveWriter(archive, frames_per_chunk=1)
    ok, rgb_encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    rgb_encoded = rgb_encoded.tobytes()
    for stream, frame, number, timestamp, frame_id, payload in [
        ("rgb", rgb, 7, 100, "color_optical_frame", b"rgb-source"),
        ("depth", depth, 8, 105, "depth_optical_frame", b"depth-source"),
    ]:
        kwargs = {}
        if stream == "rgb":
            kwargs = {"encoded_payload": rgb_encoded, "encoded_format": "png"}
        writer.append_image(
            stream, frame, frame_number=number, timestamp_ns=timestamp,
            header_timestamp_ns=timestamp, frame_id=frame_id, source_payload=payload,
            **kwargs,
        )
    metadata = [
        ("/device_0/sensor_1/Color_0/image/camera_info", "sensor_msgs/msg/CameraInfo", 100, calibration_payload),
        ("/device_0/sensor_0/Depth_0/image/camera_info", "sensor_msgs/msg/CameraInfo", 100,
         _CdrWriter().camera_info(frame_id="depth_optical_frame")),
        ("/device_0/sensor_1/Color_0/tf/ref_0", "std_msgs/msg/String", 100,
         _CdrWriter().string_message("rotation=1,0,0,0,1,0,0,0,1;translation=0.1,0,0")),
        ("/device_0/sensor_0/Depth_0/tf/ref_0", "std_msgs/msg/String", 100,
         _CdrWriter().string_message("rotation=1,0,0,0,1,0,0,0,1;translation=0.3,0,0")),
        ("/device_0/sensor_0/option/Depth_Units/value", "std_msgs/msg/String", 100,
         _CdrWriter().string_message("0.001")),
        ("/device_0/sensor_1/Color_0/image/metadata", "std_msgs/msg/String", 100,
         _CdrWriter().string_message("Frame number=7; timestamp=100")),
    ]
    for message in metadata:
        writer.append_metadata(*message)
    manifest = writer.finalize(
        {"path": "recording_001.db3", "sha256": "synthetic", "size_bytes": 0,
         "message_count": 8, "topic_count": 8},
        {"counts": {"rgb": 1, "depth": 1, "metadata": 6},
         "image_topics": {
             "rgb": "/device_0/sensor_1/Color_0/image/data/compressed",
             "depth": "/device_0/sensor_0/Depth_0/image/data/compressedDepth",
         }},
    )
    # The reader gates on verified flags. Source DB3 hash equivalence is
    # covered independently by test_archive_verification.py.
    manifest["verified"] = True
    manifest["verification"] = {"verified": True}
    (archive / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Use the real reader to validate chunks and yield ArchivedMetadataMessage
    # dataclasses, matching the production archive interface exactly.
    real_reader = NumpyArchiveReader(archive)
    assert real_reader.manifest["stream_info"]["image_topics"] == {
        "rgb": "/device_0/sensor_1/Color_0/image/data/compressed",
        "depth": "/device_0/sensor_0/Depth_0/image/data/compressedDepth",
    }
    rgb_chunk = archive / "rgb" / "rgb_000000.npz"
    with np.load(rgb_chunk, allow_pickle=False) as chunk:
        assert "frames" not in chunk.files
        assert "encoded_payload_bytes" in chunk.files
        assert "encoded_payload_offsets" in chunk.files
    encoded_record = next(real_reader.iter_encoded_images())
    assert _get(encoded_record, "payload") == rgb_encoded

    class Odometry:
        def __init__(self, **kwargs):
            assert kwargs["intrinsics"]["fx"] == pytest.approx(2.0)

        def estimate(self, _rgb, depth_m):
            assert depth_m.shape == (2, 2)
            assert depth_m.dtype == np.float32
            assert np.count_nonzero(depth_m) > 0
            return np.eye(4)

    monkeypatch.setattr(prepare_module, "RGBDOdometry", Odometry)
    observed_alignment = {}

    actual_alignment = prepare_module.align_depth_to_color

    def capture_alignment(depth_frame, depth_intrinsics, color_intrinsics,
                          depth_to_color_transform, *, depth_scale_m):
        observed_alignment["depth_scale_m"] = depth_scale_m
        observed_alignment["transform"] = np.asarray(depth_to_color_transform)
        assert depth_intrinsics["frame_id"] == "depth_optical_frame"
        assert color_intrinsics["frame_id"] == "color_optical_frame"
        return actual_alignment(
            depth_frame, depth_intrinsics, color_intrinsics,
            depth_to_color_transform, depth_scale_m=depth_scale_m,
        )

    monkeypatch.setattr(
        prepare_module, "align_depth_to_color", capture_alignment, raising=False
    )
    real_import = builtins.__import__

    def reject_pyrealsense(name, *args, **kwargs):
        if name == "pyrealsense2":
            raise AssertionError("archive preparation must not import pyrealsense2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_pyrealsense)
    args = Namespace(
        archive=str(archive), dataset_root=str(tmp_path / "dataset"), scene_id="office",
        sample_every=1, sample_offset=0, odometry_device="CUDA:0",
        odometry_max_depth=4.0, odometry_min_fitness=0.25,
        capture_session=str(capture_session_path), snapshot_pose_tolerance_ms=50.0,
    )

    # The real archive reader owns chunk validation and metadata decoding.
    monkeypatch.setattr(
        prepare_module, "NumpyArchiveReader", lambda _path: real_reader, raising=False
    )

    assert prepare_module.prepare(args) == 0
    assert (tmp_path / "dataset/office/rgb/00000007.jpg").is_file()
    assert (tmp_path / "dataset/office/depth/00000007.png").is_file()
    config = (tmp_path / "dataset/office/dataconfig.yaml").read_text()
    assert "image_width: 2" in config
    assert "image_height: 2" in config
    assert observed_alignment["depth_scale_m"] == pytest.approx(0.001)
    expected_depth_to_color = np.eye(4)
    expected_depth_to_color[0, 3] = 0.2
    np.testing.assert_allclose(observed_alignment["transform"], expected_depth_to_color)
    updated = json.loads(capture_session_path.read_text(encoding="utf-8"))
    still = updated["still_images"][0]
    assert still["recording_id"] == "office-recording"
    assert still["pose_pending"] is False
    assert still["pose_frame_number"] == 7
    assert json.loads((tmp_path / "still_000.json").read_text())["recording_id"] == "office-recording"


def test_capture_segment_wiring_stops_camera_then_closes_mp4_and_archives(tmp_path):
    from conceptgraph.scripts.record_realsense_bag import stop_and_finalize_capture

    db3 = tmp_path / "recording_001.db3"
    video = tmp_path / "recording_001.mp4"
    archive = tmp_path / "recording_001_archive"
    db3.write_bytes(b"closed db3")
    video.write_bytes(b"closed temporary video")
    events = []

    class Pipeline:
        def stop(self):
            events.append("pipeline-stop")

    class Session:
        still_images = []

        recordings = [{
            "recording_id": "recording_001", "bag": db3.name, "video": video.name,
        }]

        def stop_recording(self):
            events.append("mp4-close")

        def flush_metadata(self):
            events.append("session-save")

    def exporter(source, output):
        events.append("archive-export")
        assert Path(source).exists()
        Path(output).mkdir()
        return {"verified": True, "verification": {"verified": True}}

    session = Session()
    stop_and_finalize_capture(
        Pipeline(), session, db3, video, archive, "recording_001", exporter=exporter
    )

    assert events.index("pipeline-stop") < events.index("mp4-close") < events.index("archive-export")
    assert not db3.exists()
    assert not video.exists()
    assert archive.is_dir()
    assert session.recordings[0]["archive"] == archive.name
    assert session.recordings[0]["video"] == f"{archive.name}/review.mp4"
    assert "bag" not in session.recordings[0]


def test_missing_capture_session_skips_optional_pose_association(tmp_path, caplog):
    from conceptgraph.scripts.prepare_realsense_bag import _update_capture_session_poses

    _update_capture_session_poses(
        tmp_path / "missing_capture_session.json", "recording_001_archive", [], 50.0
    )

    assert "prepared dataset is complete" in caplog.text
    assert "pose association was skipped" in caplog.text
