"""Headless contracts for interactive RealSense capture and still-pose matching.

Proposed importable interfaces:

* ``CaptureSession(output_dir, writer_factory=..., fps=30)``
* ``start_recording(frame)``, ``write_frame(frame)``, ``stop_recording()``,
  ``take_image(frame)``, and ``save()``
* ``associate_still_captures(stills, pose_samples)``

A frame exposes ``rgb`` (HxWx3 uint8), ``frame_number``, and ``timestamp_ms``.
The writer factory accepts ``(video_path, fps, (width, height))``. These tests
intentionally avoid the camera, GUI, and GPU.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from conceptgraph.scripts.record_realsense_bag import CaptureSession
from conceptgraph.scripts.prepare_realsense_bag import associate_still_captures


class FakeWriter:
    def __init__(self, path: Path, fps: int, size: tuple[int, int]):
        self.path = Path(path)
        self.fps = fps
        self.size = size
        self.frames = []
        self.release_count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()

    def isOpened(self):
        return True

    def write(self, image):
        self.frames.append(np.asarray(image).copy())

    def release(self):
        self.release_count += 1


def frame(number: int, timestamp_ms: float, value: int = 10):
    return SimpleNamespace(
        rgb=np.full((4, 6, 3), value, dtype=np.uint8),
        frame_number=number,
        timestamp_ms=timestamp_ms,
    )


def test_recording_can_start_stop_restart_and_writes_only_active_frames(tmp_path):
    writers = []

    def writer_factory(path, fps, size):
        writer = FakeWriter(path, fps, size)
        writers.append(writer)
        return writer

    session = CaptureSession(tmp_path, writer_factory=writer_factory, fps=30)
    session.start_recording(frame(10, 100.0))
    session.write_frame(frame(11, 133.3, 20))
    session.stop_recording()
    session.write_frame(frame(12, 166.6, 30))
    session.start_recording(frame(13, 199.9, 40))
    session.stop_recording()

    assert len(writers) == 2
    assert [len(writer.frames) for writer in writers] == [2, 1]
    assert [writer.release_count for writer in writers] == [1, 1]


def test_still_image_is_saved_with_timestamp_frame_id_and_pending_pose(tmp_path):
    session = CaptureSession(tmp_path, writer_factory=lambda *_: FakeWriter(*_))

    result = session.take_image(frame(42, 1400.25, value=80))

    image_path = tmp_path / result["image"]
    metadata_path = tmp_path / result["metadata"]
    assert image_path.is_file()
    assert metadata_path.is_file()
    with Image.open(image_path) as image:
        assert image.size == (6, 4)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["frame_number"] == 42
    assert metadata["timestamp_ms"] == 1400.25
    assert metadata["camera_pose"] is None
    assert metadata["pose_pending"] is True


def test_save_finalizes_active_recording_once_and_persists_session_metadata(tmp_path):
    writers = []

    def writer_factory(path, fps, size):
        writer = FakeWriter(path, fps, size)
        writers.append(writer)
        return writer

    session = CaptureSession(tmp_path, writer_factory=writer_factory, fps=30)
    session.start_recording(frame(5, 50.0))
    session.write_frame(frame(6, 83.3))
    session.take_image(frame(6, 83.3))

    manifest_path = Path(session.save())
    session.save()  # A repeated Save must not release the same writer twice.

    assert writers[0].release_count == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["recordings"]
    assert manifest["still_images"]
    assert manifest["recordings"][0]["frame_count"] == 2


def test_metadata_capture_without_video_does_not_create_or_write_a_video_writer(tmp_path):
    writer_calls = []

    def forbidden_writer(*args):
        writer_calls.append(args)
        raise AssertionError("write_video=False must not open a video writer")

    session = CaptureSession(
        tmp_path, writer_factory=forbidden_writer, fps=30, write_video=False
    )
    video_path = tmp_path / "temporary.mp4"
    session.start_recording(
        frame(10, 100.0), bag_path=tmp_path / "temporary.db3", video_path=video_path
    )
    session.write_frame(frame(11, 133.3, 20))
    session.stop_recording()
    manifest_path = session.save()

    assert writer_calls == []
    assert not video_path.exists()
    recording = json.loads(manifest_path.read_text(encoding="utf-8"))["recordings"][0]
    assert recording["frame_count"] == 2
    assert recording["stored_frames"][0]["video_frame_index"] is None


def test_still_pose_association_prefers_exact_then_nearest_timestamp():
    pose_a = np.eye(4)
    pose_a[0, 3] = 1.0
    pose_b = np.eye(4)
    pose_b[0, 3] = 2.0
    stills = [
        {"frame_number": 4, "timestamp_ms": 100.0, "image": "a.png"},
        {"frame_number": 9, "timestamp_ms": 148.0, "image": "b.png"},
    ]
    poses = [
        {"frame_number": 3, "timestamp_ms": 90.0, "camera_pose": pose_a.tolist()},
        {"frame_number": 5, "timestamp_ms": 100.0, "camera_pose": pose_b.tolist()},
        {"frame_number": 10, "timestamp_ms": 150.0, "camera_pose": pose_a.tolist()},
    ]

    associated = associate_still_captures(stills, poses)

    assert associated[0]["pose_frame_number"] == 5
    assert associated[0]["pose_timestamp_delta_ms"] == 0.0
    np.testing.assert_allclose(associated[0]["camera_pose"], pose_b)
    assert associated[1]["pose_frame_number"] == 10
    assert associated[1]["pose_timestamp_delta_ms"] == 2.0
    np.testing.assert_allclose(associated[1]["camera_pose"], pose_a)
    assert [row["frame_number"] for row in associated] == [4, 9]


def test_still_pose_association_leaves_pose_pending_when_pose_samples_are_empty():
    still = {"frame_number": 12, "timestamp_ms": 400.0, "image": "still.png"}

    associated = associate_still_captures([still], [])

    assert len(associated) == 1
    assert associated[0]["frame_number"] == 12
    assert associated[0]["camera_pose"] is None
    assert associated[0]["pose_frame_number"] is None
    assert associated[0]["pose_timestamp_delta_ms"] is None
