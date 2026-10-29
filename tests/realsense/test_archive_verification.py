"""Focused verification tests using a synthetic rosbag2 SQLite recording."""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from conceptgraph.utils import realsense_archive
from conceptgraph.utils.realsense_archive import (
    NumpyArchiveWriter,
    decode_compressed_image_cdr,
    verify_archive,
)


RGB_TOPIC = "/device_0/sensor_1/Color_0/image/data/compressed"
DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data/compressedDepth"
METADATA_TOPIC = "/device_0/sensor_0/Depth_0/image/metadata"


def _cdr_string(value: str) -> bytes:
    encoded = value.encode("utf-8") + b"\0"
    return struct.pack("<I", len(encoded)) + encoded


def _compressed_image_cdr(image_format: str, image_data: bytes, *, frame_id: str) -> bytes:
    body = bytearray(struct.pack("<iI", 4, 123_456_789))
    body.extend(_cdr_string(frame_id))
    body.extend(b"\0" * (-len(body) % 4))
    body.extend(_cdr_string(image_format))
    body.extend(b"\0" * (-len(body) % 4))
    body.extend(struct.pack("<I", len(image_data)))
    body.extend(image_data)
    return b"\0\x01\0\0" + body


def _make_rosbag_fixture(path):
    topics = [
        (1, RGB_TOPIC, "sensor_msgs/msg/CompressedImage", "cdr", "[]"),
        (2, DEPTH_TOPIC, "sensor_msgs/msg/CompressedImage", "cdr", "[]"),
        (3, METADATA_TOPIC, "std_msgs/msg/String", "cdr", "[]"),
    ]
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[:, :4] = (220, 40, 15)
    rgb[:, 4:] = (10, 80, 200)
    ok, rgb_jpeg = cv2.imencode(".jpg", rgb[:, :, ::-1])
    assert ok
    rgb_payload = _compressed_image_cdr(
        "jpeg", rgb_jpeg.tobytes(), frame_id="camera_color_optical_frame"
    )

    depth = np.array([[0, 500, 1200], [2500, 8000, 15000]], dtype=np.uint16)
    ok, depth_png = cv2.imencode(".png", depth)
    assert ok
    depth_payload = _compressed_image_cdr(
        "16UC1; compressedDepth png",
        struct.pack("<iff", 0, 0.0, 0.0) + depth_png.tobytes(),
        frame_id="camera_depth_optical_frame",
    )
    metadata_payload = b"\x00camera metadata\xff"
    rows = [
        (1, 1, 100, rgb_payload),
        (2, 2, 200, depth_payload),
        (3, 3, 870, metadata_payload),
    ]

    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE topics (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                serialization_format TEXT NOT NULL,
                offered_qos_profiles TEXT NOT NULL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                topic_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                data BLOB NOT NULL
            );
            """
        )
        connection.executemany("INSERT INTO topics VALUES (?, ?, ?, ?, ?)", topics)
        connection.executemany("INSERT INTO messages VALUES (?, ?, ?, ?)", rows)
        connection.commit()
    finally:
        connection.close()
    return topics, rows, rgb_payload, depth_payload, metadata_payload


class _VideoCapture:
    def __init__(self, frames):
        self._frames = iter(frames)
        self.released = False

    def read(self):
        try:
            return True, next(self._frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True


def _build_archive(tmp_path, monkeypatch):
    source = tmp_path / "recording.db3"
    topics, rows, rgb_payload, depth_payload, metadata_payload = _make_rosbag_fixture(source)
    output = tmp_path / "archive"
    writer = NumpyArchiveWriter(output, frames_per_chunk=1)

    rgb_row = rows[0]
    depth_row = rows[1]
    rgb_decoded = decode_compressed_image_cdr(rgb_payload, "rgb")
    depth_decoded = decode_compressed_image_cdr(depth_payload, "depth")
    writer.append_image(
        "rgb",
        rgb_decoded.frame,
        frame_number=rgb_row[0],
        timestamp_ns=rgb_row[2],
        header_timestamp_ns=rgb_decoded.header_timestamp_ns,
        frame_id=rgb_decoded.frame_id,
        source_payload=rgb_payload,
    )
    writer.append_image(
        "depth",
        depth_decoded.frame,
        frame_number=depth_row[0],
        timestamp_ns=depth_row[2],
        header_timestamp_ns=depth_decoded.header_timestamp_ns,
        frame_id=depth_decoded.frame_id,
        source_payload=depth_payload,
    )
    writer.append_metadata(METADATA_TOPIC, "std_msgs/msg/String", rows[2][2], metadata_payload)

    topic_counts = {name: 1 for _, name, *_ in topics}
    source_info = {
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "size_bytes": source.stat().st_size,
        "message_count": len(rows),
        "topic_message_counts": topic_counts,
    }
    stream_info = {"counts": {"rgb": 1, "depth": 1, "metadata": 1}}
    writer.finalize(source_info, stream_info)
    (output / "review.mp4").write_bytes(b"mock review video")

    # Verification only needs ffprobe's decoded frame count and the decoded
    # review frame. No external codec, video file, or GPU is involved.
    monkeypatch.setattr(
        realsense_archive.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="1\n", stderr="", returncode=0),
    )
    rgb_bgr = cv2.cvtColor(rgb_decoded.frame, cv2.COLOR_RGB2BGR)
    # A real review video is intentionally lossy. Keep the synthetic decoded
    # frame one channel value away so the reported PSNR remains finite JSON.
    rgb_bgr = rgb_bgr.copy()
    rgb_bgr[0, 0, 0] = (int(rgb_bgr[0, 0, 0]) + 1) % 256
    captures = []

    def create_capture(_path):
        capture = _VideoCapture([rgb_bgr])
        captures.append(capture)
        return capture

    monkeypatch.setattr(realsense_archive.cv2, "VideoCapture", create_capture)
    return source, output, rows, captures


def _rewrite_npz(path, mutate):
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    mutate(arrays)
    replacement = path.with_suffix(".replacement.npz")
    np.savez_compressed(replacement, **arrays)
    replacement.replace(path)


def test_verify_archive_accepts_exact_arrays_metadata_and_video_frame_mapping(
    tmp_path, monkeypatch
):
    source, output, rows, captures = _build_archive(tmp_path, monkeypatch)
    original_source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    result = verify_archive(source, output)

    assert result["verified"] is True
    assert result["metadata_messages"] == 1
    assert result["review_video_frames"] == 1
    assert 0.0 < result["review_video_quality"]["rgb_to_video_mse"] < 1.0
    assert np.isfinite(result["review_video_quality"]["rgb_to_video_psnr_db"])
    assert captures[0].released
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_source_hash
    assert rows[2][2] == 870


@pytest.mark.parametrize("tamper", ["image", "metadata"])
def test_verify_archive_rejects_tampered_archived_payload(tmp_path, monkeypatch, tamper):
    source, output, _, _ = _build_archive(tmp_path, monkeypatch)
    if tamper == "image":
        _rewrite_npz(
            output / "rgb" / "rgb_000000.npz",
            lambda arrays: arrays["frames"].__setitem__(
                (0, 0, 0, 0), arrays["frames"][0, 0, 0, 0] ^ 0x01
            ),
        )
    else:
        _rewrite_npz(
            output / "metadata" / "topics_000000.npz",
            lambda arrays: arrays["payload_bytes"].__setitem__(
                0, arrays["payload_bytes"][0] ^ 0x01
            ),
        )

    with pytest.raises(RuntimeError):
        verify_archive(source, output)
