"""Acceptance tests for retaining original compressed RGB image payloads."""

from __future__ import annotations

import hashlib
import json
import struct

import cv2
import numpy as np
import pytest

from conceptgraph.utils import realsense_archive
from conceptgraph.utils.realsense_archive import (
    NumpyArchiveReader,
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


def _jpeg_frame():
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[:, :4] = (220, 40, 15)
    rgb[:, 4:] = (10, 80, 200)
    ok, encoded = cv2.imencode(".jpg", rgb[:, :, ::-1])
    assert ok
    jpeg = encoded.tobytes()
    cdr = _compressed_image_cdr(
        "jpeg", jpeg, frame_id="camera_color_optical_frame"
    )
    return decode_compressed_image_cdr(cdr, "rgb"), jpeg, cdr


def _set_verified(output):
    manifest_path = output / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verified"] = True
    manifest["verification"] = {"verified": True}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _record_value(record, name):
    return record[name] if isinstance(record, dict) else getattr(record, name)


def _rewrite_npz(path, mutate):
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    mutate(arrays)
    replacement = path.with_name(".replacement.npz")
    np.savez(replacement, **arrays)
    replacement.replace(path)


def test_rgb_chunk_stores_exact_jpeg_bytes_and_separate_frame_metadata(tmp_path):
    decoded, jpeg, cdr = _jpeg_frame()
    output = tmp_path / "archive"
    writer = NumpyArchiveWriter(output, frames_per_chunk=1)
    writer.append_image(
        "rgb",
        decoded.frame,
        frame_number=7,
        timestamp_ns=200,
        header_timestamp_ns=decoded.header_timestamp_ns,
        frame_id=decoded.frame_id,
        source_payload=cdr,
        encoded_payload=jpeg,
        encoded_format=decoded.format,
    )
    writer.finalize({"sha256": "source"}, {"counts": {"rgb": 1}})

    with np.load(output / "rgb" / "rgb_000000.npz", allow_pickle=False) as chunk:
        assert "frames" not in chunk.files
        assert chunk["encoded_payload_bytes"].dtype == np.uint8
        assert chunk["encoded_payload_bytes"].tobytes() == jpeg
        np.testing.assert_array_equal(chunk["encoded_payload_offsets"], [0, len(jpeg)])
        assert chunk["frame_numbers"].tolist() == [7]
        assert chunk["timestamps_ns"].tolist() == [200]
        assert chunk["header_timestamps_ns"].tolist() == [decoded.header_timestamp_ns]
        assert chunk["frame_ids"].tolist() == [decoded.frame_id]
        assert chunk["source_payload_sha256"].tolist() == [hashlib.sha256(cdr).hexdigest()]


def test_reader_lazily_decodes_rgb_payload_chunk_and_reads_legacy_array_chunks(
    tmp_path, monkeypatch
):
    decoded, jpeg, cdr = _jpeg_frame()
    output = tmp_path / "archive"
    writer = NumpyArchiveWriter(output, frames_per_chunk=1)
    writer.append_image(
        "rgb",
        decoded.frame,
        frame_number=4,
        timestamp_ns=100,
        header_timestamp_ns=decoded.header_timestamp_ns,
        frame_id=decoded.frame_id,
        source_payload=cdr,
        encoded_payload=jpeg,
        encoded_format=decoded.format,
    )
    writer.finalize({"sha256": "source"}, {"counts": {"rgb": 1}})
    _set_verified(output)

    # A second legacy chunk uses the original array layout and must remain readable.
    legacy = output / "rgb" / "rgb_000001.npz"
    np.savez_compressed(
        legacy,
        frames=decoded.frame[None, ...],
        frame_numbers=np.array([9], dtype=np.int64),
        timestamps_ns=np.array([300], dtype=np.int64),
        header_timestamps_ns=np.array([decoded.header_timestamp_ns], dtype=np.int64),
        frame_ids=np.array([decoded.frame_id]),
        source_payload_sha256=np.array([hashlib.sha256(cdr).hexdigest()]),
    )
    manifest_path = output / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunk_hashes"]["rgb/rgb_000001.npz"] = hashlib.sha256(legacy.read_bytes()).hexdigest()
    manifest["rgb_chunks"] = 2
    manifest["stream_info"]["counts"]["rgb"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    original_load = np.load
    loaded = []

    def tracked_load(path, *args, **kwargs):
        loaded.append(str(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", tracked_load)
    reader = NumpyArchiveReader(output)
    images = reader.iter_images("rgb")
    assert loaded == []
    first = next(images)
    assert loaded == [str(output / "rgb" / "rgb_000000.npz")]
    np.testing.assert_array_equal(_record_value(first, "frame"), decoded.frame)
    assert _record_value(first, "frame_number") == 4
    second = next(images)
    np.testing.assert_array_equal(_record_value(second, "frame"), decoded.frame)
    assert _record_value(second, "frame_number") == 9


def test_reader_exposes_byte_exact_original_rgb_payloads(tmp_path):
    decoded, jpeg, cdr = _jpeg_frame()
    output = tmp_path / "archive"
    writer = NumpyArchiveWriter(output, frames_per_chunk=1)
    writer.append_image(
        "rgb",
        decoded.frame,
        frame_number=2,
        timestamp_ns=100,
        header_timestamp_ns=decoded.header_timestamp_ns,
        frame_id=decoded.frame_id,
        source_payload=cdr,
        encoded_payload=jpeg,
        encoded_format=decoded.format,
    )
    writer.finalize({"sha256": "source"}, {"counts": {"rgb": 1}})
    _set_verified(output)

    records = list(NumpyArchiveReader(output).iter_encoded_images("rgb"))
    assert len(records) == 1
    assert _record_value(records[0], "payload") == jpeg
    assert _record_value(records[0], "frame_number") == 2
    assert _record_value(records[0], "format") == decoded.format


def test_reader_rejects_malformed_encoded_payload_offsets(tmp_path):
    decoded, jpeg, cdr = _jpeg_frame()
    output = tmp_path / "archive"
    writer = NumpyArchiveWriter(output, frames_per_chunk=1)
    writer.append_image(
        "rgb", decoded.frame, frame_number=2, timestamp_ns=100,
        header_timestamp_ns=decoded.header_timestamp_ns, frame_id=decoded.frame_id,
        source_payload=cdr, encoded_payload=jpeg, encoded_format=decoded.format,
    )
    writer.finalize({"sha256": "source"}, {"counts": {"rgb": 1}})
    _set_verified(output)
    _rewrite_npz(
        output / "rgb" / "rgb_000000.npz",
        lambda arrays: arrays.__setitem__("encoded_payload_offsets", np.array(0)),
    )

    with pytest.raises(RuntimeError, match="offsets"):
        next(NumpyArchiveReader(output).iter_images("rgb"))


def test_reader_rejects_empty_encoded_image_payload(tmp_path):
    decoded, jpeg, cdr = _jpeg_frame()
    output = tmp_path / "archive"
    writer = NumpyArchiveWriter(output, frames_per_chunk=2)
    for frame_number, timestamp in ((2, 100), (3, 200)):
        writer.append_image(
            "rgb", decoded.frame, frame_number=frame_number, timestamp_ns=timestamp,
            header_timestamp_ns=decoded.header_timestamp_ns, frame_id=decoded.frame_id,
            source_payload=cdr, encoded_payload=jpeg, encoded_format=decoded.format,
        )
    writer.finalize({"sha256": "source"}, {"counts": {"rgb": 2}})
    _set_verified(output)
    _rewrite_npz(
        output / "rgb" / "rgb_000000.npz",
        lambda arrays: arrays["encoded_payload_offsets"].__setitem__(
            2, arrays["encoded_payload_offsets"][1]
        ),
    )

    with pytest.raises(RuntimeError, match="offsets"):
        next(NumpyArchiveReader(output).iter_encoded_images())


def test_verify_archive_compares_original_jpeg_bytes_and_existing_checks(
    tmp_path, monkeypatch
):
    decoded, jpeg, rgb_cdr = _jpeg_frame()
    depth = np.array([[0, 500, 1200], [2500, 8000, 15000]], dtype=np.uint16)
    ok, depth_png = cv2.imencode(".png", depth)
    assert ok
    depth_cdr = _compressed_image_cdr(
        "16UC1; compressedDepth png",
        struct.pack("<iff", 0, 0.0, 0.0) + depth_png.tobytes(),
        frame_id="camera_depth_optical_frame",
    )
    metadata = b"\x00camera metadata\xff"
    source = tmp_path / "recording.db3"
    connection = realsense_archive.sqlite3.connect(source)
    try:
        connection.executescript(
            """
            CREATE TABLE topics (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,
                serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL, data BLOB NOT NULL
            );
            """
        )
        topics = [
            (1, RGB_TOPIC, "sensor_msgs/msg/CompressedImage", "cdr", "[]"),
            (2, DEPTH_TOPIC, "sensor_msgs/msg/CompressedImage", "cdr", "[]"),
            (3, METADATA_TOPIC, "std_msgs/msg/String", "cdr", "[]"),
        ]
        rows = [(1, 1, 100, rgb_cdr), (2, 2, 200, depth_cdr), (3, 3, 870, metadata)]
        connection.executemany("INSERT INTO topics VALUES (?, ?, ?, ?, ?)", topics)
        connection.executemany("INSERT INTO messages VALUES (?, ?, ?, ?)", rows)
        connection.commit()
    finally:
        connection.close()

    output = tmp_path / "archive"
    writer = NumpyArchiveWriter(output, frames_per_chunk=1)
    writer.append_image(
        "rgb", decoded.frame, frame_number=1, timestamp_ns=100,
        header_timestamp_ns=decoded.header_timestamp_ns, frame_id=decoded.frame_id,
        source_payload=rgb_cdr, encoded_payload=jpeg, encoded_format=decoded.format,
    )
    decoded_depth = decode_compressed_image_cdr(depth_cdr, "depth")
    writer.append_image(
        "depth", decoded_depth.frame, frame_number=2, timestamp_ns=200,
        header_timestamp_ns=decoded_depth.header_timestamp_ns, frame_id=decoded_depth.frame_id,
        source_payload=depth_cdr,
    )
    writer.append_metadata(METADATA_TOPIC, "std_msgs/msg/String", 870, metadata)
    source_info = {
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "message_count": 3,
        "topic_message_counts": {name: 1 for _, name, *_ in topics},
    }
    writer.finalize(source_info, {"counts": {"rgb": 1, "depth": 1, "metadata": 1}})
    (output / "review.mp4").write_bytes(b"mock review video")
    monkeypatch.setattr(
        realsense_archive.subprocess,
        "run",
        lambda *args, **kwargs: type("Probe", (), {"stdout": "1\n", "stderr": "", "returncode": 0})(),
    )

    bgr = cv2.cvtColor(decoded.frame, cv2.COLOR_RGB2BGR)
    bgr = bgr.copy()
    bgr[0, 0, 0] = (int(bgr[0, 0, 0]) + 1) % 256
    captures = []

    class Capture:
        def __init__(self):
            self.frames = iter([bgr])
            self.released = False

        def read(self):
            frame = next(self.frames, None)
            return (frame is not None, frame)

        def release(self):
            self.released = True

    def make_capture(_path):
        capture = Capture()
        captures.append(capture)
        return capture

    monkeypatch.setattr(realsense_archive.cv2, "VideoCapture", make_capture)
    result = verify_archive(source, output)
    assert result["verified"] is True
    assert result["metadata_messages"] == 1
    assert result["review_video_frames"] == 1
    assert captures[0].released

    # The pixel-equivalent but byte-different JPEG must fail source verification.
    with np.load(output / "rgb" / "rgb_000000.npz", allow_pickle=False) as chunk:
        arrays = {key: chunk[key].copy() for key in chunk.files}
    changed = bytearray(arrays["encoded_payload_bytes"].tobytes())
    changed[-3] ^= 1
    arrays["encoded_payload_bytes"] = np.frombuffer(changed, dtype=np.uint8).copy()
    np.savez_compressed(output / "rgb" / "replacement.npz", **arrays)
    (output / "rgb" / "replacement.npz").replace(output / "rgb" / "rgb_000000.npz")
    manifest_path = output / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunk_hashes"]["rgb/rgb_000000.npz"] = hashlib.sha256(
        (output / "rgb" / "rgb_000000.npz").read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="encoded|payload|JPEG|jpeg"):
        verify_archive(source, output)
