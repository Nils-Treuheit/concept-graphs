"""Acceptance tests for lazy reads from verified NumPy RealSense archives."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from conceptgraph.utils.realsense_archive import NumpyArchiveReader, NumpyArchiveWriter


def _make_archive(tmp_path, *, verified=True):
    output = tmp_path / "archive"
    writer = NumpyArchiveWriter(output, frames_per_chunk=1)
    rgb_frames = [
        np.full((3, 4, 3), 17, dtype=np.uint8),
        np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
    ]
    rgb_payloads = [b"rgb-source-payload-0", b"rgb-source-payload-1"]
    rgb_rows = [
        {
            "frame": rgb_frames[0],
            "frame_number": 11,
            "timestamp_ns": 100,
            "header_timestamp_ns": 90,
            "frame_id": "color_optical_frame",
            "source_payload_sha256": hashlib.sha256(rgb_payloads[0]).hexdigest(),
        },
        {
            "frame": rgb_frames[1],
            "frame_number": 14,
            "timestamp_ns": 300,
            "header_timestamp_ns": 290,
            "frame_id": "color_optical_frame",
            "source_payload_sha256": hashlib.sha256(rgb_payloads[1]).hexdigest(),
        },
    ]
    for row, payload in zip(rgb_rows, rgb_payloads, strict=True):
        writer.append_image(
            "rgb",
            row["frame"],
            frame_number=row["frame_number"],
            timestamp_ns=row["timestamp_ns"],
            header_timestamp_ns=row["header_timestamp_ns"],
            frame_id=row["frame_id"],
            source_payload=payload,
        )

    depth_frames = [
        np.array([[100, 200], [300, 400]], dtype=np.uint16),
        np.array([[500, 600], [700, 800]], dtype=np.uint16),
    ]
    depth_payloads = [b"depth-source-payload-0", b"depth-source-payload-1"]
    depth_rows = []
    for index, (frame, payload) in enumerate(zip(depth_frames, depth_payloads, strict=True)):
        row = {
            "frame": frame,
            "frame_number": 21 + index,
            "timestamp_ns": 120 + index * 200,
            "header_timestamp_ns": 110 + index * 200,
            "frame_id": "depth_optical_frame",
            "source_payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
        depth_rows.append(row)
        writer.append_image(
            "depth",
            frame,
            frame_number=row["frame_number"],
            timestamp_ns=row["timestamp_ns"],
            header_timestamp_ns=row["header_timestamp_ns"],
            frame_id=row["frame_id"],
            source_payload=payload,
        )

    metadata_rows = [
        ("/camera/color/image/metadata", "std_msgs/msg/String", 870, b"raw\\x00color\\xff"),
        ("/device/options", "rcl_interfaces/msg/ParameterEvent", 0, b"raw\\x80options"),
    ]
    for row in metadata_rows:
        writer.append_metadata(*row)
    writer.finalize(
        {"source": "synthetic.db3"},
        {"counts": {"rgb": 2, "depth": 2, "metadata": 2}},
    )
    manifest_path = output / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verified"] = bool(verified)
    if verified:
        manifest["verification"] = {"verified": True}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return output, rgb_rows, depth_rows, metadata_rows


def _record_value(record, name):
    """Read the public record field whether represented by a dataclass or mapping."""
    if isinstance(record, dict):
        return record[name]
    return getattr(record, name)


def _rewrite_npz(path, mutate):
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    mutate(arrays)
    replacement = path.with_suffix(".replacement.npz")
    np.savez_compressed(replacement, **arrays)
    replacement.replace(path)


def test_reader_lazily_yields_image_records_from_verified_chunks(tmp_path, monkeypatch):
    output, expected_rgb, expected_depth, _ = _make_archive(tmp_path)
    original_load = np.load
    loaded_paths = []

    def tracked_load(file, *args, **kwargs):
        loaded_paths.append(str(file))
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", tracked_load)
    reader = NumpyArchiveReader(output)
    rgb_iterator = reader.iter_images("rgb")

    assert loaded_paths == []
    first = next(rgb_iterator)
    assert loaded_paths == [str(output / "rgb" / "rgb_000000.npz")]
    assert np.array_equal(_record_value(first, "frame"), expected_rgb[0]["frame"])
    for key in (
        "frame_number",
        "timestamp_ns",
        "header_timestamp_ns",
        "frame_id",
        "source_payload_sha256",
    ):
        assert _record_value(first, key) == expected_rgb[0][key]

    remaining_rgb = list(rgb_iterator)
    assert len(remaining_rgb) == 1
    assert np.array_equal(_record_value(remaining_rgb[0], "frame"), expected_rgb[1]["frame"])
    depth_records = list(reader.iter_images("depth"))
    assert len(depth_records) == len(expected_depth)
    for actual, expected in zip(depth_records, expected_depth, strict=True):
        np.testing.assert_array_equal(_record_value(actual, "frame"), expected["frame"])
        for key in (
            "frame_number",
            "timestamp_ns",
            "header_timestamp_ns",
            "frame_id",
            "source_payload_sha256",
        ):
            assert _record_value(actual, key) == expected[key]


def test_reader_yields_metadata_bytes_in_archive_order_and_filters_by_topic(tmp_path):
    output, _, _, expected = _make_archive(tmp_path)
    reader = NumpyArchiveReader(output)

    records = list(reader.iter_metadata())
    assert [
        (
            _record_value(row, "topic_name"),
            _record_value(row, "topic_type"),
            _record_value(row, "timestamp_ns"),
            _record_value(row, "payload"),
        )
        for row in records
    ] == expected
    filtered = list(reader.iter_metadata(topic_name="/device/options"))
    assert len(filtered) == 1
    assert _record_value(filtered[0], "payload") == b"raw\\x80options"
    with pytest.raises(ValueError):
        list(reader.iter_metadata(topic_name="/not/in/archive"))


def test_reader_rejects_unverified_archive_and_unknown_stream(tmp_path):
    unverified_path, _, _, _ = _make_archive(tmp_path / "unverified", verified=False)
    with pytest.raises(RuntimeError):
        NumpyArchiveReader(unverified_path)

    verified_path, _, _, _ = _make_archive(tmp_path / "verified")
    reader = NumpyArchiveReader(verified_path)
    with pytest.raises(ValueError):
        list(reader.iter_images("color"))


def test_reader_rejects_missing_listed_chunk_in_verified_archive(tmp_path):
    output, _, _, _ = _make_archive(tmp_path)
    (output / "depth" / "depth_000000.npz").unlink()

    with pytest.raises(RuntimeError):
        reader = NumpyArchiveReader(output)
        list(reader.iter_images("depth"))


def test_reader_can_recheck_chunk_hashes_on_demand(tmp_path):
    output, _, _, _ = _make_archive(tmp_path)
    _rewrite_npz(
        output / "rgb" / "rgb_000000.npz",
        lambda arrays: arrays["frames"].__setitem__(
            (0, 0, 0, 0), arrays["frames"][0, 0, 0, 0] ^ 0x01
        ),
    )

    with pytest.raises(RuntimeError, match="hash mismatch"):
        list(NumpyArchiveReader(output, verify_hashes=True).iter_images("rgb"))


@pytest.mark.parametrize("damage", ["missing", "corrupt_manifest", "corrupt_chunk"])
def test_reader_rejects_missing_or_corrupt_archive_input(tmp_path, damage):
    output, _, _, _ = _make_archive(tmp_path)
    if damage == "missing":
        output = tmp_path / "does-not-exist"
    elif damage == "corrupt_manifest":
        (output / "export_manifest.json").write_text("{broken", encoding="utf-8")
    else:
        chunk = output / "rgb" / "rgb_000000.npz"
        chunk.write_bytes(b"not an NPZ archive")

    with pytest.raises(RuntimeError):
        reader = NumpyArchiveReader(output)
        if damage == "corrupt_chunk":
            list(reader.iter_images("rgb"))
