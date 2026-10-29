"""Contract tests for metadata-only archive writing.

These tests intentionally exercise the public writer interface without a DB3,
ROS installation, video encoder, camera, or GPU.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from conceptgraph.utils.realsense_archive import NumpyArchiveWriter


def _finalize_metadata(writer: NumpyArchiveWriter) -> dict:
    """Flush writer chunks with the minimum source and stream descriptors."""
    return writer.finalize(source_info={}, stream_info={})


def _metadata_chunks(output_dir):
    return sorted((output_dir / "metadata").glob("topics_*.npz"))


def _read_metadata(output_dir):
    rows = []
    for path in _metadata_chunks(output_dir):
        with np.load(path, allow_pickle=False) as chunk:
            assert set(chunk.files) >= {
                "topic_names",
                "topic_types",
                "timestamps_ns",
                "payload_offsets",
                "payload_bytes",
            }
            assert all(chunk[name].dtype.kind != "O" for name in chunk.files)
            names = chunk["topic_names"].tolist()
            types = chunk["topic_types"].tolist()
            timestamps = chunk["timestamps_ns"].tolist()
            offsets = chunk["payload_offsets"].tolist()
            payload = chunk["payload_bytes"].tobytes()

        # Offsets are N+1 int64 boundaries into concatenated payload bytes.
        assert len(offsets) == len(names) + 1
        assert np.asarray(offsets).dtype == np.dtype("int64")
        assert offsets[0] == 0
        assert offsets[-1] == len(payload)
        for index, (name, topic_type, timestamp) in enumerate(
            zip(names, types, timestamps, strict=True)
        ):
            rows.append(
                (
                    str(name),
                    str(topic_type),
                    int(timestamp),
                    payload[offsets[index] : offsets[index + 1]],
                )
            )
    return rows


def test_metadata_round_trips_exactly_across_compressed_chunk_boundaries(tmp_path):
    output_dir = tmp_path / "archive"
    writer = NumpyArchiveWriter(output_dir, frames_per_chunk=2)
    expected = [
        ("/camera/color/image/metadata", "std_msgs/msg/String", 10, b"\x00first\xff"),
        ("/camera/depth/image/metadata", "std_msgs/msg/String", 20, b"second\x00"),
        ("/device/options", "rcl_interfaces/msg/ParameterEvent", 30, bytes(range(32))),
        ("/camera/color/image/metadata", "std_msgs/msg/String", 40, b"fourth"),
        ("/camera/depth/image/metadata", "std_msgs/msg/String", 50, b"\x80last\x00"),
    ]
    for topic_name, topic_type, timestamp, payload in expected:
        writer.append_metadata(topic_name, topic_type, timestamp, payload)

    _finalize_metadata(writer)

    chunks = _metadata_chunks(output_dir)
    assert [path.name for path in chunks] == [
        "topics_000000.npz",
        "topics_000001.npz",
        "topics_000002.npz",
    ]
    assert [len(np.load(path, allow_pickle=False)["topic_names"]) for path in chunks] == [
        2,
        2,
        1,
    ]
    assert _read_metadata(output_dir) == expected
    for path in chunks:
        with zipfile.ZipFile(path) as archive:
            assert archive.namelist()
            assert all(
                entry.compress_type == zipfile.ZIP_DEFLATED
                for entry in archive.infolist()
            )


def test_metadata_empty_payloads_and_unicode_names_use_non_object_arrays(tmp_path):
    output_dir = tmp_path / "archive"
    writer = NumpyArchiveWriter(output_dir, frames_per_chunk=4)
    expected = [
        ("/cámara/metadata", "std_msgs/msg/String", 1, b""),
        ("/camera/metadata", "std_msgs/msg/String", 2, b"metadata\x00bytes"),
    ]
    for row in expected:
        writer.append_metadata(*row)

    _finalize_metadata(writer)

    [chunk_path] = _metadata_chunks(output_dir)
    with np.load(chunk_path, allow_pickle=False) as chunk:
        assert chunk["topic_names"].dtype.kind == "U"
        assert chunk["topic_types"].dtype.kind == "U"
        assert chunk["timestamps_ns"].dtype == np.dtype("int64")
        assert chunk["payload_bytes"].dtype == np.dtype("uint8")
    assert _read_metadata(output_dir) == expected


@pytest.mark.parametrize(
    "row",
    [
        ("", "std_msgs/msg/String", 1, b"payload"),
        ("/topic", "", 1, b"payload"),
        ("/topic", "std_msgs/msg/String", 1, "not bytes"),
    ],
)
def test_append_metadata_rejects_invalid_arguments(tmp_path, row):
    writer = NumpyArchiveWriter(tmp_path / "archive", frames_per_chunk=2)
    with pytest.raises(ValueError):
        writer.append_metadata(*row)


def test_metadata_timestamps_preserve_db3_order_and_first_last_rows(tmp_path):
    output_dir = tmp_path / "archive"
    writer = NumpyArchiveWriter(output_dir, frames_per_chunk=3)
    expected = [
        ("/topic", "pkg/msg/Value", 870, b"first"),
        ("/other_topic", "pkg/msg/Value", 25, b"other"),
        ("/topic", "pkg/msg/Value", 0, b"regressed"),
        ("/topic", "pkg/msg/Value", 0, b"equal timestamp"),
    ]
    for row in expected:
        writer.append_metadata(*row)

    manifest = _finalize_metadata(writer)

    assert _read_metadata(output_dir) == expected
    assert manifest["metadata_topics"]["/topic"] == {
        "message_count": 3,
        "first_timestamp_ns": 870,
        "last_timestamp_ns": 0,
    }
    assert manifest["metadata_topics"]["/other_topic"] == {
        "message_count": 1,
        "first_timestamp_ns": 25,
        "last_timestamp_ns": 25,
    }


def test_finalized_writer_rejects_more_metadata_and_a_second_finalize(tmp_path):
    writer = NumpyArchiveWriter(tmp_path / "archive", frames_per_chunk=1)
    writer.append_metadata("/topic", "pkg/msg/Value", 1, b"payload")
    _finalize_metadata(writer)

    with pytest.raises(RuntimeError):
        writer.append_metadata("/topic", "pkg/msg/Value", 2, b"later")
    with pytest.raises(RuntimeError):
        _finalize_metadata(writer)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_writer_requires_positive_chunk_size(tmp_path, chunk_size):
    with pytest.raises(ValueError):
        NumpyArchiveWriter(tmp_path / "archive", frames_per_chunk=chunk_size)
