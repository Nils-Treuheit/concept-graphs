"""Contract tests for NumpyArchiveWriter's state and input-safety rules."""

from pathlib import Path

import numpy as np
import pytest

from conceptgraph.utils.realsense_archive import NumpyArchiveWriter


def _append_rgb(writer, *, frame_number=0, timestamp_ns=10, value=17):
    frame = np.full((2, 3, 3), value, dtype=np.uint8)
    writer.append_image(
        "rgb",
        frame,
        frame_number=frame_number,
        timestamp_ns=timestamp_ns,
        header_timestamp_ns=timestamp_ns + 1,
        frame_id="camera_color_optical_frame",
        source_payload=bytes([value]),
    )
    return frame


@pytest.mark.parametrize("bad_value", [-(1 << 63) - 1, 1 << 63])
def test_rejects_out_of_range_metadata_timestamp_without_consuming_timestamp(
    tmp_path, bad_value
):
    writer = NumpyArchiveWriter(tmp_path / "metadata_timestamp", frames_per_chunk=2)

    with pytest.raises(ValueError):
        writer.append_metadata("/status", "std_msgs/msg/String", bad_value, b"bad")

    writer.append_metadata("/status", "std_msgs/msg/String", 20, b"valid")
    writer.finalize({}, {})
    with np.load(
        tmp_path / "metadata_timestamp" / "metadata" / "topics_000000.npz",
        allow_pickle=False,
    ) as chunk:
        assert chunk["timestamps_ns"].tolist() == [20]
        assert chunk["payload_bytes"].tobytes() == b"valid"


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("frame_number", -(1 << 63) - 1),
        ("frame_number", 1 << 63),
        ("timestamp_ns", -(1 << 63) - 1),
        ("timestamp_ns", 1 << 63),
        ("header_timestamp_ns", -(1 << 63) - 1),
        ("header_timestamp_ns", 1 << 63),
    ],
)
def test_rejects_out_of_range_image_integer_without_changing_writer_state(
    tmp_path, field, bad_value
):
    output = tmp_path / "image_integer"
    writer = NumpyArchiveWriter(output, frames_per_chunk=2)
    kwargs = {
        "frame_number": 0,
        "timestamp_ns": 10,
        "header_timestamp_ns": 11,
    }
    kwargs[field] = bad_value

    with pytest.raises(ValueError):
        writer.append_image(
            "rgb",
            np.full((2, 3, 3), 23, dtype=np.uint8),
            **kwargs,
            frame_id="camera_color_optical_frame",
            source_payload=b"bad",
        )

    _append_rgb(writer, frame_number=0, timestamp_ns=10, value=23)
    writer.finalize({}, {})
    with np.load(output / "rgb" / "rgb_000000.npz", allow_pickle=False) as chunk:
        assert chunk["frame_numbers"].tolist() == [0]
        assert chunk["timestamps_ns"].tolist() == [10]
        assert chunk["header_timestamps_ns"].tolist() == [11]
        assert np.all(chunk["frames"][0] == 23)


def test_append_image_snapshots_pixels_before_a_later_chunk_flush(tmp_path):
    output = tmp_path / "snapshot"
    writer = NumpyArchiveWriter(output, frames_per_chunk=2)
    caller_frame = np.full((2, 3, 3), 41, dtype=np.uint8)
    writer.append_image(
        "rgb",
        caller_frame,
        frame_number=0,
        timestamp_ns=10,
        header_timestamp_ns=11,
        frame_id="camera_color_optical_frame",
        source_payload=b"first",
    )

    caller_frame[:] = 99
    _append_rgb(writer, frame_number=1, timestamp_ns=12, value=42)
    writer.finalize({}, {})

    with np.load(output / "rgb" / "rgb_000000.npz", allow_pickle=False) as chunk:
        assert np.all(chunk["frames"][0] == 41)
        assert np.all(chunk["frames"][1] == 42)


@pytest.mark.parametrize("descriptor_name", ["source_info", "stream_info"])
def test_non_json_descriptor_fails_before_flush_and_writer_remains_usable(
    tmp_path, descriptor_name
):
    output = tmp_path / descriptor_name
    writer = NumpyArchiveWriter(output, frames_per_chunk=2)
    _append_rgb(writer, frame_number=0, timestamp_ns=10, value=51)
    source_info, stream_info = {}, {}
    if descriptor_name == "source_info":
        source_info["invalid"] = {"set value"}
    else:
        stream_info["invalid"] = {"set value"}

    with pytest.raises(ValueError):
        writer.finalize(source_info, stream_info)

    # The rejected finalize must not flush buffered image data or make the
    # writer terminal. A second append and corrected finalize must succeed.
    assert not (output / "rgb" / "rgb_000000.npz").exists()
    _append_rgb(writer, frame_number=1, timestamp_ns=12, value=52)
    manifest = writer.finalize({}, {})
    assert manifest["verified"] is False
    with np.load(output / "rgb" / "rgb_000000.npz", allow_pickle=False) as chunk:
        assert chunk["frame_numbers"].tolist() == [0, 1]


def test_finalizing_empty_writer_creates_output_and_unverified_manifest(tmp_path):
    output = tmp_path / "empty_archive"
    assert not output.exists()

    manifest = NumpyArchiveWriter(output).finalize({}, {})

    assert output.is_dir()
    assert Path(output, "export_manifest.json").is_file()
    assert manifest["verified"] is False
