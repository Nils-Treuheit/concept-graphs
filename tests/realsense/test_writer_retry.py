"""Focused writer contract tests for ordering and one-shot I/O failures."""

from pathlib import Path

import numpy as np
import pytest

from conceptgraph.utils.realsense_archive import NumpyArchiveWriter


def _append_rgb(writer, value, *, frame_number, timestamp_ns):
    frame = np.full((2, 3, 3), value, dtype=np.uint8)
    writer.append_image(
        "rgb",
        frame,
        frame_number=frame_number,
        timestamp_ns=timestamp_ns,
        header_timestamp_ns=timestamp_ns,
        frame_id="camera_color_optical_frame",
        source_payload=bytes([value]),
    )


def _finalize(writer):
    return writer.finalize(source_info={"source": "fixture"}, stream_info={})


def _rgb_frames(output_dir):
    chunks = sorted((output_dir / "rgb").glob("rgb_*.npz"))
    arrays = [np.load(path, allow_pickle=False) for path in chunks]
    return chunks, arrays


def test_ordering_is_strict_per_stream_and_rejected_rows_do_not_mutate_state(tmp_path):
    writer = NumpyArchiveWriter(tmp_path / "archive", frames_per_chunk=8)
    _append_rgb(writer, 10, frame_number=0, timestamp_ns=100)

    # The same number and timestamp are valid on an independent stream.
    writer.append_image(
        "depth",
        np.full((2, 3), 10, dtype=np.uint16),
        frame_number=0,
        timestamp_ns=100,
        header_timestamp_ns=100,
        frame_id="camera_depth_optical_frame",
        source_payload=b"depth",
    )

    with pytest.raises(ValueError):
        _append_rgb(writer, 20, frame_number=0, timestamp_ns=200)
    with pytest.raises(ValueError):
        _append_rgb(writer, 20, frame_number=1, timestamp_ns=100)

    _append_rgb(writer, 30, frame_number=1, timestamp_ns=200)
    _finalize(writer)

    chunks, arrays = _rgb_frames(tmp_path / "archive")
    assert [path.name for path in chunks] == ["rgb_000000.npz"]
    assert arrays[0]["frame_numbers"].tolist() == [0, 1]
    assert arrays[0]["timestamps_ns"].tolist() == [100, 200]
    assert arrays[0]["frames"][:, 0, 0, 0].tolist() == [10, 30]


def test_chunk_write_oserror_keeps_rows_for_retry_without_duplicate_indices(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "archive"
    output_dir.mkdir()
    writer = NumpyArchiveWriter(output_dir, frames_per_chunk=2)

    real_savez_compressed = np.savez_compressed
    failed_temp_paths = []
    fail_once = True

    def fail_once_during_chunk_write(file, *args, **kwargs):
        nonlocal fail_once
        path = Path(file)
        if fail_once and path.parent == output_dir / "rgb":
            fail_once = False
            failed_temp_paths.append(path)
            path.write_bytes(b"partial archive")
            raise OSError("simulated interrupted chunk write")
        return real_savez_compressed(file, *args, **kwargs)

    monkeypatch.setattr(np, "savez_compressed", fail_once_during_chunk_write)

    _append_rgb(writer, 11, frame_number=0, timestamp_ns=100)
    try:
        _append_rgb(writer, 22, frame_number=1, timestamp_ns=200)
    except OSError as exc:
        assert "simulated interrupted" in str(exc)

    # Implementations may flush a full chunk on append or during finalization.
    try:
        _finalize(writer)
    except OSError as exc:
        assert "simulated interrupted" in str(exc)
        _finalize(writer)

    assert not fail_once
    assert failed_temp_paths
    assert all(not path.exists() for path in failed_temp_paths)

    chunks, arrays = _rgb_frames(output_dir)
    assert [path.name for path in chunks] == ["rgb_000000.npz"]
    assert arrays[0]["frame_numbers"].tolist() == [0, 1]
    assert arrays[0]["timestamps_ns"].tolist() == [100, 200]
    assert arrays[0]["frames"][:, 0, 0, 0].tolist() == [11, 22]
    assert (output_dir / "export_manifest.json").is_file()
    assert all(not path.exists() for path in failed_temp_paths)


def test_manifest_install_oserror_can_be_retried_without_rewriting_chunks(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "archive"
    output_dir.mkdir()
    writer = NumpyArchiveWriter(output_dir, frames_per_chunk=2)
    _append_rgb(writer, 7, frame_number=0, timestamp_ns=70)
    _append_rgb(writer, 8, frame_number=1, timestamp_ns=80)

    real_replace = Path.replace
    failed_temp_paths = []
    fail_once = True

    def fail_once_installing_manifest(path, target):
        nonlocal fail_once
        target_path = Path(target)
        if fail_once and target_path.name == "export_manifest.json":
            fail_once = False
            failed_temp_paths.append(path)
            raise OSError("simulated manifest installation failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_once_installing_manifest)

    with pytest.raises(OSError, match="simulated manifest installation"):
        _finalize(writer)

    chunks_before_retry = sorted((output_dir / "rgb").glob("rgb_*.npz"))
    assert [path.name for path in chunks_before_retry] == ["rgb_000000.npz"]
    assert not (output_dir / "export_manifest.json").exists()
    assert failed_temp_paths
    assert all(not path.exists() for path in failed_temp_paths)

    _finalize(writer)

    chunks_after_retry, arrays = _rgb_frames(output_dir)
    assert chunks_after_retry == chunks_before_retry
    assert arrays[0]["frame_numbers"].tolist() == [0, 1]
    assert arrays[0]["timestamps_ns"].tolist() == [70, 80]
    assert arrays[0]["frames"][:, 0, 0, 0].tolist() == [7, 8]
    assert (output_dir / "export_manifest.json").is_file()
    assert all(not path.exists() for path in failed_temp_paths)
