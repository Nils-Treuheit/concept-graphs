"""Contract tests for rejecting non-finite values before archive finalization."""

import numpy as np
import pytest

from conceptgraph.utils.realsense_archive import NumpyArchiveWriter


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("bad_descriptor", ["source_info", "stream_info"])
def test_non_json_finite_descriptor_does_not_flush_or_finalize_writer(
    tmp_path, bad_value, bad_descriptor
):
    output_dir = tmp_path / "archive"
    writer = NumpyArchiveWriter(output_dir, frames_per_chunk=2)
    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    writer.append_image(
        "rgb",
        rgb,
        frame_number=0,
        timestamp_ns=100,
        header_timestamp_ns=90,
        frame_id="camera_color_optical_frame",
        source_payload=b"source-cdr-payload",
    )
    metadata_payload = b"\x00metadata-cdr-payload\xff"
    writer.append_metadata("/device/status", "std_msgs/msg/String", 101, metadata_payload)

    source_info = {"recording": "capture.db3"}
    stream_info = {"rgb": {"width": 5, "height": 4}}
    if bad_descriptor == "source_info":
        source_info["invalid"] = bad_value
    else:
        stream_info["rgb"]["invalid"] = bad_value

    with pytest.raises(ValueError):
        writer.finalize(source_info, stream_info)

    assert not output_dir.exists() or not list(output_dir.rglob("*.npz"))

    source_info = {"recording": "capture.db3"}
    stream_info = {"rgb": {"width": 5, "height": 4}}
    manifest = writer.finalize(source_info, stream_info)
    assert manifest["verified"] is False

    with np.load(output_dir / "rgb" / "rgb_000000.npz", allow_pickle=False) as rgb_chunk:
        np.testing.assert_array_equal(rgb_chunk["frames"][0], rgb)
    with np.load(
        output_dir / "metadata" / "topics_000000.npz", allow_pickle=False
    ) as metadata_chunk:
        assert metadata_chunk["payload_bytes"].tobytes() == metadata_payload
