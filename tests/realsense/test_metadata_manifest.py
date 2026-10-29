"""Focused contract tests for metadata writer validation and manifest output."""

import json

import pytest

from conceptgraph.utils.realsense_archive import NumpyArchiveWriter


@pytest.mark.parametrize("timestamp_ns", [1.25, True, False])
def test_append_metadata_rejects_non_integer_timestamp_without_poisoning_writer(
    tmp_path, timestamp_ns
):
    writer = NumpyArchiveWriter(tmp_path / "archive", frames_per_chunk=2)

    with pytest.raises(ValueError):
        writer.append_metadata("/x", "pkg/msg/X", timestamp_ns, b"invalid")

    # A rejected row must not affect ordering, counts, or later finalization.
    writer.append_metadata("/x", "pkg/msg/X", 10, b"first")
    writer.append_metadata("/x", "pkg/msg/X", 20, b"second")
    manifest = writer.finalize(
        {
            "topics": [
                {
                    "name": "/x",
                    "type": "pkg/msg/X",
                    "serialization_format": "cdr",
                    "offered_qos_profiles": "qos",
                    "message_count": 2,
                }
            ]
        },
        {},
    )

    assert manifest["verified"] is False
    summary = manifest["metadata_topics"]["/x"]
    assert summary == {
        "message_count": 2,
        "first_timestamp_ns": 10,
        "last_timestamp_ns": 20,
    }


def test_finalize_writes_unverified_manifest_with_topic_descriptors_and_summary(
    tmp_path,
):
    output_dir = tmp_path / "archive"
    writer = NumpyArchiveWriter(output_dir, frames_per_chunk=2)
    writer.append_metadata("/x", "pkg/msg/X", 101, b"alpha")
    writer.append_metadata("/x", "pkg/msg/X", 205, b"beta")

    source_info = {
        "topics": [
            {
                "name": "/x",
                "type": "pkg/msg/X",
                "serialization_format": "cdr",
                "offered_qos_profiles": "qos",
                "message_count": 2,
            }
        ]
    }
    manifest = writer.finalize(source_info, {})

    on_disk = json.loads((output_dir / "export_manifest.json").read_text())
    assert on_disk == manifest
    assert on_disk["verified"] is False
    assert on_disk["source_info"]["topics"] == source_info["topics"]
    assert on_disk["metadata_topics"]["/x"] == {
        "message_count": 2,
        "first_timestamp_ns": 101,
        "last_timestamp_ns": 205,
    }
