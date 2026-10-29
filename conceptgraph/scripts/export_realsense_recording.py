"""Command-line entry point for a lossless RealSense DB3 archive export."""

from __future__ import annotations

import argparse
import json

from conceptgraph.utils.realsense_archive import export_recording


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, help="Input RealSense rosbag2 .db3")
    parser.add_argument("--output-dir", required=True, help="New or empty archive directory")
    parser.add_argument("--frames-per-chunk", type=int, default=128)
    args = parser.parse_args()
    manifest = export_recording(
        args.recording, args.output_dir, frames_per_chunk=args.frames_per_chunk
    )
    print(json.dumps({
        "output_dir": manifest["output_dir"],
        "verified": manifest["verified"],
        "counts": manifest["stream_info"]["counts"],
        "metadata_topics": len(manifest["metadata_topics"]),
        "review_video": manifest["stream_info"]["review_video"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
