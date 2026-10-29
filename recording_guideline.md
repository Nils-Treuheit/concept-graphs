# RealSense room recording guide

## What the current run records

The D415 was configured at **640 × 480 and 30 FPS**. That is the camera stream rate. The current mapper does not save every camera frame: it acquires one frame, runs the full detection, segmentation, feature, object-map, and visualization work synchronously, then requests another frame.

In the supplied run, the progress output reached 40 mapped frames in about 34 seconds, or roughly **1.2 saved/mapped frames per second** over that interval. The rate varies with per-frame workload. The camera emits frames at 30 FPS, while the current application processes and saves about one selected frame per second. Intermediate sensor frames are not recorded by this loop. At that observed pace, `total_frames=1800` would take about 25 minutes; treat that as a rough estimate, not a fixed duration.

The pose tracker also reported an odometry fitness of 0.229, below its 0.25 acceptance threshold. It retained the previous pose for that frame. If this repeats, the camera path stalls while images and detections continue, so the resulting map can be spatially wrong even though capture continues.

## Recommended room sweep

1. Start with the camera still for two seconds and point it at a textured wall or a recognizable furniture arrangement. This gives RGB-D odometry a stable reference.
2. Walk a slow clockwise perimeter sweep. Keep the camera about 0.5–2 m from surfaces and move at roughly 0.1 m/s while keeping the horizon level.
3. Aim toward each wall long enough to see its corners and nearby furniture. Overlap consecutive views by about one third to one half.
4. Pause for one or two seconds at each corner. Use a small sideways translation while turning so nearby furniture shifts against the background and produces depth parallax.
5. Make a second, shorter pass for occluded areas such as behind chairs, under tables, and around doorways. Change position instead of only rotating in place.
6. Avoid fast turns, sudden stops, blank close-up walls, and views dominated by windows, mirrors, or reflective surfaces. Keep both color and depth frames well lit and sharp.
7. Watch the camera trajectory and odometry warnings. If the pose stays fixed while you move, stop and restart from a textured view rather than completing a long sweep with a frozen trajectory.

For a first trial, use a smaller frame budget such as `total_frames=600`. At the observed processing rate, that is roughly an eight-to-ten-minute run. Increase the budget only after confirming the trajectory follows the camera and the saved frames cover the room.

## Improving capture cadence and tracking

The current loop couples capture to the entire ConceptGraphs pipeline. Its effective saved-frame rate therefore changes with model and mapping workload. A stronger design is to acquire and track RGB-D frames continuously, update the pose on every camera frame, and send only selected keyframes to detection and map integration. The selected keyframe must retain its timestamp and interpolated or exact odometry pose. This keeps camera motion tracking frequent without running segmentation and object-map updates at 30 FPS.

A reasonable first keyframe policy to evaluate is one ConceptGraphs candidate every 15–30 camera frames, about every 0.5–1 second at 30 FPS, with earlier selection when translation or rotation exceeds a threshold. This should be measured on the D415 before being treated as a configured project default. Open3D RGB-D odometry is already the current pose source; the log's rejected fitness indicates tracking quality needs attention, not that the current mapper is using a separate learned visual tracker.

## Capture-first RealSense mode

The button-based record-first workflow requests matching **1280 × 720 color and depth at 30 FPS**. Start/Stop Recording finalizes a native RGB-D `.db3` and a playable RGB `.mp4`; Take Image writes a timestamped still and JSON sidecar; Save and close finalizes the session manifest and closes the app. The configured rate is not a guarantee of received frame rate. The live status reports observed cadence and gaps. Check that the camera starts with the requested streams; USB 2 connections or competing bandwidth can prevent this mode from starting. See the [librealsense release notes](https://github.com/realsenseai/librealsense/wiki/Release-Notes) and [record/playback guide](https://github.com/realsenseai/librealsense/blob/master/doc/record-and-playback.md).

The MP4 uses a fixed 30 FPS timeline. Gaps up to five seconds are filled by
repeating the previous image. Longer gaps are capped to avoid huge duplicate
videos, and `unfilled_source_gap_frames` records the omitted duration in
`capture_session.json`. Such MP4s can play shorter than the sensor recording;
use DB3 timestamps as the authoritative timing source.

The SDK recording is finalized when capture stops. Replay it with `prepare_realsense_bag` to align depth to color and estimate camera motion on every recorded frame. Pass the session's `capture_session.json` with `--capture-session` to associate each still image with the nearest odometry pose within 50 ms. The default exporter keeps every 15th frame for mapping, or about 2 frames per second at a steady 30 FPS. It stores each selected frame with the pose estimated from the complete preceding sequence. The default odometry device is CUDA:0, so a CUDA-enabled Open3D build is required.

Start with one offset (`--sample-offset 0`). Alternate offsets can add temporal samples, but each produces a separate prepared dataset and mapping run. The mapper does not merge those independent graphs automatically. Use a single sweep first, then add offsets only if object coverage is visibly incomplete.

The recorder uses the camera's synchronized RGB-D framesets. During replay, depth is aligned to the color image before odometry and export. The preparation step reads the color stream's camera intrinsics and writes them with the exported dataset. It does not lock exposure; use steady room lighting and avoid aiming directly at bright windows when tracking.

## Export a compact, Python-friendly archive

Keep the original `.db3` as the source of truth. Export it with:

```bash
uv run python -m conceptgraph.scripts.export_realsense_recording \
  --recording /path/to/recording.db3 \
  --output-dir /path/to/recording_archive
```

The output stores original compressed RGB image payloads in packed `.npz`
chunks with offsets. Those members are not recompressed because JPEG bytes are
already compressed. Depth arrays and remaining DB3 messages, including
per-frame metadata and camera information, use compressed NPZ chunks.
`export_manifest.json` records topic descriptors and verification results.
`review.mp4` is a compact HEVC video for visual inspection. The exporter checks
the RGB bytes, decoded pixel arrays, depth arrays, and metadata against the
read-only DB3 before publishing and never removes the source recording.
