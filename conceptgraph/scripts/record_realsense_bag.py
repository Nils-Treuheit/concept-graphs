"""Record RGB-D to a temporary DB3, then publish its verified archive."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any

import cv2
import numpy as np
from conceptgraph.utils.realsense_archive import export_recording

logger = logging.getLogger("conceptgraph.realsense_recording")


@dataclass(frozen=True)
class CaptureFrame:
    """One synchronized color frame with sensor-clock metadata."""

    rgb: np.ndarray
    frame_number: int
    timestamp_ms: float
    depth_frame_number: int
    depth_timestamp_ms: float


class CaptureSession:
    """Save RGB video, stills, and timestamp metadata for offline pose matching."""

    def __init__(
        self, output_dir: str | Path, writer_factory: Any = None, fps: int = 30,
        write_video: bool = True,
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / "capture_session.json").exists():
            raise FileExistsError(
                f"Capture session already exists in {self.output_dir}; choose a new output directory"
            )
        self.writer_factory = writer_factory or self._make_video_writer
        self.write_video = bool(write_video)
        self.fps = int(fps)
        self.recordings: list[dict[str, Any]] = []
        self.still_images: list[dict[str, Any]] = []
        self._writer: Any = None
        self._active_recording: dict[str, Any] | None = None
        self._previous_bgr: np.ndarray | None = None
        self._segment = 0
        self._saved_path: Path | None = None

    @property
    def is_recording(self) -> bool:
        return self._active_recording is not None

    @property
    def next_segment_number(self) -> int:
        return self._segment + 1

    @staticmethod
    def _make_video_writer(path: Path, fps: int, size: tuple[int, int]) -> Any:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size
        )
        if not writer.isOpened():
            writer.release()
            raise OSError(f"Could not open MP4 video writer for {path}")
        return writer

    def start_recording(
        self,
        frame: CaptureFrame,
        bag_path: str | Path | None = None,
        video_path: str | Path | None = None,
        recording_id: str | None = None,
    ) -> Path:
        if self._saved_path is not None:
            raise RuntimeError("This capture session has already been saved")
        if self._writer is not None:
            raise RuntimeError("A recording is already active")
        self._segment += 1
        height, width = frame.rgb.shape[:2]
        video_path = Path(video_path) if video_path else self.output_dir / f"recording_{self._segment:03d}.mp4"
        if video_path.exists():
            raise FileExistsError(f"Video already exists: {video_path}; choose a new output path")
        writer = self.writer_factory(video_path, self.fps, (width, height)) if self.write_video else None
        if writer is not None and hasattr(writer, "isOpened") and not writer.isOpened():
            writer.release()
            raise OSError(f"Could not open MP4 video writer for {video_path}")
        self._writer = writer
        self._active_recording = {
            "recording_id": recording_id or f"recording_{self._segment:03d}",
            "video": video_path.name,
            "bag": Path(bag_path).name if bag_path else None,
            "temporary_db3": str(Path(bag_path).relative_to(self.output_dir)) if bag_path else None,
            "fps": self.fps,
            "width": width,
            "height": height,
            "frame_count": 0,
            "mp4_frame_count": 0,
            "observed_frame_gaps": 0,
            "duplicated_gap_frames": 0,
            "unfilled_source_gap_frames": 0,
            "stored_frames": [],
            "first_timestamp_ms": None,
            "last_timestamp_ms": None,
            "_last_frame_number": None,
        }
        self.write_frame(frame)
        return video_path

    def write_frame(self, frame: CaptureFrame) -> None:
        if self._active_recording is None:
            return
        rgb = np.asarray(frame.rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("CaptureFrame.rgb must be an HxWx3 uint8 array")
        entry = self._active_recording
        previous_number = entry["_last_frame_number"]
        if previous_number is not None:
            gap = max(0, int(frame.frame_number) - previous_number - 1)
            entry["observed_frame_gaps"] += gap
            # Keep the constant-rate MP4 close to sensor time by holding the
            # previous image over short capture gaps. Long gaps remain explicit
            # in the metadata instead of generating minutes of duplicate video.
            fill = min(gap, self.fps * 5) if self._writer is not None else 0
            if self._writer is not None and self._previous_bgr is not None:
                for _ in range(fill):
                    self._writer.write(self._previous_bgr)
                    entry["mp4_frame_count"] += 1
                entry["duplicated_gap_frames"] += fill
                entry["unfilled_source_gap_frames"] += gap - fill
        if self._writer is not None:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self._writer.write(bgr)
            self._previous_bgr = bgr.copy()
        entry["_last_frame_number"] = int(frame.frame_number)
        entry["frame_count"] += 1
        entry["mp4_frame_count"] += 1
        timestamp_ms = float(frame.timestamp_ms)
        if entry["first_timestamp_ms"] is None:
            entry["first_timestamp_ms"] = timestamp_ms
        entry["last_timestamp_ms"] = timestamp_ms
        entry["stored_frames"].append(
            {
                "video_frame_index": entry["mp4_frame_count"] - 1 if self._writer is not None else None,
                "frame_number": int(frame.frame_number),
                "timestamp_ms": timestamp_ms,
                "depth_frame_number": int(
                    getattr(frame, "depth_frame_number", frame.frame_number)
                ),
                "depth_timestamp_ms": float(
                    getattr(frame, "depth_timestamp_ms", frame.timestamp_ms)
                ),
            }
        )

    def stop_recording(self) -> None:
        if self._active_recording is None:
            return
        writer = self._writer
        self._writer = None
        try:
            if writer is not None:
                writer.release()
        finally:
            entry = self._active_recording
            self._active_recording = None
            self._previous_bgr = None
            if entry is not None:
                entry.pop("_last_frame_number", None)
                self.recordings.append(entry)

    def take_image(self, frame: CaptureFrame) -> dict[str, str]:
        if self._saved_path is not None:
            raise RuntimeError("This capture session has already been saved")
        stem = f"still_{int(frame.frame_number):08d}_{float(frame.timestamp_ms):.3f}ms"
        image_path = self.output_dir / f"{stem}.png"
        metadata_path = self.output_dir / f"{stem}.json"
        from PIL import Image

        Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8), mode="RGB").save(image_path)
        metadata = {
            "image": image_path.name,
            "frame_number": int(frame.frame_number),
            "timestamp_ms": float(frame.timestamp_ms),
            "depth_frame_number": int(
                getattr(frame, "depth_frame_number", frame.frame_number)
            ),
            "depth_timestamp_ms": float(
                getattr(frame, "depth_timestamp_ms", frame.timestamp_ms)
            ),
            "video": self._active_recording["video"] if self._active_recording else None,
            "bag": self._active_recording["bag"] if self._active_recording else None,
            "recording_id": self._active_recording["recording_id"] if self._active_recording else None,
            "camera_pose": None,
            "pose_pending": True,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        metadata["metadata"] = metadata_path.name
        self.still_images.append(metadata)
        return {"image": image_path.name, "metadata": metadata_path.name}

    def save(self) -> Path:
        if self._saved_path is not None:
            return self._saved_path
        self.stop_recording()
        self._saved_path = self.output_dir / "capture_session.json"
        self.flush_metadata()
        return self._saved_path

    def flush_metadata(self) -> Path:
        """Persist current session metadata without closing future segments."""
        path = self.output_dir / "capture_session.json"
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "recordings": self.recordings,
            "still_images": self.still_images,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


def _require_verified_archive(manifest: dict) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("verified") is not True
        or not isinstance(manifest.get("verification"), dict)
        or manifest["verification"].get("verified") is not True
    ):
        raise RuntimeError("archive export did not confirm source verification; temporary DB3 was preserved")


def finalize_capture_segment(
    session: CaptureSession,
    db3_path,
    video_path,
    archive_dir,
    recording_id: str,
    *,
    exporter=None,
) -> dict:
    """Publish one capture segment as an archive and discard only verified temporaries."""
    session.stop_recording()
    source = Path(db3_path).expanduser().resolve()
    video = Path(video_path).expanduser().resolve()
    archive = Path(archive_dir).expanduser().resolve()
    if archive.exists() and any(archive.iterdir()):
        raise FileExistsError(f"archive destination is not empty: {archive}")
    manifest = (exporter or export_recording)(source, archive)
    _require_verified_archive(manifest)

    entry = next((row for row in reversed(session.recordings) if row.get("recording_id") == recording_id), None)
    if entry is None:
        raise RuntimeError(f"capture session has no completed recording {recording_id}")
    entry["archive"] = archive.name
    entry["video"] = f"{archive.name}/review.mp4"
    entry.pop("bag", None)
    entry.pop("temporary_db3", None)
    for still in getattr(session, "still_images", []):
        if still.get("recording_id") == recording_id:
            still["archive"] = archive.name
            still["video"] = f"{archive.name}/review.mp4"
            still.pop("bag", None)
            metadata_name = still.get("metadata")
            if metadata_name:
                (session.output_dir / metadata_name).write_text(
                    json.dumps(still, indent=2) + "\n", encoding="utf-8"
                )
    flush = getattr(session, "flush_metadata", None)
    if flush is not None:
        flush()
    elif hasattr(session, "save"):
        session.save()
    source.unlink()
    video.unlink(missing_ok=True)
    return manifest


def stop_and_finalize_capture(
    pipeline,
    session: CaptureSession,
    db3_path,
    video_path,
    archive_dir,
    recording_id: str,
    *,
    exporter=None,
) -> dict:
    """Stop the SDK first, then export and verify the completed recording."""
    pipeline.stop()
    return finalize_capture_segment(
        session, db3_path, video_path, archive_dir, recording_id, exporter=exporter
    )


def make_recording_config(
    rs: Any,
    bag_path: str | Path,
    serial: str | None,
    width: int,
    height: int,
    fps: int,
) -> Any:
    """Build a matching-resolution RGB-D recording configuration."""
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_record_to_file(str(bag_path))
    return config


def _profile_description(profile: Any, rs: Any) -> tuple[int, int, int, int, int, int]:
    depth = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color = profile.get_stream(rs.stream.color).as_video_stream_profile()
    return (
        depth.width(), depth.height(), depth.fps(),
        color.width(), color.height(), color.fps(),
    )


def record(args: argparse.Namespace) -> int:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "RealSense recording needs the optional dependency: uv sync --extra realsense"
        ) from exc

    if getattr(args, "output_dir", None):
        output_dir = Path(args.output_dir).expanduser().resolve()
        recording_id = getattr(args, "recording_id", "recording_001")
    else:
        legacy_output = Path(args.output).expanduser().resolve()
        if legacy_output.suffix.lower() != ".db3":
            raise ValueError("legacy --output must end in .db3")
        output_dir = legacy_output.parent
        recording_id = legacy_output.stem
        if legacy_output.exists():
            raise FileExistsError(f"Recording already exists: {legacy_output}; choose a new path")
    if not recording_id or Path(recording_id).name != recording_id:
        raise ValueError("recording-id must be a single path component")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_output = output_dir / f"{recording_id}_archive"
    temporary_dir = output_dir / ".recording_tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    output = temporary_dir / f"{recording_id}.db3"
    video_output = temporary_dir / f"{recording_id}.mp4"
    if output.exists() or video_output.exists() or archive_output.exists():
        raise FileExistsError(f"Capture output already exists for {recording_id}; use a new output directory")
    if args.preview and not (os.environ.get("DISPLAY") or os.name == "nt"):
        raise RuntimeError("No desktop display is available; pass --no-preview for headless recording")

    pipeline = rs.pipeline()
    config = make_recording_config(
        rs, output, args.serial, args.width, args.height, args.fps
    )
    started = False
    window_name = "RealSense RGB-D recording"
    window_created = False
    captured = 0
    estimated_gaps = 0
    previous_sensor_frame = None
    start_time = time.monotonic()
    rate_window_start = start_time
    rate_window_count = 0
    measured_fps = 0.0
    capture_session = CaptureSession(output_dir, fps=args.fps, write_video=False)
    stop_requested = [False]
    previous_sigint = None
    run_error: BaseException | None = None
    can_manage_sigint = threading.current_thread() is threading.main_thread()

    def request_stop(_signum: int, _frame: Any) -> None:
        # Avoid raising KeyboardInterrupt through librealsense's native wait
        # or stop calls. The loop observes this flag and finalizes normally.
        stop_requested[0] = True

    try:
        if can_manage_sigint:
            previous_sigint = signal.signal(signal.SIGINT, request_stop)
        profile = pipeline.start(config)
        started = True
        actual = _profile_description(profile, rs)
        requested = (args.width, args.height, args.fps)
        if actual[:3] != requested or actual[3:] != requested:
            raise RuntimeError(
                "Camera started with a different profile than requested: "
                f"depth={actual[:3]}, color={actual[3:]}, requested={requested} for both"
            )
        device = profile.get_device()
        serial = device.get_info(rs.camera_info.serial_number)
        logger.info(
            "Recording RealSense %s: depth Z16 and color RGB8 at %dx%d, %d FPS to %s",
            serial, args.width, args.height, args.fps, output,
        )
        logger.info(
            "Preview shows acquired frames. Press q or Esc, or close the window, "
            "to stop and finalize; there is no separate Save menu."
        )
        if args.preview:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            window_created = True

        while True:
            if stop_requested[0]:
                break
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                # A SIGINT can interrupt the SDK's native wait and surface as
                # a frame timeout instead of a Python KeyboardInterrupt.
                if stop_requested[0]:
                    break
                raise
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                raise RuntimeError("RealSense returned a frameset without color or depth")

            sensor_frame = int(color.get_frame_number())
            if previous_sensor_frame is not None:
                estimated_gaps += max(0, sensor_frame - previous_sensor_frame - 1)
            previous_sensor_frame = sensor_frame
            captured += 1
            rate_window_count += 1

            now = time.monotonic()
            capture_frame = CaptureFrame(
                rgb=np.asanyarray(color.get_data()).copy(),
                frame_number=sensor_frame,
                timestamp_ms=float(color.get_timestamp()),
                depth_frame_number=int(depth.get_frame_number()),
                depth_timestamp_ms=float(depth.get_timestamp()),
            )
            if not capture_session.is_recording:
                capture_session.start_recording(
                    capture_frame, bag_path=output, video_path=video_output,
                    recording_id=recording_id,
                )
            else:
                capture_session.write_frame(capture_frame)
            window_elapsed = now - rate_window_start
            if window_elapsed >= 2.0:
                measured_fps = rate_window_count / window_elapsed
                logger.info(
                    "Preview cadence %.1f FPS; %d frames received; estimated sensor-frame gaps %d",
                    measured_fps, captured, estimated_gaps,
                )
                rate_window_start = now
                rate_window_count = 0

            if args.preview:
                rgb = capture_frame.rgb
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                preview = cv2.resize(bgr, (960, 540), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    preview,
                    f"Received {measured_fps:.1f}/{args.fps} FPS | frame {sensor_frame}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                )
                cv2.putText(
                    preview,
                    "Q / Esc or close window: stop and save recording",
                    (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                )
                cv2.imshow(window_name, preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    # Some OpenCV backends report a closed window as an error.
                    break

            if args.duration_seconds and now - start_time >= args.duration_seconds:
                break
    except KeyboardInterrupt:
        # Let the SDK close its file cleanly. The first Ctrl+C requests stop;
        # subsequent Ctrl+C presses are ignored while the SDK finalizes.
        logger.info("Stop requested; finalizing the recording.")
    except BaseException as exc:
        run_error = exc
    finally:
        try:
            if started:
                try:
                    pipeline.stop()
                except KeyboardInterrupt:
                    # A non-main-thread caller cannot replace the process
                    # signal handler. Still report completion after the stop
                    # attempt rather than leaking a second interrupt.
                    logger.info("Interrupted during SDK shutdown; stop was attempted.")
                except BaseException as exc:
                    run_error = run_error or exc
        finally:
            try:
                capture_session.stop_recording()
            except BaseException as exc:
                run_error = run_error or exc
            finally:
                if window_created:
                    try:
                        cv2.destroyWindow(window_name)
                    except cv2.error:
                        logger.debug("Preview window was already closed", exc_info=True)
                if can_manage_sigint and previous_sigint is not None:
                    signal.signal(signal.SIGINT, previous_sigint)

    if run_error is not None:
        capture_session.save()
        raise run_error
    if captured == 0:
        capture_session.save()
        logger.info("Recording finalized: no camera frames were received; no archive segment was published.")
        return 0
    try:
        finalize_capture_segment(
            capture_session, output, video_output, archive_output, recording_id,
        )
    except BaseException:
        capture_session.save()
        raise
    capture_manifest = capture_session.save()

    if stop_requested[0]:
        logger.info("Stop requested; recording finalized.")

    elapsed = time.monotonic() - start_time
    average_fps = captured / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Recording finalized: %d color frames received in %.1f s (%.2f FPS average); "
        "estimated frame-number gaps=%d; verified archive=%s; metadata=%s",
        captured, elapsed, average_fps, estimated_gaps, archive_output, capture_manifest,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Capture folder for verified archives and session metadata")
    parser.add_argument("--recording-id", default="recording_001", help="Segment name without the _archive suffix")
    parser.add_argument("--serial", help="RealSense serial; defaults to the first available camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration-seconds", type=float, help="Stop automatically after this duration")
    parser.add_argument("--no-preview", dest="preview", action="store_false", help="Record without an OpenCV window")
    parser.set_defaults(preview=True)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("width, height and fps must be positive")
    if not args.recording_id or Path(args.recording_id).name != args.recording_id:
        raise SystemExit("recording-id must be a single path component")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise SystemExit("duration-seconds must be positive")
    return record(args)


if __name__ == "__main__":
    raise SystemExit(main())
