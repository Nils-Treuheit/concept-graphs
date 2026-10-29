"""Capture a RealSense RGB-D bag with a playable MP4 and still-image metadata."""

from __future__ import annotations

import argparse
from base64 import b64encode
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np

from conceptgraph.scripts.record_realsense_bag import (
    CaptureFrame,
    CaptureSession,
    stop_and_finalize_capture,
    _profile_description,
    make_recording_config,
)


def _preview_config(rs: Any, serial: str | None, width: int, height: int, fps: int) -> Any:
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    return config


class _CameraWorker(threading.Thread):
    def __init__(self, args: argparse.Namespace, events: queue.Queue):
        super().__init__(name="realsense-capture", daemon=True)
        self.args = args
        self.events = events
        self.commands: queue.Queue[str] = queue.Queue()
        self._frame_lock = threading.Lock()
        self.latest_frame: CaptureFrame | None = None
        self.latest_preview_rgb: np.ndarray | None = None
        # The archive exporter builds the final review MP4 after capture. Avoid
        # encoding a second video during the camera-rate recording.
        self.session = CaptureSession(args.output_dir, fps=args.fps, write_video=False)
        self.recording_requested = False
        self.current_bag: Path | None = None
        self.current_video: Path | None = None
        self.current_archive: Path | None = None
        self.current_recording_id: str | None = None
        self.frame_count = 0
        self._rate_start = time.monotonic()
        self._rate_frames = 0
        self.stop_after_first_frame = False
        self.save_after_first_frame = False

    def _post(self, kind: str, message: str) -> None:
        self.events.put((kind, message))

    def _check_profile(self, profile: Any, rs: Any) -> None:
        actual = _profile_description(profile, rs)
        expected = (self.args.width, self.args.height, self.args.fps)
        if actual[:3] != expected or actual[3:] != expected:
            raise RuntimeError(
                "Camera started with a different profile than requested: "
                f"depth={actual[:3]}, color={actual[3:]}, requested={expected} for both"
            )

    def _start_recording(self, pipeline: Any, rs: Any) -> None:
        segment = self.session.next_segment_number
        temporary_dir = self.args.output_dir / ".recording_tmp"
        temporary_dir.mkdir(parents=True, exist_ok=True)
        self.current_recording_id = f"recording_{segment:03d}"
        self.current_bag = temporary_dir / f"{self.current_recording_id}.db3"
        video_path = temporary_dir / f"{self.current_recording_id}.mp4"
        self.current_archive = self.args.output_dir / f"{self.current_recording_id}_archive"
        while self.current_bag.exists() or video_path.exists() or self.current_archive.exists():
            segment += 1
            self.current_recording_id = f"recording_{segment:03d}"
            self.current_bag = temporary_dir / f"{self.current_recording_id}.db3"
            video_path = temporary_dir / f"{self.current_recording_id}.mp4"
            self.current_archive = self.args.output_dir / f"{self.current_recording_id}_archive"
        self.current_video = video_path
        pipeline.stop()
        try:
            profile = pipeline.start(
                make_recording_config(
                    rs, self.current_bag, self.args.serial,
                    self.args.width, self.args.height, self.args.fps,
                )
            )
            self._check_profile(profile, rs)
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            pipeline.start(
                _preview_config(
                    rs, self.args.serial, self.args.width, self.args.height, self.args.fps
                )
            )
            raise
        self.recording_requested = True
        self.frame_count = 0
        self._post("recording", f"Recording {self.current_recording_id}; DB3 is temporary…")

    def _stop_recording(self, pipeline: Any, rs: Any) -> None:
        if not self.recording_requested:
            return
        stop_and_finalize_capture(
            pipeline, self.session, self.current_bag, self.current_video,
            self.current_archive, self.current_recording_id,
        )
        self.recording_requested = False
        self.current_bag = None
        self.current_video = None
        self.current_archive = None
        self.current_recording_id = None
        pipeline.start(
            _preview_config(
                rs, self.args.serial, self.args.width, self.args.height, self.args.fps
            )
        )
        self._post("preview", "Recording saved. Preview is ready for another segment.")

    def _save(self, pipeline: Any, pipeline_started: bool) -> None:
        if self.recording_requested:
            stop_and_finalize_capture(
                pipeline, self.session, self.current_bag, self.current_video,
                self.current_archive, self.current_recording_id,
            )
            pipeline_started = False
            self.recording_requested = False
            self.current_bag = self.current_video = self.current_archive = None
            self.current_recording_id = None
        if pipeline_started:
            pipeline.stop()
        path = self.session.save()
        self._post("saved", f"Saved capture session: {path}")

    def run(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            self.session.save()
            self._post("failed", "Install RealSense support with: uv sync --extra realsense")
            return

        pipeline = rs.pipeline()
        pipeline_started = False
        failure: str | None = None
        try:
            profile = pipeline.start(
                _preview_config(
                    rs, self.args.serial, self.args.width, self.args.height, self.args.fps
                )
            )
            pipeline_started = True
            self._check_profile(profile, rs)
            self._post(
                "preview",
                f"Live preview · {self.args.width}×{self.args.height} · requested {self.args.fps} FPS",
            )
            while True:
                try:
                    command = self.commands.get_nowait()
                except queue.Empty:
                    command = ""
                if command == "start" and not self.recording_requested:
                    self._start_recording(pipeline, rs)
                elif command == "stop" and self.recording_requested and self.session.is_recording:
                    self._stop_recording(pipeline, rs)
                elif command == "stop" and self.recording_requested:
                    self.stop_after_first_frame = True
                elif command == "save" and self.recording_requested and not self.session.is_recording:
                    self.save_after_first_frame = True
                elif command == "snapshot":
                    with self._frame_lock:
                        snapshot_frame = self.latest_frame
                    if self.recording_requested and snapshot_frame is not None:
                        result = self.session.take_image(snapshot_frame)
                        self._post(
                            "snapshot",
                            f"Saved still {result['image']}; pose will be added during archive replay.",
                        )
                    else:
                        self._post("recording", "Start recording before taking a timestamped still image.")
                elif command == "save":
                    self._save(pipeline, pipeline_started)
                    pipeline_started = False
                    return

                try:
                    frames = pipeline.wait_for_frames(timeout_ms=100)
                except RuntimeError as exc:
                    if "didn't arrive" in str(exc).lower():
                        continue
                    raise
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if not color or not depth:
                    continue
                capture_frame = CaptureFrame(
                    rgb=np.asanyarray(color.get_data()).copy(),
                    frame_number=int(color.get_frame_number()),
                    timestamp_ms=float(color.get_timestamp()),
                    depth_frame_number=int(depth.get_frame_number()),
                    depth_timestamp_ms=float(depth.get_timestamp()),
                )
                if self.recording_requested:
                    if not self.session.is_recording:
                        video_path = self.current_video
                        if video_path is None:
                            raise RuntimeError("Recording video path was not initialized")
                        self.session.start_recording(
                            capture_frame, bag_path=self.current_bag, video_path=video_path,
                            recording_id=self.current_recording_id,
                        )
                        self._post("recording_ready", "Recording RGB-D; still capture is ready.")
                    else:
                        self.session.write_frame(capture_frame)
                    self.frame_count += 1
                with self._frame_lock:
                    self.latest_frame = capture_frame
                    self.latest_preview_rgb = capture_frame.rgb

                self._rate_frames += 1
                elapsed = time.monotonic() - self._rate_start
                if elapsed >= 2.0:
                    rate = self._rate_frames / elapsed
                    state = "recording" if self.recording_requested else "preview"
                    self._post(
                        state,
                        f"{state.title()} · observed {rate:.1f} FPS / {self.args.fps} requested"
                        + (f" · {self.frame_count} frames" if self.recording_requested else ""),
                    )
                    self._rate_start = time.monotonic()
                    self._rate_frames = 0

                if self.stop_after_first_frame:
                    self.stop_after_first_frame = False
                    self._stop_recording(pipeline, rs)
                if self.save_after_first_frame:
                    self.save_after_first_frame = False
                    self._save(pipeline, pipeline_started)
                    pipeline_started = False
                    return
        except Exception as exc:
            failure = f"Capture failed: {exc}"
        finally:
            if pipeline_started:
                try:
                    pipeline.stop()
                except Exception as exc:
                    failure = f"{failure or 'Capture stopped'}; RGB-D finalization failed: {exc}"
            if self.session.is_recording:
                try:
                    self.session.stop_recording()
                except Exception as exc:
                    failure = f"{failure or 'Capture stopped'}; MP4 finalization failed: {exc}"
            try:
                self.session.save()
            except Exception as exc:
                failure = f"{failure or 'Capture ended'}; could not save capture metadata: {exc}"
            if failure:
                self._post("failed", f"{failure}\nSession metadata was saved when possible.")


def run(args: argparse.Namespace) -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError as exc:
        raise RuntimeError("This capture interface requires Python Tk support") from exc

    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "capture_session.json").exists():
        raise FileExistsError(
            f"Capture session already exists in {args.output_dir}; use a new --output-dir"
        )
    root = tk.Tk()
    root.title("ConceptGraphs · RealSense RGB-D Capture")
    root.geometry("1000x760")
    preview = tk.Label(root, text="Connecting to RealSense…", bg="#171717", fg="white")
    preview.pack(fill="both", expand=True, padx=10, pady=10)
    status = tk.Label(root, text="Starting camera…", anchor="w", justify="left")
    status.pack(fill="x", padx=10, pady=4)
    buttons = tk.Frame(root)
    buttons.pack(fill="x", padx=10, pady=10)
    events: queue.Queue = queue.Queue()
    worker = _CameraWorker(args, events)
    state = {"recording": False, "saving": False, "photo": None}

    def set_record_button() -> None:
        record_button.configure(text="Stop recording" if state["recording"] else "Start recording")

    def start_stop() -> None:
        if state["saving"]:
            return
        state["recording"] = not state["recording"]
        set_record_button()
        worker.commands.put("start" if state["recording"] else "stop")
        image_button.configure(state="disabled")

    def take_image() -> None:
        worker.commands.put("snapshot")

    def save_and_close() -> None:
        if state["saving"]:
            return
        state["saving"] = True
        save_button.configure(state="disabled", text="Saving…")
        record_button.configure(state="disabled")
        image_button.configure(state="disabled")
        worker.commands.put("save")

    record_button = tk.Button(buttons, text="Start recording", width=20, command=start_stop)
    record_button.pack(side="left", padx=(0, 8))
    image_button = tk.Button(buttons, text="Take image", width=20, state="disabled", command=take_image)
    image_button.pack(side="left", padx=8)
    save_button = tk.Button(buttons, text="Save and close", width=20, command=save_and_close)
    save_button.pack(side="left", padx=8)
    worker.start()

    def poll() -> None:
        while True:
            try:
                kind, message = events.get_nowait()
            except queue.Empty:
                break
            status.configure(text=message)
            if kind == "recording_ready":
                image_button.configure(state="normal")
            elif kind == "preview":
                image_button.configure(state="disabled")
            if kind == "failed":
                messagebox.showerror("RealSense capture", message, parent=root)
                record_button.configure(state="disabled")
                image_button.configure(state="disabled")
                save_button.configure(text="Close", state="normal", command=root.destroy)
                return
            elif kind == "saved":
                root.destroy()
                return
        with worker._frame_lock:
            rgb = worker.latest_preview_rgb
            capture = worker.latest_frame
        if rgb is not None:
            height, width = rgb.shape[:2]
            scale = min(960 / width, 660 / height, 1.0)
            if scale < 1.0:
                import cv2

                rgb = cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
                height, width = rgb.shape[:2]
            ppm = f"P6 {width} {height} 255\n".encode("ascii") + np.ascontiguousarray(rgb).tobytes()
            state["photo"] = tk.PhotoImage(data=b64encode(ppm).decode("ascii"), format="PPM")
            preview.configure(image=state["photo"], text="")
        if root.winfo_exists():
            root.after(40, poll)

    root.protocol("WM_DELETE_WINDOW", save_and_close)
    root.after(40, poll)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            save_and_close()
            while worker.is_alive():
                try:
                    root.update()
                except tk.TclError:
                    time.sleep(0.05)
                else:
                    time.sleep(0.05)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
    worker.join()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--serial", help="RealSense serial; defaults to the first available camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(args.width, args.height, args.fps) <= 0:
        raise SystemExit("width, height and fps must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
