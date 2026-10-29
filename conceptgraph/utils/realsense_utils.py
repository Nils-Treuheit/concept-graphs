"""Live RGB-D capture and odometry for Intel RealSense cameras."""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def depth_to_meters(depth_raw: np.ndarray, depth_scale: float) -> np.ndarray:
    """Convert a 2D RealSense depth frame to contiguous meters."""
    depth_raw = np.asarray(depth_raw)
    if depth_raw.ndim != 2:
        raise ValueError(f"depth must be 2D, got shape {depth_raw.shape}")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be positive and finite")
    return np.ascontiguousarray(depth_raw.astype(np.float32) * depth_scale)


def is_valid_pose(pose: np.ndarray) -> bool:
    """Return whether a 4x4 matrix is a finite rigid transform."""
    pose = np.asarray(pose)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return False
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        return False
    rotation = pose[:3, :3]
    return bool(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)
    )


def source_to_target_to_camera_to_world(
    previous_camera_to_world: np.ndarray,
    source_to_target: np.ndarray,
) -> np.ndarray:
    """Compose source-to-target odometry into a camera-to-world pose."""
    previous_camera_to_world = np.asarray(previous_camera_to_world, dtype=np.float64)
    source_to_target = np.asarray(source_to_target, dtype=np.float64)
    if not is_valid_pose(previous_camera_to_world):
        raise ValueError("previous_camera_to_world must be a finite rigid 4x4 pose")
    if not is_valid_pose(source_to_target):
        raise ValueError("source_to_target must be a finite rigid 4x4 transform")
    result = previous_camera_to_world @ np.linalg.inv(source_to_target)
    if not is_valid_pose(result):
        raise ValueError("odometry composition produced an invalid pose")
    return result


class RGBDOdometry:
    """Sequential Open3D RGB-D odometry independent of a physical camera."""

    def __init__(
        self,
        intrinsics: dict[str, float],
        device: str = "auto",
        max_depth: float = 4.0,
        min_fitness: float = 0.25,
        o3d: Any = None,
    ) -> None:
        if o3d is None:
            import open3d as o3d

        self._o3d = o3d
        self.intrinsics = intrinsics
        self.max_depth = float(max_depth)
        self.min_fitness = float(min_fitness)
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if not 0.0 <= self.min_fitness <= 1.0:
            raise ValueError("min_fitness must be between zero and one")

        if device.lower() == "auto":
            use_cuda = o3d.core.cuda.is_available()
            self.device = o3d.core.Device("CUDA:0" if use_cuda else "CPU:0")
            if not use_cuda:
                logger.warning("Open3D CUDA is unavailable; RGB-D odometry is using CPU.")
        else:
            self.device = o3d.core.Device(device)
            if str(self.device).upper().startswith("CUDA:") and not o3d.core.cuda.is_available():
                raise RuntimeError(
                    "CUDA RGB-D odometry was requested but this Open3D build has no CUDA support."
                )

        self.camera_to_world = np.eye(4, dtype=np.float64)
        self._previous_rgbd: Any = None
        self._failure_reported = False

    def _make_rgbd(self, rgb: np.ndarray, depth_m: np.ndarray) -> Any:
        color_image = self._o3d.t.geometry.Image(
            self._o3d.core.Tensor(np.ascontiguousarray(rgb), device=self.device)
        )
        depth_image = self._o3d.t.geometry.Image(
            self._o3d.core.Tensor(np.ascontiguousarray(depth_m), device=self.device)
        )
        return self._o3d.t.geometry.RGBDImage(color_image, depth_image)

    def reset(self) -> None:
        """Release the retained reference frame and restart pose accumulation."""
        self._previous_rgbd = None
        self.camera_to_world = np.eye(4, dtype=np.float64)
        self._failure_reported = False

    def estimate(self, color_bgr: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """Estimate and return camera-to-world pose for an aligned RGB-D frame."""
        color_bgr = np.asarray(color_bgr)
        depth_m = np.asarray(depth_m)
        if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
            raise ValueError(f"color must be HxWx3 BGR, got shape {color_bgr.shape}")
        if depth_m.shape != color_bgr.shape[:2]:
            raise ValueError(
                f"depth shape {depth_m.shape} does not match color {color_bgr.shape[:2]}"
            )

        rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])
        current_rgbd = self._make_rgbd(rgb, depth_m)
        if self._previous_rgbd is None:
            self._previous_rgbd = current_rgbd
            return self.camera_to_world.copy()

        k = self.intrinsics
        intrinsic_matrix = np.array(
            [[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        intrinsic = self._o3d.core.Tensor(
            intrinsic_matrix, dtype=self._o3d.core.Dtype.Float64
        )
        try:
            result = self._o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                self._previous_rgbd,
                current_rgbd,
                intrinsic,
                depth_scale=1.0,
                depth_max=self.max_depth,
                method=self._o3d.t.pipelines.odometry.Method.Hybrid,
            )
            source_to_target = result.transformation.cpu().numpy()
            if result.fitness < self.min_fitness or not is_valid_pose(source_to_target):
                if not self._failure_reported:
                    logger.warning(
                        "RealSense odometry rejected (fitness %.3f); retaining the previous pose. "
                        "Further consecutive failures are suppressed.",
                        result.fitness,
                    )
                self._failure_reported = True
            else:
                self.camera_to_world = source_to_target_to_camera_to_world(
                    self.camera_to_world, source_to_target
                )
                self._failure_reported = False
        except Exception as exc:
            if not self._failure_reported:
                logger.warning(
                    "RealSense RGB-D odometry failed; retaining prior pose: %s. "
                    "Further consecutive failures are suppressed.",
                    exc,
                )
            self._failure_reported = True
        finally:
            self._previous_rgbd = current_rgbd
        return self.camera_to_world.copy()


class RealSenseDemoApp:
    """RealSense camera adapter with an RGB-D odometry pose source.

    The class intentionally mirrors the small interface used by the existing
    Record3D live mapper. D4xx devices provide RGB-D frames but generally do
    not provide a world pose, so Open3D estimates motion between frames.
    """

    def __init__(
        self,
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        odometry_device: str = "auto",
        odometry_max_depth: float = 4.0,
        odometry_min_fitness: float = 0.25,
    ) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.odometry_device = odometry_device
        self.odometry_max_depth = odometry_max_depth
        self.odometry_min_fitness = odometry_min_fitness
        self.pipeline: Any = None
        self.align: Any = None
        self.depth_scale: float | None = None
        self.intrinsics: dict[str, float] | None = None
        self.camera_to_world = np.eye(4, dtype=np.float64)
        self._previous_rgbd: Any = None
        self._o3d: Any = None
        self._o3d_device: Any = None
        self._connected = False
        self._odometry_failure_reported = False
        self._odometry: RGBDOdometry | None = None

    def connect_to_device(self, dev_idx: int = 0) -> None:
        """Start a depth+color stream, selecting by serial or device index."""
        if self._connected:
            return
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "RealSense support is optional; install it with `uv sync --extra realsense`."
            ) from exc

        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError("No Intel RealSense device was found.")
        if self.serial:
            selected = next(
                (d for d in devices if d.get_info(rs.camera_info.serial_number) == self.serial),
                None,
            )
            if selected is None:
                found = [d.get_info(rs.camera_info.serial_number) for d in devices]
                raise RuntimeError(
                    f"RealSense serial {self.serial!r} not found; available serials: {found}"
                )
        else:
            if dev_idx < 0 or dev_idx >= len(devices):
                raise RuntimeError(
                    f"RealSense device index {dev_idx} is unavailable; found {len(devices)} device(s)."
                )
            selected = devices[dev_idx]
            self.serial = selected.get_info(rs.camera_info.serial_number)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        try:
            profile = self.pipeline.start(config)
            atexit.register(self.stop)
            self.align = rs.align(rs.stream.color)
            self.depth_scale = float(
                profile.get_device().first_depth_sensor().get_depth_scale()
            )
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            camera = color_profile.get_intrinsics()
            self.width, self.height = camera.width, camera.height
            self.intrinsics = {
                "fx": float(camera.fx),
                "fy": float(camera.fy),
                "cx": float(camera.ppx),
                "cy": float(camera.ppy),
            }
            self._initialize_odometry()
            self._connected = True
            logger.info(
                "Connected RealSense %s at %dx%d %d FPS; depth scale %.8f m/unit",
                self.serial,
                self.width,
                self.height,
                self.fps,
                self.depth_scale,
            )
        except Exception:
            self.stop()
            raise

    def _initialize_odometry(self) -> None:
        import open3d as o3d

        self._o3d = o3d
        self._odometry = RGBDOdometry(
            intrinsics=self.intrinsics,
            device=self.odometry_device,
            max_depth=self.odometry_max_depth,
            min_fitness=self.odometry_min_fitness,
            o3d=o3d,
        )
        self._o3d_device = self._odometry.device

    def write_dataset_config(self, path: str | Path) -> None:
        """Write intrinsics metadata compatible with `RealsenseDataset`."""
        if self.intrinsics is None:
            raise RuntimeError("Connect the RealSense camera before writing its dataconfig.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "dataset_name": "realsense",
            "camera_params": {
                **self.intrinsics,
                "image_height": int(self.height),
                "image_width": int(self.width),
                "png_depth_scale": 1000.0,
            },
        }
        path.write_text(yaml.safe_dump(config, sort_keys=False))

    def _make_rgbd(self, rgb: np.ndarray, depth_m: np.ndarray) -> Any:
        if self._odometry is None:
            raise RuntimeError("Initialize odometry before creating RGB-D images.")
        return self._odometry._make_rgbd(rgb, depth_m)

    def _estimate_pose(self, color_bgr: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        if self._odometry is None:
            raise RuntimeError("Initialize odometry before estimating camera poses.")
        self.camera_to_world = self._odometry.estimate(color_bgr, depth_m)
        return self.camera_to_world.copy()

    def get_frame_data(self):
        """Return BGR color, metric depth, 4x4 intrinsics, and camera pose."""
        if not self._connected:
            raise RuntimeError("Connect to the RealSense device before requesting frames.")
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except Exception as exc:
            raise RuntimeError(f"Timed out waiting for a RealSense RGB-D frame: {exc}") from exc
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense did not provide both aligned color and depth frames.")

        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_to_meters(depth_raw, self.depth_scale)
        pose = self._estimate_pose(color_bgr, depth_m)

        intrinsics = torch.eye(4, dtype=torch.float32)
        intrinsics[0, 0] = self.intrinsics["fx"]
        intrinsics[1, 1] = self.intrinsics["fy"]
        intrinsics[0, 2] = self.intrinsics["cx"]
        intrinsics[1, 2] = self.intrinsics["cy"]
        return color_bgr, depth_m, intrinsics, pose.astype(np.float32)

    def stop(self) -> None:
        """Stop the sensor and release odometry frame references."""
        self._previous_rgbd = None
        if self._odometry is not None:
            self._odometry.reset()
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                logger.debug("Ignoring error while stopping RealSense pipeline", exc_info=True)
            self.pipeline = None
        self._connected = False

    def __enter__(self):
        self.connect_to_device()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop()
