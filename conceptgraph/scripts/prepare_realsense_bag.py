"""Replay a RealSense recording, estimate every-frame poses, and export sampled RGB-D."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import cv2
import numpy as np
import yaml

from conceptgraph.utils.realsense_utils import RGBDOdometry, depth_to_meters
from conceptgraph.utils.realsense_archive import (
    NumpyArchiveReader,
    align_depth_to_color,
    decode_camera_info_cdr,
    decode_string_cdr,
    pair_rgb_depth_frames,
)

logger = logging.getLogger("conceptgraph.realsense_preparation")


def associate_still_captures(
    stills: list[dict[str, Any]],
    pose_samples: list[dict[str, Any]],
    max_delta_ms: float = 50.0,
) -> list[dict[str, Any]]:
    """Match still sidecars to the nearest RGB-D odometry timestamp.

    The signed delta is ``pose timestamp - still timestamp``. Stills outside
    the tolerance remain pending instead of receiving a misleading pose.
    """
    if max_delta_ms < 0:
        raise ValueError("max_delta_ms must be non-negative")
    valid_samples = [
        sample for sample in pose_samples
        if sample.get("timestamp_ms") is not None
        and sample.get("camera_pose") is not None
    ]
    result = []
    for still in stills:
        updated = copy.deepcopy(still)
        updated.update(
            camera_pose=None,
            pose_pending=True,
            pose_frame_number=None,
            pose_timestamp_ms=None,
            pose_timestamp_delta_ms=None,
            pose_timestamp_abs_delta_ms=None,
        )
        if not valid_samples or still.get("timestamp_ms") is None:
            result.append(updated)
            continue
        timestamp_ms = float(still["timestamp_ms"])
        nearest = min(
            valid_samples,
            key=lambda sample: abs(float(sample["timestamp_ms"]) - timestamp_ms),
        )
        delta = float(nearest["timestamp_ms"]) - timestamp_ms
        if abs(delta) <= max_delta_ms:
            updated.update(
                camera_pose=nearest["camera_pose"],
                pose_pending=False,
                pose_frame_number=nearest.get("frame_number"),
                pose_timestamp_ms=float(nearest["timestamp_ms"]),
                pose_timestamp_delta_ms=delta,
                pose_timestamp_abs_delta_ms=abs(delta),
            )
        result.append(updated)
    return result


def should_sample_frame(frame_index: int, sample_every: int, sample_offset: int = 0) -> bool:
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")
    if not 0 <= sample_offset < sample_every:
        raise ValueError("sample_offset must satisfy 0 <= offset < sample_every")
    return frame_index % sample_every == sample_offset


def get_blur_score(rgb: np.ndarray) -> float:
    """Return a Laplacian-variance sharpness score for an RGB image."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape HxWx3, got {rgb.shape}")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _stage_frame_period(
    period: list[dict[str, Any]],
    period_index: int,
    scratch_dir: Path,
    blur_tolerance: float,
    sample_offset: int,
    sample_every: int,
) -> list[dict[str, Any]]:
    """Keep near-sharpest frames in a period and stage their RGB-D payloads."""
    if not period:
        return []
    maximum = max(float(frame["score"]) for frame in period)
    threshold = maximum * (1.0 - blur_tolerance)
    candidates = [frame for frame in period if float(frame["score"]) >= threshold]
    staged = []
    for frame in candidates:
        payload_path = scratch_dir / f"{period_index:08d}_{frame['source_index']:010d}.npz"
        np.savez_compressed(
            payload_path,
            rgb=frame["rgb"],
            depth_mm=frame["depth_mm"],
            pose=frame["pose"],
        )
        staged.append({
            "source_index": frame["source_index"],
            "frame_number": frame["frame_number"],
            "timestamp_ms": frame["timestamp_ms"],
            "score": frame["score"],
            "offset_distance": abs(
                int(frame["source_index"]) % sample_every - sample_offset
            ),
            "payload_path": payload_path,
        })
    return staged


def _select_spaced_candidates(
    candidate_periods: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Choose one per period, maximizing spacing before quality and offset.

    Sharpness is the deterministic tie-breaker after temporal spacing. The
    dynamic program retains only scalar states and back-pointers, so video
    payloads can remain staged on disk until the final choices are known.
    """
    if not candidate_periods:
        return []
    for period_index, period in enumerate(candidate_periods):
        if not period:
            raise ValueError(f"candidate period {period_index} is empty")
        if period_index and min(int(c["source_index"]) for c in period) <= max(
            int(c["source_index"]) for c in candidate_periods[period_index - 1]
        ):
            raise ValueError("candidate frame ids must increase between periods")

    def feasible(minimum_gap: int) -> bool:
        reachable = [True] * len(candidate_periods[0])
        for period_index in range(1, len(candidate_periods)):
            previous = candidate_periods[period_index - 1]
            current = candidate_periods[period_index]
            next_reachable = []
            for candidate in current:
                current_index = int(candidate["source_index"])
                next_reachable.append(any(
                    reachable[previous_index]
                    and current_index - int(previous_candidate["source_index"]) >= minimum_gap
                    for previous_index, previous_candidate in enumerate(previous)
                ))
            reachable = next_reachable
        return any(reachable)

    min_source_index = min(
        int(candidate["source_index"])
        for period in candidate_periods for candidate in period
    )
    max_source_index = max(
        int(candidate["source_index"])
        for period in candidate_periods for candidate in period
    )
    low, high = 1, max_source_index - min_source_index
    while low < high:
        middle = (low + high + 1) // 2
        if feasible(middle):
            low = middle
        else:
            high = middle - 1
    best_minimum_gap = low

    # Among max-spacing sequences, maximize sharpness, then prefer the offset.
    states: list[list[tuple[float, int]]] = [[
        (float(candidate["score"]), int(candidate["offset_distance"]))
        for candidate in candidate_periods[0]
    ]]
    backpointers: list[list[int | None]] = [[None] * len(candidate_periods[0])]
    for period_index in range(1, len(candidate_periods)):
        previous = candidate_periods[period_index - 1]
        current = candidate_periods[period_index]
        period_scores = []
        period_backpointers = []
        for candidate in current:
            current_index = int(candidate["source_index"])
            eligible = [
                previous_index
                for previous_index, previous_candidate in enumerate(previous)
                if states[-1][previous_index][0] != float("-inf")
                and current_index - int(previous_candidate["source_index"]) >= best_minimum_gap
            ]
            if not eligible:
                period_scores.append((float("-inf"), 0))
                period_backpointers.append(None)
                continue
            previous_index = max(
                eligible,
                key=lambda index: (states[-1][index][0], -states[-1][index][1]),
            )
            previous_score, previous_offset_distance = states[-1][previous_index]
            period_scores.append((
                previous_score + float(candidate["score"]),
                previous_offset_distance + int(candidate["offset_distance"]),
            ))
            period_backpointers.append(previous_index)
        states.append(period_scores)
        backpointers.append(period_backpointers)

    selected_indices = [0] * len(candidate_periods)
    reachable_final = [
        index for index, (score, _) in enumerate(states[-1])
        if score != float("-inf")
    ]
    if not reachable_final:
        raise RuntimeError("maximum-spacing candidate path became infeasible")
    selected_indices[-1] = max(
        reachable_final,
        key=lambda index: (states[-1][index][0], -states[-1][index][1]),
    )
    for period_index in range(len(candidate_periods) - 1, 0, -1):
        previous_index = backpointers[period_index][selected_indices[period_index]]
        if previous_index is None:
            raise RuntimeError("candidate selection backtracking failed")
        selected_indices[period_index - 1] = previous_index
    return [
        period[index]
        for period, index in zip(candidate_periods, selected_indices)
    ]


def _write_selected_candidates(
    selected: list[dict[str, Any]], output_root: str | Path, scene: str,
) -> None:
    for candidate in selected:
        with np.load(candidate["payload_path"]) as payload:
            write_sampled_frame(
                output_root,
                scene,
                int(candidate["frame_number"]),
                float(candidate["timestamp_ms"]),
                payload["rgb"],
                payload["depth_mm"],
                payload["pose"],
            )


def write_dataset_config(
    scene_root: Path,
    width: int,
    height: int,
    intrinsics: dict[str, float],
    depth_scale: float = 1000.0,
) -> Path:
    path = scene_root / "dataconfig.yaml"
    contents = {
        "dataset_name": "realsense",
        "camera_params": {
            **intrinsics,
            "image_height": int(height),
            "image_width": int(width),
            "png_depth_scale": float(depth_scale),
        },
    }
    path.write_text(yaml.safe_dump(contents, sort_keys=False), encoding="utf-8")
    return path


def write_sampled_frame(
    output_root: str | Path,
    scene: str,
    frame_index: int,
    timestamp_ms: float,
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    pose: np.ndarray,
) -> None:
    """Write one selected frame in the existing RealsenseDataset layout."""
    from PIL import Image

    rgb = np.asarray(rgb)
    depth_mm = np.asarray(depth_mm)
    pose = np.asarray(pose, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape HxWx3, got {rgb.shape}")
    if depth_mm.shape != rgb.shape[:2] or depth_mm.dtype != np.uint16:
        raise ValueError("depth_mm must be uint16 and match the RGB height and width")
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must be a finite 4x4 matrix")

    scene_root = Path(output_root) / scene
    rgb_dir, depth_dir, pose_dir = (
        scene_root / "rgb", scene_root / "depth", scene_root / "poses"
    )
    for directory in (rgb_dir, depth_dir, pose_dir):
        directory.mkdir(parents=True, exist_ok=True)
    stem = f"{frame_index:08d}"
    Image.fromarray(rgb, mode="RGB").save(rgb_dir / f"{stem}.jpg", quality=95)
    if not cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_mm):
        raise OSError(f"Could not write depth frame {stem}")
    np.save(pose_dir / f"{stem}.npy", pose)
    manifest_path = scene_root / "recording_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as manifest:
        manifest.write(json.dumps({"frame_index": int(frame_index), "timestamp_ms": float(timestamp_ms)}) + "\n")


def _get_intrinsics(profile: Any, rs: Any) -> tuple[dict[str, float], int, int]:
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    camera = color_profile.get_intrinsics()
    return (
        {
            "fx": float(camera.fx),
            "fy": float(camera.fy),
            "cx": float(camera.ppx),
            "cy": float(camera.ppy),
        },
        int(camera.width),
        int(camera.height),
    )


def _parse_reference_transform(payload: bytes) -> np.ndarray:
    text = decode_string_cdr(payload)
    rotation_match = re.search(r"(?:^|;)rotation=([^;]+)", text)
    translation_match = re.search(r"(?:^|;)translation=([^;]+)", text)
    if not rotation_match or not translation_match:
        raise ValueError("RealSense tf/ref_0 metadata lacks rotation or translation")
    rotation = np.fromstring(rotation_match.group(1), sep=",")
    translation = np.fromstring(translation_match.group(1), sep=",")
    if rotation.size != 9 or translation.size != 3 or not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise ValueError("RealSense tf/ref_0 metadata has invalid transform values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.reshape(3, 3)
    transform[:3, 3] = translation
    return transform


def _archive_metadata(reader: NumpyArchiveReader) -> tuple[dict, dict, np.ndarray, float, dict[int, tuple[int, float]]]:
    image_topics = reader.manifest.get("stream_info", {}).get("image_topics", {})
    if not isinstance(image_topics, dict) or not {"rgb", "depth"}.issubset(image_topics):
        raise RuntimeError("archive manifest does not identify RGB and depth image topics")
    rgb_data_topic = image_topics["rgb"]
    depth_data_topic = image_topics["depth"]
    rgb_info_topic = rgb_data_topic.split("/image/data/", 1)[0] + "/image/camera_info"
    depth_info_topic = depth_data_topic.split("/image/data/", 1)[0] + "/image/camera_info"
    rgb_sensor = rgb_info_topic.rsplit("/image/camera_info", 1)[0]
    depth_sensor = depth_info_topic.rsplit("/image/camera_info", 1)[0]
    depth_scale_topic = depth_sensor.rsplit("/", 1)[0] + "/option/Depth_Units/value"
    color_metadata_topic = rgb_sensor + "/image/metadata"
    required_topics = {
        rgb_info_topic, depth_info_topic, rgb_sensor + "/tf/ref_0",
        depth_sensor + "/tf/ref_0", depth_scale_topic,
    }
    selected = {}
    frame_metadata = {}
    for message in reader.iter_metadata():
        if message.topic_name in required_topics and message.topic_name not in selected:
            selected[message.topic_name] = message.payload
        if message.topic_name == color_metadata_topic:
            try:
                text = decode_string_cdr(message.payload)
            except ValueError:
                continue
            frame_number = re.search(r"(?:^|;\s*)Frame number=(\d+)", text)
            timestamp = re.search(r"(?:^|;\s*)timestamp=([0-9]+(?:\.[0-9]+)?)", text)
            if frame_number and timestamp:
                frame_metadata[message.timestamp_ns] = (int(frame_number.group(1)), float(timestamp.group(1)))
    missing = required_topics.difference(selected)
    if missing:
        raise RuntimeError(f"required calibration metadata is missing: {', '.join(sorted(missing))}")
    rgb_cal = decode_camera_info_cdr(selected[rgb_info_topic])
    depth_cal = decode_camera_info_cdr(selected[depth_info_topic])
    color_to_ref = _parse_reference_transform(selected[rgb_sensor + "/tf/ref_0"])
    depth_to_ref = _parse_reference_transform(selected[depth_sensor + "/tf/ref_0"])
    try:
        depth_scale_m = float(decode_string_cdr(selected[depth_scale_topic]))
    except ValueError as exc:
        raise RuntimeError(f"invalid depth scale metadata: {depth_scale_topic}") from exc
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise RuntimeError(f"invalid depth scale metadata: {depth_scale_topic}")

    # RealSense records each sensor pose in its common reference frame.
    depth_to_color = np.linalg.inv(color_to_ref) @ depth_to_ref
    if not frame_metadata:
        logger.warning("Archive has no usable color image metadata; still poses will use archive timestamps")
    return rgb_cal, depth_cal, depth_to_color, depth_scale_m, frame_metadata


def _prepare_archive(args: argparse.Namespace, scene_root: Path) -> int:
    archive_path = Path(args.archive).expanduser().resolve()
    if not archive_path.is_dir():
        raise FileNotFoundError(f"RealSense archive directory does not exist: {archive_path}")
    reader = NumpyArchiveReader(archive_path)
    color_cal, depth_cal, depth_to_color, depth_scale_m, frame_metadata = _archive_metadata(reader)
    width, height = int(color_cal["width"]), int(color_cal["height"])
    intrinsics = {
        "fx": float(color_cal["k"][0, 0]), "fy": float(color_cal["k"][1, 1]),
        "cx": float(color_cal["k"][0, 2]), "cy": float(color_cal["k"][1, 2]),
    }
    odometry = RGBDOdometry(
        intrinsics=intrinsics,
        device=args.odometry_device,
        max_depth=args.odometry_max_depth,
        min_fitness=args.odometry_min_fitness,
    )
    pairs = pair_rgb_depth_frames(
        reader.iter_images("rgb"), reader.iter_images("depth"),
        max_delta_ns=getattr(args, "pair_tolerance_ms", 20.0) * 1_000_000,
    )
    frame_index = candidate_count = 0
    pose_samples = []
    candidate_periods = []
    pending_period = []
    scratch = tempfile.TemporaryDirectory(prefix="conceptgraph-realsense-frames-")
    scratch_dir = Path(scratch.name)
    started_at = time.monotonic()
    last_report = started_at
    for color_frame, depth_frame in pairs:
        rgb = color_frame.frame
        depth_aligned = align_depth_to_color(
            depth_frame.frame, depth_cal, color_cal, depth_to_color,
            depth_scale_m=depth_scale_m,
        )
        if rgb.shape[:2] != (height, width) or depth_aligned.shape != (height, width):
            raise RuntimeError(
                f"Archive frame dimensions disagree with CameraInfo at pair {frame_index}: "
                f"RGB={rgb.shape[:2]}, depth={depth_aligned.shape}, expected={(height, width)}"
            )
        depth_m = depth_aligned.astype(np.float32) * depth_scale_m
        pose = odometry.estimate(rgb[:, :, ::-1], depth_m)
        sensor_frame, timestamp_ms = frame_metadata.get(
            color_frame.timestamp_ns,
            (int(color_frame.frame_number), color_frame.timestamp_ns / 1_000_000.0),
        )
        pose_samples.append({
            "frame_number": sensor_frame,
            "timestamp_ms": timestamp_ms,
            "camera_pose": np.asarray(pose, dtype=np.float32).tolist(),
        })
        depth_mm = np.clip(np.rint(depth_m * 1000.0), 0, 65535).astype(np.uint16)
        pending_period.append({
            "source_index": frame_index,
            "frame_number": sensor_frame,
            "timestamp_ms": timestamp_ms,
            "score": get_blur_score(rgb),
            "rgb": rgb,
            "depth_mm": depth_mm,
            "pose": np.asarray(pose, dtype=np.float32),
        })
        if len(pending_period) == args.sample_every:
            staged = _stage_frame_period(
                pending_period, len(candidate_periods), scratch_dir,
                args.blur_tolerance_percent / 100.0,
                args.sample_offset, args.sample_every,
            )
            candidate_count += len(staged)
            candidate_periods.append(staged)
            pending_period.clear()
        frame_index += 1
        now = time.monotonic()
        if now - last_report >= 2.0:
            logger.info(
                "Processed %d archived RGB-D pairs (%.1f FPS), retained %d sharp candidates",
                frame_index, frame_index / max(now - started_at, 1e-6), candidate_count,
            )
            last_report = now
    if frame_index == 0:
        raise RuntimeError(f"No timestamp-matched RGB-D pairs were read from archive: {archive_path}")
    if pending_period:
        staged = _stage_frame_period(
            pending_period, len(candidate_periods), scratch_dir,
            args.blur_tolerance_percent / 100.0,
            args.sample_offset, args.sample_every,
        )
        candidate_count += len(staged)
        candidate_periods.append(staged)
    selected = _select_spaced_candidates(candidate_periods)
    _write_selected_candidates(selected, args.dataset_root, args.scene_id)
    sampled = len(selected)
    scratch.cleanup()
    write_dataset_config(
        scene_root, width, height, intrinsics, depth_scale=1.0 / depth_scale_m,
    )
    capture_session = getattr(args, "capture_session", None)
    if capture_session:
        _update_capture_session_poses(
            Path(capture_session).expanduser().resolve(), archive_path.name,
            pose_samples, getattr(args, "snapshot_pose_tolerance_ms", 50.0),
        )
    logger.info(
        "Prepared %d archived RGB-D pairs into %d sharp, spaced samples from %d-frame periods "
        "(offset %d, %.1f%% blur tolerance); dataset=%s",
        frame_index, sampled, args.sample_every, args.sample_offset,
        args.blur_tolerance_percent, scene_root,
    )
    return 0


def prepare(args: argparse.Namespace) -> int:
    if not hasattr(args, "blur_tolerance_percent"):
        args.blur_tolerance_percent = 2.5
    if not args.scene_id or Path(args.scene_id).name != args.scene_id:
        raise ValueError("scene-id must be a single directory name")
    scene_root = Path(args.dataset_root).expanduser().resolve() / args.scene_id
    if scene_root.exists() and any(scene_root.iterdir()):
        raise FileExistsError(
            f"Dataset scene already contains files: {scene_root}; choose a new scene_id"
        )
    if args.sample_every <= 0:
        raise ValueError("sample-every must be positive")
    if not 0 <= args.sample_offset < args.sample_every:
        raise ValueError("sample-offset must satisfy 0 <= offset < sample-every")
    if not 0 <= args.blur_tolerance_percent <= 100:
        raise ValueError("blur-tolerance-percent must be between 0 and 100")

    if getattr(args, "archive", None):
        return _prepare_archive(args, scene_root)

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"RealSense recording does not exist: {bag_path}")
    if bag_path.suffix.lower() not in {".db3", ".bag"}:
        raise ValueError("Input must be a librealsense .db3 recording or legacy .bag file")
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "RealSense bag replay needs the optional dependency: uv sync --extra realsense"
        ) from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)
    started = False
    frame_index = 0
    candidate_count = 0
    pose_samples: list[dict[str, Any]] = []
    candidate_periods = []
    pending_period = []
    scratch = tempfile.TemporaryDirectory(prefix="conceptgraph-realsense-frames-")
    scratch_dir = Path(scratch.name)
    started_at = time.monotonic()
    last_report = started_at
    try:
        profile = pipeline.start(config)
        started = True
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        align = rs.align(rs.stream.color)
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        intrinsics, width, height = _get_intrinsics(profile, rs)
        odometry = RGBDOdometry(
            intrinsics=intrinsics,
            device=args.odometry_device,
            max_depth=args.odometry_max_depth,
            min_fitness=args.odometry_min_fitness,
        )

        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                if playback.current_status() == rs.playback_status.stopped:
                    break
                raise
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError(f"Missing aligned color/depth at bag frame {frame_index}")

            rgb = np.asanyarray(color_frame.get_data()).copy()
            depth_raw = np.asanyarray(depth_frame.get_data()).copy()
            if rgb.shape[:2] != (height, width) or depth_raw.shape != (height, width):
                raise RuntimeError(
                    f"Aligned stream dimensions changed at frame {frame_index}: "
                    f"color={rgb.shape[:2]}, depth={depth_raw.shape}, expected={(height, width)}"
                )
            depth_m = depth_to_meters(depth_raw, depth_scale)
            pose = odometry.estimate(rgb[:, :, ::-1], depth_m)
            pose_samples.append(
                {
                    "frame_number": frame_index,
                    "timestamp_ms": float(color_frame.get_timestamp()),
                    "camera_pose": np.asarray(pose, dtype=np.float32).tolist(),
                }
            )
            depth_mm = np.clip(np.rint(depth_m * 1000.0), 0, 65535).astype(np.uint16)
            pending_period.append({
                "source_index": frame_index,
                "frame_number": frame_index,
                "timestamp_ms": float(color_frame.get_timestamp()),
                "score": get_blur_score(rgb),
                "rgb": rgb,
                "depth_mm": depth_mm,
                "pose": np.asarray(pose, dtype=np.float32),
            })
            if len(pending_period) == args.sample_every:
                staged = _stage_frame_period(
                    pending_period, len(candidate_periods), scratch_dir,
                    args.blur_tolerance_percent / 100.0,
                    args.sample_offset, args.sample_every,
                )
                candidate_count += len(staged)
                candidate_periods.append(staged)
                pending_period.clear()

            frame_index += 1
            now = time.monotonic()
            if now - last_report >= 2.0:
                logger.info(
                    "Replayed %d frames (%.1f FPS), retained %d sharp candidates",
                    frame_index, frame_index / max(now - started_at, 1e-6), candidate_count,
                )
                last_report = now

        if frame_index == 0:
            raise RuntimeError(f"No frames were read from recording: {bag_path}")
        if pending_period:
            staged = _stage_frame_period(
                pending_period, len(candidate_periods), scratch_dir,
                args.blur_tolerance_percent / 100.0,
                args.sample_offset, args.sample_every,
            )
            candidate_count += len(staged)
            candidate_periods.append(staged)
        selected = _select_spaced_candidates(candidate_periods)
        _write_selected_candidates(selected, args.dataset_root, args.scene_id)
        sampled = len(selected)
        scratch.cleanup()
        write_dataset_config(scene_root, width, height, intrinsics)
        capture_session = getattr(args, "capture_session", None)
        if capture_session:
            _update_capture_session_poses(
                Path(capture_session).expanduser().resolve(),
                bag_path.name,
                pose_samples,
                getattr(args, "snapshot_pose_tolerance_ms", 50.0),
            )
    finally:
        if started:
            pipeline.stop()

    logger.info(
        "Prepared %d source frames into %d sharp, spaced samples from %d-frame periods "
        "(offset %d, %.1f%% blur tolerance); dataset=%s",
        frame_index, sampled, args.sample_every, args.sample_offset,
        args.blur_tolerance_percent, scene_root,
    )
    logger.info(
        "Map with: python -m conceptgraph.slam.rerun_realtime_mapping "
        "--config-name rerun_realtime_mapping dataset_root=%s scene_id=%s "
        "dataset_config=%s stride=1",
        scene_root.parent, args.scene_id, scene_root / "dataconfig.yaml",
    )
    return 0


def _update_capture_session_poses(
    session_path: Path,
    bag_name: str,
    pose_samples: list[dict[str, Any]],
    tolerance_ms: float,
) -> None:
    if not session_path.is_file():
        logger.warning(
            "Capture session metadata does not exist: %s; prepared dataset is complete, "
            "but still-image pose association was skipped",
            session_path,
        )
        return
    session = json.loads(session_path.read_text(encoding="utf-8"))
    stills = session.get("still_images", [])
    recording_ids = {
        recording.get("recording_id")
        for recording in session.get("recordings", [])
        if recording.get("archive") == bag_name
    }
    selected_indexes = [
        index for index, still in enumerate(stills)
        if still.get("archive") == bag_name
        or still.get("bag") == bag_name
        or (still.get("recording_id") is not None and still.get("recording_id") in recording_ids)
    ]
    if not selected_indexes:
        logger.info("No still images in %s refer to bag %s", session_path, bag_name)
        return
    selected = [stills[index] for index in selected_indexes]
    associated = associate_still_captures(selected, pose_samples, tolerance_ms)
    for index, updated in zip(selected_indexes, associated):
        stills[index] = updated
        metadata_name = updated.get("metadata")
        if metadata_name:
            sidecar_path = session_path.parent / metadata_name
            sidecar_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    session["still_images"] = stills
    session_path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
    matched = sum(not still.get("pose_pending", True) for still in associated)
    logger.info(
        "Attached odometry poses to %d/%d still images (tolerance %.1f ms)",
        matched, len(associated), tolerance_ms,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", help="Verified RealSense archive directory")
    source.add_argument("--bag", help="Input RealSense .db3 or legacy .bag recording")
    parser.add_argument("--dataset-root", required=True, help="Root folder for extracted dataset")
    parser.add_argument("--scene-id", required=True, help="New dataset scene directory name")
    parser.add_argument(
        "--sample-every", type=int, default=15,
        help="Group input into fixed periods of N frames and export one frame per period",
    )
    parser.add_argument(
        "--sample-offset", type=int, default=0,
        help="Tie-break preference within each period, in [0, sample-every)",
    )
    parser.add_argument(
        "--blur-tolerance-percent", type=float, default=2.5,
        help="Keep period frames within this percentage of the sharpest Laplacian score",
    )
    parser.add_argument("--odometry-device", default="CUDA:0", help="Open3D device; defaults to CUDA:0")
    parser.add_argument("--odometry-max-depth", type=float, default=4.0)
    parser.add_argument("--odometry-min-fitness", type=float, default=0.25)
    parser.add_argument(
        "--pair-tolerance-ms", type=float, default=20.0,
        help="Maximum RGB/depth timestamp difference for archive pairing",
    )
    parser.add_argument(
        "--capture-session",
        help="Optional capture_session.json; matching still-image sidecars receive odometry poses",
    )
    parser.add_argument(
        "--snapshot-pose-tolerance-ms", type=float, default=50.0,
        help="Maximum sensor timestamp difference allowed for still-pose association",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.sample_every <= 0:
        raise SystemExit("sample-every must be positive")
    if not 0 <= args.sample_offset < args.sample_every:
        raise SystemExit("sample-offset must satisfy 0 <= offset < sample-every")
    if args.odometry_max_depth <= 0:
        raise SystemExit("odometry-max-depth must be positive")
    if not 0 <= args.odometry_min_fitness <= 1:
        raise SystemExit("odometry-min-fitness must be between zero and one")
    if args.snapshot_pose_tolerance_ms < 0:
        raise SystemExit("snapshot-pose-tolerance-ms must be non-negative")
    if not 0 <= args.blur_tolerance_percent <= 100:
        raise SystemExit("blur-tolerance-percent must be between 0 and 100")
    if args.pair_tolerance_ms < 0:
        raise SystemExit("pair-tolerance-ms must be non-negative")
    return prepare(args)


if __name__ == "__main__":
    raise SystemExit(main())
