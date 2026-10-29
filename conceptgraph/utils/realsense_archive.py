"""Lossless NumPy archive writer for RealSense SDK ``.db3`` recordings.

This module exports rosbag2 image and metadata streams to chunked NumPy archives.
RGB chunks retain the source compressed image bytes with frame offsets; depth
arrays and metadata payloads are compressed NPZ members.

The writer uses ``numpy.savez`` for already-compressed RGB payloads, avoiding
redundant DEFLATE work. Other chunks use ``numpy.savez_compressed``. All members
can be read with ``allow_pickle=False``. Chunks and the manifest are installed
via a temporary sibling file plus atomic rename: on a filesystem ``OSError``
the partial temporary file is removed, buffered rows and chunk counters are
retained, and ``finalize`` can be retried without losing data or duplicating
chunk indices.

The source DB3 is opened read-only. The exporter verifies saved payloads and
metadata against that source before publishing a verified manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import sqlite3
import struct
import subprocess
import uuid
import zipfile
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

__all__ = [
    "DecodedCompressedImage",
    "ArchivedImageFrame",
    "ArchivedEncodedImage",
    "ArchivedMetadataMessage",
    "NumpyArchiveWriter",
    "NumpyArchiveReader",
    "decode_compressed_image_cdr",
    "decode_camera_info_cdr",
    "decode_string_cdr",
    "pair_rgb_depth_frames",
    "align_depth_to_color",
    "export_recording",
    "verify_archive",
]


class DecodedCompressedImage(NamedTuple):
    frame: np.ndarray
    header_timestamp_ns: int
    frame_id: str
    format: str
    encoded_payload: bytes = b""


@dataclass(frozen=True)
class ArchivedImageFrame:
    """One already-decoded image array and its source DB3 timing fields."""

    frame: np.ndarray
    frame_number: int
    timestamp_ns: int
    header_timestamp_ns: int
    frame_id: str
    source_payload_sha256: str


@dataclass(frozen=True)
class ArchivedEncodedImage:
    """One original compressed RGB image payload and its source timing fields."""

    payload: bytes
    frame_number: int
    timestamp_ns: int
    header_timestamp_ns: int
    frame_id: str
    format: str
    source_payload_sha256: str


@dataclass(frozen=True)
class ArchivedMetadataMessage:
    """One raw CDR metadata payload from a compressed archive chunk."""

    topic_name: str
    topic_type: str
    timestamp_ns: int
    payload: bytes


class _CdrReader:
    """Read the primitive fields used by ROS 2 CompressedImage CDR."""

    def __init__(self, payload: bytes, *, alignment_base: int = 0) -> None:
        if len(payload) < 4:
            raise ValueError("truncated CDR encapsulation")
        representation = payload[:2]
        if representation == b"\x00\x01":
            self.endian = "<"
        elif representation == b"\x00\x00":
            self.endian = ">"
        else:
            raise ValueError(f"unsupported CDR encapsulation: {representation!r}")
        self.data = memoryview(payload)[4:]
        self.offset = 0
        self.alignment_base = int(alignment_base)

    def _take(self, size: int, alignment: int = 1) -> bytes:
        aligned_offset = self.offset + self.alignment_base
        start = self.offset + (-aligned_offset % alignment)
        end = start + size
        if end > len(self.data):
            raise ValueError("truncated CDR payload")
        self.offset = end
        return self.data[start:end].tobytes()

    def int32(self) -> int:
        return struct.unpack(self.endian + "i", self._take(4, 4))[0]

    def uint32(self) -> int:
        return struct.unpack(self.endian + "I", self._take(4, 4))[0]

    def cdr_string(self) -> str:
        length = self.uint32()
        if length < 1:
            raise ValueError("CDR strings must include a terminating NUL")
        raw = self._take(length)
        if raw[-1:] != b"\0":
            raise ValueError("CDR string is missing its terminating NUL")
        try:
            return raw[:-1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("CDR string is not valid UTF-8") from exc

    def byte_sequence(self) -> bytes:
        return self._take(self.uint32())

    def require_end(self) -> None:
        if self.offset != len(self.data):
            raise ValueError("trailing bytes after CDR CompressedImage payload")


def decode_camera_info_cdr(payload: bytes | bytearray | memoryview) -> dict:
    """Decode the calibration fields from a ROS 2 ``sensor_msgs/CameraInfo``."""
    failures = []
    for alignment_base in (0, 4):
        try:
            reader = _CdrReader(bytes(payload), alignment_base=alignment_base)
            reader.int32()  # header stamp seconds
            nanosec = reader.uint32()
            if nanosec >= 1_000_000_000:
                raise ValueError("invalid CameraInfo timestamp nanoseconds")
            frame_id = reader.cdr_string()
            height, width = reader.uint32(), reader.uint32()
            distortion_model = reader.cdr_string()

            def doubles(count: int | None = None) -> np.ndarray:
                length = reader.uint32() if count is None else count
                if length > 4096:
                    raise ValueError("CameraInfo array is unreasonably large")
                values = [
                    struct.unpack(reader.endian + "d", reader._take(8, 8))[0]
                    for _ in range(length)
                ]
                return np.asarray(values, dtype=np.float64)

            d = doubles()
            k = doubles(9).reshape(3, 3)
            r = doubles(9).reshape(3, 3)
            p = doubles(12).reshape(3, 4)
            reader.uint32()  # binning_x
            reader.uint32()  # binning_y
            for _ in range(4):
                reader.uint32()  # ROI x_offset, y_offset, height, width
            reader._take(1)  # ROI do_rectify
            reader.require_end()
            if width <= 0 or height <= 0 or not np.all(np.isfinite(k)) or not np.all(np.isfinite(d)):
                raise ValueError("CameraInfo has invalid dimensions or calibration values")
            if abs(float(k[0, 0])) < 1e-12 or abs(float(k[1, 1])) < 1e-12:
                raise ValueError("CameraInfo focal lengths must be non-zero")
            return {
                "frame_id": frame_id,
                "width": int(width),
                "height": int(height),
                "distortion_model": distortion_model,
                "d": d,
                "k": k,
                "r": r,
                "p": p,
            }
        except (ValueError, struct.error) as exc:
            failures.append(exc)
    raise ValueError(f"could not decode CameraInfo CDR payload: {failures[-1]}")


def decode_string_cdr(payload: bytes | bytearray | memoryview) -> str:
    """Decode one ROS 2 ``std_msgs/String`` CDR payload."""
    reader = _CdrReader(bytes(payload))
    value = reader.cdr_string()
    reader.require_end()
    return value


def pair_rgb_depth_frames(rgb_frames, depth_frames, *, max_delta_ns: int = 20_000_000):
    """Yield nearest one-to-one timestamp pairs from ordered RGB and depth streams."""
    if max_delta_ns < 0:
        raise ValueError("max_delta_ns must be non-negative")
    def timestamp(record):
        return int(record.timestamp_ns if hasattr(record, "timestamp_ns") else record["timestamp_ns"])

    rgb_iter, depth_iter = iter(rgb_frames), iter(depth_frames)
    rgb, depth = next(rgb_iter, None), next(depth_iter, None)
    previous_depth = None
    while rgb is not None:
        rgb_ts = timestamp(rgb)
        while depth is not None and timestamp(depth) < rgb_ts:
            previous_depth, depth = depth, next(depth_iter, None)
        candidates = [candidate for candidate in (previous_depth, depth) if candidate is not None]
        if candidates:
            candidate = min(candidates, key=lambda item: abs(timestamp(item) - rgb_ts))
            delta = abs(timestamp(candidate) - rgb_ts)
            if delta <= max_delta_ns:
                yield rgb, candidate
                if candidate is previous_depth:
                    previous_depth = None
                else:
                    depth = next(depth_iter, None)
                    previous_depth = None
        rgb = next(rgb_iter, None)


def align_depth_to_color(
    depth_raw: np.ndarray,
    depth_calibration: dict,
    color_calibration: dict,
    depth_to_color_transform: np.ndarray,
    *,
    depth_scale_m: float = 0.001,
) -> np.ndarray:
    """Project raw Z16 depth into the color camera, resolving collisions by nearest Z."""
    depth = np.asarray(depth_raw)
    transform = np.asarray(depth_to_color_transform, dtype=np.float64)
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError("depth_raw must be a 2D uint16 array")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError("depth_scale_m must be finite and positive")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("depth-to-color transform must be a finite 4x4 matrix")
    dk = np.asarray(depth_calibration.get("k"), dtype=np.float64)
    ck = np.asarray(color_calibration.get("k"), dtype=np.float64)
    d_dist = np.asarray(depth_calibration.get("d", []), dtype=np.float64)
    c_dist = np.asarray(color_calibration.get("d", []), dtype=np.float64)
    width, height = int(color_calibration.get("width", 0)), int(color_calibration.get("height", 0))
    if dk.shape != (3, 3) or ck.shape != (3, 3) or width <= 0 or height <= 0:
        raise ValueError("camera calibration has invalid intrinsics or dimensions")
    if abs(dk[0, 0]) < 1e-12 or abs(dk[1, 1]) < 1e-12 or abs(ck[0, 0]) < 1e-12 or abs(ck[1, 1]) < 1e-12:
        raise ValueError("camera focal lengths must be non-zero")
    source_h, source_w = depth.shape
    if depth_calibration.get("width") != source_w or depth_calibration.get("height") != source_h:
        raise ValueError("depth dimensions do not match depth CameraInfo")

    y, x = np.indices(depth.shape, dtype=np.float64)
    pixels = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(pixels, dk, d_dist if d_dist.size else None).reshape(-1, 2)
    z = depth.reshape(-1).astype(np.float64) * depth_scale_m
    valid = z > 0
    points = np.column_stack((normalized[valid, 0] * z[valid], normalized[valid, 1] * z[valid], z[valid]))
    if not len(points):
        return np.zeros((height, width), dtype=np.uint16)
    points_color = points @ transform[:3, :3].T + transform[:3, 3]
    valid_color = points_color[:, 2] > 0
    points_color = points_color[valid_color]
    if not len(points_color):
        return np.zeros((height, width), dtype=np.uint16)
    projected, _ = cv2.projectPoints(
        points_color.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), ck,
        c_dist if c_dist.size else None,
    )
    uv = np.rint(projected.reshape(-1, 2)).astype(np.int64)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, points_color = uv[inside], points_color[inside]
    flat_index = uv[:, 1] * width + uv[:, 0]
    z_buffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(z_buffer, flat_index, points_color[:, 2])
    aligned = np.zeros(height * width, dtype=np.uint16)
    hit = np.isfinite(z_buffer)
    aligned[hit] = np.clip(np.rint(z_buffer[hit] / depth_scale_m), 1, 65535).astype(np.uint16)
    return aligned.reshape(height, width)


def decode_compressed_image_cdr(
    payload: bytes | bytearray | memoryview, stream: str
) -> DecodedCompressedImage:
    """Decode one ROS 2 CDR CompressedImage payload to RGB or raw Z16."""
    if stream not in _IMAGE_STREAMS:
        raise ValueError("stream must be 'rgb' or 'depth'")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("payload must be bytes-like")

    reader = _CdrReader(bytes(payload))
    sec = reader.int32()
    nanosec = reader.uint32()
    if not 0 <= nanosec < 1_000_000_000:
        raise ValueError(f"invalid ROS timestamp nanoseconds: {nanosec}")
    frame_id = reader.cdr_string()
    image_format = reader.cdr_string()
    encoded = reader.byte_sequence()
    reader.require_end()
    header_timestamp_ns = sec * 1_000_000_000 + nanosec

    if stream == _RGB:
        normalized_format = image_format.lower()
        if normalized_format not in {
            "jpeg",
            "jpg",
            "png",
            "rgb8; jpeg compressed bgr8",
            "rgb8; png compressed bgr8",
            "bgr8; jpeg compressed bgr8",
            "bgr8; png compressed bgr8",
        }:
            raise ValueError(f"unsupported RGB CompressedImage format: {image_format!r}")
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("could not decode RGB JPEG image data")
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        if image_format != "16UC1; compressedDepth png":
            raise ValueError(f"unsupported depth CompressedImage format: {image_format!r}")
        if len(encoded) <= 12:
            raise ValueError("compressedDepth data is missing its 12-byte config header or PNG")
        frame = cv2.imdecode(
            np.frombuffer(encoded[12:], dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if frame is None or frame.ndim != 2 or frame.dtype != np.uint16:
            raise ValueError("compressedDepth payload did not decode to uint16 pixels")

    return DecodedCompressedImage(
        frame, header_timestamp_ns, frame_id, image_format, encoded
    )

# Image streams recognized by the writer and their required dtype/shape rules.
_RGB = "rgb"
_DEPTH = "depth"
_IMAGE_STREAMS = (_RGB, _DEPTH)
_IMAGE_DTYPES = {_RGB: np.dtype(np.uint8), _DEPTH: np.dtype(np.uint16)}


# Signed int64 bounds: every integer field must fit this range.
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _coerce_int_ns(value, name: str) -> int:
    """Coerce an integer nanosecond/sequence field to a Python ``int``.

    Only Python integers and NumPy integer scalars are accepted.  Booleans and
    floats (including NumPy floating scalars) are rejected with ``ValueError``.
    The resulting value must fit signed int64 or ``ValueError`` is raised.  All
    rejection happens before any writer-state mutation so a bad value cannot
    affect ordering, counts, or summaries.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer; got bool")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer; got {type(value).__name__}")
    result = int(value)
    if result < _INT64_MIN or result > _INT64_MAX:
        raise ValueError(
            f"{name} must fit signed int64 [{_INT64_MIN}, {_INT64_MAX}]; got {result}"
        )
    return result


def _require_json_serializable(value, name: str):
    """Return a JSON-canonical copy of ``value`` or raise ``ValueError``.

    ``finalize`` calls this before flushing any chunks so that a non-JSON-safe
    ``source_info``/``stream_info`` descriptor fails fast, leaving the writer
    unmutated and still usable (no partial flush, no terminal state).
    """
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not JSON serializable: {exc}") from exc


class _ImageStreamState:
    """Internal, per-stream buffering state for one image stream."""

    __slots__ = (
        "stream",
        "frame_shape",
        "last_frame_number",
        "last_timestamp_ns",
        "encoded_payload",
        "rows",
        "chunks_written",
    )

    def __init__(self, stream: str, frame_shape: tuple[int, ...]) -> None:
        self.stream = stream
        self.frame_shape = frame_shape
        self.last_frame_number: int | None = None
        self.last_timestamp_ns: int | None = None
        self.encoded_payload: bool | None = None
        # Each row is a compact tuple:
        # (frame, frame_number, timestamp_ns, header_timestamp_ns,
        #  frame_id, source_payload_sha256)
        self.rows: list[tuple] = []
        self.chunks_written = 0


class NumpyArchiveWriter:
    """Buffer DB3 messages and image frames into chunked NumPy archives.

    Parameters
    ----------
    output_dir:
        Destination directory for the archive.  It must either not exist or be
        empty.  Sub-directories (``metadata``, ``rgb``, ``depth``) are created
        as chunks are flushed.
    frames_per_chunk:
        Number of records buffered before a chunk is flushed.  Must be a
        positive integer.

    Raises
    ------
    ValueError
        If ``frames_per_chunk`` is not positive or the destination directory
        already exists and is non-empty.
    """

    def __init__(self, output_dir, *, frames_per_chunk: int = 30) -> None:
        self._output_dir = Path(output_dir)
        if not isinstance(frames_per_chunk, int) or isinstance(frames_per_chunk, bool):
            raise ValueError("frames_per_chunk must be a positive integer")
        if frames_per_chunk < 1:
            raise ValueError("frames_per_chunk must be a positive integer")
        self._frames_per_chunk = frames_per_chunk

        if self._output_dir.exists() and any(self._output_dir.iterdir()):
            raise ValueError(
                f"destination must not exist or be empty: {self._output_dir}"
            )

        self._finalized = False
        # Buffer of pending metadata rows: (name, type, timestamp_ns, payload)
        self._metadata_rows: list[tuple[str, str, int, bytes]] = []
        self._metadata_chunks_written = 0
        self._image_streams: dict[str, _ImageStreamState] = {}
        # Per-topic running stats for the manifest:
        # name -> {"message_count", "first_timestamp_ns", "last_timestamp_ns"}.
        self._metadata_stats: dict[str, dict] = {}
        # Absolute paths of every flushed chunk file, in write order, so
        # finalize can record their SHA-256 digests.
        self._chunk_paths: list[str] = []

    # ----------------------------------------------------------------- #
    # Metadata API
    # ----------------------------------------------------------------- #

    def append_metadata(
        self,
        topic_name: str,
        topic_type: str,
        timestamp_ns: int,
        payload: bytes,
    ) -> None:
        """Append one non-image DB3 message to the metadata stream.

        ``payload`` bytes, ``topic_name``/``topic_type`` strings, and a
        ``timestamp_ns`` are preserved in DB3 append order. Timestamps may
        regress because some RealSense static metadata topics use zero stamps.
        An invalid argument
        raises ``ValueError`` without recording the row (a failed append does
        not poison the writer).

        Raises
        ------
        ValueError
            If any argument is invalid or the timestamp decreases.
        RuntimeError
            If the writer has already been finalized.
        """
        if self._finalized:
            raise RuntimeError("writer is finalized")
        if not isinstance(topic_name, str) or topic_name == "":
            raise ValueError("topic_name must be a non-empty string")
        if not isinstance(topic_type, str) or topic_type == "":
            raise ValueError("topic_type must be a non-empty string")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError("payload must be bytes")

        # Strict integer validation happens before any writer state changes so
        # that a rejected row cannot affect ordering, counts, or summaries.
        ts = _coerce_int_ns(timestamp_ns, "timestamp_ns")
        # All validations passed: record the row (payload normalized to bytes).
        self._metadata_rows.append((topic_name, topic_type, ts, bytes(payload)))

        stats = self._metadata_stats.get(topic_name)
        if stats is None:
            self._metadata_stats[topic_name] = {
                "message_count": 1,
                "first_timestamp_ns": ts,
                "last_timestamp_ns": ts,
            }
        else:
            stats["message_count"] += 1
            stats["last_timestamp_ns"] = ts

        if len(self._metadata_rows) >= self._frames_per_chunk:
            self._flush_metadata()

    # ----------------------------------------------------------------- #
    # Image API
    # ----------------------------------------------------------------- #

    def append_image(
        self,
        stream: str,
        frame,
        *,
        frame_number: int,
        timestamp_ns: int,
        header_timestamp_ns: int,
        frame_id,
        source_payload: bytes,
        encoded_payload: bytes | None = None,
        encoded_format: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Buffer one decoded RGB or depth frame for a given stream.

        ``stream`` must be exactly ``"rgb"`` or ``"depth"``.  RGB frames are
        RGB-order ``uint8`` arrays of shape ``(H, W, 3)``; depth frames are raw
        Z16 ``uint16`` arrays of shape ``(H, W)``.  Shape and dtype are fixed by
        the first frame of each stream and must stay consistent.  Per-stream
        ``frame_number`` and ``timestamp_ns`` must be strictly increasing
        (independent per stream); a rejected frame does not mutate writer state.

        ``source_payload`` is the original (CDR) payload for the frame; its
        SHA-256 digest is stored. If ``encoded_payload`` and ``encoded_format``
        are provided for RGB, those exact compressed image bytes are stored in a
        packed, uncompressed NPZ member with offsets. Otherwise decoded arrays
        use the legacy layout.
        ``metadata`` is an optional dict (reserved).

        Raises
        ------
        ValueError
            For an unknown stream, invalid dtype/shape, or invalid arguments.
        RuntimeError
            If the writer has already been finalized.
        """
        if self._finalized:
            raise RuntimeError("writer is finalized")
        if stream not in _IMAGE_STREAMS:
            raise ValueError("stream must be 'rgb' or 'depth'")

        arr = np.asarray(frame)
        want_dtype = _IMAGE_DTYPES[stream]
        if arr.dtype != want_dtype:
            raise ValueError(
                f"{stream} frame dtype must be {want_dtype.name}; got {arr.dtype.name}"
            )
        if stream == _RGB:
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"rgb frame must have shape (H, W, 3); got {arr.shape}")
        else:
            if arr.ndim != 2:
                raise ValueError(f"depth frame must have shape (H, W); got {arr.shape}")

        if not isinstance(source_payload, (bytes, bytearray, memoryview)):
            raise ValueError("source_payload must be bytes")
        if encoded_payload is not None:
            if stream != _RGB:
                raise ValueError("encoded_payload is supported only for RGB frames")
            if not isinstance(encoded_payload, (bytes, bytearray, memoryview)):
                raise ValueError("encoded_payload must be bytes-like")
            if not encoded_payload:
                raise ValueError("encoded_payload must not be empty")
            if not isinstance(encoded_format, str) or not encoded_format:
                raise ValueError("encoded_format must be a non-empty string for encoded RGB")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be None or a dict")

        # Strict integer validation happens before any writer state changes so
        # that a rejected frame cannot affect ordering or recorded rows.
        ts = _coerce_int_ns(timestamp_ns, "timestamp_ns")
        frame_number = _coerce_int_ns(frame_number, "frame_number")
        header_ts = _coerce_int_ns(header_timestamp_ns, "header_timestamp_ns")

        # All shape/ordering validation happens before any state mutation so a
        # rejected record cannot update last-seen values, counters, or buffers.
        state = self._image_streams.get(stream)
        if state is not None:
            if state.encoded_payload != (encoded_payload is not None):
                raise ValueError(
                    f"{stream} chunk cannot mix encoded payloads and decoded arrays"
                )
            if tuple(arr.shape) != state.frame_shape:
                raise ValueError(
                    f"{stream} frame shape {arr.shape} does not match first frame "
                    f"shape {state.frame_shape}"
                )
            if (
                state.last_frame_number is not None
                and frame_number <= state.last_frame_number
            ):
                raise ValueError(
                    f"{stream} frame_numbers must be strictly increasing; "
                    f"{frame_number} <= {state.last_frame_number}"
                )
            if state.last_timestamp_ns is not None and ts <= state.last_timestamp_ns:
                raise ValueError(
                    f"{stream} timestamps_ns must be strictly increasing per "
                    f"stream; {ts} <= {state.last_timestamp_ns}"
                )
        if state is None:
            state = _ImageStreamState(stream, tuple(arr.shape))
            state.encoded_payload = encoded_payload is not None
            self._image_streams[stream] = state

        state.last_frame_number = frame_number
        state.last_timestamp_ns = ts
        # Buffer a copy so later caller mutation cannot corrupt archived data.
        state.rows.append(
            (
                bytes(encoded_payload) if encoded_payload is not None else arr.copy(),
                frame_number,
                ts,
                header_ts,
                str(frame_id),
                hashlib.sha256(bytes(source_payload)).hexdigest(),
                str(encoded_format),
            )
        )
        if len(state.rows) >= self._frames_per_chunk:
            self._flush_image(state)

    # ----------------------------------------------------------------- #
    # Finalize
    # ----------------------------------------------------------------- #

    def finalize(self, source_info: dict, stream_info: dict) -> dict:
        """Flush all partial chunks and mark the writer as finalized.

        A second ``finalize`` call, or any append after finalize, raises
        ``RuntimeError``.  An explicitly unverified ``export_manifest.json`` is
        persisted in ``output_dir`` and the same JSON-safe dict is returned.

        ``OSError`` from a chunk or manifest write propagates without marking
        the writer finalized: buffered rows and chunk counters are retained and
        partial temporary files are removed, so calling ``finalize`` again
        retries the write without losing data or duplicating chunk indices.

        Returns
        -------
        dict
            The manifest.  ``verified`` is ``False`` because verification has
            not been performed.  ``source_info``/``stream_info`` are preserved,
            ``metadata_topics`` summarizes per-topic counts and first/last
            timestamps, and ``chunk_hashes`` maps each written chunk to its
            SHA-256 digest.
        """
        if self._finalized:
            raise RuntimeError("writer is already finalized")
        if not isinstance(source_info, dict):
            raise ValueError("source_info must be a dict")
        if not isinstance(stream_info, dict):
            raise ValueError("stream_info must be a dict")

        # Validate JSON serializability BEFORE flushing any chunk: a bad
        # descriptor must fail fast, leaving no partial flush and the writer
        # still usable.  The canonical copies are reused in the manifest below.
        source_info_json = _require_json_serializable(source_info, "source_info")
        stream_info_json = _require_json_serializable(stream_info, "stream_info")

        # A flush failure (OSError) propagates with the pending rows and chunk
        # counters preserved (see the flush helpers), so finalize can be
        # retried without losing data or duplicating chunk indices.
        self._flush_metadata()
        for state in self._image_streams.values():
            self._flush_image(state)

        chunk_hashes = {}
        for path in self._chunk_paths:
            chunk_path = Path(path)
            rel = chunk_path.relative_to(self._output_dir).as_posix()
            chunk_hashes[rel] = hashlib.sha256(chunk_path.read_bytes()).hexdigest()

        manifest = {
            "output_dir": str(self._output_dir),
            "verified": False,
            "source_info": source_info_json,
            "stream_info": stream_info_json,
            "metadata_chunks": self._metadata_chunks_written,
            "rgb_chunks": self._image_streams.get(_RGB).chunks_written
            if _RGB in self._image_streams
            else 0,
            "depth_chunks": self._image_streams.get(_DEPTH).chunks_written
            if _DEPTH in self._image_streams
            else 0,
            "metadata_topics": {
                name: {
                    "message_count": stats["message_count"],
                    "first_timestamp_ns": stats["first_timestamp_ns"],
                    "last_timestamp_ns": stats["last_timestamp_ns"],
                }
                for name, stats in self._metadata_stats.items()
            },
            "chunk_hashes": chunk_hashes,
        }

        self._output_dir.mkdir(parents=True, exist_ok=True)
        # Install the manifest atomically (temporary sibling + rename).  The
        # writer is only marked finalized after a successful replace; on
        # ``OSError`` the temporary manifest is removed and a later finalize
        # call retries with the already-installed chunks (flushes are no-ops
        # for already-flushed rows, so no duplicate chunk indices).
        self._install_manifest_atomic(json.dumps(manifest, indent=2) + "\n")
        self._finalized = True
        return manifest

    # ----------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------- #

    def _flush_metadata(self) -> None:
        """Flush the current metadata buffer into one compressed chunk.

        The chunk is written to a temporary sibling file and atomically
        renamed into place.  If the write raises ``OSError``, the partial
        temporary file is removed and the pending rows and chunk counter are
        left untouched, so a later retry re-flushes the same rows under the
        same chunk index.
        """
        rows = self._metadata_rows
        if not rows:
            return

        index = self._metadata_chunks_written

        names = np.array([r[0] for r in rows], dtype="U")
        types = np.array([r[1] for r in rows], dtype="U")
        timestamps = np.array([r[2] for r in rows], dtype=np.int64)
        payloads = [bytes(r[3]) for r in rows]

        total = sum(len(p) for p in payloads)
        offsets = np.empty(len(rows) + 1, dtype=np.int64)
        offsets[0] = 0
        running = 0
        for i, p in enumerate(payloads):
            running += len(p)
            offsets[i + 1] = running
        if total > 0:
            payload_bytes = np.frombuffer(b"".join(payloads), dtype=np.uint8).copy()
        else:
            payload_bytes = np.zeros(0, dtype=np.uint8)

        out_dir = self._output_dir / "metadata"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"topics_{index:06d}.npz"
        self._write_chunk_atomic(
            path,
            topic_names=names,
            topic_types=types,
            timestamps_ns=timestamps,
            payload_offsets=offsets,
            payload_bytes=payload_bytes,
        )
        # Bookkeeping only after a durable atomic install; on failure the
        # rows persist for a retry and the chunk index is not consumed.
        self._metadata_rows = []
        self._metadata_chunks_written = index + 1
        self._chunk_paths.append(str(path))

    def _flush_image(self, state: _ImageStreamState) -> None:
        """Flush the current image buffer for ``state`` into one chunk.

        The chunk is written to a temporary sibling file and atomically
        renamed into place.  If the write raises ``OSError``, the partial
        temporary file is removed and the pending rows and chunk counter are
        left untouched, so a later retry re-flushes the same rows under the
        same chunk index.
        """
        rows = state.rows
        if not rows:
            return

        index = state.chunks_written

        frame_numbers = np.array([r[1] for r in rows], dtype=np.int64)
        timestamps = np.array([r[2] for r in rows], dtype=np.int64)
        header_timestamps = np.array([r[3] for r in rows], dtype=np.int64)
        frame_ids = np.array([r[4] for r in rows], dtype="U")
        digests = np.array([r[5] for r in rows], dtype="U")

        subdir = "rgb" if state.stream == _RGB else "depth"
        out_dir = self._output_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{subdir}_{index:06d}.npz"
        fields = {
            "frame_numbers": frame_numbers,
            "timestamps_ns": timestamps,
            "header_timestamps_ns": header_timestamps,
            "frame_ids": frame_ids,
            "source_payload_sha256": digests,
        }
        if state.encoded_payload:
            payloads = [r[0] for r in rows]
            encoded_formats = np.array([r[6] for r in rows], dtype="U")
            offsets = np.empty(len(payloads) + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum([len(payload) for payload in payloads], out=offsets[1:])
            payload_bytes = np.frombuffer(b"".join(payloads), dtype=np.uint8).copy()
            self._write_chunk_atomic(
                path,
                compressed=False,
                encoded_payload_bytes=payload_bytes,
                encoded_payload_offsets=offsets,
                encoded_formats=encoded_formats,
                **fields,
            )
        else:
            frames = np.stack([r[0] for r in rows], axis=0)
            self._write_chunk_atomic(path, frames=frames, **fields)
        # Bookkeeping only after a durable atomic install; on failure the
        # rows persist for a retry and the chunk index is not consumed.
        state.rows = []
        state.chunks_written = index + 1
        self._chunk_paths.append(str(path))

    # ----------------------------------------------------------------- #
    # Atomic write helpers
    # ----------------------------------------------------------------- #

    def _tmp_sibling(self, final_path: Path) -> Path:
        """Return a unique temporary path in the final file's directory.

        The name is a hidden sibling (leading dot) so it never matches chunk
        globs such as ``rgb_*.npz``, and for ``.npz`` targets it keeps the
        ``.npz`` suffix so ``numpy.savez_compressed`` writes exactly that path
        instead of appending another ``.npz``.
        """
        return final_path.with_name(f".{final_path.name}.partial.{uuid.uuid4().hex}")

    def _write_chunk_atomic(self, path: Path, *, compressed: bool = True, **arrays) -> None:
        """Atomically write a compressed NPZ chunk via a temp sibling file.

        ``np.savez_compressed`` targets a temporary sibling file in the same
        directory (same filesystem), which is then atomically renamed over
        ``path``.  On ``OSError`` the partial temporary file is removed and the
        error re-raised; callers keep their pending rows and counters so a
        retry re-flushes under the same chunk index.
        """
        # Keep the .npz suffix so savez_compressed writes exactly this path.
        temp = self._tmp_sibling(path)
        if not temp.name.endswith(".npz"):
            temp = temp.with_name(temp.name + ".npz")
        try:
            save = np.savez_compressed if compressed else np.savez
            save(temp, **arrays)
            temp.replace(path)
        except OSError:
            self._remove_quietly(temp)
            raise

    def _install_manifest_atomic(self, payload: str) -> None:
        """Atomically install ``export_manifest.json`` via a temp sibling.

        The temporary manifest is written in the destination directory and
        atomically renamed over ``export_manifest.json``.  On ``OSError`` the
        temporary file is removed and the error re-raised; the writer is left
        un-finalized (by the caller) so finalize can retry.  Retries after a
        failed manifest install do not rewrite chunks because flushed rows are
        already cleared, so no duplicate chunk indices are produced.
        """
        target = self._output_dir / "export_manifest.json"
        temp = self._tmp_sibling(target)
        try:
            temp.write_text(payload, encoding="utf-8")
            temp.replace(target)
        except OSError:
            self._remove_quietly(temp)
            raise

    @staticmethod
    def _remove_quietly(path: Path) -> None:
        """Remove a partial temporary file if present, ignoring failures."""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


class NumpyArchiveReader:
    """Lazily read verified RGB-D arrays and metadata without DB3 decoding.

    NPZ chunks are opened only when an iterator advances into that chunk.
    Image pixels are returned as NumPy arrays; no CDR or JPEG/PNG image decode
    is performed. Metadata payloads are returned as their original CDR bytes.
    Since the exporter verified every chunk before publication, SHA-256
    rechecking is off by default. Set ``verify_hashes=True`` for a fresh
    integrity scan while reading.
    """

    def __init__(self, archive_dir, *, verify_hashes: bool = False):
        self.archive_dir = Path(archive_dir).expanduser().resolve()
        self.verify_hashes = bool(verify_hashes)
        manifest_path = self.archive_dir / "export_manifest.json"
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not read archive manifest: {manifest_path}") from exc
        if self.manifest.get("verified") is not True or self.manifest.get("verification", {}).get("verified") is not True:
            raise RuntimeError("archive must have a successful source verification")
        self._chunk_hashes = self.manifest.get("chunk_hashes")
        self._metadata_topics = self.manifest.get("metadata_topics")
        if not isinstance(self._chunk_hashes, dict) or not isinstance(self._metadata_topics, dict):
            raise RuntimeError("archive manifest is missing chunk hashes or metadata topic summaries")

    def _checked_chunks(self, folder: str, pattern: str):
        directory = self.archive_dir / folder
        try:
            paths = sorted(directory.glob(pattern))
        except OSError as exc:
            raise RuntimeError(f"could not list archive chunks in {directory}") from exc
        expected_paths = sorted(
            relative for relative in self._chunk_hashes
            if Path(relative).parent.as_posix() == folder
            and Path(relative).name.startswith(pattern.split("*")[0])
            and Path(relative).name.endswith(pattern.split("*")[-1])
        )
        actual_paths = [path.relative_to(self.archive_dir).as_posix() for path in paths]
        if actual_paths != expected_paths:
            raise RuntimeError(f"archive {folder} chunks are incomplete or unexpected")
        expected_count = self.manifest.get("stream_info", {}).get("counts", {}).get(folder)
        if expected_count and not paths:
            raise RuntimeError(f"archive has no {folder} chunks despite a non-zero manifest count")
        for path in paths:
            relative = path.relative_to(self.archive_dir).as_posix()
            expected = self._chunk_hashes.get(relative)
            if expected is None:
                raise RuntimeError(f"archive chunk is missing from manifest: {relative}")
            if self.verify_hashes and _sha256_file(path) != expected:
                raise RuntimeError(f"archive chunk hash mismatch: {relative}")
            yield path

    @staticmethod
    def _validate_encoded_payload(payload_bytes, offsets, count: int, path: Path) -> None:
        if (
            not isinstance(payload_bytes, np.ndarray)
            or payload_bytes.ndim != 1
            or payload_bytes.dtype != np.uint8
        ):
            raise RuntimeError(f"encoded image payload bytes are invalid: {path}")
        if (
            not isinstance(offsets, np.ndarray)
            or offsets.ndim != 1
            or not np.issubdtype(offsets.dtype, np.integer)
            or count < 1
            or len(offsets) != count + 1
        ):
            raise RuntimeError(f"encoded image payload offsets are invalid: {path}")
        if (
            int(offsets[0]) != 0
            or int(offsets[-1]) != len(payload_bytes)
            or np.any(offsets[1:] <= offsets[:-1])
        ):
            raise RuntimeError(f"encoded image payload offsets are invalid: {path}")

    def iter_images(self, stream: str):
        """Yield image-frame records from ``rgb`` or ``depth`` chunks lazily."""
        if stream not in _IMAGE_STREAMS:
            raise ValueError("stream must be 'rgb' or 'depth'")
        folder = stream
        try:
            paths = self._checked_chunks(folder, f"{folder}_*.npz")
            for path in paths:
                with np.load(path, allow_pickle=False) as chunk:
                    if "frames" in chunk.files:
                        frames = chunk["frames"]
                        encoded_payloads = None
                    elif stream == _RGB and {
                        "encoded_payload_bytes", "encoded_payload_offsets"
                    }.issubset(chunk.files):
                        payload_bytes = chunk["encoded_payload_bytes"]
                        offsets = chunk["encoded_payload_offsets"]
                        if not isinstance(offsets, np.ndarray) or offsets.ndim != 1:
                            raise RuntimeError(f"RGB chunk has invalid payload offsets: {path}")
                        frames = [None] * max(0, len(offsets) - 1)
                        self._validate_encoded_payload(
                            payload_bytes, offsets, len(frames), path
                        )
                        encoded_payloads = (payload_bytes, offsets)
                    else:
                        raise RuntimeError(f"image chunk has no supported frame data: {path}")
                    fields = (
                        chunk["frame_numbers"],
                        chunk["timestamps_ns"],
                        chunk["header_timestamps_ns"],
                        chunk["frame_ids"],
                        chunk["source_payload_sha256"],
                    )
                    if any(len(values) != len(frames) for values in fields):
                        raise RuntimeError(f"image chunk has mismatched field lengths: {path}")
                    for index, frame in enumerate(frames):
                        if encoded_payloads is not None:
                            payload_bytes, offsets = encoded_payloads
                            start, end = int(offsets[index]), int(offsets[index + 1])
                            bgr = cv2.imdecode(
                                payload_bytes[start:end], cv2.IMREAD_COLOR
                            )
                            if bgr is None:
                                raise RuntimeError(
                                    f"could not decode RGB payload {index} in {path}"
                                )
                            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        yield ArchivedImageFrame(
                            frame=frame,
                            frame_number=int(fields[0][index]),
                            timestamp_ns=int(fields[1][index]),
                            header_timestamp_ns=int(fields[2][index]),
                            frame_id=str(fields[3][index]),
                            source_payload_sha256=str(fields[4][index]),
                        )
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f"could not read {stream} archive chunks") from exc

    def iter_encoded_images(self, stream: str = _RGB):
        """Yield original compressed image bytes from encoded archive chunks.

        New RGB chunks retain their exact source JPEG/PNG payloads. Legacy
        array chunks cannot provide the original encoded bytes and raise an
        error when reached.
        """
        if stream != _RGB:
            raise ValueError("encoded image payload access is currently supported for RGB")
        try:
            for path in self._checked_chunks(_RGB, "rgb_*.npz"):
                with np.load(path, allow_pickle=False) as chunk:
                    required = {
                        "encoded_payload_bytes",
                        "encoded_payload_offsets",
                        "frame_numbers",
                        "timestamps_ns",
                        "header_timestamps_ns",
                        "frame_ids",
                        "encoded_formats",
                        "source_payload_sha256",
                    }
                    if not required.issubset(chunk.files):
                        raise RuntimeError(
                            f"RGB chunk does not contain encoded image payloads: {path}"
                        )
                    payload_bytes = chunk["encoded_payload_bytes"]
                    offsets = chunk["encoded_payload_offsets"]
                    fields = (
                        chunk["frame_numbers"],
                        chunk["timestamps_ns"],
                        chunk["header_timestamps_ns"],
                        chunk["frame_ids"],
                        chunk["encoded_formats"],
                        chunk["source_payload_sha256"],
                    )
                    count = len(fields[0])
                    if any(len(values) != count for values in fields[1:]):
                        raise RuntimeError(f"RGB chunk has mismatched field lengths: {path}")
                    self._validate_encoded_payload(payload_bytes, offsets, count, path)
                    for index in range(count):
                        start, end = int(offsets[index]), int(offsets[index + 1])
                        yield ArchivedEncodedImage(
                            payload=payload_bytes[start:end].tobytes(),
                            frame_number=int(fields[0][index]),
                            timestamp_ns=int(fields[1][index]),
                            header_timestamp_ns=int(fields[2][index]),
                            frame_id=str(fields[3][index]),
                            format=str(fields[4][index]),
                            source_payload_sha256=str(fields[5][index]),
                        )
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise RuntimeError("could not read encoded RGB archive chunks") from exc

    def iter_metadata(self, topic_name: str | None = None):
        """Yield exact CDR metadata payloads, optionally for one topic."""
        if topic_name is not None and topic_name not in self._metadata_topics:
            raise ValueError(f"metadata topic is not present in the archive: {topic_name}")
        try:
            for path in self._checked_chunks("metadata", "topics_*.npz"):
                with np.load(path, allow_pickle=False) as chunk:
                    names = chunk["topic_names"]
                    types = chunk["topic_types"]
                    timestamps = chunk["timestamps_ns"]
                    offsets = chunk["payload_offsets"]
                    payload_bytes = chunk["payload_bytes"]
                    if len(offsets) != len(names) + 1 or len(types) != len(names) or len(timestamps) != len(names):
                        raise RuntimeError(f"metadata chunk has mismatched field lengths: {path}")
                    if len(offsets) and (int(offsets[0]) != 0 or int(offsets[-1]) != len(payload_bytes)):
                        raise RuntimeError(f"metadata chunk has invalid payload offsets: {path}")
                    if np.any(offsets[1:] < offsets[:-1]):
                        raise RuntimeError(f"metadata chunk payload offsets are not monotonic: {path}")
                    for index, name in enumerate(names):
                        decoded_name = str(name)
                        if topic_name is not None and decoded_name != topic_name:
                            continue
                        start, end = int(offsets[index]), int(offsets[index + 1])
                        yield ArchivedMetadataMessage(
                            topic_name=decoded_name,
                            topic_type=str(types[index]),
                            timestamp_ns=int(timestamps[index]),
                            payload=payload_bytes[start:end].tobytes(),
                        )
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise RuntimeError("could not read metadata archive chunks") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_stream_for_topic(name: str) -> str | None:
    if name.endswith("/image/data/compressedDepth"):
        return _DEPTH
    if name.endswith("/image/data/compressed"):
        return _RGB
    return None


def verify_archive(source_db3, archive_dir) -> dict:
    """Compare archived arrays and metadata byte-for-byte with a source DB3."""
    source = Path(source_db3).resolve()
    archive = Path(archive_dir).resolve()
    manifest_path = archive / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not source.is_file() or _sha256_file(source) != manifest["source_info"]["sha256"]:
        raise RuntimeError("source DB3 hash does not match the export manifest")
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        topics = {int(i): (str(n), str(t)) for i, n, t in connection.execute("SELECT id,name,type FROM topics")}
        image_topics = {
            stream: next(i for i, (name, _) in topics.items() if _image_stream_for_topic(name) == stream)
            for stream in _IMAGE_STREAMS
        }
        archived_metadata = []
        for path in sorted((archive / "metadata").glob("topics_*.npz")):
            with np.load(path, allow_pickle=False) as chunk:
                names = chunk["topic_names"]
                types = chunk["topic_types"]
                times = chunk["timestamps_ns"]
                offsets = chunk["payload_offsets"]
                raw = chunk["payload_bytes"].tobytes()
                archived_metadata.extend(
                    (str(names[i]), str(types[i]), int(times[i]), raw[int(offsets[i]):int(offsets[i + 1])])
                    for i in range(len(names))
                )
        source_metadata = (
            (topics[int(topic_id)][0], topics[int(topic_id)][1], int(ts), bytes(payload))
            for topic_id, ts, payload in connection.execute(
                "SELECT topic_id,timestamp,data FROM messages ORDER BY id"
            ) if int(topic_id) not in image_topics.values()
        )
        for index, (saved, original) in enumerate(zip(archived_metadata, source_metadata)):
            if saved != original:
                raise RuntimeError(f"metadata message {index} differs from source DB3")
        if len(archived_metadata) != int(manifest["source_info"]["message_count"]) - sum(
            int(connection.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (topic_id,)).fetchone()[0])
            for topic_id in image_topics.values()
        ):
            raise RuntimeError("metadata message count differs from source DB3")
        archived_topic_summary = {}
        for name, _, timestamp, _ in archived_metadata:
            stats = archived_topic_summary.setdefault(name, {
                "message_count": 0,
                "first_timestamp_ns": timestamp,
                "last_timestamp_ns": timestamp,
            })
            stats["message_count"] += 1
            stats["last_timestamp_ns"] = timestamp
        if archived_topic_summary != manifest.get("metadata_topics"):
            raise RuntimeError("metadata topic summary differs from archived messages")
        source_counts = {
            topics[int(topic_id)][0]: int(count)
            for topic_id, count in connection.execute(
                "SELECT topic_id,COUNT(*) FROM messages GROUP BY topic_id"
            )
        }
        if source_counts != manifest["source_info"].get("topic_message_counts"):
            raise RuntimeError("source topic counts differ from the manifest")

        for stream, topic_id in image_topics.items():
            paths = sorted((archive / stream).glob(f"{stream}_*.npz"))
            saved_count = 0
            source_rows = iter(connection.execute(
                "SELECT id,timestamp,data FROM messages WHERE topic_id=? ORDER BY id", (topic_id,)
            ))
            for path in paths:
                with np.load(path, allow_pickle=False) as chunk:
                    if "frames" in chunk.files:
                        frame_count = len(chunk["frames"])
                        encoded_bytes = encoded_offsets = None
                    elif stream == _RGB and {
                        "encoded_payload_bytes", "encoded_payload_offsets"
                    }.issubset(chunk.files):
                        encoded_bytes = chunk["encoded_payload_bytes"]
                        encoded_offsets = chunk["encoded_payload_offsets"]
                        frame_count = len(chunk["frame_numbers"])
                        NumpyArchiveReader._validate_encoded_payload(
                            encoded_bytes, encoded_offsets, frame_count, path
                        )
                    else:
                        raise RuntimeError(f"{stream} archive chunk has no supported frame data: {path}")
                    for i in range(frame_count):
                        row = next(source_rows, None)
                        if row is None:
                            raise RuntimeError(f"{stream} archive contains extra frames")
                        payload = bytes(row[2])
                        decoded = decode_compressed_image_cdr(payload, stream)
                        if encoded_bytes is not None:
                            start, end = int(encoded_offsets[i]), int(encoded_offsets[i + 1])
                            if encoded_bytes[start:end].tobytes() != decoded.encoded_payload:
                                raise RuntimeError(
                                    f"{stream} frame {saved_count} encoded payload differs from source"
                                )
                            bgr = cv2.imdecode(
                                encoded_bytes[start:end], cv2.IMREAD_COLOR
                            )
                            if bgr is None:
                                raise RuntimeError(
                                    f"{stream} frame {saved_count} encoded payload cannot be decoded"
                                )
                            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        else:
                            frame = chunk["frames"][i]
                        if not np.array_equal(frame, decoded.frame):
                            raise RuntimeError(f"{stream} frame {saved_count} differs from source pixels")
                        if hashlib.sha256(payload).hexdigest() != str(chunk["source_payload_sha256"][i]):
                            raise RuntimeError(f"{stream} frame {saved_count} source payload hash differs")
                        if int(chunk["timestamps_ns"][i]) != int(row[1]):
                            raise RuntimeError(f"{stream} frame {saved_count} timestamp differs")
                        if int(chunk["frame_numbers"][i]) != int(row[0]):
                            raise RuntimeError(f"{stream} frame {saved_count} DB3 row id differs")
                        if int(chunk["header_timestamps_ns"][i]) != decoded.header_timestamp_ns:
                            raise RuntimeError(f"{stream} frame {saved_count} header timestamp differs")
                        if str(chunk["frame_ids"][i]) != decoded.frame_id:
                            raise RuntimeError(f"{stream} frame {saved_count} frame id differs")
                        if encoded_bytes is not None and str(chunk["encoded_formats"][i]) != decoded.format:
                            raise RuntimeError(f"{stream} frame {saved_count} encoded format differs")
                        saved_count += 1
            if next(source_rows, None) is not None:
                raise RuntimeError(f"{stream} archive is missing source frames")
        for relative, expected in manifest["chunk_hashes"].items():
            if _sha256_file(archive / relative) != expected:
                raise RuntimeError(f"archive chunk hash mismatch: {relative}")
        actual_chunks = {
            path.relative_to(archive).as_posix()
            for folder in ("metadata", "rgb", "depth")
            for path in (archive / folder).glob("*.npz")
        }
        if actual_chunks != set(manifest["chunk_hashes"]):
            raise RuntimeError("archive chunk file set differs from the manifest")
        source_stream_counts = {
            stream: int(connection.execute(
                "SELECT COUNT(*) FROM messages WHERE topic_id=?", (topic_id,)
            ).fetchone()[0])
            for stream, topic_id in image_topics.items()
        }
        expected_counts = {
            "rgb": source_stream_counts[_RGB],
            "depth": source_stream_counts[_DEPTH],
            "metadata": len(archived_metadata),
        }
        if manifest["stream_info"].get("counts") != expected_counts:
            raise RuntimeError("manifest stream counts differ from verified source counts")
        expected_chunk_counts = {
            "metadata_chunks": len(list((archive / "metadata").glob("topics_*.npz"))),
            "rgb_chunks": len(list((archive / "rgb").glob("rgb_*.npz"))),
            "depth_chunks": len(list((archive / "depth").glob("depth_*.npz"))),
        }
        if any(manifest.get(key) != value for key, value in expected_chunk_counts.items()):
            raise RuntimeError("manifest chunk counts differ from archive files")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(archive / "review.mp4")],
            check=True, capture_output=True, text=True,
        )
        video_count = int(probe.stdout.strip())
        source_rgb_count = source_stream_counts[_RGB]
        if video_count != source_rgb_count:
            raise RuntimeError(f"review video frame count {video_count} != RGB source count {source_rgb_count}")
        def archived_rgb_frames():
            for rgb_path in sorted((archive / _RGB).glob("rgb_*.npz")):
                with np.load(rgb_path, allow_pickle=False) as chunk:
                    if "frames" in chunk.files:
                        yield from chunk["frames"]
                    else:
                        payload_bytes = chunk["encoded_payload_bytes"]
                        offsets = chunk["encoded_payload_offsets"]
                        for start, end in zip(offsets[:-1], offsets[1:]):
                            bgr = cv2.imdecode(
                                payload_bytes[int(start):int(end)], cv2.IMREAD_COLOR
                            )
                            if bgr is None:
                                raise RuntimeError("RGB encoded archive payload cannot be decoded")
                            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        rgb_frames = iter(archived_rgb_frames())
        capture = cv2.VideoCapture(str(archive / "review.mp4"))
        squared_error = 0
        pixel_count = 0
        max_abs_error = 0
        for index in range(source_rgb_count):
            rgb = next(rgb_frames, None)
            if rgb is None:
                capture.release()
                raise RuntimeError(f"RGB archive ended before frame {index}")
            ok, bgr = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError(f"review video ended before RGB frame {index}")
            rendered = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rendered.shape != rgb.shape:
                capture.release()
                raise RuntimeError(f"review video frame {index} has shape {rendered.shape}, expected {rgb.shape}")
            error = rendered.astype(np.int16) - rgb.astype(np.int16)
            squared_error += int(np.sum(error.astype(np.int64) ** 2))
            pixel_count += int(error.size)
            max_abs_error = max(max_abs_error, int(np.max(np.abs(error))))
        extra_ok, _ = capture.read()
        capture.release()
        if extra_ok:
            raise RuntimeError("review video has more decoded frames than the RGB archive")
        if next(rgb_frames, None) is not None:
            raise RuntimeError("RGB archive has more frames than the review video")
        mse = squared_error / max(pixel_count, 1)
        psnr_db = float("inf") if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)
        manifest["verification"] = {
            "verified": True, "source_sha256": manifest["source_info"]["sha256"],
            "metadata_messages": len(archived_metadata), "review_video_frames": video_count,
            "verified_image_arrays": True, "verified_metadata_payload_bytes": True,
            "review_video_quality": {
                "rgb_to_video_psnr_db": float(psnr_db),
                "rgb_to_video_mse": float(mse),
                "rgb_to_video_max_abs_channel_error": max_abs_error,
                "comparison": "decoded HEVC frame against exact archived RGB frame in order",
            },
        }
        manifest["verified"] = True
        temporary = manifest_path.with_name(".export_manifest.json.verify")
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
        return manifest["verification"]
    finally:
        connection.close()


def export_recording(recording, output_dir, *, frames_per_chunk: int = 128) -> dict:
    """Export a RealSense rosbag2 SQLite database to lossless compressed NPZ.

    All non-image messages, including per-frame image metadata and camera info,
    are preserved as their original CDR byte payloads in compressed metadata
    chunks. The source database is opened read-only and is never modified.
    """
    source = Path(recording).resolve()
    output = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output == source or source in output.parents:
        raise ValueError("output directory must not contain or replace the source DB3")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"destination must be empty: {output}")
    work = output.with_name(f".{output.name}.partial.{uuid.uuid4().hex}")
    before_hash = _sha256_file(source)
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    ffmpeg = None
    temp_video = None
    try:
        topics = {
            row["id"]: dict(row)
            for row in connection.execute(
                "SELECT id, name, type, serialization_format, offered_qos_profiles FROM topics"
            )
        }
        topic_counts = {
            int(row["topic_id"]): int(row["n"])
            for row in connection.execute(
                "SELECT topic_id, COUNT(*) AS n FROM messages GROUP BY topic_id"
            )
        }
        descriptors = [
            {key: topic[key] for key in ("name", "type", "serialization_format", "offered_qos_profiles")}
            for topic in topics.values()
        ]
        total = int(connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        work.mkdir(parents=True, exist_ok=False)
        writer = NumpyArchiveWriter(work, frames_per_chunk=frames_per_chunk)
        counts = {"rgb": 0, "depth": 0, "metadata": 0}
        stream_topics = {s: [t for t in topics.values() if _image_stream_for_topic(t["name"]) == s] for s in _IMAGE_STREAMS}
        if any(len(items) != 1 for items in stream_topics.values()):
            raise ValueError("recording must contain exactly one RGB and one compressedDepth image topic")
        rgb_times = [int(r[0]) for r in connection.execute(
            "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY id",
            (stream_topics[_RGB][0]["id"],),
        )]
        delta = [b - a for a, b in zip(rgb_times, rgb_times[1:]) if b > a]
        fps = 1e9 / (sum(delta) / len(delta)) if delta else 30.0
        video_path = work / "review.mp4"
        temp_video = work / ".review.partial.mp4"
        # Image dimensions are discovered from the first decoded frame, so defer
        # encoder startup until that frame has been read.
        for row in connection.execute(
            "SELECT id, topic_id, timestamp, data FROM messages ORDER BY id"
        ):
            topic = topics[int(row["topic_id"])]
            stream = _image_stream_for_topic(topic["name"])
            payload = bytes(row["data"])
            if stream is None:
                writer.append_metadata(topic["name"], topic["type"], int(row["timestamp"]), payload)
                counts["metadata"] += 1
                continue
            decoded = decode_compressed_image_cdr(payload, stream)
            writer.append_image(
                stream,
                decoded.frame,
                frame_number=int(row["id"]),
                timestamp_ns=int(row["timestamp"]),
                header_timestamp_ns=decoded.header_timestamp_ns,
                frame_id=decoded.frame_id,
                source_payload=payload,
                encoded_payload=decoded.encoded_payload if stream == _RGB else None,
                encoded_format=decoded.format if stream == _RGB else None,
            )
            counts[stream] += 1
            if stream == _RGB:
                height, width = decoded.frame.shape[:2]
                if ffmpeg is None:
                    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}", "-r", f"{fps:.9f}", "-i", "-"]
                    ffmpeg = subprocess.Popen(base + ["-an", "-c:v", "hevc_nvenc", "-preset", "p5", "-cq", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp_video)], stdin=subprocess.PIPE)
                assert ffmpeg.stdin is not None
                ffmpeg.stdin.write(np.ascontiguousarray(decoded.frame).tobytes())
        if ffmpeg is None:
            raise ValueError("recording contains no RGB frames")
        assert ffmpeg.stdin is not None
        ffmpeg.stdin.close()
        code = ffmpeg.wait()
        ffmpeg = None
        if code:
            raise OSError(f"ffmpeg encoder failed with exit status {code}")
        os.replace(temp_video, video_path)
        after_hash = _sha256_file(source)
        if before_hash != after_hash:
            raise RuntimeError("source DB3 changed during export")
        source_info = {
            "path": str(source), "sha256": before_hash, "size_bytes": source.stat().st_size,
            "message_count": total, "topic_count": len(topics), "topic_descriptors": descriptors,
            "topic_message_counts": {topics[k]["name"]: v for k, v in topic_counts.items()},
        }
        stream_info = {
            "counts": counts,
            "image_topics": {s: stream_topics[s][0]["name"] for s in _IMAGE_STREAMS},
            "review_video": {"path": "review.mp4", "codec": "hevc_nvenc", "fps": fps, "rgb_frames": counts[_RGB]},
        }
        manifest = writer.finalize(source_info, stream_info)
        verify_archive(source, work)
        if output.exists():
            output.rmdir()
        work.replace(output)
        manifest["output_dir"] = str(output)
        manifest_tmp = output / ".export_manifest.json.partial"
        manifest = json.loads((output / "export_manifest.json").read_text(encoding="utf-8"))
        manifest["output_dir"] = str(output)
        manifest_tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest_tmp.replace(output / "export_manifest.json")
        return manifest
    except Exception:
        if ffmpeg is not None:
            try:
                if ffmpeg.stdin:
                    ffmpeg.stdin.close()
                ffmpeg.terminate()
                ffmpeg.wait(timeout=10)
            except Exception:
                pass
        if temp_video is not None:
            temp_video.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
        raise
    finally:
        connection.close()
