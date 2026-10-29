"""Contract tests for ROS 2 CompressedImage CDR decoding."""

import struct

import cv2
import numpy as np
import pytest

from conceptgraph.utils.realsense_archive import (
    DecodedCompressedImage,
    decode_compressed_image_cdr,
)


def _cdr_string(value: str, endian: str) -> bytes:
    encoded = value.encode("utf-8") + b"\0"
    return struct.pack(f"{endian}I", len(encoded)) + encoded


def _cdr_compressed_image(
    *,
    endian: str = "<",
    frame_id: str = "camera_color_optical_frame",
    stamp_sec: int = 12,
    stamp_nanosec: int = 345_678_901,
    image_format: str = "jpeg",
    image_data: bytes = b"image payload",
) -> bytes:
    """Build a CDR encapsulation and CompressedImage body for either byte order."""
    body = bytearray(struct.pack(f"{endian}iI", stamp_sec, stamp_nanosec))
    body.extend(_cdr_string(frame_id, endian))
    body.extend(b"\0" * (-len(body) % 4))
    body.extend(_cdr_string(image_format, endian))
    body.extend(b"\0" * (-len(body) % 4))
    body.extend(struct.pack(f"{endian}I", len(image_data)))
    body.extend(image_data)
    # CDR encapsulation identifier 0 is big-endian; 1 is little-endian.
    encapsulation = b"\0\x01\0\0" if endian == "<" else b"\0\0\0\0"
    return encapsulation + body


@pytest.mark.parametrize("endian", ["<", ">"], ids=["little-endian", "big-endian"])
def test_decodes_jpeg_rgb_and_header_fields_for_both_byte_orders(endian):
    bgr = np.zeros((9, 13, 3), dtype=np.uint8)
    bgr[:, :5] = (20, 80, 220)
    bgr[:, 5:] = (210, 40, 10)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
    assert ok
    payload = _cdr_compressed_image(endian=endian, image_data=encoded.tobytes())

    decoded = decode_compressed_image_cdr(payload, "rgb")

    assert isinstance(decoded, DecodedCompressedImage)
    np.testing.assert_array_equal(decoded.frame, cv2.imdecode(encoded, cv2.IMREAD_COLOR)[:, :, ::-1])
    assert decoded.frame.dtype == np.uint8
    assert decoded.header_timestamp_ns == 12_345_678_901
    assert decoded.frame_id == "camera_color_optical_frame"
    assert decoded.format == "jpeg"


@pytest.mark.parametrize("endian", ["<", ">"], ids=["little-endian", "big-endian"])
def test_decodes_compressed_depth_png_without_converting_uint16(endian):
    expected = np.array(
        [[0, 1, 250, 1000], [4095, 12_000, 30_000, 65_000]], dtype=np.uint16
    )
    ok, encoded = cv2.imencode(".png", expected)
    assert ok
    config_header = struct.pack("<iff", 0, 0.0, 0.0)
    payload = _cdr_compressed_image(
        endian=endian,
        frame_id="camera_depth_optical_frame",
        image_format="16UC1; compressedDepth png",
        image_data=config_header + encoded.tobytes(),
    )

    decoded = decode_compressed_image_cdr(payload, "depth")

    assert isinstance(decoded, DecodedCompressedImage)
    np.testing.assert_array_equal(decoded.frame, expected)
    assert decoded.frame.dtype == np.uint16
    assert decoded.frame_id == "camera_depth_optical_frame"
    assert decoded.format == "16UC1; compressedDepth png"


@pytest.mark.parametrize(
    ("payload", "stream", "reason"),
    [
        (b"", "rgb", "empty payload"),
        (b"\0\x01\0\0\x01", "rgb", "truncated CDR body"),
        (b"\0\x02\0\0" + b"\0" * 20, "rgb", "unsupported encapsulation"),
        (b"\0\x01\0\0" + b"\0" * 9, "rgb", "truncated header"),
        (
            _cdr_compressed_image(image_format="png", image_data=b"not a JPEG"),
            "rgb",
            "wrong RGB format",
        ),
        (
            _cdr_compressed_image(
                image_format="16UC1; compressedDepth png", image_data=b"short"
            ),
            "depth",
            "truncated compressedDepth header",
        ),
        (
            _cdr_compressed_image(
                image_format="16UC1; compressedDepth png",
                image_data=struct.pack("<iff", 0, 0.0, 0.0) + b"not a PNG",
            ),
            "depth",
            "invalid depth image bytes",
        ),
        (
            _cdr_compressed_image(image_format="16UC1; compressedDepth png"),
            "rgb",
            "wrong stream format",
        ),
        (
            _cdr_compressed_image(image_format="jpeg", image_data=b"not a JPEG"),
            "rgb",
            "invalid RGB image bytes",
        ),
    ],
)
def test_rejects_malformed_or_unsupported_payloads(payload, stream, reason):
    del reason
    with pytest.raises(ValueError):
        decode_compressed_image_cdr(payload, stream)
