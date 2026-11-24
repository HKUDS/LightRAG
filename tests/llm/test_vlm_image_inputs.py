"""Offline tests for the unified VLM image_inputs path."""

from __future__ import annotations

import base64
import hashlib
import struct

import pytest

from lightrag.llm._vision_utils import (
    _JPEG_MAX_SCAN_STEPS,
    _dimensions_from_bytes,
    image_audit_metadata,
    image_cache_metadata,
    normalize_image_inputs,
    read_image_dimensions,
)


pytestmark = pytest.mark.offline


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8"
    b"\xcf\xc0\x00\x00\x00\x03\x00\x01\x5c\xcc\xd9\x9e\x00\x00\x00\x00"
    b"IEND\xaeB`\x82"
)
# Header-only 32x16 PNG: dimensions come from IHDR, nothing decodes pixels.
PNG_32x16 = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 32, 16)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def test_normalize_accepts_raw_base64_and_detects_png():
    result = normalize_image_inputs([_b64(PNG_BYTES)])
    assert len(result) == 1
    img = result[0]
    assert img.index == 0
    assert img.mime_type == "image/png"
    assert img.raw_bytes == PNG_BYTES
    assert img.sha256 == hashlib.sha256(PNG_BYTES).hexdigest()
    assert img.source_id is None
    assert img.source_file is None


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        # Scheme and base64 marker are ASCII case-insensitive; the type is
        # lowercased.
        pytest.param(f"DaTa:Image/Png;BaSe64,{_b64(PNG_BYTES)}", "image/png", id="url"),
        # The declared type wins over the sniffed bytes (PNG here), and the
        # image/jpg alias is canonicalized after lowercasing, in both shapes.
        pytest.param(
            f"DATA:IMAGE/JPG;BASE64,{_b64(PNG_BYTES)}", "image/jpeg", id="url-alias"
        ),
        pytest.param(
            {"base64": _b64(PNG_BYTES), "mime_type": "Image/JPG"},
            "image/jpeg",
            id="dict-alias",
        ),
    ],
)
def test_declared_mime_type_is_used_and_canonicalized(item, expected):
    [image] = normalize_image_inputs([item])
    assert image.mime_type == expected
    assert image.raw_bytes == PNG_BYTES
    assert image.base64_str == _b64(PNG_BYTES)


@pytest.mark.parametrize("marker", ["base65", "baſe64"])
def test_data_url_rejects_invalid_or_non_ascii_markers(marker):
    with pytest.raises(ValueError):
        normalize_image_inputs([f"data:image/png;{marker},{_b64(PNG_BYTES)}"])


def test_normalize_empty_returns_empty_list():
    assert normalize_image_inputs(None) == []
    assert normalize_image_inputs([]) == []


# The segment count and the buffer crossing are swept on the baseline marker;
# the progressive SOF marker only needs to be recognised once.
@pytest.mark.parametrize(
    ("metadata_segments", "frame_marker"),
    [(0, b"\xff\xc0"), (1, b"\xff\xc0"), (2, b"\xff\xc0"), (0, b"\xff\xc2")],
)
def test_jpeg_dimensions_after_metadata(tmp_path, metadata_segments, frame_marker):
    comment = b"\xff\xfe\xff\xff" + b"x" * 65533
    frame = frame_marker + b"\x00\x0b\x08\x00\x10\x00\x20\x01\x01\x11\x00"
    raw = b"\xff\xd8" + comment * metadata_segments + frame + b"\xff\xd9"
    path = tmp_path / "image.jpg"
    path.write_bytes(raw)

    assert read_image_dimensions(path) == (32, 16)
    [image] = normalize_image_inputs([_b64(raw)])
    assert (image.width, image.height) == (32, 16)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\xff\xd8\xff",
        b"\xff\xd8\xff\xe1\x00",
        b"\xff\xd8\xff\xe1\x00\x01",
        b"\xff\xd8\xff\xe1\x00\x10abc",
        b"\xff\xd8\xff\xc0\x00\x0b\x08\x00\x10\x00\x20",
    ],
)
def test_incomplete_jpeg_dimensions_are_unknown(tmp_path, raw):
    path = tmp_path / "incomplete.jpg"
    path.write_bytes(raw)
    assert read_image_dimensions(path) is None


@pytest.mark.parametrize("end_marker", [b"\xff\xd9", b"\xff\xda\x00\x02"])
def test_jpeg_dimensions_stop_before_image_data(tmp_path, end_marker):
    frame = b"\xff\xc0\x00\x0b\x08\x00\x10\x00\x20\x01\x01\x11\x00"
    raw = b"\xff\xd8" + end_marker + frame
    path = tmp_path / "no-frame.jpg"
    path.write_bytes(raw)
    assert read_image_dimensions(path) is None
    [image] = normalize_image_inputs([_b64(raw)])
    assert (image.width, image.height) == (None, None)


_JPEG_FRAME_32x16 = b"\xff\xc0\x00\x0b\x08\x00\x10\x00\x20\x01\x01\x11\x00"


@pytest.mark.parametrize(
    "padding",
    [
        pytest.param(b"\xff\xe1\x00\x02" * (_JPEG_MAX_SCAN_STEPS + 1), id="segments"),
        pytest.param(b"\xff" * (_JPEG_MAX_SCAN_STEPS + 1), id="fill-bytes"),
    ],
)
def test_jpeg_scan_is_bounded(tmp_path, padding):
    # A crafted header must not make the probe scan proportional to file
    # size: past the step budget the dimensions are unknown, even though a
    # frame header follows.
    raw = b"\xff\xd8" + padding + _JPEG_FRAME_32x16 + b"\xff\xd9"
    path = tmp_path / "crafted.jpg"
    path.write_bytes(raw)
    assert read_image_dimensions(path) is None
    assert _dimensions_from_bytes(raw) is None


def test_jpeg_scan_budget_admits_many_real_segments(tmp_path):
    # Stability control: a segment-heavy but realistic header (each segment
    # costs one step, its body is seeked past) still resolves.
    segment = b"\xff\xe2\x10\x00" + b"i" * 0x0FFE
    raw = b"\xff\xd8" + segment * 512 + _JPEG_FRAME_32x16 + b"\xff\xd9"
    path = tmp_path / "many-segments.jpg"
    path.write_bytes(raw)
    assert read_image_dimensions(path) == (32, 16)


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview])
def test_jpeg_dimensions_accept_bytes_like(wrap):
    raw = b"\xff\xd8" + _JPEG_FRAME_32x16 + b"\xff\xd9"
    assert _dimensions_from_bytes(wrap(raw)) == (32, 16)


def _webp(chunk: bytes, payload: bytes) -> bytes:
    body = b"WEBP" + chunk + len(payload).to_bytes(4, "little") + payload
    return b"RIFF" + len(body).to_bytes(4, "little") + body


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(PNG_32x16, (32, 16), id="png"),
        pytest.param(b"GIF89a\x20\x00\x10\x00" + b"\x00" * 8, (32, 16), id="gif"),
        pytest.param(
            _webp(b"VP8 ", b"\x00" * 3 + b"\x9d\x01\x2a\x20\x00\x10\x00"),
            (32, 16),
            id="webp-vp8",
        ),
        pytest.param(
            # 14-bit (width-1) / (height-1) fields: 31 and 15.
            _webp(b"VP8L", b"\x2f\x1f\xc0\x03\x00"),
            (32, 16),
            id="webp-vp8l",
        ),
        pytest.param(
            _webp(b"VP8X", b"\x00" * 4 + b"\x1f\x00\x00\x0f\x00\x00"),
            (32, 16),
            id="webp-vp8x",
        ),
    ],
)
def test_non_jpeg_dimensions_from_file_header(tmp_path, raw, expected):
    # The file entry point decodes these from a short probe; it must cover
    # the furthest offset each reader touches.
    path = tmp_path / "image.bin"
    path.write_bytes(raw + b"\x00" * 64)
    assert read_image_dimensions(path) == expected


def test_normalize_rejects_invalid_base64():
    with pytest.raises(ValueError):
        normalize_image_inputs(["this is not base64@@@!!"])


def test_normalize_rejects_unsupported_element_type():
    with pytest.raises(TypeError):
        normalize_image_inputs([12345])


def test_normalize_rejects_dict_without_base64():
    with pytest.raises(ValueError):
        normalize_image_inputs([{"mime_type": "image/png"}])


def test_cache_metadata_excludes_source_identifiers():
    images = normalize_image_inputs(
        [
            {
                "base64": _b64(PNG_32x16),
                "source_id": "leak-id",
                "source_file": "/leak/path.png",
            }
        ]
    )
    [meta] = image_cache_metadata(images)
    assert meta == {
        "index": 0,
        "mime_type": "image/png",
        "sha256": hashlib.sha256(PNG_32x16).hexdigest(),
        "bytes": len(PNG_32x16),
        "width": 32,
        "height": 16,
    }


def test_audit_metadata_includes_full_provenance_without_raw_base64():
    images = normalize_image_inputs(
        [
            {
                "base64": _b64(PNG_32x16),
                "source_id": "img-001",
                "source_file": "/tmp/foo.png",
                "modality": "image",
                "doc_id": "doc-1",
            }
        ]
    )
    [audit] = image_audit_metadata(images)
    # The audit blob carries every digest and provenance field, and never
    # re-leaks the raw base64 payload.
    assert audit == {
        "index": 0,
        "mime_type": "image/png",
        "sha256": hashlib.sha256(PNG_32x16).hexdigest(),
        "bytes": len(PNG_32x16),
        "width": 32,
        "height": 16,
        "source_id": "img-001",
        "source_file": "/tmp/foo.png",
        "modality": "image",
        "doc_id": "doc-1",
    }
