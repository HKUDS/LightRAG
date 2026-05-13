"""Shared image-input normalization for LLM bindings.

All LLM bindings accept a unified ``image_inputs`` keyword parameter. Each
element may be:

- a raw base64 string (the MIME type is inferred via ``imghdr`` / magic bytes,
  defaulting to ``image/png``);
- a data URL of the form ``data:<mime>;base64,<payload>``;
- a dict with keys ``base64`` (required) and optional ``mime_type``,
  ``source_id``, ``source_file``, ``modality``, ``doc_id``.

The provider-specific binding code converts the normalized result to its own
content-block format. The VLM pipeline uses :func:`image_cache_metadata` for
cache-key inputs (deliberately excluding ``source_id`` / ``source_file`` so the
same image at different filenames still hits the same entry) and
:func:`image_audit_metadata` for the human-readable ``original_prompt`` audit
block.
"""

from __future__ import annotations

import base64
import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[\w./+-]+);base64,(?P<data>[A-Za-z0-9+/=\s]+)$"
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
_WEBP_RIFF = b"RIFF"
_WEBP_TAG = b"WEBP"


@dataclass(frozen=True)
class NormalizedImage:
    index: int
    raw_bytes: bytes
    mime_type: str
    sha256: str
    base64_str: str
    source_id: str | None
    source_file: str | None
    modality: str | None
    doc_id: str | None


def _detect_mime(raw: bytes) -> str:
    if raw.startswith(_PNG_SIGNATURE):
        return "image/png"
    if raw.startswith(_JPEG_SIGNATURE):
        return "image/jpeg"
    if any(raw.startswith(sig) for sig in _GIF_SIGNATURES):
        return "image/gif"
    if len(raw) >= 12 and raw[0:4] == _WEBP_RIFF and raw[8:12] == _WEBP_TAG:
        return "image/webp"
    return "image/png"


def _decode_base64(data: str) -> bytes:
    cleaned = re.sub(r"\s+", "", data)
    try:
        return base64.b64decode(cleaned, validate=True)
    except (base64.binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 image data: {exc}") from exc


def _coerce_item(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        match = DATA_URL_RE.match(item.strip())
        if match:
            return {"base64": match.group("data"), "mime_type": match.group("mime")}
        return {"base64": item}
    if isinstance(item, dict):
        if "base64" not in item:
            raise ValueError("image_inputs dict element must contain a 'base64' key")
        return item
    raise TypeError(
        f"image_inputs element must be str or dict, got {type(item).__name__}"
    )


def normalize_image_inputs(
    image_inputs: list[Any] | None,
) -> list[NormalizedImage]:
    """Normalize the unified ``image_inputs`` parameter.

    Returns an empty list when ``image_inputs`` is falsy, so callers can do a
    plain ``if normalized:`` check.
    """
    if not image_inputs:
        return []

    result: list[NormalizedImage] = []
    for idx, raw_item in enumerate(image_inputs):
        item = _coerce_item(raw_item)
        raw_bytes = _decode_base64(item["base64"])
        if not raw_bytes:
            raise ValueError(f"image_inputs[{idx}] decoded to empty bytes")
        mime_type = item.get("mime_type") or _detect_mime(raw_bytes)
        sha = hashlib.sha256(raw_bytes).hexdigest()
        clean_b64 = base64.b64encode(raw_bytes).decode("ascii")
        result.append(
            NormalizedImage(
                index=idx,
                raw_bytes=raw_bytes,
                mime_type=mime_type,
                sha256=sha,
                base64_str=clean_b64,
                source_id=item.get("source_id"),
                source_file=item.get("source_file"),
                modality=item.get("modality"),
                doc_id=item.get("doc_id"),
            )
        )
    return result


def image_cache_metadata(images: list[NormalizedImage]) -> list[dict[str, Any]]:
    """Return cache-key-safe image metadata (no source identifiers)."""
    return [
        {
            "index": img.index,
            "mime_type": img.mime_type,
            "sha256": img.sha256,
            "bytes": len(img.raw_bytes),
        }
        for img in images
    ]


def image_audit_metadata(images: list[NormalizedImage]) -> list[dict[str, Any]]:
    """Return audit metadata suitable for the ``original_prompt`` block.

    Never includes the raw base64 payload — only digests and source pointers.
    """
    return [
        {
            "index": img.index,
            "mime_type": img.mime_type,
            "sha256": img.sha256,
            "bytes": len(img.raw_bytes),
            "source_id": img.source_id,
            "source_file": img.source_file,
            "modality": img.modality,
            "doc_id": img.doc_id,
        }
        for img in images
    ]


def _read_png_dimensions(data: bytes) -> tuple[int, int] | None:
    # IHDR is the first chunk; width/height are big-endian uint32 at offsets
    # 16/20 (8-byte signature + 4 length + 4 "IHDR" + 4 width + 4 height).
    if len(data) < 24 or not data.startswith(_PNG_SIGNATURE):
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _read_gif_dimensions(data: bytes) -> tuple[int, int] | None:
    # Logical screen descriptor: width/height are little-endian uint16 at
    # offsets 6/8.
    if len(data) < 10 or not any(data.startswith(sig) for sig in _GIF_SIGNATURES):
        return None
    width, height = struct.unpack("<HH", data[6:10])
    return width, height


def _read_jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    # Scan for a Start-Of-Frame marker (SOF0 / SOF2 / etc.). Skip segments by
    # their length field. We deliberately accept any SOF variant the codec
    # might emit rather than enumerating each one.
    if len(data) < 4 or not data.startswith(_JPEG_SIGNATURE):
        return None
    i = 2
    n = len(data)
    while i < n:
        if data[i] != 0xFF:
            return None
        # Skip fill bytes.
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            return None
        marker = data[i]
        i += 1
        # Standalone markers without a length field.
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > n:
            return None
        segment_len = struct.unpack(">H", data[i : i + 2])[0]
        if segment_len < 2 or i + segment_len > n:
            return None
        # SOF0..SOF15 except 0xC4 (DHT), 0xC8 (JPG reserved), 0xCC (DAC).
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            # SOF payload: precision(1) + height(2) + width(2) + …
            if i + 7 > n:
                return None
            height, width = struct.unpack(">HH", data[i + 3 : i + 7])
            return width, height
        i += segment_len
    return None


def _read_webp_dimensions(data: bytes) -> tuple[int, int] | None:
    if (
        len(data) < 30
        or data[0:4] != _WEBP_RIFF
        or data[8:12] != _WEBP_TAG
    ):
        return None
    chunk_type = data[12:16]
    if chunk_type == b"VP8 ":
        # Lossy: 3-byte tag + 3-byte sync code at offset 23, then 4 bytes
        # holding 14-bit width / 14-bit height in little-endian halves.
        if len(data) < 30:
            return None
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return width, height
    if chunk_type == b"VP8L":
        # Lossless: signature(0x2F) + 4 bytes encoding 14-bit width-1 / 14-bit
        # height-1 starting at offset 21.
        if len(data) < 25 or data[20] != 0x2F:
            return None
        b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
        width = ((b1 & 0x3F) << 8 | b0) + 1
        height = ((b3 & 0x0F) << 10 | b2 << 2 | (b1 & 0xC0) >> 6) + 1
        return width, height
    if chunk_type == b"VP8X":
        # Extended: 3 bytes width-1 / 3 bytes height-1, little-endian, at
        # offsets 24/27.
        if len(data) < 30:
            return None
        width = (data[24] | data[25] << 8 | data[26] << 16) + 1
        height = (data[27] | data[28] << 8 | data[29] << 16) + 1
        return width, height
    return None


def read_image_dimensions(path: Path) -> tuple[int, int] | None:
    """Return ``(width, height)`` for a raster image, or ``None`` if unknown.

    Reads only the file header — no Pillow dependency. Supports PNG, JPEG,
    GIF and WebP (VP8 / VP8L / VP8X). Returns ``None`` for unsupported
    formats and on any I/O or parse error so callers can fall back to a
    skipped/failure decision without raising.
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(64 * 1024)
    except OSError:
        return None
    if not header:
        return None
    for reader in (
        _read_png_dimensions,
        _read_gif_dimensions,
        _read_jpeg_dimensions,
        _read_webp_dimensions,
    ):
        try:
            dims = reader(header)
        except (struct.error, IndexError, ValueError):
            continue
        if dims:
            return dims
    return None
