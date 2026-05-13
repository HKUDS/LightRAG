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
from dataclasses import dataclass
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
