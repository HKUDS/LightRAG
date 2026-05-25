"""Offline tests for the unified VLM image_inputs path."""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import pytest

from lightrag.llm._vision_utils import (
    image_audit_metadata,
    image_cache_metadata,
    normalize_image_inputs,
)


pytestmark = pytest.mark.offline


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8"
    b"\xcf\xc0\x00\x00\x00\x03\x00\x01\x5c\xcc\xd9\x9e\x00\x00\x00\x00"
    b"IEND\xaeB`\x82"
)
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 16


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


def test_normalize_accepts_data_url_and_uses_declared_mime():
    data_url = f"data:image/jpeg;base64,{_b64(JPEG_BYTES)}"
    result = normalize_image_inputs([data_url])
    assert len(result) == 1
    assert result[0].mime_type == "image/jpeg"


def test_normalize_accepts_dict_with_metadata():
    dict_item: dict[str, Any] = {
        "base64": _b64(PNG_BYTES),
        "mime_type": "image/png",
        "source_id": "img-001",
        "source_file": "/tmp/foo.png",
        "modality": "image",
        "doc_id": "doc-1",
    }
    [img] = normalize_image_inputs([dict_item])
    assert img.source_id == "img-001"
    assert img.source_file == "/tmp/foo.png"
    assert img.modality == "image"
    assert img.doc_id == "doc-1"


def test_normalize_empty_returns_empty_list():
    assert normalize_image_inputs(None) == []
    assert normalize_image_inputs([]) == []


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
                "base64": _b64(PNG_BYTES),
                "source_id": "leak-id",
                "source_file": "/leak/path.png",
            }
        ]
    )
    [meta] = image_cache_metadata(images)
    assert "source_id" not in meta
    assert "source_file" not in meta
    assert meta["sha256"] == hashlib.sha256(PNG_BYTES).hexdigest()
    assert meta["mime_type"] == "image/png"
    assert meta["bytes"] == len(PNG_BYTES)


def test_cache_metadata_same_image_different_filename_is_identical():
    img_a = normalize_image_inputs(
        [{"base64": _b64(PNG_BYTES), "source_file": "/a/x.png"}]
    )
    img_b = normalize_image_inputs(
        [{"base64": _b64(PNG_BYTES), "source_file": "/b/y.png"}]
    )
    assert image_cache_metadata(img_a) == image_cache_metadata(img_b)


def test_audit_metadata_includes_full_provenance_without_raw_base64():
    images = normalize_image_inputs(
        [
            {
                "base64": _b64(PNG_BYTES),
                "source_id": "img-001",
                "source_file": "/tmp/foo.png",
                "modality": "image",
                "doc_id": "doc-1",
            }
        ]
    )
    [audit] = image_audit_metadata(images)
    assert audit["source_id"] == "img-001"
    assert audit["source_file"] == "/tmp/foo.png"
    assert audit["sha256"] == hashlib.sha256(PNG_BYTES).hexdigest()
    # The audit blob must never re-leak the raw base64 payload.
    assert "base64" not in audit
    assert _b64(PNG_BYTES) not in str(audit)
