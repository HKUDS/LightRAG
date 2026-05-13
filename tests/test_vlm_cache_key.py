"""Offline tests for the VLM cache-key invariants used by analyze_multimodal.

These tests verify the hash inputs we feed into ``compute_args_hash`` actually
deliver the contract documented in the LLM/VLM vision plan:

- same prompt + same image content => cache HIT (identical args_hash)
- same prompt + different image content => cache MISS (different args_hash)
- same prompt + same image content under a different file path/source_id =>
  cache HIT (provenance is for audit only and must not affect the hash)
- the audit blob written into ``original_prompt`` never embeds the raw base64
  payload, only digests and provenance pointers
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from lightrag.llm._vision_utils import (
    image_audit_metadata,
    image_cache_metadata,
    normalize_image_inputs,
)
from lightrag.utils import (
    _serialize_cache_variant,
    compute_args_hash,
    get_llm_cache_identity,
    serialize_llm_cache_identity,
)


pytestmark = pytest.mark.offline


PNG_A = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8"
    b"\xcf\xc0\x00\x00\x00\x03\x00\x01\x5c\xcc\xd9\x9e\x00\x00\x00\x00"
    b"IEND\xaeB`\x82"
)
PNG_B = PNG_A[:-12] + b"\x01" + PNG_A[-11:]  # 1-byte tweak => different hash


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _hash_for(prompt: str, images: list[dict[str, Any]] | None) -> str:
    normalized = normalize_image_inputs(images) if images else []
    identity = get_llm_cache_identity({}, role="vlm")
    return compute_args_hash(
        prompt,
        "",
        "",
        serialize_llm_cache_identity(identity),
        _serialize_cache_variant({"type": "json_object"}),
        _serialize_cache_variant(image_cache_metadata(normalized)),
    )


def test_same_prompt_same_image_yields_same_hash():
    h1 = _hash_for("describe", [{"base64": _b64(PNG_A)}])
    h2 = _hash_for("describe", [{"base64": _b64(PNG_A)}])
    assert h1 == h2


def test_same_prompt_different_image_yields_different_hash():
    h1 = _hash_for("describe", [{"base64": _b64(PNG_A)}])
    h2 = _hash_for("describe", [{"base64": _b64(PNG_B)}])
    assert h1 != h2


def test_same_image_different_source_file_still_hits():
    h1 = _hash_for(
        "describe",
        [
            {
                "base64": _b64(PNG_A),
                "source_id": "img-001",
                "source_file": "/path/a/img.png",
                "modality": "image",
                "doc_id": "doc-1",
            }
        ],
    )
    h2 = _hash_for(
        "describe",
        [
            {
                "base64": _b64(PNG_A),
                "source_id": "img-002",
                "source_file": "/different/elsewhere/copy.png",
                "modality": "image",
                "doc_id": "doc-2",
            }
        ],
    )
    assert h1 == h2


def test_different_prompt_with_same_image_yields_different_hash():
    h1 = _hash_for("describe", [{"base64": _b64(PNG_A)}])
    h2 = _hash_for("describe in english", [{"base64": _b64(PNG_A)}])
    assert h1 != h2


def test_image_present_vs_absent_yields_different_hash():
    h_text_only = _hash_for("describe", None)
    h_with_image = _hash_for("describe", [{"base64": _b64(PNG_A)}])
    assert h_text_only != h_with_image


def test_audit_block_in_original_prompt_does_not_leak_raw_base64():
    """Mirrors how _analyze_item builds the cache-entry original_prompt."""
    normalized = normalize_image_inputs(
        [
            {
                "base64": _b64(PNG_A),
                "source_id": "img-001",
                "source_file": "/tmp/a.png",
                "modality": "image",
                "doc_id": "doc-1",
            }
        ]
    )
    audit_blob = image_audit_metadata(normalized)
    prompt = "describe"
    original_prompt = (
        prompt
        + f"\n<vlm_images>{json.dumps(audit_blob, ensure_ascii=False)}</vlm_images>"
    )

    assert "<vlm_images>" in original_prompt
    assert "</vlm_images>" in original_prompt
    # sha256 digest is present; raw base64 must not be.
    assert audit_blob[0]["sha256"] in original_prompt
    assert _b64(PNG_A) not in original_prompt
