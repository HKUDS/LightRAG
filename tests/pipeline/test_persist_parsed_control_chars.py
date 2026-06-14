"""Injection point B: ``LightRAG._persist_parsed_full_docs`` strips C0
control/separator chars from the parsed body before it lands in full_docs.

This is the single convergence point for every parser engine's persist. For
RAW (legacy) the full_docs content IS the chunk source, so cleaning here is
what keeps \\x1c-\\x1f out of downstream chunks. The content_hash must be
derived from the cleaned body so dedup stays stable.

Exercised against the real method via the shared debug stand-in
(``lightrag.parser.debug.build_debug_rag``) so no real storage backends are
needed.
"""

import pytest

from lightrag.constants import FULL_DOCS_FORMAT_RAW
from lightrag.parser.debug import build_debug_rag
from lightrag.utils_pipeline import (
    compute_text_content_hash,
    strip_lightrag_doc_prefix,
)

pytestmark = pytest.mark.offline


async def test_persist_strips_control_chars_from_raw_content():
    rag = build_debug_rag()
    dirty = "head\x1cbody\x1d中\x1f文\x00\x7f tail"

    await rag._persist_parsed_full_docs(
        "doc-raw",
        {
            "content": dirty,
            "file_path": "f.pdf",
            "parse_format": FULL_DOCS_FORMAT_RAW,
            "parse_engine": "legacy",
        },
    )

    stored = rag.full_docs.data["doc-raw"]
    assert not any(c in stored["content"] for c in "\x1c\x1d\x1f\x00\x7f")
    assert stored["content"] == "headbody中文 tail"


async def test_persist_content_hash_matches_cleaned_body():
    rag = build_debug_rag()
    dirty = "alpha\x1ebeta\x1f"

    await rag._persist_parsed_full_docs(
        "doc-hash",
        {
            "content": dirty,
            "file_path": "f.pdf",
            "parse_format": FULL_DOCS_FORMAT_RAW,
            "parse_engine": "legacy",
        },
    )

    stored = rag.full_docs.data["doc-hash"]
    expected = compute_text_content_hash(
        strip_lightrag_doc_prefix("alphabeta", FULL_DOCS_FORMAT_RAW)
    )
    assert stored["content_hash"] == expected


async def test_persist_noop_for_clean_content():
    rag = build_debug_rag()
    clean = "perfectly clean body\nwith newline\ttab"

    await rag._persist_parsed_full_docs(
        "doc-clean",
        {
            "content": clean,
            "file_path": "f.pdf",
            "parse_format": FULL_DOCS_FORMAT_RAW,
            "parse_engine": "legacy",
        },
    )

    assert rag.full_docs.data["doc-clean"]["content"] == clean
