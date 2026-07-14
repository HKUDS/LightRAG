"""doc_title resolution in the native DOCX IR builder.

An explicit ``parse_metadata["doc_title"]`` (written only by the smart
assembler) wins outright — an EMPTY string is a real answer ("no title block
identified"), not a missing one. Without the key the legacy chain holds:
``first_heading``, else the file stem.
"""

from __future__ import annotations

import pytest

from lightrag.parser.docx.ir_builder import NativeDocxIRBuilder

pytestmark = pytest.mark.offline


def _normalize(parse_metadata):
    return NativeDocxIRBuilder().normalize(
        [
            {
                "uuid": "p1",
                "uuid_end": "p1",
                "heading": "前言",
                "content": "正文一句。",
                "parent_headings": [],
                "level": 1,
            }
        ],
        document_name="doc.docx",
        asset_dir_name="doc.blocks.assets",
        parse_metadata=parse_metadata,
    )


def test_explicit_doc_title_wins() -> None:
    ir = _normalize({"doc_title": "某某管理办法", "first_heading": "前言"})
    assert ir.doc_title == "某某管理办法"


def test_explicit_empty_doc_title_beats_every_fallback() -> None:
    ir = _normalize({"doc_title": "", "first_heading": "前言"})
    assert ir.doc_title == ""


def test_legacy_first_heading_chain_without_the_key() -> None:
    assert _normalize({"first_heading": "前言"}).doc_title == "前言"


def test_legacy_stem_fallback_without_any_metadata() -> None:
    assert _normalize({}).doc_title == "doc"
    assert _normalize(None).doc_title == "doc"
