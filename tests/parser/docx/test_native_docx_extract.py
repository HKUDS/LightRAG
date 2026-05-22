"""Unit tests for native docx tracked-change / comment / empty-table handling.

Locks in the contract that:
- ``w:ins`` content survives (final revised text),
- ``w:del`` / ``w:moveFrom`` content is dropped,
- ``w:commentRangeStart`` / ``w:commentRangeEnd`` / ``w:commentReference`` /
  ``w:annotationRef`` markers are dropped,
- tables whose every cell is whitespace-only are omitted from the parser
  output (no ``<table>`` placeholder, so no IRTable downstream).

These were previously emergent properties (skip lists in two places + a
run-level white-list); regressions now fail loudly.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from docx import Document
from docx.oxml.ns import qn
from lxml import etree

from lightrag.parser.docx.parse_document import (
    extract_docx_blocks,
    extract_paragraph_content,
)
from lightrag.parser.docx.table_extractor import extract_paragraph_content_table

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
PARAGRAPH_NS = {
    "w": W_NS,
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def _p(inner_xml: str):
    """Build a ``<w:p>`` lxml element from inner OOXML."""
    return etree.fromstring(f'<w:p xmlns:w="{W_NS}">{inner_xml}</w:p>'.encode("utf-8"))


# --- paragraph walker (parse_document.extract_paragraph_content) -----------


@pytest.mark.offline
def test_paragraph_keeps_w_ins_content() -> None:
    para = _p("<w:ins><w:r><w:t>kept</w:t></w:r></w:ins>")

    assert extract_paragraph_content(para, PARAGRAPH_NS) == "kept"


@pytest.mark.offline
def test_paragraph_drops_w_del_content() -> None:
    para = _p("<w:del><w:r><w:t>removed</w:t></w:r></w:del>")

    assert extract_paragraph_content(para, PARAGRAPH_NS) == ""


@pytest.mark.offline
def test_paragraph_drops_w_movefrom_content() -> None:
    para = _p("<w:moveFrom><w:r><w:t>moved away</w:t></w:r></w:moveFrom>")

    assert extract_paragraph_content(para, PARAGRAPH_NS) == ""


@pytest.mark.offline
def test_paragraph_drops_comment_markers_but_keeps_surrounding_text() -> None:
    para = _p(
        '<w:commentRangeStart w:id="0"/>'
        "<w:r><w:t>visible</w:t></w:r>"
        '<w:commentRangeEnd w:id="0"/>'
        "<w:r>"
        '  <w:rPr><w:rStyle w:val="CommentReference"/></w:rPr>'
        '  <w:commentReference w:id="0"/>'
        "</w:r>"
    )

    assert extract_paragraph_content(para, PARAGRAPH_NS) == "visible"


# --- table-cell walker (table_extractor.extract_paragraph_content_table) ---


@pytest.mark.offline
def test_table_cell_keeps_w_ins_content() -> None:
    para = _p("<w:ins><w:r><w:t>kept</w:t></w:r></w:ins>")

    assert extract_paragraph_content_table(para, qn) == "kept"


@pytest.mark.offline
def test_table_cell_drops_w_del_content() -> None:
    para = _p("<w:del><w:r><w:t>removed</w:t></w:r></w:del>")

    assert extract_paragraph_content_table(para, qn) == ""


@pytest.mark.offline
def test_table_cell_drops_comment_markers() -> None:
    para = _p(
        '<w:commentRangeStart w:id="0"/>'
        "<w:r><w:t>cell-text</w:t></w:r>"
        '<w:commentRangeEnd w:id="0"/>'
    )

    assert extract_paragraph_content_table(para, qn) == "cell-text"


# --- end-to-end: empty tables vanish from blocks output --------------------


def _populate_cell(cell, text: str) -> None:
    cell.paragraphs[0].text = text


def _build_docx_with_three_tables() -> BytesIO:
    """One real table, one all-empty table, one whitespace-only table.

    Returns a BytesIO containing the rendered .docx so it can be fed to
    ``extract_docx_blocks`` via a tempfile-backed path or directly.
    """
    doc = Document()
    doc.add_paragraph("intro")

    real = doc.add_table(rows=1, cols=2)
    _populate_cell(real.rows[0].cells[0], "A")
    _populate_cell(real.rows[0].cells[1], "B")

    empty = doc.add_table(rows=2, cols=2)
    # leave every cell at python-docx default (empty paragraph)
    assert all(c.text == "" for row in empty.rows for c in row.cells)

    whitespace = doc.add_table(rows=1, cols=2)
    _populate_cell(whitespace.rows[0].cells[0], "   ")
    _populate_cell(whitespace.rows[0].cells[1], "\t")

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


@pytest.mark.offline
def test_empty_tables_are_skipped(tmp_path) -> None:
    docx_bytes = _build_docx_with_three_tables()
    docx_path = tmp_path / "three_tables.docx"
    docx_path.write_bytes(docx_bytes.getvalue())

    blocks = extract_docx_blocks(str(docx_path))

    table_placeholders = sum(b["content"].count("<table>") for b in blocks)
    assert table_placeholders == 1, (
        "exactly one real table should survive; empty + whitespace tables "
        "must be dropped before the placeholder is emitted"
    )
    assert any('"A"' in b["content"] and '"B"' in b["content"] for b in blocks)
