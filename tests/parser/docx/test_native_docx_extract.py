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
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree

from lightrag.parser.docx import parse_document as parse_doc
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


# --- end-to-end: every heading becomes its own block ----------------------


def _add_heading(doc, text: str, level: int) -> None:
    """Append a heading paragraph with an explicit ``w:outlineLvl``.

    Setting outlineLvl directly on the paragraph (rather than relying on the
    template's built-in Heading styles) keeps ``get_heading_level`` detection
    deterministic. ``level`` is 1-based (1 = H1); outlineLvl val is 0-based.
    """
    para = doc.add_paragraph(text)
    pPr = para._p.get_or_add_pPr()
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), str(level - 1))
    pPr.append(outline)


@pytest.mark.offline
def test_each_heading_becomes_its_own_block(tmp_path) -> None:
    """Headings with no body must each form a standalone block.

    Layout: H1 (no body) → H2 (no body) → H3 (with body) → H1 (no body, last).
    Previously the empty H1/H2 were folded into the next heading's block; now
    every recognized heading starts its own block. A heading with no following
    body yields a block whose ``content`` is just the heading text, while a
    heading with body merges that body into the same block. ``parent_headings``
    must track the outline hierarchy correctly.
    """
    doc = Document()
    _add_heading(doc, "Chapter One", level=1)
    _add_heading(doc, "Section 1.1", level=2)
    _add_heading(doc, "Subsection 1.1.1", level=3)
    doc.add_paragraph("Body text under subsection.")
    _add_heading(doc, "Appendix", level=1)

    buf = BytesIO()
    doc.save(buf)
    docx_path = tmp_path / "headings.docx"
    docx_path.write_bytes(buf.getvalue())

    # fixlevel=0 mirrors the production path (pipeline.py): split at every
    # heading level, no token-based splitting or small-block merging.
    blocks = extract_docx_blocks(str(docx_path), fixlevel=0)

    # The content line carries a markdown ``#`` prefix matching the level,
    # while the ``heading`` field stays clean (no prefix).
    summary = [
        (b["heading"], b["content"], b["level"], b["parent_headings"]) for b in blocks
    ]
    assert summary == [
        ("Chapter One", "# Chapter One", 1, []),
        ("Section 1.1", "## Section 1.1", 2, ["Chapter One"]),
        (
            "Subsection 1.1.1",
            "### Subsection 1.1.1\nBody text under subsection.",
            3,
            ["Chapter One", "Section 1.1"],
        ),
        ("Appendix", "# Appendix", 1, []),
    ]


@pytest.mark.offline
def test_heading_markdown_prefix_capped_at_six(tmp_path) -> None:
    """Heading content lines get a markdown ``#`` prefix matching the level,
    capped at 6 ``#`` (a level-7 heading still renders ``######``)."""
    doc = Document()
    _add_heading(doc, "Top", level=1)
    doc.add_paragraph("body under top.")
    _add_heading(doc, "Deep Seven", level=7)
    doc.add_paragraph("body under deep.")

    buf = BytesIO()
    doc.save(buf)
    docx_path = tmp_path / "deep.docx"
    docx_path.write_bytes(buf.getvalue())

    blocks = extract_docx_blocks(str(docx_path), fixlevel=0)

    summary = [(b["heading"], b["content"].split("\n")[0], b["level"]) for b in blocks]
    assert summary == [
        ("Top", "# Top", 1),
        # level 7 outline → clamped to six "#".
        ("Deep Seven", "###### Deep Seven", 7),
    ]


@pytest.mark.offline
def test_existing_markdown_heading_keeps_content_but_metadata_is_clean(
    tmp_path,
) -> None:
    doc = Document()
    _add_heading(doc, "# Already MD", level=1)
    doc.add_paragraph("Body.")

    buf = BytesIO()
    doc.save(buf)
    docx_path = tmp_path / "markdown-heading.docx"
    docx_path.write_bytes(buf.getvalue())

    parse_metadata: dict[str, str] = {}
    blocks = extract_docx_blocks(
        str(docx_path), fixlevel=0, parse_metadata=parse_metadata
    )

    assert parse_metadata["first_heading"] == "Already MD"
    assert blocks[0]["heading"] == "Already MD"
    assert blocks[0]["parent_headings"] == []
    assert blocks[0]["content"].splitlines()[0] == "# Already MD"


@pytest.mark.offline
def test_split_long_block_promoted_markdown_anchor_metadata_is_clean(
    monkeypatch,
) -> None:
    monkeypatch.setattr(parse_doc, "MAX_BLOCK_CONTENT_TOKENS", 25)
    monkeypatch.setattr(parse_doc, "IDEAL_BLOCK_CONTENT_TOKENS", 20)
    monkeypatch.setattr(parse_doc, "estimate_tokens", len)

    blocks = parse_doc.split_long_block(
        "Top",
        [
            {"text": "aaaaa", "para_id": "p1", "is_table": False},
            {"text": "bbbbbbbbbb", "para_id": "p2", "is_table": False},
            {"text": "# Anchor", "para_id": "p3", "is_table": False},
            {"text": "cccccccccc", "para_id": "p4", "is_table": False},
        ],
        parent_headings=[],
        block_level=1,
    )

    assert blocks[1]["heading"] == "Anchor"
    assert blocks[1]["parent_headings"] == ["Top"]
    assert blocks[1]["content"].splitlines()[0] == "# Anchor"


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
