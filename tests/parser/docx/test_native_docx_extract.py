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
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree

from lightrag.parser.docx.parse_document import (
    DocxContentError,
    extract_docx_blocks,
    extract_paragraph_content,
)
from lightrag.parser.docx.smart_heading.features import StyleAttributes
from lightrag.parser.docx.table_extractor import (
    TableExtractor,
    extract_paragraph_content_table,
)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
PARAGRAPH_NS = {
    "w": W_NS,
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def _p(inner_xml: str):
    """Build a ``<w:p>`` lxml element from inner OOXML."""
    return etree.fromstring(f'<w:p xmlns:w="{W_NS}">{inner_xml}</w:p>'.encode("utf-8"))


@pytest.mark.offline
@pytest.mark.parametrize("in_table", [False, True])
def test_direct_list_level_with_style_inherited_numid(tmp_path, in_table) -> None:
    doc = Document()
    style = doc.styles.add_style("InheritedList", WD_STYLE_TYPE.PARAGRAPH)
    style.element.get_or_add_pPr().append(
        etree.fromstring(f'<w:numPr xmlns:w="{W_NS}"><w:numId w:val="42"/></w:numPr>')
    )
    doc.part.numbering_part.element.append(
        etree.fromstring(
            f'<w:abstractNum xmlns:w="{W_NS}" w:abstractNumId="42">'
            '<w:lvl w:ilvl="0"><w:start w:val="1"/>'
            '<w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/>'
            '<w:pStyle w:val="InheritedList"/></w:lvl>'
            '<w:lvl w:ilvl="1"><w:start w:val="1"/>'
            '<w:numFmt w:val="lowerLetter"/><w:lvlText w:val="%2)"/>'
            "</w:lvl></w:abstractNum>"
        )
    )
    doc.part.numbering_part.element.append(
        etree.fromstring(
            f'<w:num xmlns:w="{W_NS}" w:numId="42">'
            '<w:abstractNumId w:val="42"/></w:num>'
        )
    )
    container = doc.add_table(rows=1, cols=1).cell(0, 0) if in_table else doc
    for text, level in [("nested", 1), ("next nested", 1), ("top", None)]:
        para = container.add_paragraph(text, style="InheritedList")
        if level is not None:
            para._p.get_or_add_pPr().append(
                etree.fromstring(
                    f'<w:numPr xmlns:w="{W_NS}"><w:ilvl w:val="{level}"/></w:numPr>'
                )
            )
    path = tmp_path / "inherited-list.docx"
    doc.save(path)

    content = "\n".join(block["content"] for block in extract_docx_blocks(str(path)))
    assert "a) nested" in content
    assert "b) next nested" in content
    assert "2. top" in content


@pytest.mark.offline
@pytest.mark.parametrize("in_table", [False, True])
def test_style_list_level_with_numid_from_base_style(tmp_path, in_table) -> None:
    # A child style writes only w:ilvl and inherits w:numId from its base style;
    # Word renders "1. a) b) 2." for this layout.
    doc = Document()
    base = doc.styles.add_style("ListBase", WD_STYLE_TYPE.PARAGRAPH)
    base.element.get_or_add_pPr().append(
        etree.fromstring(f'<w:numPr xmlns:w="{W_NS}"><w:numId w:val="43"/></w:numPr>')
    )
    child = doc.styles.add_style("ListChild", WD_STYLE_TYPE.PARAGRAPH)
    child.base_style = base
    child.element.get_or_add_pPr().append(
        etree.fromstring(f'<w:numPr xmlns:w="{W_NS}"><w:ilvl w:val="1"/></w:numPr>')
    )
    doc.part.numbering_part.element.append(
        etree.fromstring(
            f'<w:abstractNum xmlns:w="{W_NS}" w:abstractNumId="43">'
            '<w:lvl w:ilvl="0"><w:start w:val="1"/>'
            '<w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/></w:lvl>'
            '<w:lvl w:ilvl="1"><w:start w:val="1"/>'
            '<w:numFmt w:val="lowerLetter"/><w:lvlText w:val="%2)"/>'
            "</w:lvl></w:abstractNum>"
        )
    )
    doc.part.numbering_part.element.append(
        etree.fromstring(
            f'<w:num xmlns:w="{W_NS}" w:numId="43">'
            '<w:abstractNumId w:val="43"/></w:num>'
        )
    )
    container = doc.add_table(rows=1, cols=1).cell(0, 0) if in_table else doc
    for text, style in [
        ("parent", "ListBase"),
        ("child", "ListChild"),
        ("next child", "ListChild"),
        ("next parent", "ListBase"),
    ]:
        container.add_paragraph(text, style=style)
    path = tmp_path / "style-level-list.docx"
    doc.save(path)

    content = "\n".join(block["content"] for block in extract_docx_blocks(str(path)))
    assert "1. parent" in content
    assert "a) child" in content
    assert "b) next child" in content
    assert "2. next parent" in content


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


def _pm(inner_xml: str):
    """Build a ``<w:p>`` element that also declares the ``m`` (math) namespace."""
    return etree.fromstring(
        f'<w:p xmlns:w="{W_NS}" xmlns:m="{PARAGRAPH_NS["m"]}">{inner_xml}</w:p>'.encode(
            "utf-8"
        )
    )


@pytest.mark.offline
def test_table_cell_converts_inline_equation() -> None:
    # Regression: the m:oMath branch used a bare ``from omml import`` (absolute)
    # that raised ModuleNotFoundError once a table cell actually contained an
    # equation. It must resolve the package-relative ``.omml`` module instead.
    para = _pm(
        "<w:r><w:t>before </w:t></w:r>"
        "<m:oMath><m:r><m:t>x+1</m:t></m:r></m:oMath>"
        "<w:r><w:t> after</w:t></w:r>"
    )

    assert (
        extract_paragraph_content_table(para, qn)
        == "before <equation>x+1</equation> after"
    )


@pytest.mark.offline
def test_table_cell_converts_block_equation() -> None:
    para = _pm(
        "<m:oMathPara><m:oMath><m:r><m:t>E=mc^2</m:t></m:r></m:oMath></m:oMathPara>"
    )

    assert extract_paragraph_content_table(para, qn) == "<equation>E=mc^2</equation>"


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
@pytest.mark.parametrize("continuation", [None, "continue"])
def test_empty_table_cell_ends_vertical_merge(continuation: str | None) -> None:
    document = Document()
    table = document.add_table(rows=3, cols=2)
    merged = table.cell(0, 0).merge(table.cell(1, 0))
    _populate_cell(merged, "Operations")
    _populate_cell(table.cell(0, 1), "Chair")
    _populate_cell(table.cell(1, 1), "Desk")
    _populate_cell(table.cell(2, 1), "Unassigned clipboard")

    restart, continued, independent = (row.tc_lst[0] for row in table._tbl.tr_lst)
    merge_marker = continued.tcPr.find(qn("w:vMerge"))
    if continuation is not None:
        merge_marker.set(qn("w:val"), continuation)
    para_ids = ("00000001", "00000002", "00000003")
    for cell, para_id in zip((restart, continued, independent), para_ids):
        cell.find(qn("w:p")).set(qn("w14:paraId"), para_id)

    source = BytesIO()
    document.save(source)
    source.seek(0)
    result = TableExtractor.extract_with_metadata(Document(source).tables[0])

    assert result["rows"] == [
        ["Operations", "Chair"],
        ["Operations", "Desk"],
        ["", "Unassigned clipboard"],
    ]
    assert [row[0] for row in result["para_ids"]] == [
        para_ids[0],
        para_ids[0],
        para_ids[2],
    ]
    assert [row[0] for row in result["para_ids_end"]] == list(para_ids)


def _filled_table(document, rows: int, cols: int):
    table = document.add_table(rows=rows, cols=cols)
    for row in range(rows):
        for column in range(cols):
            _populate_cell(table.cell(row, column), f"r{row}c{column}")
    return table


def _skip_grid_columns(row, *, before: int = 0, after: int = 0) -> None:
    """Drop cells from a row's ends and record them as gridBefore/gridAfter."""
    tr = row._tr
    for _ in range(before):
        tr.remove(tr.tc_lst[0])
    for _ in range(after):
        tr.remove(tr.tc_lst[-1])
    tr_pr = tr.get_or_add_trPr()
    for tag, count in (("w:gridBefore", before), ("w:gridAfter", after)):
        if count:
            node = OxmlElement(tag)
            node.set(qn("w:val"), str(count))
            tr_pr.append(node)


def _saved_table_metadata(document, **kwargs) -> dict:
    source = BytesIO()
    document.save(source)
    source.seek(0)
    return TableExtractor.extract_with_metadata(Document(source).tables[0], **kwargs)


@pytest.mark.offline
def test_grid_before_row_keeps_cells_in_their_columns() -> None:
    document = Document()
    table = _filled_table(document, 3, 3)
    _skip_grid_columns(table.rows[1], before=1)
    _skip_grid_columns(table.rows[2], after=2)
    skipped_row = table.rows[1]._tr
    for tc, para_id in zip(skipped_row.tc_lst, ("00000011", "00000012")):
        tc.find(qn("w:p")).set(qn("w14:paraId"), para_id)

    result = _saved_table_metadata(document)

    assert result["rows"] == [
        ["r0c0", "r0c1", "r0c2"],
        ["", "r1c1", "r1c2"],
        ["r2c0", "", ""],
    ]
    assert result["para_ids"][1] == [None, "00000011", "00000012"]


@pytest.mark.offline
def test_vertical_merge_below_grid_before_row() -> None:
    document = Document()
    table = _filled_table(document, 2, 3)
    table.cell(0, 2).merge(table.cell(1, 2)).text = "Total"
    _skip_grid_columns(table.rows[0], before=1)

    # The continuation in row 2 sits under grid column 2, which row 1
    # reaches only after its skipped first column.
    assert _saved_table_metadata(document)["rows"] == [
        ["", "r0c1", "Total"],
        ["r1c0", "r1c1", "Total"],
    ]


@pytest.mark.offline
def test_column_without_cell_ends_vertical_merge() -> None:
    document = Document()
    table = _filled_table(document, 3, 2)
    table.cell(0, 0).merge(table.cell(2, 0)).text = "Total"
    # Row 2 loses its continuation cell: grid column 0 has no cell there.
    _skip_grid_columns(table.rows[1], before=1)

    # Row 3's continuation has no merge directly above it to repeat.
    assert _saved_table_metadata(document)["rows"] == [
        ["Total", "r0c1"],
        ["", "r1c1"],
        ["", "r2c1"],
    ]


@pytest.mark.offline
@pytest.mark.parametrize("breaker", ["gridAfter", "normal cell"])
def test_orphaned_continuation_keeps_its_own_para_ids(breaker: str) -> None:
    document = Document()
    table = _filled_table(document, 3, 2)
    table.cell(0, 1).merge(table.cell(2, 1)).text = "Total"
    if breaker == "gridAfter":
        _skip_grid_columns(table.rows[1], after=1)
    else:
        tc_pr = table.rows[1]._tr.tc_lst[1].tcPr
        tc_pr.remove(tc_pr.find(qn("w:vMerge")))
    orphan = table.rows[2]._tr.tc_lst[1]
    paragraphs = orphan.findall(qn("w:p"))
    if len(paragraphs) == 1:
        orphan.append(OxmlElement("w:p"))
        paragraphs = orphan.findall(qn("w:p"))
    paragraphs[0].set(qn("w14:paraId"), "0000000A")
    paragraphs[-1].set(qn("w14:paraId"), "0000000B")

    result = _saved_table_metadata(document)

    # No merge reaches row 3: no text, but its paraIds still bound the range.
    assert [row[1] for row in result["rows"]] == ["Total", "", ""]
    assert result["para_ids"][2][1] == "0000000A"
    assert result["para_ids_end"][2][1] == "0000000B"


def _wrap(element, wrapper: str) -> None:
    """Move ``element`` into a content control or custom XML element."""
    parent = element.getparent()
    index = parent.index(element)
    if wrapper == "sdt":
        outer = OxmlElement("w:sdt")
        outer.append(OxmlElement("w:sdtPr"))
        inner = OxmlElement("w:sdtContent")
        outer.append(inner)
    else:
        outer = inner = OxmlElement("w:customXml")
        outer.set(qn("w:element"), "item")
    parent.remove(element)
    inner.append(element)
    parent.insert(index, outer)


@pytest.mark.offline
@pytest.mark.parametrize("wrapper", ["sdt", "customXml"])
@pytest.mark.parametrize("merged_col", [0, 1])
def test_wrapped_continuation_cell_keeps_vertical_merge(
    wrapper: str, merged_col: int
) -> None:
    document = Document()
    table = _filled_table(document, 3, 2)
    table.cell(0, merged_col).merge(table.cell(2, merged_col)).text = "Total"
    _wrap(table.rows[1]._tr.tc_lst[merged_col], wrapper)

    rows = _saved_table_metadata(document)["rows"]

    other = 1 - merged_col
    assert [row[merged_col] for row in rows] == ["Total"] * 3
    assert [row[other] for row in rows] == [f"r{i}c{other}" for i in range(3)]


@pytest.mark.offline
@pytest.mark.parametrize("wrapper", ["sdt", "customXml"])
def test_wrapped_row_counts_like_a_direct_row(wrapper: str) -> None:
    document = Document()
    table = _filled_table(document, 3, 2)
    table.rows[2]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    _wrap(table.rows[1]._tr, wrapper)

    result = _saved_table_metadata(document)

    assert result["rows"] == [
        ["r0c0", "r0c1"],
        ["r1c0", "r1c1"],
        ["r2c0", "r2c1"],
    ]
    assert result["header_indices"] == [2]


def _wrap_in_placeholder(element) -> None:
    """Wrap ``element`` in a content control still showing its placeholder."""
    _wrap(element, "sdt")
    element.getparent().getparent().find(qn("w:sdtPr")).append(
        OxmlElement("w:showingPlcHdr")
    )


@pytest.mark.offline
def test_placeholder_cell_keeps_its_column_without_text() -> None:
    document = Document()
    table = _filled_table(document, 2, 3)
    _wrap_in_placeholder(table.rows[0]._tr.tc_lst[1])

    result = _saved_table_metadata(document, style_attributes=StyleAttributes())

    assert result["rows"] == [["r0c0", "", "r0c2"], ["r1c0", "r1c1", "r1c2"]]
    assert result["cell_features"][0][1] == ("", None, False)


@pytest.mark.offline
def test_placeholder_row_is_left_out() -> None:
    document = Document()
    table = _filled_table(document, 3, 2)
    _wrap_in_placeholder(table.rows[1]._tr)

    assert _saved_table_metadata(document)["rows"] == [
        ["r0c0", "r0c1"],
        ["r2c0", "r2c1"],
    ]


@pytest.mark.offline
@pytest.mark.parametrize("with_numbering", [False, True])
def test_placeholder_cell_content_is_not_extracted(
    monkeypatch, with_numbering: bool
) -> None:
    """Nothing in a hidden cell is extracted: no image export, no list count."""
    import lightrag.parser.docx.table_extractor as table_extractor

    document = Document()
    table = _filled_table(document, 1, 2)
    hidden_tc = table.rows[0]._tr.tc_lst[1]
    hidden_tc.find(qn("w:p")).set(qn("w14:paraId"), "0000000C")
    _wrap_in_placeholder(hidden_tc)

    touched: list[str] = []
    real_extract = table_extractor.extract_paragraph_content_table

    def recording_extract(para_elem, *args, **kwargs):
        text = real_extract(para_elem, *args, **kwargs)
        touched.append(text)
        return text

    class RecordingResolver:
        def reset_tracking_state(self) -> None:
            pass

        def get_label(self, para_elem) -> str:
            touched.append("label:" + real_extract(para_elem, qn))
            return ""

    monkeypatch.setattr(
        table_extractor, "extract_paragraph_content_table", recording_extract
    )
    result = _saved_table_metadata(
        document, numbering_resolver=RecordingResolver() if with_numbering else None
    )

    assert result["rows"] == [["r0c0", ""]]
    assert result["para_ids"][0][1] == "0000000C"
    assert all("r0c1" not in entry for entry in touched)
    assert any("r0c0" in entry for entry in touched)


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

    # The native parser splits at every heading level and never does
    # token-based splitting or small-block merging (that is the chunker's job).
    blocks = extract_docx_blocks(str(docx_path))

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

    blocks = extract_docx_blocks(str(docx_path))

    summary = [(b["heading"], b["content"].split("\n")[0], b["level"]) for b in blocks]
    assert summary == [
        ("Top", "# Top", 1),
        # level 7 outline → clamped to six "#".
        ("Deep Seven", "###### Deep Seven", 7),
    ]


def _add_outline_para_with_softbreak(doc, head_text: str, body_text: str, level: int):
    """Append one ``<w:p>`` carrying an outline level whose runs are
    ``head_text`` + a soft line break (``<w:br/>``) + ``body_text``.

    Mirrors the real-world WPS/Word pattern where an author sets an outline
    level on a paragraph but types the body with Shift+Enter (a soft break)
    instead of a new paragraph, so heading + body live in a single paragraph.
    """
    para = doc.add_paragraph(head_text)
    pPr = para._p.get_or_add_pPr()
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), str(level - 1))
    pPr.append(outline)
    para.add_run().add_break()  # default WD_BREAK.LINE → soft <w:br/> → '\n'
    para.add_run(body_text)
    return para


@pytest.mark.offline
def test_oversize_outline_heading_with_softbreak_splits(tmp_path) -> None:
    """An over-long heading paragraph that contains a soft break must split:
    the first line stays the heading, the remainder becomes body text. No
    DocxContentError, and the body content is preserved in full."""
    head = "A、处置程序"
    body = "施工现场一旦发生事故时应立即采取相应的应急处置措施并以最快速度报警。" * 8
    assert len(head) + 1 + len(body) > 200  # would trip validate_heading_length

    doc = Document()
    _add_outline_para_with_softbreak(doc, head, body, level=4)

    buf = BytesIO()
    doc.save(buf)
    docx_path = tmp_path / "softbreak.docx"
    docx_path.write_bytes(buf.getvalue())

    blocks = extract_docx_blocks(str(docx_path))

    # The short first line becomes the heading; the long remainder is body.
    heading_block = next(b for b in blocks if b["heading"] == head)
    assert body in heading_block["content"]
    # Heading field carries only the first line, not the body.
    assert body not in heading_block["heading"]


@pytest.mark.offline
def test_oversize_outline_heading_no_softbreak_demoted_to_body(tmp_path) -> None:
    """A single-line over-long outline paragraph (no soft break) is demoted to
    body text rather than raising — content preserved, not treated as a heading."""
    long_text = "施工现场应急处置措施说明，" * 30  # well over 200 chars, no '\n'
    assert len(long_text) > 200

    doc = Document()
    _add_heading(doc, long_text, level=2)

    buf = BytesIO()
    doc.save(buf)
    docx_path = tmp_path / "demote.docx"
    docx_path.write_bytes(buf.getvalue())

    blocks = extract_docx_blocks(str(docx_path))

    # The text survives as body content, and no block adopts it as a heading.
    assert any(long_text in b["content"] for b in blocks)
    assert all(b["heading"] != long_text for b in blocks)


@pytest.mark.offline
def test_short_outline_paragraph_still_heading(tmp_path) -> None:
    """A within-limit outline paragraph must still be recognized as a heading —
    the over-long handling never demotes short headings."""
    doc = Document()
    _add_heading(doc, "处置程序", level=3)
    doc.add_paragraph("body under heading.")

    buf = BytesIO()
    doc.save(buf)
    docx_path = tmp_path / "short.docx"
    docx_path.write_bytes(buf.getvalue())

    blocks = extract_docx_blocks(str(docx_path))

    heading_block = next(b for b in blocks if b["heading"] == "处置程序")
    assert heading_block["level"] == 3
    assert heading_block["content"].startswith("### 处置程序")


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
    blocks = extract_docx_blocks(str(docx_path), parse_metadata=parse_metadata)

    assert parse_metadata["first_heading"] == "Already MD"
    assert blocks[0]["heading"] == "Already MD"
    assert blocks[0]["parent_headings"] == []
    assert blocks[0]["content"].splitlines()[0] == "# Already MD"


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


# --- invalid (non-ZIP) .docx files surface an accurate error ---------------


@pytest.mark.offline
@pytest.mark.parametrize(
    ("header", "format_hint"),
    [
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"junk", "Word 97-2003"),
        (b"{\\rtf1\\ansi}", "RTF"),
        (b"%PDF-1.7\n%binary", "PDF"),
        (b"<?xml version='1.0'?><html></html>", "HTML or XML"),
        (b"plain text masquerading as docx", "not a valid DOCX"),
    ],
)
def test_invalid_docx_raises_accurate_error(tmp_path, header, format_hint) -> None:
    """A file that exists but is not a ZIP/OOXML package must fail with an
    accurate message — never the misleading python-docx "Package not found".

    The native worker confirms the file exists before parsing, so the only way
    ``PackageNotFoundError`` can fire here is a corrupt file or a non-DOCX
    payload wearing a .docx extension. The error must name the real format and
    keep the file path.
    """
    docx_path = tmp_path / "masquerade.docx"
    docx_path.write_bytes(header)

    with pytest.raises(DocxContentError) as excinfo:
        extract_docx_blocks(str(docx_path))

    message = str(excinfo.value)
    assert "Package not found" not in message
    assert "valid DOCX" in message
    assert str(docx_path) in message
    assert format_hint in message


@pytest.mark.offline
def test_empty_docx_file_reports_truncation(tmp_path) -> None:
    """A 0-byte .docx must be diagnosed as empty/truncated, not 'not found'."""
    docx_path = tmp_path / "empty.docx"
    docx_path.write_bytes(b"")

    with pytest.raises(DocxContentError) as excinfo:
        extract_docx_blocks(str(docx_path))

    message = str(excinfo.value)
    assert "Package not found" not in message
    assert "empty" in message
    assert str(docx_path) in message
