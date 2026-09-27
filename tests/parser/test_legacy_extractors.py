"""Tests for legacy text extraction helpers."""

from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

import pytest
from docx import Document
from docx.oxml import parse_xml
from openpyxl import Workbook
from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.util import Inches

from lightrag.parser.legacy.extractors import extract_text


_NS_URI = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS = {"main": _NS_URI}


def _inject_cached_value(
    data: bytes, cell_ref: str, cached_value: str | int | float
) -> bytes:
    """Patch a worksheet's XML so ``cell_ref`` carries a cached ``<v>`` value.

    ``openpyxl`` writes formula expressions but never a cached calculated value,
    so we inject one to exercise the ``data_only=True`` read path. String results
    are tagged ``t="str"`` to match how Excel records a text-valued formula.
    """

    root = ET.fromstring(data)
    for cell in root.findall(".//main:c", _NS):
        if cell.attrib.get("r") != cell_ref:
            continue
        if isinstance(cached_value, str):
            cell.set("t", "str")
        value_node = cell.find("main:v", _NS)
        if value_node is None:
            value_node = ET.SubElement(cell, f"{{{_NS_URI}}}v")
        value_node.text = str(cached_value)
        break
    return ET.tostring(root, encoding="utf-8", xml_declaration=False)


def _patch_xlsx(
    file_bytes: bytes, injections: dict[str, tuple[str, str | int | float]]
) -> bytes:
    """Rewrite worksheet parts in ``file_bytes`` with cached formula values.

    ``injections`` maps a worksheet part name (e.g. ``xl/worksheets/sheet1.xml``)
    to a ``(cell_ref, cached_value)`` pair.
    """

    source = BytesIO(file_bytes)
    patched = BytesIO()
    with ZipFile(source, "r") as zin, ZipFile(patched, "w", ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in injections:
                cell_ref, cached_value = injections[item.filename]
                data = _inject_cached_value(data, cell_ref, cached_value)
            zout.writestr(item, data)
    return patched.getvalue()


def _make_xlsx_bytes(*, cached_formula_value: str | int | float | None) -> bytes:
    """Build a minimal single-sheet workbook with one formula cell.

    ``openpyxl`` writes the formula expression, but not a cached calculated
    value. When ``cached_formula_value`` is given we patch the worksheet XML so
    the extractor exercises the ``data_only=True`` path; when it is ``None`` the
    workbook carries no cache and the formula-text fallback path is exercised.
    """

    bio = BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws["A1"] = 1
    ws["A2"] = 2
    ws["B1"] = "=SUM(A1:A2)"
    wb.save(bio)

    if cached_formula_value is None:
        return bio.getvalue()

    return _patch_xlsx(
        bio.getvalue(), {"xl/worksheets/sheet1.xml": ("B1", cached_formula_value)}
    )


def _make_multi_sheet_xlsx_bytes() -> bytes:
    """Two sheets of differing shape, each with a cached formula result.

    Exercises the per-sheet title matching (``wb_formulas[sheet.title]``), the
    cross-view dimension union, and a string-valued cached result.
    """

    bio = BytesIO()
    wb = Workbook()
    numbers = wb.active
    numbers.title = "Numbers"
    numbers["A1"] = 1
    numbers["A2"] = 2
    numbers["B1"] = "=SUM(A1:A2)"

    words = wb.create_sheet("Words")
    words["A1"] = "foo"
    words["B1"] = '=A1&"bar"'
    wb.save(bio)

    return _patch_xlsx(
        bio.getvalue(),
        {
            "xl/worksheets/sheet1.xml": ("B1", 3),
            "xl/worksheets/sheet2.xml": ("B1", "foobar"),
        },
    )


@pytest.mark.offline
def test_extract_text_xlsx_uses_cached_formula_value():
    file_bytes = _make_xlsx_bytes(cached_formula_value=3)

    text = extract_text(file_bytes, "xlsx")

    assert "3" in text
    assert "=SUM(A1:A2)" not in text


@pytest.mark.offline
def test_extract_text_xlsx_falls_back_to_formula_text_when_cache_missing():
    file_bytes = _make_xlsx_bytes(cached_formula_value=None)

    text = extract_text(file_bytes, "xlsx")

    assert "=SUM(A1:A2)" in text


@pytest.mark.offline
def test_extract_text_xlsx_handles_multiple_sheets_and_string_results():
    file_bytes = _make_multi_sheet_xlsx_bytes()

    text = extract_text(file_bytes, "xlsx")

    # Both sheets are emitted, matched to their own formula view by title.
    assert "Sheet: Numbers" in text
    assert "Sheet: Words" in text
    # Numeric cached result preferred over the SUM formula text.
    assert "3" in text
    assert "=SUM(A1:A2)" not in text
    # String cached result preferred over the concatenation formula text.
    assert "foobar" in text
    assert '=A1&"bar"' not in text


@pytest.mark.offline
@pytest.mark.parametrize("group_depth", [0, 1, 2])
def test_extract_text_pptx_preserves_grouped_text_and_order(group_depth):
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])

    def add_text(shapes, text):
        shapes.add_textbox(0, 0, Inches(2), Inches(1)).text = text

    add_text(slide.shapes, "Before")
    shapes = slide.shapes
    for _ in range(group_depth):
        shapes = shapes.add_group_shape().shapes
    add_text(shapes, "Grouped paragraph\nSecond paragraph\vSoft break")
    shapes.add_connector(MSO_CONNECTOR.STRAIGHT, 0, 0, Inches(1), Inches(1))
    shapes.add_group_shape()  # Empty groups must not add spurious newlines.
    add_text(shapes, "")  # Keep the existing empty-textbox newline behavior.
    add_text(shapes, "Group sibling")
    add_text(slide.shapes, "After")
    next_slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    add_text(next_slide.shapes, "Next slide")
    file_bytes = BytesIO()
    presentation.save(file_bytes)

    assert extract_text(file_bytes.getvalue(), "pptx") == (
        "Before\nGrouped paragraph\nSecond paragraph\vSoft break\n\n"
        "Group sibling\nAfter\nNext slide\n"
    )


_W_URI = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W_DECL = f'xmlns:w="{_W_URI}"'


def _docx_bytes(build) -> bytes:
    """Save a python-docx document after ``build`` has added its content."""

    document = Document()
    build(document)
    out = BytesIO()
    document.save(out)
    return out.getvalue()


def _append_xml(document, xml: str) -> None:
    """Insert raw WordprocessingML before the body's closing ``w:sectPr``."""

    body = document.element.body
    body.insert(len(body) - 1, parse_xml(xml))


def _run(text: str) -> str:
    return f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r>'


@pytest.mark.offline
def test_extract_text_docx_reads_blocks_inside_content_controls_and_custom_xml():
    table = (
        "<w:tbl><w:tblPr/><w:tblGrid><w:gridCol/></w:tblGrid>"
        f"<w:tr><w:tc><w:p>{_run('Cell in a control')}</w:p></w:tc></w:tr></w:tbl>"
    )

    def build(document):
        document.add_paragraph("Before")
        _append_xml(
            document,
            f"<w:sdt {_W_DECL}><w:sdtPr/><w:sdtContent>"
            f"<w:p>{_run('In a control')}</w:p>{table}"
            "</w:sdtContent></w:sdt>",
        )
        _append_xml(
            document,
            f'<w:customXml {_W_DECL} w:element="clause">'
            f"<w:p>{_run('In custom XML')}</w:p></w:customXml>",
        )
        document.add_paragraph("After")

    assert extract_text(_docx_bytes(build), "docx") == (
        "Before\nIn a control\n\nCell in a control\n\nIn custom XML\nAfter"
    )


@pytest.mark.offline
def test_extract_text_docx_reads_runs_nested_in_a_paragraph():
    paragraphs = [
        _run("Signed by ")
        + f"<w:sdt><w:sdtPr/><w:sdtContent>{_run('Jane Doe')}</w:sdtContent></w:sdt>",
        _run("Dose: ")
        + f'<w:ins w:id="1" w:author="a">{_run("20 mg")}</w:ins>'
        + '<w:del w:id="2" w:author="a"><w:r><w:tab/><w:delText>10 mg</w:delText></w:r></w:del>',
        _run("Updated ")
        + f'<w:fldSimple w:instr="DATE">{_run("2024-05-01")}</w:fldSimple>',
        f'<w:smartTag w:uri="urn:x" w:element="place">{_run("Reutlingen")}</w:smartTag>',
    ]

    def build(document):
        for content in paragraphs:
            _append_xml(document, f"<w:p {_W_DECL}>{content}</w:p>")

    assert extract_text(_docx_bytes(build), "docx") == (
        "Signed by Jane Doe\nDose: 20 mg\nUpdated 2024-05-01\nReutlingen"
    )


@pytest.mark.offline
def test_extract_text_docx_leaves_out_runs_that_are_not_the_paragraph_text():
    mc = "http://schemas.openxmlformats.org/markup-compatibility/2006"
    content = (
        f'<w:moveFrom w:id="3" w:author="a">{_run("OLD PLACE")}</w:moveFrom>'
        f'<w:moveTo w:id="4" w:author="a">{_run("NEW PLACE")}</w:moveTo>'
        "<w:r><w:ruby><w:rubyPr/><w:rt><w:r><w:t>kan</w:t></w:r></w:rt>"
        "<w:rubyBase><w:r><w:t>漢</w:t></w:r></w:rubyBase></w:ruby></w:r>"
        f'<w:r><mc:AlternateContent xmlns:mc="{mc}"><mc:Choice Requires="wps">'
        f"<w:txbxContent><w:p>{_run('BOX')}</w:p></w:txbxContent></mc:Choice>"
        f"<mc:Fallback><w:pict><w:txbxContent><w:p>{_run('BOX COPY')}</w:p>"
        "</w:txbxContent></w:pict></mc:Fallback></mc:AlternateContent></w:r>"
        f'<mc:AlternateContent xmlns:mc="{mc}"><mc:Choice Requires="w14">'
        f"{_run('CHOICE')}</mc:Choice><mc:Fallback>{_run('FALLBACK')}</mc:Fallback>"
        "</mc:AlternateContent>"
    )

    def build(document):
        _append_xml(document, f"<w:p {_W_DECL}>{content}</w:p>")

    assert extract_text(_docx_bytes(build), "docx") == "NEW PLACE漢CHOICE"


@pytest.mark.offline
def test_extract_text_docx_reads_nested_runs_in_table_cells():
    def build(document):
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "Name"
        tc = table.cell(0, 1)._tc
        tc.p_lst[0].append(
            parse_xml(
                f"<w:sdt {_W_DECL}><w:sdtPr/>"
                f"<w:sdtContent>{_run('Jane Doe')}</w:sdtContent></w:sdt>"
            )
        )
        tc.append(
            parse_xml(
                f"<w:sdt {_W_DECL}><w:sdtPr/><w:sdtContent>"
                f"<w:p>{_run('second line')}</w:p></w:sdtContent></w:sdt>"
            )
        )

    assert extract_text(_docx_bytes(build), "docx") == "Name\tJane Doe<br>second line"


@pytest.mark.offline
def test_extract_text_docx_reads_a_plain_document_as_before():
    """Direct runs, a tab, a line break, a hyperlink and a table: unchanged."""

    def build(document):
        paragraph = document.add_paragraph("one\ttwo")
        paragraph.add_run().add_break()
        paragraph.add_run("three")
        _append_xml(
            document,
            f"<w:p {_W_DECL}>{_run('see ')}"
            f'<w:hyperlink w:anchor="x">{_run("the link")}</w:hyperlink></w:p>',
        )
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "A"
        table.cell(0, 1).text = "B"

    data = _docx_bytes(build)
    document = Document(BytesIO(data))
    assert [p.text for p in document.paragraphs] == ["one\ttwo\nthree", "see the link"]
    assert [c.text for c in document.tables[0].rows[0].cells] == ["A", "B"]
    assert extract_text(data, "docx") == "one\ttwo\nthree\nsee the link\n\nA\tB"
