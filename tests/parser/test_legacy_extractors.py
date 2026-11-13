"""Tests for legacy text extraction helpers."""

from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

import pytest
from docx import Document
from docx.oxml.shared import OxmlElement
from docx.oxml.ns import qn
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


def _retype_b1_formula(
    file_bytes: bytes, attrs: dict[str, str], formula_text: str | None
) -> bytes:
    """Rewrite the ``<f>`` node of cell B1 in ``sheet1.xml``.

    ``openpyxl`` cannot write array or data-table formulas without cached
    values the way other producers do, so the tests build an ordinary formula
    workbook and retype its formula node here. Any cached ``<v>`` is kept.
    """

    patched = BytesIO()
    with (
        ZipFile(BytesIO(file_bytes)) as source,
        ZipFile(patched, "w", ZIP_DEFLATED) as target,
    ):
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                root = ET.fromstring(data)
                cell = root.find(".//main:c[@r='B1']", _NS)
                formula_node = cell.find("main:f", _NS)
                for key, value in attrs.items():
                    formula_node.set(key, value)
                formula_node.text = formula_text
                data = ET.tostring(root, encoding="utf-8")
            target.writestr(item, data)
    return patched.getvalue()


@pytest.mark.offline
@pytest.mark.parametrize("formula", ["=A1:A2*2", "=_xlfn._xlws.FILTER(A1:A2,A1:A2>1)"])
@pytest.mark.parametrize("cached_value", [None, 0, 3, "calculated"])
def test_extract_text_xlsx_preserves_array_formula_content(formula, cached_value):
    """Array formula objects must yield cached values or their formula text."""
    file_bytes = _retype_b1_formula(
        _make_xlsx_bytes(cached_formula_value=cached_value),
        {"t": "array", "ref": "B1:B2"},
        formula.removeprefix("="),
    )

    text = extract_text(file_bytes, "xlsx")

    expected = formula if cached_value is None else str(cached_value)
    assert text.splitlines()[1] == f"1\t{expected}"
    assert "ArrayFormula object" not in text


@pytest.mark.offline
@pytest.mark.parametrize("cached_value", [None, 0, 3, "calculated"])
def test_extract_text_xlsx_never_indexes_data_table_formula_object(cached_value):
    """A data-table formula carries no text: cached value or an empty cell.

    openpyxl >= 3.1 reads it as a ``DataTableFormula`` object and older
    versions as a bare ``"="``; neither may reach the extracted text.
    """
    file_bytes = _retype_b1_formula(
        _make_xlsx_bytes(cached_formula_value=cached_value),
        {"t": "dataTable", "ref": "B1:B1", "dt2D": "0", "dtr": "0", "r1": "A1"},
        None,
    )

    text = extract_text(file_bytes, "xlsx")

    expected = "" if cached_value is None else str(cached_value)
    assert text.splitlines()[1] == f"1\t{expected}"
    assert "DataTableFormula" not in text
    assert " object at 0x" not in text


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


def _docx_bytes(document) -> bytes:
    file_bytes = BytesIO()
    document.save(file_bytes)
    return file_bytes.getvalue()


@pytest.mark.offline
def test_extract_text_docx_keeps_horizontally_merged_cell_text_once():
    document = Document()
    table = document.add_table(rows=2, cols=3)
    table.cell(0, 0).merge(table.cell(0, 2)).text = "Quarterly report"
    for column, text in enumerate(("Q1", "Q2", "Q3")):
        table.cell(1, column).text = text

    # The merged cell still takes three columns, so the rows stay aligned.
    assert extract_text(_docx_bytes(document), "docx") == (
        "Quarterly report\t\t\nQ1\tQ2\tQ3"
    )


@pytest.mark.offline
def test_extract_text_docx_repeats_vertically_merged_cell_text_per_row():
    document = Document()
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).merge(table.cell(1, 0)).text = "Region"
    table.cell(0, 1).text = "North"
    table.cell(1, 1).text = "South"

    assert extract_text(_docx_bytes(document), "docx") == (
        "Region\tNorth\nRegion\tSouth"
    )


@pytest.mark.offline
def test_extract_text_docx_block_merged_cell_text_is_once_per_row():
    document = Document()
    table = document.add_table(rows=2, cols=3)
    table.cell(0, 0).merge(table.cell(1, 1)).text = "Block"
    table.cell(0, 2).text = "A"
    table.cell(1, 2).text = "B"

    assert extract_text(_docx_bytes(document), "docx") == "Block\t\tA\nBlock\t\tB"


def _filled_table(document, rows: int, cols: int):
    table = document.add_table(rows=rows, cols=cols)
    for row in range(rows):
        for column in range(cols):
            table.cell(row, column).text = f"r{row}c{column}"
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


@pytest.mark.offline
def test_extract_text_docx_grid_before_row_keeps_cells_in_their_columns():
    document = Document()
    table = _filled_table(document, 3, 3)
    _skip_grid_columns(table.rows[1], before=1)

    assert extract_text(_docx_bytes(document), "docx") == (
        "r0c0\tr0c1\tr0c2\n\tr1c1\tr1c2\nr2c0\tr2c1\tr2c2"
    )


@pytest.mark.offline
def test_extract_text_docx_grid_after_row_keeps_trailing_columns():
    document = Document()
    table = _filled_table(document, 2, 3)
    _skip_grid_columns(table.rows[0], after=2)

    assert extract_text(_docx_bytes(document), "docx") == "r0c0\t\t\nr1c0\tr1c1\tr1c2"


@pytest.mark.offline
def test_extract_text_docx_vertical_merge_below_grid_before_row():
    document = Document()
    table = _filled_table(document, 2, 3)
    table.cell(0, 2).merge(table.cell(1, 2)).text = "Total"
    _skip_grid_columns(table.rows[0], before=1)

    # The continuation in row 2 sits under grid column 2, which row 1
    # reaches only after its skipped first column.
    assert extract_text(_docx_bytes(document), "docx") == (
        "\tr0c1\tTotal\nr1c0\tr1c1\tTotal"
    )


@pytest.mark.offline
def test_extract_text_docx_vertical_merge_continuation_without_cell_above():
    document = Document()
    table = _filled_table(document, 1, 2)
    v_merge = OxmlElement("w:vMerge")
    table.cell(0, 0)._tc.get_or_add_tcPr().append(v_merge)

    # A malformed continuation in the first row has nothing to repeat.
    assert extract_text(_docx_bytes(document), "docx") == "\tr0c1"


def _make_encrypted_pdf_bytes(text: str, *, password: str, algorithm: str) -> bytes:
    """Build a one-page PDF showing ``text`` and encrypt it with pypdf."""
    from pypdf import PdfReader, PdfWriter

    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
    ]
    out = BytesIO()
    out.write(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n%s\nendobj\n" % (number, body))
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, xref)
    )

    writer = PdfWriter(clone_from=PdfReader(BytesIO(out.getvalue())))
    writer.encrypt(user_password=password, algorithm=algorithm)
    encrypted = BytesIO()
    writer.write(encrypted)
    return encrypted.getvalue()


@pytest.mark.offline
@pytest.mark.parametrize("algorithm", ["AES-128", "AES-256"])
def test_extract_text_pdf_decrypts_aes_encrypted_pdf(algorithm):
    # pypdf decrypts AES only through an installed crypto backend
    # (``cryptography``, else ``pycryptodome``); its pure-Python fallback raises
    # DependencyError. LightRAG ships ``cryptography`` for this, so this pins
    # that dropping it breaks encrypted-PDF uploads loudly here, not in prod.
    file_bytes = _make_encrypted_pdf_bytes(
        "Encrypted hello", password="secret", algorithm=algorithm
    )

    assert "Encrypted hello" in extract_text(file_bytes, "pdf", pdf_password="secret")


@pytest.mark.offline
def test_extract_text_pdf_rejects_wrong_password():
    file_bytes = _make_encrypted_pdf_bytes(
        "Encrypted hello", password="secret", algorithm="AES-256"
    )

    with pytest.raises(Exception, match="Incorrect PDF password"):
        extract_text(file_bytes, "pdf", pdf_password="wrong")

    with pytest.raises(Exception, match="no password provided"):
        extract_text(file_bytes, "pdf")
