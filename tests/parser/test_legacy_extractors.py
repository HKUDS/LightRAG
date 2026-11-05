"""Tests for legacy text extraction helpers."""

from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

import pytest
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


def _make_ragged_rows_xlsx_bytes() -> bytes:
    """Rows of unequal width plus a formula view wider than its value view.

    read_only mode may emit ragged rows for sparse tails; the extractor must
    pad both views to the per-row width instead of dropping the trailing
    cells (or misaligning the value/formula pairing).
    """
    bio = BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "Ragged"
    ws["A1"] = "header-a"
    ws["C1"] = "header-c"          # B1 empty -> ragged row 1
    ws["A2"] = "x"
    ws["B2"] = "y"
    ws["C2"] = "z"
    ws["D2"] = "w"                 # wider than row 1
    ws["A3"] = "short"             # narrow row again
    wb.save(bio)
    return bio.getvalue()


@pytest.mark.offline
def test_extract_text_xlsx_pads_ragged_rows():
    file_bytes = _make_ragged_rows_xlsx_bytes()

    text = extract_text(file_bytes, "xlsx")

    lines = [l for l in text.splitlines() if l and not set(l) <= {"=", " "}]
    # Row 1: A and C survive with the empty B padded, in column order.
    assert any("header-a" in l and "header-c" in l for l in lines)
    # Row 2: all four cells land on one line, order preserved.
    assert any(l.strip().split("\t")[:4] == ["x", "y", "z", "w"] for l in lines)
    # Row 3 stays intact and is not misaligned by the wider row 2.
    assert any(l.strip().startswith("short") for l in lines)


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
