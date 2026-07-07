"""Tests for legacy text extraction helpers."""

from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

import pytest
from openpyxl import Workbook

from lightrag.parser.legacy.extractors import extract_text


def _make_xlsx_bytes(*, cached_formula_value: str | int | float | None) -> bytes:
    """Build a minimal workbook with one formula cell.

    ``openpyxl`` writes the formula expression, but not a cached calculated
    value. We patch the worksheet XML so the extractor can exercise both the
    ``data_only=True`` path and the formula fallback path.
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

    source = BytesIO(bio.getvalue())
    patched = BytesIO()
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with ZipFile(source, "r") as zin, ZipFile(patched, "w", ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                root = ET.fromstring(data)
                for cell in root.findall(".//main:c", ns):
                    if cell.attrib.get("r") != "B1":
                        continue
                    value_node = cell.find("main:v", ns)
                    if value_node is None:
                        value_node = ET.SubElement(
                            cell,
                            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v",
                        )
                    value_node.text = str(cached_formula_value)
                    break
                data = ET.tostring(root, encoding="utf-8", xml_declaration=False)
            zout.writestr(item, data)

    return patched.getvalue()


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
