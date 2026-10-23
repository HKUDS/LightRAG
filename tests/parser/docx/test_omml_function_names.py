"""Preserve function names spread across multiple OMML text runs."""

from __future__ import annotations

from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import pytest
from docx import Document
from docx.oxml import parse_xml

from lightrag.parser.docx.omml import convert_omml_to_latex
from lightrag.parser.docx.parse_document import extract_docx_blocks

pytestmark = pytest.mark.offline

MATH_NS = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'


def _function_xml(parts: tuple[str, ...], *, scripted: bool = False) -> str:
    runs = "".join(f"<m:r><m:t>{escape(part)}</m:t></m:r>" for part in parts)
    if scripted:
        runs = (
            f"<m:sSubSup><m:e>{runs}</m:e>"
            "<m:sub><m:r><m:t>1</m:t></m:r></m:sub>"
            "<m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSubSup>"
        )
    return (
        f"<m:oMath {MATH_NS}><m:func><m:fName>{runs}</m:fName>"
        "<m:e><m:r><m:t>x</m:t></m:r></m:e></m:func></m:oMath>"
    )


@pytest.mark.parametrize(
    ("parts", "scripted", "expected"),
    [
        (("c", "os"), False, r"\cos_{}^{}{x}"),
        (("l", "im"), False, r"\lim\limits_{}^{}{x}"),
        (("foo", "bar"), False, r"{foobar}^{}_{}{x}"),
        (("cos",), False, r"\cos_{}^{}{x}"),
        (("c", "os"), True, r"\cos_{1}^{2}{x}"),
    ],
)
def test_function_name_keeps_all_runs(parts, scripted, expected):
    assert (
        convert_omml_to_latex(ET.fromstring(_function_xml(parts, scripted=scripted)))
        == expected
    )


@pytest.mark.parametrize("parts", [("cos",), ("c", "os")])
def test_split_function_name_survives_docx_extraction(tmp_path, parts):
    doc = Document()
    paragraph = doc.add_paragraph("Before ")
    paragraph._p.append(parse_xml(_function_xml(parts)))
    paragraph.add_run(" after")
    path = tmp_path / "function.docx"
    doc.save(path)

    blocks = extract_docx_blocks(str(path))

    assert len(blocks) == 1
    assert r"Before <equation>\cos_{}^{}{x}</equation> after" in blocks[0]["content"]
