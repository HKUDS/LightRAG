"""OOXML on/off properties read the same way in every native DOCX reader.

``ST_OnOff`` switches a property off with ``0``, ``false`` or ``off``; a bare
element and every other spelling leave it on. Each reader below used to carry
its own copy of that list, and two of the copies had drifted (``none`` in place
of ``off``; ``off`` missing).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from lightrag.parser.docx.numbering_resolver import NumberingResolver
from lightrag.parser.docx.ooxml import W_NS, is_on
from lightrag.parser.docx.smart_heading.features import (
    extract_paragraph_physical_features,
    parse_styles_attributes,
)

pytestmark = pytest.mark.offline

ON_OFF_VALUES = [
    (None, True),
    ("1", True),
    ("true", True),
    ("on", True),
    ("0", False),
    ("false", False),
    ("off", False),
]


def _flag(tag: str, value: str | None):
    elem = OxmlElement(tag)
    if value is not None:
        elem.set(qn("w:val"), value)
    return elem


@pytest.mark.parametrize(("value", "expected"), ON_OFF_VALUES)
def test_is_on(value, expected) -> None:
    assert is_on(_flag("w:b", value)) is expected


@pytest.mark.parametrize(("value", "expected"), ON_OFF_VALUES)
def test_smart_heading_reads_bold_and_page_break_before(
    tmp_path: Path, value, expected
) -> None:
    doc = Document()
    para = doc.add_paragraph()
    para._p.get_or_add_pPr().append(_flag("w:pageBreakBefore", value))
    run = para.add_run("Quarterly results")
    run._r.get_or_add_rPr().append(_flag("w:b", value))
    path = tmp_path / "features.docx"
    doc.save(path)

    feats = extract_paragraph_physical_features(
        Document(str(path)).paragraphs[0]._p, parse_styles_attributes(str(path))
    )

    assert feats.all_bold is expected
    assert feats.page_break_before is expected


@pytest.mark.parametrize(("value", "expected"), ON_OFF_VALUES)
def test_empty_paragraph_record_reads_page_break_before(
    tmp_path: Path, value, expected
) -> None:
    from lightrag.parser.docx.parse_document import (
        _read_document_records,
        parse_styles_outline_levels,
    )

    doc = Document()
    doc.add_paragraph("Body text.")
    doc.add_paragraph()._p.get_or_add_pPr().append(_flag("w:pageBreakBefore", value))
    path = tmp_path / "empty.docx"
    doc.save(path)

    records = _read_document_records(
        Document(str(path)),
        NumberingResolver(str(path)),
        parse_styles_outline_levels(str(path)),
        None,
        {},
        style_attributes=parse_styles_attributes(str(path)),
    )

    (empty,) = [r for r in records if r.kind == "empty_para"]
    assert empty.page_break_before is expected


@pytest.mark.parametrize(("value", "expected"), ON_OFF_VALUES)
def test_numbering_reads_is_lgl(tmp_path: Path, value, expected) -> None:
    val = "" if value is None else f' w:val="{value}"'
    numbering = (
        f'<w:numbering xmlns:w="{W_NS}">'
        '<w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0">'
        '<w:start w:val="1"/><w:numFmt w:val="decimal"/>'
        f'<w:lvlText w:val="%1."/><w:isLgl{val}/>'
        "</w:lvl></w:abstractNum>"
        '<w:num w:numId="1"><w:abstractNumId w:val="1"/></w:num>'
        "</w:numbering>"
    )
    path = tmp_path / "numbering.docx"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/numbering.xml", numbering)

    resolver = NumberingResolver(str(path))

    assert resolver.abstract_nums["1"][0]["isLgl"] is expected


def test_absent_is_lgl_stays_off(tmp_path: Path) -> None:
    numbering = (
        f'<w:numbering xmlns:w="{W_NS}">'
        '<w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0">'
        '<w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/>'
        "</w:lvl></w:abstractNum></w:numbering>"
    )
    path = tmp_path / "numbering.docx"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/numbering.xml", numbering)

    assert NumberingResolver(str(path)).abstract_nums["1"][0]["isLgl"] is False
