"""Native DOCX parsing of content controls (w:sdt) and custom XML markup.

Locks in the rules of ``lightrag/parser/docx/content_controls.py``: wrapped
content is read at every level, except a table of contents or a control still
showing its placeholder.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from lightrag.parser.docx.parse_document import extract_docx_blocks
from lightrag.parser.docx.smart_heading.features import (
    StyleAttributes,
    extract_paragraph_physical_features,
)
from lightrag.parser.docx.table_extractor import TableExtractor

pytestmark = pytest.mark.offline


def _wrap(
    element, wrapper: str = "sdt", *, gallery: str | None = None, placeholder=False
):
    """Move ``element`` into a content control or a custom XML element."""
    parent = element.getparent()
    index = parent.index(element)
    if wrapper == "sdt":
        outer = OxmlElement("w:sdt")
        sdt_pr = OxmlElement("w:sdtPr")
        if gallery is not None:
            doc_part = OxmlElement("w:docPartObj")
            gallery_node = OxmlElement("w:docPartGallery")
            gallery_node.set(qn("w:val"), gallery)
            doc_part.append(gallery_node)
            sdt_pr.append(doc_part)
        if placeholder:
            sdt_pr.append(OxmlElement("w:showingPlcHdr"))
        outer.append(sdt_pr)
        inner = OxmlElement("w:sdtContent")
        outer.append(inner)
    else:
        outer = inner = OxmlElement("w:customXml")
        outer.set(qn("w:element"), "item")
    parent.remove(element)
    inner.append(element)
    parent.insert(index, outer)


def _text(document, tmp_path) -> str:
    path = tmp_path / "doc.docx"
    document.save(path)
    return "\n".join(block["content"] for block in extract_docx_blocks(str(path)))


def _table_rows(document) -> list:
    source = BytesIO()
    document.save(source)
    source.seek(0)
    return TableExtractor.extract_with_metadata(Document(source).tables[0])["rows"]


def test_wrapped_body_blocks_are_read_in_order(tmp_path) -> None:
    document = Document()
    document.add_paragraph("Before")
    _wrap(document.add_paragraph("Clause in a control")._p)
    _wrap(document.add_paragraph("Tagged paragraph")._p, "customXml")
    table = document.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "Cell of a wrapped table"
    _wrap(table._tbl)
    _wrap(document.add_paragraph("Cover title")._p, gallery="Cover Pages")
    document.add_paragraph("After")

    text = _text(document, tmp_path)

    order = [
        "Before",
        "Clause in a control",
        "Tagged paragraph",
        "Cell of a wrapped table",
        "Cover title",
        "After",
    ]
    positions = [text.find(item) for item in order]
    assert -1 not in positions, text
    assert positions == sorted(positions)


def test_toc_and_placeholder_block_controls_are_left_out(tmp_path) -> None:
    document = Document()
    document.add_paragraph("Before")
    _wrap(
        document.add_paragraph("Chapter 1 Introduction\t3")._p,
        gallery="Table of Contents",
    )
    _wrap(
        document.add_paragraph("Click or tap here to enter text.")._p, placeholder=True
    )
    document.add_paragraph("After")

    text = _text(document, tmp_path)

    assert "Before" in text and "After" in text
    assert "Chapter 1" not in text
    assert "Click or tap" not in text


def _paragraph_with_inline_control(document, *, placeholder: bool, gallery=None):
    paragraph = document.add_paragraph("Name: ")
    run = paragraph.add_run("Ada" if not placeholder else "Click here")
    paragraph.add_run(" (end)")
    _wrap(run._r, placeholder=placeholder, gallery=gallery)
    return paragraph


@pytest.mark.parametrize("placeholder", [False, True])
def test_inline_control_text_is_read_unless_placeholder(tmp_path, placeholder) -> None:
    document = Document()
    _paragraph_with_inline_control(document, placeholder=placeholder)

    text = _text(document, tmp_path)

    assert "Name:" in text and "(end)" in text
    assert ("Ada" in text) is not placeholder
    assert "Click here" not in text


def test_inline_toc_control_is_still_read(tmp_path) -> None:
    # The smart-heading pass takes it as TOC evidence and decides what to keep.
    document = Document()
    _paragraph_with_inline_control(
        document, placeholder=False, gallery="Table of Contents"
    )

    assert "Ada" in _text(document, tmp_path)


@pytest.mark.parametrize("placeholder", [False, True])
def test_inline_placeholder_in_table_cell(placeholder) -> None:
    document = Document()
    table = document.add_table(rows=1, cols=1)
    paragraph = table.cell(0, 0).paragraphs[0]
    paragraph.add_run("Name: ")
    run = paragraph.add_run("Click here" if placeholder else "Ada")
    _wrap(run._r, placeholder=placeholder)

    assert _table_rows(document) == [["Name:" if placeholder else "Name: Ada"]]


@pytest.mark.parametrize(
    ("wrap_kwargs", "expected"),
    [
        ({}, "Wrapped field"),
        ({"wrapper": "customXml"}, "Wrapped field"),
        ({"placeholder": True}, ""),
    ],
)
def test_cell_paragraph_in_block_control(wrap_kwargs, expected) -> None:
    document = Document()
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Label"
    table.cell(0, 1).text = "Wrapped field"
    _wrap(table.cell(0, 1).paragraphs[0]._p, **wrap_kwargs)

    assert _table_rows(document) == [["Label", expected]]


def test_placeholder_run_is_not_a_smart_heading_feature() -> None:
    document = Document()
    paragraph = _paragraph_with_inline_control(document, placeholder=True)

    features = extract_paragraph_physical_features(paragraph._p, StyleAttributes())

    text = "".join(rf.text for rf in features.run_features)
    assert "Click here" not in text
    assert "Name:" in text


# --- bibliography controls: read, but marked for drop_references -----------


def _bibliography(document, entries, *, heading: str | None):
    """Append Word's bibliography building block: gallery + w:bibliography."""
    paragraphs = []
    if heading is not None:
        paragraphs.append(document.add_heading(heading, level=1)._p)
    entry_ps = [document.add_paragraph(entry)._p for entry in entries]
    for p in entry_ps:
        _wrap(p)
    # Merge the entries into ONE w:bibliography control, as Word writes it.
    first = entry_ps[0].getparent().getparent()
    content = first.find(qn("w:sdtContent"))
    first.find(qn("w:sdtPr")).append(OxmlElement("w:bibliography"))
    for p in entry_ps[1:]:
        sdt = p.getparent().getparent()
        content.append(p)
        sdt.getparent().remove(sdt)
    paragraphs.append(first)
    if heading is not None:
        # The gallery wrapper holds the heading and the bibliography control.
        _wrap(paragraphs[0], gallery="Bibliographies")
        gallery = paragraphs[0].getparent()
        gallery.append(first)


def _blocks(document, tmp_path) -> list:
    path = tmp_path / "doc.docx"
    document.save(path)
    return extract_docx_blocks(str(path))


def test_bibliography_with_heading_is_one_references_block(tmp_path) -> None:
    document = Document()
    document.add_heading("Conclusion", level=1)
    document.add_paragraph("We conclude.")
    _bibliography(document, ["[1] Foo.", "[2] Bar."], heading="Works Cited")
    document.add_heading("Appendix", level=1)
    document.add_paragraph("Extra.")

    blocks = _blocks(document, tmp_path)

    by_heading = {b["heading"]: b for b in blocks}
    assert by_heading["Works Cited"].get("session_type") == "references"
    assert "[1] Foo." in by_heading["Works Cited"]["content"]
    assert "[2] Bar." in by_heading["Works Cited"]["content"]
    assert "session_type" not in by_heading["Conclusion"]
    assert "session_type" not in by_heading["Appendix"]


def test_bibliography_without_heading_gets_its_own_block(tmp_path) -> None:
    document = Document()
    document.add_heading("Conclusion", level=1)
    document.add_paragraph("We conclude.")
    _bibliography(document, ["[1] Foo."], heading=None)
    document.add_paragraph("Closing remark.")

    blocks = _blocks(document, tmp_path)

    assert [(b["heading"], b.get("session_type")) for b in blocks] == [
        ("Conclusion", None),
        ("Conclusion", "references"),
        ("Conclusion", None),
    ]
    assert blocks[1]["content"] == "[1] Foo."
    assert "Closing remark." in blocks[2]["content"]


def test_references_session_type_reaches_the_ir() -> None:
    from lightrag.parser.docx.ir_builder import NativeDocxIRBuilder

    rows = [
        {
            "uuid": "p1",
            "uuid_end": "p1",
            "heading": "Body",
            "content": "Text.",
            "parent_headings": [],
            "level": 1,
        },
        {
            "uuid": "p2",
            "uuid_end": "p2",
            "heading": "Body",
            "content": "[1] Foo.",
            "parent_headings": [],
            "level": 1,
            "session_type": "references",
        },
    ]

    ir = NativeDocxIRBuilder().normalize(
        rows, document_name="doc.docx", asset_dir_name="doc.blocks.assets"
    )

    assert [b.session_type for b in ir.blocks] == ["body", "references"]


_M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _table_with_bibliography_cell(document, other: str | None):
    """A table whose last cell is a bibliography; ``other`` fills the first."""
    table = document.add_table(rows=1, cols=2 if other is not None else 1)
    bib_cell = table.cell(0, table._column_count - 1)
    bib_cell.text = "[1] Foo."
    paragraph = bib_cell.paragraphs[0]._p
    _wrap(paragraph)
    paragraph.getparent().getparent().find(qn("w:sdtPr")).append(
        OxmlElement("w:bibliography")
    )
    if other == "text":
        table.cell(0, 0).text = "Budget: 100"
    elif other == "equation":
        from lxml import etree

        table.cell(0, 0).paragraphs[0]._p.append(
            etree.fromstring(
                f'<m:oMath xmlns:m="{_M_NS}"><m:r><m:t>E=mc^2</m:t></m:r></m:oMath>'
            )
        )


def test_table_holding_only_a_bibliography_is_a_references_block(tmp_path) -> None:
    document = Document()
    document.add_heading("Conclusion", level=1)
    document.add_paragraph("We conclude.")
    _table_with_bibliography_cell(document, other=None)

    blocks = _blocks(document, tmp_path)

    refs = [b for b in blocks if b.get("session_type") == "references"]
    assert len(refs) == 1 and "[1] Foo." in refs[0]["content"]
    assert "We conclude." not in refs[0]["content"]


@pytest.mark.parametrize(
    ("other", "marker"), [("text", "Budget: 100"), ("equation", "E=mc")]
)
def test_table_mixing_bibliography_and_other_content_stays_body(
    tmp_path, other, marker
) -> None:
    # Keeping the bibliography text beats dropping the other cells with it.
    document = Document()
    document.add_heading("Conclusion", level=1)
    _table_with_bibliography_cell(document, other=other)

    blocks = _blocks(document, tmp_path)

    assert all(b.get("session_type") is None for b in blocks)
    content = "\n".join(b["content"] for b in blocks)
    assert marker in content and "[1] Foo." in content


def test_placeholder_does_not_mask_leading_padding() -> None:
    from lightrag.parser.docx.smart_heading.features import leading_pad_em

    document = Document()
    padded = document.add_paragraph(" " * 28 + "签名")
    masked = document.add_paragraph()
    run = masked.add_run("Click here")
    masked.add_run(" " * 28 + "签名")
    _wrap(run._r, placeholder=True)

    assert leading_pad_em(masked._p) == leading_pad_em(padded._p) > 0
