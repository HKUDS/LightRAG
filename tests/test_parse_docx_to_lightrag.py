"""Unit tests for the new native docx → LightRAG content_list path.

These tests cover ``paragraphs_to_content_list`` and the asset-bytes filter
in ``parse_docx_to_lightrag_content_list``. They use synthetic ``Paragraph``
instances so the test does not depend on a real ``.docx`` fixture or
python-docx behavior.
"""

import pytest

from lightrag.extraction.docx_extractor import Paragraph
from lightrag.extraction.parse_document import (
    paragraphs_to_content_list,
)


@pytest.mark.offline
def test_heading_paragraph_to_section_header_item():
    paragraphs = [
        Paragraph(text="Chapter 1", outline_level=0),
        Paragraph(text="Body text under chapter.", outline_level=9),
    ]
    items = paragraphs_to_content_list(paragraphs, images={}, asset_dir_rel="x.assets")

    assert items == [
        {"type": "section_header", "text": "Chapter 1", "text_level": 1},
        {"type": "text", "text": "Body text under chapter."},
    ]


@pytest.mark.offline
def test_table_paragraph_to_table_item():
    rows = [["序号", "器件"], ["1", "A"], ["2", "B"]]
    paragraphs = [
        Paragraph(
            text="<table>...</table>",
            outline_level=9,
            is_table=True,
            table_json=rows,
        )
    ]
    items = paragraphs_to_content_list(paragraphs, images={}, asset_dir_rel="x.assets")

    assert len(items) == 1
    item = items[0]
    assert item["type"] == "table"
    assert item["rows"] == rows
    assert item["num_rows"] == 3
    assert item["num_cols"] == 2


@pytest.mark.offline
def test_inline_drawing_split_into_image_item_and_preserves_order():
    images = {"rId7": ("pic7.png", b"PNG-bytes")}
    paragraphs = [
        Paragraph(
            text='before <drawing id="9" name="Image 9" /> after',
            outline_level=9,
            has_drawing=True,
            drawing_rIds=["rId7"],
        )
    ]
    items = paragraphs_to_content_list(
        paragraphs, images=images, asset_dir_rel="demo.blocks.assets"
    )

    types = [it["type"] for it in items]
    assert types == ["text", "image", "text"]
    assert items[0]["text"] == "before"
    assert items[1]["id"] == "9"
    assert items[1]["img_path"] == "demo.blocks.assets/pic7.png"
    assert items[1]["image_caption"] == ["Image 9"]
    assert items[2]["text"] == "after"


@pytest.mark.offline
def test_inline_equation_split_into_equation_item():
    paragraphs = [
        Paragraph(
            text="see <equation>E = m c^2</equation> here",
            outline_level=9,
        )
    ]
    items = paragraphs_to_content_list(paragraphs, images={}, asset_dir_rel="x.assets")
    types = [it["type"] for it in items]
    assert types == ["text", "equation", "text"]
    assert items[1]["text"] == "E = m c^2"


@pytest.mark.offline
def test_drawing_without_matching_rid_does_not_emit_image():
    paragraphs = [
        Paragraph(
            text='only-text <drawing id="1" name="x" />',
            outline_level=9,
            drawing_rIds=[],  # no rId attached → no image item
        )
    ]
    items = paragraphs_to_content_list(paragraphs, images={}, asset_dir_rel="x.assets")
    types = [it["type"] for it in items]
    assert types == ["text"]
    assert items[0]["text"] == "only-text"


@pytest.mark.offline
def test_empty_or_whitespace_paragraphs_skipped():
    paragraphs = [
        Paragraph(text="", outline_level=9),
        Paragraph(text="   \n  ", outline_level=9),
        Paragraph(text="real", outline_level=9),
    ]
    items = paragraphs_to_content_list(paragraphs, images={}, asset_dir_rel="x.assets")
    assert items == [{"type": "text", "text": "real"}]


@pytest.mark.offline
def test_section_header_text_level_is_one_based():
    # outline_level 0 (top heading) → text_level 1
    # outline_level 2 (third level) → text_level 3
    paragraphs = [
        Paragraph(text="H1", outline_level=0),
        Paragraph(text="H3", outline_level=2),
    ]
    items = paragraphs_to_content_list(paragraphs, images={}, asset_dir_rel="x.assets")
    assert items[0]["text_level"] == 1
    assert items[1]["text_level"] == 3


@pytest.mark.offline
def test_parse_docx_to_lightrag_content_list_filters_unreferenced_assets(monkeypatch):
    """Only image bytes that are actually referenced by the content_list are
    returned, mirroring the behavior expected by the LightRAG Document writer
    (writing a stray image we never cite would dirty the asset directory)."""
    from lightrag.extraction import parse_document as pd

    paragraphs = [
        Paragraph(
            text='hi <drawing id="1" name="A" />',
            outline_level=9,
            drawing_rIds=["rIdA"],
        )
    ]
    images_map = {
        "rIdA": ("a.png", b"A-BYTES"),
        "rIdB": ("b.png", b"B-BYTES"),  # not referenced
    }

    monkeypatch.setattr(pd, "extract_docx_paragraphs", lambda _b: paragraphs)
    monkeypatch.setattr(pd, "extract_docx_images", lambda _b: images_map)

    content_list, asset_blobs = pd.parse_docx_to_lightrag_content_list(
        b"docx-bytes",
        source_file="report.docx",
        doc_id="doc-1",
    )

    assert any(it.get("type") == "image" for it in content_list)
    assert "a.png" in asset_blobs and asset_blobs["a.png"] == b"A-BYTES"
    assert "b.png" not in asset_blobs


@pytest.mark.offline
def test_parse_docx_to_lightrag_content_list_uses_source_stem_for_asset_dir(
    monkeypatch,
):
    """Asset paths must use ``<source-stem>.blocks.assets`` so they line up
    with ``LightRAG._write_lightrag_document_from_content_list``."""
    from lightrag.extraction import parse_document as pd

    paragraphs = [
        Paragraph(
            text='<drawing id="1" name="A" />',
            outline_level=9,
            drawing_rIds=["r1"],
        )
    ]
    monkeypatch.setattr(pd, "extract_docx_paragraphs", lambda _b: paragraphs)
    monkeypatch.setattr(pd, "extract_docx_images", lambda _b: {"r1": ("p.png", b"x")})

    content_list, _ = pd.parse_docx_to_lightrag_content_list(
        b"docx", source_file="MI012 技术说明书.docx", doc_id="doc-1"
    )
    image_item = next(it for it in content_list if it["type"] == "image")
    assert image_item["img_path"].startswith("MI012 技术说明书.blocks.assets/")
    assert image_item["img_path"].endswith("p.png")
