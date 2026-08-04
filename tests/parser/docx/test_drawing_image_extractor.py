"""Unit tests for drawing placeholder emission in ``drawing_image_extractor``.

Focus: the ``path``/``src`` contract. ``path`` is local-only (a file the
extractor materialized under ``<base>.blocks.assets/``); linked/external
image references travel through ``src`` and never through ``path``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from lightrag.parser.docx.drawing_image_extractor import (
    IMAGE_REL_TYPE,
    DrawingExtractionContext,
    DrawingRelationship,
    extract_drawing_placeholder_from_element,
    extract_vml_image_placeholder_from_element,
    parse_drawing_attributes,
)

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_V = "urn:schemas-microsoft-com:vml"


def _drawing_element(blip_attr: str, rel_id: str) -> ET.Element:
    return ET.fromstring(
        f'<w:drawing xmlns:w="{_W}" xmlns:wp="{_WP}" xmlns:a="{_A}" xmlns:r="{_R}">'
        "<wp:inline>"
        '<wp:docPr id="7" name="fig"/>'
        f'<a:blip r:{blip_attr}="{rel_id}"/>'
        "</wp:inline>"
        "</w:drawing>"
    )


def _pict_element(rel_id: str) -> ET.Element:
    return ET.fromstring(
        f'<w:pict xmlns:w="{_W}" xmlns:v="{_V}" xmlns:r="{_R}">'
        '<v:shape id="s1" alt="vml fig">'
        f'<v:imagedata r:id="{rel_id}"/>'
        "</v:shape>"
        "</w:pict>"
    )


def _context_with_relationship(rel: DrawingRelationship) -> DrawingExtractionContext:
    ctx = DrawingExtractionContext(docx_path=Path("unused.docx"))
    ctx.relationships[rel.rel_id] = rel
    return ctx


def _external_rel(target: str, image_format: str) -> DrawingRelationship:
    return DrawingRelationship(
        rel_id="rId9",
        target=target,
        target_mode="External",
        rel_type=IMAGE_REL_TYPE,
        image_format=image_format,
    )


@pytest.mark.offline
@pytest.mark.parametrize(
    ("target", "image_format"),
    [
        ("https://example.com/diagrams/architecture.png", "png"),
        ("../images/legacy.gif", "gif"),
    ],
)
def test_linked_blip_target_lands_in_src_not_path(
    target: str, image_format: str
) -> None:
    ctx = _context_with_relationship(_external_rel(target, image_format))
    placeholder = extract_drawing_placeholder_from_element(
        _drawing_element("link", "rId9"), ctx
    )

    attrs = parse_drawing_attributes(placeholder)
    assert attrs["src"] == target
    assert attrs["format"] == image_format
    assert "path" not in attrs


@pytest.mark.offline
def test_vml_external_imagedata_target_lands_in_src_not_path() -> None:
    ctx = _context_with_relationship(
        _external_rel("https://example.com/shape.png", "png")
    )
    placeholder = extract_vml_image_placeholder_from_element(_pict_element("rId9"), ctx)

    attrs = parse_drawing_attributes(placeholder)
    assert attrs["src"] == "https://example.com/shape.png"
    assert attrs["format"] == "png"
    assert "path" not in attrs


@pytest.mark.offline
def test_embedded_blip_still_exports_to_local_path(tmp_path: Path) -> None:
    docx_path = tmp_path / "doc.docx"
    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("word/media/image1.png", b"PNGDATA")

    export_dir = tmp_path / "doc.blocks.assets"
    export_dir.mkdir()
    ctx = DrawingExtractionContext(
        docx_path=docx_path,
        export_dir_name="doc.blocks.assets",
        export_dir_path=export_dir,
    )
    ctx.relationships["rId1"] = DrawingRelationship(
        rel_id="rId1",
        target="media/image1.png",
        target_mode="",
        rel_type=IMAGE_REL_TYPE,
        part_name="/word/media/image1.png",
        content_type="image/png",
        image_format="png",
    )

    placeholder = extract_drawing_placeholder_from_element(
        _drawing_element("embed", "rId1"), ctx
    )

    attrs = parse_drawing_attributes(placeholder)
    assert attrs["path"] == "doc.blocks.assets/image1.png"
    assert attrs["format"] == "png"
    assert "src" not in attrs
    assert (export_dir / "image1.png").read_bytes() == b"PNGDATA"
