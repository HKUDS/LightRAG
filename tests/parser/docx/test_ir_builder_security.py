"""Security-focused tests for native DOCX IR drawing asset handling."""

from __future__ import annotations

import pytest

from lightrag.parser.docx.ir_builder import NativeDocxIRBuilder


def _build_ir(content: str):
    return NativeDocxIRBuilder().normalize(
        [
            {
                "uuid": "p1",
                "uuid_end": "p1",
                "heading": "Section",
                "content": content,
                "parent_headings": [],
                "level": 1,
            }
        ],
        document_name="doc.docx",
        asset_dir_name="doc.blocks.assets",
    )


@pytest.mark.offline
def test_native_docx_ir_accepts_safe_local_drawing_asset() -> None:
    ir = _build_ir(
        '<drawing id="1" name="fig" format="png" path="doc.blocks.assets/fig.png" />'
    )

    drawing = ir.blocks[0].drawings[0]
    assert drawing.asset_ref == "fig.png"
    assert len(ir.assets) == 1
    assert ir.assets[0].ref == "fig.png"
    assert ir.assets[0].suggested_name == "fig.png"
    assert ir.assets[0].source is None


@pytest.mark.offline
@pytest.mark.parametrize(
    "path",
    [
        "doc.blocks.assets/../secret.png",
        "doc.blocks.assets//tmp/secret.png",
        r"doc.blocks.assets/..\secret.png",
    ],
)
def test_native_docx_ir_rejects_unsafe_local_drawing_asset(path: str) -> None:
    ir = _build_ir(f'<drawing id="1" name="fig" format="png" path="{path}" />')

    drawing = ir.blocks[0].drawings[0]
    assert drawing.asset_ref == ""
    assert ir.assets == []


@pytest.mark.offline
def test_native_docx_ir_passes_external_src_through() -> None:
    ir = _build_ir(
        '<drawing id="1" name="fig" format="gif" src="../images/legacy.gif" />'
    )

    drawing = ir.blocks[0].drawings[0]
    assert drawing.asset_ref == ""
    assert drawing.src == "../images/legacy.gif"
    assert drawing.fmt == "gif"
    assert ir.assets == []


@pytest.mark.offline
def test_native_docx_ir_infers_format_from_external_src() -> None:
    # No format attribute: the builder falls back to inferring it from the
    # external reference in src.
    ir = _build_ir('<drawing id="1" name="fig" src="../images/legacy.gif" />')

    drawing = ir.blocks[0].drawings[0]
    assert drawing.fmt == "gif"


@pytest.mark.offline
def test_native_docx_ir_extensionless_external_src_keeps_empty_format() -> None:
    # _infer_format_from_target returns None for an extensionless URL; the
    # builder must coerce that to "" (never the string "None").
    ir = _build_ir('<drawing id="1" name="fig" src="https://example.com/image" />')

    drawing = ir.blocks[0].drawings[0]
    assert drawing.fmt == ""
    assert drawing.src == "https://example.com/image"


@pytest.mark.offline
def test_native_docx_ir_drops_non_asset_path_without_salvage() -> None:
    # ``path`` is local-only under the new contract: a non-asset path (here a
    # remote URL that predates the extractor emitting ``src``) is dropped
    # rather than migrated into ``src``.
    ir = _build_ir(
        '<drawing id="1" name="fig" format="png" path="https://example.com/x.png" />'
    )

    drawing = ir.blocks[0].drawings[0]
    assert drawing.asset_ref == ""
    assert drawing.src == ""
    assert ir.assets == []
