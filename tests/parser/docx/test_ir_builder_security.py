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
    assert drawing.path_override is None
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
    assert drawing.path_override is None
    assert ir.assets == []


@pytest.mark.offline
def test_native_docx_ir_preserves_non_asset_external_drawing_path() -> None:
    ir = _build_ir(
        '<drawing id="1" name="fig" format="gif" path="../images/legacy.gif" />'
    )

    drawing = ir.blocks[0].drawings[0]
    assert drawing.asset_ref == ""
    assert drawing.path_override == "../images/legacy.gif"
    assert ir.assets == []
