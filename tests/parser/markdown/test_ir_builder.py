"""Unit tests for ``NativeMarkdownIRBuilder`` (markers + side tables → IRDoc)."""

from __future__ import annotations

from lightrag.parser.markdown.extract import (
    drawing_marker,
    equation_marker,
    table_marker,
)
from lightrag.parser.markdown.ir_builder import NativeMarkdownIRBuilder


def _normalize(blocks, meta, *, document_name="doc.md"):
    return NativeMarkdownIRBuilder().normalize(
        blocks,
        document_name=document_name,
        asset_dir_name="doc.blocks.assets",
        parse_metadata=meta,
    )


def test_pipe_table_becomes_json_irtable_with_header_grid():
    blocks = [
        {
            "heading": "H",
            "level": 1,
            "parent_headings": [],
            "content": table_marker("t1"),
        }
    ]
    meta = {
        "md_tables": {
            "t1": {"kind": "pipe", "rows": [["a", "b"]], "header": [["H1", "H2"]]}
        }
    }
    ir = _normalize(blocks, meta)
    (table,) = ir.blocks[0].tables
    assert "{{TBL:" in ir.blocks[0].content_template
    assert table.rows == [["a", "b"]]
    assert table.html is None
    assert table.table_header == [["H1", "H2"]]
    assert (table.num_rows, table.num_cols) == (1, 2)


def test_html_table_becomes_html_irtable_with_thead_and_dims():
    html = "<table><thead><tr><th>K</th><th>V</th></tr></thead><tbody><tr><td>a</td><td>1</td></tr></tbody></table>"
    blocks = [
        {
            "heading": "H",
            "level": 1,
            "parent_headings": [],
            "content": table_marker("t1"),
        }
    ]
    ir = _normalize(blocks, {"md_tables": {"t1": {"kind": "html", "html": html}}})
    (table,) = ir.blocks[0].tables
    assert table.rows is None
    assert table.html and table.html.startswith("<table>")
    assert isinstance(table.table_header, str) and "<thead>" in table.table_header
    assert table.body_override and "<thead>" in table.body_override
    assert (table.num_rows, table.num_cols) == (2, 2)


def test_equation_is_block_level():
    blocks = [
        {
            "heading": "H",
            "level": 1,
            "parent_headings": [],
            "content": equation_marker("e1"),
        }
    ]
    ir = _normalize(blocks, {"md_equations": {"e1": "E = mc^2"}})
    (eq,) = ir.blocks[0].equations
    assert eq.is_block is True
    assert eq.latex == "E = mc^2"
    assert "{{EQ:" in ir.blocks[0].content_template


def test_local_drawing_dedups_assetspec_across_occurrences():
    content = f"{drawing_marker('d1')} {drawing_marker('d2')}"
    blocks = [{"heading": "H", "level": 1, "parent_headings": [], "content": content}]
    meta = {
        "md_drawings": {
            "d1": {"kind": "local", "asset_ref": "sha:1", "fmt": "png", "src": "a.png"},
            "d2": {"kind": "local", "asset_ref": "sha:1", "fmt": "png", "src": "a.png"},
        },
        "md_assets": {"sha:1": {"suggested_name": "a.png", "data": b"BYTES"}},
    }
    ir = _normalize(blocks, meta)
    assert len(ir.blocks[0].drawings) == 2
    assert all(d.asset_ref == "sha:1" for d in ir.blocks[0].drawings)
    # One shared on-disk asset, carried as bytes for the writer to materialize.
    assert len(ir.assets) == 1
    assert ir.assets[0].ref == "sha:1"
    assert ir.assets[0].source == b"BYTES"


def test_external_drawing_uses_path_override_and_no_asset():
    blocks = [
        {
            "heading": "H",
            "level": 1,
            "parent_headings": [],
            "content": drawing_marker("d1"),
        }
    ]
    meta = {
        "md_drawings": {
            "d1": {
                "kind": "external",
                "url": "http://x/y.png",
                "fmt": "png",
                "src": "http://x/y.png",
            }
        }
    }
    ir = _normalize(blocks, meta)
    (drawing,) = ir.blocks[0].drawings
    assert drawing.path_override == "http://x/y.png"
    assert drawing.asset_ref == ""
    assert ir.assets == []


def test_doc_title_from_first_h1_else_filename():
    ir = _normalize(
        [{"heading": "First", "level": 1, "parent_headings": [], "content": "# First"}],
        {},
    )
    assert ir.doc_title == "First"
    ir2 = _normalize(
        [{"heading": "Sub", "level": 2, "parent_headings": [], "content": "## Sub"}],
        {},
        document_name="myfile.md",
    )
    assert ir2.doc_title == "myfile"


def test_positions_use_heading_anchor():
    ir = _normalize(
        [{"heading": "Sec", "level": 2, "parent_headings": [], "content": "## Sec"}],
        {},
    )
    (pos,) = ir.blocks[0].positions
    assert pos.type == "heading"
    assert pos.anchor == "Sec"


def test_document_format_from_suffix():
    ir = _normalize(
        [{"heading": "X", "level": 1, "parent_headings": [], "content": "# X"}],
        {},
        document_name="a.textpack",
    )
    assert ir.document_format == "textpack"
