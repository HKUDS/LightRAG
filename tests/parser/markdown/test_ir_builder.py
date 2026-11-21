"""Unit tests for ``NativeMarkdownIRBuilder`` (markers + side tables → IRDoc)."""

from __future__ import annotations

import pytest

from lightrag.parser.markdown.extract import (
    drawing_marker,
    equation_marker,
    table_marker,
)
from lightrag.parser.markdown.ir_builder import NativeMarkdownIRBuilder


pytestmark = pytest.mark.offline


def _normalize(blocks, meta, *, document_name="doc.md"):
    return NativeMarkdownIRBuilder().normalize(
        blocks,
        document_name=document_name,
        asset_dir_name="doc.blocks.assets",
        parse_metadata=meta,
    )


@pytest.mark.parametrize(
    "marker",
    [table_marker("author"), equation_marker("author"), drawing_marker("author")],
)
def test_undeclared_source_markers_remain_literal(marker):
    ir = _normalize([{"content": marker}], {})
    (block,) = ir.blocks
    assert block.content_template == marker
    assert not block.tables
    assert not block.equations
    assert not block.drawings


@pytest.mark.parametrize(
    "token, marker, meta, collection, key",
    [
        ("{{TBL:tb1}}", table_marker("t1"), {"md_tables": {"t1": {}}}, "tables", "tb1"),
        (
            "{{EQ:eq1}}",
            equation_marker("e1"),
            {"md_equations": {"e1": "x"}},
            "equations",
            "eq1",
        ),
        (
            "{{EQI:eq1}}",
            equation_marker("e1"),
            {"md_equations": {"e1": "x"}},
            "equations",
            "eq1",
        ),
        (
            "{{IMG:im1}}",
            drawing_marker("d1"),
            {"md_drawings": {"d1": {"kind": "external"}}},
            "drawings",
            "im1",
        ),
    ],
)
def test_generated_placeholder_keys_do_not_alias_source_tokens(
    token, marker, meta, collection, key
):
    ir = _normalize([{"content": marker}, {"content": token}], meta)
    (item,) = getattr(ir.blocks[0], collection)
    assert item.placeholder_key != key
    assert ir.blocks[1].content_template == token


@pytest.mark.parametrize(
    "body, rows, header",
    [
        pytest.param(
            [["a", "b"]], [["H1", "H2"], ["a", "b"]], [["H1", "H2"]], id="aligned"
        ),
        pytest.param(
            [["x"], ["y", "z", "EXTRA"]],
            [["H1", "H2", ""], ["x", "", ""], ["y", "z", "EXTRA"]],
            [["H1", "H2", ""]],
            id="ragged-rows-widen-to-widest",
        ),
    ],
)
def test_pipe_table_becomes_json_irtable_with_header_grid(body, rows, header):
    meta = {
        "md_tables": {"t1": {"kind": "pipe", "rows": body, "header": [["H1", "H2"]]}}
    }
    ir = _normalize([{"content": table_marker("t1")}], meta)
    (table,) = ir.blocks[0].tables
    assert "{{TBL:" in ir.blocks[0].content_template
    assert table.html is None
    assert table.rows == rows
    assert table.table_header == header
    assert (table.num_rows, table.num_cols) == (len(rows), len(rows[0]))


_HTML_TABLES = [
    pytest.param(
        "<table><thead><tr><th>K</th><th>V</th></tr></thead>"
        "<tbody><tr><td>a</td><td>1</td></tr></tbody></table>",
        "<thead><tr><th>K</th><th>V</th></tr></thead>",
        id="explicit-end-tags",
    ),
    pytest.param(
        "<table><thead><tr><th>K<th>V</thead><tbody><tr><td>a<td>1</tbody></table>",
        "<thead><tr><th>K<th>V</thead>",
        id="optional-end-tags",
    ),
]


@pytest.mark.parametrize("html, thead", _HTML_TABLES)
def test_html_table_becomes_html_irtable_with_thead_and_dims(html, thead):
    ir = _normalize(
        [{"content": table_marker("t1")}],
        {"md_tables": {"t1": {"kind": "html", "html": html}}},
    )
    (table,) = ir.blocks[0].tables
    assert table.rows is None
    assert table.html == html
    assert table.table_header == thead
    assert table.body_override and thead in table.body_override
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


def test_document_metadata_and_heading_position():
    # The first-H1 title is covered end to end by test_e2e_sidecar.py.
    ir = _normalize(
        [{"heading": "Sec", "level": 2, "parent_headings": [], "content": "## Sec"}],
        {},
        document_name="myfile.TextPack",
    )
    assert ir.doc_title == "myfile"
    assert ir.document_format == "textpack"
    (pos,) = ir.blocks[0].positions
    assert (pos.type, pos.anchor) == ("heading", "Sec")
