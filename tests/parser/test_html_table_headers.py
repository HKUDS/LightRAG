"""Blank HTML header detection must use rendered character references."""

import pytest

from lightrag.parser._html_table import extract_thead_html
from lightrag.parser.markdown.extract import table_marker
from lightrag.parser.markdown.ir_builder import NativeMarkdownIRBuilder

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "blank", ["", " \t\n", "&nbsp;", "&#160;", "&#xA0;", "&ensp;", "&emsp;"]
)
def test_entity_only_blank_headers_are_not_repeated(blank: str) -> None:
    html = f"<table><thead><tr><th colspan='2'>{blank}</th></tr></thead></table>"

    assert extract_thead_html(html) is None


@pytest.mark.parametrize(
    "text", ["Revenue", "&lt;ID&gt;", "&amp;", "&nbsp;Net profit&nbsp;"]
)
def test_visible_headers_keep_their_original_markup(text: str) -> None:
    header = f"<thead><tr><th colspan='2'>{text}</th></tr></thead>"

    assert extract_thead_html(f"<table>{header}</table>") == header


def test_markdown_ir_keeps_the_table_but_not_a_blank_header() -> None:
    html = (
        "<table><thead><tr><th>&nbsp;</th></tr></thead>"
        "<tbody><tr><td>42</td></tr></tbody></table>"
    )
    ir = NativeMarkdownIRBuilder().normalize(
        [{"content": table_marker("t1")}],
        document_name="table.md",
        asset_dir_name="table.blocks.assets",
        parse_metadata={"md_tables": {"t1": {"kind": "html", "html": html}}},
    )
    (table,) = ir.blocks[0].tables

    assert table.table_header is None
    assert table.html == html
    assert table.body_override and "<td>42</td>" in table.body_override
    assert (table.num_rows, table.num_cols) == (2, 1)
