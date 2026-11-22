"""Blank HTML header detection must use rendered character references."""

import pytest

from lightrag.parser._html_table import extract_thead_html
from lightrag.parser.markdown.extract import table_marker
from lightrag.parser.markdown.ir_builder import NativeMarkdownIRBuilder

pytestmark = pytest.mark.offline


# Tag removal happens before entity decoding; the ordinary header is
# tests/parser/markdown/test_ir_builder.py::test_html_table_becomes_html_irtable_with_thead_and_dims.
@pytest.mark.parametrize("text", ["&lt;ID&gt;"])
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
