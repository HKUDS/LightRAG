"""HTML slicing must retain offsets into the original Unicode text."""

import pytest

from lightrag.parser._html_table import (
    extract_thead_html,
    html_table_inner_body,
    unwrap_html_table,
)

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("prefix", ["", "<p>İstanbul</p>"])
@pytest.mark.parametrize("tag", ["table", "TABLE"])
def test_unwrap_preserves_unicode_and_complete_closing_tag(prefix, tag):
    table = f'<{tag} data-city="İstanbul"><tr><td>İzmir</td></tr></{tag}>'
    assert unwrap_html_table(f"<html><body>{prefix}{table}</body></html>") == table


@pytest.mark.parametrize("tag", ["thead", "THEAD"])
def test_header_slice_preserves_unicode_before_and_inside_header(tag):
    header = f"<{tag}><tr><th>İstanbul</th></tr></{tag}>"
    assert extract_thead_html(f'<table title="İzmir">{header}</table>') == header


def test_inner_body_does_not_include_part_of_closing_tag():
    body = "<tr><td>İstanbul</td></tr>"
    assert html_table_inner_body(f'<TABLE title="İzmir">{body}</TABLE>') == body
