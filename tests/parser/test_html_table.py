"""HTML table dimensions must not depend on optional cell/row end tags."""

from __future__ import annotations

import pytest

from lightrag.parser._html_table import extract_html_table_info

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        (
            "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>",
            (2, 2),
        ),
        (
            "<table><tr><td>a<td>b</tr><tr><td>c<td>d</tr></table>",
            (2, 2),
        ),
        (
            "<table><tr><td>a</td><td>b</td><tr><td>c</td><td>d</td></table>",
            (2, 2),
        ),
        ("<table><tr><td>a<td>b<tr><td>c<td>d</table>", (2, 2)),
        (
            "<table><thead><tr><th>A<th>B</thead><tbody><tr><td>a<td>b</tbody></table>",
            (2, 2),
        ),
        (
            '<table><tr><th colspan="3">Heading<tr><td>a<td>b<td>c</table>',
            (2, 3),
        ),
        ("<tr><td>a<td>b<tr><td>c<td>d", (2, 2)),
        (
            "<table><tr><td>outer<table><tr><td>nested-a<td>nested-b"
            "</table><td>peer</table>",
            (1, 2),
        ),
        (
            "<table><tr><td>outer<table><tr><td>nested-a</td>"
            "<td>nested-b</td></tr></table></td><td>peer</td></tr></table>",
            (1, 2),
        ),
        ("<table><tr></tr><tr><td>a</td></tr></table>", (1, 1)),
    ],
)
def test_table_dimensions_with_optional_end_tags(html, expected):
    info = extract_html_table_info(html)
    assert (info.num_rows, info.num_cols) == expected
