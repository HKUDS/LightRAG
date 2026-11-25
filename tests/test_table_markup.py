"""The JSON body of a ``<table>`` tag must never end the tag early.

Every reader finds a table with a non-greedy ``<table …>(.*?)</table>``, so a
cell holding the text ``</table>`` cut the body short wherever the rows were
written with a bare ``json.dumps``. These tests pin the shared encoder.
"""

from __future__ import annotations

import json
import random

import pytest

from lightrag.table_markup import (
    TABLE_TAG_RE,
    detect_table_format,
    dumps_json_table_body,
    escape_json_table_body,
    extract_table_id,
    parse_table_tag,
    split_html_rows,
)

pytestmark = pytest.mark.offline


_TRICKY_CELLS = [
    "</table>",
    "</TABLE>",
    "</tAbLe >",
    "a</table>b</table>c",
    "\\</table>",  # a backslash right before the tag
    "<\\/table>",  # text that already looks escaped
    "\\\\</",
    '"</table>"',
    "</",
    "<//",
    "<<//table>>",
    "</equation></cite></td>",
    "表格</table>结束",
    "no markup at all",
    "",
]


@pytest.mark.parametrize("cell", _TRICKY_CELLS)
def test_body_never_holds_a_closing_tag_and_round_trips(cell: str) -> None:
    rows = [["Tag", "Meaning"], [cell, "x"], ["tail", cell]]

    body = dumps_json_table_body(rows)

    assert "</table>" not in body.lower()
    assert json.loads(body) == rows


@pytest.mark.parametrize("cell", _TRICKY_CELLS)
def test_tag_built_from_the_body_parses_back_whole(cell: str) -> None:
    rows = [["Tag", "Meaning"], [cell, "x"], ["tail", cell]]
    tag = f'<table id="tb-1" format="json">{dumps_json_table_body(rows)}</table>'

    match = TABLE_TAG_RE.match(tag)
    assert match is not None
    assert match.end() == len(tag)  # the first "</table>" is the real one
    assert parse_table_tag(tag) == ('id="tb-1" format="json"', rows)


@pytest.mark.parametrize(
    "cell",
    [
        "25 m<sup>2</sup>",
        "CO<sub>2</sub>",
        "<equation>E = mc^2</equation>",
        '<cite type="table" refid="tb-9">Table 9</cite>',
        "<table>",  # an opening tag alone ends nothing
        "</table >",  # no reader accepts this as a closing tag
        "</tables>",
        "a/b </ c",
        "",
    ],
)
def test_only_the_closing_table_tag_is_rewritten(cell: str) -> None:
    """Everything but ``</table>`` keeps the encoding it always had.

    Cells routinely hold balanced inner markup; other scanners pair those
    tags up across the whole chunk, so their closers must stay as written.
    """
    rows = [["a", cell], [{"k": cell}, 3]]

    assert dumps_json_table_body(rows) == json.dumps(rows, ensure_ascii=False)


def test_inner_markup_next_to_a_closing_table_tag_is_left_alone() -> None:
    body = dumps_json_table_body([["m<sup>2</sup></table><equation>x</equation>"]])

    assert body == '[["m<sup>2</sup><\\/table><equation>x</equation>"]]'


def test_escape_is_idempotent() -> None:
    once = escape_json_table_body(json.dumps([["</table>", "\\</"]]))

    assert escape_json_table_body(once) == once


def test_random_cells_round_trip() -> None:
    """Differential check against ``json.loads`` over the hazard alphabet."""
    rng = random.Random(4450)
    alphabet = [
        "<",
        "/",
        "\\",
        '"',
        ">",
        "table>",
        "TABLE>",
        "</table>",
        "t",
        " ",
        "表",
    ]
    for _ in range(2000):
        rows = [
            ["".join(rng.choices(alphabet, k=rng.randint(0, 12))) for _ in range(3)]
            for _ in range(rng.randint(1, 3))
        ]
        body = dumps_json_table_body(rows)
        assert "</table>" not in body.lower()
        assert json.loads(body) == rows
        assert escape_json_table_body(body) == body


def test_cell_markup_does_not_capture_a_later_real_tag() -> None:
    """Inner markup in a cell stays balanced, so the extraction cleaner still
    reaches the real tags that follow the table.

    Escaping every ``</`` in the body left a cell's ``<equation>`` opener
    without its closer; the cleaner then paired it with the next real
    ``</equation>`` and that equation kept its internal ``id``.
    """
    from lightrag.chunk_schema import strip_internal_multimodal_markup_for_extraction

    rows = [
        ["E", "<equation>x</equation>"],
        ["area", "25 m<sup>2</sup>"],
        ["</table>", "closes a table"],
    ]
    chunk = (
        f'<table id="tb-doc-0001" format="json">{dumps_json_table_body(rows)}</table>\n'
        '<equation id="eq-doc-0001" format="latex">y</equation>\n'
        'see <cite type="table" refid="tb-doc-0002">Table 2</cite>'
    )

    table_line, equation_line, cite_line = (
        strip_internal_multimodal_markup_for_extraction(chunk).split("\n")
    )

    assert equation_line == '<equation format="latex">y</equation>'
    assert cite_line == "see Table 2"
    assert parse_table_tag(table_line) == ('format="json"', rows)


def test_detect_table_format_explicit_attr() -> None:
    assert detect_table_format('id="t1" format="json"', "[]") == "json"
    assert detect_table_format("format='html'", "<tr></tr>") == "html"
    # Unknown formats fall through (force the caller to use char fallback).
    assert detect_table_format('format="markdown"', "...") is None


def test_detect_table_format_sniff_when_attrs_silent() -> None:
    assert detect_table_format("", '[{"a":1}]') == "json"
    assert detect_table_format("", "<tr><td>x</td></tr>") == "html"
    # Body that doesn't look like JSON or HTML → unknown.
    assert detect_table_format("", "plain text rows") is None


def test_split_html_rows_extracts_tr_elements() -> None:
    body = (
        "<thead><tr><th>h</th></tr></thead>"
        "<tbody><tr><td>a</td></tr><tr><td>b</td></tr></tbody>"
    )
    rows = split_html_rows(body)
    assert rows is not None
    assert len(rows) == 3
    # Each row carries its parent wrapper so the chunk serialiser can
    # rebuild <thead>/<tbody> instead of dropping them silently.
    assert [w for w, _ in rows] == ["thead", "tbody", "tbody"]
    assert all(tr.startswith("<tr") and tr.endswith("</tr>") for _, tr in rows)


def test_split_html_rows_no_tr_returns_none() -> None:
    assert split_html_rows("just text, no rows") is None
    assert split_html_rows("") is None


def test_extract_table_id_variants() -> None:
    assert extract_table_id('id="tb-1" format="json"') == "tb-1"
    assert extract_table_id("format='html' id='tb-h'") == "tb-h"
    assert extract_table_id('format="json"') is None
    assert extract_table_id("") is None
    # ``id`` that is only the tail of another attribute name must not match —
    # neither a hyphen-joined one (``data-id``) nor a word-char prefix
    # (``grid``); a standalone ``id`` alongside such an attribute still wins.
    assert extract_table_id('data-id="x"') is None
    assert extract_table_id('grid="y"') is None
    assert extract_table_id('data-id="x" id="tb-9"') == "tb-9"
