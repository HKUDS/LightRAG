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
    dumps_json_table_body,
    escape_json_table_body,
    parse_table_tag,
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

    assert "</" not in body
    assert json.loads(body) == rows


@pytest.mark.parametrize("cell", _TRICKY_CELLS)
def test_tag_built_from_the_body_parses_back_whole(cell: str) -> None:
    rows = [["Tag", "Meaning"], [cell, "x"], ["tail", cell]]
    tag = f'<table id="tb-1" format="json">{dumps_json_table_body(rows)}</table>'

    match = TABLE_TAG_RE.match(tag)
    assert match is not None
    assert match.end() == len(tag)  # the first "</table>" is the real one
    assert parse_table_tag(tag) == ('id="tb-1" format="json"', rows)


def test_body_without_closing_tag_text_is_plain_json_dumps() -> None:
    """Tables without ``</`` keep the encoding they always had."""
    rows = [["a", "b/c"], ["<tr>", "1 < 2"], [{"k": "v"}, 3]]

    assert dumps_json_table_body(rows) == json.dumps(rows, ensure_ascii=False)


def test_escape_is_idempotent() -> None:
    once = escape_json_table_body(json.dumps([["</table>", "\\</"]]))

    assert escape_json_table_body(once) == once


def test_random_cells_round_trip() -> None:
    """Differential check against ``json.loads`` over the hazard alphabet."""
    rng = random.Random(4450)
    alphabet = ["<", "/", "\\", '"', ">", "t", "a", "b", "l", "e", " ", "\n", "表"]
    for _ in range(2000):
        rows = [
            ["".join(rng.choices(alphabet, k=rng.randint(0, 12))) for _ in range(3)]
            for _ in range(rng.randint(1, 3))
        ]
        body = dumps_json_table_body(rows)
        assert "</" not in body
        assert json.loads(body) == rows
