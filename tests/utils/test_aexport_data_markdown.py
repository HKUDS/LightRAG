"""aexport_data(file_format="md"): every value stays inside its table cell.

Descriptions extracted from tables often hold a literal ``|``. Written into a
Markdown table row unescaped, it ends the cell early, so the rest of the value
spills into an extra column that renderers drop. A line break ends the row,
and a stored backslash right before a pipe can cancel the pipe's escape.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from lightrag.utils import aexport_data
from tests.utils._export_fakes import NullVdb, make_networkx_storage

pytestmark = pytest.mark.offline

_RELATIONSHIP = {"__id__": "rel-1", "content": "net | gross"}


class _VdbWithClientStorage(NullVdb):
    """Feeds the raw VectorDB Relationships section (FaissVectorDBStorage shape)."""

    @property
    def client_storage(self) -> dict[str, Any]:
        return {"data": [dict(_RELATIONSHIP)]}


def _cells(row: str) -> list[str]:
    """Split a row the way a backslash-parity renderer does, then decode.

    A pipe ends a cell unless an odd number of backslashes precedes it. Each
    cell is decoded by inverting the export's two rewrites only: ``<br>`` is a
    line break, and ``2n`` backslashes plus ``\\|`` are ``n`` backslashes
    plus ``|``. Every other byte must come back as written.
    """
    cells, current, i = [], "", 0
    body = row.strip()[1:]
    while i < len(body):
        char = body[i]
        if char == "\\" and i + 1 < len(body):
            current += char + body[i + 1]
            i += 2
            continue
        if char == "|":
            cells.append(current)
            current = ""
        else:
            current += char
        i += 1
    assert current.strip() == "", f"row does not end with a pipe: {row!r}"
    return [
        re.sub(
            r"(\\*)\\\|",
            lambda m: m.group(1)[: len(m.group(1)) // 2] + "|",
            cell.strip().replace("<br>", "\n"),
        )
        for cell in cells
    ]


@pytest.mark.asyncio
async def test_values_stay_in_their_cells_and_read_back_unchanged(
    tmp_path: Path,
) -> None:
    graph = nx.Graph()
    graph.add_node(
        "REVENUE",
        description="Q1 | Q2 totals for `List<T>` *net*",
        source_id="chunk-1<SEP>chunk-2",
    )
    graph.add_node("ACME\nCORP", description="vendor", source_id="chunk-3")
    graph.add_node("A\\|B", description="path C:\\x", source_id="chunk-5")
    graph.add_edge(
        "REVENUE", "ACME\nCORP", description="reported | audited", source_id="chunk-4"
    )
    output = tmp_path / "export.md"

    await aexport_data(
        make_networkx_storage(graph),
        NullVdb(),
        _VdbWithClientStorage(),
        str(output),
        file_format="md",
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    tables = {}
    for heading in ("## Entities", "## Relations", "## Relationships"):
        start = lines.index(heading) + 2
        end = lines.index("", start) if "" in lines[start:] else len(lines)
        header, _rule, *rows = lines[start:end]
        assert [len(_cells(r)) for r in rows] == [len(_cells(header))] * len(rows)
        tables[heading] = [dict(zip(_cells(header), _cells(r))) for r in rows]

    entities = {r["entity_name"]: r for r in tables["## Entities"]}
    assert set(entities) == set(graph.nodes)
    for name, row in entities.items():
        assert row["source_id"] == graph.nodes[name]["source_id"]
        assert row["graph_data"] == str(graph.nodes[name])
    (relation,) = tables["## Relations"]
    edge = graph.edges["REVENUE", "ACME\nCORP"]
    assert relation["graph_data"] == str(edge)
    (relationship,) = tables["## Relationships"]
    assert relationship["data"] == str(_RELATIONSHIP)
