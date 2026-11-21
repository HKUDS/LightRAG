"""aexport_data(file_format="md"): a ``|`` in a value stays inside its cell.

Descriptions extracted from tables often hold a literal ``|``. Written into a
Markdown table row unescaped, it ends the cell early, so the rest of the value
spills into an extra column that renderers drop.
"""

from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import pytest

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.utils import aexport_data

pytestmark = pytest.mark.offline


def _make_storage(graph: nx.Graph) -> NetworkXStorage:
    storage = NetworkXStorage.__new__(NetworkXStorage)
    storage.workspace = "test"
    storage.namespace = "chunk_entity_relation"

    async def _get_graph():
        return graph

    storage._get_graph = _get_graph
    return storage


class _Vdb:
    async def get_by_id(self, _id: str) -> None:
        return None


def _cells(row: str) -> list[str]:
    """Split a Markdown table row on its unescaped pipes."""
    return [
        cell.strip().replace("\\|", "|")
        for cell in re.split(r"(?<!\\)\|", row.strip())[1:-1]
    ]


@pytest.mark.asyncio
async def test_pipe_in_value_stays_in_its_cell(tmp_path: Path) -> None:
    graph = nx.Graph()
    graph.add_node("REVENUE", description="Q1 | Q2 totals", source_id="chunk-1")
    graph.add_edge(
        "REVENUE", "ACME", description="reported | audited", source_id="chunk-2"
    )
    output = tmp_path / "export.md"

    await aexport_data(
        _make_storage(graph), _Vdb(), _Vdb(), str(output), file_format="md"
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    tables = {}
    for heading in ("## Entities", "## Relations"):
        start = lines.index(heading) + 2
        end = lines.index("", start)
        header, _rule, *rows = lines[start:end]
        tables[heading] = [dict(zip(_cells(header), _cells(r))) for r in rows]
        assert all(len(_cells(r)) == len(_cells(header)) for r in rows)

    revenue = next(r for r in tables["## Entities"] if r["entity_name"] == "REVENUE")
    assert "'description': 'Q1 | Q2 totals'" in revenue["graph_data"]
    (relation,) = tables["## Relations"]
    assert "'description': 'reported | audited'" in relation["graph_data"]
