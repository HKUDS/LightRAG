"""aexport_data(file_format="md"): every value stays inside its table cell.

Descriptions extracted from tables often hold a literal ``|``. Written into a
Markdown table row unescaped, it ends the cell early, so the rest of the value
spills into an extra column that renderers drop. A line break ends the row,
and the ``<SEP>`` joining merged source IDs renders as an unknown HTML tag.
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


class _VdbWithClientStorage(NullVdb):
    """Feeds the raw VectorDB Relationships section (FaissVectorDBStorage shape)."""

    @property
    def client_storage(self) -> dict[str, Any]:
        return {"data": [{"__id__": "rel-1", "content": "net | gross"}]}


def _cells(row: str) -> list[str]:
    """Split a Markdown table row on its unescaped pipes."""
    return [
        cell.strip().replace("\\|", "|")
        for cell in re.split(r"(?<!\\)\|", row.strip())[1:-1]
    ]


@pytest.mark.asyncio
async def test_values_stay_in_their_cells(tmp_path: Path) -> None:
    graph = nx.Graph()
    graph.add_node(
        "REVENUE", description="Q1 | Q2 totals", source_id="chunk-1<SEP>chunk-2"
    )
    graph.add_node("ACME\nCORP", description="vendor", source_id="chunk-3")
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
    assert set(entities) == {"REVENUE", "ACME<br>CORP"}
    assert entities["REVENUE"]["source_id"] == "chunk-1&lt;SEP>chunk-2"
    assert "'description': 'Q1 | Q2 totals'" in entities["REVENUE"]["graph_data"]
    (relation,) = tables["## Relations"]
    assert "'description': 'reported | audited'" in relation["graph_data"]
    (relationship,) = tables["## Relationships"]
    assert "'content': 'net | gross'" in relationship["data"]
