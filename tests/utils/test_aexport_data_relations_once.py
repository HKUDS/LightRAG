"""aexport_data: each relation is exported once, not once per direction.

Relations are undirected in every graph backend: ``has_edge(a, b)`` and
``has_edge(b, a)`` both answer for the same edge. The export used to probe
every *ordered* entity pair, so each relation came out twice, once per
orientation, doubling the Relations section of the file.
"""

from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx
import pytest

from lightrag.utils import aexport_data
from tests.utils._export_fakes import NullVdb, make_networkx_storage

pytestmark = pytest.mark.offline


def _relation_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("# RELATIONS") + 1
    end = lines.index("", start)
    return list(csv.DictReader(lines[start:end]))


@pytest.mark.asyncio
async def test_undirected_relation_is_exported_once(tmp_path: Path) -> None:
    graph = nx.Graph()
    graph.add_edge("ALICE", "BOB", description="knows", source_id="chunk-1")
    graph.add_edge("CAROL", "BOB", description="works with", source_id="chunk-2")
    graph.add_node("DAVE", source_id="chunk-3")
    output = tmp_path / "export.csv"

    await aexport_data(
        make_networkx_storage(graph),
        NullVdb(),
        NullVdb(),
        str(output),
        file_format="csv",
    )

    rows = _relation_rows(output)
    exported = sorted(
        tuple(sorted((row["src_entity"], row["tgt_entity"]))) for row in rows
    )
    assert exported == [("ALICE", "BOB"), ("BOB", "CAROL")]
    assert {row["source_id"] for row in rows} == {"chunk-1", "chunk-2"}
