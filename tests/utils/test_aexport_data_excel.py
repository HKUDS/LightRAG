"""aexport_data: the Excel branch must keep working across pandas majors.

The export builds DataFrames from lists of dicts and writes them through
``pd.ExcelWriter(engine="xlsxwriter")``. pandas 3.0 changed the default dtype
for text columns (object -> str) and dropped a number of legacy code paths, so
this pins the observable output: one sheet per non-empty section, no index
column, and ``None`` cells left empty rather than rendered as text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lightrag.utils import aexport_data

pytestmark = pytest.mark.offline


class _FakeGraph:
    """Minimal graph storage surface consumed by aexport_data."""

    def __init__(self, nodes: dict[str, Any], edges: dict[tuple[str, str], Any]):
        self._nodes = nodes
        self._edges = edges

    async def get_all_labels(self) -> list[str]:
        return list(self._nodes)

    async def get_node(self, entity_name: str) -> Any:
        return self._nodes.get(entity_name)

    async def has_edge(self, src: str, tgt: str) -> bool:
        return (src, tgt) in self._edges

    async def get_edge(self, src: str, tgt: str) -> Any:
        return self._edges.get((src, tgt))


class _FakeVdb:
    def __init__(self, data: list[dict[str, Any]] | None = None):
        self._data = data or []

    @property
    async def client_storage(self) -> dict[str, Any]:
        return {"data": self._data}

    async def get_by_id(self, _id: str) -> None:  # pragma: no cover - unused here
        raise AssertionError("include_vector_data=False must not query the VDB")


def _read_sheets(path: Path) -> dict[str, list[list[Any]]]:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.load_workbook(path)
    try:
        return {
            name: [list(row) for row in workbook[name].iter_rows(values_only=True)]
            for name in workbook.sheetnames
        }
    finally:
        workbook.close()


@pytest.mark.asyncio
async def test_excel_export_writes_one_sheet_per_non_empty_section(
    tmp_path: Path,
) -> None:
    graph = _FakeGraph(
        nodes={
            "ALICE": {"entity_type": "person", "source_id": "chunk-1"},
            "BOB": {"entity_type": "person", "source_id": "chunk-2"},
        },
        edges={("ALICE", "BOB"): {"weight": 1.0, "source_id": "chunk-1"}},
    )
    relationships_vdb = _FakeVdb([{"__id__": "rel-1", "src_id": "ALICE"}])
    output = tmp_path / "export.xlsx"

    await aexport_data(
        graph,
        _FakeVdb(),
        relationships_vdb,
        str(output),
        file_format="excel",
    )

    sheets = _read_sheets(output)
    assert set(sheets) == {"Entities", "Relations", "Relationships"}

    # index=False: the first column is a real field, not the DataFrame index.
    assert sheets["Entities"][0] == ["entity_name", "source_id", "graph_data"]
    assert sheets["Entities"][1][:2] == ["ALICE", "chunk-1"]
    assert sheets["Entities"][2][:2] == ["BOB", "chunk-2"]

    assert sheets["Relations"][0] == [
        "src_entity",
        "tgt_entity",
        "source_id",
        "graph_data",
    ]
    assert len(sheets["Relations"]) == 2
    assert sheets["Relations"][1][:3] == ["ALICE", "BOB", "chunk-1"]

    assert sheets["Relationships"][0] == ["relationship_id", "data"]
    assert sheets["Relationships"][1][0] == "rel-1"


@pytest.mark.asyncio
async def test_excel_export_leaves_missing_source_ids_empty(tmp_path: Path) -> None:
    """A source-less node yields source_id=None, which must not become text."""
    graph = _FakeGraph(
        nodes={"ALICE": {"entity_type": "person"}},
        edges={},
    )
    output = tmp_path / "export.xlsx"

    await aexport_data(
        graph,
        _FakeVdb(),
        _FakeVdb([{"__id__": "rel-1"}]),
        str(output),
        file_format="excel",
    )

    sheets = _read_sheets(output)
    # No edges and no relations rows -> that sheet is skipped entirely.
    assert set(sheets) == {"Entities", "Relationships"}
    header, row = sheets["Entities"]
    assert row[header.index("source_id")] is None
    assert row[header.index("graph_data")] == str({"entity_type": "person"})
