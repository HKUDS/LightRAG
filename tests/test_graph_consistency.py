"""Tests for graph/VDB consistency utilities and the edge-count drift fix.

These are offline tests that use only the NetworkX in-memory graph backend
and a mock VDB — no external services required.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any


# ---------------------------------------------------------------------------
# Helpers / minimal stubs
# ---------------------------------------------------------------------------


def _make_graph_storage(edges: list[tuple[str, str]]) -> MagicMock:
    """Return a mock BaseGraphStorage pre-populated with the given edges."""
    storage = MagicMock()
    storage.get_all_edges = AsyncMock(
        return_value=[{"source": s, "target": t} for s, t in edges]
    )
    removed: list[tuple[str, str]] = []
    storage._removed = removed

    async def _remove_edges(pairs):
        removed.extend(pairs)

    storage.remove_edges = _remove_edges
    return storage


def _make_vdb(present_ids: set[str]) -> MagicMock:
    """Return a mock BaseVectorStorage that knows about `present_ids`."""
    vdb = MagicMock()

    async def _get_by_ids(ids: list[str]) -> list[Any]:
        return [{"id": i} if i in present_ids else None for i in ids]

    vdb.get_by_ids = _get_by_ids
    vdb.upsert = AsyncMock()
    vdb.delete = AsyncMock()
    return vdb


# ---------------------------------------------------------------------------
# check_graph_consistency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_consistency_no_edges():
    from lightrag.utils_graph import check_graph_consistency

    graph = _make_graph_storage([])
    vdb = _make_vdb(set())
    report = await check_graph_consistency(graph, vdb)
    assert report["total_graph_edges"] == 0
    assert report["orphan_graph_edges"] == []


@pytest.mark.asyncio
async def test_check_consistency_all_present():
    from lightrag.utils_graph import check_graph_consistency
    from lightrag.utils import compute_mdhash_id

    edges = [("Alice", "Bob"), ("Bob", "Carol")]
    # Build the expected VDB IDs (same normalisation the utility uses)
    ids = set()
    for s, t in edges:
        ns, nt = sorted([s, t])
        ids.add(compute_mdhash_id(ns + nt, prefix="rel-"))

    graph = _make_graph_storage(edges)
    vdb = _make_vdb(ids)
    report = await check_graph_consistency(graph, vdb)
    assert report["total_graph_edges"] == 2
    assert report["total_vdb_relations"] == 2
    assert report["orphan_graph_edges"] == []


@pytest.mark.asyncio
async def test_check_consistency_detects_orphans():
    from lightrag.utils_graph import check_graph_consistency
    from lightrag.utils import compute_mdhash_id

    edges = [("Alice", "Bob"), ("Bob", "Carol")]
    # Only Alice-Bob is in the VDB
    ns, nt = sorted(["Alice", "Bob"])
    present = {compute_mdhash_id(ns + nt, prefix="rel-")}

    graph = _make_graph_storage(edges)
    vdb = _make_vdb(present)
    report = await check_graph_consistency(graph, vdb)
    assert report["total_graph_edges"] == 2
    assert report["total_vdb_relations"] == 1
    assert len(report["orphan_graph_edges"]) == 1
    orphan_src, orphan_tgt = report["orphan_graph_edges"][0]
    assert {orphan_src, orphan_tgt} == {"Bob", "Carol"}


@pytest.mark.asyncio
async def test_check_consistency_strips_agtype_quotes():
    """PostgreSQL/AGE returns entity_id values wrapped in double-quotes."""
    from lightrag.utils_graph import check_graph_consistency
    from lightrag.utils import compute_mdhash_id

    # Simulate AGE wrapping the values in extra double-quotes
    graph = MagicMock()
    graph.get_all_edges = AsyncMock(
        return_value=[{"source": '"Alice"', "target": '"Bob"'}]
    )

    ns, nt = sorted(["Alice", "Bob"])
    present = {compute_mdhash_id(ns + nt, prefix="rel-")}
    vdb = _make_vdb(present)

    report = await check_graph_consistency(graph, vdb)
    assert report["orphan_graph_edges"] == []


# ---------------------------------------------------------------------------
# repair_graph_consistency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_dry_run_does_not_mutate():
    from lightrag.utils_graph import repair_graph_consistency

    edges = [("Alice", "Bob")]
    graph = _make_graph_storage(edges)
    vdb = _make_vdb(set())  # No VDB entries → orphan

    report = await repair_graph_consistency(graph, vdb, dry_run=True)
    assert len(report["orphan_graph_edges"]) == 1
    assert report["repaired"] is False
    assert graph._removed == []  # Nothing was actually removed


@pytest.mark.asyncio
async def test_repair_removes_orphan_edges():
    from lightrag.utils_graph import repair_graph_consistency

    edges = [("Alice", "Bob"), ("Bob", "Carol")]
    from lightrag.utils import compute_mdhash_id

    ns, nt = sorted(["Alice", "Bob"])
    present = {compute_mdhash_id(ns + nt, prefix="rel-")}

    graph = _make_graph_storage(edges)
    vdb = _make_vdb(present)

    report = await repair_graph_consistency(graph, vdb)
    assert report["repaired"] is True
    assert len(report["orphan_graph_edges"]) == 1
    # The orphan edge was passed to remove_edges
    assert len(graph._removed) == 1
    orphan_src, orphan_tgt = graph._removed[0]
    assert {orphan_src, orphan_tgt} == {"Bob", "Carol"}


@pytest.mark.asyncio
async def test_repair_no_op_when_consistent():
    from lightrag.utils_graph import repair_graph_consistency
    from lightrag.utils import compute_mdhash_id

    edges = [("Alice", "Bob")]
    ns, nt = sorted(["Alice", "Bob"])
    present = {compute_mdhash_id(ns + nt, prefix="rel-")}

    graph = _make_graph_storage(edges)
    vdb = _make_vdb(present)

    report = await repair_graph_consistency(graph, vdb)
    assert report["repaired"] is False
    assert graph._removed == []


# ---------------------------------------------------------------------------
# VDB rollback inside _merge_entities_impl
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_rolls_back_graph_edge_on_vdb_failure():
    """When a VDB upsert throws during merge, the matching graph edge must be
    removed so graph and VDB stay in sync (no orphan edges)."""
    from lightrag.utils_graph import _merge_entities_impl

    # --- graph storage ---
    graph = MagicMock()
    graph.has_node = AsyncMock(side_effect=lambda n: n in {"A", "B", "C"})

    async def _get_node(name):
        return {
            "entity_id": name,
            "description": name,
            "entity_type": "UNKNOWN",
            "source_id": "chunk1",
            "file_path": "",
        }

    graph.get_node = _get_node
    graph.upsert_node = AsyncMock()
    graph.get_node_edges = AsyncMock(
        side_effect=lambda n: [("A", "C")] if n == "A" else []
    )

    async def _get_edge(src, tgt):
        return {
            "description": "A relates to C",
            "keywords": "test",
            "source_id": "chunk1",
            "file_path": "",
            "weight": 1.0,
        }

    graph.get_edge = _get_edge

    rolled_back: list[list[tuple[str, str]]] = []

    async def _remove_edges(pairs):
        rolled_back.append(list(pairs))

    graph.remove_edges = _remove_edges
    graph.delete_node = AsyncMock()
    graph.index_done_callback = AsyncMock(return_value=True)
    graph.upsert_edge = AsyncMock()

    # --- entities VDB ---
    ent_vdb = MagicMock()
    ent_vdb.upsert = AsyncMock()
    ent_vdb.delete = AsyncMock()
    ent_vdb.get_by_id = AsyncMock(return_value=None)
    ent_vdb.index_done_callback = AsyncMock(return_value=True)

    # --- relationships VDB — upsert always fails ---
    rel_vdb = MagicMock()
    rel_vdb.delete = AsyncMock()
    rel_vdb.upsert = AsyncMock(side_effect=Exception("embedder timeout"))
    rel_vdb.index_done_callback = AsyncMock(return_value=True)

    # Run the merge (A → B, with C as a neighbour of A)
    result = await _merge_entities_impl(
        graph,
        ent_vdb,
        rel_vdb,
        source_entities=["A"],
        target_entity="B",
    )

    # The graph write happened (upsert_edge was called)
    graph.upsert_edge.assert_awaited_once()
    # The failed VDB upsert triggered a rollback via remove_edges
    assert len(rolled_back) == 1, "remove_edges should have been called once"
    rolled = rolled_back[0]
    assert len(rolled) == 1
    src, tgt = rolled[0]
    assert {src, tgt} == {"B", "C"}
