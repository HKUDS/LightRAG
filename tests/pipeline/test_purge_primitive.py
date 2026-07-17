"""Candidate-driven purge/rebuild primitive (issue #3400, Phase 1).

``_purge_kg_contributions`` is the shared lower-level primitive behind
whole-document purge (``_purge_doc_chunks_and_kg``) and — in later phases —
custom-chunk patch rollback. These tests drive it with in-memory fakes:

- candidates may be explicit (journal/prewrite driven) or discovered from the
  per-doc ``full_entities`` / ``full_relations`` rows;
- a candidate absent from the graph is an idempotent no-op, never an error
  (candidates are a recovery SUPERSET);
- ``patch_only=True`` must keep the base document's recovery rows;
- ``strict_rebuild=True`` propagates missing-extraction-cache errors instead
  of silently succeeding.
"""

from __future__ import annotations

import asyncio

import pytest

from lightrag import LightRAG
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.exceptions import KGRebuildCacheMissingError
from lightrag.utils import compute_mdhash_id, make_relation_chunk_key


class _KV:
    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})
        self.deleted: list[str] = []

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(k) for k in keys]

    async def upsert(self, data):
        self.data.update(data)

    async def delete(self, ids):
        self.deleted.extend(ids)
        for k in ids:
            self.data.pop(k, None)

    async def index_done_callback(self):
        pass


class _Vdb:
    def __init__(self):
        self.deleted: list[str] = []

    async def delete(self, ids):
        self.deleted.extend(ids)

    async def index_done_callback(self):
        pass


class _Graph:
    def __init__(self, nodes: dict | None = None, edges: dict | None = None):
        self.nodes = dict(nodes or {})
        self.edges = dict(edges or {})
        self.removed_nodes: list[str] = []
        self.removed_edges: list[tuple[str, str]] = []

    async def get_nodes_batch(self, names):
        return {n: dict(self.nodes[n]) if n in self.nodes else None for n in names}

    async def get_edges_batch(self, pairs):
        out = {}
        for p in pairs:
            s, t = p["src"], p["tgt"]
            edge = self.edges.get((s, t)) or self.edges.get((t, s))
            out[(s, t)] = dict(edge) if edge else None
        return out

    async def get_nodes_edges_batch(self, names):
        return {n: [] for n in names}

    async def remove_nodes(self, names):
        self.removed_nodes.extend(names)
        for n in names:
            self.nodes.pop(n, None)

    async def remove_edges(self, pairs):
        self.removed_edges.extend(pairs)
        for s, t in pairs:
            self.edges.pop((s, t), None)
            self.edges.pop((t, s), None)


def _node(name: str, sources: list[str]) -> dict:
    return {
        "entity_id": name,
        "description": name,
        "source_id": GRAPH_FIELD_SEP.join(sources),
        "entity_type": "X",
        "file_path": "f",
    }


def _edge(sources: list[str]) -> dict:
    return {
        "description": "rel",
        "keywords": "k",
        "weight": 1.0,
        "source_id": GRAPH_FIELD_SEP.join(sources),
        "file_path": "f",
    }


def _make_rag(
    *,
    graph: _Graph,
    full_entities: _KV | None = None,
    full_relations: _KV | None = None,
    entity_chunks: _KV | None = None,
    relation_chunks: _KV | None = None,
    text_chunks: _KV | None = None,
    llm_cache: _KV | None = None,
) -> LightRAG:
    rag = LightRAG.__new__(LightRAG)
    rag.chunk_entity_relation_graph = graph
    rag.full_entities = full_entities if full_entities is not None else _KV()
    rag.full_relations = full_relations if full_relations is not None else _KV()
    rag.entity_chunks = entity_chunks if entity_chunks is not None else _KV()
    rag.relation_chunks = relation_chunks if relation_chunks is not None else _KV()
    rag.chunks_vdb = _Vdb()
    rag.entities_vdb = _Vdb()
    rag.relationships_vdb = _Vdb()
    rag.text_chunks = text_chunks if text_chunks is not None else _KV()
    rag.llm_response_cache = llm_cache if llm_cache is not None else _KV()

    async def _noop_insert_done(*args, **kwargs):
        return None

    rag._insert_done = _noop_insert_done
    rag._build_global_config = lambda: {"llm_model_max_async": 1}
    return rag


def _status():
    return {"latest_message": "", "history_messages": []}, asyncio.Lock()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_whole_doc_purge_from_recovery_rows():
    """Wrapper path (regression): candidates come from the per-doc rows;
    fully-owned entity/relation are removed from graph, vector, and tracking,
    and the recovery rows are deleted at the end."""
    graph = _Graph(
        nodes={"ALICE": _node("ALICE", ["c1"]), "ACME": _node("ACME", ["c1"])},
        edges={("ACME", "ALICE"): _edge(["c1"])},
    )
    rag = _make_rag(
        graph=graph,
        full_entities=_KV({"d1": {"entity_names": ["ALICE", "ACME"], "count": 2}}),
        full_relations=_KV({"d1": {"relation_pairs": [["ACME", "ALICE"]], "count": 1}}),
        entity_chunks=_KV(
            {
                "ALICE": {"chunk_ids": ["c1"]},
                "ACME": {"chunk_ids": ["c1"]},
            }
        ),
        relation_chunks=_KV(
            {make_relation_chunk_key("ACME", "ALICE"): {"chunk_ids": ["c1"]}}
        ),
    )
    status, lock = _status()

    await rag._purge_doc_chunks_and_kg(
        "d1", ["c1"], pipeline_status=status, pipeline_status_lock=lock
    )

    assert set(graph.removed_nodes) == {"ALICE", "ACME"}
    assert graph.removed_edges == [("ACME", "ALICE")]
    assert rag.chunks_vdb.deleted == ["c1"]
    assert rag.text_chunks.deleted == ["c1"]
    assert set(rag.entities_vdb.deleted) == {
        compute_mdhash_id("ALICE", prefix="ent-"),
        compute_mdhash_id("ACME", prefix="ent-"),
    }
    # Recovery rows removed last (whole-doc mode).
    assert rag.full_entities.deleted == ["d1"]
    assert rag.full_relations.deleted == ["d1"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_explicit_candidates_bypass_recovery_rows():
    """Explicit candidates (journal/prewrite callers) must be used verbatim —
    the per-doc rows are not consulted for discovery."""
    graph = _Graph(nodes={"ALICE": _node("ALICE", ["c1"])})

    class _ExplodingKV(_KV):
        async def get_by_id(self, key):  # discovery read would blow up
            raise AssertionError("full-index row must not be read")

    rag = _make_rag(
        graph=graph,
        full_entities=_ExplodingKV(),
        full_relations=_ExplodingKV(),
        entity_chunks=_KV({"ALICE": {"chunk_ids": ["c1"]}}),
    )
    status, lock = _status()

    await rag._purge_kg_contributions(
        "d1",
        ["c1"],
        candidate_entities=["ALICE"],
        candidate_relations=[],
        patch_only=True,
        pipeline_status=status,
        pipeline_status_lock=lock,
    )

    assert graph.removed_nodes == ["ALICE"]
    # patch_only: base-document recovery rows untouched.
    assert rag.full_entities.deleted == []
    assert rag.full_relations.deleted == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_absent_candidates_are_idempotent_noops():
    """A candidate that never reached the graph (prewritten superset, or a
    previous partial purge already removed it) must be skipped silently."""
    graph = _Graph()  # empty graph — nothing exists
    rag = _make_rag(graph=graph)
    status, lock = _status()

    await rag._purge_kg_contributions(
        "d1",
        ["c1"],
        candidate_entities=["GHOST"],
        candidate_relations=[("GHOST", "PHANTOM")],
        pipeline_status=status,
        pipeline_status_lock=lock,
    )

    assert graph.removed_nodes == []
    assert graph.removed_edges == []
    # Chunks are still deleted (they are the doc's own data)...
    assert rag.chunks_vdb.deleted == ["c1"]
    # ...and whole-doc mode still clears the recovery rows.
    assert rag.full_entities.deleted == ["d1"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_empty_chunk_ids_returns_without_touching_storage():
    graph = _Graph(nodes={"ALICE": _node("ALICE", ["c1"])})
    rag = _make_rag(graph=graph)
    status, lock = _status()

    await rag._purge_kg_contributions(
        "d1", [], pipeline_status=status, pipeline_status_lock=lock
    )

    assert graph.removed_nodes == []
    assert rag.chunks_vdb.deleted == []
    assert rag.full_entities.deleted == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_purge_deletes_chunks_only_after_graph_repair(monkeypatch):
    """Safe destructive ordering (Phase 2): graph/vector/tracking cleanup is
    flushed BEFORE the source chunks are deleted, and the recovery rows go
    last — so a crash at any point leaves either the chunks or the anchors
    (or both) for a retry."""
    graph = _Graph(
        nodes={"ALICE": _node("ALICE", ["c1"])},
    )
    rag = _make_rag(
        graph=graph,
        full_entities=_KV({"d1": {"entity_names": ["ALICE"], "count": 1}}),
        full_relations=_KV({"d1": {"relation_pairs": [], "count": 0}}),
        entity_chunks=_KV({"ALICE": {"chunk_ids": ["c1"]}}),
    )
    order: list[str] = []

    orig_remove_nodes = graph.remove_nodes

    async def spy_remove_nodes(names):
        order.append("graph.remove_nodes")
        await orig_remove_nodes(names)

    graph.remove_nodes = spy_remove_nodes

    async def spy_insert_done(*a, **k):
        order.append("flush")

    rag._insert_done = spy_insert_done

    orig_chunk_delete = rag.chunks_vdb.delete

    async def spy_chunk_delete(ids):
        order.append("chunks.delete")
        await orig_chunk_delete(ids)

    monkeypatch.setattr(rag.chunks_vdb, "delete", spy_chunk_delete)

    orig_fe_delete = rag.full_entities.delete

    async def spy_fe_delete(ids):
        order.append("anchors.delete")
        await orig_fe_delete(ids)

    monkeypatch.setattr(rag.full_entities, "delete", spy_fe_delete)

    status, lock = _status()
    await rag._purge_doc_chunks_and_kg(
        "d1", ["c1"], pipeline_status=status, pipeline_status_lock=lock
    )

    assert order.index("graph.remove_nodes") < order.index("flush")
    assert order.index("flush") < order.index("chunks.delete")
    assert order.index("chunks.delete") < order.index("anchors.delete")


@pytest.mark.offline
@pytest.mark.asyncio
async def test_graph_delete_failure_keeps_chunks_and_anchors():
    """A failure while repairing graph contributions must leave both the
    chunks and the recovery rows untouched — fully retryable."""
    graph = _Graph(nodes={"ALICE": _node("ALICE", ["c1"])})

    async def boom(names):
        raise RuntimeError("graph delete boom")

    graph.remove_nodes = boom
    rag = _make_rag(
        graph=graph,
        full_entities=_KV({"d1": {"entity_names": ["ALICE"], "count": 1}}),
        entity_chunks=_KV({"ALICE": {"chunk_ids": ["c1"]}}),
    )
    status, lock = _status()

    with pytest.raises(Exception, match="Failed to delete entities"):
        await rag._purge_doc_chunks_and_kg(
            "d1", ["c1"], pipeline_status=status, pipeline_status_lock=lock
        )

    assert rag.chunks_vdb.deleted == []
    assert rag.text_chunks.deleted == []
    assert rag.full_entities.deleted == []
    assert "d1" in rag.full_entities.data


@pytest.mark.offline
@pytest.mark.asyncio
async def test_chunk_delete_failure_keeps_recovery_rows(monkeypatch):
    """A failure while deleting chunks must keep the recovery rows so the
    purge can be repeated."""
    graph = _Graph(nodes={"ALICE": _node("ALICE", ["c1"])})
    rag = _make_rag(
        graph=graph,
        full_entities=_KV({"d1": {"entity_names": ["ALICE"], "count": 1}}),
        entity_chunks=_KV({"ALICE": {"chunk_ids": ["c1"]}}),
    )

    async def boom(ids):
        raise RuntimeError("chunk delete boom")

    monkeypatch.setattr(rag.chunks_vdb, "delete", boom)
    status, lock = _status()

    with pytest.raises(Exception, match="Failed to delete document chunks"):
        await rag._purge_doc_chunks_and_kg(
            "d1", ["c1"], pipeline_status=status, pipeline_status_lock=lock
        )

    assert rag.full_entities.deleted == []
    assert "d1" in rag.full_entities.data


@pytest.mark.offline
@pytest.mark.asyncio
async def test_strict_rebuild_failure_propagates_through_purge():
    """An entity that keeps sources from another chunk becomes a rebuild
    target; with strict_rebuild=True and no extraction cache the purge must
    FAIL (retaining a retryable dirty state), not silently 'succeed'."""
    graph = _Graph(nodes={"ALICE": _node("ALICE", ["c1", "c2"])})
    rag = _make_rag(
        graph=graph,
        entity_chunks=_KV({"ALICE": {"chunk_ids": ["c1", "c2"]}}),
        # c2 survives but has no cached extraction to rebuild from.
        text_chunks=_KV({"c2": {"content": "x", "llm_cache_list": []}}),
    )
    status, lock = _status()

    with pytest.raises(Exception) as excinfo:
        await rag._purge_kg_contributions(
            "d1",
            ["c1"],
            candidate_entities=["ALICE"],
            candidate_relations=[],
            strict_rebuild=True,
            pipeline_status=status,
            pipeline_status_lock=lock,
        )
    assert isinstance(excinfo.value.__cause__, KGRebuildCacheMissingError)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_non_strict_rebuild_keeps_best_effort_behavior():
    """Same partial-cache scenario without strict_rebuild: historical
    best-effort behavior (warn + continue) is preserved for existing callers."""
    graph = _Graph(nodes={"ALICE": _node("ALICE", ["c1", "c2"])})
    rag = _make_rag(
        graph=graph,
        entity_chunks=_KV({"ALICE": {"chunk_ids": ["c1", "c2"]}}),
        text_chunks=_KV({"c2": {"content": "x", "llm_cache_list": []}}),
    )
    status, lock = _status()

    await rag._purge_kg_contributions(
        "d1",
        ["c1"],
        candidate_entities=["ALICE"],
        candidate_relations=[],
        pipeline_status=status,
        pipeline_status_lock=lock,
    )
    # Tracking narrowed to the surviving source.
    assert rag.entity_chunks.data["ALICE"]["chunk_ids"] == ["c2"]
