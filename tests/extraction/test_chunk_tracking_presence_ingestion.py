"""Regression tests (#3609 follow-up): the ingestion merge paths must not
reseed a present-but-empty chunk-tracking row from the graph's ``source_id``.

The entity/relation tracking row is AUTHORITATIVE; a graph object's
``source_id`` is only a truncated view of it and may legitimately still name
chunks a previous purge already pruned. PR #3660 fixed the absent-vs-empty
conflation in the edit/rename paths (``utils_graph.py``); these tests pin the
same rule for the three ingestion sites in ``operate.py``:

- ``_merge_nodes_then_upsert`` (entity merge on insert),
- ``_merge_edges_then_upsert`` (relation merge on insert),
- the edge-endpoint node update inside ``_merge_edges_then_upsert``.

Before the fix each site fell back to the graph ``source_id`` whenever the
stored ``chunk_ids`` list was falsy — resurrecting stale attribution into the
authoritative store through an ordinary document insert, the most reachable
path of all. The absent-row direction (fall back to ``source_id``) is pinned
by twin tests so the guard is narrowed, not widened.
"""

from __future__ import annotations

import pytest

import lightrag.operate as operate
from lightrag.operate import _merge_edges_then_upsert, _merge_nodes_then_upsert
from lightrag.utils import make_relation_chunk_key

pytestmark = pytest.mark.offline


class _FakeTokenizer:
    def encode(self, s: str):
        return list(range(len(s)))


class _MemGraph:
    """Minimal in-memory graph for real get -> merge -> upsert round-trips."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict = {}

    async def get_node(self, name):
        return self.nodes.get(name)

    async def has_node(self, name):
        return name in self.nodes

    async def upsert_node(self, name, node_data):
        self.nodes[name] = dict(node_data)

    async def has_edge(self, s, t):
        return (s, t) in self.edges or (t, s) in self.edges

    async def get_edge(self, s, t):
        return self.edges.get((s, t)) or self.edges.get((t, s))

    async def upsert_edge(self, s, t, edge_data):
        self.edges[(s, t)] = dict(edge_data)

    async def get_nodes_batch(self, names):
        return {n: self.nodes.get(n) for n in names}

    async def get_edges_batch(self, pairs):
        out = {}
        for p in pairs:
            s, t = (p["src"], p["tgt"]) if isinstance(p, dict) else p
            out[(s, t)] = self.edges.get((s, t)) or self.edges.get((t, s))
        return out


class _MemVdb:
    async def upsert(self, data):
        pass

    async def delete(self, ids):
        pass


class _MemKV:
    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(k) for k in keys]

    async def upsert(self, data):
        self.data.update(data)

    async def index_done_callback(self):
        pass


def _cfg() -> dict:
    return {
        "tokenizer": _FakeTokenizer(),
        "summary_context_size": 1_000_000,
        "summary_max_tokens": 1_000_000,
        "force_llm_summary_on_merge": 6,
        "source_ids_limit_method": operate.SOURCE_IDS_LIMIT_METHOD_KEEP,
        "max_source_ids_per_entity": 10_000,
        "max_source_ids_per_relation": 10_000,
        "max_file_paths": 100,
        "file_path_more_placeholder": "...",
    }


def _node(source_id: str) -> dict:
    return {
        "entity_name": "A",
        "entity_type": "person",
        "description": "Entity A",
        "source_id": source_id,
        "file_path": "doc.txt",
        "timestamp": 1,
    }


def _rel(source_id: str) -> dict:
    return {
        "weight": 1.0,
        "source_id": source_id,
        "description": "rel",
        "keywords": "k",
        "file_path": "doc.txt",
    }


async def _graph_with_node(source_id: str = "c-stale") -> _MemGraph:
    g = _MemGraph()
    await g.upsert_node(
        "A",
        {
            "entity_id": "A",
            "entity_type": "person",
            "description": "Entity A",
            "source_id": source_id,
            "file_path": "doc.txt",
        },
    )
    return g


async def _graph_with_edge_endpoints(source_id: str = "c-node") -> _MemGraph:
    g = await _graph_with_node(source_id)
    await g.upsert_node(
        "B",
        {
            "entity_id": "B",
            "entity_type": "person",
            "description": "Entity B",
            "source_id": source_id,
            "file_path": "doc.txt",
        },
    )
    return g


# --- _merge_nodes_then_upsert (entity ingestion merge) ----------------------


@pytest.mark.asyncio
async def test_node_merge_does_not_reseed_present_but_empty_entity_row():
    """A curated-empty entity row plus a stale graph source_id: re-extracting
    the entity must append only the genuinely new chunk, never resurrect the
    purged one from the graph's truncated view."""
    g = await _graph_with_node("c-stale")
    ecs = _MemKV({"A": {"chunk_ids": [], "count": 0}})

    await _merge_nodes_then_upsert(
        "A", [_node("c-new")], g, None, _cfg(), entity_chunks_storage=ecs
    )

    assert ecs.data["A"]["chunk_ids"] == ["c-new"]


@pytest.mark.asyncio
async def test_node_merge_seeds_from_graph_when_row_absent():
    """Absent-row pin: with no tracking row at all, the graph source_id is
    still the seed (migration/initialization semantics are unchanged)."""
    g = await _graph_with_node("c-old")
    ecs = _MemKV()

    await _merge_nodes_then_upsert(
        "A", [_node("c-new")], g, None, _cfg(), entity_chunks_storage=ecs
    )

    assert ecs.data["A"]["chunk_ids"] == ["c-old", "c-new"]


# --- _merge_edges_then_upsert (relation ingestion merge) --------------------


@pytest.mark.asyncio
async def test_edge_merge_does_not_reseed_present_but_empty_relation_row():
    g = await _graph_with_edge_endpoints()
    await g.upsert_edge("A", "B", _rel("c-stale"))
    rcs = _MemKV({make_relation_chunk_key("A", "B"): {"chunk_ids": [], "count": 0}})

    await _merge_edges_then_upsert(
        "A",
        "B",
        [_rel("c-new")],
        g,
        _MemVdb(),
        _MemVdb(),
        _cfg(),
        relation_chunks_storage=rcs,
    )

    assert rcs.data[make_relation_chunk_key("A", "B")]["chunk_ids"] == ["c-new"]


@pytest.mark.asyncio
async def test_edge_merge_seeds_from_graph_when_row_absent():
    g = await _graph_with_edge_endpoints()
    await g.upsert_edge("A", "B", _rel("c-old"))
    rcs = _MemKV()

    await _merge_edges_then_upsert(
        "A",
        "B",
        [_rel("c-new")],
        g,
        _MemVdb(),
        _MemVdb(),
        _cfg(),
        relation_chunks_storage=rcs,
    )

    assert rcs.data[make_relation_chunk_key("A", "B")]["chunk_ids"] == [
        "c-old",
        "c-new",
    ]


# --- edge-endpoint node update inside _merge_edges_then_upsert --------------


@pytest.mark.asyncio
async def test_edge_endpoint_update_does_not_reseed_present_but_empty_row():
    """The endpoint-node update branch merges the edge's chunks into each
    existing node's tracking; a curated-empty row must stay the baseline."""
    g = await _graph_with_edge_endpoints("c-stale")
    ecs = _MemKV(
        {
            "A": {"chunk_ids": [], "count": 0},
            "B": {"chunk_ids": [], "count": 0},
        }
    )

    await _merge_edges_then_upsert(
        "A",
        "B",
        [_rel("c-new")],
        g,
        _MemVdb(),
        _MemVdb(),
        _cfg(),
        entity_chunks_storage=ecs,
    )

    assert ecs.data["A"]["chunk_ids"] == ["c-new"]
    assert ecs.data["B"]["chunk_ids"] == ["c-new"]


@pytest.mark.asyncio
async def test_edge_endpoint_update_seeds_from_graph_when_row_absent():
    g = await _graph_with_edge_endpoints("c-node")
    ecs = _MemKV()

    await _merge_edges_then_upsert(
        "A",
        "B",
        [_rel("c-new")],
        g,
        _MemVdb(),
        _MemVdb(),
        _cfg(),
        entity_chunks_storage=ecs,
    )

    assert ecs.data["A"]["chunk_ids"] == ["c-node", "c-new"]
    assert ecs.data["B"]["chunk_ids"] == ["c-node", "c-new"]
