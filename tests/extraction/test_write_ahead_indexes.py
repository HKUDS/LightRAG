"""Write-ahead recovery indexes in ``merge_nodes_and_edges`` (issue #3400, Phase 2).

The merge must persist and flush the full candidate superset to
``full_entities`` / ``full_relations`` BEFORE the first graph mutation, so a
crash at any later point always leaves a durable recovery anchor. The
historical post-merge "Phase 3" write (derived from in-memory results, with
swallowed exceptions) is gone; anchor persistence failures now abort the
merge before it mutates anything.
"""

from __future__ import annotations

import asyncio

import pytest

import lightrag.operate as operate
from lightrag.exceptions import IndexFlushError
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.operate import merge_nodes_and_edges


class _FakeTokenizer:
    def encode(self, s: str):
        return list(range(len(s)))


class _OrderLog:
    """Shared event recorder so stores can prove cross-store write ordering."""

    def __init__(self):
        self.events: list[str] = []

    def add(self, event: str):
        self.events.append(event)

    def first(self, prefix: str) -> int:
        for i, e in enumerate(self.events):
            if e.startswith(prefix):
                return i
        return -1


class _MemGraph:
    def __init__(self, order: _OrderLog):
        self.order = order
        self.nodes: dict[str, dict] = {}
        self.edges: dict = {}

    async def get_node(self, name):
        return self.nodes.get(name)

    async def has_node(self, name):
        return name in self.nodes

    async def upsert_node(self, name, node_data):
        self.order.add(f"graph.upsert_node:{name}")
        self.nodes[name] = dict(node_data)

    async def has_edge(self, s, t):
        return (s, t) in self.edges or (t, s) in self.edges

    async def get_edge(self, s, t):
        return self.edges.get((s, t)) or self.edges.get((t, s))

    async def upsert_edge(self, s, t, edge_data):
        self.order.add(f"graph.upsert_edge:{s}~{t}")
        self.edges[(s, t)] = dict(edge_data)


class _MemVdb:
    async def upsert(self, data):
        pass

    async def delete(self, ids):
        pass


class _MemKV:
    def __init__(self, order: _OrderLog, name: str):
        self.order = order
        self.name = name
        self.data: dict = {}

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(k) for k in keys]

    async def upsert(self, data):
        self.order.add(f"{self.name}.upsert")
        self.data.update(data)

    async def index_done_callback(self):
        self.order.add(f"{self.name}.flush")


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


def _node_dp(name: str, src: str) -> dict:
    return {
        "entity_name": name,
        "entity_type": "person",
        "description": f"{name} desc",
        "source_id": src,
        "file_path": "d.txt",
        "timestamp": 1,
    }


def _chunk_results(src: str = "c1"):
    """ALICE + ACME entities and an ALICE~ACME edge; BOB appears only as a
    relation endpoint (never extracted standalone)."""
    maybe_nodes = {
        "ALICE": [_node_dp("ALICE", src)],
        "ACME": [_node_dp("ACME", src)],
    }
    maybe_edges = {
        ("ALICE", "BOB"): [
            {
                "src_id": "ALICE",
                "tgt_id": "BOB",
                "weight": 1.0,
                "description": "rel",
                "keywords": "k",
                "source_id": src,
                "file_path": "d.txt",
                "timestamp": 1,
            }
        ]
    }
    return [(maybe_nodes, maybe_edges)]


async def _merge(chunk_results, order: _OrderLog, **overrides):
    initialize_share_data()
    stores = {
        "full_entities": _MemKV(order, "full_entities"),
        "full_relations": _MemKV(order, "full_relations"),
        "entity_chunks": _MemKV(order, "entity_chunks"),
        "relation_chunks": _MemKV(order, "relation_chunks"),
    }
    stores.update({k: v for k, v in overrides.items() if k in stores})
    graph = overrides.get("graph") or _MemGraph(order)
    await merge_nodes_and_edges(
        chunk_results,
        graph,
        _MemVdb(),
        _MemVdb(),
        _cfg(),
        full_entities_storage=stores["full_entities"],
        full_relations_storage=stores["full_relations"],
        doc_id="d1",
        pipeline_status={"history_messages": []},
        pipeline_status_lock=asyncio.Lock(),
        entity_chunks_storage=stores["entity_chunks"],
        relation_chunks_storage=stores["relation_chunks"],
    )
    return graph, stores


@pytest.mark.offline
@pytest.mark.asyncio
async def test_anchors_written_and_flushed_before_first_graph_mutation():
    order = _OrderLog()
    await _merge(_chunk_results(), order)

    first_mutation = min(
        i
        for i in (order.first("graph.upsert_node"), order.first("graph.upsert_edge"))
        if i >= 0
    )
    for prefix in (
        "full_entities.upsert",
        "full_relations.upsert",
        "full_entities.flush",
        "full_relations.flush",
    ):
        idx = order.first(prefix)
        assert idx >= 0, f"{prefix} never happened"
        assert idx < first_mutation, (
            f"{prefix} (at {idx}) must precede the first graph mutation "
            f"(at {first_mutation}): {order.events}"
        )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_anchor_rows_contain_candidate_superset():
    """The prewritten entity anchor includes relation endpoints (BOB) that
    edge processing may create, and the relation anchor the sorted pair."""
    order = _OrderLog()
    _, stores = await _merge(_chunk_results(), order)

    assert sorted(stores["full_entities"].data["d1"]["entity_names"]) == [
        "ACME",
        "ALICE",
        "BOB",
    ]
    assert stores["full_relations"].data["d1"]["relation_pairs"] == [["ALICE", "BOB"]]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_empty_extraction_still_writes_both_anchor_rows():
    """Zero candidates must still overwrite both rows (a reprocess yielding
    nothing must not leave the previous attempt's stale anchors)."""
    order = _OrderLog()
    _, stores = await _merge([({}, {})], order)

    assert stores["full_entities"].data["d1"] == {"entity_names": [], "count": 0}
    assert stores["full_relations"].data["d1"] == {"relation_pairs": [], "count": 0}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reprocess_with_fewer_candidates_overwrites_stale_anchor():
    order = _OrderLog()
    initialize_share_data()
    full_entities = _MemKV(order, "full_entities")
    full_entities.data["d1"] = {"entity_names": ["STALE"], "count": 1}

    _, stores = await _merge(_chunk_results(), order, full_entities=full_entities)
    assert "STALE" not in stores["full_entities"].data["d1"]["entity_names"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_anchor_upsert_failure_aborts_merge_before_mutation():
    """The historical Phase-3 write swallowed exceptions; the write-ahead
    write must propagate AND nothing may have been merged into the graph."""
    order = _OrderLog()

    class _FailingKV(_MemKV):
        async def upsert(self, data):
            raise RuntimeError("anchor upsert boom")

    graph = _MemGraph(order)
    with pytest.raises(RuntimeError, match="anchor upsert boom"):
        await _merge(
            _chunk_results(),
            order,
            full_entities=_FailingKV(order, "full_entities"),
            graph=graph,
        )
    assert graph.nodes == {} and graph.edges == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_anchor_flush_failure_aborts_merge_before_mutation():
    order = _OrderLog()

    class _FlushFailKV(_MemKV):
        async def index_done_callback(self):
            raise RuntimeError("anchor flush boom")

    graph = _MemGraph(order)
    with pytest.raises(IndexFlushError):
        await _merge(
            _chunk_results(),
            order,
            full_relations=_FlushFailKV(order, "full_relations"),
            graph=graph,
        )
    assert graph.nodes == {} and graph.edges == {}
