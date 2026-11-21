"""Regression tests: an edge's weight counts each distinct source once
(issue #3367 sibling of description accumulation).

``_merge_edges_then_upsert`` finalizes an edge's weight by summing the newly
extracted relations' weights with the stored scalar. It used to sum *every* new
relation, so re-feeding an already-stored source (any reprocess or resume)
re-added its weight, growing 1 -> 2 -> 3 per reprocess, and a source whose
extraction stated the relation twice in one batch counted twice. It now sums one
weight per source not already reflected in the stored scalar, filtering on the
edge's own ``already_source_ids`` (consistent with ``already_weights``), and
ignores falsy source ids. The deletion rebuild's side of the per-chunk rule is
pinned in ``test_relation_evidence_floor_repair.py``.

These drive the real merge round-trip (unit ``_merge_edges_then_upsert`` and the
top-level ``merge_nodes_and_edges`` orchestrator) against in-memory stores, no DB
or LLM.
"""

from __future__ import annotations

import asyncio

import pytest

import lightrag.operate as operate
from lightrag.operate import _merge_edges_then_upsert
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import make_relation_chunk_key


class _FakeTokenizer:
    def encode(self, s: str):  # token count == char count; thresholds kept slack
        return list(range(len(s)))


class _MemGraph:
    """Minimal in-memory graph: real get_node/get_edge -> merge -> upsert round-trip."""

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
    """No-op vector store so the edge-merge entity-vdb path doesn't require a
    real backend."""

    async def upsert(self, data):
        pass

    async def delete(self, ids):
        pass


class _MemKV:
    """No-op-ish KV that serves get_by_id / get_by_ids from a dict."""

    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(k) for k in keys]

    async def upsert(self, data):
        self.data.update(data)

    async def index_done_callback(self):
        # In-memory store: the Phase-0 write-ahead flush barrier is a no-op.
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


async def _edge_graph_with_nodes() -> _MemGraph:
    """A graph with endpoint nodes A, B pre-created (edge merges assume they
    exist). Returns the graph so a test can seed a specific stored edge."""
    g = _MemGraph()
    for name in ("A", "B"):
        await g.upsert_node(
            name,
            {
                "entity_id": name,
                "description": name,
                "source_id": "c1",
                "entity_type": "X",
                "file_path": "f",
            },
        )
    return g


def _rel(src, weight: float = 1.0) -> dict:
    return {
        "weight": weight,
        "source_id": src,
        "description": "rel",
        "keywords": "k",
        "file_path": "f",
    }


# --- unit tests of the edge weight logic -----------------------------------
#
# Every relation an extraction emits carries weight 1.0, so the evidence floor
# (weight >= distinct real sources) lifts an under-counted weight back up on its
# own. A test of a lost or skipped contribution therefore needs a stored boost
# above the floor, or it cannot fail.


@pytest.mark.offline
@pytest.mark.asyncio
async def test_edge_weight_counts_each_source_once_across_reprocess():
    """#3367 sibling: re-feeding an already-stored source (reprocess/resume) adds
    nothing, while each genuinely new source still adds its weight."""
    g = await _edge_graph_with_nodes()
    cfg = _cfg()
    weights: list[float] = []
    for src in ["c1", "c1", "c2", "c2", "c3"]:
        await _merge_edges_then_upsert(
            "A", "B", [_rel(src)], g, _MemVdb(), _MemVdb(), cfg
        )
        weights.append((await g.get_edge("A", "B"))["weight"])
    assert weights == [1.0, 1.0, 2.0, 2.0, 3.0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_edge_weight_not_double_counted_within_one_batch():
    """One merge call's edges_data can carry two records for the SAME source
    (one extraction stated the relation twice). Each source contributes weight
    once per call, not once per record; another source in the batch still counts."""
    g = await _edge_graph_with_nodes()
    await _merge_edges_then_upsert(
        "A",
        "B",
        [_rel("c1"), _rel("c1"), _rel("c2")],
        g,
        _MemVdb(),
        _MemVdb(),
        _cfg(),
    )
    edge = await g.get_edge("A", "B")
    assert edge["source_id"].split(GRAPH_FIELD_SEP) == ["c1", "c2"]
    assert edge["weight"] == 2.0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_edge_weight_adds_new_source_when_chunks_store_is_ahead():
    """relation_chunks_storage can run AHEAD of the graph edge (chunks upserted,
    edge not yet updated). A genuinely new source re-fed then must still add its
    weight: the filter uses the EDGE's own source_ids (consistent with the stored
    scalar), not the chunk store. The stored boost keeps the evidence floor (2)
    from hiding a skipped contribution."""
    g = await _edge_graph_with_nodes()
    g.edges[("A", "B")] = _rel("c1", weight=5.0)
    rcs = _MemKV({make_relation_chunk_key("A", "B"): {"chunk_ids": ["c1", "c2"]}})
    await _merge_edges_then_upsert(
        "A",
        "B",
        [_rel("c2")],
        g,
        _MemVdb(),
        _MemVdb(),
        _cfg(),
        relation_chunks_storage=rcs,
    )
    assert (await g.get_edge("A", "B"))["weight"] == 6.0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_edge_merge_repairs_legacy_weight_below_evidence_floor():
    g = await _edge_graph_with_nodes()
    cfg = _cfg()
    g.edges[("A", "B")] = _rel("c1", weight=0.25)

    await _merge_edges_then_upsert("A", "B", [_rel("c2")], g, _MemVdb(), _MemVdb(), cfg)

    edge = await g.get_edge("A", "B")
    assert set(edge["source_id"].split(GRAPH_FIELD_SEP)) == {"c1", "c2"}
    assert edge["weight"] == 2.0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_edge_weight_stays_consistent_when_new_source_hits_id_limit():
    """When a genuinely new source is KEEP-dropped by max_source_ids_per_relation,
    the stored weight must still match the persisted source count (the drop path
    short-circuits before the weight grows), so weight does not outrun the
    sources actually kept."""
    g = await _edge_graph_with_nodes()
    g.edges[("A", "B")] = _rel("c1")
    cfg = _cfg()
    cfg["max_source_ids_per_relation"] = 1  # c1 already stored; a new c2 is dropped
    await _merge_edges_then_upsert("A", "B", [_rel("c2")], g, _MemVdb(), _MemVdb(), cfg)
    edge = await g.get_edge("A", "B")
    persisted = len([s for s in edge["source_id"].split(GRAPH_FIELD_SEP) if s])
    assert persisted == 1
    assert edge["weight"] == 1.0


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("falsy_source", [None, ""])
async def test_edge_weight_ignores_falsy_source(falsy_source):
    """A falsy source_id (None or "") is excluded from the stored source list, so
    it must not add weight either -- otherwise weight would outgrow the persisted
    source count. Over a stored c1, a falsy-source item keeps weight at 1."""
    g = await _edge_graph_with_nodes()
    g.edges[("A", "B")] = _rel("c1")
    await _merge_edges_then_upsert(
        "A", "B", [_rel(falsy_source)], g, _MemVdb(), _MemVdb(), _cfg()
    )
    edge = await g.get_edge("A", "B")
    persisted = len([s for s in edge["source_id"].split(GRAPH_FIELD_SEP) if s])
    assert persisted == 1
    assert edge["weight"] == 1.0


# --- End-to-end via the real orchestrator merge_nodes_and_edges -------------
#
# The tests above drive the unit _merge_edges_then_upsert helper. This one drives
# the top-level two-phase merge the pipeline actually calls (Phase 1 entities ->
# Phase 2 relations) with in-memory stores, to prove the non-accumulation
# invariant holds through the full reprocess path (weight, plus description as an
# incidental corroboration of the already-merged #3395 fix).


def _node_dp(name: str, desc: str, src: str, ts: int = 1) -> dict:
    return {
        "entity_name": name,
        "entity_type": "person",
        "description": desc,
        "source_id": src,
        "file_path": "d.txt",
        "timestamp": ts,
    }


async def _orchestrate(chunk_results, g, stores, cfg, doc_id="d"):
    from lightrag.operate import merge_nodes_and_edges

    await merge_nodes_and_edges(
        chunk_results,
        g,
        stores["entity_vdb"],
        stores["relationships_vdb"],
        cfg,
        full_entities_storage=stores["full_entities"],
        full_relations_storage=stores["full_relations"],
        doc_id=doc_id,
        pipeline_status={"history_messages": []},
        pipeline_status_lock=asyncio.Lock(),
        entity_chunks_storage=stores["entity_chunks"],
        relation_chunks_storage=stores["relation_chunks"],
    )


def _stores() -> dict:
    return {
        "entity_vdb": _MemVdb(),
        "relationships_vdb": _MemVdb(),
        "full_entities": _MemKV(),
        "full_relations": _MemKV(),
        "entity_chunks": _MemKV(),
        "relation_chunks": _MemKV(),
    }


def _edge_dp(src_id: str, tgt_id: str, src: str) -> dict:
    return {
        "src_id": src_id,
        "tgt_id": tgt_id,
        "weight": 1.0,
        "description": "rel",
        "keywords": "k",
        "source_id": src,
        "file_path": "d.txt",
        "timestamp": 1,
    }


def _one_chunk(src: str, alice_desc: str = "Alice is an engineer."):
    """A single extracted chunk: entities ALICE + ACME and edge ALICE~ACME,
    all attributed to source `src`. The extraction states the relation in both
    directions, as LLM output does; the orchestrator folds both under one sorted
    key, so the edge must still count this chunk once."""
    maybe_nodes = {
        "ALICE": [_node_dp("ALICE", alice_desc, src)],
        "ACME": [_node_dp("ACME", "Acme is a company.", src)],
    }
    maybe_edges = {
        ("ALICE", "ACME"): [_edge_dp("ALICE", "ACME", src)],
        ("ACME", "ALICE"): [_edge_dp("ACME", "ALICE", src)],
    }
    return [(maybe_nodes, maybe_edges)]


def _node_frag_count(g, name: str) -> int:
    return len([d for d in g.nodes[name]["description"].split(GRAPH_FIELD_SEP) if d])


@pytest.mark.offline
@pytest.mark.asyncio
async def test_orchestrator_weight_and_description_track_distinct_sources():
    """Through the real orchestrator: reprocessing a doc (d1 twice, and d1 again
    after d2) grows neither the entity description nor the edge weight, while a
    DISTINCT doc grows both; a chunk stating the relation in both directions
    counts once."""
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data()
    g = _MemGraph()
    stores = _stores()
    cfg = _cfg()
    observed = []
    for src, desc, doc_id in [
        ("c1", "Alice is an engineer.", "d1"),
        ("c1", "Alice is an engineer.", "d1"),
        ("c2", "Alice leads the team.", "d2"),
        ("c1", "Alice is an engineer.", "d1"),
    ]:
        await _orchestrate(_one_chunk(src, desc), g, stores, cfg, doc_id=doc_id)
        observed.append(
            (
                _node_frag_count(g, "ALICE"),
                (await g.get_edge("ALICE", "ACME"))["weight"],
            )
        )
    assert observed == [(1, 1.0), (1, 1.0), (2, 2.0), (2, 2.0)]
