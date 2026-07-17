"""Regression tests: edge weight must not accumulate across reprocess/resume
(issue #3367 sibling of description accumulation).

``_merge_edges_then_upsert`` finalizes an edge's weight by summing the newly
extracted relations' weights with the stored scalar. Before the fix it summed
*every* new relation, so re-feeding an already-stored source (any reprocess or
resume) re-added its weight, growing 1 -> 2 -> 3 per reprocess. The fix only
sums weights of sources not already reflected in the stored scalar, filtering on
the edge's own ``already_source_ids`` (consistent with ``already_weights``), and
ignores falsy source ids.

These drive the real merge round-trip (unit ``_merge_edges_then_upsert`` and the
top-level ``merge_nodes_and_edges`` orchestrator) against in-memory stores, no DB
or LLM. Split out from #3373 as a focused, independent fix.
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


async def _reprocess_edge_weights(sources: list[str]) -> list[float]:
    """Merge one edge per source in order; return the persisted weight after
    each merge. Endpoint nodes are pre-created; a no-op vdb sidesteps the
    unrelated entity_vdb path so this test isolates the weight logic."""
    g = await _edge_graph_with_nodes()
    cfg = _cfg()
    weights: list[float] = []
    for src in sources:
        await _merge_edges_then_upsert(
            "A", "B", [_rel(src)], g, _MemVdb(), _MemVdb(), cfg
        )
        weights.append((await g.get_edge("A", "B"))["weight"])
    return weights


# --- unit tests of the edge weight logic -----------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_edge_weight_not_accumulated_on_reprocess():
    """#3367 sibling: re-merging the same edge from the SAME source keeps weight
    fixed. Each source contributes 1.0 and an already-reflected source is not
    re-summed, so reprocess/resume does not inflate weight."""
    assert await _reprocess_edge_weights(["c1", "c1", "c1"]) == [1.0, 1.0, 1.0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_edge_weight_grows_across_distinct_sources():
    """Over-suppress guard: distinct sources (legitimate multi-document
    evidence) must still accumulate weight."""
    assert await _reprocess_edge_weights(["c1", "c2", "c3"]) == [1.0, 2.0, 3.0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_edge_weight_adds_only_the_new_source():
    """A reprocessed source followed by a genuinely new one adds only the new
    one's weight, not the re-fed duplicate's."""
    assert await _reprocess_edge_weights(["c1", "c1", "c2"]) == [1.0, 1.0, 2.0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_edge_weight_recovers_to_one_when_edge_missing_but_chunks_stored():
    """Partial-write recovery: relation_chunks_storage already has the source but
    the graph edge does not exist yet (a crash between the two writes). The weight
    filter must NOT treat those sources as already-weighted -- already_weights is
    empty, so filtering them would recover the edge with weight 0. With no stored
    scalar, sum all contributions (weight = 1, not 0)."""
    g = await _edge_graph_with_nodes()
    cfg = _cfg()
    rcs = _MemKV({make_relation_chunk_key("A", "B"): {"chunk_ids": ["c1"]}})
    await _merge_edges_then_upsert(
        "A",
        "B",
        [_rel("c1")],
        g,
        _MemVdb(),
        _MemVdb(),
        cfg,
        relation_chunks_storage=rcs,
    )
    edge = await g.get_edge("A", "B")
    assert edge is not None
    assert edge["weight"] == 1.0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_edge_weight_adds_new_source_when_chunks_store_is_ahead():
    """relation_chunks_storage can run AHEAD of the graph edge (chunks upserted,
    edge not yet updated). A genuinely new source re-fed then must still add
    weight -- the filter uses the EDGE's own source_ids (consistent with the
    stored scalar), not the chunk store, so the new source is not wrongly skipped
    (which would under-count the weight)."""
    g = await _edge_graph_with_nodes()
    cfg = _cfg()
    # Existing edge reflects only c1 (weight 1)...
    g.edges[("A", "B")] = _rel("c1")
    # ...but relation_chunks already lists c1 AND c2 (ahead of the edge).
    rcs = _MemKV({make_relation_chunk_key("A", "B"): {"chunk_ids": ["c1", "c2"]}})
    await _merge_edges_then_upsert(
        "A",
        "B",
        [_rel("c2")],
        g,
        _MemVdb(),
        _MemVdb(),
        cfg,
        relation_chunks_storage=rcs,
    )
    assert (await g.get_edge("A", "B"))["weight"] == 2.0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_edge_weight_not_double_counted_on_reversed_refeed():
    """Undirected edge: stored as (A,B) from c1, then re-fed reversed as (B,A)
    from the SAME c1. get_edge is symmetric, so the edge's own source_ids see c1
    and the re-fed duplicate is dropped -- weight stays 1, no direction-flip
    double count."""
    g = await _edge_graph_with_nodes()
    g.edges[("A", "B")] = _rel("c1")
    await _merge_edges_then_upsert(
        "B", "A", [_rel("c1")], g, _MemVdb(), _MemVdb(), _cfg()
    )
    assert (await g.get_edge("A", "B"))["weight"] == 1.0


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
# The tests above drive the unit _merge_edges_then_upsert helper. These drive the
# top-level two-phase merge the pipeline actually calls (Phase 1 entities ->
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


def _one_chunk(src: str, alice_desc: str = "Alice is an engineer."):
    """A single extracted chunk: entities ALICE + ACME and edge ALICE~ACME,
    all attributed to source `src`."""
    maybe_nodes = {
        "ALICE": [_node_dp("ALICE", alice_desc, src)],
        "ACME": [_node_dp("ACME", "Acme is a company.", src)],
    }
    maybe_edges = {
        ("ACME", "ALICE"): [
            {
                "src_id": "ALICE",
                "tgt_id": "ACME",
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


def _node_frag_count(g, name: str) -> int:
    return len([d for d in g.nodes[name]["description"].split(GRAPH_FIELD_SEP) if d])


@pytest.mark.offline
@pytest.mark.asyncio
async def test_orchestrator_reprocess_same_doc_does_not_accumulate():
    """Merging the SAME extracted doc twice (a reprocess/resume) must not grow
    entity descriptions or edge weight through the real orchestrator."""
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data()
    g = _MemGraph()
    stores = _stores()
    cfg = _cfg()
    for _ in range(2):
        await _orchestrate(_one_chunk("c1"), g, stores, cfg, doc_id="d1")
    assert _node_frag_count(g, "ALICE") == 1
    assert (await g.get_edge("ALICE", "ACME"))["weight"] == 1.0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_orchestrator_distinct_docs_accumulate():
    """Two DISTINCT docs (distinct sources + descriptions) legitimately grow the
    entity description to two fragments and the edge weight to two."""
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data()
    g = _MemGraph()
    stores = _stores()
    cfg = _cfg()
    await _orchestrate(
        _one_chunk("c1", "Alice is an engineer."), g, stores, cfg, doc_id="d1"
    )
    await _orchestrate(
        _one_chunk("c2", "Alice leads the team."), g, stores, cfg, doc_id="d2"
    )
    assert _node_frag_count(g, "ALICE") == 2
    assert (await g.get_edge("ALICE", "ACME"))["weight"] == 2.0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_orchestrator_reprocess_after_distinct_docs_is_stable():
    """After two distinct docs (c1, c2), reprocessing c1 again must not grow the
    description or weight past two."""
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data()
    g = _MemGraph()
    stores = _stores()
    cfg = _cfg()
    await _orchestrate(
        _one_chunk("c1", "Alice is an engineer."), g, stores, cfg, doc_id="d1"
    )
    await _orchestrate(
        _one_chunk("c2", "Alice leads the team."), g, stores, cfg, doc_id="d2"
    )
    await _orchestrate(
        _one_chunk("c1", "Alice is an engineer."), g, stores, cfg, doc_id="d1"
    )
    assert _node_frag_count(g, "ALICE") == 2
    assert (await g.get_edge("ALICE", "ACME"))["weight"] == 2.0
