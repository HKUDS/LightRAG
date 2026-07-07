"""Regression tests for #3367.

``_handle_entity_relation_summary`` — the single choke point both
``_merge_nodes_then_upsert`` and ``_merge_edges_then_upsert`` route their
combined ``already_stored + newly_extracted`` description list through — used to
join the list without cross-deduping stored vs new. So re-merging a doc whose
entities/relations already existed (any reprocess/resume) re-appended the same
description, growing the fragment count by one per reprocess.

These tests pin: (a) a reprocess does not accumulate; (b) identical descriptions
collapse to one; and — guarding against over-dedup — (c) distinct descriptions
are kept in first-seen order.
"""

from __future__ import annotations

import pytest

import lightrag.operate as operate
from lightrag.operate import (
    _handle_entity_relation_summary,
    _merge_edges_then_upsert,
    _merge_nodes_then_upsert,
)
from lightrag.constants import GRAPH_FIELD_SEP


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

    async def upsert_node(self, name, node_data):
        self.nodes[name] = dict(node_data)

    async def has_edge(self, s, t):
        return (s, t) in self.edges or (t, s) in self.edges

    async def get_edge(self, s, t):
        return self.edges.get((s, t)) or self.edges.get((t, s))

    async def upsert_edge(self, s, t, edge_data):
        self.edges[(s, t)] = dict(edge_data)


class _MemVdb:
    """No-op vector store so the edge-merge entity-vdb path doesn't require a
    real backend (and sidesteps the unrelated entity_vdb=None crash)."""

    async def upsert(self, data):
        pass

    async def delete(self, ids):
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


_D = "Alice is a software engineer at Acme."
_D2 = "Alice leads the backend team."


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reprocess_does_not_accumulate_node_descriptions():
    """Re-merging the same single description N times keeps one fragment, not N."""
    g = _MemGraph()
    cfg = _cfg()
    batch = {
        "entity_name": "ALICE",
        "entity_type": "person",
        "description": _D,
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
        "timestamp": 1,
    }
    for _ in range(3):
        await _merge_nodes_then_upsert("ALICE", [dict(batch)], g, None, cfg)

    frags = g.nodes["ALICE"]["description"].split(GRAPH_FIELD_SEP)
    assert frags == [_D]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reprocess_does_not_accumulate_dirty_descriptions():
    """A description that sanitization changes (e.g. an XML-illegal control char
    is stripped) must also not accumulate. Stored descriptions are already
    sanitized while a re-extracted one is raw, so dedup must compare on the
    sanitized text; a raw comparison would miss the duplicate and keep growing.
    """
    g = _MemGraph()
    cfg = _cfg()
    dirty = "Alice\x08 is a software engineer at Acme."  # \x08 is stripped by sanitize
    batch = {
        "entity_name": "ALICE",
        "entity_type": "person",
        "description": dirty,
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
        "timestamp": 1,
    }
    for _ in range(3):
        await _merge_nodes_then_upsert("ALICE", [dict(batch)], g, None, cfg)

    frags = g.nodes["ALICE"]["description"].split(GRAPH_FIELD_SEP)
    assert len(frags) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_summary_collapses_identical_descriptions():
    """A list of identical descriptions collapses to a single fragment."""
    joined, llm_used = await _handle_entity_relation_summary(
        "Relation", "ALICE~ACME", [_D, _D], GRAPH_FIELD_SEP, _cfg(), None
    )
    assert joined.split(GRAPH_FIELD_SEP) == [_D]
    # dedup to one -> len==1 early return, so no LLM summary is invoked.
    assert llm_used is False


@pytest.mark.offline
@pytest.mark.asyncio
async def test_distinct_descriptions_are_kept_in_first_seen_order():
    """Over-dedup guard: distinct descriptions survive; only exact dups drop,
    order-preserving keep-first."""
    joined, _ = await _handle_entity_relation_summary(
        "Relation", "X", [_D, _D2, _D], GRAPH_FIELD_SEP, _cfg(), None
    )
    assert joined.split(GRAPH_FIELD_SEP) == [_D, _D2]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_keep_first_preserves_earliest_occurrence_order():
    """Keep-first: the first occurrence position wins, later dups are dropped."""
    joined, _ = await _handle_entity_relation_summary(
        "Relation", "X", [_D2, _D, _D2], GRAPH_FIELD_SEP, _cfg(), None
    )
    assert joined.split(GRAPH_FIELD_SEP) == [_D2, _D]


async def _reprocess_edge_weights(sources: list[str]) -> list[float]:
    """Merge one edge per source in order; return the persisted weight after
    each merge. Endpoint nodes are pre-created; a no-op vdb sidesteps the
    unrelated entity_vdb crash so this test isolates the weight logic."""
    g = _MemGraph()
    cfg = _cfg()
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

    def _edge(src: str) -> dict:
        return {
            "weight": 1.0,
            "source_id": src,
            "description": "rel",
            "keywords": "k",
            "file_path": "f",
        }

    weights: list[float] = []
    for src in sources:
        await _merge_edges_then_upsert(
            "A", "B", [_edge(src)], g, _MemVdb(), _MemVdb(), cfg
        )
        weights.append((await g.get_edge("A", "B"))["weight"])
    return weights


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
async def test_edge_merge_with_entity_vdb_none_does_not_crash():
    """entity_vdb=None must not raise UnboundLocalError when an edge merge adds a
    new endpoint entity. The vdb upsert call must stay inside its `entity_vdb is
    not None` guard (it references vdb_data, which is assigned only inside it)."""
    g = _MemGraph()
    cfg = _cfg()
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

    def _edge(src: str) -> dict:
        return {
            "weight": 1.0,
            "source_id": src,
            "description": "rel",
            "keywords": "k",
            "file_path": "f",
        }

    # entity_vdb=None + genuinely new sources is the path that used to crash.
    for src in ("c1", "c2", "c3"):
        await _merge_edges_then_upsert("A", "B", [_edge(src)], g, None, None, cfg)

    assert await g.get_edge("A", "B") is not None
