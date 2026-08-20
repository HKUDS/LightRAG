"""Regression tests: the two remaining relation-weight-contract gaps left by the
evidence-floor change (#3676 follow-up).

1. ``_rebuild_single_relationship`` (document purge/resume and
   ``lightrag-rebuild-vdb``) wrote ``sum(cached fragment weights)`` alongside the
   full surviving chunk list with no floor. Cached fragments only exist for those
   surviving chunks whose extraction cache is still present -- none at all on the
   degraded/structural-fallback path -- so the rebuilt weight could land below
   the edge's own evidence count, minting exactly the rows ``aedit_relation``
   then refuses to edit.

2. ``_merge_edges_then_upsert`` seeded its relation chunk tracking baseline from
   a legacy edge's ``source_id`` (and from a stored tracking row) filtering only
   falsy entries, so the historical no-evidence placeholders
   (``manual_creation`` / ``UNKNOWN``) were written into ``relation_chunks`` --
   the authoritative chunk list -- where purge and audit read them back as real
   surviving chunks.

Both run against in-memory duck-typed stores: no DB, no LLM.
"""

from __future__ import annotations

import pytest

from lightrag.constants import GRAPH_FIELD_SEP, RELATION_NO_EVIDENCE_SOURCE_IDS
from lightrag.operate import _merge_edges_then_upsert, _rebuild_single_relationship
from lightrag.utils import Tokenizer, TokenizerInterface, make_relation_chunk_key


class _DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]):
        return "".join(chr(t) for t in tokens)


class _MemGraph:
    """Minimal in-memory graph covering the calls both functions make."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple[str, str], dict] = {}

    async def get_node(self, node_id):
        return self.nodes.get(node_id)

    async def has_node(self, node_id) -> bool:
        return node_id in self.nodes

    async def upsert_node(self, node_id, node_data) -> None:
        self.nodes[node_id] = dict(node_data)

    async def has_edge(self, src, tgt) -> bool:
        return (src, tgt) in self.edges or (tgt, src) in self.edges

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt)) or self.edges.get((tgt, src))

    async def upsert_edge(self, src, tgt, edge_data) -> None:
        self.edges[(src, tgt)] = dict(edge_data)


class _MemVdb:
    async def upsert(self, data) -> None:
        pass

    async def delete(self, ids) -> None:
        pass


class _MemKV:
    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(k) for k in keys]

    async def upsert(self, data) -> None:
        self.data.update(data)

    async def index_done_callback(self) -> None:
        pass


def _cfg(max_source_ids_per_relation: int = 10) -> dict:
    return {
        "tokenizer": Tokenizer("dummy", _DummyTokenizer()),
        "summary_context_size": 1_000_000,
        "summary_max_tokens": 1_000_000,
        "force_llm_summary_on_merge": 6,
        "source_ids_limit_method": "KEEP",
        "max_source_ids_per_entity": 10,
        "max_source_ids_per_relation": max_source_ids_per_relation,
        "max_file_paths": 10,
        "file_path_more_placeholder": "...",
    }


async def _graph_with_edge(weight: float, source_id: str) -> _MemGraph:
    graph = _MemGraph()
    for name in ("A", "B"):
        await graph.upsert_node(
            name,
            {
                "entity_id": name,
                "entity_type": "ENT",
                "description": f"Node {name}",
                "source_id": source_id,
                "file_path": "f1.txt",
            },
        )
    await graph.upsert_edge(
        "A",
        "B",
        {
            "description": "Relation AB",
            "keywords": "legacy_kw",
            "weight": weight,
            "source_id": source_id,
            "file_path": "f1.txt",
        },
    )
    return graph


async def _rebuild(
    graph: _MemGraph,
    chunk_ids: list[str],
    chunk_relationships: dict,
    *,
    global_config: dict | None = None,
) -> dict:
    """Drive one rebuild and return the persisted edge."""
    await _rebuild_single_relationship(
        knowledge_graph_inst=graph,
        relationships_vdb=_MemVdb(),
        entities_vdb=_MemVdb(),
        src="A",
        tgt="B",
        chunk_ids=chunk_ids,
        chunk_relationships=chunk_relationships,
        llm_response_cache=_MemKV(),
        global_config=global_config or _cfg(),
        structural_fallback=True,
    )
    return await graph.get_edge("A", "B")


def _fragment(weight: float = 1.0) -> dict:
    return {
        "weight": weight,
        "description": "rel desc",
        "keywords": "kw",
        "file_path": "f1.txt",
    }


# --- 1. rebuild must not mint under-floor rows ------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_lifts_legacy_weight_when_no_fragment_cache_survives():
    """Degraded rebuild: the extraction cache is gone, so the only weight input is
    the legacy stored scalar (1.0) while three chunks survive as the edge's
    evidence. The rebuilt row must be lifted to the evidence floor."""
    graph = await _graph_with_edge(weight=1.0, source_id="c1")

    edge = await _rebuild(graph, ["c1", "c2", "c3"], {})

    assert edge["source_id"] == GRAPH_FIELD_SEP.join(["c1", "c2", "c3"])
    assert edge["weight"] == 3.0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_lifts_weight_when_fragment_cache_is_partial():
    """Partial cache: only one of three surviving chunks still has a cached
    fragment, so the summed fragment weight (1.0) under-counts the evidence the
    rebuilt ``source_id`` advertises."""
    graph = await _graph_with_edge(weight=1.0, source_id="c1")

    edge = await _rebuild(
        graph,
        ["c1", "c2", "c3"],
        {"c1": {("A", "B"): [_fragment(1.0)]}},
    )

    assert edge["source_id"] == GRAPH_FIELD_SEP.join(["c1", "c2", "c3"])
    assert edge["weight"] == 3.0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_preserves_a_larger_explicit_boost():
    """Over-repair guard: the floor lifts an undersized weight, it never lowers a
    legitimate importance boost."""
    graph = await _graph_with_edge(weight=10.0, source_id="c1")

    edge = await _rebuild(graph, ["c1", "c2"], {})

    assert edge["weight"] == 10.0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_floors_on_the_post_limit_chunk_list():
    """The floor counts the chunks actually stored on the edge, not the full
    surviving list: ``aedit_relation`` validates against the stored ``source_id``,
    so a truncated row must not carry a weight above its visible evidence."""
    graph = await _graph_with_edge(weight=1.0, source_id="c1")

    edge = await _rebuild(
        graph,
        ["c1", "c2", "c3"],
        {},
        global_config=_cfg(max_source_ids_per_relation=2),
    )

    assert edge["source_id"] == GRAPH_FIELD_SEP.join(["c1", "c2"])
    assert edge["weight"] == 2.0


# --- 2. no placeholder may enter relation chunk tracking -------------------


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("placeholder", sorted(RELATION_NO_EVIDENCE_SOURCE_IDS))
async def test_merge_keeps_legacy_placeholder_out_of_chunk_tracking(placeholder):
    """A manually created legacy edge carries a no-evidence placeholder in its
    ``source_id``. When extraction later merges a real chunk into it, neither the
    authoritative tracking row nor the edge may keep the placeholder."""
    graph = await _graph_with_edge(weight=1.0, source_id=placeholder)
    storage_key = make_relation_chunk_key("A", "B")
    tracking = _MemKV()

    await _merge_edges_then_upsert(
        "A",
        "B",
        [{**_fragment(1.0), "source_id": "c1"}],
        graph,
        _MemVdb(),
        _MemVdb(),
        _cfg(),
        relation_chunks_storage=tracking,
    )

    assert tracking.data[storage_key]["chunk_ids"] == ["c1"]
    assert tracking.data[storage_key]["count"] == 1
    edge = await graph.get_edge("A", "B")
    assert edge["source_id"] == "c1"
    # Unchanged by this fix: the stored manual scalar stays an importance boost
    # on top of the newly contributed evidence.
    assert edge["weight"] == 2.0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_merge_repairs_a_tracking_row_that_already_holds_a_placeholder():
    """Tracking rows written before the exclusion existed are repaired on the next
    merge, and every real chunk id survives that repair in order."""
    graph = await _graph_with_edge(
        weight=2.0, source_id=GRAPH_FIELD_SEP.join(["manual_creation", "c1"])
    )
    storage_key = make_relation_chunk_key("A", "B")
    tracking = _MemKV({storage_key: {"chunk_ids": ["manual_creation", "c1"]}})

    await _merge_edges_then_upsert(
        "A",
        "B",
        [{**_fragment(1.0), "source_id": "c2"}],
        graph,
        _MemVdb(),
        _MemVdb(),
        _cfg(),
        relation_chunks_storage=tracking,
    )

    assert tracking.data[storage_key]["chunk_ids"] == ["c1", "c2"]
    edge = await graph.get_edge("A", "B")
    assert edge["source_id"] == GRAPH_FIELD_SEP.join(["c1", "c2"])
    assert edge["weight"] >= 2.0
