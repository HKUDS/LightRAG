"""Fusion order is consumed as-is downstream (plan D5 invariant, test A8).

Every fixture here is built so a distance-descending re-sort WOULD reorder
the records (lexical-only sentinel distances sit below real vector
similarities). The tests assert the real retrieval entry points
(``_get_vector_context``, ``_get_node_data``) and the real downstream
truncation (``process_chunks_unified``) keep the fusion order instead.
Dispatch is pinned at the same seams: flag off or unsupported backend never
reaches ``hybrid_query``; the registry-wide proof lives in
``tests/kg/test_hybrid_capability.py``.
"""

from __future__ import annotations

import pytest

from lightrag.base import QueryParam
from lightrag.operate import _get_node_data, _get_vector_context
from lightrag.utils import process_chunks_unified

pytestmark = pytest.mark.offline

# Fusion order: the lexical-only hit outranks the vector hit. A distance
# sort (descending similarity) would flip them: 0.9 > 0.2.
_FUSED_CHUNKS = [
    {"id": "c-lex", "content": "lexical hit", "file_path": "a.txt", "distance": 0.2},
    {"id": "c-vec", "content": "vector hit", "file_path": "b.txt", "distance": 0.9},
]
_FUSED_ENTITIES = [
    {"entity_name": "LexOnly", "created_at": 1, "distance": 0.2},
    {"entity_name": "VecHit", "created_at": 2, "distance": 0.9},
]


def _distance_sorted(records: list[dict], key: str) -> list[str]:
    return [r[key] for r in sorted(records, key=lambda r: -r["distance"])]


class _HybridVDB:
    """Hybrid-capable fake whose two legs are recorded separately."""

    supports_hybrid_query = True
    cosine_better_than_threshold = 0.2

    def __init__(self, fused: list[dict], key: str):
        self._fused = fused
        self._key = key
        self.query_calls: list[tuple] = []
        self.hybrid_calls: list[tuple] = []

    async def query(self, query, top_k, query_embedding=None):
        self.query_calls.append((query, top_k))
        return [dict(r) for r in reversed(self._fused)]

    async def hybrid_query(self, query, top_k, query_embedding=None):
        self.hybrid_calls.append((query, top_k))
        return [dict(r) for r in self._fused]


class _UnsupportedVDB(_HybridVDB):
    """Same shape, flag False: hybrid_query must never be reached."""

    supports_hybrid_query = False


class _FakeGraph:
    """Batch graph reads keyed by name; enough for _get_node_data's path."""

    async def get_nodes_batch(self, names):
        return {n: {"entity_name": n, "description": f"d-{n}"} for n in names}

    async def node_degrees_batch(self, names):
        return {n: 1 for n in names}

    async def get_nodes_edges_batch(self, names):
        return {n: [] for n in names}

    async def get_edges_batch(self, pairs):
        return {}

    async def edge_degrees_batch(self, pairs):
        return {}


async def test_vector_context_consumes_fusion_order_not_distance():
    vdb = _HybridVDB(_FUSED_CHUNKS, "id")
    assert _distance_sorted(_FUSED_CHUNKS, "id") == ["c-vec", "c-lex"]

    chunks = await _get_vector_context(
        "query", vdb, QueryParam(enable_hybrid=True), query_embedding=[0.1]
    )

    assert [c["chunk_id"] for c in chunks] == ["c-lex", "c-vec"]
    assert vdb.hybrid_calls == [("query", 20)]
    assert vdb.query_calls == []


async def test_unified_truncation_keeps_the_fusion_head():
    vdb = _HybridVDB(_FUSED_CHUNKS, "id")
    chunks = await _get_vector_context(
        "query", vdb, QueryParam(enable_hybrid=True), query_embedding=[0.1]
    )

    # Real downstream truncation: chunk_top_k keeps the FIRST record, which
    # must be the fusion head, not the distance-sort head.
    kept = await process_chunks_unified(
        query="query",
        unique_chunks=chunks,
        query_param=QueryParam(enable_hybrid=True, chunk_top_k=1),
        global_config={},
        source_type="vector",
    )

    assert [c["chunk_id"] for c in kept] == ["c-lex"]


async def test_vector_context_flag_off_stays_on_plain_query():
    vdb = _HybridVDB(_FUSED_CHUNKS, "id")

    chunks = await _get_vector_context(
        "query", vdb, QueryParam(enable_hybrid=False), query_embedding=[0.1]
    )

    assert [c["chunk_id"] for c in chunks] == ["c-vec", "c-lex"]
    assert vdb.query_calls == [("query", 20)]
    assert vdb.hybrid_calls == []


async def test_vector_context_unsupported_backend_never_reaches_hybrid():
    vdb = _UnsupportedVDB(_FUSED_CHUNKS, "id")

    chunks = await _get_vector_context(
        "query", vdb, QueryParam(enable_hybrid=True), query_embedding=[0.1]
    )

    assert [c["chunk_id"] for c in chunks] == ["c-vec", "c-lex"]
    assert vdb.query_calls == [("query", 20)]
    assert vdb.hybrid_calls == []


async def test_node_data_consumes_fusion_order_not_distance():
    vdb = _HybridVDB(_FUSED_ENTITIES, "entity_name")
    assert _distance_sorted(_FUSED_ENTITIES, "entity_name") == ["VecHit", "LexOnly"]

    node_datas, relations = await _get_node_data(
        "query",
        _FakeGraph(),
        vdb,
        QueryParam(enable_hybrid=True, top_k=5),
        query_embedding=[0.1],
    )

    assert [n["entity_name"] for n in node_datas] == ["LexOnly", "VecHit"]
    assert node_datas[0]["rank"] == 1
    assert relations == []
    assert vdb.hybrid_calls == [("query", 5)]
    assert vdb.query_calls == []


async def test_node_data_flag_off_stays_on_plain_query():
    vdb = _HybridVDB(_FUSED_ENTITIES, "entity_name")

    node_datas, _ = await _get_node_data(
        "query",
        _FakeGraph(),
        vdb,
        QueryParam(enable_hybrid=False, top_k=5),
        query_embedding=[0.1],
    )

    assert [n["entity_name"] for n in node_datas] == ["VecHit", "LexOnly"]
    assert vdb.query_calls == [("query", 5)]
    assert vdb.hybrid_calls == []
