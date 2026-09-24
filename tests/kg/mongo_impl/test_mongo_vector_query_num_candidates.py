"""Regression test: MongoVectorDBStorage.query's numCandidates must scale
with top_k, not stay fixed below it.

All tests use mocks -- no running MongoDB instance required. Follows the
mocking conventions of tests/kg/mongo_impl/test_mongo_deferred_embedding.py.
"""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

pytest.importorskip(
    "pymongo",
    reason="pymongo is required for Mongo storage tests",
)

from lightrag.kg.mongo_impl import (
    MongoVectorDBStorage,
    _VECTOR_SEARCH_MAX_NUM_CANDIDATES,
    _VECTOR_SEARCH_MIN_NUM_CANDIDATES,
    _VECTOR_SEARCH_NUM_CANDIDATES_MULTIPLIER,
)

pytestmark = pytest.mark.offline


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    async def to_list(self, length=None):
        return list(self._docs)


def _make_storage(*, cosine_threshold=0.2):
    async def embed(texts, **kwargs):
        return np.random.rand(len(texts), 8).astype(np.float32)

    storage = MongoVectorDBStorage(
        namespace="entities",
        workspace="test",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": cosine_threshold
            },
        },
        embedding_func=embed,
        meta_fields={"content"},
    )
    storage._data = MagicMock()
    storage._data.aggregate = AsyncMock(return_value=_AsyncCursor([]))
    return storage


def _num_candidates_from_call(storage) -> int:
    pipeline = storage._data.aggregate.call_args[0][0]
    return pipeline[0]["$vectorSearch"]["numCandidates"]


@pytest.mark.asyncio
async def test_num_candidates_scales_above_top_k_when_top_k_is_large():
    """A top_k above the old fixed 100 must not be handed a smaller
    numCandidates -- Atlas Vector Search cannot return more documents than
    it was told to scan, so limit > numCandidates silently caps the result
    set (or the query is rejected outright)."""
    storage = _make_storage()

    await storage.query("q", top_k=500, query_embedding=[0.1] * 8)

    num_candidates = _num_candidates_from_call(storage)
    assert num_candidates >= 500
    assert num_candidates == min(
        500 * _VECTOR_SEARCH_NUM_CANDIDATES_MULTIPLIER,
        _VECTOR_SEARCH_MAX_NUM_CANDIDATES,
    )


@pytest.mark.asyncio
async def test_num_candidates_keeps_previous_floor_for_small_top_k():
    """A small top_k (the common case) keeps the previous fixed candidate
    pool rather than shrinking it, preserving prior recall behaviour."""
    storage = _make_storage()

    await storage.query("q", top_k=5, query_embedding=[0.1] * 8)

    assert _num_candidates_from_call(storage) == _VECTOR_SEARCH_MIN_NUM_CANDIDATES


@pytest.mark.asyncio
async def test_num_candidates_is_capped_at_atlas_ceiling():
    """A very large top_k (MAX_QUERY_TOP_K allows up to 1000) must not push
    numCandidates past Atlas's own 10,000 hard limit."""
    storage = _make_storage()

    await storage.query("q", top_k=1000, query_embedding=[0.1] * 8)

    assert _num_candidates_from_call(storage) == _VECTOR_SEARCH_MAX_NUM_CANDIDATES
