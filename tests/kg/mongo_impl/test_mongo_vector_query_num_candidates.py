"""Regression test: MongoVectorDBStorage.query must keep $vectorSearch's
numCandidates >= limit and both within Atlas's [1..10000] bound.

Atlas rejects the whole query otherwise ("limit should be less than or equal
to numCandidates"), so a fixed numCandidates of 100 failed every query whose
top_k exceeded 100.

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

from lightrag.kg.mongo_impl import MongoVectorDBStorage

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


def _vector_search_stage(storage) -> dict:
    pipeline = storage._data.aggregate.call_args[0][0]
    return pipeline[0]["$vectorSearch"]


@pytest.mark.parametrize(
    ("top_k", "expected_num_candidates", "expected_limit"),
    [
        # top_k at or below the previous fixed pool keeps it unchanged,
        # including the default top_k (40) and chunk_top_k (20).
        pytest.param(5, 100, 5, id="small"),
        pytest.param(100, 100, 100, id="at-floor"),
        # Above it, the pool widens to top_k so limit never exceeds it.
        pytest.param(101, 101, 101, id="just-above-floor"),
        pytest.param(10_000, 10_000, 10_000, id="atlas-ceiling"),
        # Core-API callers may exceed Atlas's ceiling; both values clamp.
        pytest.param(20_000, 10_000, 10_000, id="above-atlas-ceiling"),
    ],
)
@pytest.mark.asyncio
async def test_vector_search_bounds(top_k, expected_num_candidates, expected_limit):
    storage = _make_storage()

    await storage.query("q", top_k=top_k, query_embedding=[0.1] * 8)

    stage = _vector_search_stage(storage)
    assert (stage["numCandidates"], stage["limit"]) == (
        expected_num_candidates,
        expected_limit,
    )
    assert stage["limit"] <= stage["numCandidates"] <= 10_000
