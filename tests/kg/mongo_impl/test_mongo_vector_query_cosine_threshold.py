"""MongoVectorDBStorage.query() filters on Atlas Vector Search's
vectorSearchScore. For a cosine-similarity index, Atlas normalizes that
score to (1 + cosine_similarity) / 2, in [0, 1] -- not raw cosine
similarity, which every other backend (faiss, postgres, nano) compares
cosine_better_than_threshold against. Comparing the threshold straight
against the normalized score both rescales the effective cutoff (a
default 0.2 threshold only requires raw cosine similarity >= -0.6) and
reports the wrong value in the "distance" field.
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

pytest.importorskip("pymongo", reason="pymongo is required for Mongo storage tests")

from lightrag.kg.mongo_impl import MongoVectorDBStorage

pytestmark = pytest.mark.offline


class MockEmbeddingFunc:
    def __init__(self, dim=8):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts, **kwargs):
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    async def to_list(self, length=None):
        return list(self._docs)


def _make_storage(threshold=0.2):
    storage = MongoVectorDBStorage(
        namespace="entities",
        workspace="test",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": threshold},
        },
        embedding_func=MockEmbeddingFunc(),
        meta_fields={"content"},
    )
    storage._data = MagicMock()
    storage._index_name = "test_index"
    return storage


@pytest.mark.asyncio
async def test_match_stage_uses_the_rescaled_threshold():
    """The server-side $match must compare against (1 + threshold) / 2, the
    same scale Atlas reports vectorSearchScore on, not the raw threshold."""
    storage = _make_storage(threshold=0.2)
    storage._data.aggregate = AsyncMock(return_value=_AsyncCursor([]))

    await storage.query("test", top_k=5, query_embedding=[0.1] * 8)

    pipeline = storage._data.aggregate.await_args[0][0]
    match_stage = next(stage for stage in pipeline if "$match" in stage)
    assert match_stage["$match"]["score"]["$gte"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_distance_is_converted_back_to_raw_cosine_similarity():
    """A doc surviving the server-side filter carries Atlas's normalized
    score; the returned "distance" must be the raw cosine similarity."""
    storage = _make_storage(threshold=0.2)
    storage._data.aggregate = AsyncMock(
        return_value=_AsyncCursor([{"_id": "v1", "score": 0.85, "content": "match"}])
    )

    results = await storage.query("test", top_k=5, query_embedding=[0.1] * 8)

    assert len(results) == 1
    assert results[0]["distance"] == pytest.approx(0.7)
