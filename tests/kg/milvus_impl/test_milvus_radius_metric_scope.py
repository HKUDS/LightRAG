"""cosine_better_than_threshold is documented as a raw cosine-similarity
value everywhere in LightRAG. MilvusVectorDBStorage.query() passed it
straight through as Milvus's "radius" search param regardless of the
collection's configured metric_type. Milvus's radius semantics are
metric-dependent: for L2 it is an upper distance bound (smaller distance is
closer), so a cosine-scale value like 0.2 becomes an almost-unreachable
tight cap and silently starves retrieval. Radius must only be derived from
cosine_better_than_threshold when the collection's metric actually is
COSINE.

All tests use mocks -- no running Milvus instance required. Mirrors the
fixture setup in test_milvus_off_event_loop.py.
"""

import asyncio
import logging

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from lightrag.kg.milvus_impl import MilvusVectorDBStorage

pytestmark = pytest.mark.offline


class MockEmbeddingFunc:
    def __init__(self, dim=8):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts, **kwargs):
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


@pytest.fixture(autouse=True)
def patch_namespace_lock():
    cache: dict = {}

    def factory(namespace, workspace=None, enable_logging=False):
        key = (namespace, workspace or "")
        lock = cache.get(key)
        if lock is None:
            lock = asyncio.Lock()
            cache[key] = lock
        return lock

    with patch("lightrag.kg.milvus_impl.get_namespace_lock", side_effect=factory):
        yield cache


def _make_storage(*, metric_type: str, threshold: float = 0.2):
    storage = MilvusVectorDBStorage(
        namespace="entities",
        workspace="test",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": threshold,
                "metric_type": metric_type,
            },
        },
        embedding_func=MockEmbeddingFunc(),
        meta_fields={"content"},
    )
    storage._client = MagicMock()
    storage._client.has_collection.return_value = True
    storage._client.load_collection = MagicMock()
    storage._initialized = True
    return storage


@pytest.mark.asyncio
async def test_cosine_metric_applies_radius():
    s = _make_storage(metric_type="COSINE", threshold=0.2)
    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return [[]]

    s._client.search = MagicMock(side_effect=fake_search)

    await s.query("hello", top_k=5, query_embedding=[0.1] * 8)

    assert captured["search_params"]["params"]["radius"] == 0.2


@pytest.mark.asyncio
async def test_l2_metric_does_not_apply_cosine_scale_radius():
    s = _make_storage(metric_type="L2", threshold=0.2)
    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return [[]]

    s._client.search = MagicMock(side_effect=fake_search)

    await s.query("hello", top_k=5, query_embedding=[0.1] * 8)

    assert "radius" not in captured["search_params"]["params"]


@pytest.mark.asyncio
async def test_ip_metric_does_not_apply_cosine_scale_radius():
    s = _make_storage(metric_type="IP", threshold=0.2)
    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return [[]]

    s._client.search = MagicMock(side_effect=fake_search)

    await s.query("hello", top_k=5, query_embedding=[0.1] * 8)

    assert "radius" not in captured["search_params"]["params"]


def test_non_cosine_metric_warns_that_threshold_is_ignored(
    caplog: pytest.LogCaptureFixture,
):
    """cosine_better_than_threshold is required at construction regardless of
    metric_type, so an operator on L2/IP who set it gets no other signal that
    it is never applied at query time."""
    lightrag_logger = logging.getLogger("lightrag")
    previous_propagate = lightrag_logger.propagate
    lightrag_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="lightrag"):
            _make_storage(metric_type="L2", threshold=0.2)
    finally:
        lightrag_logger.propagate = previous_propagate

    assert "not COSINE" in caplog.text


def test_cosine_metric_does_not_warn(caplog: pytest.LogCaptureFixture):
    lightrag_logger = logging.getLogger("lightrag")
    previous_propagate = lightrag_logger.propagate
    lightrag_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="lightrag"):
            _make_storage(metric_type="COSINE", threshold=0.2)
    finally:
        lightrag_logger.propagate = previous_propagate

    assert "not COSINE" not in caplog.text
