"""Regression test: QdrantVectorDBStorage.query() must not block the event loop.

query() calls the synchronous QdrantClient.query_points() directly inside an
async method. A synchronous network round trip run that way occupies the
single-threaded asyncio event loop for its whole duration, so every other
coroutine in the process (other users' concurrent queries, pipeline
background work, health checks) stalls until Qdrant responds. Under load or
against a slow Qdrant instance this makes the whole server appear to hang.

Uses mocks only -- no running Qdrant instance required.
"""

import asyncio
import time

import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

pytest.importorskip(
    "qdrant_client",
    reason="qdrant-client is required for Qdrant storage tests",
)

from lightrag.kg.qdrant_impl import QdrantVectorDBStorage  # noqa: E402

pytestmark = pytest.mark.offline

# How long the fake Qdrant call blocks synchronously.
BLOCKING_SECONDS = 0.3


class MockEmbeddingFunc:
    def __init__(self, dim=8):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts, **kwargs):
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


@pytest.fixture(autouse=True)
def patch_namespace_lock():
    cache: dict[tuple[str, str | None], asyncio.Lock] = {}

    def factory(namespace, workspace=None, enable_logging=False):
        key = (namespace, workspace or "")
        lock = cache.get(key)
        if lock is None:
            lock = asyncio.Lock()
            cache[key] = lock
        return lock

    with patch("lightrag.kg.qdrant_impl.get_namespace_lock", side_effect=factory):
        yield cache


def _make_storage(embed_func, *, namespace="entities", workspace="test_ws"):
    storage = QdrantVectorDBStorage.__new__(QdrantVectorDBStorage)
    storage.workspace = workspace
    storage.namespace = namespace
    storage.effective_workspace = workspace
    storage.model_suffix = "mock"
    storage.final_namespace = f"lightrag_vdb_{namespace}_mock"
    storage.meta_fields = {"content"}
    storage.embedding_func = embed_func
    storage.cosine_better_than_threshold = 0.2
    storage._max_batch_size = 10
    storage._max_upsert_payload_bytes = 16 * 1024 * 1024
    storage._max_upsert_points_per_batch = 128
    storage._max_delete_points_per_batch = 1000
    storage._pending_vector_docs = {}
    storage._pending_vector_deletes = set()

    from lightrag.kg.qdrant_impl import get_namespace_lock

    storage._flush_lock = get_namespace_lock(
        namespace=storage.final_namespace, workspace=storage.effective_workspace
    )
    return storage


def _blocking_query_points(**_kwargs):
    """Stand-in for a slow synchronous Qdrant network round trip."""
    time.sleep(BLOCKING_SECONDS)
    return SimpleNamespace(points=[])


@pytest.mark.asyncio
async def test_query_does_not_block_event_loop():
    storage = _make_storage(MockEmbeddingFunc())
    storage._client = MagicMock()
    storage._client.query_points = MagicMock(side_effect=_blocking_query_points)

    start = asyncio.get_event_loop().time()
    # Run the vector query concurrently with a plain asyncio.sleep of the
    # same duration. If query() blocks the event loop for BLOCKING_SECONDS,
    # the sleep cannot make progress meanwhile and the two waits serialize
    # into roughly 2 * BLOCKING_SECONDS. Off the event loop, they overlap
    # and the total stays close to BLOCKING_SECONDS.
    await asyncio.gather(
        storage.query("hello", top_k=5, query_embedding=[0.1] * 8),
        asyncio.sleep(BLOCKING_SECONDS),
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < BLOCKING_SECONDS * 1.5, (
        f"query() appears to block the event loop: {elapsed:.3f}s elapsed for "
        f"two concurrent {BLOCKING_SECONDS}s waits (expected ~{BLOCKING_SECONDS}s)"
    )
