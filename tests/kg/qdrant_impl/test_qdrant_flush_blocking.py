"""Regression test: flushing pending vector upserts must not block the event loop.

``_flush_pending_vector_ops`` is the write path every document-ingestion run
goes through: ``index_done_callback`` (called after every pipeline batch)
flushes the buffered upserts by calling the synchronous
``QdrantClient.upsert()`` directly inside this async method. A synchronous
network round trip run that way occupies the single-threaded asyncio event
loop for its whole duration, so every other coroutine in the process (other
workspaces' pipelines, concurrent queries, health checks, the API server's
request handling) stalls until Qdrant responds. Under load, or against a
slow Qdrant instance, routine document ingestion makes the whole server
appear to hang -- the same failure mode already fixed for ``query()``, just
on the write path instead of the read path.

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

from lightrag.kg.qdrant_impl import (  # noqa: E402
    QdrantVectorDBStorage,
    _PendingVectorDoc,
)

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


def _blocking_upsert(**_kwargs):
    """Stand-in for a slow synchronous Qdrant network round trip."""
    time.sleep(BLOCKING_SECONDS)
    return SimpleNamespace()


@pytest.mark.asyncio
async def test_flush_upsert_does_not_block_event_loop():
    storage = _make_storage(MockEmbeddingFunc())
    storage._client = MagicMock()
    storage._client.upsert = MagicMock(side_effect=_blocking_upsert)

    # Pre-embedded pending doc, as a flush would find mid-pipeline: the
    # embedding step is already covered elsewhere, so this isolates the
    # actual server write.
    storage._pending_vector_docs["doc-1"] = _PendingVectorDoc(
        source={"id": "doc-1", "workspace_id": "test_ws", "created_at": 0},
        content="hello world",
        vector=[0.1] * 8,
    )

    start = asyncio.get_event_loop().time()
    # Run the flush concurrently with a plain asyncio.sleep of the same
    # duration. If the flush blocks the event loop for BLOCKING_SECONDS,
    # the sleep cannot make progress meanwhile and the two waits serialize
    # into roughly 2 * BLOCKING_SECONDS. Off the event loop, they overlap
    # and the total stays close to BLOCKING_SECONDS.
    await asyncio.gather(
        storage._flush_pending_vector_ops(),
        asyncio.sleep(BLOCKING_SECONDS),
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < BLOCKING_SECONDS * 1.5, (
        f"flushing pending vector upserts appears to block the event loop: "
        f"{elapsed:.3f}s elapsed for two concurrent {BLOCKING_SECONDS}s waits "
        f"(expected ~{BLOCKING_SECONDS}s)"
    )
