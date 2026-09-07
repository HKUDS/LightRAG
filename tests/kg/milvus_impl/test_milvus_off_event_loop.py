"""MilvusVectorDBStorage.query() and _flush_pending_vector_ops() call the
synchronous MilvusClient SDK (blocking gRPC) directly inside async methods.
Calling it without offloading would block the whole event loop for the
duration of every search/upsert/delete round trip, stalling every other
concurrent task (LLM calls, other storage I/O) sharing the loop.

All tests use mocks -- no running Milvus instance required. Mirrors the
fixture setup in test_milvus_deferred_embedding.py.
"""

import asyncio
import threading

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


def _make_storage(embed_func, *, namespace="entities", workspace="test"):
    storage = MilvusVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=embed_func,
        meta_fields={"content"},
    )
    storage._client = MagicMock()
    storage._client.has_collection.return_value = True
    storage._client.load_collection = MagicMock()
    storage._initialized = True
    return storage


@pytest.mark.asyncio
async def test_query_runs_search_off_the_event_loop_thread():
    main_thread_id = threading.get_ident()
    call_thread_id = {}

    def fake_search(**kwargs):
        call_thread_id["id"] = threading.get_ident()
        return [[]]

    s = _make_storage(MockEmbeddingFunc())
    s._client.search = MagicMock(side_effect=fake_search)

    result = await s.query("hello", top_k=5, query_embedding=[0.1] * 8)

    assert result == []
    assert call_thread_id["id"] != main_thread_id


@pytest.mark.asyncio
async def test_flush_runs_upsert_off_the_event_loop_thread():
    main_thread_id = threading.get_ident()
    call_thread_id = {}

    def fake_upsert(**kwargs):
        call_thread_id["id"] = threading.get_ident()
        return {"upsert_count": 1}

    s = _make_storage(MockEmbeddingFunc())
    s._client.upsert = MagicMock(side_effect=fake_upsert)
    s._client.delete = MagicMock(return_value={"delete_count": 0})

    await s.upsert({"v1": {"content": "hello"}})
    await s.index_done_callback()

    assert call_thread_id["id"] != main_thread_id


@pytest.mark.asyncio
async def test_cancelling_flush_defers_until_write_completes_then_clears_buffer():
    """asyncio.to_thread only cancels the awaiting future -- an in-flight
    Milvus upsert keeps running in the background thread. A bare cancel
    here would release _flush_lock and return to the caller while that
    write (and its buffer bookkeeping) is still pending, letting a
    concurrent flush interleave with the orphaned write. Cancellation must
    instead be deferred until the write, and the buffer pop that follows
    it, have actually finished."""
    call_started = threading.Event()
    release_call = threading.Event()

    def fake_upsert(**kwargs):
        call_started.set()
        release_call.wait(timeout=5)
        return {"upsert_count": 1}

    s = _make_storage(MockEmbeddingFunc())
    s._client.upsert = MagicMock(side_effect=fake_upsert)
    s._client.delete = MagicMock(return_value={"delete_count": 0})

    await s.upsert({"v1": {"content": "hello"}})

    task = asyncio.ensure_future(s.index_done_callback())
    for _ in range(500):
        if call_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert call_started.is_set()

    task.cancel()
    # Let the background write finish so the deferred cancellation can
    # resolve -- release_call must be set before awaiting the cancelled
    # task, since the cancellation is held back until the write completes.
    release_call.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    s._client.upsert.assert_called_once()
    # The write actually landed, so the buffer must reflect that outcome
    # -- not the stale "still pending" state a bare cancel would leave.
    assert s._pending_vector_docs == {}
    assert not s._flush_lock.locked()


@pytest.mark.asyncio
async def test_flush_runs_delete_off_the_event_loop_thread():
    main_thread_id = threading.get_ident()
    call_thread_id = {}

    def fake_delete(**kwargs):
        call_thread_id["id"] = threading.get_ident()
        return {"delete_count": 1}

    s = _make_storage(MockEmbeddingFunc())
    s._client.upsert = MagicMock(return_value={"upsert_count": 0})
    s._client.delete = MagicMock(side_effect=fake_delete)

    await s.delete(["v1"])
    await s.index_done_callback()

    assert call_thread_id["id"] != main_thread_id
