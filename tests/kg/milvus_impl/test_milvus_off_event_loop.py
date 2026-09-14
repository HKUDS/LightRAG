"""MilvusVectorDBStorage.query(), _flush_pending_vector_ops(),
delete_entity_relation(), _query_rows_by_ids() (used by get_by_ids() /
get_vectors_by_ids()), and get_by_id() call the synchronous MilvusClient SDK
(blocking gRPC) directly inside async methods. Calling it without offloading
would block the whole event loop for the duration of every
search/upsert/delete/query round trip, stalling every other concurrent task
(LLM calls, other storage I/O) sharing the loop.

The negative-case tests pin the other half of the contract: a call that
never reaches the client (a buffer-only read/prune) must not touch the
executor at all.

All tests use mocks -- no running Milvus instance required. Mirrors the
fixture setup in test_milvus_deferred_embedding.py.
"""

import asyncio
import threading
import time

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from lightrag.constants import MILVUS_SUBMIT_LIMIT
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
    """run_in_milvus_executor only cancels the awaiting future -- an in-flight
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


@pytest.mark.asyncio
async def test_blocking_calls_use_the_dedicated_milvus_pool():
    """Off the loop is not enough: the calls must also stay off the process
    DEFAULT executor. That pool is shared with
    UnifiedLock._acquire_mp_lock_in_executor, login password hashing and the
    document routes' stat() calls, and its wait queue is unbounded -- parking
    multi-second Milvus round trips there makes namespace-lock acquisition and
    /login queue behind Milvus. Threads from the dedicated pool are named
    'lightrag-milvus*'; asyncio's default pool names them 'asyncio_*'."""
    thread_names: dict[str, str] = {}

    def record(op):
        def _fake(**kwargs):
            thread_names[op] = threading.current_thread().name
            return {"upsert_count": 1} if op == "upsert" else {"delete_count": 1}

        return _fake

    def fake_load(*args, **kwargs):
        thread_names["load_collection"] = threading.current_thread().name

    def fake_search(**kwargs):
        thread_names["search"] = threading.current_thread().name
        return [[]]

    s = _make_storage(MockEmbeddingFunc())
    s._client.load_collection = MagicMock(side_effect=fake_load)
    s._client.search = MagicMock(side_effect=fake_search)
    s._client.upsert = MagicMock(side_effect=record("upsert"))
    s._client.delete = MagicMock(side_effect=record("delete"))

    await s.query("hello", top_k=5, query_embedding=[0.1] * 8)
    await s.upsert({"v1": {"content": "hello"}})
    await s.delete(["v2"])
    await s.index_done_callback()

    assert set(thread_names) == {"load_collection", "search", "upsert", "delete"}
    offenders = {
        op: name
        for op, name in thread_names.items()
        if not name.startswith("lightrag-milvus")
    }
    assert offenders == {}


@pytest.mark.asyncio
async def test_delete_entity_relation_runs_query_and_delete_off_the_event_loop_thread():
    main_thread_id = threading.get_ident()
    call_thread_id = {}

    def fake_query(**kwargs):
        call_thread_id["query"] = threading.get_ident()
        return [{"id": "rel-1"}]

    def fake_delete(**kwargs):
        call_thread_id["delete"] = threading.get_ident()
        return {"delete_count": 1}

    s = _make_storage(MockEmbeddingFunc())
    s._client.query = MagicMock(side_effect=fake_query)
    s._client.delete = MagicMock(side_effect=fake_delete)

    await s.delete_entity_relation("entity-1")

    assert call_thread_id["query"] != main_thread_id
    assert call_thread_id["delete"] != main_thread_id


@pytest.mark.asyncio
async def test_delete_entity_relation_no_server_rows_still_offloads_the_query():
    """No matching rows means delete() is never called, but the query()
    that discovers that must still be offloaded -- it is the same blocking
    round trip regardless of how many rows come back."""
    main_thread_id = threading.get_ident()
    call_thread_id = {}

    def fake_query(**kwargs):
        call_thread_id["query"] = threading.get_ident()
        return []

    s = _make_storage(MockEmbeddingFunc())
    s._client.query = MagicMock(side_effect=fake_query)
    s._client.delete = MagicMock()

    await s.delete_entity_relation("entity-1")

    assert call_thread_id["query"] != main_thread_id
    s._client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_entity_relation_without_client_does_not_touch_executor():
    """Negative case: with no server client, delete_entity_relation only
    prunes the in-memory pending-upsert buffer -- a path that already
    worked correctly before this fix and must stay untouched by it."""
    s = _make_storage(MockEmbeddingFunc())
    s._client = None
    s._pending_vector_docs = {
        "v1": type("P", (), {"source": {"src_id": "entity-1", "tgt_id": "other"}})()
    }

    await s.delete_entity_relation("entity-1")

    assert "v1" not in s._pending_vector_docs


@pytest.mark.asyncio
async def test_get_by_ids_runs_query_off_the_event_loop_thread():
    """get_by_ids (and get_vectors_by_ids, which shares the same
    _query_rows_by_ids helper) must offload the paged query() calls."""
    main_thread_id = threading.get_ident()
    call_thread_id = {}

    def fake_query(**kwargs):
        call_thread_id["query"] = threading.get_ident()
        return [{"id": "v1", "content": "hello"}]

    s = _make_storage(MockEmbeddingFunc())
    s._client.query = MagicMock(side_effect=fake_query)

    result = await s.get_by_ids(["v1"])

    assert result == [{"id": "v1", "content": "hello"}]
    assert call_thread_id["query"] != main_thread_id


@pytest.mark.asyncio
async def test_get_by_id_runs_query_off_the_event_loop_thread():
    main_thread_id = threading.get_ident()
    call_thread_id = {}

    def fake_query(**kwargs):
        call_thread_id["query"] = threading.get_ident()
        return [{"id": "v1", "content": "hello"}]

    s = _make_storage(MockEmbeddingFunc())
    s._client.query = MagicMock(side_effect=fake_query)

    result = await s.get_by_id("v1")

    assert result == {"id": "v1", "content": "hello"}
    assert call_thread_id["query"] != main_thread_id


@pytest.mark.asyncio
async def test_get_by_id_returns_buffered_value_without_touching_client():
    """Negative case: a read-your-writes hit against the pending-upsert
    buffer must short-circuit before ever reaching the client/executor --
    unchanged behavior this fix must not disturb."""
    s = _make_storage(MockEmbeddingFunc())
    s._client.query = MagicMock()
    s._pending_vector_docs = {
        "v1": type("P", (), {"source": {"content": "buffered"}})()
    }

    result = await s.get_by_id("v1")

    assert result == {"content": "buffered", "id": "v1"}
    s._client.query.assert_not_called()


@pytest.mark.asyncio
async def test_new_call_sites_use_dedicated_pool_not_default():
    """Same requirement as test_blocking_calls_use_the_dedicated_milvus_pool
    above, extended to the three call sites this fix adds -- offloading to
    the wrong (shared, unbounded-queue) pool reintroduces the exact
    contention problem #3862 introduced this pool to avoid."""
    thread_names: dict[str, str] = {}

    def record(op, result):
        def _fake(**kwargs):
            thread_names[op] = threading.current_thread().name
            return result

        return _fake

    s = _make_storage(MockEmbeddingFunc())
    s._client.query = MagicMock(side_effect=record("query", [{"id": "rel-1"}]))
    s._client.delete = MagicMock(side_effect=record("delete", {"delete_count": 1}))

    await s.delete_entity_relation("entity-1")

    assert thread_names["query"].startswith("lightrag-milvus")
    assert thread_names["delete"].startswith("lightrag-milvus")


@pytest.mark.asyncio
async def test_concurrent_searches_are_capped_by_the_submit_limit():
    """The dedicated pool's wait queue is unbounded like any other
    ThreadPoolExecutor's, so submissions are ceilinged by a semaphore
    (MILVUS_SUBMIT_LIMIT) rather than by the pool itself. Without that ceiling
    every concurrent search gets its own thread up to the pool's width."""
    state = {"live": 0, "peak": 0}
    guard = threading.Lock()

    def fake_search(**kwargs):
        with guard:
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
        time.sleep(0.1)
        with guard:
            state["live"] -= 1
        return [[]]

    s = _make_storage(MockEmbeddingFunc())
    s._client.search = MagicMock(side_effect=fake_search)

    await asyncio.gather(
        *(
            s.query("hello", top_k=5, query_embedding=[0.1] * 8)
            for _ in range(MILVUS_SUBMIT_LIMIT + 4)
        )
    )

    assert s._client.search.call_count == MILVUS_SUBMIT_LIMIT + 4
    assert state["peak"] <= MILVUS_SUBMIT_LIMIT
