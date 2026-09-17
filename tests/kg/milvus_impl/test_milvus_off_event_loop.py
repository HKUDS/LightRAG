"""MilvusVectorDBStorage.query(), _flush_pending_vector_ops(),
delete_entity_relation(), _query_rows_by_ids() (used by get_by_ids() /
get_vectors_by_ids()), get_by_id() and drop() call the synchronous
MilvusClient SDK (blocking gRPC) directly inside async methods. Calling it
without offloading would block the whole event loop for the duration of every
search/upsert/delete/query/collection-management round trip, stalling every
other concurrent task (LLM calls, other storage I/O) sharing the loop.

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
from lightrag.kg import milvus_impl
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
async def test_cancelling_delete_entity_relation_defers_until_delete_completes_then_prunes():
    """run_in_milvus_executor only cancels the awaiting future -- an in-flight
    Milvus delete keeps running in the background thread. A bare cancel here
    would release _flush_lock and return to the caller while that delete (and
    the pending-buffer prune that follows it) is still pending, letting a
    concurrent flush reinsert a relation the server already deleted.
    Cancellation must instead be deferred until the delete, and the prune
    that follows it, have actually finished."""
    call_started = threading.Event()
    release_call = threading.Event()

    def fake_delete(**kwargs):
        call_started.set()
        release_call.wait(timeout=5)
        return {"delete_count": 1}

    s = _make_storage(MockEmbeddingFunc())
    s._client.query = MagicMock(return_value=[{"id": "rel-1"}])
    s._client.delete = MagicMock(side_effect=fake_delete)
    s._pending_vector_docs = {
        "rel-1": type("P", (), {"source": {"src_id": "entity-1", "tgt_id": "other"}})()
    }

    task = asyncio.ensure_future(s.delete_entity_relation("entity-1"))
    for _ in range(500):
        if call_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert call_started.is_set()

    task.cancel()
    # Let the background delete finish so the deferred cancellation can
    # resolve -- release_call must be set before awaiting the cancelled
    # task, since the cancellation is held back until the delete completes.
    release_call.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    s._client.delete.assert_called_once()
    # The delete actually landed, so the buffer must reflect that outcome --
    # not the stale "still pending" state a bare cancel would leave, which
    # would let a later flush reinsert the relation the server just deleted.
    assert s._pending_vector_docs == {}
    assert not s._flush_lock.locked()


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
async def test_drop_runs_collection_management_off_the_event_loop_thread():
    """drop() rebuilds the collection through four blocking SDK calls
    (has_collection / drop_collection / create_collection / load_collection).
    A collection drop + recreate is the longest round trip this class makes,
    so running it on the loop thread stalls every concurrent request for its
    whole duration."""
    main_thread_id = threading.get_ident()
    thread_names: dict[str, str] = {}

    def record(op, result=None):
        def _fake(*args, **kwargs):
            thread_names[op] = threading.current_thread().name
            return result

        return _fake

    s = _make_storage(MockEmbeddingFunc())
    s._client.has_collection = MagicMock(side_effect=record("has_collection", True))
    s._client.drop_collection = MagicMock(side_effect=record("drop_collection"))
    s._client.create_collection = MagicMock(side_effect=record("create_collection"))
    s._client.load_collection = MagicMock(side_effect=record("load_collection"))

    result = await s.drop()

    assert result["status"] == "success"
    assert set(thread_names) == {
        "has_collection",
        "drop_collection",
        "create_collection",
        "load_collection",
    }
    for op, name in thread_names.items():
        assert name != threading.current_thread().name, op
        # The dedicated pool, not the default executor: parking a collection
        # rebuild there would queue every namespace-lock acquisition and every
        # login behind it.
        assert name.startswith("lightrag-milvus"), op
    assert threading.get_ident() == main_thread_id


@pytest.mark.asyncio
async def test_drop_clears_the_pending_buffers():
    """Unchanged behavior the offloading must not disturb: buffered writes are
    discarded before the collection goes away, so a later flush cannot
    resurrect rows into the freshly recreated collection."""
    s = _make_storage(MockEmbeddingFunc())
    s._pending_vector_docs = {"v1": object()}
    s._pending_vector_deletes = {"v2"}

    result = await s.drop()

    assert result["status"] == "success"
    assert s._pending_vector_docs == {}
    assert s._pending_vector_deletes == set()


@pytest.mark.asyncio
async def test_cancelling_drop_defers_until_the_collection_is_recreated():
    """run_in_milvus_executor only cancels the awaiting future -- an in-flight
    drop_collection keeps running in the background thread. A bare cancel
    between the drop and the recreate would leave the namespace with NO
    collection at all, and unlike the residues Consistency without
    transactions accepts, that one never heals: nothing recreates it at run
    time and the next initialize() would re-run the legacy migration this
    method exists to avoid. Cancellation must be deferred until the empty
    replacement exists."""
    call_started = threading.Event()
    release_call = threading.Event()

    def fake_drop_collection(*args, **kwargs):
        call_started.set()
        release_call.wait(timeout=5)

    s = _make_storage(MockEmbeddingFunc())
    s._client.drop_collection = MagicMock(side_effect=fake_drop_collection)

    task = asyncio.ensure_future(s.drop())
    for _ in range(500):
        if call_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert call_started.is_set()

    task.cancel()
    # Let the background drop finish so the deferred cancellation can resolve --
    # release_call must be set before awaiting the cancelled task, since the
    # cancellation is held back until the recreate completes.
    release_call.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    # The collection was actually rebuilt, not left missing.
    s._client.drop_collection.assert_called_once()
    s._client.create_collection.assert_called_once()
    s._client.load_collection.assert_called_once()
    assert not s._flush_lock.locked()


@pytest.mark.asyncio
async def test_drop_reports_a_server_error_as_an_error_dict():
    """Unchanged behavior: a failure raised inside the pool must surface as
    drop()'s error dict, not propagate -- clear_documents reads the status."""
    s = _make_storage(MockEmbeddingFunc())
    s._client.drop_collection = MagicMock(side_effect=RuntimeError("boom"))

    result = await s.drop()

    assert result["status"] == "error"
    assert "boom" in result["message"]
    s._client.create_collection.assert_not_called()


async def _drop_blocking_on(storage, blocked_call: str):
    """Start drop() and suspend it inside the window where the collection is
    gone. Returns (task, release) -- call release() to let the rebuild finish.
    """
    call_started = threading.Event()
    release_call = threading.Event()

    def _fake(*args, **kwargs):
        call_started.set()
        release_call.wait(timeout=5)

    setattr(storage._client, blocked_call, MagicMock(side_effect=_fake))
    task = asyncio.ensure_future(storage.drop())
    for _ in range(500):
        if call_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert call_started.is_set()
    return task, release_call.set


@pytest.mark.asyncio
async def test_query_waits_out_the_drop_recreate_window():
    """drop() yields the loop between drop_collection and the recreate, and
    query() holds no lock at all -- so without a gate a concurrent search in
    the same worker hits a collection that does not exist and fails the
    request. Before the offloading the synchronous body made that impossible;
    the reader must wait the rebuild out, as it effectively did then."""
    order: list[str] = []

    s = _make_storage(MockEmbeddingFunc())
    s._client.create_collection = MagicMock(
        side_effect=lambda *a, **k: order.append("create")
    )
    s._client.search = MagicMock(side_effect=lambda **k: order.append("search") or [[]])

    drop_task, release = await _drop_blocking_on(s, "drop_collection")
    query_task = asyncio.ensure_future(
        s.query("hello", top_k=5, query_embedding=[0.1] * 8)
    )

    # Give the query every chance to slip into the window.
    for _ in range(5):
        await asyncio.sleep(0)
    assert order == [], "query reached the client while the collection was gone"

    release()
    # Timeouts, not bare awaits: a gate that is never reopened must fail this
    # test, not hang the run.
    assert (await asyncio.wait_for(drop_task, timeout=5))["status"] == "success"
    assert await asyncio.wait_for(query_task, timeout=5) == []
    assert order == ["create", "search"]


@pytest.mark.parametrize("read", ["get_by_id", "get_by_ids"])
@pytest.mark.asyncio
async def test_id_lookups_wait_at_the_gate_while_the_collection_is_gone(read):
    """The id lookups take _flush_lock only for their buffer phase and release
    it before the server leg, so -- unlike a reader that arrives after drop()
    already holds the lock -- one already past that phase can resume inside the
    rebuild window. get_by_id's except-Exception would then report the missing
    collection as a plain None: a broken backend read as a missing row.

    Driven through the gate directly rather than through a drop(): what the
    lock happens to serialize depends on whether UnifiedLock.__aexit__
    suspends, and the reader must hold regardless of that.
    """
    s = _make_storage(MockEmbeddingFunc())
    s._client.query = MagicMock(return_value=[{"id": "v1", "content": "hi"}])

    gate = milvus_impl.get_collection_gate(s.final_namespace)
    gate.close()
    task = asyncio.ensure_future(
        s.get_by_id("v1") if read == "get_by_id" else s.get_by_ids(["v1"])
    )
    for _ in range(5):
        await asyncio.sleep(0)
    s._client.query.assert_not_called()

    gate.reopen()
    result = await asyncio.wait_for(task, timeout=5)

    expected = {"id": "v1", "content": "hi"}
    assert result == (expected if read == "get_by_id" else [expected])
    s._client.query.assert_called_once()


@pytest.mark.asyncio
async def test_query_re_gates_after_the_embedding_round_trip():
    """A reader that cleared the gate at entry can then sit in an embedding
    round trip for seconds. Checking once per method would leave that reader
    free to issue its search into a rebuild that started meanwhile, so every
    submission re-checks -- not just the first one in the method."""
    order: list[str] = []
    embedding_started = asyncio.Event()
    release_embedding = asyncio.Event()

    class SlowEmbeddingFunc(MockEmbeddingFunc):
        async def __call__(self, texts, **kwargs):
            embedding_started.set()
            await release_embedding.wait()
            return await super().__call__(texts, **kwargs)

    s = _make_storage(SlowEmbeddingFunc())
    s._client.create_collection = MagicMock(
        side_effect=lambda *a, **k: order.append("create")
    )
    s._client.search = MagicMock(side_effect=lambda **k: order.append("search") or [[]])

    # No query_embedding: the reader must go through embedding_func.
    query_task = asyncio.ensure_future(s.query("hello", top_k=5))
    await asyncio.wait_for(embedding_started.wait(), timeout=5)

    # The reader is past its entry gate and suspended. Now start the drop.
    drop_task, release_drop = await _drop_blocking_on(s, "drop_collection")
    release_embedding.set()
    for _ in range(5):
        await asyncio.sleep(0)
    assert "search" not in order, "search was issued while the collection was gone"

    release_drop()
    assert (await asyncio.wait_for(drop_task, timeout=5))["status"] == "success"
    assert await asyncio.wait_for(query_task, timeout=5) == []
    assert order == ["create", "search"]


@pytest.mark.asyncio
async def test_aliased_instances_share_one_collection_gate():
    """Two instances can resolve to the same collection (two LightRAG objects,
    or workspaces collapsed by MILVUS_WORKSPACE). They already share
    _flush_lock, keyed on final_namespace; the gate must be keyed the same way
    or a drop through one leaves the other's readers free to hit the window."""
    a = _make_storage(MockEmbeddingFunc())
    b = _make_storage(MockEmbeddingFunc())
    assert a.final_namespace == b.final_namespace
    assert milvus_impl.get_collection_gate(
        a.final_namespace
    ) is milvus_impl.get_collection_gate(b.final_namespace)

    b._client.search = MagicMock(return_value=[[]])
    drop_task, release = await _drop_blocking_on(a, "drop_collection")

    read_task = asyncio.ensure_future(
        b.query("hello", top_k=5, query_embedding=[0.1] * 8)
    )
    for _ in range(5):
        await asyncio.sleep(0)
    b._client.search.assert_not_called()

    release()
    assert (await asyncio.wait_for(drop_task, timeout=5))["status"] == "success"
    assert await asyncio.wait_for(read_task, timeout=5) == []


def test_each_loop_gets_its_own_collection_gate():
    """asyncio.Event is loop-bound in fact if not in signature: one that
    actually blocks binds itself, and a later loop blocking on it raises
    'bound to a different event loop'. A gate shared across loops would
    therefore turn a read that merely fails during a clear into a hang or a
    spurious error, so the registry keys on the running loop -- which is also
    what makes a storage object survive successive asyncio.run() calls."""
    s = _make_storage(MockEmbeddingFunc())
    namespace = s.final_namespace

    seen = {}

    async def bind_the_gate(key):
        gate = milvus_impl.get_collection_gate(namespace)
        seen[key] = gate
        # Make a reader actually block, which is what binds the event.
        gate.close()
        waiter = asyncio.ensure_future(gate.acquire_read())
        await asyncio.sleep(0)
        gate.reopen()
        await asyncio.wait_for(waiter, timeout=5)
        gate.release_read()

    asyncio.run(bind_the_gate("first"))
    # Would raise "bound to a different event loop" on a shared gate.
    asyncio.run(bind_the_gate("second"))

    assert seen["first"] is not seen["second"]
    # The closed first loop must not be retained.
    assert len(milvus_impl._COLLECTION_GATES) <= 1


@pytest.mark.asyncio
async def test_drop_waits_for_an_in_flight_read_before_removing_the_collection():
    """The gate is a lease, not just a flag. A reader that cleared the flag can
    still be between that check and its call landing on a pool thread, so a
    rebuild starting there would remove the collection underneath a read that
    is already on its way. drop() must wait for the reads in flight -- and only
    for those: the lease covers one SDK round trip, never a reader's embedding
    work."""
    order: list[str] = []
    read_started = threading.Event()
    release_read = threading.Event()

    def slow_search(**kwargs):
        order.append("search")
        read_started.set()
        release_read.wait(timeout=5)
        return [[]]

    s = _make_storage(MockEmbeddingFunc())
    s._client.search = MagicMock(side_effect=slow_search)
    s._client.drop_collection = MagicMock(
        side_effect=lambda *a, **k: order.append("drop")
    )

    query_task = asyncio.ensure_future(
        s.query("hello", top_k=5, query_embedding=[0.1] * 8)
    )
    for _ in range(500):
        if read_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert read_started.is_set()

    drop_task = asyncio.ensure_future(s.drop())
    # Real time, not bare ticks: drop() has to take _flush_lock, spawn its
    # inner task and round-trip has_collection through the pool before it could
    # reach drop_collection, which is far more than a few loop iterations. Give
    # an ungated drop ample room to violate this, and break as soon as it does.
    for _ in range(50):
        if "drop" in order:
            break
        await asyncio.sleep(0.02)
    assert "drop" not in order, "collection was removed under an in-flight read"

    release_read.set()
    assert await asyncio.wait_for(query_task, timeout=5) == []
    assert (await asyncio.wait_for(drop_task, timeout=5))["status"] == "success"
    assert order == ["search", "drop"]


@pytest.mark.asyncio
async def test_a_shutdown_cancelling_every_task_still_recreates_the_collection():
    """drop() runs its rebuild in a task of its own, which a shutdown that
    cancels every pending task (asyncio.run's _cancel_all_tasks, an ASGI
    teardown) reaches DIRECTLY -- and a direct cancellation is one
    _wait_deferring_cancellation re-raises rather than defers. Were the rebuild
    a sequence of awaits, the cancel would land between them and leave the
    namespace with no collection, which nothing recreates at run time. One
    submission is what makes the pool thread carry it to the end regardless."""
    order: list[str] = []
    drop_started = threading.Event()
    release_drop = threading.Event()

    def blocking_drop(*args, **kwargs):
        order.append("drop")
        drop_started.set()
        release_drop.wait(timeout=5)

    s = _make_storage(MockEmbeddingFunc())
    s._client.drop_collection = MagicMock(side_effect=blocking_drop)
    s._client.create_collection = MagicMock(
        side_effect=lambda *a, **k: order.append("create")
    )

    task = asyncio.ensure_future(s.drop())
    for _ in range(500):
        if drop_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert drop_started.is_set()

    # The shutdown: cancel everything but this test's own task, which reaches
    # drop()'s caller task and its inner rebuild task alike.
    for pending in asyncio.all_tasks():
        if pending is not asyncio.current_task():
            pending.cancel()

    release_drop.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The pool thread finished the rebuild even though every task was cancelled.
    for _ in range(500):
        if "create" in order:
            break
        await asyncio.sleep(0.01)
    assert order == ["drop", "create"], (
        "the collection was left missing after a shutdown cancellation"
    )
    s._client.load_collection.assert_called()


@pytest.mark.asyncio
async def test_a_shutdown_cannot_release_the_writer_lock_mid_rebuild():
    """The rebuild survives a shutdown, but that is only half of it: while the
    pool thread still has a collection to remove, _flush_lock -- and above it
    the caller's destructive_busy reservation -- must stay held. Unwinding
    early would let a writer land a row the rebuild then erases, which is the
    one outcome Consistency without transactions never allows. The submission
    is uninterruptible and runs in the caller's own task precisely so no
    cancellation can unwind it early."""
    rebuild_started = threading.Event()
    release_rebuild = threading.Event()

    def blocking_has_collection(*args, **kwargs):
        rebuild_started.set()
        release_rebuild.wait(timeout=5)
        return True

    s = _make_storage(MockEmbeddingFunc())
    s._client.has_collection = MagicMock(side_effect=blocking_has_collection)

    task = asyncio.ensure_future(s.drop())
    for _ in range(500):
        if rebuild_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert rebuild_started.is_set()
    assert s._flush_lock.locked()

    for pending in asyncio.all_tasks():
        if pending is not asyncio.current_task():
            pending.cancel()

    # Real time, so an unwinding caller would have every chance to release it.
    for _ in range(25):
        await asyncio.sleep(0.02)
        if not s._flush_lock.locked():
            break
    assert s._flush_lock.locked(), "writer lock released while the rebuild ran"

    release_rebuild.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not s._flush_lock.locked()
    s._client.create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_a_shutdown_during_delete_entity_relation_still_prunes():
    """The prune is bookkeeping that must not be separated from the delete
    that made it necessary: a buffered upsert left behind is one a later flush
    re-sends, resurrecting a relation the server already removed. A shutdown
    cancelling every task must therefore not be able to land between them --
    which is why the prune is the commit hook of the same uncancellable
    submission rather than a statement after the await."""
    delete_started = threading.Event()
    release_delete = threading.Event()

    def blocking_delete(**kwargs):
        delete_started.set()
        release_delete.wait(timeout=5)
        return {"delete_count": 1}

    s = _make_storage(MockEmbeddingFunc())
    s._client.query = MagicMock(return_value=[{"id": "rel-1"}])
    s._client.delete = MagicMock(side_effect=blocking_delete)
    s._pending_vector_docs = {
        "rel-1": type("P", (), {"source": {"src_id": "entity-1", "tgt_id": "other"}})()
    }

    task = asyncio.ensure_future(s.delete_entity_relation("entity-1"))
    for _ in range(500):
        if delete_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert delete_started.is_set()

    for pending in asyncio.all_tasks():
        if pending is not asyncio.current_task():
            pending.cancel()

    release_delete.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    s._client.delete.assert_called_once()
    assert s._pending_vector_docs == {}, (
        "the delete landed but its buffer prune was skipped"
    )
    assert not s._flush_lock.locked()


@pytest.mark.asyncio
async def test_the_relation_prune_runs_inside_the_delete_submission():
    """Structural pin for the case above. The prune is not a statement after
    the await, nor on_committed -- that hook is an ensure_future task of its
    own, which an all-tasks sweep can cancel in the window between the delete
    landing and the hook's first step. It is inside the submitted callable, so
    the pool thread carries delete and prune to the end together. Asserting the
    prune happens on a pool thread is what pins that."""

    class _RecordingDocs(dict):
        popped_on: list[str] = []

        def pop(self, *args, **kwargs):
            _RecordingDocs.popped_on.append(threading.current_thread().name)
            return super().pop(*args, **kwargs)

    _RecordingDocs.popped_on = []

    s = _make_storage(MockEmbeddingFunc())
    s._client.query = MagicMock(return_value=[{"id": "rel-1"}])
    s._client.delete = MagicMock(return_value={"delete_count": 1})
    s._pending_vector_docs = _RecordingDocs(
        {
            "rel-1": type(
                "P", (), {"source": {"src_id": "entity-1", "tgt_id": "other"}}
            )()
        }
    )

    await s.delete_entity_relation("entity-1")

    assert s._pending_vector_docs == {}
    assert _RecordingDocs.popped_on, "the prune never ran"
    for name in _RecordingDocs.popped_on:
        assert name.startswith("lightrag-milvus"), (
            f"the prune ran on {name}, not inside the delete submission"
        )


@pytest.mark.asyncio
async def test_a_failed_drop_reopens_the_reader_gate():
    """The gate must be reopened even when the rebuild raises: a reader held
    behind a gate nothing reopens would hang forever, which is strictly worse
    than the error it was being spared."""
    s = _make_storage(MockEmbeddingFunc())
    s._client.drop_collection = MagicMock(side_effect=RuntimeError("boom"))
    s._client.search = MagicMock(return_value=[[]])

    assert (await s.drop())["status"] == "error"

    # Would hang instead of returning if the gate stayed closed.
    result = await asyncio.wait_for(
        s.query("hello", top_k=5, query_embedding=[0.1] * 8), timeout=5
    )
    assert result == []


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
