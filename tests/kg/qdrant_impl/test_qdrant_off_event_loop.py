"""QdrantVectorDBStorage runs every request-path QdrantClient call off the
event loop, in the dedicated bounded ``lightrag-qdrant`` pool.

QdrantClient is synchronous; awaited inline, each round trip stalls every
other coroutine in the process. These tests pin three things:

* every request-path call (query, flush upsert/delete, delete_entity_relation,
  is_empty, the id lookups, drop) runs on a ``lightrag-qdrant`` thread -- not
  on the loop and not on the default executor;
* a write made under ``_flush_lock`` cannot be outlived by its cancelled
  caller: the lock is held until the write and its buffer bookkeeping are
  done, so a later writer cannot overtake it on the server;
* concurrent calls are capped at ``QDRANT_SUBMIT_LIMIT`` however many the
  caller issues at once, and the loop stays responsive meanwhile.

Mocks only -- no running Qdrant instance required.
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "qdrant_client",
    reason="qdrant-client is required for Qdrant storage tests",
)

from lightrag.constants import QDRANT_SUBMIT_LIMIT  # noqa: E402
from lightrag.kg import qdrant_impl  # noqa: E402
from lightrag.kg.qdrant_impl import (  # noqa: E402
    QdrantVectorDBStorage,
    _PendingVectorDoc,
    compute_mdhash_id_for_qdrant,
)

pytestmark = pytest.mark.offline

POOL_PREFIX = "lightrag-qdrant"


class MockEmbeddingFunc:
    def __init__(self, dim=4):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts, **kwargs):
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


def _make_storage(*, namespace="entities", workspace="test_ws"):
    storage = QdrantVectorDBStorage.__new__(QdrantVectorDBStorage)
    storage.workspace = workspace
    storage.namespace = namespace
    storage.effective_workspace = workspace
    storage.model_suffix = "mock"
    storage.final_namespace = f"lightrag_vdb_{namespace}_mock"
    storage.meta_fields = {"content", "src_id", "tgt_id"}
    storage.embedding_func = MockEmbeddingFunc()
    storage.cosine_better_than_threshold = 0.2
    storage._max_batch_size = 10
    storage._max_upsert_payload_bytes = 16 * 1024 * 1024
    storage._max_upsert_points_per_batch = 128
    storage._max_delete_points_per_batch = 1000
    storage._pending_vector_docs = {}
    storage._pending_vector_deletes = set()
    storage._flush_lock = asyncio.Lock()
    storage._client = MagicMock()
    return storage


def _pending(doc_id, **source):
    return _PendingVectorDoc(
        source={"id": doc_id, "workspace_id": "test_ws", "created_at": 0, **source},
        content="c",
        vector=[0.1, 0.2, 0.3, 0.4],
    )


class _ThreadRecorder:
    """Records the thread name of every call made through it."""

    def __init__(self):
        self.names: list[str] = []

    def __call__(self, result):
        def fake(*args, **kwargs):
            self.names.append(threading.current_thread().name)
            return result

        return MagicMock(side_effect=fake)

    def assert_all_in_pool(self, expected_calls):
        assert len(self.names) == expected_calls, self.names
        assert all(n.startswith(POOL_PREFIX) for n in self.names), self.names


# ---------------------------------------------------------------------------
# Every request-path call runs in the dedicated pool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_runs_in_the_qdrant_pool():
    s = _make_storage()
    rec = _ThreadRecorder()
    point = SimpleNamespace(payload={"id": "a", "created_at": 1}, score=0.9)
    s._client.query_points = rec(SimpleNamespace(points=[point]))

    result = await s.query("q", top_k=3, query_embedding=[0.1] * 4)

    assert result == [{"id": "a", "created_at": 1, "distance": 0.9}]
    rec.assert_all_in_pool(1)


@pytest.mark.asyncio
async def test_flush_runs_upsert_and_delete_in_the_qdrant_pool():
    s = _make_storage()
    rec = _ThreadRecorder()
    s._client.upsert = rec(None)
    s._client.delete = rec(None)
    s._pending_vector_docs["a"] = _pending("a")
    s._pending_vector_deletes.add("b")

    await s.index_done_callback()

    rec.assert_all_in_pool(2)
    assert s._pending_vector_docs == {}
    assert s._pending_vector_deletes == set()


@pytest.mark.asyncio
async def test_delete_entity_relation_runs_scroll_and_delete_in_the_qdrant_pool():
    s = _make_storage(namespace="relationships")
    rec = _ThreadRecorder()
    s._client.scroll = rec(([SimpleNamespace(id="p1")], None))
    s._client.delete = rec(None)
    s._pending_vector_docs["r1"] = _pending("r1", src_id="E", tgt_id="X")
    s._pending_vector_docs["r2"] = _pending("r2", src_id="Y", tgt_id="Z")

    await s.delete_entity_relation("E")

    rec.assert_all_in_pool(2)
    assert set(s._pending_vector_docs) == {"r2"}


@pytest.mark.asyncio
async def test_drop_runs_server_side_steps_in_the_qdrant_pool():
    s = _make_storage()
    rec = _ThreadRecorder()
    s._client.collection_exists = rec(True)
    s._client.delete = rec(None)
    s._pending_vector_docs["a"] = _pending("a")

    with patch.object(qdrant_impl, "_find_legacy_collection", return_value=None):
        result = await s.drop()

    assert result["status"] == "success"
    rec.assert_all_in_pool(2)
    assert s._pending_vector_docs == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.is_empty(),
        lambda s: s.get_by_id("a"),
        lambda s: s.get_by_ids(["a", "b"]),
        lambda s: s.get_vectors_by_ids(["a"]),
    ],
    ids=["is_empty", "get_by_id", "get_by_ids", "get_vectors_by_ids"],
)
async def test_reads_run_in_the_qdrant_pool(call):
    s = _make_storage()
    rec = _ThreadRecorder()
    s._client.scroll = rec(([], None))
    s._client.retrieve = rec([])

    await call(s)

    rec.assert_all_in_pool(1)


@pytest.mark.asyncio
async def test_exact_count_is_exact_scoped_and_runs_in_the_qdrant_pool():
    s = _make_storage()
    rec = _ThreadRecorder()
    s._client.count = rec(SimpleNamespace(count=4))

    assert await s.get_exact_count() == 4
    rec.assert_all_in_pool(1)
    kwargs = s._client.count.call_args.kwargs
    assert kwargs["exact"] is True
    assert "test_ws" in repr(kwargs["count_filter"])


@pytest.mark.asyncio
async def test_strict_read_raises_where_get_by_ids_reports_nothing():
    s = _make_storage()
    s._client.retrieve = MagicMock(side_effect=ConnectionError("qdrant down"))

    assert await s.get_by_ids(["a"]) == []
    with pytest.raises(ConnectionError):
        await s.get_by_ids_strict(["a"])


# ---------------------------------------------------------------------------
# A cancelled writer cannot release _flush_lock mid-write
# ---------------------------------------------------------------------------


_RELEASES: list[threading.Event] = []


@pytest.fixture(autouse=True)
def _release_blocked_calls():
    """Unblock every gated call at teardown, so a test that fails before its
    own release does not leave a pool worker parked for the next test."""
    yield
    while _RELEASES:
        _RELEASES.pop().set()


def _release_event() -> threading.Event:
    release = threading.Event()
    _RELEASES.append(release)
    return release


def _gated(result=None):
    """A fake client call that blocks until released, flagging its start."""
    started = threading.Event()
    release = _release_event()

    def fake(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5), "test never released the call"
        return result

    return started, release, MagicMock(side_effect=fake)


async def _wait_for(event):
    for _ in range(500):
        if event.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("call never started")


@pytest.mark.asyncio
async def test_cancelled_flush_holds_the_lock_until_the_write_lands():
    s = _make_storage()
    started, release, s._client.upsert = _gated()
    s._pending_vector_docs["a"] = _pending("a")

    task = asyncio.ensure_future(s.index_done_callback())
    await _wait_for(started)
    # Cancel twice, the way a shutdown sweep can: a single shield would let
    # the second cancellation escape while the write is still running.
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.05)
    assert s._flush_lock.locked(), "lock released while the upsert is in flight"

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    s._client.upsert.assert_called_once()
    # The write landed, so its buffer pop ran too.
    assert s._pending_vector_docs == {}
    assert not s._flush_lock.locked()


@pytest.mark.asyncio
async def test_drop_after_a_cancelled_flush_is_not_resurrected():
    """The defect this PR's first version introduced: a cancelled flush
    released the lock while its upsert ran on, drop() then cleared the
    workspace, and the stale upsert landed afterwards."""
    s = _make_storage()
    events: list[str] = []
    started = threading.Event()

    def slow_upsert(**kwargs):
        started.set()
        time.sleep(0.2)
        events.append("upsert")

    s._client.upsert = MagicMock(side_effect=slow_upsert)
    s._client.collection_exists = MagicMock(return_value=True)
    s._client.delete = MagicMock(side_effect=lambda **kw: events.append("drop"))
    s._pending_vector_docs["a"] = _pending("a")

    task = asyncio.ensure_future(s.index_done_callback())
    await _wait_for(started)
    task.cancel()

    with patch.object(qdrant_impl, "_find_legacy_collection", return_value=None):
        drop_result = await s.drop()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert drop_result["status"] == "success"
    assert events == ["upsert", "drop"]


@pytest.mark.asyncio
async def test_failed_write_under_cancellation_keeps_the_buffer():
    s = _make_storage()
    started = threading.Event()
    release = _release_event()

    def failing_upsert(**kwargs):
        started.set()
        release.wait(timeout=5)
        raise RuntimeError("qdrant down")

    s._client.upsert = MagicMock(side_effect=failing_upsert)
    s._pending_vector_docs["a"] = _pending("a")

    task = asyncio.ensure_future(s.index_done_callback())
    await _wait_for(started)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert set(s._pending_vector_docs) == {"a"}
    assert not s._flush_lock.locked()


@pytest.mark.asyncio
async def test_cancelled_delete_entity_relation_still_prunes_after_the_delete():
    s = _make_storage(namespace="relationships")
    s._client.scroll = MagicMock(return_value=([SimpleNamespace(id="p1")], None))
    started, release, s._client.delete = _gated()
    s._pending_vector_docs["r1"] = _pending("r1", src_id="E", tgt_id="X")

    task = asyncio.ensure_future(s.delete_entity_relation("E"))
    await _wait_for(started)
    task.cancel()
    await asyncio.sleep(0.05)
    assert s._flush_lock.locked()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert s._pending_vector_docs == {}
    assert not s._flush_lock.locked()


@pytest.mark.asyncio
async def test_cancelled_drop_holds_the_lock_until_the_delete_lands():
    s = _make_storage()
    s._client.collection_exists = MagicMock(return_value=True)
    started, release, s._client.delete = _gated()

    with patch.object(qdrant_impl, "_find_legacy_collection", return_value=None):
        task = asyncio.ensure_future(s.drop())
        await _wait_for(started)
        task.cancel()
        await asyncio.sleep(0.05)
        assert s._flush_lock.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert not s._flush_lock.locked()


@pytest.mark.asyncio
async def test_cancelled_read_returns_at_once():
    """Reads hold no lock, so they must NOT wait out their round trip."""
    s = _make_storage()
    started, release, s._client.query_points = _gated(SimpleNamespace(points=[]))

    task = asyncio.ensure_future(s.query("q", top_k=1, query_embedding=[0.1] * 4))
    await _wait_for(started)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
    finally:
        release.set()


# ---------------------------------------------------------------------------
# Concurrency ceiling and loop responsiveness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_burst_of_queries_is_capped_and_the_loop_stays_responsive():
    s = _make_storage()
    state = {"in_flight": 0, "peak": 0}
    guard = threading.Lock()

    def slow_query(**kwargs):
        with guard:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
        time.sleep(0.05)
        with guard:
            state["in_flight"] -= 1
        return SimpleNamespace(points=[])

    s._client.query_points = MagicMock(side_effect=slow_query)

    ticks = 0
    stop = asyncio.Event()

    async def heartbeat():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    hb = asyncio.ensure_future(heartbeat())
    n = QDRANT_SUBMIT_LIMIT * 4
    await asyncio.gather(
        *(s.query("q", top_k=1, query_embedding=[0.1] * 4) for _ in range(n))
    )
    stop.set()
    await hb

    assert s._client.query_points.call_count == n
    assert state["peak"] == QDRANT_SUBMIT_LIMIT
    # n calls of 50 ms in waves of QDRANT_SUBMIT_LIMIT take ~200 ms; a loop
    # blocked by the calls would tick a handful of times at most.
    assert ticks >= 10, ticks


@pytest.mark.asyncio
async def test_writers_and_readers_share_one_ceiling():
    """A flush and a burst of reads together never exceed the ceiling."""
    s = _make_storage()
    state = {"in_flight": 0, "peak": 0}
    guard = threading.Lock()

    def slow(result):
        def fake(**kwargs):
            with guard:
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            time.sleep(0.05)
            with guard:
                state["in_flight"] -= 1
            return result

        return MagicMock(side_effect=fake)

    s._client.upsert = slow(None)
    s._client.retrieve = slow([])
    s._pending_vector_docs["a"] = _pending("a")

    reads = [s.get_by_ids([f"id-{i}"]) for i in range(QDRANT_SUBMIT_LIMIT * 2)]
    await asyncio.gather(s.index_done_callback(), *reads)

    assert state["peak"] <= QDRANT_SUBMIT_LIMIT
    assert s._pending_vector_docs == {}


def test_pool_width_matches_the_submit_limit():
    assert qdrant_impl.get_qdrant_executor()._max_workers == QDRANT_SUBMIT_LIMIT


@pytest.mark.asyncio
async def test_flush_point_ids_are_unchanged():
    """Moving the write into the pool must not change what is written."""
    s = _make_storage()
    s._client.upsert = MagicMock()
    s._client.delete = MagicMock()
    s._pending_vector_docs["a"] = _pending("a")
    s._pending_vector_deletes.add("b")

    await s.index_done_callback()

    (points,) = [c.kwargs["points"] for c in s._client.upsert.call_args_list]
    assert [p.id for p in points] == [
        compute_mdhash_id_for_qdrant("a", prefix="test_ws")
    ]
    selector = s._client.delete.call_args.kwargs["points_selector"]
    assert selector.points == [compute_mdhash_id_for_qdrant("b", prefix="test_ws")]
