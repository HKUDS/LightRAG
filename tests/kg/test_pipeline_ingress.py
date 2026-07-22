"""Phase 1 tests for the workspace-scoped pipeline ingress.

Covers the three-channel mailbox contract (document / auto-rescan dirty flag /
sticky manual retries with bounded terminal ids), the single-process
``AsyncioPipelineIngress`` (real ``asyncio.Queue`` document channel, paired
``task_done``, event-driven waits, owning-loop fast fail), the Manager-backed
multiprocess mailbox (cross-process discovery through the shared proxy table,
sticky survival after publisher death, observe-only waiting), and the
shared_storage registry lifecycle (workspace isolation, idempotent
initialization, teardown-only finalize).
"""

import asyncio
import multiprocessing as mp
from pathlib import Path

import pytest

import lightrag.kg.shared_storage as shared_storage
from lightrag.kg.pipeline_ingress import (
    MANUAL_RETRY_ACKED,
    MANUAL_RETRY_CANCELLED_BY_CLEAR,
    AsyncioPipelineIngress,
    PipelineIngress,
    PipelineIngressMessage,
    _BoundedTerminalIds,
)
from lightrag.kg.shared_storage import (
    finalize_pipeline_ingress,
    finalize_share_data,
    get_final_namespace,
    get_pipeline_ingress,
    initialize_pipeline_ingress,
    initialize_share_data,
)

pytestmark = pytest.mark.offline


def _doc(doc_id: str) -> PipelineIngressMessage:
    return PipelineIngressMessage(kind="document", doc_id=doc_id)


def _manual(request_id: str) -> PipelineIngressMessage:
    return PipelineIngressMessage(
        kind="rescan", retry_failed=True, request_id=request_id
    )


@pytest.fixture
def single_process_share_data():
    """Fresh single-process shared data per test (isolated registry/loop)."""
    finalize_share_data()
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


# ---------------------------------------------------------------------------
# Registry lifecycle (single-process)
# ---------------------------------------------------------------------------


async def test_get_before_initialize_raises():
    finalize_share_data()
    with pytest.raises(ValueError, match="not initialized"):
        await get_pipeline_ingress("ws")


async def test_workspace_isolation_and_idempotent_resolution(
    single_process_share_data,
):
    a1 = await get_pipeline_ingress("wsA")
    a2 = await get_pipeline_ingress("wsA")
    b = await get_pipeline_ingress("wsB")
    assert a1 is a2
    assert a1 is not b
    assert isinstance(a1, PipelineIngress)

    a1.put_document(_doc("d-a"))
    assert a1.has_work() and not b.has_work()
    assert [m.doc_id for m in a1.drain_documents()] == ["d-a"]
    assert b.drain_documents() == []


async def test_initialize_pipeline_ingress_is_idempotent(single_process_share_data):
    await initialize_pipeline_ingress("wsA")
    first = await get_pipeline_ingress("wsA")
    await initialize_pipeline_ingress("wsA")
    assert (await get_pipeline_ingress("wsA")) is first


async def test_finalize_pipeline_ingress_drops_and_recreates(
    single_process_share_data,
):
    ingress = await get_pipeline_ingress("wsA")
    ingress.request_manual_retry("r1", _manual("r1"))
    await finalize_pipeline_ingress("wsA")
    fresh = await get_pipeline_ingress("wsA")
    assert fresh is not ingress
    assert not fresh.has_work()  # teardown really dropped the sticky request


def test_mailbox_wait_timeout_is_required_and_clamped():
    """A SIGKILLed waiter must strand its server thread at most for the
    bounded window — unbounded waits are rejected by construction."""
    from lightrag.kg.pipeline_ingress import (
        MAX_MAILBOX_WAIT_SECONDS,
        PipelineIngressMailbox,
    )

    assert PipelineIngressMailbox._bounded_wait(999.0) == MAX_MAILBOX_WAIT_SECONDS
    assert PipelineIngressMailbox._bounded_wait(-1.0) == 0.0
    assert PipelineIngressMailbox._bounded_wait(0.2) == 0.2
    mailbox = PipelineIngressMailbox()  # in-process instance, no Manager needed
    with pytest.raises(TypeError):
        mailbox.wait_for_documents()  # timeout is required, no None default
    assert mailbox.wait_for_documents(0.01) is False
    assert mailbox.wait_for_items(0.01) is False


def test_cross_loop_access_fast_fails():
    """A single-process ingress belongs to the loop that created it."""
    finalize_share_data()
    initialize_share_data(workers=1)
    try:
        asyncio.run(get_pipeline_ingress("wsA"))  # created on loop A
        with pytest.raises(RuntimeError, match="another event loop"):
            asyncio.run(get_pipeline_ingress("wsA"))  # loop B → fast fail
    finally:
        finalize_share_data()


def test_finalize_storages_never_touches_ingress():
    """Ownership guard: the ingress is workspace-shared, so no LightRAG
    instance teardown path may finalize it (only finalize_share_data /
    explicit workspace teardown / tests)."""
    lightrag_src = (
        Path(shared_storage.__file__).parent.parent / "lightrag.py"
    ).read_text(encoding="utf-8")
    assert "finalize_pipeline_ingress" not in lightrag_src


# ---------------------------------------------------------------------------
# AsyncioPipelineIngress (single-process semantics)
# ---------------------------------------------------------------------------


async def test_single_process_document_channel_is_asyncio_queue(
    single_process_share_data,
):
    ingress = await get_pipeline_ingress("wsA")
    assert isinstance(ingress, AsyncioPipelineIngress)
    assert isinstance(ingress.document_messages, asyncio.Queue)
    # No Manager involvement in single-process mode.
    assert shared_storage._manager is None
    assert shared_storage._pipeline_ingress_proxies is None


async def test_get_document_event_driven_and_task_done_paired(
    single_process_share_data,
):
    ingress = await get_pipeline_ingress("wsA")

    waiter = asyncio.create_task(ingress.get_document())
    await asyncio.sleep(0)
    assert not waiter.done()  # parked on queue.get(), no polling loop to feed

    ingress.put_document(_doc("d1"))
    msg = await asyncio.wait_for(waiter, timeout=1.0)
    assert msg.doc_id == "d1"

    # task_done() paired immediately: join() must complete at once.
    await asyncio.wait_for(ingress.document_messages.join(), timeout=0.1)
    assert not ingress.work_event.is_set()  # cleared once no work remains

    # drain_documents pairs task_done too.
    ingress.put_document(_doc("d2"))
    ingress.put_document(_doc("d3"))
    assert [m.doc_id for m in ingress.drain_documents(limit=1)] == ["d2"]
    assert [m.doc_id for m in ingress.drain_documents()] == ["d3"]
    await asyncio.wait_for(ingress.document_messages.join(), timeout=0.1)


async def test_document_wait_isolated_from_control_channels(
    single_process_share_data,
):
    """Pending manual/auto entries must NOT satisfy a document wait."""
    ingress = await get_pipeline_ingress("wsA")
    ingress.request_auto_rescan()
    ingress.request_manual_retry("r1", _manual("r1"))

    waiter = asyncio.create_task(ingress.get_document())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), timeout=0.2)

    ingress.put_document(_doc("d1"))
    assert (await asyncio.wait_for(waiter, timeout=1.0)).doc_id == "d1"


async def test_wait_for_items_covers_all_channels(single_process_share_data):
    ingress = await get_pipeline_ingress("wsA")

    supervisor_wait = asyncio.create_task(ingress.wait_for_items())
    await asyncio.sleep(0)
    assert not supervisor_wait.done()

    ingress.request_auto_rescan()
    await asyncio.wait_for(supervisor_wait, timeout=1.0)

    # Consuming the only work re-arms the wait instead of spinning.
    assert ingress.consume_auto_rescan() is True
    rearmed = asyncio.create_task(ingress.wait_for_items())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(rearmed), timeout=0.2)
    ingress.request_manual_retry("r1", _manual("r1"))
    await asyncio.wait_for(rearmed, timeout=1.0)


async def test_consume_auto_rescan_atomic_exchange(single_process_share_data):
    ingress = await get_pipeline_ingress("wsA")
    assert ingress.consume_auto_rescan() is False
    ingress.request_auto_rescan()
    ingress.request_auto_rescan()  # idempotent set
    assert ingress.has_work()
    assert ingress.consume_auto_rescan() is True
    assert ingress.consume_auto_rescan() is False
    assert not ingress.has_work()
    assert not ingress.work_event.is_set()


async def test_manual_retry_fifo_ack_and_terminal_replay_guard(
    single_process_share_data,
):
    ingress = await get_pipeline_ingress("wsA")
    for rid in ("r1", "r2", "r3"):
        assert ingress.request_manual_retry(rid, _manual(rid)) is True
    # Re-requesting a pending id is a no-op and keeps its FIFO position.
    assert ingress.request_manual_retry("r2", _manual("r2")) is True
    assert [m.request_id for m in ingress.snapshot_manual_retries()] == [
        "r1",
        "r2",
        "r3",
    ]

    assert ingress.peek_next_manual_retry().request_id == "r1"
    assert ingress.peek_next_manual_retry().request_id == "r1"  # peek ≠ consume
    assert ingress.ack_manual_retry("r1") is True
    assert ingress.ack_manual_retry("r1") is True  # idempotent
    assert ingress.peek_next_manual_retry().request_id == "r2"

    # Terminal id refuses replay.
    assert ingress.request_manual_retry("r1", _manual("r1")) is False
    assert ingress.counts()["manual_retries"] == 2


async def test_clear_tombstones_pending_manual_ids(single_process_share_data):
    ingress = await get_pipeline_ingress("wsA")
    ingress.put_document(_doc("d1"))
    ingress.request_auto_rescan()
    ingress.request_manual_retry("r1", _manual("r1"))

    ingress.clear()
    assert not ingress.has_work()
    # The un-ACKed id was tombstoned: a delayed replay must not re-enter.
    assert ingress.request_manual_retry("r1", _manual("r1")) is False
    # Fresh ids keep working; the terminal set survived the clear.
    assert ingress.request_manual_retry("r2", _manual("r2")) is True
    assert ingress.counts()["terminal_manual_request_ids"] == 1


async def test_manual_request_message_validation(single_process_share_data):
    ingress = await get_pipeline_ingress("wsA")
    with pytest.raises(ValueError, match="non-empty request_id"):
        ingress.request_manual_retry("", _manual(""))
    with pytest.raises(ValueError, match="retry_failed=True"):
        ingress.request_manual_retry("r1", _doc("d1"))
    with pytest.raises(ValueError, match="matching request_id"):
        ingress.request_manual_retry("r1", _manual("other"))


def test_bounded_terminal_ids_fifo_eviction():
    ids = _BoundedTerminalIds(capacity=2)
    ids.add("a", MANUAL_RETRY_ACKED)
    ids.add("b", MANUAL_RETRY_CANCELLED_BY_CLEAR)
    ids.add("a", MANUAL_RETRY_CANCELLED_BY_CLEAR)  # first terminal state wins
    assert ids.get("a") == MANUAL_RETRY_ACKED
    ids.add("c", MANUAL_RETRY_ACKED)  # evicts "a" (oldest)
    assert "a" not in ids
    assert "b" in ids and "c" in ids


# ---------------------------------------------------------------------------
# Manager-backed mailbox (multiprocess mode, real Manager server)
# ---------------------------------------------------------------------------


def _child_publish(proxies_dict, final_namespace):
    """Worker-side publisher: resolves the SAME server-side mailbox through
    the shared proxy table, publishes, then exits (its death must not affect
    the sticky request)."""
    mailbox = proxies_dict.get(final_namespace)
    mailbox.put_document(PipelineIngressMessage(kind="document", doc_id="from-child"))
    mailbox.request_manual_retry(
        "req-child",
        PipelineIngressMessage(
            kind="rescan", retry_failed=True, request_id="req-child"
        ),
    )


@pytest.fixture
def multiprocess_share_data():
    """Real Manager-backed shared data (workers=2); one server per test."""
    finalize_share_data()
    initialize_share_data(workers=2)
    yield
    finalize_share_data()


async def test_multiprocess_cross_process_publish_and_sticky_survival(
    multiprocess_share_data,
):
    ingress = await get_pipeline_ingress("wsM")
    other_ws = await get_pipeline_ingress("wsOther")
    final_namespace = get_final_namespace("pipeline_ingress", "wsM")

    child = mp.get_context("spawn").Process(
        target=_child_publish,
        args=(shared_storage._pipeline_ingress_proxies, final_namespace),
    )
    child.start()
    await asyncio.to_thread(child.join, 30)
    assert child.exitcode == 0

    # Cross-workspace isolation: the sibling workspace saw nothing.
    assert not other_ws.has_work()

    assert await asyncio.to_thread(ingress.wait_for_documents, 5.0)
    assert [m.doc_id for m in ingress.drain_documents()] == ["from-child"]

    # The publisher process is dead; the sticky request lives in the server.
    assert ingress.peek_next_manual_retry().request_id == "req-child"
    assert ingress.counts()["manual_retries"] == 1

    # ACK + bounded-window replay guard, across proxies.
    assert ingress.ack_manual_retry("req-child") is True
    assert ingress.request_manual_retry("req-child", _manual("req-child")) is False

    # Same-workspace resolution from this process yields the same mailbox.
    again = await get_pipeline_ingress("wsM")
    assert again.counts()["terminal_manual_request_ids"] == 1


async def test_multiprocess_wait_isolation_and_cancelled_waiter_steals_nothing(
    multiprocess_share_data,
):
    ingress = await get_pipeline_ingress("wsM")

    # Predicate isolation: pending manual/auto must not satisfy a document wait.
    ingress.request_auto_rescan()
    ingress.request_manual_retry("r1", _manual("r1"))
    assert await asyncio.to_thread(ingress.wait_for_documents, 0.3) is False
    assert await asyncio.to_thread(ingress.wait_for_items, 0.3) is True

    # A cancelled waiter's orphaned thread only OBSERVES — it can never take
    # a message with it (wait_for_documents has no dequeue path).
    waiter = asyncio.create_task(asyncio.to_thread(ingress.wait_for_documents, 5.0))
    await asyncio.sleep(0.2)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    ingress.put_document(_doc("kept"))
    await asyncio.sleep(0.3)  # give the orphaned thread time to wake and exit
    assert [m.doc_id for m in ingress.drain_documents()] == ["kept"]

    # clear() tombstones the pending manual id server-side as well.
    ingress.clear()
    assert ingress.request_manual_retry("r1", _manual("r1")) is False
    assert not ingress.has_work()


async def test_multiprocess_finalize_keeps_discovery_table_intact(
    multiprocess_share_data,
):
    """Per-workspace finalize in multiprocess mode only clears the LOCAL
    proxy cache: popping the shared discovery entry while sibling workers
    still hold cached proxies would split-brain the workspace (the entry
    itself holds no refcount — Manager-container values unpickle
    ``manager_owned`` — so a later resolver would mint a brand-new mailbox
    while old workers keep using the orphaned one)."""
    ingress = await get_pipeline_ingress("wsM")
    ingress.request_manual_retry("r-sticky", _manual("r-sticky"))
    final_namespace = get_final_namespace("pipeline_ingress", "wsM")

    await finalize_pipeline_ingress("wsM")
    assert final_namespace in shared_storage._pipeline_ingress_proxies

    # Re-resolution reaches the SAME server-side mailbox: state preserved.
    again = await get_pipeline_ingress("wsM")
    assert again.peek_next_manual_retry().request_id == "r-sticky"
