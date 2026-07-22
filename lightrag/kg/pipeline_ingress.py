"""Workspace-scoped pipeline ingress: the notification channel that replaces
``request_pending``.

One ingress instance exists per workspace (see ``shared_storage.
get_pipeline_ingress``).  It carries three channels with three distinct
reliability grades — deliberately NOT a single destructive FIFO:

* **document** — ephemeral doc-id notifications published by enqueue after the
  ``doc_status`` row is persisted.  Messages may be lost or duplicated freely:
  ``doc_status`` is the source of truth and the next processing run's initial
  scan rebuilds anything dropped.  Consumers re-read storage by id and skip
  stale/terminal/duplicate ids.
* **auto rescan** — a single dirty flag.  Publishers that cannot name precise
  doc ids (busy-refused processing requests, partial doc-status commits,
  recovery paths) set it; the pipeline supervisor is the only consumer and
  resets it atomically via :meth:`consume_auto_rescan`.
* **manual retry** — sticky FIFO requests carrying user intent from
  ``/documents/scan`` and ``/documents/reprocess_failed``.  A request stays in
  the mailbox until explicitly ACKed *after* the FAILED→PENDING reset has been
  persisted, so a consumer crash at any point either leaves the request sticky
  (re-executed by the next run) or leaves the documents already PENDING
  (recovered by the automatic paths).  Terminal request ids (ACKED or
  CANCELLED_BY_CLEAR) are remembered in a bounded FIFO set so a delayed replay
  of the same id is refused — a *bounded-window* idempotency guarantee, not an
  indefinite exactly-once (ids are server-generated UUIDs, so eviction of very
  old ids is harmless in practice).

Waiting primitives are split by consumer role: a feeder that only routes
documents must wait on the document channel alone (``wait_for_documents`` /
:meth:`AsyncioPipelineIngress.get_document`); waiting on all three channels
(``wait_for_items``) is reserved for a supervisor that actually consumes the
control channels — otherwise a pending manual/auto entry keeps ``has_work()``
true and turns the wait into a busy loop.

This module is dependency-free on purpose: :mod:`lightrag.kg.shared_storage`
imports it to register :class:`PipelineIngressMailbox` on the LightRAG
``SyncManager`` subclass and to build the per-workspace registry.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from multiprocessing.managers import BaseProxy
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

# Terminal states for manual retry request ids.
MANUAL_RETRY_ACKED = "acked"
MANUAL_RETRY_CANCELLED_BY_CLEAR = "cancelled_by_clear"

# Capacity of the bounded terminal-request-id set (FIFO eviction, oldest id
# first).  Sizing note: ids enter this set only via explicit human actions
# (one per /scan or /reprocess_failed call), so 4096 covers any realistic
# replay window; after eviction an extremely old id could in theory be
# accepted again — the documented bounded-window limit of the idempotency
# guarantee.
TERMINAL_MANUAL_REQUEST_IDS_CAPACITY = 4096

# Hard ceiling for one mailbox wait RPC.  A Manager server runs each client
# connection on its own thread; a client SIGKILLed while a wait_for_* call is
# blocked server-side is NOT detected until the wait returns (the thread is
# parked on the condition, not in recv(), so the EOF path never fires).  A
# bounded wait caps that stranded window; waiters needing longer horizons
# simply loop.
MAX_MAILBOX_WAIT_SECONDS = 30.0


@dataclass(frozen=True)
class PipelineIngressMessage:
    """One ingress notification.

    ``document`` messages carry only the ``doc_id`` — never the status row
    itself — so consumers re-read storage and stay idempotent.  ``rescan``
    messages request a full status re-query: automatic ones
    (``retry_failed=False``) travel through the auto-rescan dirty flag, manual
    ones (``retry_failed=True``) through the sticky manual-retry channel keyed
    by ``request_id``.
    """

    kind: Literal["document", "rescan"]
    doc_id: str | None = None
    retry_failed: bool = False
    request_id: str | None = None


class _BoundedTerminalIds:
    """Insertion-ordered ``request_id -> terminal state`` map with FIFO eviction.

    NOT thread-safe on its own — every access happens under the owning
    container's lock (the mailbox condition, or the asyncio single-thread
    guarantee).
    """

    def __init__(self, capacity: int = TERMINAL_MANUAL_REQUEST_IDS_CAPACITY):
        self._capacity = max(1, int(capacity))
        self._ids: OrderedDict[str, str] = OrderedDict()

    def add(self, request_id: str, state: str) -> None:
        # Re-adding refreshes neither order nor state: the first terminal
        # state wins so a late ACK cannot overwrite CANCELLED_BY_CLEAR.
        if request_id in self._ids:
            return
        self._ids[request_id] = state
        while len(self._ids) > self._capacity:
            self._ids.popitem(last=False)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._ids

    def get(self, request_id: str) -> Optional[str]:
        return self._ids.get(request_id)

    def __len__(self) -> int:
        return len(self._ids)


def _validate_manual_request(request_id: str, msg: PipelineIngressMessage) -> None:
    if not request_id:
        raise ValueError("manual retry requires a non-empty request_id")
    if msg.kind != "rescan" or not msg.retry_failed or msg.request_id != request_id:
        raise ValueError(
            "manual retry message must be kind='rescan' with retry_failed=True "
            f"and a matching request_id (got {msg!r} for {request_id!r})"
        )


@runtime_checkable
class PipelineIngress(Protocol):
    """Surface shared by the single-process and Manager-backed implementations.

    Waiting primitives are intentionally NOT part of the protocol — they differ
    by mode (``await get_document()`` on :class:`AsyncioPipelineIngress`;
    ``wait_for_documents(timeout)`` bridged via ``asyncio.to_thread`` on the
    Manager proxy) and Phase 2's feeder branches on the mode anyway.
    """

    def put_document(self, msg: PipelineIngressMessage) -> None: ...

    def request_auto_rescan(self) -> None: ...

    def consume_auto_rescan(self) -> bool: ...

    def request_manual_retry(
        self, request_id: str, msg: PipelineIngressMessage
    ) -> bool: ...

    def drain_documents(
        self, limit: Optional[int] = None
    ) -> List[PipelineIngressMessage]: ...

    def has_work(self) -> bool: ...

    def peek_next_manual_retry(self) -> Optional[PipelineIngressMessage]: ...

    def snapshot_manual_retries(self) -> List[PipelineIngressMessage]: ...

    def ack_manual_retry(self, request_id: str) -> bool: ...

    def clear(self) -> None: ...

    def counts(self) -> Dict[str, Any]: ...


class PipelineIngressMailbox:
    """Server-side ingress for multi-process mode.

    The instance lives in the Manager SERVER process; workers only hold a
    :class:`_PipelineIngressMailboxProxy`, so it survives any worker's death
    (including SIGKILL) as long as the Manager server itself is alive.  Every
    proxy method is exactly one RPC and is atomic server-side under one
    ``threading.Condition``; only the two ``wait_for_*`` calls block, each
    occupying just its own client connection's server thread while sleeping on
    the condition (never while holding it across an RPC boundary).

    ``clear()`` is reserved for destructive operations: it tombstones every
    still-pending manual request id as ``CANCELLED_BY_CLEAR`` *before* wiping
    the three active channels, so a delayed replay of an already-accepted
    request cannot re-enter the mailbox.  The terminal-id set itself is never
    cleared here — only workspace teardown (dropping the whole mailbox) or
    FIFO capacity eviction retires entries.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._documents: deque[PipelineIngressMessage] = deque()
        self._auto_rescan_pending = False
        self._manual_retries: OrderedDict[str, PipelineIngressMessage] = OrderedDict()
        self._terminal_ids = _BoundedTerminalIds()

    # -- publishing -----------------------------------------------------

    def put_document(self, msg: PipelineIngressMessage) -> None:
        with self._cond:
            self._documents.append(msg)
            self._cond.notify_all()

    def request_auto_rescan(self) -> None:
        with self._cond:
            self._auto_rescan_pending = True
            self._cond.notify_all()

    def request_manual_retry(
        self, request_id: str, msg: PipelineIngressMessage
    ) -> bool:
        """Sticky-register a manual retry request; False iff refused.

        Refusal happens only for ids already in the terminal set (ACKED or
        CANCELLED_BY_CLEAR) — the bounded-window replay guard.  Re-requesting
        an id that is still pending is an idempotent no-op (True) and does not
        change its FIFO position.
        """
        _validate_manual_request(request_id, msg)
        with self._cond:
            if request_id in self._terminal_ids:
                return False
            if request_id not in self._manual_retries:
                self._manual_retries[request_id] = msg
                self._cond.notify_all()
            return True

    # -- consuming ------------------------------------------------------

    def consume_auto_rescan(self) -> bool:
        """Atomic exchange: return the dirty flag and reset it.

        The supervisor is the sole caller.  If the strict follow-up query
        fails, the caller MUST re-arm via :meth:`request_auto_rescan` before
        propagating, otherwise the request is lost.
        """
        with self._cond:
            pending = self._auto_rescan_pending
            self._auto_rescan_pending = False
            return pending

    def drain_documents(
        self, limit: Optional[int] = None
    ) -> List[PipelineIngressMessage]:
        with self._cond:
            if limit is None or limit >= len(self._documents):
                drained = list(self._documents)
                self._documents.clear()
            else:
                drained = [self._documents.popleft() for _ in range(max(0, limit))]
            return drained

    def peek_next_manual_retry(self) -> Optional[PipelineIngressMessage]:
        """Earliest pending manual request WITHOUT removal (FIFO).

        The consumption primitive: one request per quiescence cycle, ACKed only
        after its FAILED→PENDING reset is persisted.
        """
        with self._cond:
            for msg in self._manual_retries.values():
                return msg
            return None

    def snapshot_manual_retries(self) -> List[PipelineIngressMessage]:
        """Read-only FIFO copy of pending manual requests (diagnostics/tests)."""
        with self._cond:
            return list(self._manual_retries.values())

    def ack_manual_retry(self, request_id: str) -> bool:
        """Retire ``request_id`` as ACKED; idempotent, always True.

        Valid only after the reset-to-PENDING write for this request has been
        persisted.  Unknown ids are recorded as ACKED too — ids are
        server-generated UUIDs, so this only makes crash-retried ACKs
        idempotent and can never poison a future request.
        """
        with self._cond:
            self._manual_retries.pop(request_id, None)
            self._terminal_ids.add(request_id, MANUAL_RETRY_ACKED)
            return True

    def clear(self) -> None:
        with self._cond:
            for request_id in self._manual_retries:
                self._terminal_ids.add(request_id, MANUAL_RETRY_CANCELLED_BY_CLEAR)
            self._manual_retries.clear()
            self._documents.clear()
            self._auto_rescan_pending = False

    # -- observing ------------------------------------------------------

    def has_work(self) -> bool:
        with self._cond:
            return bool(
                self._documents or self._auto_rescan_pending or self._manual_retries
            )

    def counts(self) -> Dict[str, Any]:
        with self._cond:
            return {
                "documents": len(self._documents),
                "auto_rescan_pending": self._auto_rescan_pending,
                "manual_retries": len(self._manual_retries),
                "terminal_manual_request_ids": len(self._terminal_ids),
            }

    @staticmethod
    def _bounded_wait(timeout: float) -> float:
        """Clamp a wait to :data:`MAX_MAILBOX_WAIT_SECONDS`.

        ``timeout`` is required and bounded by design: an unbounded wait would
        park this connection's server thread forever if the waiting client is
        SIGKILLed (see :data:`MAX_MAILBOX_WAIT_SECONDS`).  Callers loop for
        longer horizons.
        """
        return min(max(0.0, float(timeout)), MAX_MAILBOX_WAIT_SECONDS)

    def wait_for_documents(self, timeout: float) -> bool:
        """Block until the DOCUMENT channel is non-empty; never dequeues.

        The feeder's only waiting primitive: its predicate deliberately
        ignores the control channels, so a pending manual/auto entry (owned by
        the supervisor) cannot wake — let alone busy-loop — the feeder.  A
        cancelled/timed-out waiter has observed nothing and can never steal a
        message.  ``timeout`` is required and clamped (see :meth:`_bounded_wait`).
        """
        with self._cond:
            return self._cond.wait_for(
                lambda: bool(self._documents), self._bounded_wait(timeout)
            )

    def wait_for_items(self, timeout: float) -> bool:
        """Block until ANY channel has work; supervisor/diagnostics only.

        Never hand this to a consumer that does not consume the control
        channels — their pending entries keep the predicate true and the wait
        degenerates into a busy loop.  ``timeout`` is required and clamped
        (see :meth:`_bounded_wait`).
        """
        with self._cond:
            return self._cond.wait_for(
                self._has_work_locked, self._bounded_wait(timeout)
            )

    def _has_work_locked(self) -> bool:
        return bool(
            self._documents or self._auto_rescan_pending or self._manual_retries
        )


class _PipelineIngressMailboxProxy(BaseProxy):
    """Explicit proxy for :class:`PipelineIngressMailbox`.

    ``BaseProxy`` has no dynamic ``__getattr__`` — declaring ``_exposed_``
    alone would NOT make the methods callable, so each one gets an explicit
    ``_callmethod`` wrapper (deterministic, unlike AutoProxy).  A feeder
    blocking in ``wait_for_documents`` should hold its own proxy instance:
    calls on one proxy serialize on its connection.
    """

    _exposed_ = (
        "put_document",
        "request_auto_rescan",
        "request_manual_retry",
        "consume_auto_rescan",
        "drain_documents",
        "peek_next_manual_retry",
        "snapshot_manual_retries",
        "ack_manual_retry",
        "clear",
        "has_work",
        "counts",
        "wait_for_documents",
        "wait_for_items",
    )

    def put_document(self, msg: PipelineIngressMessage) -> None:
        self._callmethod("put_document", (msg,))

    def request_auto_rescan(self) -> None:
        self._callmethod("request_auto_rescan")

    def request_manual_retry(
        self, request_id: str, msg: PipelineIngressMessage
    ) -> bool:
        return self._callmethod("request_manual_retry", (request_id, msg))

    def consume_auto_rescan(self) -> bool:
        return self._callmethod("consume_auto_rescan")

    def drain_documents(
        self, limit: Optional[int] = None
    ) -> List[PipelineIngressMessage]:
        return self._callmethod("drain_documents", (limit,))

    def peek_next_manual_retry(self) -> Optional[PipelineIngressMessage]:
        return self._callmethod("peek_next_manual_retry")

    def snapshot_manual_retries(self) -> List[PipelineIngressMessage]:
        return self._callmethod("snapshot_manual_retries")

    def ack_manual_retry(self, request_id: str) -> bool:
        return self._callmethod("ack_manual_retry", (request_id,))

    def clear(self) -> None:
        self._callmethod("clear")

    def has_work(self) -> bool:
        return self._callmethod("has_work")

    def counts(self) -> Dict[str, Any]:
        return self._callmethod("counts")

    def wait_for_documents(self, timeout: float) -> bool:
        return self._callmethod("wait_for_documents", (timeout,))

    def wait_for_items(self, timeout: float) -> bool:
        return self._callmethod("wait_for_items", (timeout,))


class AsyncioPipelineIngress:
    """Single-process ingress: pure asyncio, event-driven, zero polling.

    The document channel is a real ``asyncio.Queue`` (feeders simply
    ``await get_document()``); manual/auto state lives in plain containers.
    Every synchronous mutation happens on the owning event loop's thread, so
    ``work_event.set()/clear()`` need no await and no extra locking.  The
    instance is bound to the loop it was created on — cross-loop use is a
    bug the registry rejects via :attr:`owning_loop`.

    Unlike the Manager mailbox, a process-level SIGKILL loses this instance
    (and any sticky manual request in it) with the process — the documented
    single-process persistence boundary; ``doc_status`` plus the next run's
    initial scan recover the document backlog.
    """

    def __init__(self) -> None:
        self.owning_loop = asyncio.get_running_loop()
        self.document_messages: asyncio.Queue[PipelineIngressMessage] = asyncio.Queue()
        self._auto_rescan_pending = False
        self._manual_retries: OrderedDict[str, PipelineIngressMessage] = OrderedDict()
        self._terminal_ids = _BoundedTerminalIds()
        self.work_event = asyncio.Event()

    # -- publishing -----------------------------------------------------

    def put_document(self, msg: PipelineIngressMessage) -> None:
        self.document_messages.put_nowait(msg)
        self.work_event.set()

    def request_auto_rescan(self) -> None:
        self._auto_rescan_pending = True
        self.work_event.set()

    def request_manual_retry(
        self, request_id: str, msg: PipelineIngressMessage
    ) -> bool:
        _validate_manual_request(request_id, msg)
        if request_id in self._terminal_ids:
            return False
        if request_id not in self._manual_retries:
            self._manual_retries[request_id] = msg
            self.work_event.set()
        return True

    # -- consuming ------------------------------------------------------

    def consume_auto_rescan(self) -> bool:
        pending = self._auto_rescan_pending
        self._auto_rescan_pending = False
        self._maybe_clear_event()
        return pending

    async def get_document(self) -> PipelineIngressMessage:
        """Await the next document message (the single-process feeder wait).

        ``task_done()`` is paired immediately — no await separates the
        ``get()`` from it, so cancellation can never leave the queue's
        unfinished counter inflated.  A consumer that re-publishes the message
        creates a NEW queue item; this one is already accounted for.
        """
        msg = await self.document_messages.get()
        self.document_messages.task_done()
        self._maybe_clear_event()
        return msg

    def drain_documents(
        self, limit: Optional[int] = None
    ) -> List[PipelineIngressMessage]:
        drained: List[PipelineIngressMessage] = []
        while limit is None or len(drained) < limit:
            try:
                msg = self.document_messages.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.document_messages.task_done()
            drained.append(msg)
        self._maybe_clear_event()
        return drained

    def peek_next_manual_retry(self) -> Optional[PipelineIngressMessage]:
        for msg in self._manual_retries.values():
            return msg
        return None

    def snapshot_manual_retries(self) -> List[PipelineIngressMessage]:
        return list(self._manual_retries.values())

    def ack_manual_retry(self, request_id: str) -> bool:
        self._manual_retries.pop(request_id, None)
        self._terminal_ids.add(request_id, MANUAL_RETRY_ACKED)
        self._maybe_clear_event()
        return True

    def clear(self) -> None:
        for request_id in self._manual_retries:
            self._terminal_ids.add(request_id, MANUAL_RETRY_CANCELLED_BY_CLEAR)
        self._manual_retries.clear()
        self.drain_documents()
        self._auto_rescan_pending = False
        self._maybe_clear_event()

    # -- observing ------------------------------------------------------

    def has_work(self) -> bool:
        return bool(
            not self.document_messages.empty()
            or self._auto_rescan_pending
            or self._manual_retries
        )

    def counts(self) -> Dict[str, Any]:
        return {
            "documents": self.document_messages.qsize(),
            "auto_rescan_pending": self._auto_rescan_pending,
            "manual_retries": len(self._manual_retries),
            "terminal_manual_request_ids": len(self._terminal_ids),
        }

    async def wait_for_items(self) -> None:
        """Wait until ANY channel has work; supervisor/diagnostics only.

        Same caveat as the mailbox variant: a waiter that does not consume the
        control channels busy-loops on their pending entries.  Double-check
        after ``clear()`` — all state changes are same-loop synchronous, so no
        await separates the checks and a stale-set event cannot spin.
        """
        while not self.has_work():
            self.work_event.clear()
            if self.has_work():
                break
            await self.work_event.wait()

    def _maybe_clear_event(self) -> None:
        if not self.has_work():
            self.work_event.clear()
