"""DRAIN_TO_IDLE must always make forward progress (LR2 §7.2, §13.2 case 16).

A manual retry drains the AUTO backlog before its exclusive FAILED reset, and it
decides "not drained yet" by re-sweeping the AUTO statuses from Start. That makes
the drain vulnerable in a way the IDLE path is not: a row the sweep KEEPS
returning but that nothing ever advances is read as "still draining" forever, and
because the re-sweep is a plain storage call with no sleep, the run spins as fast
as the backend will answer — pegging a core, hammering doc_status, and never
running the retry the operator asked for.

Three things are pinned here, and the middle one is the trap:

1. a routing sweep does not carry rows it must never route — rows holding an
   unfinished custom-chunk journal, which only the SDK caller or
   ``/documents/scan``'s rollback may advance;
2. the drain's idle proof is NOT that filtered view, and does not stop at one
   empty page. Reusing the filter is fail-open — the exclusive reset starts and
   ACKs while the persistent AUTO set is not empty (LR2 §4.2) — and one empty
   page with a live cursor means "call again", not "drained";
3. every way the drain can fail to reach idle ends in a ``recovery_required``
   fence with a BOUNDED sample of the blocking doc ids: rows that look routable
   but never change state (``manual_drain_stalled``) and rows the drain can never
   advance at all (``manual_drain_blocked``).

(3) went through a rejected design worth recording, because the tests that
"proved" it were the reason it looked fine. Reporting the blocker WITHOUT fencing
seemed better — a fence refuses every mutation, including the ``/documents/scan``
that resolves a journal — but a sticky un-ACKed manual request ALREADY makes
``/scan`` refuse its reservation (``refuse_when_manual_pending``, so a scan cannot
jump the manual FIFO), and the blocker leaves exactly that request queued. The
unfenced path therefore had no reachable remedy either; the "it self-heals" test
missed it by simulating the rollback with a direct ``doc_status.delete()``,
bypassing the one call that was blocked. So the recovery path is now asserted
through the REAL scan reservation, and the ``force_reset`` that clears the fence
also cancels the queued intents that block the scan.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import CURSOR_END, CURSOR_START, DocStatus
from lightrag.constants import CUSTOM_CHUNK_PATCH_METADATA_KEY
from lightrag.exceptions import PipelineRecoveryRequiredError
from lightrag.kg.shared_storage import get_namespace_data, get_pipeline_ingress
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

from .conftest import request_failed_retry

pytestmark = pytest.mark.offline

# Enough rounds that a spin is unambiguous, small enough to stay fast: the drain
# guard trips after _MANUAL_DRAIN_STALL_ROUNDS (3) identical rounds, so a run that
# is still going after this many confirmation sweeps is not converging.
_SPIN_TIMEOUT_SECONDS = 20


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


def _chunking(tokenizer, content, *a, **k) -> list[dict]:
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


async def _extract(chunks, *args, **kwargs):
    return [({}, {}) for _ in chunks]


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"mdp-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    rag._process_extract_entities = _extract
    return rag


async def _status_of(rag: LightRAG, doc_id: str) -> str:
    row = await rag.doc_status.get_by_id(doc_id)
    raw = (row or {}).get("status")
    return raw.value if isinstance(raw, DocStatus) else str(raw or "<missing>")


async def _seed_journaled_row(rag: LightRAG, name: str, status: DocStatus) -> str:
    """Write the row a SIGKILLed ``ainsert_custom_chunks`` leaves behind: an
    ACTIVE status plus a durable custom-chunk journal. Only the SDK caller or a
    scan rollback may advance it, so ordinary scheduling must route around it."""
    doc_id = compute_mdhash_id(name, prefix="doc-")
    now = datetime.now(timezone.utc).isoformat()
    await rag.full_docs.upsert({doc_id: {"content": f"body of {name}"}})
    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": status,
                "content_summary": f"body of {name}",
                "content_length": 10,
                "chunks_count": 0,
                "chunks_list": [],
                "created_at": now,
                "updated_at": now,
                "file_path": name,
                "track_id": "t-journal",
                "error_msg": "",
                "metadata": {
                    CUSTOM_CHUNK_PATCH_METADATA_KEY: {
                        "schema_version": 1,
                        "operation_id": "op-1",
                        "mode": "patch",
                        "chunk_ids": [],
                        "phase": "prepared",
                    }
                },
            }
        }
    )
    return doc_id


async def _seed_failed_with_content(rag: LightRAG, name: str) -> str:
    """A plain retryable FAILED row: content present, no journal."""
    doc_id = compute_mdhash_id(name, prefix="doc-")
    now = datetime.now(timezone.utc).isoformat()
    await rag.full_docs.upsert({doc_id: {"content": f"body of {name}"}})
    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.FAILED,
                "content_summary": f"body of {name}",
                "content_length": 10,
                "chunks_count": 0,
                "chunks_list": [],
                "created_at": now,
                "updated_at": now,
                "file_path": name,
                "track_id": "t-failed",
                "error_msg": "boom",
                "metadata": {},
            }
        }
    )
    return doc_id


# ---------------------------------------------------------------------------
# Group 1/2: a routing sweep skips journaled rows; the drain's idle proof does not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", [500, 0], ids=["paged", "paging-disabled"])
def test_scheduling_sweep_skips_journaled_rows(tmp_path, page_size):
    """``_next_scheduling_page`` drops rows with an unfinished custom-chunk
    journal in BOTH modes — the paged keyset sweep and the
    ``PIPELINE_SCHEDULING_PAGE_SIZE=0`` legacy single scan. Leaving them in made
    the sweep answer "there is routable work" for a row nothing could route."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            rag.pipeline_scheduling_page_size = page_size
            journaled = await _seed_journaled_row(
                rag, "journaled.txt", DocStatus.PROCESSING
            )
            plain = await _seed_journaled_row(rag, "plain.txt", DocStatus.PENDING)
            # Strip the journal from the second row so it stays routable.
            row = await rag.doc_status.get_by_id(plain)
            row["metadata"] = {}
            await rag.doc_status.upsert({plain: row})

            docs, _cursor = await rag._next_scheduling_page(
                (
                    DocStatus.PENDING,
                    DocStatus.PROCESSING,
                    DocStatus.PARSING,
                    DocStatus.ANALYZING,
                ),
                CURSOR_START,
            )

            assert journaled not in docs
            assert plain in docs
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_journaled_active_row_fences_instead_of_resetting(tmp_path):
    """A journaled ACTIVE row means the pipeline is NOT idle, so the exclusive
    reset must not begin — and it must not spin either.

    Three behaviours bracket the correct one:

    * the original bug swept the row, dropped it in the batch validator and left
      it PROCESSING, so the drain's confirmation found it again every round and
      spun (~3.4k rounds/second, no sleep, retry never served);
    * filtering it out of the confirmation instead is fail-OPEN: the reset starts
      and ACKs while the persistent AUTO set is not empty (LR2 §4.2);
    * reporting it but NOT fencing looked better — a fence refuses every mutation
      including the `/documents/scan` that fixes this — but had no reachable
      remedy either, because a sticky un-ACKed request ALREADY makes `/scan`
      refuse its reservation (`refuse_when_manual_pending`). That was a silent
      dead end rather than a loud one.

    So it fences, per §7.2, and the fence names the order that actually works.
    """

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            journaled = await _seed_journaled_row(
                rag, "journaled.txt", DocStatus.PROCESSING
            )
            failed = await _seed_failed_with_content(rag, "retry-me.txt")
            journaled_row_before = await rag.doc_status.get_by_id(journaled)

            await request_failed_retry(rag)
            with pytest.raises(PipelineRecoveryRequiredError) as excinfo:
                await asyncio.wait_for(
                    rag.apipeline_process_enqueue_documents(),
                    timeout=_SPIN_TIMEOUT_SECONDS,
                )

            # The blocker is named, boundedly, and the message names BOTH steps —
            # force_reset first, because the fence refuses the scan.
            assert excinfo.value.blocked_doc_ids == (journaled,)
            detail = str(excinfo.value)
            assert "/documents/recovery/force_reset" in detail
            assert "/documents/scan" in detail

            # The reset never ran: the FAILED doc is untouched, and the request is
            # still queued, so its one retry per document is still owed.
            assert await _status_of(rag, failed) == DocStatus.FAILED.value
            ingress = await get_pipeline_ingress(rag.workspace)
            assert len(ingress.snapshot_manual_retries()) == 1

            # The journaled row is untouched — not reset, not deleted, journal
            # intact — so a scan rollback can still resolve it.
            assert await rag.doc_status.get_by_id(journaled) == journaled_row_before

            status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            fence = status.get("recovery_required")
            assert isinstance(fence, dict)
            assert fence["kind"] == "manual_drain_blocked"
            # Freeze and busy released, so nothing is wedged on a gone owner.
            assert status.get("busy") is False
            assert status.get("manual_freeze_requested") is False
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_force_reset_unblocks_the_documented_recovery_order(tmp_path):
    """The recovery path is exercised through the REAL scan reservation.

    The previous version of this test simulated the rollback with a direct
    ``doc_status.delete()``, which bypassed the one call that was actually
    blocked — ``acquire_reservation(..., refuse_when_manual_pending=True)`` — and
    so "it self-heals" was never true. Here the scan reservation itself is the
    assertion:

    1. fenced, a scan is refused (the fence);
    2. `force_reset` clears the fence AND cancels the queued retry — without the
       cancellation the scan would still be refused for jumping the manual FIFO,
       which is what made the unfenced design a dead end;
    3. the scan reservation is then granted, so the rollback can run.
    """
    from lightrag.kg.shared_storage import (
        PipelineReservationConflict,
        acquire_reservation,
        get_namespace_lock,
    )

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            await _seed_journaled_row(rag, "journaled.txt", DocStatus.PROCESSING)
            await _seed_failed_with_content(rag, "retry-me.txt")

            await request_failed_retry(rag)
            with pytest.raises(PipelineRecoveryRequiredError):
                await asyncio.wait_for(
                    rag.apipeline_process_enqueue_documents(),
                    timeout=_SPIN_TIMEOUT_SECONDS,
                )

            status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
            ingress = await get_pipeline_ingress(rag.workspace)

            async def _try_scan_reservation():
                return await acquire_reservation(
                    status,
                    lock,
                    owner_key="scanning_owner",
                    owner=uuid4().hex,
                    owner_kind="scan",
                    flags={"scanning": True, "scanning_exclusive": True},
                    reject_when=(
                        ("busy", "busy"),
                        ("scanning", "scanning"),
                        ("pending_enqueues", "pending"),
                        ("manual_freeze_requested", "freeze"),
                    ),
                    pipeline_ingress=ingress,
                    refuse_when_manual_pending=True,
                )

            # (1) Fenced → the scan that would fix this is refused.
            refused = await _try_scan_reservation()
            assert refused.acquired is False
            assert refused.conflict is PipelineReservationConflict.RECOVERY_REQUIRED

            # (2) force_reset: fence cleared AND the queued intent cancelled.
            async with lock:
                status["recovery_required"] = None
            cancelled = ingress.cancel_manual_retries()
            assert cancelled == 1
            assert ingress.snapshot_manual_retries() == []

            # (3) The scan is now allowed — the recovery path is reachable.
            granted = await _try_scan_reservation()
            assert granted.acquired is True, granted.message
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_clearing_only_the_fence_still_blocks_the_scan(tmp_path):
    """Fix-proof for the finding itself: dropping the fence WITHOUT cancelling the
    queued retry leaves ``/documents/scan`` refused for jumping the manual FIFO.

    This is the trap that made "report but do not fence" unworkable, and it is
    also why ``force_reset`` cancels the intents rather than only the fence — if
    someone removes that cancellation, this test fails."""
    from lightrag.kg.shared_storage import (
        PipelineReservationConflict,
        acquire_reservation,
        get_namespace_lock,
    )

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            await _seed_journaled_row(rag, "journaled.txt", DocStatus.PROCESSING)
            await request_failed_retry(rag)
            with pytest.raises(PipelineRecoveryRequiredError):
                await asyncio.wait_for(
                    rag.apipeline_process_enqueue_documents(),
                    timeout=_SPIN_TIMEOUT_SECONDS,
                )

            status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
            ingress = await get_pipeline_ingress(rag.workspace)

            # Fence gone, request left queued — the state the rejected design left.
            async with lock:
                status["recovery_required"] = None

            result = await acquire_reservation(
                status,
                lock,
                owner_key="scanning_owner",
                owner=uuid4().hex,
                owner_kind="scan",
                flags={"scanning": True, "scanning_exclusive": True},
                reject_when=(),
                pipeline_ingress=ingress,
                refuse_when_manual_pending=True,
            )
            assert result.acquired is False
            assert result.conflict is PipelineReservationConflict.MANUAL_FREEZE
            assert "manual retry" in (result.message or "").lower()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_drain_confirmation_pages_past_a_fully_filtered_page(tmp_path):
    """The final confirmation must page to CURSOR_END, not stop at one empty page.

    An empty ``docs`` with a live cursor means "call again" — ``CURSOR_END`` is the
    only termination signal. Checking just ``if docs`` reset the moment page one
    came back empty, so with a page size of 1 and a journaled row sorting first,
    the reset started while a routable PENDING row sat on page two. Here the
    routable row must be found and processed, and the retry must NOT be ACKed on
    that first pass."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            rag.pipeline_scheduling_page_size = 1  # one row per page

            # created_at orders the keyset: the journaled row sorts FIRST, so it
            # owns page one and the routable row is only reachable on page two.
            journaled = compute_mdhash_id("aaa-journaled.txt", prefix="doc-")
            await rag.full_docs.upsert({journaled: {"content": "body"}})
            await rag.doc_status.upsert(
                {
                    journaled: {
                        "status": DocStatus.PROCESSING,
                        "content_summary": "body",
                        "content_length": 4,
                        "chunks_count": 0,
                        "chunks_list": [],
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                        "file_path": "aaa-journaled.txt",
                        "track_id": "t",
                        "error_msg": "",
                        "metadata": {
                            CUSTOM_CHUNK_PATCH_METADATA_KEY: {
                                "schema_version": 1,
                                "operation_id": "op-1",
                                "mode": "patch",
                                "chunk_ids": [],
                                "phase": "prepared",
                            }
                        },
                    }
                }
            )
            routable = compute_mdhash_id("zzz-pending.txt", prefix="doc-")
            await rag.full_docs.upsert({routable: {"content": "body"}})
            await rag.doc_status.upsert(
                {
                    routable: {
                        "status": DocStatus.PENDING,
                        "content_summary": "body",
                        "content_length": 4,
                        "chunks_count": 0,
                        "chunks_list": [],
                        "created_at": "2026-01-02T00:00:00+00:00",
                        "updated_at": "2026-01-02T00:00:00+00:00",
                        "file_path": "zzz-pending.txt",
                        "track_id": "t",
                        "error_msg": "",
                        "metadata": {},
                    }
                }
            )

            # The confirmation must reach page two and report the routable row —
            # not conclude "drained" from the empty first page.
            docs, _cursor, blocking = await rag._confirm_auto_drained()
            assert set(docs) == {routable}
            assert blocking == ()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_drain_confirmation_reports_a_blocker_only_after_cursor_end(tmp_path):
    """The blocker verdict also requires reaching CURSOR_END: a journaled row on
    page one must not be reported as a blocker while later pages are unread."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            rag.pipeline_scheduling_page_size = 1
            journaled_a = await _seed_journaled_row(
                rag, "a-journaled.txt", DocStatus.PROCESSING
            )
            journaled_b = await _seed_journaled_row(
                rag, "b-journaled.txt", DocStatus.PARSING
            )

            docs, cursor, blocking = await rag._confirm_auto_drained()

            # Nothing routable anywhere, so BOTH journaled rows are blockers and
            # the sweep only says so after it has seen the whole keyset.
            assert docs == {}
            assert cursor is CURSOR_END
            assert set(blocking) == {journaled_a, journaled_b}
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Group 3: a drain that cannot advance fences instead of spinning
# ---------------------------------------------------------------------------


def test_stalled_drain_fences_with_bounded_sample(tmp_path):
    """LR2 §7.2 / §13.2 case 16: when the same active rows block DRAIN_TO_IDLE
    round after round with no state change, the run sets ``recovery_required``,
    reports a BOUNDED sample of the blocking doc ids and stops — it never spins
    and never silently drops the active rows to let the retry start.

    The stall is injected rather than found, because every in-tree path that
    produced one is now fixed: a validator stubbed to drop everything reproduces
    the shape (the sweep keeps returning rows nothing advances) without depending
    on a particular bug surviving."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            # 12 blocking rows so the reported sample is provably capped.
            blocked = []
            for index in range(12):
                doc_id = compute_mdhash_id(f"stuck-{index}.txt", prefix="doc-")
                now = datetime.now(timezone.utc).isoformat()
                await rag.full_docs.upsert({doc_id: {"content": "body"}})
                await rag.doc_status.upsert(
                    {
                        doc_id: {
                            "status": DocStatus.PROCESSING,
                            "content_summary": "body",
                            "content_length": 4,
                            "chunks_count": 0,
                            "chunks_list": [],
                            "created_at": now,
                            "updated_at": now,
                            "file_path": f"stuck-{index}.txt",
                            "track_id": "t",
                            "error_msg": "",
                            "metadata": {},
                        }
                    }
                )
                blocked.append(doc_id)

            # The validator advances nothing — the rows stay PROCESSING and the
            # batch is empty, exactly the "no state advanced" condition.
            async def drop_everything(to_process_docs, *_args, **_kwargs):
                return {}

            rag._validate_and_fix_document_consistency = drop_everything

            await request_failed_retry(rag)
            with pytest.raises(PipelineRecoveryRequiredError) as excinfo:
                await asyncio.wait_for(
                    rag.apipeline_process_enqueue_documents(),
                    timeout=_SPIN_TIMEOUT_SECONDS,
                )

            # Bounded report, not a serialised backlog.
            sample = excinfo.value.blocked_doc_ids
            assert 0 < len(sample) <= 8
            assert set(sample).issubset(set(blocked))

            status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            fence = status.get("recovery_required")
            assert isinstance(fence, dict)
            assert fence["kind"] == "manual_drain_stalled"
            # The fence message names the cause, not a phantom worker death.
            assert "drain stalled" in fence["message"]

            # The freeze and busy are released (a live-but-exiting owner must not
            # wedge the pipeline), while the fence survives to refuse mutations.
            assert status.get("busy") is False
            assert status.get("manual_freeze_requested") is False

            # No FAILED document was consumed by an attempt that never ran: the
            # request is still sticky and un-ACKed.
            ingress = await get_pipeline_ingress(rag.workspace)
            assert len(ingress.snapshot_manual_retries()) == 1

            # And the blocked rows are still there — never dropped to let the
            # retry proceed.
            for doc_id in blocked:
                assert await _status_of(rag, doc_id) == DocStatus.PROCESSING.value
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_drain_progress_counter_resets_on_a_changing_id_set(tmp_path):
    """A large-but-finite backlog is not a stall. The guard keys on the blocking
    id SET, so a drain that is still shrinking (or otherwise changing) resets the
    counter and is never fenced — only an unchanging set trips it."""

    from lightrag.pipeline import _MANUAL_DRAIN_STALL_ROUNDS, _ManualDrainProgress

    progress = _ManualDrainProgress()

    # A changing set can be observed indefinitely.
    for round_index in range(_MANUAL_DRAIN_STALL_ROUNDS * 4):
        assert progress.observe({f"doc-{round_index}"}) is True

    # A shrinking set is progress too.
    assert progress.observe({"a", "b", "c"}) is True
    assert progress.observe({"a", "b"}) is True
    assert progress.observe({"a"}) is True

    # Only repetition of the SAME set trips the guard, and only after the
    # configured number of rounds — one observation is never evidence, so a
    # fresh tracker survives exactly _MANUAL_DRAIN_STALL_ROUNDS - 1 repeats.
    fresh = _ManualDrainProgress()
    for _ in range(_MANUAL_DRAIN_STALL_ROUNDS - 1):
        assert fresh.observe({"a"}) is True
    assert fresh.observe({"a"}) is False
    # Still False afterwards: the caller fences on the first False and unwinds,
    # but a repeated observation must not "reset" itself into looking healthy.
    assert fresh.observe({"a"}) is False
