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
   page with a live cursor means "call again", not "drained". A row the drain
   cannot advance is reported as a blocker and the reset is refused;
3. whatever else ever reaches a no-progress state, the drain detects it, fences
   the workspace with ``recovery_required`` plus a BOUNDED sample of the blocking
   doc ids, and stops.

(2) and (3) differ in one deliberate way: a blocker with a documented remedy
(``/documents/scan``) must NOT fence, because ``recovery_required`` refuses every
mutation including that scan; a blocker with no known remedy fences.
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
from lightrag.exceptions import (
    PipelineDrainBlockedError,
    PipelineRecoveryRequiredError,
)
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


def test_journaled_active_row_blocks_the_reset_without_fencing(tmp_path):
    """A journaled ACTIVE row means the pipeline is NOT idle, so the exclusive
    reset must not begin — and it must not spin either.

    Two failure modes bracket the correct behaviour:

    * the original bug swept the row, dropped it in the batch validator and left
      it PROCESSING, so the drain's confirmation found it again every round and
      spun (~3.4k rounds/second, no sleep, retry never served);
    * filtering it out of the confirmation sweep instead is fail-OPEN: the reset
      starts and ACKs while the persistent AUTO set is not empty, which is exactly
      what LR2 §4.2 forbids.

    So the drain reports the blocker, refuses the reset and exits. The retry stays
    OWED — the request is un-ACKed and the FAILED document is untouched — and the
    workspace is NOT fenced, because the remedy (`/documents/scan`'s rollback) is
    itself a mutation a fence would refuse.
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
            with pytest.raises(PipelineDrainBlockedError) as excinfo:
                await asyncio.wait_for(
                    rag.apipeline_process_enqueue_documents(),
                    timeout=_SPIN_TIMEOUT_SECONDS,
                )

            # The blocker is named, boundedly, and points at the remedy.
            assert excinfo.value.blocked_doc_ids == (journaled,)
            assert "/documents/scan" in str(excinfo.value)

            # The reset never ran: the FAILED doc is untouched and the request is
            # still queued, so its one retry per document is still owed.
            assert await _status_of(rag, failed) == DocStatus.FAILED.value
            ingress = await get_pipeline_ingress(rag.workspace)
            assert len(ingress.snapshot_manual_retries()) == 1

            # The journaled row is untouched — not reset, not deleted, journal
            # intact — so /documents/scan can still roll it back.
            after = await rag.doc_status.get_by_id(journaled)
            assert after == journaled_row_before

            # NOT fenced: uploads and — crucially — the scan that fixes this are
            # still allowed.
            status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            assert not status.get("recovery_required")
            # And the freeze/busy were released, so nothing is wedged.
            assert status.get("busy") is False
            assert status.get("manual_freeze_requested") is False
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_the_queued_retry_is_served_once_the_blocker_is_gone(tmp_path):
    """The blocked request self-heals: it stays queued, and the next trigger
    serves it as soon as the journal is gone (what a scan rollback leaves behind).

    This is the pay-off for refusing rather than fencing — no operator has to
    re-issue the retry, and no force-reset stands between them and the fix."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            journaled = await _seed_journaled_row(
                rag, "journaled.txt", DocStatus.PROCESSING
            )
            failed = await _seed_failed_with_content(rag, "retry-me.txt")

            await request_failed_retry(rag)
            with pytest.raises(PipelineDrainBlockedError):
                await asyncio.wait_for(
                    rag.apipeline_process_enqueue_documents(),
                    timeout=_SPIN_TIMEOUT_SECONDS,
                )

            # What a /documents/scan rollback leaves: journal gone, row resolved.
            await rag.doc_status.delete([journaled])

            # Same still-queued request, no new one published.
            await asyncio.wait_for(
                rag.apipeline_process_enqueue_documents(),
                timeout=_SPIN_TIMEOUT_SECONDS,
            )

            assert await _status_of(rag, failed) == DocStatus.PROCESSED.value
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []  # ACKed
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
