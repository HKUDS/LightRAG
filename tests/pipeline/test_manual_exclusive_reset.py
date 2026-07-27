"""Manual retry EXCLUSIVE_RESET protocol (LR2 Phase 3).

A manual retry no longer sweeps FAILED inline with workers. Instead it freezes
new ingress, drains the AUTO backlog to idle, then — with NO worker running —
pages FAILED→PENDING (preserving created_at), ACKs, clears the freeze and
processes the reset docs. These tests pin the Phase-3-specific observables that
the shared failed-retry semantics suite does not: the reset is PAGED (bounded),
the freeze rejects new enqueue reservations while draining, created_at is
preserved, and a FAILED stub without content is left FAILED.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import CURSOR_START, DocStatus
from lightrag.kg.shared_storage import (
    MANUAL_PHASE_DRAIN_TO_IDLE,
    PipelineReservationConflict,
    acquire_enqueue_reservation,
    get_namespace_data,
    get_namespace_lock,
    get_pipeline_ingress,
    make_owner_record,
)
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

from .conftest import request_failed_retry

pytestmark = pytest.mark.offline


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


class _CountingExtract:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    async def __call__(self, chunks, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in chunks]


async def _build_rag(tmp_path, extract: _CountingExtract) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"mxr-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    rag._process_extract_entities = extract
    return rag


async def _status_of(rag: LightRAG, doc_id: str) -> str:
    row = await rag.doc_status.get_by_id(doc_id)
    raw = (row or {}).get("status")
    return raw.value if isinstance(raw, DocStatus) else str(raw or "<missing>")


async def _make_failed(rag: LightRAG, name: str, extract: _CountingExtract) -> str:
    extract.fail = True
    await rag.apipeline_enqueue_documents(input=f"body of {name}", file_paths=name)
    await rag.apipeline_process_enqueue_documents()
    doc_id = compute_mdhash_id(name, prefix="doc-")
    assert await _status_of(rag, doc_id) == DocStatus.FAILED.value
    return doc_id


def test_manual_drains_auto_backlog_before_reset(tmp_path):
    """LR2 §13.2 case 5: a manual retry arriving with an AUTO backlog freezes
    new ingress, drains the AUTO (PENDING) backlog to idle FIRST, then resets
    FAILED→PENDING and processes those too — all in one run."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            # A pre-existing FAILED doc (the manual retry target).
            failed_id = await _make_failed(rag, "old.txt", extract)
            # An AUTO-backlog PENDING doc enqueued but not yet processed.
            extract.fail = False
            await rag.apipeline_enqueue_documents(
                input="fresh body", file_paths="new.txt"
            )
            pending_id = compute_mdhash_id("new.txt", prefix="doc-")
            assert await _status_of(rag, pending_id) == DocStatus.PENDING.value

            await request_failed_retry(rag)
            await rag.apipeline_process_enqueue_documents()

            # AUTO backlog drained AND the FAILED doc reset+reprocessed.
            assert await _status_of(rag, pending_id) == DocStatus.PROCESSED.value
            assert await _status_of(rag, failed_id) == DocStatus.PROCESSED.value
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []  # ACKed
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_refail_during_process_stays_failed(tmp_path):
    """LR2 §7.4 / §13.2 case 3: a doc the manual reset returns to PENDING that
    FAILS AGAIN during the PROCESS phase stays FAILED — the SAME request never
    resets it a second time (no retry loop)."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            failed_id = await _make_failed(rag, "a.txt", extract)
            calls_after_failure = extract.calls

            # Extraction still fails: the reset returns it to PENDING, it is
            # processed once, fails again → FAILED, and is NOT reset again by
            # the SAME request (no loop).
            await request_failed_retry(rag)
            await rag.apipeline_process_enqueue_documents()

            assert await _status_of(rag, failed_id) == DocStatus.FAILED.value
            # Exactly ONE extra attempt for this request (no retry loop).
            assert extract.calls == calls_after_failure + 1
            # A second run with no new signal must not retry again.
            await rag.apipeline_process_enqueue_documents()
            assert extract.calls == calls_after_failure + 1
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_reset_idempotent_across_mid_reset_failure(tmp_path):
    """LR2 §13.2 case 2 / §7.3 "为什么不需要 generation": a reset that fails on a
    later FAILED page leaves the already-reset rows PENDING; the next run
    re-runs the reset from Start and does NOT re-reset them (they are no longer
    on a FAILED page). Every doc is processed exactly once."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            rag.pipeline_scheduling_page_size = 1  # one FAILED per page
            id_a = await _make_failed(rag, "a.txt", extract)
            id_b = await _make_failed(rag, "b.txt", extract)
            calls_after_setup = extract.calls

            # Fail the SECOND FAILED page fetch of the reset (first page resets
            # one doc, second raises → run aborts, request stays sticky).
            original_page = rag.doc_status.get_docs_by_statuses_page
            state = {"failed_pages": 0, "armed": True}

            async def flaky(statuses, *, limit, position, strict=False):
                if list(statuses) == [DocStatus.FAILED]:
                    state["failed_pages"] += 1
                    if state["failed_pages"] == 2 and state["armed"]:
                        state["armed"] = False
                        raise RuntimeError("mid-reset page boom")
                return await original_page(
                    statuses, limit=limit, position=position, strict=strict
                )

            rag.doc_status.get_docs_by_statuses_page = flaky

            extract.fail = False
            await request_failed_retry(rag)
            with pytest.raises(RuntimeError, match="mid-reset page boom"):
                await rag.apipeline_process_enqueue_documents()

            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries()  # still sticky, not ACKed
            # Freeze cleared on the aborted exit (no wedge).
            status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            assert status["manual_freeze_requested"] is False

            # Next run completes: both docs processed exactly once, no double reset.
            await rag.apipeline_process_enqueue_documents()
            assert await _status_of(rag, id_a) == DocStatus.PROCESSED.value
            assert await _status_of(rag, id_b) == DocStatus.PROCESSED.value
            # Exactly two extra attempts (a + b, once each) — the already-reset
            # a.txt is not reset/processed a second time.
            assert extract.calls == calls_after_setup + 2
            assert ingress.snapshot_manual_retries() == []  # ACKed
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_exclusive_reset_pages_failed_backlog(tmp_path):
    """With paging on and a small page, the exclusive reset pages the FAILED
    backlog (never one giant scan) and every FAILED doc is reset+reprocessed."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            rag.pipeline_scheduling_page_size = 2
            ids = [await _make_failed(rag, f"f{i}.txt", extract) for i in range(5)]

            page_calls = {"failed": 0}
            original_page = rag.doc_status.get_docs_by_statuses_page

            async def counting_page(statuses, *, limit, position, strict=False):
                if list(statuses) == [DocStatus.FAILED]:
                    page_calls["failed"] += 1
                return await original_page(
                    statuses, limit=limit, position=position, strict=strict
                )

            rag.doc_status.get_docs_by_statuses_page = counting_page

            extract.fail = False
            await request_failed_retry(rag)
            await rag.apipeline_process_enqueue_documents()

            # Every FAILED doc was reset and reprocessed.
            for doc_id in ids:
                assert await _status_of(rag, doc_id) == DocStatus.PROCESSED.value
            # Paged: 5 docs / page 2 → at least 3 FAILED page fetches (2+2+1),
            # i.e. never a single unbounded scan.
            assert page_calls["failed"] >= 3
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_exclusive_reset_preserves_created_at(tmp_path):
    """FAILED→PENDING reset preserves the immutable created_at (LR2 §4.4/§7.3)."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed(rag, "keep.txt", extract)
            created_before = (await rag.doc_status.get_by_id(doc_id))["created_at"]

            extract.fail = False
            await request_failed_retry(rag)
            await rag.apipeline_process_enqueue_documents()

            row = await rag.doc_status.get_by_id(doc_id)
            assert row["status"] in (DocStatus.PROCESSED, DocStatus.PROCESSED.value)
            assert row["created_at"] == created_before
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_exclusive_reset_leaves_stub_without_content_failed(tmp_path):
    """A FAILED doc whose full_docs content is gone is unprocessable: the
    exclusive reset must leave it FAILED (matches the validator's preserve),
    never reset it into an immediate re-fail loop."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed(rag, "stub.txt", extract)
            # Simulate a content-less stub: drop the full_docs row.
            await rag.full_docs.delete([doc_id])

            extract.fail = False
            await request_failed_retry(rag)
            await rag.apipeline_process_enqueue_documents()

            # Left FAILED — never reset to PENDING (which would immediately
            # re-fail on the missing content).
            assert await _status_of(rag, doc_id) == DocStatus.FAILED.value
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_begin_manual_drain_freezes_new_enqueue_reservation(tmp_path):
    """While a manual drain holds the freeze, a NEW enqueue reservation is
    rejected with MANUAL_FREEZE (→ 409); clearing the freeze re-admits it."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
            token = "busy-tok"
            status.update(
                {"busy": True, "busy_owner": make_owner_record(token, "processing")}
            )

            assert await rag._begin_manual_drain("req-1", token, status, lock) is True
            assert status["manual_freeze_requested"] is True
            assert status["manual_phase"] == MANUAL_PHASE_DRAIN_TO_IDLE
            assert status["manual_owner"]["request_id"] == "req-1"
            assert status["manual_owner"]["owner_token"] == token

            reject_when = (("manual_freeze_requested", "manual retry draining"),)
            blocked = await acquire_enqueue_reservation(
                status, lock, token="up-1", reject_when=reject_when
            )
            assert blocked.acquired is False
            assert blocked.conflict is PipelineReservationConflict.MANUAL_FREEZE

            # Clearing the freeze re-admits enqueues.
            await rag._end_manual_drain(token, status, lock)
            assert status["manual_freeze_requested"] is False
            assert status["manual_owner"] is None
            admitted = await acquire_enqueue_reservation(
                status, lock, token="up-2", reject_when=reject_when
            )
            assert admitted.acquired is True
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_exclusive_reset_returns_false_when_not_owner(tmp_path):
    """LR2 §7.7 item 8 / §13.2 case 11: the exclusive reset is owner-checked. If
    the busy owner changed (a dead-owner reaper reclaim handed the slot to a new
    owner), a stale run's reset must not run — it returns False and touches
    nothing, leaving every FAILED row intact for the real owner to reset."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed(rag, "a.txt", extract)
            status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
            # The slot is owned by a DIFFERENT token than the stale run's.
            status.update(
                {"busy": True, "busy_owner": make_owner_record("owner-B", "processing")}
            )

            wrote = {"called": False}
            orig_upsert = rag.doc_status.upsert

            async def spy_upsert(data):
                wrote["called"] = True
                return await orig_upsert(data)

            rag.doc_status.upsert = spy_upsert

            ok = await rag._run_exclusive_failed_reset(
                "req-stale", "owner-A", status, lock
            )
            assert ok is False
            assert wrote["called"] is False  # no page write by the stale run
            # phase never advanced to EXCLUSIVE_RESET (the enter transition was
            # owner-checked and refused).
            assert status["manual_resetting"] is False
            assert await _status_of(rag, doc_id) == DocStatus.FAILED.value
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_reset_page_rejects_write_when_owner_changed(tmp_path):
    """_reset_failed_page raises (discards the late write) rather than persist a
    FAILED→PENDING page once the freeze owner no longer matches — so a reaper
    reclaim mid-reset can never be overwritten by the stale task."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            await _make_failed(rag, "a.txt", extract)
            status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
            status.update(
                {"busy": True, "busy_owner": make_owner_record("owner-B", "processing")}
            )

            docs, _ = await rag._next_failed_page(CURSOR_START)
            assert docs  # a real FAILED page to attempt

            wrote = {"called": False}
            orig_upsert = rag.doc_status.upsert

            async def spy_upsert(data):
                wrote["called"] = True
                return await orig_upsert(data)

            rag.doc_status.upsert = spy_upsert

            # token "owner-A" != current owner "owner-B" → reject before write.
            with pytest.raises(RuntimeError, match="lost freeze ownership"):
                await rag._reset_failed_page(docs, "owner-A", status, lock)
            assert wrote["called"] is False
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_next_failed_page_single_scan_when_paging_off(tmp_path):
    """page_size<=0 collapses the FAILED reset to one strict scan ending at
    CURSOR_END (legacy path); paging on returns a keyset cursor."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            await _make_failed(rag, "a.txt", extract)
            await _make_failed(rag, "b.txt", extract)

            rag.pipeline_scheduling_page_size = 0
            docs, cursor = await rag._next_failed_page(CURSOR_START)
            assert set(docs) and cursor is not None
            from lightrag.base import CURSOR_END

            assert cursor is CURSOR_END
            assert all(
                d.status in (DocStatus.FAILED, DocStatus.FAILED.value)
                for d in docs.values()
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_reset_page_propagates_a_strict_full_docs_failure(tmp_path):
    """Fix-proof: the content probe used the NON-strict ``get_by_id``, whose
    implementations may report a transport failure (or, on OpenSearch, an index
    that is merely not ready) as a best-effort ``None``. This loop read that as
    "unprocessable stub" and skipped the doc, so a storage failure could skip
    EVERY doc, still page to End, and ACK the manual retry having reset nothing.
    The strict read must propagate instead."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            await _make_failed(rag, "a.txt", extract)
            status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
            docs, _ = await rag._next_failed_page(CURSOR_START)
            assert docs

            assert rag.full_docs.supports_strict_point_reads is True

            async def _boom(doc_id):
                raise RuntimeError("full_docs transport failure")

            rag.full_docs.get_by_id_strict = _boom
            # A non-strict read that softens the same failure to None would make
            # this a silent skip; the strict path must surface it.
            rag.full_docs.get_by_id = _boom

            wrote = {"called": False}
            orig_upsert = rag.doc_status.upsert

            async def spy_upsert(data):
                wrote["called"] = True
                return await orig_upsert(data)

            rag.doc_status.upsert = spy_upsert

            with pytest.raises(RuntimeError, match="full_docs transport failure"):
                await rag._reset_failed_page(docs, "tok", status, lock)
            assert wrote["called"] is False
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_reset_page_counts_unconfirmed_misses_without_strict_reads(tmp_path):
    """A backend that cannot confirm absence keeps the conservative skip, but the
    skipped docs are surfaced so the operator does not read silence as success."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            await _make_failed(rag, "a.txt", extract)
            status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            status["history_messages"] = []
            lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
            docs, _ = await rag._next_failed_page(CURSOR_START)
            assert docs

            # Pretend the backend lacks strict point reads and misses the row.
            type(rag.full_docs).supports_strict_point_reads = False
            try:

                async def _miss(doc_id):
                    return None

                rag.full_docs.get_by_id = _miss
                reset = await rag._reset_failed_page(docs, "tok", status, lock)
            finally:
                type(rag.full_docs).supports_strict_point_reads = True

            assert reset == 0
            assert any(
                "cannot confirm whether their content is really absent" in m
                for m in status["history_messages"]
            ), status["history_messages"]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_worker_status_write_is_discarded_when_ownership_was_reclaimed(tmp_path):
    """LR2 §7.7 items 3/4/7: a run whose ownership was reclaimed must DISCARD its
    late status write, not stamp a final status over the new owner's work.

    Fix-proof: the worker write path did a plain ``doc_status.upsert`` with no
    owner check, so a late result from a run declared dead could re-FAIL a
    document the manual exclusive reset had just moved to PENDING — ACKing the
    retry with the document still failed.
    """

    async def _run():
        from lightrag.pipeline import _BatchRunContext
        from lightrag.parser.registry import parser_specs_snapshot

        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed(rag, "a.txt", extract)
            status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
            status["history_messages"] = []
            # Ownership now belongs to owner-B; our run still holds owner-A.
            status.update(
                {"busy": True, "busy_owner": make_owner_record("owner-B", "processing")}
            )

            ctx = _BatchRunContext(
                pipeline_status=status,
                pipeline_status_lock=lock,
                semaphore=asyncio.Semaphore(1),
                total_files=1,
                parse_queues={"native": asyncio.Queue()},
                parser_specs=parser_specs_snapshot(),
                q_analyze=asyncio.Queue(),
                q_process=asyncio.Queue(),
                run_owner_token="owner-A",
            )

            rows = await rag.doc_status.get_by_id(doc_id)
            status_doc = (await rag._next_failed_page(CURSOR_START))[0][doc_id]
            before = rows.get("status")

            wrote = {"called": False}
            orig_upsert = rag.doc_status.upsert

            async def spy_upsert(data):
                wrote["called"] = True
                return await orig_upsert(data)

            rag.doc_status.upsert = spy_upsert

            await rag._upsert_doc_status_transition(
                doc_id=doc_id,
                status=DocStatus.PROCESSED,
                status_doc=status_doc,
                file_path="a.txt",
                ctx=ctx,
            )

            # Nothing written, and the row is untouched.
            assert wrote["called"] is False
            after = (await rag.doc_status.get_by_id(doc_id)).get("status")
            assert after == before

            # The current owner's write still goes through.
            ctx.run_owner_token = "owner-B"
            await rag._upsert_doc_status_transition(
                doc_id=doc_id,
                status=DocStatus.PROCESSED,
                status_doc=status_doc,
                file_path="a.txt",
                ctx=ctx,
            )
            assert wrote["called"] is True
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_parsed_content_write_is_discarded_when_ownership_was_reclaimed(tmp_path):
    """The same §7.7 guarantee at the ONE write a worker makes outside the status
    chokepoint: ``_persist_parsed_full_docs``.

    Fix-proof: it wrote ``full_docs`` and patched ``doc_status.content_hash``
    with no owner check at all, so a parse result returning after a reaper
    reclaim would overwrite the body the NEW owner had re-parsed and stamp a row
    it now owns. It is reached as ``ctx.rag._persist_parsed_full_docs`` from
    inside a parser, so the check reads the run context the batch publishes on
    the instance — which is what makes third-party parsers guarded too.
    """

    async def _run():
        from lightrag.pipeline import _BatchRunContext
        from lightrag.parser.registry import parser_specs_snapshot

        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed(rag, "a.txt", extract)
            status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
            status.update(
                {"busy": True, "busy_owner": make_owner_record("owner-B", "processing")}
            )

            ctx = _BatchRunContext(
                pipeline_status=status,
                pipeline_status_lock=lock,
                semaphore=asyncio.Semaphore(1),
                total_files=1,
                parse_queues={"native": asyncio.Queue()},
                parser_specs=parser_specs_snapshot(),
                q_analyze=asyncio.Queue(),
                q_process=asyncio.Queue(),
                run_owner_token="owner-A",  # reclaimed: owner-B holds busy now
            )
            rag._active_run_ctx = ctx

            full_before = await rag.full_docs.get_by_id(doc_id)
            status_before = await rag.doc_status.get_by_id(doc_id)

            result = await rag._persist_parsed_full_docs(
                doc_id,
                {
                    "content": "text a stale run re-parsed",
                    "file_path": "a.txt",
                    "parse_format": "raw",
                },
            )

            # Neither store touched, and no content_hash handed back.
            assert result is None
            assert await rag.full_docs.get_by_id(doc_id) == full_before
            assert await rag.doc_status.get_by_id(doc_id) == status_before

            # The current owner's parse result still lands.
            ctx.run_owner_token = "owner-B"
            content_hash = await rag._persist_parsed_full_docs(
                doc_id,
                {
                    "content": "text the real owner parsed",
                    "file_path": "a.txt",
                    "parse_format": "raw",
                },
            )
            assert content_hash
            after = await rag.full_docs.get_by_id(doc_id)
            assert after["content"] == "text the real owner parsed"
            assert (await rag.doc_status.get_by_id(doc_id))[
                "content_hash"
            ] == content_hash
        finally:
            rag._active_run_ctx = None
            await rag.finalize_storages()

    asyncio.run(_run())


def test_persist_parsed_patches_content_hash_without_rewriting_the_row(tmp_path):
    """LR2 §5.6: ``_persist_parsed_full_docs`` syncs ``content_hash`` through the
    TARGETED field update, not a read-modify-write ``upsert``.

    Two scalars change. Round-tripping the whole row for them dragged this
    document's ``chunks_list`` through memory on every parse — the exact pattern
    the targeted-update API was added to replace. Pinned by asserting the call
    shape AND that the untouched fields survive."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = "doc-patch"
            chunks = [f"chunk-{index}" for index in range(50)]
            await rag.full_docs.upsert({doc_id: {"content": "old body"}})
            await rag.doc_status.upsert(
                {
                    doc_id: {
                        "status": DocStatus.PARSING,
                        "content_summary": "old body",
                        "content_length": 8,
                        "chunks_count": len(chunks),
                        "chunks_list": chunks,
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                        "file_path": "patch.txt",
                        "track_id": "t-patch",
                        "error_msg": "",
                        "metadata": {"keep": "me"},
                    }
                }
            )

            targeted: list[tuple[str, dict, bool]] = []
            full_upserts: list[dict] = []
            real_update = rag.doc_status.update_doc_status_fields
            real_upsert = rag.doc_status.upsert

            async def spy_update(target, fields, *, missing_ok=False):
                targeted.append((target, dict(fields), missing_ok))
                return await real_update(target, fields, missing_ok=missing_ok)

            async def spy_upsert(data):
                full_upserts.append(dict(data))
                return await real_upsert(data)

            rag.doc_status.update_doc_status_fields = spy_update
            rag.doc_status.upsert = spy_upsert

            content_hash = await rag._persist_parsed_full_docs(
                doc_id,
                {
                    "content": "freshly parsed body",
                    "file_path": "patch.txt",
                    "parse_format": "raw",
                },
            )

            # One targeted write, carrying ONLY the two fields that changed.
            assert len(targeted) == 1
            patched_id, fields, missing_ok = targeted[0]
            assert patched_id == doc_id
            assert set(fields) == {"content_hash", "updated_at"}
            assert fields["content_hash"] == content_hash
            # A row that vanished is not an error on this path.
            assert missing_ok is True
            # No full-row rewrite of doc_status at all.
            assert full_upserts == []

            # And nothing else on the row moved — chunks, metadata, created_at.
            row = await rag.doc_status.get_by_id(doc_id)
            assert row["chunks_list"] == chunks
            assert row["chunks_count"] == len(chunks)
            assert row["metadata"] == {"keep": "me"}
            assert row["created_at"] == "2026-01-01T00:00:00+00:00"
            assert row["content_hash"] == content_hash
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
