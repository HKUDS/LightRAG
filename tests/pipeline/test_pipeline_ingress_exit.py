"""has_work-only atomic exit + cancellation/recovery closure (Phase 3).

The quiescence decision now covers the document channel: a message resident in
the mailbox at a batch boundary keeps the run busy (CONTINUE_DOCUMENT) and the
refetch resolves it with the feeder's drain→strict-verify protocol — live docs
come back as the next batch, provably-stale notifications are compacted so a
has_work-only exit can neither livelock on them nor strand a doc behind a
released ``busy``.  Cancellation (user or internal error) exits BEFORE the
decision so the ingress is fully retained for the next explicit trigger, and
the custom-chunks exit handoff consults ``has_work()`` alongside the
transitional ``request_pending`` flag.

All on a real LightRAG with in-memory JSON storages (offline).
"""

import asyncio
from collections import Counter
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.kg.pipeline_ingress import PipelineIngressMessage
from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_namespace_lock,
    get_pipeline_ingress,
)
from lightrag.pipeline import PipelineNextDecision, PipelineNextStep
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id
from lightrag.utils_pipeline import CUSTOM_CHUNK_PATCH_METADATA_KEY

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


def _chunking(tokenizer, content, *args) -> list[dict]:
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


class _MarkerExtract:
    """Process-layer extraction that counts calls per document marker and can
    block whichever marker is armed until released."""

    def __init__(self):
        self.calls: Counter = Counter()
        self.block_marker: str | None = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, chunks, *args, **kwargs):
        content = " ".join(str(v.get("content", "")) for v in chunks.values())
        marker = content.split(" ", 1)[0] if content else ""
        self.calls[marker] += 1
        if self.block_marker and self.block_marker in content:
            self.started.set()
            await self.release.wait()
        return [({}, {}) for _ in chunks]


async def _build_rag(tmp_path, extract, *, max_parallel_insert: int = 1) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"exit-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_chunking,
        max_parallel_insert=max_parallel_insert,
    )
    await rag.initialize_storages()
    rag._process_extract_entities = extract
    return rag


def _status_text(row) -> str:
    status = row["status"]
    return getattr(status, "value", str(status)).replace("DocStatus.", "").lower()


async def _pipeline_ns(rag):
    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
    return status, lock


def _doc_msg(doc_id: str) -> PipelineIngressMessage:
    return PipelineIngressMessage(kind="document", doc_id=doc_id)


def _manual_msg(request_id: str) -> PipelineIngressMessage:
    return PipelineIngressMessage(
        kind="rescan", retry_failed=True, request_id=request_id
    )


async def test_decision_priority_and_consumption_semantics(tmp_path):
    """Priority is manual > auto > document > request_pending > release, and
    each step consumes ONLY its own signal: manual is peeked (nothing removed),
    auto is an atomic exchange, the document check never drains (the refetch
    does), request_pending is the transitional fallback, and RELEASED clears
    busy/busy_owner in the same critical section."""
    rag = await _build_rag(tmp_path, _MarkerExtract())
    try:
        status, lock = await _pipeline_ns(rag)
        ingress = await get_pipeline_ingress(rag.workspace)

        ingress.request_manual_retry("req-1", _manual_msg("req-1"))
        ingress.request_auto_rescan()
        ingress.put_document(_doc_msg("doc-x"))
        async with lock:
            status.update({"busy": True, "busy_owner": None, "request_pending": True})

        d1 = await rag._decide_pipeline_next_step(status, lock, ingress)
        assert d1.step is PipelineNextStep.CONTINUE_MANUAL
        assert d1.manual_request_id == "req-1"
        counts = ingress.counts()
        # Manual wins and consumes nothing: every other signal is intact.
        assert counts["manual_retries"] == 1
        assert counts["auto_rescan_pending"] is True
        assert counts["documents"] == 1
        assert status.get("request_pending") is True

        ingress.ack_manual_retry("req-1")
        d2 = await rag._decide_pipeline_next_step(status, lock, ingress)
        assert d2.step is PipelineNextStep.CONTINUE_AUTO
        counts = ingress.counts()
        assert counts["auto_rescan_pending"] is False  # atomically consumed
        assert counts["documents"] == 1  # untouched

        d3 = await rag._decide_pipeline_next_step(status, lock, ingress)
        assert d3.step is PipelineNextStep.CONTINUE_DOCUMENT
        # The decision only PEEKS the channel; draining belongs to the refetch.
        assert ingress.counts()["documents"] == 1
        assert status.get("request_pending") is True  # not consumed by document

        ingress.drain_documents()
        d4 = await rag._decide_pipeline_next_step(status, lock, ingress)
        assert d4.step is PipelineNextStep.CONTINUE_AUTO
        assert status.get("request_pending") is False  # transitional consumed

        d5 = await rag._decide_pipeline_next_step(status, lock, ingress)
        assert d5.step is PipelineNextStep.RELEASED
        assert status.get("busy") is False
        assert status.get("busy_owner") is None
    finally:
        await rag.finalize_storages()


async def test_document_published_at_quiescence_starts_next_batch(tmp_path):
    """Fix-proof (has_work-only exit): a document whose ONLY signal is its
    mailbox message — published after the batch's feeder stopped, with the
    transitional ``request_pending`` flag cleared to simulate the Phase 4
    world — must keep the run busy (CONTINUE_DOCUMENT) and be processed as the
    next batch.  Without the document-channel check the decision releases
    ``busy`` and the doc strands in PENDING with its message resident."""
    extract = _MarkerExtract()
    rag = await _build_rag(tmp_path, extract)
    try:
        status, lock = await _pipeline_ns(rag)
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        b_id = compute_mdhash_id("b.txt", prefix="doc-")

        orig_batch = rag._run_pipeline_batch
        injected = False

        async def batch_then_inject(to_process_docs, **kwargs):
            nonlocal injected
            await orig_batch(to_process_docs, **kwargs)
            if not injected:
                injected = True
                # Lands AFTER this batch's feeder was cancelled and BEFORE the
                # quiescence decision — the only window the feeder cannot see.
                await rag.apipeline_enqueue_documents(
                    input="BBBB body", file_paths="b.txt"
                )
                async with lock:
                    status["request_pending"] = False  # kill the dual-write mask

        rag._run_pipeline_batch = batch_then_inject

        await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), timeout=10)

        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "processed"
        assert extract.calls["BBBB"] == 1
        assert status.get("busy") is False
        ingress = await get_pipeline_ingress(rag.workspace)
        assert ingress.counts()["documents"] == 0  # message resolved, not resident
    finally:
        await rag.finalize_storages()


async def test_stale_document_message_is_compacted_and_released(tmp_path):
    """Fix-proof (no livelock, no stranding): a notification for a doc with no
    live doc_status row — an enqueue-only doc deleted before any run, or a
    notification that outlived its doc — is drained, proven stale by the
    complete strict scan, and dropped; the run then releases ``busy`` instead
    of looping CONTINUE_DOCUMENT forever on a message nothing else consumes."""
    rag = await _build_rag(tmp_path, _MarkerExtract())
    try:
        status, _lock = await _pipeline_ns(rag)
        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.put_document(_doc_msg("doc-stale-ghost"))

        await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), timeout=10)

        assert status.get("busy") is False
        assert ingress.counts()["documents"] == 0  # compacted, not resident
        assert ingress.has_work() is False
    finally:
        await rag.finalize_storages()


async def test_refetch_document_failure_republishes_and_arms_auto(tmp_path):
    """Compensation: the CONTINUE_DOCUMENT refetch drains destructively BEFORE
    its strict scan, so a scan failure must restore the drained messages to
    the mailbox and arm auto-rescan as the backstop before propagating —
    otherwise the drained notifications die with the exception."""
    rag = await _build_rag(tmp_path, _MarkerExtract())
    try:
        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.consume_auto_rescan()
        ingress.put_document(_doc_msg("doc-a"))
        ingress.put_document(_doc_msg("doc-b"))

        async def dead_scan(*args, **kwargs):
            raise ConnectionError("doc_status backend transient failure")

        rag.doc_status.get_docs_by_statuses = dead_scan

        decision = PipelineNextDecision(PipelineNextStep.CONTINUE_DOCUMENT)
        with pytest.raises(ConnectionError):
            await rag._refetch_for_decision(decision, ingress)

        republished = {m.doc_id for m in ingress.drain_documents()}
        assert republished == {"doc-a", "doc-b"}
        assert ingress.counts()["auto_rescan_pending"] is True
    finally:
        await rag.finalize_storages()


async def test_enqueue_publishes_documents_inside_status_lock(tmp_path, monkeypatch):
    """Fix-proof (atomic producer handoff): the document publish must happen
    INSIDE the pipeline_status critical section that sets ``request_pending``
    — the consumer's exit decision reads the mailbox and releases ``busy``
    under the same lock, so a publish outside it can land just after a
    quiescing run released, stranding an enqueue-only doc in an idle mailbox
    once Phase 4 removes the transitional flag.  Serialization is the lock's,
    flavor-independent; the probe records whether the enqueue task holds the
    pipeline_status namespace lock at the moment it publishes."""
    import lightrag.kg.shared_storage as shared_storage_module

    rag = await _build_rag(tmp_path, _MarkerExtract())
    try:
        held: dict = {}
        orig_aenter = shared_storage_module.NamespaceLock.__aenter__
        orig_aexit = shared_storage_module.NamespaceLock.__aexit__

        async def tracing_aenter(self):
            result = await orig_aenter(self)
            held.setdefault(asyncio.current_task(), set()).add(self._namespace)
            return result

        async def tracing_aexit(self, *args):
            held.get(asyncio.current_task(), set()).discard(self._namespace)
            return await orig_aexit(self, *args)

        monkeypatch.setattr(
            shared_storage_module.NamespaceLock, "__aenter__", tracing_aenter
        )
        monkeypatch.setattr(
            shared_storage_module.NamespaceLock, "__aexit__", tracing_aexit
        )

        ingress = await get_pipeline_ingress(rag.workspace)
        publish_lock_states: list[bool] = []
        published_ids: list[str] = []
        orig_put = ingress.put_documents

        def probing_put(msgs):
            namespaces = held.get(asyncio.current_task(), set())
            publish_lock_states.append("pipeline_status" in namespaces)
            published_ids.extend(m.doc_id for m in msgs)
            return orig_put(msgs)

        ingress.put_documents = probing_put

        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")

        a_id = compute_mdhash_id("a.txt", prefix="doc-")
        assert published_ids == [a_id]
        assert publish_lock_states == [True]
    finally:
        await rag.finalize_storages()


async def test_refetch_compensation_failure_does_not_mask_original_error(tmp_path):
    """When even the compensation RPCs fail (Manager outage), the ORIGINAL
    strict-scan failure must still propagate — the compensation error is
    logged, never raised over it.  The drained messages are lost to the
    outage, but their docs are PENDING rows recovered by the next trigger."""
    rag = await _build_rag(tmp_path, _MarkerExtract())
    try:
        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.put_document(_doc_msg("doc-a"))

        async def dead_scan(*args, **kwargs):
            raise ConnectionError("original strict-scan failure")

        def dead_put(msg):
            raise RuntimeError("manager down: put_document")

        def dead_arm():
            raise RuntimeError("manager down: request_auto_rescan")

        rag.doc_status.get_docs_by_statuses = dead_scan
        ingress.put_document = dead_put
        ingress.request_auto_rescan = dead_arm

        decision = PipelineNextDecision(PipelineNextStep.CONTINUE_DOCUMENT)
        with pytest.raises(ConnectionError, match="original strict-scan failure"):
            await rag._refetch_for_decision(decision, ingress)
    finally:
        await rag.finalize_storages()


async def test_user_cancel_preserves_ingress_and_stops_run(tmp_path):
    """Fix-proof (cancel checked BEFORE the decision): cancellation stops the
    whole run and leaves the ingress fully retained — the auto-rescan flag
    armed during the batch must survive to the next explicit trigger.  Without
    the post-batch cancellation check the quiescence decision consumes the
    flag (CONTINUE_AUTO), the loop-top handler then discards the refetch on
    its way out, and the wake-up is lost."""
    extract = _MarkerExtract()
    extract.block_marker = "AAAA"
    rag = await _build_rag(tmp_path, extract)
    try:
        status, lock = await _pipeline_ns(rag)
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")

        proc = asyncio.create_task(rag.apipeline_process_enqueue_documents())
        await asyncio.wait_for(extract.started.wait(), timeout=5)  # A mid-flight

        async with lock:
            status["cancellation_requested"] = True
        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.consume_auto_rescan()  # clear enqueue-era noise
        ingress.request_auto_rescan()  # the signal that must survive the cancel

        extract.release.set()
        await asyncio.wait_for(proc, timeout=10)

        assert status.get("busy") is False
        assert status.get("cancellation_requested") is False  # bookkeeping ran
        # No new epoch consumed the flag on the way out: it waits for the next
        # explicit trigger, ingress fully retained.
        assert ingress.counts()["auto_rescan_pending"] is True
    finally:
        extract.release.set()
        await rag.finalize_storages()


async def test_cancel_after_auto_decision_restores_consumed_signal(tmp_path):
    """Fix-proof (consume-then-discard closure): a cancel landing AFTER the
    decision consumed the auto-rescan flag but BEFORE its refetched docs enter
    a batch must not lose the signal — the loop-top cancel exit re-arms it, so
    the cancel leaves the ingress as if the run had stopped before consuming."""
    extract = _MarkerExtract()
    extract.block_marker = "AAAA"
    rag = await _build_rag(tmp_path, extract)
    try:
        status, lock = await _pipeline_ns(rag)
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        b_id = compute_mdhash_id("b.txt", prefix="doc-")
        ingress = await get_pipeline_ingress(rag.workspace)

        orig_batch = rag._run_pipeline_batch
        injected = False

        async def batch_then_arm_auto(to_process_docs, **kwargs):
            nonlocal injected
            await orig_batch(to_process_docs, **kwargs)
            if not injected:
                injected = True
                # A PENDING doc whose ONLY remaining signal is the auto flag.
                await rag.apipeline_enqueue_documents(
                    input="BBBB body", file_paths="b.txt"
                )
                ingress.drain_documents()
                async with lock:
                    status["request_pending"] = False
                ingress.request_auto_rescan()

        rag._run_pipeline_batch = batch_then_arm_auto

        orig_decide = rag._decide_pipeline_next_step
        cancelled_after = []

        async def decide_then_cancel(ps, ps_lock, ing):
            decision = await orig_decide(ps, ps_lock, ing)
            if decision.step is PipelineNextStep.CONTINUE_AUTO and not cancelled_after:
                cancelled_after.append(True)
                # Lands in the exact window: decision consumed the flag, the
                # refetched docs have not reached a batch yet.
                async with ps_lock:
                    ps["cancellation_requested"] = True
            return decision

        rag._decide_pipeline_next_step = decide_then_cancel

        extract.release.set()
        await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), timeout=10)

        assert cancelled_after  # the window was exercised
        assert status.get("busy") is False
        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "pending"
        # The consumed flag was restored on the cancel exit.
        assert ingress.counts()["auto_rescan_pending"] is True
    finally:
        extract.release.set()
        await rag.finalize_storages()


async def test_cancel_after_document_decision_restores_consumed_signal(tmp_path):
    """Same window for CONTINUE_DOCUMENT: the refetch destructively drained
    the doc's notification; a cancel before the batch takes the docs over must
    leave a recovery signal — the drained message is represented by its
    PENDING row and the restored auto-rescan flag (the canonical
    lost-notification signal, as for channel overflow)."""
    extract = _MarkerExtract()
    extract.block_marker = "AAAA"
    rag = await _build_rag(tmp_path, extract)
    try:
        status, lock = await _pipeline_ns(rag)
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        b_id = compute_mdhash_id("b.txt", prefix="doc-")
        ingress = await get_pipeline_ingress(rag.workspace)

        orig_batch = rag._run_pipeline_batch
        injected = False

        async def batch_then_inject(to_process_docs, **kwargs):
            nonlocal injected
            await orig_batch(to_process_docs, **kwargs)
            if not injected:
                injected = True
                # A PENDING doc whose ONLY signal is its resident message.
                await rag.apipeline_enqueue_documents(
                    input="BBBB body", file_paths="b.txt"
                )
                async with lock:
                    status["request_pending"] = False

        rag._run_pipeline_batch = batch_then_inject

        orig_decide = rag._decide_pipeline_next_step
        cancelled_after = []

        async def decide_then_cancel(ps, ps_lock, ing):
            decision = await orig_decide(ps, ps_lock, ing)
            if (
                decision.step is PipelineNextStep.CONTINUE_DOCUMENT
                and not cancelled_after
            ):
                cancelled_after.append(True)
                async with ps_lock:
                    ps["cancellation_requested"] = True
            return decision

        rag._decide_pipeline_next_step = decide_then_cancel

        extract.release.set()
        await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), timeout=10)

        assert cancelled_after
        assert status.get("busy") is False
        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "pending"
        # The message was drained by the refetch; the restored flag is the
        # recovery signal that survives the cancel.
        assert ingress.counts()["auto_rescan_pending"] is True
    finally:
        extract.release.set()
        await rag.finalize_storages()


async def test_validation_failure_after_document_decision_restores_signal(tmp_path):
    """Fix-proof (non-cancel escape closure): a transient validation failure
    landing AFTER a CONTINUE_DOCUMENT refetch destructively drained the doc's
    notification — but BEFORE a batch took the docs over — must not strand the
    doc: the finally's bookkeeping re-arms auto-rescan on ANY exit with an
    uncommitted consumption, not just on a cancel."""
    extract = _MarkerExtract()
    extract.block_marker = "AAAA"
    rag = await _build_rag(tmp_path, extract)
    try:
        status, lock = await _pipeline_ns(rag)
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        b_id = compute_mdhash_id("b.txt", prefix="doc-")
        ingress = await get_pipeline_ingress(rag.workspace)

        orig_batch = rag._run_pipeline_batch
        orig_validate = rag._validate_and_fix_document_consistency
        injected = False

        async def failing_validate(*args, **kwargs):
            raise ConnectionError("full_docs transient failure in validation")

        async def batch_then_inject(to_process_docs, **kwargs):
            nonlocal injected
            await orig_batch(to_process_docs, **kwargs)
            if not injected:
                injected = True
                # A PENDING doc whose ONLY signal is its resident message —
                # and a validator that fails once that message is drained.
                await rag.apipeline_enqueue_documents(
                    input="BBBB body", file_paths="b.txt"
                )
                async with lock:
                    status["request_pending"] = False
                rag._validate_and_fix_document_consistency = failing_validate

        rag._run_pipeline_batch = batch_then_inject

        extract.release.set()
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(
                rag.apipeline_process_enqueue_documents(), timeout=10
            )

        rag._validate_and_fix_document_consistency = orig_validate
        assert status.get("busy") is False
        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "pending"
        # The drained notification survives the escape as the restored flag.
        assert ingress.counts()["auto_rescan_pending"] is True
    finally:
        extract.release.set()
        await rag.finalize_storages()


async def test_validation_failure_after_auto_decision_restores_signal(tmp_path):
    """Same escape for CONTINUE_AUTO: the decision consumed the auto flag;
    a validation failure before the batch takes over must re-arm it."""
    extract = _MarkerExtract()
    extract.block_marker = "AAAA"
    rag = await _build_rag(tmp_path, extract)
    try:
        status, lock = await _pipeline_ns(rag)
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        b_id = compute_mdhash_id("b.txt", prefix="doc-")
        ingress = await get_pipeline_ingress(rag.workspace)

        orig_batch = rag._run_pipeline_batch
        orig_validate = rag._validate_and_fix_document_consistency
        injected = False

        async def failing_validate(*args, **kwargs):
            raise ConnectionError("full_docs transient failure in validation")

        async def batch_then_arm_auto(to_process_docs, **kwargs):
            nonlocal injected
            await orig_batch(to_process_docs, **kwargs)
            if not injected:
                injected = True
                await rag.apipeline_enqueue_documents(
                    input="BBBB body", file_paths="b.txt"
                )
                ingress.drain_documents()
                async with lock:
                    status["request_pending"] = False
                ingress.request_auto_rescan()
                rag._validate_and_fix_document_consistency = failing_validate

        rag._run_pipeline_batch = batch_then_arm_auto

        extract.release.set()
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(
                rag.apipeline_process_enqueue_documents(), timeout=10
            )

        rag._validate_and_fix_document_consistency = orig_validate
        assert status.get("busy") is False
        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "pending"
        assert ingress.counts()["auto_rescan_pending"] is True  # re-armed
    finally:
        extract.release.set()
        await rag.finalize_storages()


async def test_internal_error_halt_preserves_ingress_and_surfaces_reason(tmp_path):
    """An internal-error abort exits like a cancel — no new epoch, ingress
    retained — but surfaces the actionable halt message instead of a plain
    stop, so the operator knows processing halted on a storage error."""
    extract = _MarkerExtract()
    extract.block_marker = "AAAA"
    rag = await _build_rag(tmp_path, extract)
    try:
        status, lock = await _pipeline_ns(rag)
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")

        proc = asyncio.create_task(rag.apipeline_process_enqueue_documents())
        await asyncio.wait_for(extract.started.wait(), timeout=5)

        async with lock:
            status.update(
                {
                    "cancellation_requested": True,
                    "cancellation_reason": "internal_error",
                    "cancellation_detail": "storage boom sentinel",
                }
            )
        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.consume_auto_rescan()
        ingress.request_auto_rescan()

        extract.release.set()
        await asyncio.wait_for(proc, timeout=10)

        assert status.get("busy") is False
        assert status.get("cancellation_requested") is False
        assert ingress.counts()["auto_rescan_pending"] is True  # retained
        assert "Pipeline halted on internal storage error" in status.get(
            "latest_message", ""
        )
        assert "storage boom sentinel" in status.get("latest_message", "")
    finally:
        extract.release.set()
        await rag.finalize_storages()


async def test_journaled_pending_doc_does_not_hold_busy(tmp_path):
    """A journaled custom-chunk doc is owned by the scan/custom-chunk recovery
    flow: its PENDING row keeps being stripped by the consistency validator,
    so its resident notification must be compacted (the strict scan proves the
    row exists but the validator owns it) and the run must release ``busy`` —
    not bounce CONTINUE_DOCUMENT forever between the two."""
    rag = await _build_rag(tmp_path, _MarkerExtract())
    try:
        status, lock = await _pipeline_ns(rag)
        await rag.apipeline_enqueue_documents(input="JJJJ body", file_paths="j.txt")
        j_id = compute_mdhash_id("j.txt", prefix="doc-")

        row = await rag.doc_status.get_by_id(j_id)
        row["metadata"] = {
            **(row.get("metadata") or {}),
            CUSTOM_CHUNK_PATCH_METADATA_KEY: {"phase": "staged"},
        }
        await rag.doc_status.upsert({j_id: row})
        async with lock:
            status["request_pending"] = False  # isolate the document channel

        await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), timeout=10)

        assert status.get("busy") is False
        row = await rag.doc_status.get_by_id(j_id)
        assert _status_text(row) == "pending"  # untouched by the pipeline
        assert (row.get("metadata") or {}).get(CUSTOM_CHUNK_PATCH_METADATA_KEY) == {
            "phase": "staged"
        }
        ingress = await get_pipeline_ingress(rag.workspace)
        assert ingress.counts()["documents"] == 0
    finally:
        await rag.finalize_storages()


async def test_custom_chunks_exit_hands_off_on_ingress_only_work(tmp_path):
    """Fix-proof (handoff dual-check): a document enqueued while
    ``ainsert_custom_chunks`` holds ``busy`` may be visible ONLY through the
    ingress mailbox once the transitional flag is cleared.  The exit decision
    consults ``has_work()`` alongside ``request_pending`` and hands the slot
    to a processing run; flag-only (Phase 2) would release and strand the doc
    in PENDING behind an idle pipeline."""
    extract = _MarkerExtract()
    rag = await _build_rag(tmp_path, extract)
    try:
        status, lock = await _pipeline_ns(rag)
        b_id = compute_mdhash_id("b.txt", prefix="doc-")

        orig_cleanup = rag._insert_done_with_cleanup
        injected = False

        async def cleanup_then_inject():
            nonlocal injected
            if not injected:
                injected = True
                await rag.apipeline_enqueue_documents(
                    input="BBBB body", file_paths="b.txt"
                )
                async with lock:
                    status["request_pending"] = False  # kill the dual-write mask
            await orig_cleanup()

        rag._insert_done_with_cleanup = cleanup_then_inject

        await asyncio.wait_for(
            rag.ainsert_custom_chunks("base text", ["alice is here"], doc_id="doc-cc"),
            timeout=10,
        )

        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "processed"
        assert extract.calls["BBBB"] == 1
        assert status.get("busy") is False
    finally:
        await rag.finalize_storages()


async def test_custom_chunks_exit_hands_off_when_probe_fails(tmp_path):
    """A has_work probe failure fails TOWARD handoff: the driven run re-probes
    the mailbox itself, so a transient flake self-heals and the enqueued doc
    still processes — releasing instead would silently defer it (and any
    committed manual retry) to the next unrelated trigger."""
    extract = _MarkerExtract()
    rag = await _build_rag(tmp_path, extract)
    try:
        status, lock = await _pipeline_ns(rag)
        b_id = compute_mdhash_id("b.txt", prefix="doc-")
        ingress = await get_pipeline_ingress(rag.workspace)

        orig_cleanup = rag._insert_done_with_cleanup
        orig_has_work = ingress.has_work
        injected = False

        def flaky_has_work():
            ingress.has_work = orig_has_work  # one transient flake
            raise RuntimeError("manager down: has_work")

        async def cleanup_then_inject():
            nonlocal injected
            if not injected:
                injected = True
                await rag.apipeline_enqueue_documents(
                    input="BBBB body", file_paths="b.txt"
                )
                async with lock:
                    status["request_pending"] = False  # kill the dual-write mask
                ingress.has_work = flaky_has_work  # trip the exit probe
            await orig_cleanup()

        rag._insert_done_with_cleanup = cleanup_then_inject

        await asyncio.wait_for(
            rag.ainsert_custom_chunks("base text", ["alice is here"], doc_id="doc-cc"),
            timeout=10,
        )

        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "processed"
        assert status.get("busy") is False
    finally:
        await rag.finalize_storages()
