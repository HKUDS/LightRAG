"""Offline tests for the pipeline's internal-abort path (PR #3187).

Two layers:

* Pure helpers — ``_cancellation_label`` / ``_raise_if_cancelled`` /
  ``_cancellation_requested`` — drive the user-cancel vs internal-error
  distinction directly.
* End-to-end — enqueue a real document and drive
  ``apipeline_process_enqueue_documents`` with a storage flush forced to fail,
  asserting the *current* semantics (not idealized ones, per review):
    - the doc that triggers the flush error is FAILED with ``str(IndexFlushError)``
      and a "Merging stage failed" status message (NOT a cancellation label);
    - the finally cleanup surfaces an actionable "Pipeline halted on internal
      storage error" message (and makes it latest_message) on the normal break
      exit path, not just the generic "stopped" line;
    - ``_discard_pending_index_ops`` is the observable internal-abort signal;
    - the post-merge / pre-PROCESSED cancellation guard prevents an in-flight
      sibling document from being mis-marked PROCESSED (deterministic injection
      via a ``merge_nodes_and_edges`` wrapper, not a parallelism race);
    - ``_process_worker`` survives an unhandled per-doc error without wedging
      ``q_process.join()``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

import lightrag.pipeline as pipeline_module
from lightrag import LightRAG
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.exceptions import PipelineCancelledException
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.pipeline import _BatchRunContext
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

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


def _deterministic_chunking(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    return [
        {"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0},
        {"tokens": 1, "content": f"{content}::chunk2", "chunk_order_index": 1},
    ]


def _status_to_text(status: object) -> str:
    if isinstance(status, DocStatus):
        return status.value
    return str(status).replace("DocStatus.", "").lower()


async def _build_rag(tmp_path, *, max_parallel_insert: int = 1) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"abort-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_deterministic_chunking,
        max_parallel_insert=max_parallel_insert,
    )
    await rag.initialize_storages()
    return rag


def _make_status_doc(doc_id: str) -> DocProcessingStatus:
    now = datetime.now(timezone.utc).isoformat()
    return DocProcessingStatus(
        content_summary=f"summary-{doc_id}",
        content_length=10,
        file_path=f"{doc_id}.txt",
        status=DocStatus.PENDING,
        created_at=now,
        updated_at=now,
        track_id=None,
        content_hash=f"hash-{doc_id}",
    )


# ===========================================================================
# Pure helpers
# ===========================================================================


@pytest.mark.asyncio
async def test_cancellation_label_internal_with_detail(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        status = {
            "cancellation_reason": "internal_error",
            "cancellation_detail": "MilvusVectorDBStorage[entities]: boom",
        }
        assert rag._cancellation_label(status) == (
            "Cancelled by internal error: MilvusVectorDBStorage[entities]: boom"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_cancellation_label_internal_without_detail(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        status = {"cancellation_reason": "internal_error", "cancellation_detail": None}
        assert rag._cancellation_label(status) == (
            "Cancelled by internal error: unknown"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_cancellation_label_user(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        assert rag._cancellation_label({}) == "User cancelled"
        assert (
            rag._cancellation_label({"cancellation_reason": None}) == "User cancelled"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_raise_if_cancelled(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        lock = asyncio.Lock()
        # Not requested -> no raise.
        await rag._raise_if_cancelled({"cancellation_requested": False}, lock)
        # Requested -> PipelineCancelledException.
        with pytest.raises(PipelineCancelledException):
            await rag._raise_if_cancelled({"cancellation_requested": True}, lock)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_cancellation_requested_returns_bool(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        lock = asyncio.Lock()
        assert await rag._cancellation_requested({}, lock) is False
        assert (
            await rag._cancellation_requested({"cancellation_requested": True}, lock)
            is True
        )
    finally:
        await rag.finalize_storages()


# ===========================================================================
# e2e — IndexFlushError aborts the batch
# ===========================================================================


def _fail_flush(monkeypatch, storage):
    """Force a storage's index_done_callback to raise (simulating a flush
    failure) so _insert_done wraps it in IndexFlushError."""

    async def boom():
        raise RuntimeError("vdb flush boom")

    monkeypatch.setattr(storage, "index_done_callback", boom)


@pytest.mark.asyncio
async def test_index_flush_error_marks_failed_with_real_semantics(
    tmp_path, monkeypatch
):
    rag = await _build_rag(tmp_path)
    try:
        content = "internal abort document"
        file_path = "abort.txt"
        doc_id = compute_mdhash_id(file_path, prefix="doc-")
        await rag.apipeline_enqueue_documents(input=content, file_paths=file_path)

        _fail_flush(monkeypatch, rag.chunks_vdb)
        await rag.apipeline_process_enqueue_documents()

        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert doc_status is not None
        assert _status_to_text(doc_status["status"]) == "failed"
        # The triggering doc records str(IndexFlushError) — NOT a cancel label
        # (it goes through _finalize_doc_failure's non-cancel branch).
        assert "index flush failed" in doc_status["error_msg"]
        assert "Cancelled by internal error" not in doc_status["error_msg"]

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        history = "\n".join(pipeline_status.get("history_messages", []))
        assert "Merging stage failed in document" in history
        # The finally cleanup surfaces the actionable halt reason on the
        # normal break exit path (not just the generic "stopped" line), and
        # makes it the latest_message so it is what the user sees.
        assert "Pipeline halted on internal storage error" in history
        assert "Pipeline halted on internal storage error" in pipeline_status.get(
            "latest_message", ""
        )
        # Cancellation flags are reset by the finally block on the way out.
        assert pipeline_status.get("cancellation_requested") is False
        assert pipeline_status.get("cancellation_reason") is None
        assert pipeline_status.get("cancellation_detail") is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_run_pipeline_batch_discards_pending_on_internal_abort(
    tmp_path, monkeypatch
):
    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(
            input="discard document", file_paths="discard.txt"
        )
        _fail_flush(monkeypatch, rag.chunks_vdb)

        discard_calls = 0
        orig_discard = rag._discard_pending_index_ops

        async def spy_discard(*, skip_enqueue_owned=True):
            nonlocal discard_calls
            discard_calls += 1
            await orig_discard(skip_enqueue_owned=skip_enqueue_owned)

        monkeypatch.setattr(rag, "_discard_pending_index_ops", spy_discard)

        await rag.apipeline_process_enqueue_documents()

        # _run_pipeline_batch discards the shared buffers once on internal abort.
        assert discard_calls >= 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_cascade_guard_prevents_processed_after_sibling_abort(
    tmp_path, monkeypatch
):
    """The post-merge / pre-PROCESSED guard ([pipeline.py] _raise_if_cancelled)
    bails a doc out as cancelled when a sibling already aborted — so it is NOT
    mis-marked PROCESSED and _insert_done is NOT re-run on the torn-down buffer.

    Deterministic injection: wrap merge_nodes_and_edges so the abort flag is
    flipped right after merge completes (i.e. between the two guards), then
    assert the guard fires.
    """
    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(
            input="cascade document", file_paths="cascade.txt"
        )
        doc_id = compute_mdhash_id("cascade.txt", prefix="doc-")

        orig_merge = pipeline_module.merge_nodes_and_edges

        async def merge_then_abort(**kwargs):
            result = await orig_merge(**kwargs)
            status = kwargs["pipeline_status"]
            lock = kwargs["pipeline_status_lock"]
            async with lock:
                status["cancellation_requested"] = True
                status["cancellation_reason"] = "internal_error"
                status["cancellation_detail"] = "sibling abort"
            return result

        monkeypatch.setattr(pipeline_module, "merge_nodes_and_edges", merge_then_abort)

        insert_done_calls = 0
        orig_insert_done = rag._insert_done

        async def spy_insert_done(*a, **k):
            nonlocal insert_done_calls
            insert_done_calls += 1
            await orig_insert_done(*a, **k)

        monkeypatch.setattr(rag, "_insert_done", spy_insert_done)

        await rag.apipeline_process_enqueue_documents()

        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert _status_to_text(doc_status["status"]) == "failed"
        # The guard fired BEFORE the PROCESSED transition + _insert_done.
        assert insert_done_calls == 0

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        history = "\n".join(pipeline_status.get("history_messages", []))
        assert "Cancelled by internal error" in history
    finally:
        await rag.finalize_storages()


# ===========================================================================
# _process_worker resilience
# ===========================================================================


async def _make_ctx(rag: LightRAG) -> tuple[_BatchRunContext, dict, Any]:
    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status.clear()
    pipeline_status.update(
        {
            "busy": True,
            "history_messages": [],
            "latest_message": "",
            "cancellation_requested": False,
            "cancellation_reason": None,
            "cancellation_detail": None,
        }
    )
    ctx = _BatchRunContext(
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        semaphore=asyncio.Semaphore(2),
        total_files=1,
        q_native=asyncio.Queue(),
        q_mineru=asyncio.Queue(),
        q_docling=asyncio.Queue(),
        q_analyze=asyncio.Queue(),
        q_process=asyncio.Queue(),
    )
    return ctx, pipeline_status, pipeline_status_lock


@pytest.mark.asyncio
async def test_process_worker_survives_unhandled_error(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        ctx, status, _ = await _make_ctx(rag)

        async def boom(**kwargs):
            raise RuntimeError("worker boom")

        monkeypatch.setattr(rag, "process_single_document", boom)
        await ctx.q_process.put(("doc-1", _make_status_doc("doc-1"), {}))

        worker = asyncio.create_task(rag._process_worker(ctx))
        try:
            # join() returning proves the worker did NOT die — it drained the
            # item (task_done) instead of hanging the queue forever.
            await asyncio.wait_for(ctx.q_process.join(), timeout=2.0)
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

        assert status["cancellation_requested"] is True
        assert status["cancellation_reason"] == "internal_error"
        assert "process worker unhandled error" in status["cancellation_detail"]
    finally:
        await rag.finalize_storages()
