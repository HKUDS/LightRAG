"""Offline tests for /cancel_pipeline propagation into PARSE and ANALYZE.

Tests target the worker-level cancellation contract added alongside the
existing PROCESS-stage support:

* ``_parse_worker`` and ``_analyze_worker`` check ``cancellation_requested``
  at the top of every loop iteration, drain queued items as FAILED with a
  ``"User cancelled during {stage}: ..."`` ``error_msg``, and ``task_done()``
  each one so ``q.join()`` in ``_run_pipeline_batch`` returns.
* ``analyze_multimodal`` fails fast: the first item that raises (or a
  ``cancellation_requested`` flip observed by the poll loop) cancels every
  still-running sibling task, preserves already-completed item results in
  the sidecar, and re-raises the original exception type.

Tests construct ``_BatchRunContext`` and call worker methods directly to
avoid the cross-task races inherent in driving the full
``apipeline_process_enqueue_documents`` entry point.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.exceptions import MultimodalAnalysisError, PipelineCancelledException
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.pipeline import _BatchRunContext
from lightrag.utils import EmbeddingFunc, Tokenizer


pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 8)


async def _noop_llm(prompt, **kwargs):  # pragma: no cover - never invoked
    return ""


def _build_rag(tmp_path: Path, *, vlm_func=None) -> LightRAG:
    role_configs = {}
    for spec in ROLES:
        if spec.name == "vlm" and vlm_func is not None:
            role_configs[spec.name] = RoleLLMConfig(func=vlm_func)
        else:
            role_configs[spec.name] = RoleLLMConfig()
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"cancel-{tmp_path.name}",
        llm_model_func=vlm_func or _noop_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=1024,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        vlm_process_enable=True,
        role_llm_configs=role_configs,
    )


async def _shutdown_role_workers(rag: LightRAG) -> None:
    """Explicitly shut down each role wrapper's priority-queue workers.

    finalize_storages() only finalizes storages — it does NOT touch the
    per-role priority_limit worker pools. If a test triggered any role
    LLM calls whose worker is still in ``await asyncio.sleep(...)`` when
    pytest closes the function-scoped event loop, the leaked worker
    tasks raise "Task was destroyed but it is pending" / "Event loop is
    closed" and (worse, observed on macOS Python 3.12) prevent the
    pytest process from exiting cleanly. Call this before
    ``finalize_storages()`` to drain workers under a live loop first.
    """
    for func in rag.role_llm_funcs.values():
        try:
            await rag._shutdown_llm_wrapper(func)
        except Exception as exc:
            logging.getLogger("lightrag").warning(
                f"role worker shutdown raised during test teardown: {exc}"
            )


async def _make_ctx(rag: LightRAG) -> tuple[_BatchRunContext, dict, Any]:
    """Build a fresh _BatchRunContext bound to the RAG's workspace.

    The pipeline_status dict and lock come from the same shared_storage
    keyspace that production code uses, so worker reads of the
    cancellation flag observe whatever the test writes.
    """
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
        }
    )
    ctx = _BatchRunContext(
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        semaphore=asyncio.Semaphore(2),
        total_files=0,
        q_native=asyncio.Queue(),
        q_mineru=asyncio.Queue(),
        q_docling=asyncio.Queue(),
        q_analyze=asyncio.Queue(),
        q_process=asyncio.Queue(),
    )
    return ctx, pipeline_status, pipeline_status_lock


def _make_status_doc(doc_id: str) -> DocProcessingStatus:
    now = datetime.now(timezone.utc).isoformat()
    return DocProcessingStatus(
        content_summary=f"summary-{doc_id}",
        content_length=10,
        file_path=f"{doc_id}.pdf",
        status=DocStatus.PENDING,
        created_at=now,
        updated_at=now,
        track_id=None,
        content_hash=f"hash-{doc_id}",
    )


async def _run_worker_until_drained(
    worker_coro_factory,
    queue: asyncio.Queue,
    *,
    timeout: float = 2.0,
) -> None:
    """Spin up the worker, await q.join(), then cancel the worker — same
    teardown sequence as ``_run_pipeline_batch``."""
    worker = asyncio.create_task(worker_coro_factory())
    try:
        await asyncio.wait_for(queue.join(), timeout=timeout)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_parse_worker_drains_queue_when_cancelled_before_start(tmp_path):
    """Cancellation set BEFORE the worker pulls any item: parser must not
    run, every queued doc is FAILED with a friendly message, q.join()
    returns quickly."""
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        ctx, pipeline_status, _ = await _make_ctx(rag)

        rag.parse_native = AsyncMock(
            side_effect=AssertionError("parse_native must not be called")
        )

        for i in range(3):
            doc_id = f"doc-{i}"
            await rag.full_docs.upsert(
                {doc_id: {"content": "hello", "file_path": f"{doc_id}.pdf"}}
            )
            await rag.doc_status.upsert(
                {
                    doc_id: {
                        "status": DocStatus.PENDING.value,
                        "content_summary": f"sum-{doc_id}",
                        "content_length": 5,
                        "file_path": f"{doc_id}.pdf",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "track_id": "t",
                    }
                }
            )
            await ctx.q_native.put((doc_id, _make_status_doc(doc_id)))

        pipeline_status["cancellation_requested"] = True

        start = time.monotonic()
        await _run_worker_until_drained(
            lambda: rag._parse_worker("native", ctx.q_native, ctx),
            ctx.q_native,
        )
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"queue drain should be fast, took {elapsed:.2f}s"
        assert rag.parse_native.await_count == 0

        cancel_messages = [
            m
            for m in pipeline_status["history_messages"]
            if "User cancelled during parse" in m
        ]
        assert len(cancel_messages) == 3

        for i in range(3):
            doc_id = f"doc-{i}"
            row = await rag.doc_status.get_by_id(doc_id)
            assert row is not None
            assert row.get("status") == DocStatus.FAILED.value
            assert "User cancelled during parse" in (row.get("error_msg") or "")
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_analyze_worker_drains_queue_when_cancelled_before_start(tmp_path):
    """ANALYZE-worker symmetric to the PARSE test above."""
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        ctx, pipeline_status, _ = await _make_ctx(rag)

        rag.analyze_multimodal = AsyncMock(
            side_effect=AssertionError("analyze_multimodal must not be called")
        )

        for i in range(3):
            doc_id = f"doc-{i}"
            await rag.doc_status.upsert(
                {
                    doc_id: {
                        "status": DocStatus.ANALYZING.value,
                        "content_summary": f"sum-{doc_id}",
                        "content_length": 5,
                        "file_path": f"{doc_id}.pdf",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "track_id": "t",
                    }
                }
            )
            await ctx.q_analyze.put(
                (doc_id, _make_status_doc(doc_id), {"content": "x"})
            )

        pipeline_status["cancellation_requested"] = True

        start = time.monotonic()
        await _run_worker_until_drained(
            lambda: rag._analyze_worker(ctx),
            ctx.q_analyze,
        )
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"queue drain should be fast, took {elapsed:.2f}s"
        assert rag.analyze_multimodal.await_count == 0

        cancel_messages = [
            m
            for m in pipeline_status["history_messages"]
            if "User cancelled during analyze" in m
        ]
        assert len(cancel_messages) == 3

        for i in range(3):
            row = await rag.doc_status.get_by_id(f"doc-{i}")
            assert row is not None
            assert row.get("status") == DocStatus.FAILED.value
            assert "User cancelled during analyze" in (row.get("error_msg") or "")
    finally:
        await rag.finalize_storages()


# Drawing sidecar fixture used by both in-flight cancellation and fail-fast
# tests. Three items so we can have one slow / one fast-failing / one slow-
# successful task and observe partial-result preservation.
def _write_three_item_sidecar(tmp_path: Path) -> tuple[str, dict, Path]:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir(exist_ok=True)
    blocks_path = parsed_dir / "doc.blocks.jsonl"
    blocks_path.write_text(
        json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n",
        encoding="utf-8",
    )
    sidecar_path = parsed_dir / "doc.drawings.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "drawings": {
                    "im-A": {"caption": "A", "path": "ignored-A"},
                    "im-B": {"caption": "B", "path": "ignored-B"},
                    "im-C": {"caption": "C", "path": "ignored-C"},
                }
            }
        ),
        encoding="utf-8",
    )
    parsed_data = {"blocks_path": str(blocks_path)}
    return "doc-1", parsed_data, sidecar_path


@pytest.mark.asyncio
async def test_analyze_multimodal_inflight_cancellation_polls_flag(
    tmp_path, monkeypatch
):
    """User sets cancellation_requested while VLM tasks are running.
    analyze_multimodal should observe the flag at the next poll boundary
    (≤ 0.5s), cancel pending tasks, write the sidecar with partial
    results, and raise PipelineCancelledException."""

    async def slow_vlm(prompt, **kwargs):
        # 1.2s is short enough that even when the priority-queue worker
        # finishes the in-flight call after we've already raised (the
        # role wrapper does not propagate outer-future cancellation to
        # the worker), the post-analyze cleanup is bounded.
        await asyncio.sleep(1.2)
        return json.dumps(
            {"name": "x", "type": "Chart", "description": "should not arrive"}
        )

    rag = _build_rag(tmp_path, vlm_func=slow_vlm)
    await rag.initialize_storages()
    try:
        doc_id, parsed_data, sidecar_path = _write_three_item_sidecar(tmp_path)

        # Bypass image-bytes validation: _analyze_drawing normally reads
        # and validates the image file. Replace with a controlled mock so
        # the only async work is the (slow_vlm) call we manage above.
        async def fake_analyze_drawing(item_id, item, sidecar_dir):
            await slow_vlm("dummy")  # honors the cancellation timing
            return (
                {
                    "name": item_id,
                    "type": "Chart",
                    "description": "ok",
                    "status": "success",
                    "analyze_time": int(time.time()),
                },
                f"cache-{item_id}",
            )

        # analyze_multimodal defines _analyze_drawing as a local closure,
        # so we can't monkeypatch it directly. Instead patch the helper
        # it relies on (slow_vlm via the role wrapper); we accept the
        # closure's image pre-validation and supply a minimal PNG fixture.
        from tests.test_pipeline_analyze_multimodal import PNG_BYTES

        for letter in ("A", "B", "C"):
            (tmp_path / "parsed" / f"im-{letter}.png").write_bytes(PNG_BYTES)
        sidecar_path.write_text(
            json.dumps(
                {
                    "drawings": {
                        f"im-{letter}": {
                            "caption": letter,
                            "path": str(tmp_path / "parsed" / f"im-{letter}.png"),
                        }
                        for letter in ("A", "B", "C")
                    }
                }
            ),
            encoding="utf-8",
        )

        # Use plain dict + asyncio.Lock so the poll loop's lock
        # acquisition has no chance of contending with the real
        # NamespaceLock used during LightRAG initialization paths.
        pipeline_status: dict = {
            "busy": True,
            "history_messages": [],
            "latest_message": "",
            "cancellation_requested": False,
        }
        pipeline_status_lock = asyncio.Lock()

        async def flip_after(delay: float):
            await asyncio.sleep(delay)
            async with pipeline_status_lock:
                pipeline_status["cancellation_requested"] = True

        flipper = asyncio.create_task(flip_after(0.1))

        start = time.monotonic()
        with pytest.raises(PipelineCancelledException):
            await asyncio.wait_for(
                rag.analyze_multimodal(
                    doc_id=doc_id,
                    file_path="fixture.pdf",
                    parsed_data=parsed_data,
                    process_options="i",
                    pipeline_status=pipeline_status,
                    pipeline_status_lock=pipeline_status_lock,
                ),
                timeout=15.0,
            )
        elapsed = time.monotonic() - start
        await flipper

        # Must cancel well before the 1.2s sleep — poll interval is 0.5s.
        assert elapsed < 1.0, f"in-flight cancel took {elapsed:.2f}s (>1.0s)"

        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        # Sidecar should have been written even though we raised — every
        # item carries a llm_analyze_result entry (cancelled / failure).
        for letter in ("A", "B", "C"):
            item = payload["drawings"][f"im-{letter}"]
            assert "llm_analyze_result" in item
            assert item["llm_analyze_result"]["status"] in ("failure", "success")
    finally:
        await _shutdown_role_workers(rag)
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_analyze_multimodal_fail_fast_preserves_successes(tmp_path):
    """One item raises quickly; one already completed; one would have
    taken longer. analyze_multimodal must not wait for the slow item,
    must preserve the completed item's result in the sidecar, and must
    raise MultimodalAnalysisError (not PipelineCancelledException)."""
    from tests.test_pipeline_analyze_multimodal import PNG_BYTES

    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    for letter in ("A", "B", "C"):
        (parsed_dir / f"im-{letter}.png").write_bytes(PNG_BYTES)

    blocks_path = parsed_dir / "doc.blocks.jsonl"
    blocks_path.write_text(
        json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n",
        encoding="utf-8",
    )
    sidecar_path = parsed_dir / "doc.drawings.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "drawings": {
                    f"im-{letter}": {
                        "caption": letter,
                        "path": str(parsed_dir / f"im-{letter}.png"),
                    }
                    for letter in ("A", "B", "C")
                }
            }
        ),
        encoding="utf-8",
    )
    parsed_data = {"blocks_path": str(blocks_path)}

    # Per-call behaviour: call 1 succeeds quickly (~0.05s), call 2 fails
    # quickly (~0.1s), call 3 would take 5s — we want to prove fail-fast
    # cancels call 3 rather than wait. Ordering by call_count rather than
    # by item identifier because the VLM role wrapper does not surface
    # the item filename in its kwargs (only image_inputs bytes).
    call_count = {"n": 0}
    call_lock = asyncio.Lock()

    async def vlm_func(prompt, **kwargs):
        async with call_lock:
            call_count["n"] += 1
            seq = call_count["n"]
        if seq == 1:
            await asyncio.sleep(0.05)
            return json.dumps({"name": "first", "type": "Chart", "description": "ok"})
        if seq == 2:
            await asyncio.sleep(0.1)
            raise MultimodalAnalysisError("forced failure")
        # 1.2s instead of 5s: still proves fail-fast doesn't wait (test
        # checks elapsed < 0.8s) but keeps post-analyze cleanup bounded
        # since the worker keeps running this sleep until completion.
        await asyncio.sleep(1.2)
        return json.dumps({"name": "late", "type": "Chart", "description": "late"})

    rag = _build_rag(tmp_path, vlm_func=vlm_func)
    await rag.initialize_storages()
    try:
        pipeline_status: dict = {
            "busy": True,
            "history_messages": [],
            "latest_message": "",
            "cancellation_requested": False,
        }
        pipeline_status_lock = asyncio.Lock()

        start = time.monotonic()
        with pytest.raises(MultimodalAnalysisError):
            await asyncio.wait_for(
                rag.analyze_multimodal(
                    doc_id="doc-1",
                    file_path="fixture.pdf",
                    parsed_data=parsed_data,
                    process_options="i",
                    pipeline_status=pipeline_status,
                    pipeline_status_lock=pipeline_status_lock,
                ),
                timeout=15.0,
            )
        elapsed = time.monotonic() - start

        # Without fail-fast we'd have waited for the 1.2s sleep on the
        # third call. 0.8s gives the second-call failure path room
        # while still catching any regression that waits for call 3.
        assert elapsed < 0.8, f"fail-fast still waited {elapsed:.2f}s for slow task"

        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        statuses = sorted(
            payload["drawings"][f"im-{letter}"]["llm_analyze_result"]["status"]
            for letter in ("A", "B", "C")
        )
        # Three items → one success (call 1), one failure (call 2), and
        # one cancelled (call 3 was killed by fail-fast). All represented
        # as failure status_strings except for the success.
        assert statuses == ["failure", "failure", "success"]

        # Find which item ended up cancelled — its message must say so.
        cancelled_items = [
            r["message"]
            for r in (
                payload["drawings"][f"im-{letter}"]["llm_analyze_result"]
                for letter in ("A", "B", "C")
            )
            if r["status"] == "failure" and "cancelled" in r["message"]
        ]
        assert len(cancelled_items) == 1
        forced_items = [
            r["message"]
            for r in (
                payload["drawings"][f"im-{letter}"]["llm_analyze_result"]
                for letter in ("A", "B", "C")
            )
            if r["status"] == "failure" and "forced failure" in r["message"]
        ]
        assert len(forced_items) == 1
    finally:
        await _shutdown_role_workers(rag)
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_analyze_multimodal_pre_schedule_cancellation_skips_task_creation(
    tmp_path, monkeypatch
):
    """``cancellation_requested`` is already True when analyze_multimodal
    enters the sidecar processing loop. The pre-schedule check must
    raise immediately, before any per-item VLM task is even constructed
    — not merely cancel them before the scheduler yields. Covers the
    small window between ``_analyze_worker``'s boundary check and the
    per-sidecar task spawn that the polling loop alone would miss.

    Asserts both ``vlm_invocations == 0`` (no work executed) AND that
    ``asyncio.create_task`` was never called for any
    ``_run_with_progress_log`` coroutine — distinguishing the
    early-raise implementation from a poll-then-cancel implementation
    that would still construct and immediately cancel each task.
    """
    from tests.test_pipeline_analyze_multimodal import PNG_BYTES

    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    image_path = parsed_dir / "im-X.png"
    image_path.write_bytes(PNG_BYTES)
    blocks_path = parsed_dir / "doc.blocks.jsonl"
    blocks_path.write_text(
        json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n",
        encoding="utf-8",
    )
    sidecar_path = parsed_dir / "doc.drawings.json"
    sidecar_path.write_text(
        json.dumps({"drawings": {"im-X": {"caption": "X", "path": str(image_path)}}}),
        encoding="utf-8",
    )
    parsed_data = {"blocks_path": str(blocks_path)}

    vlm_invocations = 0

    async def tripwire_vlm(prompt, **kwargs):
        nonlocal vlm_invocations
        vlm_invocations += 1
        return json.dumps(
            {"name": "X", "type": "Chart", "description": "must not be called"}
        )

    # Spy on asyncio.create_task to count per-item tasks spawned by
    # analyze_multimodal. The per-item coroutine is _run_with_progress_log
    # (a closure defined inside analyze_multimodal), so filter by qualname.
    progress_log_tasks_created = 0
    original_create_task = asyncio.create_task

    def spy_create_task(coro, *args, **kwargs):
        nonlocal progress_log_tasks_created
        name = getattr(coro, "__qualname__", "") or getattr(
            getattr(coro, "cr_code", None), "co_qualname", ""
        )
        if "_run_with_progress_log" in name:
            progress_log_tasks_created += 1
        return original_create_task(coro, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", spy_create_task)

    rag = _build_rag(tmp_path, vlm_func=tripwire_vlm)
    await rag.initialize_storages()
    try:
        pipeline_status: dict = {
            "busy": True,
            "history_messages": [],
            "latest_message": "",
            "cancellation_requested": True,  # set BEFORE the call
        }
        pipeline_status_lock = asyncio.Lock()

        with pytest.raises(PipelineCancelledException):
            await rag.analyze_multimodal(
                doc_id="doc-1",
                file_path="fixture.pdf",
                parsed_data=parsed_data,
                process_options="i",
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
            )

        # Stronger than "no work ran": the per-item task object was
        # never even constructed. A poll-then-cancel implementation
        # would still spawn and cancel — this assertion rules that out.
        assert progress_log_tasks_created == 0
        assert vlm_invocations == 0
    finally:
        await _shutdown_role_workers(rag)
        await rag.finalize_storages()
