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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.exceptions import MultimodalAnalysisError, PipelineCancelledException
from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_namespace_lock,
    get_pipeline_ingress,
)
from lightrag.pipeline import _BatchRunContext
from lightrag.parser.exceptions import ParsePipelineCancelled
from lightrag.parser.llm_bridge import SyncLLMBridge
from lightrag.parser.registry import parser_specs_snapshot
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
        parse_queues={
            "native": asyncio.Queue(),
            "mineru": asyncio.Queue(),
            "docling": asyncio.Queue(),
        },
        parser_specs=parser_specs_snapshot(),
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
    timeout: float = 15.0,
) -> None:
    """Spin up the worker, await q.join(), then cancel the worker — same
    teardown sequence as ``_run_pipeline_batch``.

    The join is raced against the worker task: a worker that dies stops
    calling ``task_done()``, so waiting on the join alone would sit out the
    whole timeout and then report a bare ``TimeoutError`` instead of the
    worker's own exception. ``timeout`` is only a hang guard for a worker
    that stays alive but never drains; a passing run returns as soon as the
    queue is empty, so its size costs nothing."""
    worker = asyncio.create_task(worker_coro_factory())
    join_task = asyncio.create_task(queue.join())
    try:
        done, _ = await asyncio.wait(
            {worker, join_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if worker in done:
            # Re-raises the worker's exception; a clean return is still a bug
            # because the worker loop is supposed to run until cancelled.
            worker.result()
            raise AssertionError("worker exited before draining its queue")
        if join_task not in done:
            raise AssertionError(f"queue did not drain within {timeout}s hang guard")
    finally:
        for task in (join_task, worker):
            task.cancel()
        await asyncio.gather(join_task, worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_parse_worker_drains_queue_when_cancelled_before_start(
    tmp_path, monkeypatch
):
    """Cancellation set BEFORE the worker pulls any item: parser must not
    run, every queued doc is FAILED with a friendly message, and q.join()
    returns (bounded by the drain helper's hang guard, not a latency
    assertion)."""
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        ctx, pipeline_status, _ = await _make_ctx(rag)

        # The worker resolves its parser via the registry; if the boundary
        # cancellation check works, get_parser is never reached.
        get_parser_spy = Mock(side_effect=AssertionError("parser must not be resolved"))
        monkeypatch.setattr("lightrag.pipeline.get_parser", get_parser_spy)

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
            await ctx.parse_queues["native"].put((doc_id, _make_status_doc(doc_id)))

        pipeline_status["cancellation_requested"] = True

        await _run_worker_until_drained(
            lambda: rag._parse_worker("native", ctx.parse_queues["native"], ctx),
            ctx.parse_queues["native"],
        )

        assert get_parser_spy.call_count == 0

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
async def test_pipeline_cancel_interrupts_inflight_native_parser_llm(
    tmp_path, monkeypatch
):
    """The batch watcher must unblock a native parser waiting on an LLM."""
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        _ctx, pipeline_status, pipeline_status_lock = await _make_ctx(rag)
        doc_id = "doc-inflight-smart-heading"
        status_doc = _make_status_doc(doc_id)
        await rag.full_docs.upsert(
            {
                doc_id: {
                    "content": "source",
                    "file_path": status_doc.file_path,
                }
            }
        )
        await rag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PENDING.value,
                    "content_summary": status_doc.content_summary,
                    "content_length": status_doc.content_length,
                    "file_path": status_doc.file_path,
                    "created_at": status_doc.created_at,
                    "updated_at": status_doc.updated_at,
                    "track_id": "t",
                }
            }
        )

        submit_started = asyncio.Event()

        class _BlockingNativeParser:
            async def parse(self, parse_ctx):
                loop = asyncio.get_running_loop()

                async def _submit(_prompt, **_kwargs):
                    submit_started.set()
                    await asyncio.Future()

                bridge = SyncLLMBridge(
                    loop,
                    _submit,
                    cancel_events=(
                        (
                            parse_ctx.pipeline_cancel_event,
                            ParsePipelineCancelled,
                        ),
                    ),
                    poll_interval=0.02,
                )
                await asyncio.to_thread(bridge, "title block prompt")
                raise AssertionError("bridge cancellation should interrupt parse")

        monkeypatch.setattr(
            "lightrag.pipeline.get_parser", lambda *_a, **_k: _BlockingNativeParser()
        )
        batch = asyncio.create_task(
            rag._run_pipeline_batch(
                {doc_id: status_doc},
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                ingress=await get_pipeline_ingress(rag.workspace),
            )
        )
        # Both bounds are hang guards, not latency assertions: a green run
        # returns as soon as the event fires / the batch finishes. The first
        # wait races the batch so a batch that fails before reaching the LLM
        # surfaces its own exception instead of a timeout.
        started = asyncio.create_task(submit_started.wait())
        done, _ = await asyncio.wait(
            {started, batch}, timeout=15.0, return_when=asyncio.FIRST_COMPLETED
        )
        if started not in done:
            started.cancel()
            await asyncio.gather(started, return_exceptions=True)
            if batch in done:
                batch.result()
                raise AssertionError("batch finished before the parser LLM call")
            batch.cancel()
            await asyncio.gather(batch, return_exceptions=True)
            raise AssertionError("parser LLM call never started within hang guard")
        async with pipeline_status_lock:
            pipeline_status["cancellation_requested"] = True

        await asyncio.wait_for(batch, timeout=15.0)
        row = await rag.doc_status.get_by_id(doc_id)
        assert row is not None
        assert row["status"] == DocStatus.FAILED.value
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

        await _run_worker_until_drained(
            lambda: rag._analyze_worker(ctx),
            ctx.q_analyze,
        )

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
async def test_analyze_multimodal_inflight_cancellation_polls_flag(tmp_path):
    """User sets cancellation_requested while VLM tasks are running.
    analyze_multimodal must observe the flag through its poll loop while
    the VLM calls are still blocked -- not after they return -- cancel the
    item tasks, write the sidecar with the cancelled results, and raise
    PipelineCancelledException.

    The VLM never returns on its own, so noticing the flag only once a call
    finished cannot pass. How soon the poll loop reacts
    (``POLL_INTERVAL_SECONDS``) is deliberately not timed: a latency bound
    is what made this test flaky on loaded runners."""

    # Signals that a VLM call has actually started, i.e. analyze_multimodal
    # is past its pre-schedule cancellation check and the item tasks exist.
    vlm_inflight = asyncio.Event()
    # The VLM call blocks until the test releases it, after the cancellation
    # has been raised. vlm_finished records whether any call ever returned.
    release_vlm = asyncio.Event()
    vlm_finished = asyncio.Event()

    async def blocked_vlm(prompt, **kwargs):
        vlm_inflight.set()
        try:
            # Hang guard only: a poll loop that misses the flag must fail
            # the assertions below rather than stall the suite.
            await asyncio.wait_for(release_vlm.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pass
        vlm_finished.set()
        return json.dumps(
            {"name": "x", "type": "Chart", "description": "should not arrive"}
        )

    rag = _build_rag(tmp_path, vlm_func=blocked_vlm)
    await rag.initialize_storages()
    try:
        doc_id, parsed_data, sidecar_path = _write_three_item_sidecar(tmp_path)

        # The real _analyze_drawing closure runs and validates the image
        # bytes before calling the VLM, so give each item a minimal PNG.
        from .test_pipeline_analyze_multimodal import PNG_BYTES

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

        # Flip the flag off the first VLM call rather than off a wall-clock
        # delay. analyze_multimodal re-checks cancellation immediately BEFORE
        # spawning the item tasks, so a flag already set by then raises on
        # that pre-schedule path: no task ever runs and the sidecar is never
        # rewritten, which is a different code path than the in-flight one
        # this test covers. A fixed delay only wins that race on an idle
        # machine -- on a loaded CI runner the startup work outlasts it and
        # the test fails on the missing llm_analyze_result entries. Gating on
        # vlm_inflight makes "flag set while tasks are running" an ordering
        # guarantee instead of a timing bet.
        cancellation_requested = asyncio.Event()

        async def flip_when_inflight():
            await vlm_inflight.wait()
            async with pipeline_status_lock:
                pipeline_status["cancellation_requested"] = True
                cancellation_requested.set()

        flipper = asyncio.create_task(flip_when_inflight())

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
        # Never plain-await the flipper: if analyze_multimodal raised without
        # ever reaching the VLM, vlm_inflight stays clear and the wait would
        # hang the suite instead of failing the assertions below.
        flipper.cancel()
        await asyncio.gather(flipper, return_exceptions=True)

        # A raise with the flag never set means the pre-schedule check (or an
        # earlier boundary) fired instead -- not the in-flight path under test.
        assert cancellation_requested.is_set(), (
            "cancellation was never requested while VLM ran"
        )
        # The ordering check: the raise came while every VLM call was still
        # blocked, so the poll loop -- not a check after the call returned --
        # is what observed the flag.
        assert not vlm_finished.is_set(), (
            "a VLM call returned before cancellation was raised; the flag was "
            "observed after the call, not by the poll loop"
        )

        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        # Sidecar should have been written even though we raised, with every
        # interrupted item recorded as a cancelled failure.
        for letter in ("A", "B", "C"):
            item = payload["drawings"][f"im-{letter}"]
            assert "llm_analyze_result" in item
            result = item["llm_analyze_result"]
            assert result["status"] == "failure"
            assert result["message"] == "cancelled"
    finally:
        # The role wrapper does not propagate outer-future cancellation to
        # its priority-queue worker, so the in-flight call is still blocked.
        # Release it first, or the worker shutdown waits out the hang guard.
        release_vlm.set()
        await _shutdown_role_workers(rag)
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_analyze_multimodal_fail_fast_preserves_successes(tmp_path):
    """One item raises; one already completed; one waits for release.
    analyze_multimodal must not wait for the blocked item,
    must preserve the completed item's result in the sidecar, and must
    raise MultimodalAnalysisError (not PipelineCancelledException)."""
    from .test_pipeline_analyze_multimodal import PNG_BYTES

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
    # quickly (~0.1s), call 3 cannot finish until teardown releases it.
    # Fail-fast must cancel call 3 rather than wait. Order by call_count rather than
    # by item identifier because the VLM role wrapper does not surface
    # the item filename in its kwargs (only image_inputs bytes).
    call_count = {"n": 0}
    call_lock = asyncio.Lock()
    release_slow = asyncio.Event()
    slow_completed = asyncio.Event()

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
        await release_slow.wait()
        slow_completed.set()
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
        # The slow call cannot complete before teardown. A regression that
        # records cancellation but still waits for it hits the timeout above.
        assert not slow_completed.is_set(), "fail-fast waited for the slow task"

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
        # Cancelling the caller can leave the role worker running the VLM.
        # Release it before shutdown so teardown never waits on slow work.
        release_slow.set()
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
    from .test_pipeline_analyze_multimodal import PNG_BYTES

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
