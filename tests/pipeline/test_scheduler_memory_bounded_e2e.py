"""Process-level bounded-memory acceptance for the pipeline and scan (LR2 §18.6).

The helper-level tracemalloc tests in ``test_scheduler_memory_bounded.py`` prove
that one scheduling/reset page is released before the next.  They cannot prove
that the composition is bounded: the supervisor, feeder, asyncio queues, worker
tasks or filesystem scan could retain one object per document even though every
helper is individually page-bounded.

These tests run each backlog size in a fresh spawned interpreter and sample that
process's resident set size (RSS):

* ``apipeline_process_enqueue_documents`` is the production entry point.  Its
  supervisor, in-batch feeder, parse workers, analyze workers and process workers
  are all real.  External parsing/KG work and persistent stores are bounded
  stubs so retained database rows do not drown out the scheduler signal.
* ``run_scanning_process`` walks a real directory through ``DocumentManager`` and
  the production classify/batch/enqueue lifecycle.  Its storage and final
  processing drive are non-retaining offline stubs.

The assertion is a growth comparison, never a machine-specific absolute RSS
budget.  A 10x backlog at fixed page/queue/batch sizes may pay allocator noise,
but it must not pay anything close to 10x transient memory.
"""

from __future__ import annotations

import asyncio
import gc
import importlib
import logging
import multiprocessing
import os
import sys
import threading
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any
from uuid import uuid4

import psutil
import pytest

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    DocProcessingStatus,
    DocSchedulingRecord,
    DocStatus,
    DocStatusPage,
    SourceAbsent,
)
from lightrag.constants import FULL_DOCS_FORMAT_RAW

pytestmark = pytest.mark.offline

_PIPELINE_PAGE_SIZE = 64
_QUEUE_SIZE = 4
_RSS_SAMPLE_SECONDS = 0.001
_RSS_FIXED_HEADROOM = 8 * 1024 * 1024
_BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _PeakRSS:
    """Sample this process's RSS from a constant-memory helper thread."""

    def __init__(self) -> None:
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline = 0
        self.peak = 0

    def start(self) -> None:
        gc.collect()
        self.baseline = self._process.memory_info().rss
        self.peak = self.baseline

        def _sample() -> None:
            while not self._stop.wait(_RSS_SAMPLE_SECONDS):
                self.peak = max(self.peak, self._process.memory_info().rss)

        self._thread = threading.Thread(
            target=_sample, name="rss-acceptance-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> dict[str, int]:
        self.peak = max(self.peak, self._process.memory_info().rss)
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=5)
        assert not self._thread.is_alive()
        return {
            "baseline": self.baseline,
            "peak": self.peak,
            "growth": max(0, self.peak - self.baseline),
        }


class _StreamingPipelineDocStatus:
    """Generate scheduling rows by cursor and retain only one bit per document.

    A JSON fake would itself keep every status row and every terminal update in
    the worker process, making RSS correctly grow with the *storage* rather than
    revealing whether the scheduler grows.  This fake models an external indexed
    backend: pages are generated on demand, while a preallocated bytearray records
    which rows reached a terminal state without allocating during the measurement.
    """

    supports_strict_point_reads = True

    def __init__(self, total: int) -> None:
        self.total = total
        self._terminal = bytearray(total)
        self.completed = 0
        self.page_calls = 0
        self.feeder_hydrations = 0
        self.parsing_writes = 0
        self.analyzing_writes = 0
        self.processed_writes = 0

    @staticmethod
    def _index(doc_id: str) -> int:
        return int(doc_id.rsplit("-", 1)[1])

    @staticmethod
    def _doc_id(index: int) -> str:
        return f"doc-{index:09d}"

    @staticmethod
    def _created_at(index: int) -> str:
        return (_BASE_TIME + timedelta(microseconds=index)).isoformat()

    def _status_doc(self, index: int) -> DocProcessingStatus:
        created_at = self._created_at(index)
        return DocProcessingStatus(
            # Match the production summary ceiling closely enough that an
            # accidental backlog-sized accumulator has a measurable RSS slope.
            content_summary=f"rss-{index:09d}-" + ("s" * 230),
            content_length=32,
            status=(
                DocStatus.PROCESSED if self._terminal[index] else DocStatus.PENDING
            ),
            created_at=created_at,
            updated_at=created_at,
            file_path=f"file-{index:09d}.txt",
            track_id="rss",
            content_hash=f"hash-{index:09d}",
            chunks_count=0,
            chunks_list=[],
            metadata={},
        )

    async def get_docs_by_statuses_page(
        self,
        statuses,
        *,
        limit: int,
        position=CURSOR_START,
        strict: bool = False,
    ) -> DocStatusPage:
        del strict
        wanted = {
            value if isinstance(value, DocStatus) else DocStatus(value)
            for value in statuses
        }
        if DocStatus.PENDING not in wanted:
            return DocStatusPage(docs={}, next_position=CURSOR_END)

        index = int(position.opaque) + 1 if isinstance(position, CursorAfter) else 0
        docs: dict[str, DocSchedulingRecord] = {}
        # Leave exactly one slot in the first epoch for a pre-published document
        # message. The production feeder must hydrate and admit that document,
        # proving this is not merely a supervisor/worker test with an idle feeder.
        page_limit = limit - 1 if self.page_calls == 0 else limit
        while index < self.total and len(docs) < page_limit:
            if not self._terminal[index]:
                doc_id = self._doc_id(index)
                created_at = self._created_at(index)
                docs[doc_id] = DocSchedulingRecord(
                    id=doc_id,
                    status=DocStatus.PENDING,
                    created_at=created_at,
                    updated_at=created_at,
                    file_path=f"file-{index:09d}.txt",
                    track_id="rss",
                    has_custom_chunk_journal=False,
                )
            index += 1
        self.page_calls += 1
        next_position = (
            CURSOR_END if index >= self.total else CursorAfter(str(index - 1))
        )
        return DocStatusPage(docs=docs, next_position=next_position)

    async def get_full_docs_by_ids(
        self, doc_ids, *, strict: bool = False
    ) -> dict[str, DocProcessingStatus]:
        del strict
        doc_ids = list(doc_ids)
        feeder_doc_id = self._doc_id(_PIPELINE_PAGE_SIZE - 1)
        if doc_ids == [feeder_doc_id]:
            self.feeder_hydrations += 1
        rows = {}
        for doc_id in doc_ids:
            index = self._index(doc_id)
            if not self._terminal[index]:
                rows[doc_id] = self._status_doc(index)
        return rows

    async def get_by_id(self, doc_id: str) -> dict[str, Any] | None:
        index = self._index(doc_id)
        if self._terminal[index]:
            return None
        row = self._status_doc(index)
        return {
            "status": row.status,
            "content_summary": row.content_summary,
            "content_length": row.content_length,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "file_path": row.file_path,
            "track_id": row.track_id,
            "content_hash": row.content_hash,
            "chunks_count": 0,
            "chunks_list": [],
            "metadata": {},
        }

    async def get_doc_by_content_hash(
        self, content_hash: str, *, exclude_doc_id: str | None = None
    ) -> None:
        del content_hash, exclude_doc_id
        return None

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        for doc_id, payload in data.items():
            status = payload.get("status")
            status = status if isinstance(status, DocStatus) else DocStatus(status)
            if status is DocStatus.PARSING:
                self.parsing_writes += 1
            elif status is DocStatus.ANALYZING:
                self.analyzing_writes += 1
            elif status is DocStatus.PROCESSED:
                self.processed_writes += 1
                index = self._index(doc_id)
                if not self._terminal[index]:
                    self._terminal[index] = 1
                    self.completed += 1

    async def delete(self, doc_ids) -> None:
        for doc_id in doc_ids:
            index = self._index(doc_id)
            if not self._terminal[index]:
                self._terminal[index] = 1
                self.completed += 1


class _StreamingFullDocs:
    """Return one small raw body at a time and retain no document rows."""

    supports_strict_point_reads = True

    @staticmethod
    def _row(doc_id: str) -> dict[str, Any]:
        index = int(doc_id.rsplit("-", 1)[1])
        return {
            "content": f"bounded rss body {index}",
            "file_path": f"file-{index:09d}.txt",
            "parse_format": FULL_DOCS_FORMAT_RAW,
            "content_hash": f"hash-{index:09d}",
            "process_options": "",
        }

    async def get_by_id(self, doc_id: str) -> dict[str, Any]:
        return self._row(doc_id)

    async def get_by_id_strict(self, doc_id: str) -> dict[str, Any]:
        return self._row(doc_id)


async def _rss_process_single_document(
    self,
    *,
    doc_id: str,
    status_doc: DocProcessingStatus,
    parsed_data: dict[str, Any],
    ctx,
) -> None:
    """Offline KG/LLM stub behind the real production process worker."""

    del parsed_data
    async with ctx.pipeline_status_lock:
        ctx.processed_count += 1
    await asyncio.sleep(0)
    await self._upsert_doc_status_transition(
        ctx=ctx,
        doc_id=doc_id,
        status=DocStatus.PROCESSED,
        status_doc=status_doc,
        file_path=status_doc.file_path,
        extra_fields={"chunks_count": 0, "chunks_list": []},
    )


async def _run_pipeline_rss(total: int, working_dir: str) -> dict[str, int]:
    import numpy as np

    from lightrag import LightRAG
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        get_pipeline_ingress,
        initialize_pipeline_status,
        initialize_share_data,
    )
    from lightrag.kg.pipeline_ingress import PipelineIngressMessage
    from lightrag.utils import EmbeddingFunc, Tokenizer

    class _Tokenizer:
        def encode(self, content: str) -> list[int]:
            return [ord(ch) for ch in content]

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(token) for token in tokens)

    async def _embedding(texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 8), dtype=float)

    async def _llm(*args, **kwargs) -> str:
        return "ok"

    initialize_share_data()
    workspace = f"rss-pipeline-{uuid4().hex}"
    await initialize_pipeline_status(workspace=workspace)
    rag = LightRAG(
        working_dir=working_dir,
        workspace=workspace,
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_embedding
        ),
        tokenizer=Tokenizer("rss-tokenizer", _Tokenizer()),
        max_parallel_insert=2,
        max_parallel_parse_native=2,
        max_parallel_analyze=2,
        queue_size_parse=_QUEUE_SIZE,
        queue_size_analyze=_QUEUE_SIZE,
        queue_size_insert=_QUEUE_SIZE,
        pipeline_scheduling_page_size=_PIPELINE_PAGE_SIZE,
    )
    doc_status = _StreamingPipelineDocStatus(total)
    rag.doc_status = doc_status
    rag.full_docs = _StreamingFullDocs()
    rag.process_single_document = MethodType(_rss_process_single_document, rag)
    ingress = await get_pipeline_ingress(workspace)
    ingress.put_document(
        PipelineIngressMessage(
            kind="document",
            doc_id=doc_status._doc_id(_PIPELINE_PAGE_SIZE - 1),
        )
    )

    sampler = _PeakRSS()
    sampler.start()
    try:
        await rag.apipeline_process_enqueue_documents()
    finally:
        rss = sampler.stop()
        finalize_share_data()

    if doc_status.completed != total:
        raise AssertionError(
            f"pipeline completed {doc_status.completed}/{total} documents"
        )
    if doc_status.parsing_writes < total:
        raise AssertionError("parse workers did not process every document")
    if doc_status.analyzing_writes < total:
        raise AssertionError("analyze workers did not process every document")
    if doc_status.processed_writes != total:
        raise AssertionError("process workers did not process every document")
    if doc_status.feeder_hydrations < 1:
        raise AssertionError("production feeder did not hydrate/admit its document")
    if doc_status.page_calls < max(2, total // _PIPELINE_PAGE_SIZE):
        raise AssertionError("production supervisor did not traverse multiple pages")
    return {
        **rss,
        "completed": doc_status.completed,
        "feeder_hydrations": doc_status.feeder_hydrations,
        "pages": doc_status.page_calls,
    }


class _ScanDocStatus:
    async def resolve_doc_source_strict(self, canonical_source_key: str):
        del canonical_source_key
        return SourceAbsent()

    async def get_full_docs_by_ids(self, doc_ids, *, strict: bool = False):
        del doc_ids, strict
        return {}


class _ScanFullDocs:
    supports_strict_point_reads = True

    async def get_by_id(self, doc_id: str):
        del doc_id
        return None

    async def get_by_id_strict(self, doc_id: str):
        del doc_id
        return None


class _ScanRag:
    """Non-retaining storage/processing edge around the production scan."""

    def __init__(self) -> None:
        self.workspace = f"rss-scan-{uuid4().hex}"
        self.doc_status = _ScanDocStatus()
        self.full_docs = _ScanFullDocs()
        self.addon_params = {}
        self.enqueued = 0
        self.process_calls = 0

    async def apipeline_enqueue_documents(self, _input: str, **kwargs):
        del kwargs
        self.enqueued += 1
        return "enqueued"

    async def apipeline_enqueue_error_documents(self, *args, **kwargs) -> None:
        raise AssertionError(f"unexpected scan enqueue error: {args!r} {kwargs!r}")

    async def arollback_failed_custom_chunk_patches(self, **kwargs):
        del kwargs
        return {"rolled_back": [], "failed": []}

    async def apipeline_reset_failed_for_scan(
        self, request_id: str, *, scan_owner_token: str | None = None
    ) -> bool:
        del scan_owner_token
        from lightrag.kg.shared_storage import get_pipeline_ingress

        ingress = await get_pipeline_ingress(self.workspace)
        ingress.ack_manual_retry(request_id)
        return True

    async def apipeline_process_enqueue_documents(self) -> None:
        self.process_calls += 1


def _import_document_routes():
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        return importlib.import_module("lightrag.api.routers.document_routes")
    finally:
        sys.argv = original_argv


async def _run_scan_rss(total: int, input_dir: str) -> dict[str, int]:
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        get_scan_job_store,
        initialize_pipeline_status,
        initialize_share_data,
    )

    routes = _import_document_routes()
    routes.global_args = SimpleNamespace(scan_enqueue_batch_size=_PIPELINE_PAGE_SIZE)

    directory = Path(input_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # Real directory entries, created before the baseline so the test measures
    # discovery/classification/enqueue rather than fixture construction.
    for index in range(total):
        (directory / f"rss-{index:09d}-{'x' * 80}.txt").touch()

    rag = _ScanRag()
    initialize_share_data()
    await initialize_pipeline_status(workspace=rag.workspace)
    manager = routes.DocumentManager(str(directory))
    router = routes.create_document_routes(rag, manager)
    scan_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "scan_for_new_documents"
    ][-1]
    managed_tasks: set[asyncio.Task] = set()
    sampler = _PeakRSS()
    sampler.start()
    job = None
    try:
        response = await scan_endpoint(managed_tasks)
        while managed_tasks:
            await asyncio.gather(*list(managed_tasks))
        job = get_scan_job_store(rag.workspace).get(response.track_id)
    finally:
        rss = sampler.stop()
        finalize_share_data()

    if response.status != "scanning_started":
        raise AssertionError(f"unexpected /scan response: {response!r}")
    if job is None or job.get("status") != "completed":
        raise AssertionError(f"/scan job did not reach completed: {job!r}")
    if rag.enqueued != total:
        raise AssertionError(f"scan enqueued {rag.enqueued}/{total} files")
    if rag.process_calls != 1:
        raise AssertionError(
            f"scan drove processing {rag.process_calls} times instead of once"
        )
    return {**rss, "enqueued": rag.enqueued}


def _child_entry(
    kind: str,
    total: int,
    working_dir: str,
    connection,
) -> None:
    logging.disable(logging.CRITICAL)
    try:
        if kind == "pipeline":
            result = asyncio.run(_run_pipeline_rss(total, working_dir))
        elif kind == "scan":
            result = asyncio.run(_run_scan_rss(total, working_dir))
        else:  # pragma: no cover - parent controls this literal
            raise ValueError(f"unknown RSS workload {kind!r}")
        connection.send(("ok", result))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _measure_in_child(kind: str, total: int, working_dir: Path) -> dict[str, int]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_entry,
        args=(kind, total, str(working_dir), child),
        name=f"lightrag-{kind}-rss-{total}",
    )
    process.start()
    child.close()
    try:
        if not parent.poll(180):
            process.terminate()
            process.join(timeout=10)
            pytest.fail(f"{kind} RSS child timed out at backlog {total}")
        status, payload = parent.recv()
    finally:
        parent.close()
    process.join(timeout=30)
    assert not process.is_alive(), f"{kind} RSS child did not exit"
    assert process.exitcode == 0, (
        f"{kind} RSS child exited {process.exitcode}: {payload}"
    )
    assert status == "ok", payload
    return payload


def _assert_rss_does_not_track_backlog(
    label: str, small: dict[str, int], large: dict[str, int]
) -> None:
    limit = max(
        3 * small["growth"],
        small["growth"] + _RSS_FIXED_HEADROOM,
    )
    assert large["growth"] < limit, (
        f"{label} RSS grew with the backlog: "
        f"{small['growth'] / 1024 / 1024:.1f} MiB transient growth at small "
        f"vs {large['growth'] / 1024 / 1024:.1f} MiB at 10x; "
        f"allowed {limit / 1024 / 1024:.1f} MiB"
    )


def test_full_worker_feeder_pipeline_rss_does_not_track_backlog(tmp_path):
    """The production supervisor + feeder + three worker layers stay bounded."""

    small = _measure_in_child("pipeline", 1_000, tmp_path / "pipeline-small")
    large = _measure_in_child("pipeline", 10_000, tmp_path / "pipeline-large")

    assert large["completed"] == 10 * small["completed"]
    assert large["pages"] > 10
    _assert_rss_does_not_track_backlog("full pipeline", small, large)


def test_real_directory_scan_rss_does_not_track_file_count(tmp_path):
    """Production discovery/classification/enqueue over a real directory is flat."""

    small = _measure_in_child("scan", 1_000, tmp_path / "scan-small")
    large = _measure_in_child("scan", 10_000, tmp_path / "scan-large")

    assert large["enqueued"] == 10 * small["enqueued"]
    _assert_rss_does_not_track_backlog("real-directory scan", small, large)
