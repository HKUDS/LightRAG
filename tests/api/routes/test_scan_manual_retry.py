"""Scan-driven manual retry intent (Phase 0).

The scan endpoint publishes a sticky manual retry request AFTER its
reservation is granted; ``run_scanning_process`` carries the request id so
its driven processing run consumes it (the only path pulling FAILED docs
back in), with a classification-failure fallback drive in the finally and a
cancellation contract that leaves the request sticky.
"""

import asyncio
import importlib
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag import LightRAG  # noqa: E402
from lightrag.base import DocStatus  # noqa: E402
from lightrag.kg.pipeline_ingress import PipelineIngressMessage  # noqa: E402
from lightrag.kg.shared_storage import get_pipeline_ingress  # noqa: E402
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id  # noqa: E402

DocumentManager = _document_routes.DocumentManager
run_scanning_process = _document_routes.run_scanning_process

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


class _FlippableExtract:
    def __init__(self):
        self.fail = True

    async def __call__(self, chunks, *args, **kwargs):
        if self.fail:
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in chunks]


async def _build_rag(tmp_path, extract) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"scanmr-{uuid4().hex[:8]}",
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


async def _make_failed_doc(rag, extract) -> str:
    extract.fail = True
    await rag.apipeline_enqueue_documents(input="doc body", file_paths="a.txt")
    await rag.apipeline_process_enqueue_documents()
    doc_id = compute_mdhash_id("a.txt", prefix="doc-")
    row = await rag.doc_status.get_by_id(doc_id)
    assert row["status"] == DocStatus.FAILED
    return doc_id


async def _publish_manual(rag) -> str:
    ingress = await get_pipeline_ingress(rag.workspace)
    request_id = uuid4().hex
    ingress.request_manual_retry(
        request_id,
        PipelineIngressMessage(kind="rescan", retry_failed=True, request_id=request_id),
    )
    return request_id


def test_scan_drive_consumes_manual_request_and_retries_failed(tmp_path):
    """No-new-files scan: its drive run peeks the scan's sticky request and
    pulls the FAILED doc back in (the ONLY automatic-trigger path doing so)."""

    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(tmp_path / "inputs"))
            await run_scanning_process(rag, doc_manager, manual_request_id=request_id)

            row = await rag.doc_status.get_by_id(doc_id)
            assert row["status"] == DocStatus.PROCESSED
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []  # consumed + ACKed
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_classification_failure_fallback_drives_sticky_request(tmp_path):
    """Classification raising must not strand the published intent: the
    finally drives the queue once (storage-only) and the run consumes it."""

    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(tmp_path / "inputs"))

            def boom():
                raise RuntimeError("classification boom")

            doc_manager.scan_directory_for_new_files = boom
            await run_scanning_process(rag, doc_manager, manual_request_id=request_id)

            row = await rag.doc_status.get_by_id(doc_id)
            assert row["status"] == DocStatus.PROCESSED
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_new_files_all_fail_to_enqueue_still_drives_sticky_request(
    tmp_path, monkeypatch
):
    """new_files are found but EVERY one fails to enqueue (duplicate / empty
    body / extraction error), so pipeline_index_files never drives the queue
    and returns False.  ``queue_drive_attempted`` must stay False so the
    classification-failure fallback still gives this scan's sticky manual
    request its run.

    Fix-proof: the branch used to set ``queue_drive_attempted = True`` before
    the call, so the fallback was skipped and the FAILED doc stayed stranded
    until an unrelated trigger."""

    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(tmp_path / "inputs"))

            # A brand-new basename (no doc_status row) → classified as new;
            # the file never has to exist on disk because enqueue is stubbed
            # to fail for every file below.
            doc_manager.scan_directory_for_new_files = lambda: [
                Path(str(tmp_path / "inputs" / "brand_new.txt"))
            ]

            async def _enqueue_fails(rag, file_path, track_id=None, from_scan=False):
                return (False, None)

            monkeypatch.setattr(
                _document_routes, "pipeline_enqueue_file", _enqueue_fails
            )

            await run_scanning_process(rag, doc_manager, manual_request_id=request_id)

            row = await rag.doc_status.get_by_id(doc_id)
            assert row["status"] == DocStatus.PROCESSED  # fallback drove it
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []  # consumed + ACKed
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_cancelled_scan_keeps_request_sticky_and_does_not_drive(tmp_path):
    """ANY cancellation (shutdown or explicit — indistinguishable here) skips
    the fallback drive: the doc stays FAILED and the request stays sticky for
    the next trigger."""

    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(tmp_path / "inputs"))

            def cancelled():
                raise asyncio.CancelledError()

            doc_manager.scan_directory_for_new_files = cancelled
            with pytest.raises(asyncio.CancelledError):
                await run_scanning_process(
                    rag, doc_manager, manual_request_id=request_id
                )

            row = await rag.doc_status.get_by_id(doc_id)
            assert row["status"] == DocStatus.FAILED  # not driven
            ingress = await get_pipeline_ingress(rag.workspace)
            assert [m.request_id for m in ingress.snapshot_manual_retries()] == [
                request_id
            ]  # sticky, next trigger picks it up
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
