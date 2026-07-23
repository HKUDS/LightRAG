"""FAILED-retry semantics split (Phase 0 of the ingress-mailbox migration).

Automatic pipeline runs resume only ``_AUTO_RESUME_DOC_STATUSES`` (PENDING +
dead-process orphans); a FAILED document re-enters exclusively through a
sticky manual retry request in the workspace ingress — consumed earliest-first,
one per quiescence cycle, ACKed only after its FAILED→PENDING reset persists.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.kg.pipeline_ingress import PipelineIngressMessage
from lightrag.kg.shared_storage import get_pipeline_ingress
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


def _chunking(
    tokenizer,
    content,
    split_by_character,
    split_by_character_only,
    chunk_overlap_token_size,
    chunk_token_size,
) -> list[dict]:
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


class _CountingExtract:
    """Instance-level extraction stub: succeeds or always fails, counting calls."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    async def __call__(self, chunks, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in chunks]


async def _build_rag(tmp_path, workspace: str, extract: _CountingExtract) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=workspace,
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


async def _make_failed_doc(rag: LightRAG, name: str, extract: _CountingExtract) -> str:
    """Enqueue + process one doc while extraction fails → FAILED."""
    extract.fail = True
    await rag.apipeline_enqueue_documents(input=f"body of {name}", file_paths=name)
    await rag.apipeline_process_enqueue_documents()
    doc_id = compute_mdhash_id(name, prefix="doc-")
    assert await _status_of(rag, doc_id) == DocStatus.FAILED.value
    return doc_id


def test_upload_triggered_run_does_not_retry_failed(tmp_path):
    """Fix-proof for the semantics split: before it, ANY run (e.g. one
    triggered by an unrelated upload) re-queued every FAILED document."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, f"frs-{uuid4().hex[:8]}", extract)
        try:
            failed_id = await _make_failed_doc(rag, "a.txt", extract)
            failed_calls = extract.calls

            # An unrelated upload triggers a run: the new doc processes,
            # the FAILED one is NOT touched (no extract call, still FAILED).
            extract.fail = False
            await rag.apipeline_enqueue_documents(input="body b", file_paths="b.txt")
            await rag.apipeline_process_enqueue_documents()

            b_id = compute_mdhash_id("b.txt", prefix="doc-")
            assert await _status_of(rag, b_id) == DocStatus.PROCESSED.value
            assert await _status_of(rag, failed_id) == DocStatus.FAILED.value
            assert extract.calls == failed_calls + 1  # only b.txt was extracted
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_manual_retry_processes_failed_and_acks(tmp_path):
    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, f"frs-{uuid4().hex[:8]}", extract)
        try:
            failed_id = await _make_failed_doc(rag, "a.txt", extract)
            ingress = await get_pipeline_ingress(rag.workspace)

            extract.fail = False
            request_id = await request_failed_retry(rag)
            await rag.apipeline_process_enqueue_documents()

            assert await _status_of(rag, failed_id) == DocStatus.PROCESSED.value
            # Request was ACKed (terminal): nothing pending, replay refused.
            assert ingress.snapshot_manual_retries() == []
            assert (
                ingress.request_manual_retry(
                    request_id,
                    PipelineIngressMessage(
                        kind="rescan", retry_failed=True, request_id=request_id
                    ),
                )
                is False
            )

            # A second, DIFFERENT request is a fresh attempt (no doc is
            # FAILED any more — legitimately-empty manual scan ACKs too).
            second = await request_failed_retry(rag)
            await rag.apipeline_process_enqueue_documents()
            assert ingress.snapshot_manual_retries() == []
            assert second != request_id
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_two_requests_fifo_one_attempt_each(tmp_path):
    """R1 and R2 pending together: each grants exactly ONE retry attempt,
    consumed earliest-first, one per quiescence cycle — a doc that fails
    again inside R1's attempt stays FAILED until R2's cycle, and after both
    it stays FAILED with no retry loop."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, f"frs-{uuid4().hex[:8]}", extract)
        try:
            failed_id = await _make_failed_doc(rag, "a.txt", extract)
            calls_after_failure = extract.calls
            ingress = await get_pipeline_ingress(rag.workspace)

            r1 = await request_failed_retry(rag)
            r2 = await request_failed_retry(rag)
            assert [m.request_id for m in ingress.snapshot_manual_retries()] == [
                r1,
                r2,
            ]

            # extraction keeps failing: ONE run serves both requests at its
            # successive quiescence cycles — exactly two extra attempts.
            await rag.apipeline_process_enqueue_documents()

            assert await _status_of(rag, failed_id) == DocStatus.FAILED.value
            assert extract.calls == calls_after_failure + 2
            assert ingress.snapshot_manual_retries() == []  # both ACKed

            # No further signal: another run must not retry again.
            await rag.apipeline_process_enqueue_documents()
            assert extract.calls == calls_after_failure + 2
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_strict_scan_failure_keeps_request_sticky(tmp_path):
    """A manual request is ACKed only after a COMPLETE strict scan and the
    persisted resets — a failing scan leaves it sticky for the next run."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, f"frs-{uuid4().hex[:8]}", extract)
        try:
            failed_id = await _make_failed_doc(rag, "a.txt", extract)
            ingress = await get_pipeline_ingress(rag.workspace)
            request_id = await request_failed_retry(rag)

            original = rag.doc_status.get_docs_by_statuses
            boom = {"armed": True}

            async def flaky(statuses, strict=False):
                if strict and boom["armed"]:
                    boom["armed"] = False
                    raise RuntimeError("strict scan boom")
                return await original(statuses, strict=strict)

            rag.doc_status.get_docs_by_statuses = flaky
            with pytest.raises(RuntimeError, match="strict scan boom"):
                await rag.apipeline_process_enqueue_documents()

            # Not ACKed: the request survived the failed run...
            assert [m.request_id for m in ingress.snapshot_manual_retries()] == [
                request_id
            ]

            # ...and the next run executes it.
            extract.fail = False
            await rag.apipeline_process_enqueue_documents()
            assert await _status_of(rag, failed_id) == DocStatus.PROCESSED.value
            assert ingress.snapshot_manual_retries() == []
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_auto_rescan_rearmed_when_strict_refetch_fails(tmp_path):
    """CONTINUE_AUTO consumed the dirty flag; a failing strict refetch must
    re-arm it before propagating (sole-consumer compensation rule)."""

    async def _run():
        from lightrag.pipeline import PipelineNextDecision, PipelineNextStep

        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, f"frs-{uuid4().hex[:8]}", extract)
        try:
            ingress = await get_pipeline_ingress(rag.workspace)

            async def boom(statuses, strict=False):
                raise RuntimeError("refetch boom")

            rag.doc_status.get_docs_by_statuses = boom
            decision = PipelineNextDecision(PipelineNextStep.CONTINUE_AUTO)
            with pytest.raises(RuntimeError, match="refetch boom"):
                await rag._refetch_for_decision(decision, ingress)
            assert ingress.consume_auto_rescan() is True  # re-armed

            # A manual continuation stays sticky on its own — no re-arm.
            with pytest.raises(RuntimeError, match="refetch boom"):
                await rag._refetch_for_decision(
                    PipelineNextDecision(
                        PipelineNextStep.CONTINUE_MANUAL, manual_request_id="r1"
                    ),
                    ingress,
                )
            assert ingress.consume_auto_rescan() is False
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_interrupted_statuses_still_auto_resume(tmp_path):
    """Dead-process orphans (PROCESSING/PARSING/ANALYZING) self-heal on any
    run — only FAILED is fenced behind manual intent."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, f"frs-{uuid4().hex[:8]}", extract)
        try:
            await rag.apipeline_enqueue_documents(input="body a", file_paths="a.txt")
            doc_id = compute_mdhash_id("a.txt", prefix="doc-")
            row = await rag.doc_status.get_by_id(doc_id)
            await rag.doc_status.upsert(
                {doc_id: {**row, "status": DocStatus.PROCESSING}}
            )

            await rag.apipeline_process_enqueue_documents()
            assert await _status_of(rag, doc_id) == DocStatus.PROCESSED.value
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_manual_request_on_empty_queue_acks_as_complete(tmp_path):
    """A legitimately-empty complete manual scan satisfies the request:
    ACK and release, no loop, no stranded sticky entry."""

    async def _run():
        extract = _CountingExtract()
        rag = await _build_rag(tmp_path, f"frs-{uuid4().hex[:8]}", extract)
        try:
            ingress = await get_pipeline_ingress(rag.workspace)
            await request_failed_retry(rag)
            await rag.apipeline_process_enqueue_documents()
            assert ingress.snapshot_manual_retries() == []  # ACKed
            assert not ingress.has_work()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
