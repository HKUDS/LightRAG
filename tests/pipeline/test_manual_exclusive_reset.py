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
