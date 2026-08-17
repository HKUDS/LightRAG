"""PROCESSED must be the commit record (issue #3400, Phase 2).

``_process_document`` historically upserted ``doc_status=PROCESSED`` (which
doc-status backends persist immediately) BEFORE ``_insert_done()`` flushed the
graph/vector/chunk stores — so a crash or flush failure in that window left a
durably-PROCESSED document whose data never hit disk.

These tests drive the real pipeline (JSON storages, offline) and assert the
reordered semantics:

- happy path: the PROCESSED upsert happens strictly AFTER the merge-stage
  data flush;
- flush failure: NO PROCESSED write ever reaches doc_status — the document
  goes straight to FAILED.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
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
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"flushorder-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_deterministic_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


def _status_value(status: object) -> str:
    if isinstance(status, DocStatus):
        return status.value
    return str(status).replace("DocStatus.", "").lower()


@pytest.mark.asyncio
async def test_processed_upsert_happens_after_data_flush(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(input="order doc", file_paths="order.txt")
        doc_id = compute_mdhash_id("order.txt", prefix="doc-")

        events: list[str] = []

        orig_insert_done = rag._insert_done

        async def spy_insert_done(*a, **k):
            events.append("insert_done")
            await orig_insert_done(*a, **k)

        monkeypatch.setattr(rag, "_insert_done", spy_insert_done)

        orig_upsert = rag.doc_status.upsert

        async def spy_upsert(data):
            for rec_id, rec in data.items():
                if rec_id == doc_id:
                    events.append(f"status:{_status_value(rec.get('status'))}")
            await orig_upsert(data)

        monkeypatch.setattr(rag.doc_status, "upsert", spy_upsert)

        await rag.apipeline_process_enqueue_documents()

        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert _status_value(doc_status["status"]) == "processed"

        processed_idx = events.index("status:processed")
        flush_indexes = [i for i, e in enumerate(events) if e == "insert_done"]
        assert flush_indexes, f"no _insert_done recorded: {events}"
        # The merge-stage flush precedes the PROCESSED commit record.
        assert any(i < processed_idx for i in flush_indexes), (
            f"PROCESSED (at {processed_idx}) was written before any data "
            f"flush: {events}"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_flush_failure_never_writes_processed(tmp_path, monkeypatch):
    """With the merge-stage flush failing, doc_status must never see a
    PROCESSED write for the document — not even transiently (the old order
    durably wrote PROCESSED first and only later corrected it to FAILED)."""
    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(input="fail doc", file_paths="fail.txt")
        doc_id = compute_mdhash_id("fail.txt", prefix="doc-")

        statuses_written: list[str] = []
        orig_upsert = rag.doc_status.upsert

        async def spy_upsert(data):
            for rec_id, rec in data.items():
                if rec_id == doc_id:
                    statuses_written.append(_status_value(rec.get("status")))
            await orig_upsert(data)

        monkeypatch.setattr(rag.doc_status, "upsert", spy_upsert)

        async def flush_boom():
            raise RuntimeError("vdb flush boom")

        monkeypatch.setattr(rag.entities_vdb, "index_done_callback", flush_boom)

        await rag.apipeline_process_enqueue_documents()

        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert _status_value(doc_status["status"]) == "failed"
        assert "processed" not in statuses_written, (
            f"a PROCESSED write reached doc_status despite the flush "
            f"failure: {statuses_written}"
        )
    finally:
        await rag.finalize_storages()
