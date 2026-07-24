"""Every FAILED write goes through the ``mark_doc_failed`` funnel (Phase 1).

Fix-proof: before the audit, FAILED rows were written by plain upserts and
carried no ``failure_generation`` — logical generation 0, leaking into every
manual cohort forever. After it, every FAILED path (extraction failure via
``_upsert_doc_status_transition``, enqueue-time error documents, in-batch
duplicate markers, post-parse content duplicates) assigns a generation and
stamps the failing attempt.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.utils import EmbeddingFunc, Tokenizer

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


async def _failing_extract(chunks, *args, **kwargs):
    raise RuntimeError("extract fail sentinel")


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"ffw-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    rag._process_extract_entities = _failing_extract
    return rag


def test_extraction_failure_carries_generation_and_attempt(tmp_path):
    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            await rag.apipeline_enqueue_documents(input="body a", file_paths="a.txt")
            await rag.apipeline_process_enqueue_documents()
            match = await rag.doc_status.get_doc_by_file_basename("a.txt")
            assert match is not None
            _, row = match
            assert str(row.get("status")) == DocStatus.FAILED.value
            assert int(row.get("failure_generation") or 0) >= 1
            assert row.get("failure_attempt_id")
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_enqueue_error_documents_carry_generation(tmp_path):
    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            await rag.apipeline_enqueue_error_documents(
                [
                    {
                        "file_path": "broken.pdf",
                        "error_description": "bad file",
                        "original_error": "parse failed",
                    }
                ],
                track_id="track-err",
            )
            failed = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
            assert failed, "error document was not persisted"
            rows = await rag.doc_status.get_by_ids(list(failed.keys()))
            assert all(int(r.get("failure_generation") or 0) >= 1 for r in rows)
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_batch_duplicate_markers_carry_generation(tmp_path):
    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            # Same canonical basename twice in one enqueue: the second entry
            # becomes a dup-* FAILED marker — through the funnel it must
            # carry a generation like every other FAILED row.
            await rag.apipeline_enqueue_documents(
                input=["body 1", "body 2"],
                file_paths=["dup.txt", "dup.txt"],
            )
            failed = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
            dup_ids = [doc_id for doc_id in failed if doc_id.startswith("dup-")]
            assert dup_ids, "no duplicate marker row was created"
            rows = await rag.doc_status.get_by_ids(dup_ids)
            assert all(int(r.get("failure_generation") or 0) >= 1 for r in rows)
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
