"""``delete_llm_cache=True`` must not report an unqualified success when part of
the cache is unreachable (issue #3833).

LLM cache rows written during extraction are found for cleanup only through
the owning chunk's ``llm_cache_list``. A row whose key was never attached — a
sibling chunk raised and the collector was discarded, a hard kill, or a
suppressed ``update_chunk_cache_list`` failure — is invisible to
``adelete_by_doc_id``: the chunks go, the rows stay, and the deletion used to
come back as ``status="success"`` with a message saying nothing about it.

The deletion still succeeds (the residue is documented and self-healing only on
re-ingest), but the result message and the pipeline history now say that the
cache deletion was incomplete, and how many chunks had nothing to follow. The
notice is limited to documents whose extraction actually ran: a
``process_options='!'`` document never wrote cache rows, so its reference-less
chunks are expected.

The gap is measured while the chunk rows still exist. A purge that fails after
removing them leaves a retry with nothing to count, so the count is persisted
in the deletion retry metadata next to the cache ids and inherited by the
retry, which must still qualify its success.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import get_namespace_data
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


def _two_chunk_chunking(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    return [
        {"tokens": 1, "content": f"{content}::chunk{i}", "chunk_order_index": i}
        for i in range(2)
    ]


def _wire_fake_extraction(rag: LightRAG) -> None:
    """One ALICE--ACME relation per chunk; writes no LLM cache rows itself."""

    async def fake_extract(chunks, *args, **kwargs):
        results = []
        for chunk_id in chunks:
            nodes = {
                name: [
                    {
                        "entity_name": name,
                        "entity_type": "person",
                        "description": f"{name} description",
                        "source_id": chunk_id,
                        "file_path": "d.txt",
                        "timestamp": 1,
                    }
                ]
                for name in ("ALICE", "ACME")
            }
            edges = {
                ("ACME", "ALICE"): [
                    {
                        "src_id": "ACME",
                        "tgt_id": "ALICE",
                        "description": "works at",
                        "keywords": "employment",
                        "weight": 1.0,
                        "source_id": chunk_id,
                        "file_path": "d.txt",
                        "timestamp": 1,
                    }
                ]
            }
            results.append((nodes, edges))
        return results

    rag._process_extract_entities = fake_extract


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"cache-notice-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_two_chunk_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    _wire_fake_extraction(rag)
    return rag


async def _ingest(rag: LightRAG, process_options: str | None = None) -> str:
    doc_id = compute_mdhash_id("d.txt", prefix="doc-")
    kwargs = {"process_options": process_options} if process_options else {}
    await rag.apipeline_enqueue_documents(
        "alice works at acme", ids=[doc_id], file_paths=["d.txt"], **kwargs
    )
    await rag.apipeline_process_enqueue_documents()
    row = await rag.doc_status.get_by_id(doc_id)
    status = row.get("status")
    status_text = status.value if isinstance(status, DocStatus) else str(status)
    assert status_text == DocStatus.PROCESSED.value, row
    return doc_id


async def _chunk_ids(rag: LightRAG, doc_id: str) -> list[str]:
    row = await rag.doc_status.get_by_id(doc_id)
    chunk_ids = list(row.get("chunks_list") or [])
    assert len(chunk_ids) == 2, chunk_ids
    return chunk_ids


async def _attach_cache_rows(rag: LightRAG, chunk_ids: list[str]) -> list[str]:
    """Write one extraction cache row per chunk and attach it via ``llm_cache_list``.

    Mirrors what ``use_llm_func_with_cache`` + ``update_chunk_cache_list`` leave
    behind after a successful extraction. The fake extraction above writes
    neither, so the fixture stands in for it.
    """
    cache_ids: list[str] = []
    for chunk_id in chunk_ids:
        cache_id = f"default:extract:{uuid4().hex}"
        await rag.llm_response_cache.upsert(
            {
                cache_id: {
                    "return": "extracted",
                    "cache_type": "extract",
                    "chunk_id": chunk_id,
                }
            }
        )
        row = await rag.text_chunks.get_by_id(chunk_id)
        assert row is not None
        row["llm_cache_list"] = [cache_id]
        await rag.text_chunks.upsert({chunk_id: row})
        cache_ids.append(cache_id)
    return cache_ids


async def _history(rag: LightRAG) -> list[str]:
    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    return list(pipeline_status.get("history_messages", []))


@pytest.mark.asyncio
async def test_unreachable_cache_rows_qualify_the_success(tmp_path):
    """One chunk has no cache reference: delete succeeds, but says it is incomplete."""
    rag = await _build_rag(tmp_path)
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        # Attach a row to the first chunk only; the second chunk's rows (if any)
        # are the stranded case from the issue — nothing points at them.
        (attached_cache_id,) = await _attach_cache_rows(rag, chunk_ids[:1])

        result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert result.status == "success", result.message
        assert result.status_code == 200
        assert "LLM cache deletion is incomplete" in result.message
        assert "1 of 2 chunks carried no llm_cache_list references" in result.message
        # What could be found was still deleted, and the document is gone.
        assert await rag.llm_response_cache.get_by_id(attached_cache_id) is None
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        # The pipeline history — what the WebUI shows — carries the same notice.
        assert any("LLM cache deletion is incomplete" in m for m in await _history(rag))
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_every_chunk_referenced_reports_a_plain_success(tmp_path):
    """No false alarm when every chunk's cache rows were reachable and deleted."""
    rag = await _build_rag(tmp_path)
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        cache_ids = await _attach_cache_rows(rag, chunk_ids)

        result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert result.status == "success", result.message
        assert (
            result.message
            == f"Successfully deleted {len(cache_ids)} LLM cache entries for document {doc_id}"
        )
        assert "incomplete" not in result.message
        for cache_id in cache_ids:
            assert await rag.llm_response_cache.get_by_id(cache_id) is None
        assert not any("incomplete" in m for m in await _history(rag))
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_no_notice_when_cache_deletion_was_not_requested(tmp_path):
    """``delete_llm_cache=False`` promised nothing about the cache, so nothing to qualify."""
    rag = await _build_rag(tmp_path)
    try:
        doc_id = await _ingest(rag)
        await _chunk_ids(rag, doc_id)  # both chunks carry no reference

        result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=False)

        assert result.status == "success", result.message
        assert result.message == f"Document {doc_id} successfully deleted"
        assert not any("incomplete" in m for m in await _history(rag))
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_no_notice_for_a_document_whose_extraction_never_ran(tmp_path):
    """``process_options='!'`` skips extraction: reference-less chunks are expected."""
    rag = await _build_rag(tmp_path)
    try:
        doc_id = await _ingest(rag, process_options="!")
        row = await rag.doc_status.get_by_id(doc_id)
        assert row["metadata"]["skip_kg"] is True
        await _chunk_ids(rag, doc_id)

        result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert result.status == "success", result.message
        assert "incomplete" not in result.message
        assert not any("incomplete" in m for m in await _history(rag))
    finally:
        await rag.finalize_storages()


def _fail_once(monkeypatch, obj, attr: str, exc_message: str) -> None:
    """Wrap an async method to raise on its first call only."""
    calls = {"n": 0}
    original = getattr(obj, attr)

    async def wrapper(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(exc_message)
        return await original(*args, **kwargs)

    monkeypatch.setattr(obj, attr, wrapper)


@pytest.mark.asyncio
async def test_notice_survives_a_retry_after_the_chunk_rows_are_gone(
    tmp_path, monkeypatch
):
    """A retry that can no longer examine the chunks inherits the recorded gap.

    Fail the purge at the relation-anchor delete, which runs AFTER the chunk
    rows are removed. The retry's own count is then zero — ``get_by_ids``
    returns nothing — so it must fall back to what the first attempt persisted
    and still report the cache deletion as incomplete.
    """
    rag = await _build_rag(tmp_path)
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        (attached_cache_id,) = await _attach_cache_rows(rag, chunk_ids[:1])

        _fail_once(
            monkeypatch, rag.full_relations, "delete", "relations anchor delete boom"
        )
        first = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        assert first.status == "fail", first.message
        assert first.status_code == 500

        # Nothing is left to re-examine: the purge removed the chunk rows...
        assert all(row is None for row in await rag.text_chunks.get_by_ids(chunk_ids))
        # ...but the gap was persisted next to the cache ids before it ran.
        row = await rag.doc_status.get_by_id(doc_id)
        assert row["metadata"]["deletion_llm_cache_gap"] == {
            "chunks_without_refs": 1,
            "chunks_examined": 2,
        }
        assert row["metadata"]["deletion_llm_cache_ids"] == [attached_cache_id]

        monkeypatch.undo()
        second = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert second.status == "success", second.message
        assert "LLM cache deletion is incomplete" in second.message
        assert "1 of 2 chunks carried no llm_cache_list references" in second.message
        assert await rag.llm_response_cache.get_by_id(attached_cache_id) is None
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert any("LLM cache deletion is incomplete" in m for m in await _history(rag))
    finally:
        await rag.finalize_storages()
