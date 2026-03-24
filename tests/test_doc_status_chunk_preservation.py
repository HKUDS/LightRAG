import asyncio
from datetime import datetime, timezone
from types import MethodType
from uuid import uuid4

import numpy as np
import pytest

import lightrag.lightrag as lightrag_module
from lightrag.base import DocStatus
from lightrag.lightrag import LightRAG
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


def _failing_chunking(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    raise RuntimeError("chunking fail sentinel")


def _status_to_text(status: object) -> str:
    if isinstance(status, DocStatus):
        return status.value
    return str(status).replace("DocStatus.", "").lower()


async def _build_rag(tmp_path, test_name: str, chunking_func) -> LightRAG:
    workspace = f"{test_name}_{uuid4().hex[:8]}"
    rag = LightRAG(
        working_dir=str(tmp_path / test_name),
        workspace=workspace,
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=8192,
            func=_dummy_embedding,
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=chunking_func,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


async def _seed_chunk_cache_entries(
    rag: LightRAG, chunk_ids: list[str], prefix: str
) -> list[str]:
    updates = {}
    cache_records = {}
    cache_ids: list[str] = []

    for idx, chunk_id in enumerate(chunk_ids):
        chunk_data = await rag.text_chunks.get_by_id(chunk_id)
        assert chunk_data is not None
        cache_id = f"{prefix}-cache-{idx}"
        chunk_data["llm_cache_list"] = [cache_id]
        updates[chunk_id] = chunk_data
        cache_records[cache_id] = {"cache_type": "extract", "return": f"cached-{idx}"}
        cache_ids.append(cache_id)

    await rag.text_chunks.upsert(updates)
    await rag.llm_response_cache.upsert(cache_records)
    return cache_ids


@pytest.mark.asyncio
async def test_extract_failure_preserves_chunks_and_allows_delete_with_cache_cleanup(
    tmp_path, monkeypatch
):
    rag = await _build_rag(tmp_path, "extract_failure_cleanup", _deterministic_chunking)
    try:
        content = "extract failure document"
        file_path = "extract_failure.txt"
        doc_id = compute_mdhash_id(content, prefix="doc-")
        await rag.apipeline_enqueue_documents(input=content, file_paths=file_path)

        async def fail_extract(self, chunks, pipeline_status, pipeline_status_lock):
            raise RuntimeError("extract fail sentinel")

        rag._process_extract_entities = MethodType(fail_extract, rag)

        await rag.apipeline_process_enqueue_documents()

        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert doc_status is not None
        assert _status_to_text(doc_status["status"]) == "failed"
        chunk_ids = doc_status.get("chunks_list", [])
        assert len(chunk_ids) == 2
        assert doc_status.get("chunks_count") == 2

        cache_ids = await _seed_chunk_cache_entries(rag, chunk_ids, "extract")

        result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        assert result.status == "success"

        deleted_chunks = await rag.text_chunks.get_by_ids(chunk_ids)
        assert all(item is None for item in deleted_chunks)
        deleted_cache = [
            await rag.llm_response_cache.get_by_id(cid) for cid in cache_ids
        ]
        assert all(item is None for item in deleted_cache)
        assert await rag.doc_status.get_by_id(doc_id) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_extract_failure_before_chunking_preserves_previous_chunk_snapshot(
    tmp_path,
):
    rag = await _build_rag(tmp_path, "extract_failure_pre_chunking", _failing_chunking)
    try:
        content = "chunking failure document"
        file_path = "chunking_failure.txt"
        doc_id = compute_mdhash_id(content, prefix="doc-")
        await rag.apipeline_enqueue_documents(input=content, file_paths=file_path)

        previous_chunks = ["chunk-old-1", "chunk-old-2", "chunk-old-3"]
        existing = await rag.doc_status.get_by_id(doc_id)
        assert existing is not None
        await rag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.FAILED,
                    "content_summary": existing["content_summary"],
                    "content_length": existing["content_length"],
                    "chunks_count": len(previous_chunks),
                    "chunks_list": previous_chunks,
                    "created_at": existing["created_at"],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "file_path": existing["file_path"],
                    "track_id": existing["track_id"],
                    "error_msg": "previous failure",
                    "metadata": {"source": "test"},
                }
            }
        )

        await rag.apipeline_process_enqueue_documents()

        failed_status = await rag.doc_status.get_by_id(doc_id)
        assert failed_status is not None
        assert _status_to_text(failed_status["status"]) == "failed"
        assert failed_status.get("chunks_list") == previous_chunks
        assert failed_status.get("chunks_count") == len(previous_chunks)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_merge_failure_preserves_chunks_and_skip_cache_cleanup_when_disabled(
    tmp_path, monkeypatch
):
    rag = await _build_rag(
        tmp_path, "merge_failure_keep_cache", _deterministic_chunking
    )
    try:
        content = "merge failure document"
        file_path = "merge_failure.txt"
        doc_id = compute_mdhash_id(content, prefix="doc-")
        await rag.apipeline_enqueue_documents(input=content, file_paths=file_path)

        async def ok_extract(self, chunks, pipeline_status, pipeline_status_lock):
            return {"chunk_count": len(chunks)}

        async def fail_merge(**kwargs):
            raise RuntimeError("merge fail sentinel")

        rag._process_extract_entities = MethodType(ok_extract, rag)
        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", fail_merge)

        await rag.apipeline_process_enqueue_documents()

        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert doc_status is not None
        assert _status_to_text(doc_status["status"]) == "failed"
        chunk_ids = doc_status.get("chunks_list", [])
        assert len(chunk_ids) == 2
        assert doc_status.get("chunks_count") == 2

        cache_ids = await _seed_chunk_cache_entries(rag, chunk_ids, "merge")
        result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=False)
        assert result.status == "success"

        remaining_cache = [
            await rag.llm_response_cache.get_by_id(cid) for cid in cache_ids
        ]
        assert all(item is not None for item in remaining_cache)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_validate_and_fix_consistency_preserves_chunks_on_reset(tmp_path):
    rag = await _build_rag(tmp_path, "reset_preserve_chunks", _deterministic_chunking)
    try:
        failed_doc_id = "doc-failed-reset"
        processing_doc_id = "doc-processing-reset"
        inferred_count_doc_id = "doc-inferred-count-reset"

        now = datetime.now(timezone.utc).isoformat()
        await rag.full_docs.upsert(
            {
                failed_doc_id: {"content": "failed doc", "file_path": "failed.txt"},
                processing_doc_id: {
                    "content": "processing doc",
                    "file_path": "processing.txt",
                },
                inferred_count_doc_id: {
                    "content": "inferred count doc",
                    "file_path": "inferred.txt",
                },
            }
        )
        await rag.doc_status.upsert(
            {
                failed_doc_id: {
                    "status": DocStatus.FAILED,
                    "content_summary": "failed",
                    "content_length": 10,
                    "chunks_count": 2,
                    "chunks_list": ["f-1", "f-2"],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "failed.txt",
                    "track_id": "track-1",
                    "error_msg": "old error",
                    "metadata": {
                        "old": True,
                        "processing_start_time": 123,
                        "processing_end_time": 456,
                        "error_type": "file_extraction_error",
                    },
                },
                processing_doc_id: {
                    "status": DocStatus.PROCESSING,
                    "content_summary": "processing",
                    "content_length": 12,
                    "chunks_count": 1,
                    "chunks_list": ["p-1"],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "processing.txt",
                    "track_id": "track-2",
                    "error_msg": "old error",
                    "metadata": {
                        "old": True,
                        "processing_start_time": 999,
                    },
                },
                inferred_count_doc_id: {
                    "status": DocStatus.FAILED,
                    "content_summary": "inferred",
                    "content_length": 14,
                    "chunks_list": ["i-1", "i-2", "i-3"],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "inferred.txt",
                    "track_id": "track-3",
                    "error_msg": "old error",
                    "metadata": {
                        "old": True,
                        "processing_end_time": 777,
                    },
                },
            }
        )

        failed_docs = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
        processing_docs = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSING)
        to_process_docs = {**failed_docs, **processing_docs}

        pipeline_status = {"latest_message": "", "history_messages": []}
        await rag._validate_and_fix_document_consistency(
            to_process_docs=to_process_docs,
            pipeline_status=pipeline_status,
            pipeline_status_lock=asyncio.Lock(),
        )

        failed_reset = await rag.doc_status.get_by_id(failed_doc_id)
        assert failed_reset is not None
        assert _status_to_text(failed_reset["status"]) == "pending"
        assert failed_reset.get("chunks_list") == ["f-1", "f-2"]
        assert failed_reset.get("chunks_count") == 2
        assert failed_reset.get("metadata") == {"old": True}

        processing_reset = await rag.doc_status.get_by_id(processing_doc_id)
        assert processing_reset is not None
        assert _status_to_text(processing_reset["status"]) == "pending"
        assert processing_reset.get("chunks_list") == ["p-1"]
        assert processing_reset.get("chunks_count") == 1
        assert processing_reset.get("metadata") == {"old": True}

        inferred_count_reset = await rag.doc_status.get_by_id(inferred_count_doc_id)
        assert inferred_count_reset is not None
        assert _status_to_text(inferred_count_reset["status"]) == "pending"
        assert inferred_count_reset.get("chunks_list") == ["i-1", "i-2", "i-3"]
        assert inferred_count_reset.get("chunks_count") == 3
        assert inferred_count_reset.get("metadata") == {"old": True}
    finally:
        await rag.finalize_storages()
