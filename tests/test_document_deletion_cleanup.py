import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest

import lightrag.lightrag as lightrag_module
from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _single_chunking_func(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
):
    return [
        {
            "content": content,
            "tokens": len(content),
            "chunk_order_index": 0,
        }
    ]


async def _dummy_llm(*args, **kwargs) -> str:
    return "<|COMPLETE|>"


async def _dummy_embed(texts: list[str], **kwargs) -> np.ndarray:
    await asyncio.sleep(0)
    return np.ones((len(texts), 4), dtype=np.float32)


async def _make_rag(tmp_path: str, workspace: str) -> LightRAG:
    tokenizer = Tokenizer("test-tokenizer", _SimpleTokenizerImpl())
    rag = LightRAG(
        working_dir=tmp_path,
        workspace=workspace,
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=4,
            max_token_size=8192,
            func=_dummy_embed,
            model_name="test-embedding",
        ),
        tokenizer=tokenizer,
        chunking_func=_single_chunking_func,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


@pytest.mark.offline
async def test_failed_doc_preserves_chunks_list_on_extraction_error(
    tmp_path, monkeypatch
):
    rag = await _make_rag(str(tmp_path), workspace="ws_extract_fail")
    try:
        monkeypatch.setattr(
            rag,
            "_process_extract_entities",
            AsyncMock(side_effect=RuntimeError("extract failed")),
        )

        content = "doc content that will fail during extraction stage"
        await rag.ainsert(content)

        doc_id = compute_mdhash_id(content, prefix="doc-")
        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert doc_status is not None
        assert doc_status["status"] == DocStatus.FAILED

        chunk_ids = doc_status.get("chunks_list")
        assert isinstance(chunk_ids, list)
        assert chunk_ids, "FAILED doc should keep chunks_list for deletion cleanup"
        assert doc_status.get("chunks_count") == len(chunk_ids)

        first_chunk_id = chunk_ids[0]
        assert await rag.text_chunks.get_by_id(first_chunk_id) is not None
        assert await rag.chunks_vdb.get_by_id(first_chunk_id) is not None

        cache_id = "cache-test-1"
        chunk_data = await rag.text_chunks.get_by_id(first_chunk_id)
        assert chunk_data is not None
        chunk_data["llm_cache_list"] = [cache_id]
        await rag.text_chunks.upsert({first_chunk_id: chunk_data})
        await rag.llm_response_cache.upsert({cache_id: {"content": "cached"}})

        result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        assert result.status == "success"

        assert await rag.text_chunks.get_by_id(first_chunk_id) is None
        assert await rag.chunks_vdb.get_by_id(first_chunk_id) is None
        assert await rag.llm_response_cache.get_by_id(cache_id) is None
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.offline
async def test_failed_doc_preserves_chunks_list_on_merge_error(tmp_path, monkeypatch):
    rag = await _make_rag(str(tmp_path), workspace="ws_merge_fail")
    try:
        monkeypatch.setattr(rag, "_process_extract_entities", AsyncMock(return_value=[]))
        monkeypatch.setattr(
            lightrag_module,
            "merge_nodes_and_edges",
            AsyncMock(side_effect=RuntimeError("merge failed")),
        )

        content = "doc content that will fail during merge stage"
        await rag.ainsert(content)

        doc_id = compute_mdhash_id(content, prefix="doc-")
        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert doc_status is not None
        assert doc_status["status"] == DocStatus.FAILED

        chunk_ids = doc_status.get("chunks_list")
        assert isinstance(chunk_ids, list)
        assert chunk_ids, "FAILED doc should keep chunks_list for deletion cleanup"
        assert doc_status.get("chunks_count") == len(chunk_ids)
    finally:
        await rag.finalize_storages()
