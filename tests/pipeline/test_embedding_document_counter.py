"""Ordinary ingestion must use the embedding adapter's document token budget."""

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


class _CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


@pytest.mark.asyncio
async def test_document_budget_is_enforced_before_vector_storage(tmp_path):
    document = "工单驳回"
    requests = []

    def count_document(text: str) -> int:
        return len("D:" + text)

    async def embed(texts: list[str]) -> np.ndarray:
        requests.extend("D:" + text for text in texts)
        return np.ones((len(texts), 2), dtype=np.float32)

    async def unexpected_llm(*args, **kwargs):
        pytest.fail("Vector-only ingestion must not call the LLM.")

    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"document-budget-{tmp_path.name}",
        tokenizer=Tokenizer("chars", _CharacterTokenizer()),
        embedding_func=EmbeddingFunc(
            embedding_dim=2,
            func=embed,
            max_token_size=4,
            document_token_counter=count_document,
        ),
        embedding_chunk_overlap_token_size=0,
        llm_model_func=unexpected_llm,
    )
    await rag.initialize_storages()
    try:
        await rag.apipeline_enqueue_documents(
            document, ids=["doc-budget"], process_options="!"
        )
        await rag.apipeline_process_enqueue_documents()
        status = await rag.doc_status.get_by_id("doc-budget")
        assert status["status"] == "processed"
        chunks = [
            await rag.text_chunks.get_by_id(chunk_id)
            for chunk_id in status["chunks_list"]
        ]
        assert requests == ["D:工单", "D:驳回"]
        assert [(chunk["content"], chunk["tokens"]) for chunk in chunks] == [
            ("工单", 2),
            ("驳回", 2),
        ]
    finally:
        await rag.finalize_storages()
