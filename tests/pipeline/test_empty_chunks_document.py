"""Regression coverage for documents whose chunker produces no chunks."""

from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id


pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    raise AssertionError("an empty chunk set must not invoke the LLM")


def _empty_chunking(*args, **kwargs) -> list[dict]:
    return []


@pytest.mark.asyncio
async def test_empty_chunk_document_reaches_processed_with_empty_anchors(tmp_path):
    """The real pipeline must commit an empty-chunk document normally.

    In particular, the ``if not chunks`` branch must continue through entity
    extraction and merge.  Merge Phase 0 persists the two present-but-empty
    recovery anchor rows before the document is marked PROCESSED.
    """
    workspace = f"empty-chunks-{uuid4().hex[:8]}"
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=workspace,
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=8192,
            func=_dummy_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_empty_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()

    try:
        doc_id = compute_mdhash_id("empty.txt", prefix="doc-")
        await rag.apipeline_enqueue_documents(
            "content intentionally removed by the chunker",
            ids=[doc_id],
            file_paths=["empty.txt"],
        )
        await rag.apipeline_process_enqueue_documents()

        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.PROCESSED
        assert status["chunks_count"] == 0
        assert status["chunks_list"] == []
        entity_anchor = await rag.full_entities.get_by_id(doc_id)
        relation_anchor = await rag.full_relations.get_by_id(doc_id)
        assert entity_anchor["entity_names"] == []
        assert entity_anchor["count"] == 0
        assert relation_anchor["relation_pairs"] == []
        assert relation_anchor["count"] == 0
    finally:
        await rag.finalize_storages()
