"""Document dates are validated and stored only on full document records."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

pytestmark = pytest.mark.offline


class _CharTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


async def _embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _llm(*_args, **_kwargs) -> str:
    return ""


def _chunking(
    tokenizer,
    content,
    split_by_character,
    split_by_character_only,
    chunk_overlap_token_size,
    chunk_token_size,
) -> list[dict]:
    return [{"tokens": len(content), "content": content, "chunk_order_index": 0}]


@pytest.fixture(autouse=True)
def _shared_storage():
    initialize_share_data()
    yield
    finalize_share_data()


async def _build_rag(tmp_path, label: str) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / label),
        workspace=f"document-date-{uuid4().hex[:8]}",
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=8192,
            func=_embedding,
        ),
        tokenizer=Tokenizer("document-date-test", _CharTokenizer()),
        chunking_func=_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document_date",
    ["2018/10/01", "2018-02-30", "", "2018-10-01T00:00:00Z"],
)
async def test_ainsert_rejects_invalid_document_date(tmp_path, document_date):
    rag = await _build_rag(tmp_path, "invalid")
    try:
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            await rag.ainsert(
                "historical facts",
                ids="doc-invalid",
                document_date=document_date,
            )

        assert await rag.full_docs.get_by_id("doc-invalid") is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_ainsert_persists_date_only_on_full_doc_and_keeps_chunk_ids_stable(
    tmp_path,
):
    dated = await _build_rag(tmp_path, "dated")
    undated = await _build_rag(tmp_path, "undated")
    try:
        await dated.ainsert(
            "same historical facts",
            ids="stable-doc",
            file_paths="organization.txt",
            document_date="2018-10-01",
        )
        await undated.ainsert(
            "same historical facts",
            ids="stable-doc",
            file_paths="organization.txt",
        )

        dated_full_doc = await dated.full_docs.get_by_id("stable-doc")
        undated_full_doc = await undated.full_docs.get_by_id("stable-doc")
        dated_status = await dated.doc_status.get_by_id("stable-doc")
        undated_status = await undated.doc_status.get_by_id("stable-doc")

        assert dated_full_doc["document_date"] == "2018-10-01"
        assert "document_date" not in undated_full_doc
        assert "document_date" not in dated_status.get("metadata", {})
        assert dated_status["chunks_list"] == undated_status["chunks_list"]

        for chunk_id in dated_status["chunks_list"]:
            chunk = await dated.text_chunks.get_by_id(chunk_id)
            assert "document_date" not in chunk
    finally:
        await dated.finalize_storages()
        await undated.finalize_storages()


@pytest.mark.asyncio
async def test_batch_document_dates_are_aligned_one_to_one(tmp_path):
    rag = await _build_rag(tmp_path, "batch")
    try:
        await rag.apipeline_enqueue_documents(
            input=["facts from 2018", "facts without a date"],
            ids=["doc-2018", "doc-undated"],
            document_dates=["2018-10-01", None],
        )

        dated = await rag.full_docs.get_by_id("doc-2018")
        undated = await rag.full_docs.get_by_id("doc-undated")
        assert dated["document_date"] == "2018-10-01"
        assert "document_date" not in undated

        with pytest.raises(ValueError, match="Number of document_dates"):
            await rag.apipeline_enqueue_documents(
                input=["one", "two"],
                ids=["doc-one", "doc-two"],
                document_dates=["2018-10-01"],
            )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_pending_parse_rewrite_preserves_document_date(tmp_path):
    rag = await _build_rag(tmp_path, "parse-rewrite")
    try:
        file_path = "organization.txt"
        doc_id = compute_mdhash_id(file_path, prefix="doc-")
        await rag.apipeline_enqueue_documents(
            input="",
            file_paths=file_path,
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine="legacy",
            document_dates=["2018-10-01"],
        )

        await rag._persist_parsed_full_docs(
            doc_id,
            {
                "content": "parsed historical facts",
                "file_path": file_path,
                "parse_format": "raw",
                "parse_engine": "legacy",
            },
        )

        parsed = await rag.full_docs.get_by_id(doc_id)
        assert parsed["content"] == "parsed historical facts"
        assert parsed["document_date"] == "2018-10-01"
    finally:
        await rag.finalize_storages()
