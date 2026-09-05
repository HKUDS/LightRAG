"""Query prompts enrich recalled chunks from document-level dates."""

from __future__ import annotations

import json

import pytest

from lightrag.base import QueryParam
from lightrag.operate import _merge_all_chunks, naive_query
from lightrag.utils import Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _ChunksVDB:
    cosine_better_than_threshold = 0.0

    def __init__(self):
        self.rows = [
            {
                "id": "chunk-2018",
                "content": "Alice leads Acme.",
                "file_path": "organization-2018.txt",
                "full_doc_id": "doc-2018",
            },
            {
                "id": "chunk-2024",
                "content": "Alice leads Acme.",
                "file_path": "organization-2024.txt",
                "full_doc_id": "doc-2024",
            },
            {
                "id": "chunk-undated",
                "content": "Alice leads Acme.",
                "file_path": "organization.txt",
                "full_doc_id": "doc-undated",
            },
        ]

    async def query(self, *_args, **_kwargs):
        return [dict(row) for row in self.rows]


class _FullDocs:
    def __init__(self):
        self.rows = {
            "doc-2018": {"content": "full 2018", "document_date": "2018-10-01"},
            "doc-2024": {"content": "full 2024", "document_date": "2024-10-01"},
            "doc-undated": {"content": "legacy full doc"},
        }
        self.requests: list[list[str]] = []

    async def get_by_ids(self, ids: list[str]):
        self.requests.append(ids)
        return [self.rows.get(doc_id) for doc_id in ids]


def _config(query_model):
    return {
        "tokenizer": Tokenizer("document-date-test", _CharTokenizer()),
        "role_llm_funcs": {"query": query_model},
        "max_total_tokens": 100_000,
        "min_rerank_score": 0.0,
        "enable_content_headings": False,
        "user_prompt_prefix": "",
    }


@pytest.mark.parametrize(
    "prompt_key", ["kg_query_context", "naive_query_context"], ids=["kg", "naive"]
)
def test_query_context_prompt_defines_document_date_semantics(prompt_key):
    from lightrag.prompt import PROMPTS

    prompt = " ".join(PROMPTS[prompt_key].split())
    assert "optional `document_date` field" in prompt
    assert "source document's as-of date" in prompt
    assert "ISO 8601" in prompt
    assert "ingestion or processing time" in prompt
    assert "`YYYY`, `YYYY-MM`, or `YYYY-MM-DD`" in prompt


@pytest.mark.parametrize(
    "prompt_key", ["rag_response", "naive_rag_response"], ids=["kg", "naive"]
)
def test_response_prompt_handles_dated_evidence_without_inventing_precision(
    prompt_key,
):
    from lightrag.prompt import PROMPTS

    prompt = PROMPTS[prompt_key]
    assert "dated chunk's time-dependent facts" in prompt
    assert "not its ingestion time" in prompt
    assert "alongside relevant time-dependent claims" in prompt
    assert "dated sources disagree about the current state" in prompt
    assert "prefer the clearly later evidence" in prompt
    assert "neither date is clearly later" in prompt
    assert "point-in-time or trend questions" in prompt
    assert "organize dated evidence chronologically" in prompt
    assert "Preserve the supplied precision" in prompt
    assert "do not infer a missing month or day" in prompt


@pytest.mark.asyncio
async def test_naive_answer_prompt_contains_dates_loaded_from_full_docs():
    captured: dict[str, str] = {}

    async def _query_model(query, **kwargs):
        captured["query"] = query
        captured["system_prompt"] = kwargs["system_prompt"]
        return "answer"

    chunks_vdb = _ChunksVDB()
    for row in chunks_vdb.rows:
        row["document_date"] = "1900-01-01"
    full_docs = _FullDocs()
    result = await naive_query(
        "Who leads Acme?",
        chunks_vdb,
        QueryParam(
            mode="naive",
            enable_rerank=False,
            chunk_top_k=3,
            max_total_tokens=100_000,
        ),
        _config(_query_model),
        full_docs_db=full_docs,
    )

    assert result.content == "answer"
    prompt = captured["system_prompt"]
    assert '"document_date": "2018-10-01"' in prompt
    assert '"document_date": "2024-10-01"' in prompt
    assert prompt.count('"document_date"') == 2
    assert full_docs.requests == [["doc-2018", "doc-2024", "doc-undated"]]

    chunks_by_path = {
        chunk["file_path"]: chunk for chunk in result.raw_data["data"]["chunks"]
    }
    assert chunks_by_path["organization-2018.txt"]["document_date"] == "2018-10-01"
    assert chunks_by_path["organization-2024.txt"]["document_date"] == "2024-10-01"
    assert "document_date" not in chunks_by_path["organization.txt"]
    assert all(row["document_date"] == "1900-01-01" for row in chunks_vdb.rows)


@pytest.mark.asyncio
async def test_shared_kg_chunk_merge_enriches_every_retrieval_source(monkeypatch):
    async def _entity_chunks(*_args, **_kwargs):
        return [
            {
                "chunk_id": "chunk-entity",
                "content": "entity context",
                "file_path": "entity.txt",
                "full_doc_id": "doc-2018",
                "document_date": "1900-01-01",
            }
        ]

    async def _relation_chunks(*_args, **_kwargs):
        return [
            {
                "chunk_id": "chunk-relation",
                "content": "relation context",
                "file_path": "relation.txt",
                "full_doc_id": "doc-undated",
                "document_date": "1900-01-01",
            }
        ]

    monkeypatch.setattr(
        "lightrag.operate._find_related_text_unit_from_entities", _entity_chunks
    )
    monkeypatch.setattr(
        "lightrag.operate._find_related_text_unit_from_relations", _relation_chunks
    )

    text_chunks_db = type(
        "_TextChunks",
        (),
        {"global_config": {"enable_content_headings": False}},
    )()
    merged = await _merge_all_chunks(
        filtered_entities=[{"entity_name": "Alice"}],
        filtered_relations=[{"src_id": "Alice", "tgt_id": "Acme"}],
        vector_chunks=[
            {
                "chunk_id": "chunk-vector",
                "content": "vector context",
                "file_path": "vector.txt",
                "full_doc_id": "doc-2024",
                "document_date": "1900-01-01",
            }
        ],
        text_chunks_db=text_chunks_db,
        query_param=QueryParam(mode="hybrid", enable_rerank=False),
        full_docs_db=_FullDocs(),
    )

    by_id = {chunk["chunk_id"]: chunk for chunk in merged}
    assert by_id["chunk-vector"]["document_date"] == "2024-10-01"
    assert by_id["chunk-entity"]["document_date"] == "2018-10-01"
    assert "document_date" not in by_id["chunk-relation"]

    rendered = "\n".join(json.dumps(chunk, sort_keys=True) for chunk in merged)
    assert "2018-10-01" in rendered
    assert "2024-10-01" in rendered
