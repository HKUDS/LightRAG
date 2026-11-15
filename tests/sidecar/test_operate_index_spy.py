"""Spy tests: operate kg/naive paths invoke ``index_figures_on_chunks``."""

from __future__ import annotations

import pytest

from lightrag.base import QueryParam
from lightrag.operate import _build_context_str, naive_query
from lightrag.sidecar.query_attachments import (
    DRAWING_CANDIDATE_WHITELIST_KEY,
    IndexFiguresResult,
    clear_drawings_index_cache,
)
from lightrag.utils import Tokenizer
from tests.sidecar.conftest_query_attachments import (
    FakeFullDocs,
    FakeTextChunks,
    SYNTH_DOC_ID,
    SYNTH_IM_0001,
)


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _mock_tokenizer() -> Tokenizer:
    return Tokenizer("mock-tokenizer", _SimpleTokenizerImpl())


@pytest.fixture(autouse=True)
def _clear_drawings_cache():
    clear_drawings_index_cache()
    yield
    clear_drawings_index_cache()


@pytest.mark.offline
class TestOperateIndexSpy:
    @pytest.mark.asyncio
    async def test_build_context_str_invokes_index_figures_on_chunks(
        self, monkeypatch, synthetic_sidecar_uri: str
    ):
        """kg path: real ``_build_context_str`` must call Index with full_docs_db."""
        calls: list[dict] = []

        async def spy_index(truncated_chunks, text_chunks_db, full_docs_db):
            calls.append(
                {
                    "truncated_chunks": truncated_chunks,
                    "text_chunks_db": text_chunks_db,
                    "full_docs_db": full_docs_db,
                }
            )
            return IndexFiguresResult(whitelist=[])

        monkeypatch.setattr("lightrag.operate.index_figures_on_chunks", spy_index)

        full_docs = FakeFullDocs(synthetic_sidecar_uri)
        chunk_id = f"{SYNTH_DOC_ID}-chunk-010"
        text_chunks = FakeTextChunks({})
        merged_chunks = [
            {
                "chunk_id": chunk_id,
                "content": f'cite <drawing id="{SYNTH_IM_0001}"/>',
                "file_path": "doc.md",
            }
        ]

        _, raw_data = await _build_context_str(
            entities_context=[{"entity_name": "E1"}],
            relations_context=[],
            merged_chunks=merged_chunks,
            query="test query",
            query_param=QueryParam(mode="hybrid", enable_rerank=False),
            global_config={
                "tokenizer": _mock_tokenizer(),
                "max_total_tokens": 100_000,
            },
            text_chunks_db=text_chunks,
            full_docs_db=full_docs,
        )

        assert len(calls) == 1
        assert calls[0]["full_docs_db"] is full_docs
        assert calls[0]["text_chunks_db"] is text_chunks
        chunk_ids = [c.get("chunk_id") for c in calls[0]["truncated_chunks"]]
        assert chunk_id in chunk_ids
        assert DRAWING_CANDIDATE_WHITELIST_KEY in raw_data.get("metadata", {})

    @pytest.mark.asyncio
    async def test_naive_query_invokes_index_figures_on_chunks(
        self, monkeypatch, synthetic_sidecar_uri: str
    ):
        """naive path: real ``naive_query`` must call Index before only_need_context return."""
        calls: list[dict] = []

        async def spy_index(processed_chunks, text_chunks_db, full_docs_db):
            calls.append(
                {
                    "processed_chunks": processed_chunks,
                    "text_chunks_db": text_chunks_db,
                    "full_docs_db": full_docs_db,
                }
            )
            return IndexFiguresResult(whitelist=[])

        monkeypatch.setattr("lightrag.operate.index_figures_on_chunks", spy_index)

        chunk_id = f"{SYNTH_DOC_ID}-chunk-010"
        fake_chunks = [
            {
                "chunk_id": chunk_id,
                "content": f'<drawing id="{SYNTH_IM_0001}"/>',
                "file_path": "doc.md",
            }
        ]

        async def fake_get_vector_context(*_args, **_kwargs):
            return fake_chunks

        monkeypatch.setattr(
            "lightrag.operate._get_vector_context", fake_get_vector_context
        )

        full_docs = FakeFullDocs(synthetic_sidecar_uri)
        text_chunks = FakeTextChunks({})

        result = await naive_query(
            "test query",
            chunks_vdb=object(),
            query_param=QueryParam(
                mode="naive", only_need_context=True, enable_rerank=False
            ),
            global_config={
                "tokenizer": _mock_tokenizer(),
                "max_total_tokens": 100_000,
                "role_llm_funcs": {"query": lambda *_a, **_k: "unused"},
            },
            text_chunks_db=text_chunks,
            full_docs_db=full_docs,
        )

        assert result is not None
        assert len(calls) == 1
        assert calls[0]["full_docs_db"] is full_docs
        assert calls[0]["text_chunks_db"] is text_chunks
        chunk_ids = [c.get("chunk_id") for c in calls[0]["processed_chunks"]]
        assert chunk_id in chunk_ids
        assert DRAWING_CANDIDATE_WHITELIST_KEY in result.raw_data.get("metadata", {})
