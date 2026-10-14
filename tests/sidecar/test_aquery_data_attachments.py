"""Integration: query entrypoints accept full_docs_db (regression for #3739)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from lightrag.base import QueryParam, QueryResult
from lightrag.lightrag import LightRAG
from lightrag.operate import _build_context_str, kg_query, naive_query
from lightrag.sidecar.query_attachments import enrich_raw_data_attachments
from tests.sidecar.conftest_query_attachments import (
    FakeFullDocs,
    SYNTH_IM_0002,
)


@pytest.mark.offline
class TestFullDocsDbWiring:
    def test_operate_signatures_accept_full_docs_db(self):
        for fn in (kg_query, naive_query, _build_context_str):
            assert "full_docs_db" in inspect.signature(fn).parameters

    @pytest.mark.asyncio
    async def test_aquery_data_no_results_includes_empty_attachments(self, monkeypatch):
        """No-results failure must still expose data.attachments: []."""

        async def fake_kg_query(*_args, **_kwargs):
            return None

        monkeypatch.setattr("lightrag.lightrag.kg_query", fake_kg_query)

        rag = SimpleNamespace(
            text_chunks=None,
            full_docs=None,
            llm_response_cache=None,
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            chunks_vdb=None,
            _build_global_config=lambda: {},
            _query_done=lambda: _noop(),
        )

        result = await LightRAG.aquery_data(
            rag,
            "hello world integration test query",
            param=QueryParam(mode="local"),
        )

        assert result["status"] == "failure"
        assert result["data"]["attachments"] == []

    @pytest.mark.asyncio
    async def test_aquery_data_enriches_via_lightrag(
        self, monkeypatch, synthetic_sidecar_uri
    ):
        """LightRAG.aquery_data must pass full_docs_db and enrich attachments."""
        raw_data = {
            "status": "success",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [
                    {
                        "chunk_id": "doc-aaa-chunk-1",
                        "content": "text",
                        "reference_id": "1",
                        "file_path": "x.md",
                    }
                ],
                "references": [],
            },
            "metadata": {
                "drawing_candidate_whitelist": [SYNTH_IM_0002],
            },
        }
        captured: dict = {}

        async def fake_kg_query(*_args, **kwargs):
            captured["full_docs_db"] = kwargs.get("full_docs_db")
            return QueryResult(
                content="ctx",
                raw_data=raw_data,
                llm_generated=False,
            )

        monkeypatch.setattr("lightrag.lightrag.kg_query", fake_kg_query)

        rag = SimpleNamespace(
            text_chunks=None,
            full_docs=FakeFullDocs(synthetic_sidecar_uri),
            llm_response_cache=None,
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            chunks_vdb=None,
            _build_global_config=lambda: {},
            _query_done=lambda: _noop(),
        )

        result = await LightRAG.aquery_data(
            rag,
            "hello world integration test query",
            param=QueryParam(mode="local"),
        )

        assert captured.get("full_docs_db") is rag.full_docs
        assert "attachments" in result.get("data", {})
        assert len(result["data"]["attachments"]) == 1
        assert result["data"]["attachments"][0]["im_id"] == SYNTH_IM_0002
        assert result["metadata"]["drawing_candidate_whitelist"] == [SYNTH_IM_0002]

    @pytest.mark.asyncio
    async def test_aquery_llm_enriches_via_lightrag(
        self, monkeypatch, synthetic_sidecar_uri
    ):
        """LightRAG.aquery_llm must pass full_docs_db and enrich attachments."""
        raw_data = {
            "status": "success",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [
                    {
                        "chunk_id": "doc-aaa-chunk-1",
                        "content": "text",
                        "reference_id": "1",
                        "file_path": "x.md",
                    }
                ],
                "references": [],
            },
            "metadata": {
                "drawing_candidate_whitelist": [SYNTH_IM_0002],
            },
        }
        captured: dict = {}

        async def fake_kg_query(*_args, **kwargs):
            captured["full_docs_db"] = kwargs.get("full_docs_db")
            return QueryResult(
                content="answer text",
                raw_data=raw_data,
                llm_generated=True,
                is_streaming=False,
            )

        monkeypatch.setattr("lightrag.lightrag.kg_query", fake_kg_query)

        rag = SimpleNamespace(
            text_chunks=None,
            full_docs=FakeFullDocs(synthetic_sidecar_uri),
            llm_response_cache=None,
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            chunks_vdb=None,
            _build_global_config=lambda: {},
            _query_done=lambda: _noop(),
        )

        result = await LightRAG.aquery_llm(
            rag,
            "hello world integration test query",
            param=QueryParam(mode="local"),
        )

        assert captured.get("full_docs_db") is rag.full_docs
        assert "attachments" in result.get("data", {})
        assert len(result["data"]["attachments"]) == 1
        assert result["data"]["attachments"][0]["im_id"] == SYNTH_IM_0002
        assert result["metadata"]["drawing_candidate_whitelist"] == [SYNTH_IM_0002]

    @pytest.mark.asyncio
    async def test_aquery_llm_streaming_enriches_attachments(
        self, monkeypatch, synthetic_sidecar_uri
    ):
        """Streaming aquery_llm must still enrich attachments from whitelist (PR-1 full display)."""

        async def _fake_stream():
            yield "tok"

        raw_data = {
            "status": "success",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [
                    {
                        "chunk_id": "doc-aaa-chunk-1",
                        "content": "text",
                        "reference_id": "1",
                        "file_path": "x.md",
                    }
                ],
                "references": [],
            },
            "metadata": {
                "drawing_candidate_whitelist": [SYNTH_IM_0002],
            },
        }

        async def fake_kg_query(*_args, **_kwargs):
            return QueryResult(
                content=None,
                raw_data=raw_data,
                llm_generated=True,
                is_streaming=True,
                response_iterator=_fake_stream(),
            )

        monkeypatch.setattr("lightrag.lightrag.kg_query", fake_kg_query)

        rag = SimpleNamespace(
            text_chunks=None,
            full_docs=FakeFullDocs(synthetic_sidecar_uri),
            llm_response_cache=None,
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            chunks_vdb=None,
            _build_global_config=lambda: {},
            _query_done=lambda: _noop(),
        )

        result = await LightRAG.aquery_llm(
            rag,
            "hello world integration test query",
            param=QueryParam(mode="local"),
        )

        assert result["llm_response"]["is_streaming"] is True
        assert result["llm_response"]["response_iterator"] is not None
        assert len(result["data"]["attachments"]) == 1
        assert result["data"]["attachments"][0]["im_id"] == SYNTH_IM_0002
        assert result["metadata"]["drawing_candidate_whitelist"] == [SYNTH_IM_0002]

    @pytest.mark.asyncio
    async def test_aquery_llm_no_results_includes_empty_attachments(self, monkeypatch):
        async def fake_kg_query(*_args, **_kwargs):
            return None

        monkeypatch.setattr("lightrag.lightrag.kg_query", fake_kg_query)

        rag = SimpleNamespace(
            text_chunks=None,
            full_docs=None,
            llm_response_cache=None,
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            chunks_vdb=None,
            _build_global_config=lambda: {},
            _query_done=lambda: _noop(),
        )

        result = await LightRAG.aquery_llm(
            rag,
            "hello world integration test query",
            param=QueryParam(mode="local"),
        )

        assert result["status"] == "failure"
        assert result["data"]["attachments"] == []

    @pytest.mark.asyncio
    async def test_enrich_from_whitelist(self, synthetic_sidecar_uri):
        raw = {
            "data": {"chunks": []},
            "metadata": {"drawing_candidate_whitelist": [SYNTH_IM_0002]},
        }
        out = await enrich_raw_data_attachments(
            raw,
            FakeFullDocs(synthetic_sidecar_uri),
        )
        assert len(out["data"]["attachments"]) == 1


async def _noop() -> None:
    return None
