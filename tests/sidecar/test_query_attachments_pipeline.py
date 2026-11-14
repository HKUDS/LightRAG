"""Full pipeline: aquery_* → Collect → Index → apply → Enrich → return."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from lightrag.base import QueryParam, QueryResult
from lightrag.lightrag import LightRAG
from lightrag.sidecar.query_attachments import (
    _fetch_chunk_records,
    apply_index_figures_to_raw_data,
    clear_drawings_index_cache,
    collect_im_ids_on_truncated_chunks,
    index_figures_on_chunks,
)
from tests.sidecar.conftest_query_attachments import (
    GoldenPipelineInputs,
    SYNTH_IM_0001,
    SYNTH_IM_0001_TITLE,
    SYNTH_IM_0002,
    SYNTH_IM_0002_TITLE,
    SYNTH_JPG_1,
    assert_valid_attachment,
    build_cross_bucket_dedupe_pipeline_inputs,
    build_golden_pipeline_inputs,
    remove_sidecar_asset,
)


@pytest.fixture(autouse=True)
def _clear_drawings_cache():
    clear_drawings_index_cache()
    yield
    clear_drawings_index_cache()


@pytest.fixture
def golden_pipeline_inputs(synthetic_sidecar_uri: str) -> GoldenPipelineInputs:
    return build_golden_pipeline_inputs(synthetic_sidecar_uri)


@pytest.fixture
def cross_bucket_dedupe_pipeline_inputs(
    synthetic_sidecar_uri: str,
) -> GoldenPipelineInputs:
    return build_cross_bucket_dedupe_pipeline_inputs(synthetic_sidecar_uri)


def _make_rag(inputs: GoldenPipelineInputs) -> SimpleNamespace:
    return SimpleNamespace(
        text_chunks=inputs.text_chunks_db,
        full_docs=inputs.full_docs_db,
        llm_response_cache=None,
        chunk_entity_relation_graph=None,
        entities_vdb=None,
        relationships_vdb=None,
        chunks_vdb=None,
        _build_global_config=lambda: {},
        _query_done=_noop,
    )


async def _noop() -> None:
    return None


async def _build_fake_indexed_raw_data(
    inputs: GoldenPipelineInputs,
    captured: dict[str, Any],
    *,
    text_chunks_db: Any,
    full_docs_db: Any,
) -> dict[str, Any]:
    """Run Collect → Index → apply inside a fake kg/naive query (no pre-filled whitelist)."""
    truncated_chunks = inputs.truncated_chunks
    captured["full_docs_db"] = full_docs_db
    captured["text_chunks_db"] = text_chunks_db

    chunk_ids = [c["chunk_id"] for c in truncated_chunks if c.get("chunk_id")]
    records = await _fetch_chunk_records(text_chunks_db, chunk_ids)
    collected = collect_im_ids_on_truncated_chunks(truncated_chunks, records)
    captured["collected"] = collected

    index_result = await index_figures_on_chunks(
        truncated_chunks,
        text_chunks_db,
        full_docs_db,
    )
    captured["index_whitelist"] = list(index_result.whitelist)

    raw_data = {
        "status": "success",
        "data": {
            "entities": [],
            "relationships": [],
            "chunks": [dict(c) for c in truncated_chunks],
            "references": [],
        },
        "metadata": {},
    }
    apply_index_figures_to_raw_data(raw_data, index_result)
    captured["applied_whitelist"] = list(
        raw_data["metadata"]["drawing_candidate_whitelist"]
    )
    return raw_data


def _make_query_result(
    raw_data: dict[str, Any],
    *,
    llm_generated: bool,
    is_streaming: bool = False,
    response_iterator: AsyncIterator[str] | None = None,
) -> QueryResult:
    return QueryResult(
        content="answer text" if llm_generated else "ctx",
        raw_data=raw_data,
        llm_generated=llm_generated,
        is_streaming=is_streaming,
        response_iterator=response_iterator,
    )


def _make_fake_kg_query(
    inputs: GoldenPipelineInputs,
    captured: dict[str, Any],
    *,
    llm_generated: bool,
    is_streaming: bool = False,
    response_iterator: AsyncIterator[str] | None = None,
) -> Callable[..., Any]:
    async def fake_kg_query(*args, **kwargs):
        text_chunks_db = args[4] if len(args) > 4 else kwargs.get("text_chunks_db")
        raw_data = await _build_fake_indexed_raw_data(
            inputs,
            captured,
            text_chunks_db=text_chunks_db,
            full_docs_db=kwargs.get("full_docs_db"),
        )
        return _make_query_result(
            raw_data,
            llm_generated=llm_generated,
            is_streaming=is_streaming,
            response_iterator=response_iterator,
        )

    return fake_kg_query


def _make_fake_naive_query(
    inputs: GoldenPipelineInputs,
    captured: dict[str, Any],
    *,
    llm_generated: bool,
    is_streaming: bool = False,
    response_iterator: AsyncIterator[str] | None = None,
) -> Callable[..., Any]:
    async def fake_naive_query(*_args, **kwargs):
        raw_data = await _build_fake_indexed_raw_data(
            inputs,
            captured,
            text_chunks_db=kwargs.get("text_chunks_db"),
            full_docs_db=kwargs.get("full_docs_db"),
        )
        return _make_query_result(
            raw_data,
            llm_generated=llm_generated,
            is_streaming=is_streaming,
            response_iterator=response_iterator,
        )

    return fake_naive_query


def _assert_pipeline_stages(
    *,
    captured: dict[str, Any],
    result: dict[str, Any],
    inputs: GoldenPipelineInputs,
    rag: SimpleNamespace,
) -> None:
    expected_collected = inputs.expected_collected
    expected_whitelist = inputs.resolved_whitelist
    expected_attachments = inputs.resolved_attachment_ids

    assert captured["full_docs_db"] is rag.full_docs
    assert captured["text_chunks_db"] is rag.text_chunks
    assert captured["collected"] == expected_collected
    assert captured["index_whitelist"] == expected_whitelist
    assert captured["applied_whitelist"] == expected_whitelist

    assert result["status"] == "success"
    assert result["metadata"]["drawing_candidate_whitelist"] == expected_whitelist
    attachments = result["data"]["attachments"]
    assert [a["im_id"] for a in attachments] == expected_attachments
    for attachment in attachments:
        assert_valid_attachment(attachment)
    for chunk in result["data"]["chunks"]:
        assert "figure_ids" not in chunk

    title_by_im_id = {
        SYNTH_IM_0001: SYNTH_IM_0001_TITLE,
        SYNTH_IM_0002: SYNTH_IM_0002_TITLE,
    }
    for attachment in attachments:
        im_id = attachment["im_id"]
        assert attachment.get("title") == title_by_im_id[im_id]


@pytest.mark.offline
class TestQueryAttachmentsPipeline:
    @pytest.mark.asyncio
    async def test_aquery_data_full_pipeline_golden_path(
        self, monkeypatch, golden_pipeline_inputs
    ):
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            "lightrag.lightrag.kg_query",
            _make_fake_kg_query(golden_pipeline_inputs, captured, llm_generated=False),
        )

        rag = _make_rag(golden_pipeline_inputs)
        result = await LightRAG.aquery_data(
            rag,
            "hello world integration test query",
            param=QueryParam(mode="local"),
        )

        _assert_pipeline_stages(
            captured=captured,
            result=result,
            inputs=golden_pipeline_inputs,
            rag=rag,
        )

    @pytest.mark.asyncio
    async def test_aquery_llm_full_pipeline_golden_path(
        self, monkeypatch, golden_pipeline_inputs
    ):
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            "lightrag.lightrag.kg_query",
            _make_fake_kg_query(golden_pipeline_inputs, captured, llm_generated=True),
        )

        rag = _make_rag(golden_pipeline_inputs)
        result = await LightRAG.aquery_llm(
            rag,
            "hello world integration test query",
            param=QueryParam(mode="local"),
        )

        _assert_pipeline_stages(
            captured=captured,
            result=result,
            inputs=golden_pipeline_inputs,
            rag=rag,
        )
        assert result["llm_response"]["content"] == "answer text"
        assert result["llm_response"]["is_streaming"] is False
        assert result["llm_response"]["llm_generated"] is True

    @pytest.mark.asyncio
    async def test_aquery_data_naive_full_pipeline_golden_path(
        self, monkeypatch, golden_pipeline_inputs
    ):
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            "lightrag.lightrag.naive_query",
            _make_fake_naive_query(
                golden_pipeline_inputs, captured, llm_generated=False
            ),
        )

        rag = _make_rag(golden_pipeline_inputs)
        result = await LightRAG.aquery_data(
            rag,
            "hello world integration test query",
            param=QueryParam(mode="naive"),
        )

        _assert_pipeline_stages(
            captured=captured,
            result=result,
            inputs=golden_pipeline_inputs,
            rag=rag,
        )

    @pytest.mark.asyncio
    async def test_aquery_llm_naive_full_pipeline_golden_path(
        self, monkeypatch, golden_pipeline_inputs
    ):
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            "lightrag.lightrag.naive_query",
            _make_fake_naive_query(
                golden_pipeline_inputs, captured, llm_generated=True
            ),
        )

        rag = _make_rag(golden_pipeline_inputs)
        result = await LightRAG.aquery_llm(
            rag,
            "hello world integration test query",
            param=QueryParam(mode="naive"),
        )

        _assert_pipeline_stages(
            captured=captured,
            result=result,
            inputs=golden_pipeline_inputs,
            rag=rag,
        )
        assert result["llm_response"]["content"] == "answer text"
        assert result["llm_response"]["is_streaming"] is False
        assert result["llm_response"]["llm_generated"] is True

    @pytest.mark.asyncio
    async def test_aquery_data_cross_bucket_dedupe(
        self, monkeypatch, cross_bucket_dedupe_pipeline_inputs
    ):
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            "lightrag.lightrag.kg_query",
            _make_fake_kg_query(
                cross_bucket_dedupe_pipeline_inputs, captured, llm_generated=False
            ),
        )

        rag = _make_rag(cross_bucket_dedupe_pipeline_inputs)
        result = await LightRAG.aquery_data(
            rag,
            "hello world integration test query",
            param=QueryParam(mode="local"),
        )

        _assert_pipeline_stages(
            captured=captured,
            result=result,
            inputs=cross_bucket_dedupe_pipeline_inputs,
            rag=rag,
        )

    @pytest.mark.asyncio
    async def test_aquery_llm_streaming_full_pipeline(
        self, monkeypatch, golden_pipeline_inputs
    ):
        async def _fake_stream():
            yield "tok"

        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            "lightrag.lightrag.kg_query",
            _make_fake_kg_query(
                golden_pipeline_inputs,
                captured,
                llm_generated=True,
                is_streaming=True,
                response_iterator=_fake_stream(),
            ),
        )

        rag = _make_rag(golden_pipeline_inputs)
        result = await LightRAG.aquery_llm(
            rag,
            "hello world integration test query",
            param=QueryParam(mode="local"),
        )

        _assert_pipeline_stages(
            captured=captured,
            result=result,
            inputs=golden_pipeline_inputs,
            rag=rag,
        )
        assert result["llm_response"]["is_streaming"] is True
        assert result["llm_response"]["response_iterator"] is not None
        assert result["llm_response"]["content"] is None
        assert result["llm_response"]["llm_generated"] is True

    @pytest.mark.asyncio
    async def test_aquery_data_index_vs_enrich_missing_jpg(
        self, monkeypatch, synthetic_sidecar, synthetic_sidecar_uri
    ):
        """Index whitelists without disk verify; Enrich skips missing jpg."""
        remove_sidecar_asset(synthetic_sidecar, SYNTH_JPG_1)
        inputs = replace(
            build_golden_pipeline_inputs(synthetic_sidecar_uri),
            expected_attachment_ids=[SYNTH_IM_0002],
        )

        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            "lightrag.lightrag.kg_query",
            _make_fake_kg_query(inputs, captured, llm_generated=False),
        )

        rag = _make_rag(inputs)
        result = await LightRAG.aquery_data(
            rag,
            "hello world integration test query",
            param=QueryParam(mode="local"),
        )

        _assert_pipeline_stages(
            captured=captured,
            result=result,
            inputs=inputs,
            rag=rag,
        )
        assert SYNTH_IM_0001 in result["metadata"]["drawing_candidate_whitelist"]
        assert SYNTH_IM_0001 not in [a["im_id"] for a in result["data"]["attachments"]]
