"""
Regression tests for lightrag.rerank.ali_rerank and the "aliyun" request/response
format branches of generic_rerank_api.

ali_rerank (DashScope / Aliyun reranking) is the only rerank provider wrapper that
uses a nested request body (top_n and return_documents live under "parameters",
not at the payload top level) and a nested response body (scores live under
"output.results", not "results"). None of this was exercised anywhere in the test
suite before this file: existing coverage in tests/chunker/test_rerank_chunking.py
only drives cohere_rerank / generic_rerank_api with response_format="standard".
"""

from unittest.mock import patch

import pytest

from lightrag.rerank import ali_rerank
from tests._rerank_session import mock_rerank_session


@pytest.mark.offline
class TestAliRerankRequestFormat:
    """ali_rerank must build DashScope's nested input/parameters payload shape,
    not the flat top_n/documents shape used by the standard (Jina/Cohere)
    providers."""

    @pytest.mark.asyncio
    async def test_payload_nests_query_and_documents_under_input(self):
        session, captured = mock_rerank_session({"output": {"results": []}})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await ali_rerank(
                query="capital of France",
                documents=["Paris is the capital.", "Tokyo is a city."],
                api_key="test-key",
            )

        payload = captured["kwargs"]["json"]
        assert payload["input"] == {
            "query": "capital of France",
            "documents": ["Paris is the capital.", "Tokyo is a city."],
        }
        # The flat, standard-format keys must not appear at the top level.
        assert "query" not in payload
        assert "documents" not in payload

    @pytest.mark.asyncio
    async def test_top_n_and_return_documents_go_under_parameters(self):
        session, captured = mock_rerank_session({"output": {"results": []}})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await ali_rerank(
                query="q",
                documents=["a", "b"],
                api_key="test-key",
                top_n=1,
            )

        payload = captured["kwargs"]["json"]
        # top_n must not leak to the top level of the payload.
        assert "top_n" not in payload
        assert payload["parameters"]["top_n"] == 1
        # ali_rerank always passes return_documents=False through to
        # generic_rerank_api, and the aliyun branch adds any non-None
        # return_documents value to "parameters" unconditionally.
        assert payload["parameters"]["return_documents"] is False

    @pytest.mark.asyncio
    async def test_extra_body_merges_into_parameters_not_top_level(self):
        session, captured = mock_rerank_session({"output": {"results": []}})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await ali_rerank(
                query="q",
                documents=["a"],
                api_key="test-key",
                extra_body={"instruction": "focus on dates"},
            )

        payload = captured["kwargs"]["json"]
        assert payload["parameters"]["instruction"] == "focus on dates"
        assert "instruction" not in payload

    @pytest.mark.asyncio
    async def test_model_and_default_base_url(self):
        session, captured = mock_rerank_session({"output": {"results": []}})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await ali_rerank(query="q", documents=["a"], api_key="test-key")

        assert captured["kwargs"]["json"]["model"] == "gte-rerank-v2"
        assert "dashscope.aliyuncs.com" in captured["args"][0]


@pytest.mark.offline
class TestAliRerankResponseFormat:
    """ali_rerank must read scores from output.results, DashScope's nested
    response shape, not from a top-level "results" key."""

    @pytest.mark.asyncio
    async def test_parses_nested_output_results(self):
        session, _ = mock_rerank_session(
            {
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.95},
                        {"index": 0, "relevance_score": 0.42},
                    ]
                },
                "usage": {"total_tokens": 10},
            }
        )

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            result = await ali_rerank(
                query="q",
                documents=["doc0", "doc1"],
                api_key="test-key",
            )

        assert result == [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.42},
        ]

    @pytest.mark.asyncio
    async def test_missing_output_key_returns_empty_without_raising(self):
        """A response with no "output" at all (e.g. an unexpected provider
        payload shape) must degrade to an empty result, not raise."""
        session, _ = mock_rerank_session({"unexpected": "shape"})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            result = await ali_rerank(
                query="q",
                documents=["doc0"],
                api_key="test-key",
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_output_results_not_a_list_returns_empty_and_warns(self):
        """If "output.results" is present but malformed (not a list), the
        aliyun branch must warn and fall back to an empty list rather than
        crash on next on a non-iterable."""
        session, _ = mock_rerank_session({"output": {"results": {"not": "a list"}}})

        with (
            patch("lightrag.rerank.aiohttp.ClientSession", return_value=session),
            patch("lightrag.rerank.logger.warning") as mock_warning,
        ):
            result = await ali_rerank(
                query="q",
                documents=["doc0"],
                api_key="test-key",
            )

        assert result == []
        # One warning for the malformed "output.results" shape itself, and a
        # second for the resulting empty result set.
        assert mock_warning.call_count == 2
        assert "to be list" in mock_warning.call_args_list[0].args[0]

    @pytest.mark.asyncio
    async def test_empty_results_list_returns_empty(self):
        session, _ = mock_rerank_session({"output": {"results": []}})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            result = await ali_rerank(
                query="q",
                documents=["doc0", "doc1"],
                api_key="test-key",
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_result_items_are_dropped_not_fatal(self):
        """Out-of-range / wrong-typed entries inside output.results must be
        discarded individually rather than failing the whole call, matching
        the standard-format malformed-item handling already covered in
        tests/chunker/test_rerank_chunking.py and tests/llm/test_jina_rerank.py."""
        session, _ = mock_rerank_session(
            {
                "output": {
                    "results": [
                        None,
                        {"index": 5, "relevance_score": 0.9},  # out of range
                        {"index": 0, "relevance_score": "bad"},
                        {"index": 1, "relevance_score": 0.6},
                    ]
                }
            }
        )

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            result = await ali_rerank(
                query="q",
                documents=["doc0", "doc1"],
                api_key="test-key",
            )

        assert result == [{"index": 1, "relevance_score": 0.6}]


@pytest.mark.offline
class TestAliRerankApiKeyResolution:
    """ali_rerank falls back from explicit api_key to DASHSCOPE_API_KEY, then
    to the generic RERANK_BINDING_API_KEY, like the other provider wrappers."""

    @pytest.mark.asyncio
    async def test_explicit_api_key_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-dashscope-key")
        session, captured = mock_rerank_session({"output": {"results": []}})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await ali_rerank(query="q", documents=["a"], api_key="explicit-key")

        assert captured["kwargs"]["headers"]["Authorization"] == "Bearer explicit-key"

    @pytest.mark.asyncio
    async def test_falls_back_to_dashscope_api_key_env_var(self, monkeypatch):
        monkeypatch.delenv("RERANK_BINDING_API_KEY", raising=False)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-dashscope-key")
        session, captured = mock_rerank_session({"output": {"results": []}})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await ali_rerank(query="q", documents=["a"])

        assert (
            captured["kwargs"]["headers"]["Authorization"] == "Bearer env-dashscope-key"
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_generic_rerank_binding_api_key(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.setenv("RERANK_BINDING_API_KEY", "generic-binding-key")
        session, captured = mock_rerank_session({"output": {"results": []}})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await ali_rerank(query="q", documents=["a"])

        assert (
            captured["kwargs"]["headers"]["Authorization"]
            == "Bearer generic-binding-key"
        )

    @pytest.mark.asyncio
    async def test_dashscope_api_key_preferred_over_generic_binding_key(
        self, monkeypatch
    ):
        monkeypatch.setenv("RERANK_BINDING_API_KEY", "generic-binding-key")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
        session, captured = mock_rerank_session({"output": {"results": []}})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await ali_rerank(query="q", documents=["a"])

        assert captured["kwargs"]["headers"]["Authorization"] == "Bearer dashscope-key"


@pytest.mark.offline
class TestAliRerankEmptyDocuments:
    @pytest.mark.asyncio
    async def test_empty_document_list_returns_empty(self):
        """Only the result is pinned. Whether an HTTP request is sent for an
        empty list is deliberately left open, so that a later short-circuit
        (the query path never reranks an empty list) does not break this test.
        """
        session, _ = mock_rerank_session({"output": {"results": []}})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            result = await ali_rerank(query="q", documents=[], api_key="test-key")

        assert result == []
