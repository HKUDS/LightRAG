"""
Regression tests for lightrag.rerank.jina_rerank and the standard-format
branches of generic_rerank_api that are specific to Jina.

jina_rerank is the only standard-format provider wrapper that passes
return_documents=False through to generic_rerank_api. cohere_rerank passes
return_documents=None, which generic_rerank_api's standard-format branch
("if return_documents is not None") deliberately omits from the payload --
Jina's explicit False is what puts the key in the request at all. None of
this request-shape or api-key-resolution behavior was exercised anywhere in
the test suite before this file: existing coverage in
tests/chunker/test_rerank_chunking.py only drives cohere_rerank, and
tests/test_ali_rerank.py only drives the nested "aliyun" format.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from lightrag.rerank import jina_rerank


def _mock_session(response_json, status=200, headers=None):
    """Build a mock aiohttp.ClientSession whose session.post(...) resolves,
    as an async context manager, to a response carrying response_json."""
    mock_response = Mock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=response_json)
    mock_response.text = AsyncMock(return_value=str(response_json))
    mock_response.request_info = None
    mock_response.history = None
    mock_response.headers = headers or {}
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = Mock()
    mock_session.post = Mock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    return mock_session


def _capturing_session(response_json):
    """Like _mock_session, but also records the kwargs passed to session.post
    so the request payload/headers can be asserted on."""
    captured = {}
    mock_response = Mock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_json)
    mock_response.request_info = None
    mock_response.history = None
    mock_response.headers = {}
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    def capture_post(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return mock_response

    mock_session = Mock()
    mock_session.post = Mock(side_effect=capture_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    return mock_session, captured


@pytest.mark.offline
class TestJinaRerankRequestFormat:
    """jina_rerank must build the flat, standard request payload and, unlike
    cohere_rerank, must always forward return_documents=False at the top
    level rather than omitting it."""

    @pytest.mark.asyncio
    async def test_payload_uses_flat_query_and_documents(self):
        session, captured = _capturing_session({"results": []})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await jina_rerank(
                query="capital of France",
                documents=["Paris is the capital.", "Tokyo is a city."],
                api_key="test-key",
            )

        payload = captured["kwargs"]["json"]
        assert payload["query"] == "capital of France"
        assert payload["documents"] == [
            "Paris is the capital.",
            "Tokyo is a city.",
        ]
        # Jina's payload is flat, not nested under "input" like Aliyun's.
        assert "input" not in payload
        assert "parameters" not in payload

    @pytest.mark.asyncio
    async def test_return_documents_false_is_always_sent(self):
        """jina_rerank always passes return_documents=False through, and the
        standard-format branch includes any non-None return_documents value
        in the payload. cohere_rerank passes None instead, so the key is
        absent for Cohere -- this is the behavior that tells the two apart.
        """
        session, captured = _capturing_session({"results": []})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await jina_rerank(query="q", documents=["a", "b"], api_key="test-key")

        payload = captured["kwargs"]["json"]
        assert payload["return_documents"] is False

    @pytest.mark.asyncio
    async def test_top_n_sent_at_top_level(self):
        session, captured = _capturing_session({"results": []})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await jina_rerank(
                query="q", documents=["a", "b"], api_key="test-key", top_n=1
            )

        assert captured["kwargs"]["json"]["top_n"] == 1

    @pytest.mark.asyncio
    async def test_extra_body_merges_into_top_level_payload(self):
        session, captured = _capturing_session({"results": []})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await jina_rerank(
                query="q",
                documents=["a"],
                api_key="test-key",
                extra_body={"instruction": "focus on dates"},
            )

        payload = captured["kwargs"]["json"]
        assert payload["instruction"] == "focus on dates"

    @pytest.mark.asyncio
    async def test_model_and_default_base_url(self):
        session, captured = _capturing_session({"results": []})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await jina_rerank(query="q", documents=["a"], api_key="test-key")

        assert (
            captured["kwargs"]["json"]["model"] == "jina-reranker-v2-base-multilingual"
        )
        assert captured["args"][0] == "https://api.jina.ai/v1/rerank"


@pytest.mark.offline
class TestJinaRerankResponseFormat:
    """jina_rerank must read scores from a top-level "results" key, the
    standard shape, not Aliyun's nested "output.results"."""

    @pytest.mark.asyncio
    async def test_parses_top_level_results(self):
        session = _mock_session(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.42},
                ]
            }
        )

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            result = await jina_rerank(
                query="q", documents=["doc0", "doc1"], api_key="test-key"
            )

        assert result == [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.42},
        ]

    @pytest.mark.asyncio
    async def test_empty_results_list_returns_empty(self):
        session = _mock_session({"results": []})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            result = await jina_rerank(
                query="q", documents=["doc0", "doc1"], api_key="test-key"
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_missing_results_key_returns_empty_without_raising(self):
        session = _mock_session({"unexpected": "shape"})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            result = await jina_rerank(
                query="q", documents=["doc0"], api_key="test-key"
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_result_items_are_dropped_not_fatal(self):
        """Out-of-range / wrong-typed entries must be discarded individually
        rather than failing the whole call."""
        session = _mock_session(
            {
                "results": [
                    None,
                    {"index": 5, "relevance_score": 0.9},  # out of range
                    {"index": 0, "relevance_score": "bad"},
                    {"index": 1, "relevance_score": 0.6},
                ]
            }
        )

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            result = await jina_rerank(
                query="q", documents=["doc0", "doc1"], api_key="test-key"
            )

        assert result == [{"index": 1, "relevance_score": 0.6}]


@pytest.mark.offline
class TestJinaRerankApiKeyResolution:
    """jina_rerank falls back from explicit api_key to JINA_API_KEY, then
    to the generic RERANK_BINDING_API_KEY, like the other provider
    wrappers, and JINA_API_KEY takes priority over the generic key."""

    @pytest.mark.asyncio
    async def test_explicit_api_key_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "env-jina-key")
        session, captured = _capturing_session({"results": []})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await jina_rerank(query="q", documents=["a"], api_key="explicit-key")

        assert captured["kwargs"]["headers"]["Authorization"] == "Bearer explicit-key"

    @pytest.mark.asyncio
    async def test_falls_back_to_jina_api_key_env_var(self, monkeypatch):
        monkeypatch.delenv("RERANK_BINDING_API_KEY", raising=False)
        monkeypatch.setenv("JINA_API_KEY", "env-jina-key")
        session, captured = _capturing_session({"results": []})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await jina_rerank(query="q", documents=["a"])

        assert captured["kwargs"]["headers"]["Authorization"] == "Bearer env-jina-key"

    @pytest.mark.asyncio
    async def test_falls_back_to_generic_rerank_binding_api_key(self, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        monkeypatch.setenv("RERANK_BINDING_API_KEY", "generic-binding-key")
        session, captured = _capturing_session({"results": []})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await jina_rerank(query="q", documents=["a"])

        assert (
            captured["kwargs"]["headers"]["Authorization"]
            == "Bearer generic-binding-key"
        )

    @pytest.mark.asyncio
    async def test_jina_api_key_preferred_over_generic_binding_key(self, monkeypatch):
        monkeypatch.setenv("RERANK_BINDING_API_KEY", "generic-binding-key")
        monkeypatch.setenv("JINA_API_KEY", "jina-key")
        session, captured = _capturing_session({"results": []})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            await jina_rerank(query="q", documents=["a"])

        assert captured["kwargs"]["headers"]["Authorization"] == "Bearer jina-key"


@pytest.mark.offline
class TestJinaRerankEmptyDocuments:
    @pytest.mark.asyncio
    async def test_empty_document_list_sends_empty_documents_and_returns_empty(self):
        session, captured = _capturing_session({"results": []})

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            result = await jina_rerank(query="q", documents=[], api_key="test-key")

        assert captured["kwargs"]["json"]["documents"] == []
        assert result == []
