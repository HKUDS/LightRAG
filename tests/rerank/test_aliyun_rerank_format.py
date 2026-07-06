"""
Unit tests for Aliyun DashScope rerank request/response format handling.

Aliyun serves two incompatible rerank API formats depending on the model:
- gte-rerank-* and qwen3-vl-rerank use the nested format:
  request {"model", "input": {"query", "documents"}, "parameters": {...}}
  response {"output": {"results": [...]}}
- qwen3-rerank series use the flat (standard) format:
  request {"model", "query", "documents", "top_n", ...}
  response {"results": [...]}

Reference: https://help.aliyun.com/zh/model-studio/text-rerank-api
Regression tests for https://github.com/HKUDS/LightRAG/issues/3084
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from lightrag.rerank import ali_rerank, is_aliyun_flat_rerank_model


def make_mock_session(response_json, captured_payload):
    """Create a mock aiohttp.ClientSession that captures the request payload"""
    mock_response = Mock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_json)
    mock_response.request_info = None
    mock_response.history = None
    mock_response.headers = {}
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    def capture_post(*args, **kwargs):
        captured_payload.update(kwargs.get("json", {}))
        return mock_response

    mock_session = Mock()
    mock_session.post = Mock(side_effect=capture_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    return mock_session


class TestIsAliyunFlatRerankModel:
    """Test suite for the flat-format model detection helper"""

    def test_qwen3_rerank_is_flat(self):
        assert is_aliyun_flat_rerank_model("qwen3-rerank") is True

    def test_qwen3_rerank_variant_is_flat(self):
        assert is_aliyun_flat_rerank_model("qwen3-rerank-8b") is True

    def test_detection_is_case_insensitive(self):
        assert is_aliyun_flat_rerank_model("Qwen3-Rerank") is True

    def test_qwen3_vl_rerank_is_nested(self):
        assert is_aliyun_flat_rerank_model("qwen3-vl-rerank") is False

    def test_gte_rerank_v2_is_nested(self):
        assert is_aliyun_flat_rerank_model("gte-rerank-v2") is False

    def test_gte_rerank_is_nested(self):
        assert is_aliyun_flat_rerank_model("gte-rerank") is False


@pytest.mark.offline
class TestAliyunNestedFormat:
    """Tests for the nested input/parameters format (gte-rerank-*, qwen3-vl-rerank)"""

    @pytest.mark.asyncio
    async def test_gte_rerank_v2_uses_nested_payload(self):
        """gte-rerank-v2 should send nested input/parameters payload"""
        captured_payload = {}
        response_json = {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.3},
                ]
            }
        }
        mock_session = make_mock_session(response_json, captured_payload)

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=mock_session):
            result = await ali_rerank(
                query="test query",
                documents=["doc1", "doc2"],
                model="gte-rerank-v2",
                api_key="test-key",
                top_n=2,
            )

        # Request: nested format with input/parameters objects
        assert captured_payload["model"] == "gte-rerank-v2"
        assert captured_payload["input"] == {
            "query": "test query",
            "documents": ["doc1", "doc2"],
        }
        assert captured_payload["parameters"]["top_n"] == 2
        assert "query" not in captured_payload
        assert "documents" not in captured_payload
        assert "top_n" not in captured_payload

        # Response: results parsed from output.results
        assert result == [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.3},
        ]

    @pytest.mark.asyncio
    async def test_qwen3_vl_rerank_uses_nested_payload(self):
        """qwen3-vl-rerank shares the nested format with gte-rerank-v2"""
        captured_payload = {}
        response_json = {"output": {"results": [{"index": 0, "relevance_score": 0.8}]}}
        mock_session = make_mock_session(response_json, captured_payload)

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=mock_session):
            result = await ali_rerank(
                query="test query",
                documents=["doc1"],
                model="qwen3-vl-rerank",
                api_key="test-key",
                top_n=1,
            )

        assert captured_payload["input"] == {
            "query": "test query",
            "documents": ["doc1"],
        }
        assert captured_payload["parameters"]["top_n"] == 1
        assert "query" not in captured_payload
        assert result == [{"index": 0, "relevance_score": 0.8}]

    @pytest.mark.asyncio
    async def test_nested_format_puts_extra_body_in_parameters(self):
        """extra_body should be merged into the parameters object"""
        captured_payload = {}
        response_json = {"output": {"results": [{"index": 0, "relevance_score": 0.8}]}}
        mock_session = make_mock_session(response_json, captured_payload)

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=mock_session):
            await ali_rerank(
                query="test query",
                documents=["doc1"],
                model="gte-rerank-v2",
                api_key="test-key",
                extra_body={"custom_param": "value"},
            )

        assert captured_payload["parameters"]["custom_param"] == "value"
        assert "custom_param" not in captured_payload


@pytest.mark.offline
class TestAliyunFlatFormat:
    """Tests for the flat format used by qwen3-rerank series (issue #3084)"""

    @pytest.mark.asyncio
    async def test_qwen3_rerank_uses_flat_payload(self):
        """qwen3-rerank should send the flat payload with top-level query/documents"""
        captured_payload = {}
        response_json = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.3},
            ]
        }
        mock_session = make_mock_session(response_json, captured_payload)

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=mock_session):
            result = await ali_rerank(
                query="test query",
                documents=["doc1", "doc2"],
                model="qwen3-rerank",
                api_key="test-key",
                top_n=2,
            )

        # Request: flat format without input/parameters nesting
        assert captured_payload["model"] == "qwen3-rerank"
        assert captured_payload["query"] == "test query"
        assert captured_payload["documents"] == ["doc1", "doc2"]
        assert captured_payload["top_n"] == 2
        assert "input" not in captured_payload
        assert "parameters" not in captured_payload
        # The flat endpoint doesn't support return_documents
        assert "return_documents" not in captured_payload

        # Response: results parsed from top-level results
        assert result == [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.3},
        ]

    @pytest.mark.asyncio
    async def test_qwen3_rerank_flat_response_not_parsed_as_nested(self):
        """Flat responses must not be looked up under output.results (issue #3084)"""
        captured_payload = {}
        # A flat response has no "output" key; the old nested-only parsing
        # returned an empty result list for it
        response_json = {
            "object": "list",
            "results": [{"index": 0, "relevance_score": 0.95}],
            "usage": {"total_tokens": 79},
        }
        mock_session = make_mock_session(response_json, captured_payload)

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=mock_session):
            result = await ali_rerank(
                query="test query",
                documents=["doc1"],
                model="qwen3-rerank",
                api_key="test-key",
            )

        assert result == [{"index": 0, "relevance_score": 0.95}]

    @pytest.mark.asyncio
    async def test_flat_format_puts_extra_body_at_top_level(self):
        """extra_body (e.g. instruct) should be merged into the flat payload"""
        captured_payload = {}
        response_json = {"results": [{"index": 0, "relevance_score": 0.9}]}
        mock_session = make_mock_session(response_json, captured_payload)

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=mock_session):
            await ali_rerank(
                query="test query",
                documents=["doc1"],
                model="qwen3-rerank",
                api_key="test-key",
                extra_body={"instruct": "Retrieve relevant passages."},
            )

        assert captured_payload["instruct"] == "Retrieve relevant passages."
        assert "parameters" not in captured_payload
