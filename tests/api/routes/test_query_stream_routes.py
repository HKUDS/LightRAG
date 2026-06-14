"""
Verify the /query and /query/stream endpoint response types.

Ensures:
  - /query  → application/json  (no streaming, backward-compatible)
  - /query/stream → application/x-ndjson
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
    "LIGHTRAG_API_PREFIX",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")


def _build_client():
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        from lightrag.api.config import parse_args
        from lightrag.api.lightrag_server import create_app

        args = parse_args()
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            return TestClient(create_app(args))
    finally:
        sys.argv = original_argv


class TestQueryRouteJsonOnly:
    """The /query endpoint must stay JSON-only to preserve backward compatibility."""

    def test_openapi_spec_declares_json_response(self):
        client = _build_client()
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()

        paths = spec.get("paths", {})
        query_path = paths.get("/query", {})
        assert query_path, "/query must be in OpenAPI paths"

        post_op = query_path.get("post", {})
        responses = post_op.get("responses", {})
        ok_resp = responses.get("200", {})
        content = ok_resp.get("content", {})

        # The /query endpoint must declare application/json — NOT ndjson
        assert (
            "application/json" in content
        ), "/query must declare application/json in OpenAPI spec"
        assert (
            "application/x-ndjson" not in content
        ), "/query must NOT declare application/x-ndjson — streaming belongs to /query/stream"

    def test_query_route_exists_and_accepts_post(self):
        client = _build_client()
        # A minimal POST to /query should reach the route (it'll 422 or 500
        # since we don't have a real LLM, but it should NOT 404/405)
        response = client.post("/query", json={"query": "test", "mode": "mix"})
        assert response.status_code not in (
            404,
            405,
        ), "/query route must exist and accept POST"


class TestQueryStreamRoute:
    """The /query/stream endpoint must serve application/x-ndjson."""

    def test_openapi_spec_declares_ndjson_response(self):
        client = _build_client()
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()

        paths = spec.get("paths", {})
        stream_path = paths.get("/query/stream", {})
        assert stream_path, "/query/stream must be in OpenAPI paths"

        post_op = stream_path.get("post", {})
        responses = post_op.get("responses", {})
        ok_resp = responses.get("200", {})
        content = ok_resp.get("content", {})

        # The /query/stream endpoint must declare application/x-ndjson
        assert (
            "application/x-ndjson" in content
        ), "/query/stream must declare application/x-ndjson in OpenAPI spec"

    def test_stream_route_exists_and_accepts_post(self):
        client = _build_client()
        response = client.post("/query/stream", json={"query": "test", "mode": "mix"})
        assert response.status_code not in (
            404,
            405,
        ), "/query/stream route must exist and accept POST"


class TestQueryStreamResponseContentType:
    """When the mock LLM returns a non-streaming result, /query/stream must
    still set the correct Content-Type header."""

    def test_stream_response_has_ndjson_content_type(self):
        """Even without a real LLM, the streaming response must carry the
        correct media type header."""

        original_argv = sys.argv.copy()
        try:
            sys.argv = ["lightrag-server"]
            from lightrag.api.config import parse_args
            from lightrag.api.lightrag_server import create_app

            args = parse_args()

            mock_rag = MagicMock()
            mock_result = {
                "llm_response": {
                    "is_streaming": False,
                    "content": "test response",
                },
                "data": {"references": []},
            }
            # Return a coroutine
            mock_rag.aquery_llm = MagicMock()

            async def _fake_aquery(*a, **kw):
                return mock_result

            mock_rag.aquery_llm.side_effect = _fake_aquery

            with patch("lightrag.api.lightrag_server.LightRAG", return_value=mock_rag):
                app = create_app(args)

            client = TestClient(app)
            response = client.post(
                "/query/stream",
                json={
                    "query": "test",
                    "mode": "mix",
                    "include_references": True,
                },
            )
            content_type = response.headers.get("content-type", "")
            assert (
                "application/x-ndjson" in content_type
            ), f"/query/stream must return application/x-ndjson, got: {content_type}"
        finally:
            sys.argv = original_argv
