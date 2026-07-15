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
        assert "application/json" in content, (
            "/query must declare application/json in OpenAPI spec"
        )
        assert "application/x-ndjson" not in content, (
            "/query must NOT declare application/x-ndjson — streaming belongs to /query/stream"
        )

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
        assert "application/x-ndjson" in content, (
            "/query/stream must declare application/x-ndjson in OpenAPI spec"
        )

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
            assert "application/x-ndjson" in content_type, (
                f"/query/stream must return application/x-ndjson, got: {content_type}"
            )
        finally:
            sys.argv = original_argv


class TestQueryStreamProtocolOrder:
    """Verify NDJSON line ordering: references must be the first line when
    include_progress is False (default); progress lines may precede references
    only when include_progress=True."""

    @staticmethod
    def _build_client_with_mock():
        original_argv = sys.argv.copy()
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
            "data": {"references": [{"reference_id": "1", "file_path": "/doc.pdf"}]},
        }

        async def _fake_aquery(*a, **kw):
            # If a progress_callback was passed, simulate one event.
            cb = kw.get("progress_callback")
            if cb:
                await cb("extracting_keywords")
            return mock_result

        mock_rag.aquery_llm = MagicMock(side_effect=_fake_aquery)

        with patch("lightrag.api.lightrag_server.LightRAG", return_value=mock_rag):
            app = create_app(args)

        client = TestClient(app)
        return client, original_argv

    @staticmethod
    def _parse_ndjson(body: str) -> list[dict]:
        lines = []
        for line in body.strip().split("\n"):
            line = line.strip()
            if line:
                lines.append(__import__("json").loads(line))
        return lines

    def test_references_first_without_progress(self):
        """Default (include_progress=False): references must be the first line."""
        client, original_argv = self._build_client_with_mock()
        try:
            response = client.post(
                "/query/stream",
                json={
                    "query": "test",
                    "mode": "mix",
                    "include_references": True,
                },
            )
            assert response.status_code == 200
            lines = self._parse_ndjson(response.text)
            assert len(lines) > 0
            # First line must be references, NOT progress
            assert "references" in lines[0], (
                f"Default stream must start with references, got: {lines[0]}"
            )
            # No progress lines should appear
            assert not any("progress" in item for item in lines), (
                "Default stream must not contain progress lines"
            )
            assert not any("response_time" in item for item in lines), (
                "Default stream must not contain timing metadata"
            )
        finally:
            sys.argv = original_argv

    def test_progress_precedes_references_when_opted_in(self):
        """include_progress=True: progress lines appear before references."""
        client, original_argv = self._build_client_with_mock()
        try:
            response = client.post(
                "/query/stream",
                json={
                    "query": "test",
                    "mode": "mix",
                    "include_references": True,
                    "include_progress": True,
                },
            )
            assert response.status_code == 200
            lines = self._parse_ndjson(response.text)
            assert len(lines) >= 2
            # First line should be a progress event
            assert "progress" in lines[0], (
                f"include_progress stream should start with progress, got: {lines[0]}"
            )
            # A references line must exist after progress
            ref_lines = [item for item in lines if "references" in item]
            assert len(ref_lines) > 0, "references line must be present"
            # The first progress line must come before the first references line
            first_progress_idx = next(
                i for i, item in enumerate(lines) if "progress" in item
            )
            first_ref_idx = next(
                i for i, item in enumerate(lines) if "references" in item
            )
            assert first_progress_idx < first_ref_idx, (
                "progress must precede references when include_progress=True"
            )
            assert "response_time" in lines[-1], (
                "include_progress stream must end with timing metadata"
            )
        finally:
            sys.argv = original_argv
