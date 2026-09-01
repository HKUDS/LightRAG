"""The `llm_generated` verdict on /query and /query/stream.

A query that finds no context is answered with the canned
``PROMPTS["fail_response"]`` and the ``only_need_context`` /
``only_need_prompt`` debug switches return the retrieved context or the
constructed prompt — none of these call the answering LLM, and no client can
tell that from the returned text. Clients that label machine-written output
(the WebUI's AI-content notice) need the server to say so, so the flag travels
on both endpoints:

- ``/query``: a field on the JSON response.
- ``/query/stream``: on the single COMPLETE-response line. Streamed chunks come
  from the LLM by construction and carry no flag, so a client's default holds.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lightrag.base import QueryResult
from lightrag.prompt import PROMPTS

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "OPENAI_API_KEY",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "WHITELIST_PATHS",
    "LIGHTRAG_API_PREFIX",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("TOKEN_SECRET", "")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    # openai keeps the app importable in a minimal test environment (the
    # optional `ollama` package is not always installed); no provider is
    # reached — aquery_llm itself is mocked.
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    auth_module = sys.modules.get("lightrag.api.auth")
    if auth_module is not None:
        monkeypatch.setattr(auth_module.auth_handler, "accounts", {}, raising=False)
    utils_api_module = sys.modules.get("lightrag.api.utils_api")
    if utils_api_module is not None:
        monkeypatch.setattr(utils_api_module, "auth_configured", False, raising=False)

    import lightrag.api.config as config

    config._global_args = None
    config._initialized = False
    yield
    config._global_args = None
    config._initialized = False


def _build_client(llm_response: dict):
    """A client whose `aquery_llm` returns exactly this `llm_response` block."""
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        from lightrag.api.config import parse_args
        from lightrag.api.lightrag_server import create_app

        args = parse_args()

        mock_rag = MagicMock()

        async def _fake_aquery(*_a, **_kw):
            return {"llm_response": llm_response, "data": {"references": []}}

        mock_rag.aquery_llm = MagicMock(side_effect=_fake_aquery)

        with patch("lightrag.api.lightrag_server.LightRAG", return_value=mock_rag):
            app = create_app(args)
        return TestClient(app)
    finally:
        sys.argv = original_argv


def _ndjson_lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# The core result carries the verdict.
# --------------------------------------------------------------------------- #
def test_query_result_is_llm_generated_by_default():
    """Only the paths that skip the answering LLM opt out, so the default must
    stay True — every LLM path constructs QueryResult without the argument."""
    assert QueryResult(content="an answer").llm_generated is True


@pytest.mark.asyncio
async def test_canned_no_context_result_is_not_llm_generated():
    """The empty-query short-circuit returns the canned reply without ever
    reaching a model."""
    from lightrag.operate import kg_query

    result = await kg_query(
        "",
        None,
        None,
        None,
        None,
        MagicMock(),
        {},
    )

    assert result.content == PROMPTS["fail_response"]
    assert result.llm_generated is False


# --------------------------------------------------------------------------- #
# /query
# --------------------------------------------------------------------------- #
def test_query_reports_a_generated_answer():
    client = _build_client(
        {"is_streaming": False, "content": "A real answer.", "llm_generated": True}
    )

    body = client.post("/query", json={"query": "test query", "mode": "mix"}).json()

    assert body["response"] == "A real answer."
    assert body["llm_generated"] is True


def test_query_reports_the_canned_no_context_reply():
    client = _build_client(
        {
            "is_streaming": False,
            "content": PROMPTS["fail_response"],
            "llm_generated": False,
        }
    )

    body = client.post("/query", json={"query": "test query", "mode": "mix"}).json()

    assert body["response"] == PROMPTS["fail_response"]
    assert body["llm_generated"] is False


def test_query_placeholder_for_empty_content_is_not_generated():
    """The route substitutes its own text when the result carries none; that
    substitute is the route's, not a model's."""
    client = _build_client({"is_streaming": False, "content": ""})

    body = client.post("/query", json={"query": "test query", "mode": "mix"}).json()

    assert body["response"] == "No relevant context found for the query."
    assert body["llm_generated"] is False


def test_query_defaults_to_generated_when_the_result_omits_the_flag():
    """Result shapes predating the flag are produced by the LLM paths only."""
    client = _build_client({"is_streaming": False, "content": "An answer."})

    body = client.post("/query", json={"query": "test query", "mode": "mix"}).json()

    assert body["llm_generated"] is True


# --------------------------------------------------------------------------- #
# /query/stream
# --------------------------------------------------------------------------- #
def test_stream_complete_response_line_carries_the_verdict():
    client = _build_client(
        {
            "is_streaming": False,
            "content": PROMPTS["fail_response"],
            "llm_generated": False,
        }
    )

    response = client.post("/query/stream", json={"query": "test query", "mode": "mix"})
    lines = _ndjson_lines(response)

    content_lines = [line for line in lines if "response" in line]
    assert len(content_lines) == 1
    assert content_lines[0]["response"] == PROMPTS["fail_response"]
    assert content_lines[0]["llm_generated"] is False


def test_streamed_chunks_carry_no_verdict():
    """Chunks exist only when the LLM is streaming, so a per-chunk flag would
    be noise — the client's default already reads them as generated."""

    async def _chunks():
        for part in ("Hello", " world"):
            yield part

    client = _build_client({"is_streaming": True, "response_iterator": _chunks()})

    response = client.post("/query/stream", json={"query": "test query", "mode": "mix"})
    lines = _ndjson_lines(response)

    content_lines = [line for line in lines if "response" in line]
    assert [line["response"] for line in content_lines] == ["Hello", " world"]
    assert all("llm_generated" not in line for line in content_lines)
