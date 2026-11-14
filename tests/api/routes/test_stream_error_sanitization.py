"""Streamed error lines must never carry raw exception text (CWE-209).

Once a ``StreamingResponse`` has started, a failure can no longer become an
HTTP 500, so ``/query/stream``, ``/api/generate`` and ``/api/chat`` report it as
an inline NDJSON line instead. Those lines used to be built from ``str(e)``,
which put provider endpoint URLs and deployment names into the response body,
bypassing ``internal_server_error`` that the non-streaming paths use. They now
go through ``stream_error_message``: the client gets a generic message plus a
correlation id, and the log line carries the detail under the same id.

Each route test drives the real router with an LLM stream that yields one chunk
and then raises, i.e. the failure lands after the response has begun.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_ollama_api = importlib.import_module("lightrag.api.routers.ollama_api")
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
_utils_api = importlib.import_module("lightrag.api.utils_api")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

_SECRET = "Upstream failure at https://internal-llm.corp.example:8443/v1 deployment=secret-deploy-123"
# Every needle must contain a character outside [0-9a-f]: the response carries
# a random hex correlation id, and a bare all-hex needle such as "8443" would
# match inside it now and then.
_NEEDLES = ("internal-llm.corp.example", ":8443/v1", "secret-deploy-123", _SECRET)
_SAFE_MESSAGE = re.compile(r"^Internal server error \(error_id: ([0-9a-f]{12})\)$")


class _ProviderError(RuntimeError):
    pass


async def _failing_stream(*_args, **_kwargs):
    yield "partial "
    raise _ProviderError(_SECRET)


def _assert_no_secret(text: str) -> None:
    for needle in _NEEDLES:
        assert needle not in text, f"stream body leaked {needle!r}"


def _ndjson(body: str) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def _error_id(message: str) -> str:
    match = _SAFE_MESSAGE.match(message)
    assert match, f"not a client-safe error message: {message!r}"
    return match.group(1)


def _assert_logged_under(records, error_id: str) -> None:
    """The operator-facing detail lives in the log, joined by the same id."""
    matching = [r for r in records if f"error_id={error_id}" in r.getMessage()]
    assert len(matching) == 1
    assert _SECRET in matching[0].getMessage()
    assert matching[0].exc_info is not None


@pytest.fixture(autouse=True)
def _open_auth(monkeypatch):
    monkeypatch.setattr(_utils_api, "auth_configured", False)


# --------------------------------------------------------------------------- #
# stream_error_message
# --------------------------------------------------------------------------- #


def test_stream_error_message_hides_text_and_logs_it_with_the_id(
    lightrag_log_records,
):
    try:
        raise _ProviderError(_SECRET)
    except _ProviderError as e:
        message = _utils_api.stream_error_message(e, "Streaming probe")

    _assert_no_secret(message)
    _assert_logged_under(lightrag_log_records, _error_id(message))


def test_stream_error_message_mints_a_fresh_id_per_call():
    exc = _ProviderError(_SECRET)
    first = _utils_api.stream_error_message(exc, "probe")
    second = _utils_api.stream_error_message(exc, "probe")
    assert _error_id(first) != _error_id(second)


# --------------------------------------------------------------------------- #
# /query/stream
# --------------------------------------------------------------------------- #


class _QueryRAG:
    def __init__(self, *, fail_before_stream: bool = False):
        self.fail_before_stream = fail_before_stream

    async def aquery_llm(self, *_args, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback:
            await callback("extracting_keywords")
        if self.fail_before_stream:
            raise _ProviderError(_SECRET)
        return {
            "llm_response": {
                "is_streaming": True,
                "response_iterator": _failing_stream(),
            },
            "data": {"references": []},
        }


def _query_client(rag) -> TestClient:
    app = FastAPI()
    app.include_router(_query_routes.create_query_routes(rag))
    return TestClient(app)


@pytest.mark.parametrize("include_progress", [False, True])
def test_query_stream_mid_stream_failure_is_sanitized(
    include_progress, lightrag_log_records
):
    response = _query_client(_QueryRAG()).post(
        "/query/stream",
        json={"query": "test", "mode": "mix", "include_progress": include_progress},
    )

    assert response.status_code == 200
    _assert_no_secret(response.text)
    lines = _ndjson(response.text)
    assert {"response": "partial "} in lines
    errors = [line["error"] for line in lines if "error" in line]
    assert len(errors) == 1
    _assert_logged_under(lightrag_log_records, _error_id(errors[0]))


def test_query_stream_background_query_failure_is_sanitized(lightrag_log_records):
    """With progress enabled the query runs in a task after the stream opened."""
    response = _query_client(_QueryRAG(fail_before_stream=True)).post(
        "/query/stream",
        json={"query": "test", "mode": "mix", "include_progress": True},
    )

    assert response.status_code == 200
    _assert_no_secret(response.text)
    lines = _ndjson(response.text)
    assert lines[0] == {"progress": "extracting_keywords"}
    assert set(lines[-1]) == {"error"}
    assert not any("response_time" in line for line in lines)
    _assert_logged_under(lightrag_log_records, _error_id(lines[-1]["error"]))


# --------------------------------------------------------------------------- #
# /api/generate and /api/chat
# --------------------------------------------------------------------------- #


class _OllamaRAG:
    def __init__(self):
        self.ollama_server_infos = SimpleNamespace(
            LIGHTRAG_MODEL="lightrag:latest",
            LIGHTRAG_SIZE=1,
            LIGHTRAG_CREATED_AT="now",
            LIGHTRAG_DIGEST="sha",
        )
        self.role_llm_kwargs = {"query": None}
        self.llm_model_kwargs = {}
        self.role_llm_funcs = {"query": self._llm}

    async def _llm(self, *_args, **_kwargs):
        return _failing_stream()

    async def aquery(self, *_args, **_kwargs):
        return _failing_stream()


@pytest.fixture
def ollama_client(monkeypatch):
    async def _count_tokens(_text):
        return 0

    monkeypatch.setattr(_ollama_api, "aestimate_tokens", _count_tokens)
    app = FastAPI()
    app.include_router(_ollama_api.OllamaAPI(_OllamaRAG()).router, prefix="/api")
    return TestClient(app)


def _assert_ollama_error_stream(response, records, content_of) -> None:
    assert response.status_code == 200
    _assert_no_secret(response.text)
    lines = _ndjson(response.text)
    assert content_of(lines[0]) == "partial "
    error_line = next(line for line in lines if "error" in line)
    message = error_line["error"].removeprefix("\n\nError: ")
    assert content_of(error_line) == f"\n\nError: {message}"
    _assert_logged_under(records, _error_id(message))
    assert lines[-1]["done"] is True


def test_generate_stream_failure_is_sanitized(ollama_client, lightrag_log_records):
    response = ollama_client.post(
        "/api/generate",
        json={"model": "lightrag:latest", "prompt": "hello", "stream": True},
    )
    _assert_ollama_error_stream(
        response, lightrag_log_records, lambda line: line["response"]
    )


@pytest.mark.parametrize(
    "content",
    [
        "/bypass hello",  # direct-LLM branch
        "what does the knowledge base say about widgets",  # RAG branch
    ],
)
def test_chat_stream_failure_is_sanitized(ollama_client, lightrag_log_records, content):
    response = ollama_client.post(
        "/api/chat",
        json={
            "model": "lightrag:latest",
            "messages": [{"role": "user", "content": content}],
            "stream": True,
        },
    )
    _assert_ollama_error_stream(
        response, lightrag_log_records, lambda line: line["message"]["content"]
    )
