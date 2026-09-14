"""Validation refusals must reach the client as refusals, not as 500s.

`QueryRequest` rejects short and empty queries by raising a ``ValueError``
inside a validator. Pydantic v2 attaches the exception OBJECT to the error's
``ctx['error']``, which ``json.dumps`` cannot serialize — so the shared
validation handler has to run the errors through ``jsonable_encoder`` before
putting them in a response body. Without it every one of these refusals came
back as an opaque 500 while the server logged an unhandled TypeError.

Structural errors (a missing field, a `max_length` breach) carry no
``ctx['error']`` and were unaffected, which is what kept this hidden.
"""

import importlib
import sys

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_qr = importlib.import_module("lightrag.api.routers.query_routes")
_server = importlib.import_module("lightrag.api.lightrag_server")
sys.argv = _original_argv

QueryRequest = _qr.QueryRequest

pytestmark = pytest.mark.offline


@pytest.fixture
def client():
    """Mount the REAL handler on stand-in routes shaped like the query ones."""
    app = FastAPI()
    app.add_exception_handler(
        RequestValidationError, _server.validation_exception_handler
    )

    @app.post("/query")
    async def _query(request: QueryRequest):  # pragma: no cover - never reached
        return {"response": "ok"}

    @app.post("/query/stream")
    async def _stream(request: QueryRequest):  # pragma: no cover - never reached
        return {"response": "ok"}

    @app.post("/query/data")
    async def _data(request: QueryRequest):  # pragma: no cover - never reached
        return {"status": "success"}

    # raise_server_exceptions=False so a handler that blows up shows up as the
    # 500 a real client would see, instead of surfacing here as a test error.
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("path", ["/query", "/query/stream"])
@pytest.mark.parametrize(
    "body",
    [
        {"query": "中"},  # below the weighted RAG minimum
        {"query": "ab"},  # below the plain minimum
        {"query": "   "},  # empty after stripping
        {"query": "", "mode": "bypass"},  # empty on the direct-LLM path
    ],
)
def test_query_refusals_are_422_with_a_serializable_body(client, path, body):
    response = client.post(path, json=body)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list) and detail
    assert detail[0]["type"] == "value_error"
    assert detail[0]["loc"][0] == "body"
    assert detail[0]["msg"].startswith("Value error, ")


@pytest.mark.parametrize(
    "body", [{"query": "中"}, {"query": "   "}, {"query": "", "mode": "bypass"}]
)
def test_query_data_refusals_keep_their_own_400_envelope(client, body):
    response = client.post("/query/data", json=body)

    assert response.status_code == 400
    payload = response.json()
    assert payload["status"] == "failure"
    assert payload["message"].startswith("Validation error: body")
    assert "Value error, " in payload["message"]
    assert payload["data"] == {} and payload["metadata"] == {}


def test_structural_errors_still_answer_422(client):
    """The path that always worked, pinned so the fix does not narrow it."""
    response = client.post("/query", json={})

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "missing"
