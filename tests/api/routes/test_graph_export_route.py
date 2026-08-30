"""Tests for GET /graph/export.

Unlike ``GET /graphs`` (a bounded, connected subgraph for the interactive
viewer), this dumps the whole graph via ``LightRAG.aexport_data`` for
offline review/backup. The route's own job -- writing to a server-managed
temp file rather than a client-supplied path, streaming it back, and
cleaning up afterward -- is what's under test here; ``aexport_data``'s own
export logic is covered separately in
``tests/utils/test_aexport_data_portable_relationships.py``.
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_graph_routes = importlib.import_module("lightrag.api.routers.graph_routes")
sys.argv = _original_argv

create_graph_routes = _graph_routes.create_graph_routes

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_HEADERS = {"X-API-Key": _API_KEY}


def _write_fixed_content(path: str, file_format: str, include_vector_data: bool):
    Path(path).write_text(
        f"format={file_format} include_vector_data={include_vector_data}",
        encoding="utf-8",
    )


def _build_client() -> TestClient:
    rag = SimpleNamespace(
        aexport_data=AsyncMock(side_effect=_write_fixed_content),
    )
    app = FastAPI()
    app.include_router(create_graph_routes(rag, api_key=_API_KEY))
    return TestClient(app), rag


def test_export_returns_the_generated_file_as_an_attachment():
    client, rag = _build_client()

    response = client.get("/graph/export", headers=_HEADERS)

    assert response.status_code == 200
    assert response.content == b"format=csv include_vector_data=False"
    assert "lightrag_graph_export.csv" in response.headers["content-disposition"]
    assert response.headers["content-type"].startswith("text/csv")
    rag.aexport_data.assert_awaited_once()
    _, kwargs = rag.aexport_data.await_args
    assert kwargs["file_format"] == "csv"
    assert kwargs["include_vector_data"] is False


def test_export_forwards_format_and_include_vector_data():
    client, rag = _build_client()

    response = client.get(
        "/graph/export?file_format=excel&include_vector_data=true",
        headers=_HEADERS,
    )

    assert response.status_code == 200
    assert response.content == b"format=excel include_vector_data=True"
    assert "lightrag_graph_export.xlsx" in response.headers["content-disposition"]
    _, kwargs = rag.aexport_data.await_args
    assert kwargs["file_format"] == "excel"
    assert kwargs["include_vector_data"] is True


def test_export_rejects_an_unsupported_format():
    client, _rag = _build_client()

    response = client.get("/graph/export?file_format=pdf", headers=_HEADERS)

    assert response.status_code == 422


def test_export_requires_auth():
    client, _rag = _build_client()

    response = client.get("/graph/export")

    assert response.status_code in (401, 403)


def test_export_cleans_up_the_temp_file_after_the_response(tmp_path):
    captured_path: dict[str, str] = {}

    async def _capture_and_write(path, file_format, include_vector_data):
        captured_path["path"] = path
        Path(path).write_text("content", encoding="utf-8")

    rag = SimpleNamespace(aexport_data=AsyncMock(side_effect=_capture_and_write))
    app = FastAPI()
    app.include_router(create_graph_routes(rag, api_key=_API_KEY))
    client = TestClient(app)

    response = client.get("/graph/export", headers=_HEADERS)

    assert response.status_code == 200
    assert "path" in captured_path
    # TestClient's background-task execution runs the FileResponse's cleanup
    # synchronously before returning, so the temp file is gone by the time
    # the request completes.
    assert not Path(captured_path["path"]).exists()


def test_export_500s_and_removes_the_temp_file_when_aexport_data_fails():
    captured_path: dict[str, str] = {}

    async def _capture_then_fail(path, file_format, include_vector_data):
        captured_path["path"] = path
        Path(path).write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    rag = SimpleNamespace(aexport_data=AsyncMock(side_effect=_capture_then_fail))
    app = FastAPI()
    app.include_router(create_graph_routes(rag, api_key=_API_KEY))
    client = TestClient(app)

    response = client.get("/graph/export", headers=_HEADERS)

    assert response.status_code == 500
    assert "path" in captured_path
    assert not Path(captured_path["path"]).exists()
