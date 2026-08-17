"""HTTP contract for ``GET /documents/supported_file_types``.

The endpoint exposes the live upload allowlist
(``DocumentManager.supported_extensions``) plus the parser capability matrix
(``DocumentManager.engine_capabilities``) so the WebUI can pre-validate
filenames — including ``[engine]``-hinted ones — without uploading the file
first. Allowlist/routing *behaviour* is covered by the registry and routing
suites (e.g. ``test_supported_extensions_markdown.py``); these tests pin the
HTTP layer: auth, serialization, cache headers, and that the matrix follows
the dynamic registry instead of a hardcoded engine set.
"""

import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# document_routes parses argv at import time; guard it like the sibling tests.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_registry = importlib.import_module("lightrag.parser.registry")
sys.argv = _original_argv

create_document_routes = _document_routes.create_document_routes
DocumentManager = _document_routes.DocumentManager

pytestmark = pytest.mark.offline

_HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient over a fresh app; parser env cleared for a default deployment."""
    for var in (
        "LIGHTRAG_PARSER",
        "MINERU_API_MODE",
        "MINERU_LOCAL_ENDPOINT",
        "MINERU_API_TOKEN",
        "DOCLING_ENDPOINT",
    ):
        monkeypatch.delenv(var, raising=False)
    app = FastAPI()
    app.include_router(
        create_document_routes(
            SimpleNamespace(),
            DocumentManager(str(tmp_path)),
            api_key="test-key",
        )
    )
    return TestClient(app)


def test_default_deployment_contract(client):
    resp = client.get("/documents/supported_file_types", headers=_HEADERS)
    assert resp.status_code == 200

    body = resp.json()
    exts = body["supported_extensions"]
    assert isinstance(exts, list) and exts
    assert all(e.startswith(".") and e == e.lower() for e in exts)
    assert ".md" in exts and ".txt" in exts

    engines = body["engines"]
    assert set(engines) >= {"native", "legacy"}
    # External engines only join once their endpoint is configured.
    assert "mineru" not in engines and "docling" not in engines
    # Internal (non-user-selectable) engines are never advertised.
    assert "reuse" not in engines and "passthrough" not in engines
    assert engines["native"] == [".docx", ".md", ".textpack"]
    for suffixes in engines.values():
        assert all(s.startswith(".") and s == s.lower() for s in suffixes)


def test_requires_auth(client):
    resp = client.get("/documents/supported_file_types")
    assert resp.status_code in (401, 403)


def test_cache_headers(client):
    resp = client.get("/documents/supported_file_types", headers=_HEADERS)
    assert resp.headers["Cache-Control"] == "no-store"
    assert resp.headers["Vary"] == "LIGHTRAG-WORKSPACE"


def test_mineru_routing_splits_allowlist_and_matrix(client, monkeypatch):
    """A hint-only suffix appears in the matrix but not the bare allowlist."""
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://localhost:8888")
    monkeypatch.setenv("LIGHTRAG_PARSER", "jpg:mineru;*:legacy-R")

    body = client.get("/documents/supported_file_types", headers=_HEADERS).json()

    # Bare .jpg routes to mineru via the rule; bare .png falls through to
    # legacy which cannot parse it, so only the matrix advertises png.
    assert ".jpg" in body["supported_extensions"]
    assert ".png" not in body["supported_extensions"]
    assert ".png" in body["engines"]["mineru"]
    assert ".jpg" in body["engines"]["mineru"]


def test_matrix_tracks_dynamic_registry(client):
    """Third-party engines registered at runtime appear without code changes."""
    spec = _registry.ParserSpec(
        engine_name="fooengine",
        impl="x:Y",
        suffixes=frozenset({"foo"}),
        queue_group="fooengine",
    )
    try:
        _registry.register_parser(spec)
        body = client.get("/documents/supported_file_types", headers=_HEADERS).json()
        assert body["engines"]["fooengine"] == [".foo"]
    finally:
        _registry._REGISTRY.pop("fooengine", None)
