import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_base = importlib.import_module("lightrag.base")
sys.argv = _original_argv

create_document_routes = _document_routes.create_document_routes
DocumentManager = _document_routes.DocumentManager
UNKNOWN_FILE_SOURCE = _document_routes.UNKNOWN_FILE_SOURCE
DocStatus = _base.DocStatus

pytestmark = pytest.mark.offline


def _doc(file_path: str, status: DocStatus = DocStatus.PROCESSED) -> dict:
    # DocStatusStorage.get_by_id returns the raw stored row (a plain dict),
    # not a DocProcessingStatus -- see json_doc_status_impl.py::get_by_id and
    # every other backend's get_by_id. The fake mirrors that contract so this
    # test would have caught the handler treating the result as an object.
    return {
        "content_summary": "a summary",
        "content_length": 10,
        "file_path": file_path,
        "status": status.value,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "metadata": {},
    }


class _FakeDocStatusStorage:
    def __init__(self, docs: dict[str, dict]):
        self.docs = docs

    async def get_by_id(self, id: str):
        return self.docs.get(id)


@pytest.fixture
def _client(tmp_path):
    doc_manager = DocumentManager(str(tmp_path))

    (doc_manager.input_dir / "report.pdf").write_bytes(b"%PDF-1.4 fake content")

    docs = {
        "doc-with-file": _doc("report.pdf"),
        "doc-missing-on-disk": _doc("vanished.pdf"),
        "doc-text-only": _doc(UNKNOWN_FILE_SOURCE),
    }
    fake_doc_status = _FakeDocStatusStorage(docs)

    app = FastAPI()
    app.include_router(
        create_document_routes(
            SimpleNamespace(doc_status=fake_doc_status),
            doc_manager,
            api_key="test-key",
        )
    )
    return TestClient(app)


_headers = {"X-API-Key": "test-key"}


def test_download_returns_the_source_file(_client):
    response = _client.get("/documents/doc-with-file/file", headers=_headers)

    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 fake content"
    assert "report.pdf" in response.headers["content-disposition"]


def test_download_requires_auth(_client):
    response = _client.get("/documents/doc-with-file/file")

    assert response.status_code in (401, 403)


def test_download_404s_for_unknown_doc_id(_client):
    response = _client.get("/documents/no-such-doc/file", headers=_headers)

    assert response.status_code == 404


def test_download_404s_when_document_has_no_source_file(_client):
    response = _client.get("/documents/doc-text-only/file", headers=_headers)

    assert response.status_code == 404


def test_download_404s_when_file_removed_from_disk(_client):
    response = _client.get("/documents/doc-missing-on-disk/file", headers=_headers)

    assert response.status_code == 404


def test_download_400s_for_whitespace_only_doc_id(_client):
    # "%20" is a non-empty path segment, so FastAPI routes it to {doc_id}="
    # ", which the handler's .strip() collapses to empty.
    response = _client.get("/documents/%20/file", headers=_headers)

    assert response.status_code == 400
