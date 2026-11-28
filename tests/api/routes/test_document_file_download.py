import importlib
import os
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

    # move_file_to_parsed_dir relocates every source file here once parsing
    # finishes -- it does not leave a copy behind in input_dir -- so this is
    # the normal on-disk location for a processed document's source, not an
    # edge case. Confirmed against a real lightrag-server run: uploading a
    # .txt file and downloading it back 404'd ("missing from disk") until the
    # handler was taught to check this directory too.
    parsed_dir = doc_manager.input_dir / "__parsed__"
    parsed_dir.mkdir()
    (parsed_dir / "archived.pdf").write_bytes(b"%PDF-1.4 archived content")

    docs = {
        "doc-with-file": _doc("report.pdf"),
        "doc-with-archived-file": _doc("archived.pdf"),
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


def test_download_finds_the_file_in_the_parsed_archive_dir(_client):
    response = _client.get("/documents/doc-with-archived-file/file", headers=_headers)

    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 archived content"
    assert "archived.pdf" in response.headers["content-disposition"]


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


def test_find_downloadable_source_file_prefers_newest_archived_variant(tmp_path):
    # A document reprocessed more than once can leave several numbered
    # variants in __parsed__ (report.pdf, report_001.pdf, ...); the helper
    # must pick the current one deterministically rather than whatever
    # os.scandir happens to yield first.
    find_downloadable_source_file = _document_routes.find_downloadable_source_file
    doc_manager = DocumentManager(str(tmp_path))
    parsed_dir = doc_manager.input_dir / "__parsed__"
    parsed_dir.mkdir()

    older = parsed_dir / "report.pdf"
    newer = parsed_dir / "report_001.pdf"
    older.write_bytes(b"stale")
    newer.write_bytes(b"current")
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_700_000_100, 1_700_000_100))

    found = find_downloadable_source_file(doc_manager.input_dir, "report.pdf")

    assert found == newer
    assert found.read_bytes() == b"current"
