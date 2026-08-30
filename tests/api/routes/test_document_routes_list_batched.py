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
DocProcessingStatus = _base.DocProcessingStatus
DocStatus = _base.DocStatus

pytestmark = pytest.mark.offline


def _doc(status: DocStatus, suffix: str) -> DocProcessingStatus:
    return DocProcessingStatus(
        content_summary=f"{status.value} summary {suffix}",
        content_length=10,
        file_path=f"{suffix}.pdf",
        status=status,
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        metadata={},
    )


class _FakeDocStatusStorage:
    def __init__(self):
        self.docs = {
            "processed-doc": _doc(DocStatus.PROCESSED, "processed"),
            "pending-doc": _doc(DocStatus.PENDING, "pending"),
            "failed-doc": _doc(DocStatus.FAILED, "failed"),
            "processed-doc-2": _doc(DocStatus.PROCESSED, "processed-2"),
        }
        self.batch_calls = 0
        self.single_calls = 0

    async def get_docs_by_statuses(self, statuses, strict: bool = False):
        self.batch_calls += 1
        wanted = {s.value if hasattr(s, "value") else s for s in statuses}
        return {
            doc_id: doc
            for doc_id, doc in self.docs.items()
            if doc.status.value in wanted
        }

    async def get_docs_by_status(self, status):
        self.single_calls += 1
        return {
            doc_id: doc for doc_id, doc in self.docs.items() if doc.status == status
        }


_fake_doc_status = _FakeDocStatusStorage()
_app = FastAPI()
_app.include_router(
    create_document_routes(
        SimpleNamespace(doc_status=_fake_doc_status),
        SimpleNamespace(),
        api_key="test-key",
    )
)
_client = TestClient(_app)
_headers = {"X-API-Key": "test-key"}


def test_documents_listing_uses_single_batched_read():
    response = _client.get("/documents", headers=_headers)
    assert response.status_code == 200

    body = response.json()["statuses"]
    assert _fake_doc_status.batch_calls == 1
    assert _fake_doc_status.single_calls == 0

    # Documents are grouped by their own status.
    assert [d["id"] for d in body["processed"]] == [
        "processed-doc",
        "processed-doc-2",
    ]
    assert [d["id"] for d in body["pending"]] == ["pending-doc"]
    assert [d["id"] for d in body["failed"]] == ["failed-doc"]
    # Statuses with no documents are not emitted.
    assert "processing" not in body


def test_documents_listing_caps_total_documents():
    # 4 docs, cap at 1000 → all returned; total counts per bucket
    response = _client.get("/documents", headers=_headers)
    assert response.status_code == 200
    body = response.json()["statuses"]
    total = sum(len(docs) for docs in body.values())
    assert total == 4
