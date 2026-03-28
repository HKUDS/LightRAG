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
DocStatusStorage = _base.DocStatusStorage

pytestmark = pytest.mark.offline


def _doc(status: DocStatus, suffix: str) -> DocProcessingStatus:
    return DocProcessingStatus(
        content_summary=f"{status.value} summary",
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
            "parsing-doc": _doc(DocStatus.PARSING, "parsing"),
            "analyzing-doc": _doc(DocStatus.ANALYZING, "analyzing"),
        }

    async def get_docs_paginated(
        self,
        status_filter=None,
        status_filters=None,
        page=1,
        page_size=50,
        sort_field="updated_at",
        sort_direction="desc",
    ):
        selected_statuses = DocStatusStorage.resolve_status_filter_values(
            status_filter=status_filter,
            status_filters=status_filters,
        )
        documents = [
            (doc_id, doc)
            for doc_id, doc in self.docs.items()
            if selected_statuses is None or doc.status.value in selected_statuses
        ]
        return documents[:page_size], len(documents)

    async def get_all_status_counts(self):
        return {"processed": 1, "parsing": 1, "analyzing": 1}


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


def test_documents_paginated_accepts_status_filter():
    response = _client.post(
        "/documents/paginated",
        headers=_headers,
        json={
            "status_filter": "processed",
            "page": 1,
            "page_size": 10,
            "sort_field": "updated_at",
            "sort_direction": "desc",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pagination"]["total_count"] == 1
    assert [doc["id"] for doc in payload["documents"]] == ["processed-doc"]


def test_documents_paginated_status_filters_override_status_filter():
    response = _client.post(
        "/documents/paginated",
        headers=_headers,
        json={
            "status_filter": "processed",
            "status_filters": ["parsing", "analyzing"],
            "page": 1,
            "page_size": 10,
            "sort_field": "updated_at",
            "sort_direction": "desc",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pagination"]["total_count"] == 2
    assert [doc["id"] for doc in payload["documents"]] == [
        "parsing-doc",
        "analyzing-doc",
    ]
