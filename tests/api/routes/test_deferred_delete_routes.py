"""Read-only operator discovery for durable deferred deletion journals."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.deletion_journal import (
    DeferredDeletionJournal,
    DeferredDeletionJournalStore,
    DeferredDeletionStage,
)

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv


class _Rag:
    workspace = "default"

    def __init__(self, working_dir: str) -> None:
        self.working_dir = working_dir


def _client(tmp_path) -> TestClient:
    app = FastAPI()
    app.include_router(
        _document_routes.create_document_routes(
            _Rag(str(tmp_path)), SimpleNamespace(), api_key="test-key"
        )
    )
    return TestClient(app)


def test_deferred_delete_list_and_status_expose_only_operator_safe_details(tmp_path):
    store = DeferredDeletionJournalStore(tmp_path, "default")
    retryable = DeferredDeletionJournal.new("retryable-001", ["doc-a", "doc-b"], True)
    retryable.aggregate_targets(
        entities={"Shared": ["chunk-a"]},
        relationships={("Shared", "Other"): ["chunk-a"]},
    )
    retryable.mark_rebuild_pending()
    retryable.mark_failed_retryable("cache evidence unavailable")
    store.save(retryable)

    committed = DeferredDeletionJournal.new("committed-001", ["doc-c"], False)
    committed.mark_rebuild_pending()
    committed.mark_finalization_pending()
    committed.mark_committed()
    store.save(committed)

    client = _client(tmp_path)
    headers = {"X-API-Key": "test-key"}
    listed = client.get("/documents/delete_document/deferred", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["total_count"] == 1
    assert listed.json()["journals"] == [
        {
            "attempt_count": 1,
            "batch_id": "retryable-001",
            "created_at": retryable.created_at,
            "current_document_index": 0,
            "document_count": 2,
            "entities_to_rebuild_count": 1,
            "error_detail": "cache evidence unavailable",
            "relationships_to_rebuild_count": 1,
            "stage": "FAILED_RETRYABLE",
            "updated_at": retryable.updated_at,
        }
    ]

    detail = client.get(
        "/documents/delete_document/deferred/retryable-001", headers=headers
    )
    assert detail.status_code == 200
    assert detail.json()["document_ids"] == ["doc-a", "doc-b"]
    assert detail.json()["delete_llm_cache"] is True
    assert detail.json()["entities_to_rebuild"] == {"Shared": ["chunk-a"]}
    assert detail.json()["relationships_to_rebuild"] == [
        {"chunk_ids": ["chunk-a"], "source": "Other", "target": "Shared"}
    ]
    assert "document_metadata" not in detail.json()

    missing = client.get(
        "/documents/delete_document/deferred/missing-001", headers=headers
    )
    assert missing.status_code == 404


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/documents/delete_document/deferred"),
        ("get", "/documents/delete_document/deferred/missing-001"),
        ("post", "/documents/delete_document/missing-001/resume"),
    ],
)
def test_deferred_delete_operator_endpoints_require_auth(tmp_path, method, path):
    response = getattr(_client(tmp_path), method)(path)

    assert response.status_code in {401, 403}


def test_deferred_delete_resume_returns_404_without_scheduling_unknown_batch(tmp_path):
    response = _client(tmp_path).post(
        "/documents/delete_document/missing-001/resume",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 404


def test_deferred_delete_routes_appear_in_openapi_with_response_models(tmp_path):
    spec = _client(tmp_path).get("/openapi.json").json()

    list_path = spec["paths"]["/documents/delete_document/deferred"]["get"]
    detail_path = spec["paths"]["/documents/delete_document/deferred/{batch_id}"]["get"]
    resume_path = spec["paths"]["/documents/delete_document/{batch_id}/resume"]["post"]
    assert list_path["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("DeferredDeletionJournalListResponse")
    assert detail_path["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("DeferredDeletionJournalDetailResponse")
    assert resume_path["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("ResumeDeferredDeleteResponse")


@pytest.mark.asyncio
async def test_resumed_commit_cleans_persisted_source_paths_when_requested(tmp_path):
    rag = _Rag(str(tmp_path))
    store = DeferredDeletionJournalStore(tmp_path, "default")
    journal = DeferredDeletionJournal.new("cleanup-on-resume", ["doc-a"], False, True)
    journal.source_file_paths = {"doc-a": "uploads/a.pdf"}
    journal.mark_rebuild_pending()
    journal.mark_finalization_pending()
    journal.mark_committed()
    store.save(journal)
    doc_manager = SimpleNamespace(input_dir=tmp_path / "input")

    with patch(
        "lightrag.api.routers.document_routes.delete_file_variants_by_file_path",
        return_value=(["uploads/a.pdf"], []),
    ) as delete_variants:
        await _document_routes.cleanup_deferred_delete_files(
            rag, doc_manager, journal.batch_id
        )
        # A repeated resume of the committed batch must not touch the path
        # again: it may now refer to a newly recreated user file.
        await _document_routes.cleanup_deferred_delete_files(
            rag, doc_manager, journal.batch_id
        )

    delete_variants.assert_called_once_with(doc_manager.input_dir, "uploads/a.pdf")
    persisted = store.load(journal.batch_id)
    assert persisted is not None
    assert persisted.source_file_cleanup_claimed is True


@pytest.mark.asyncio
async def test_initial_commit_uses_the_same_durable_file_cleanup_helper_once(tmp_path):
    rag = _Rag(str(tmp_path))
    doc_manager = SimpleNamespace(input_dir=tmp_path / "input")
    pipeline_status = {
        "busy": True,
        "destructive_busy": True,
        "history_messages": [],
        "request_pending": False,
    }

    with (
        patch(
            "lightrag.kg.shared_storage.get_namespace_data",
            new=AsyncMock(return_value=pipeline_status),
        ),
        patch(
            "lightrag.kg.shared_storage.get_namespace_lock",
            return_value=__import__("asyncio").Lock(),
        ),
        patch(
            "lightrag.api.routers.document_routes.run_deferred_batch_delete",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    batch_id="initial-commit", stage=DeferredDeletionStage.COMMITTED
                )
            ),
        ),
        patch(
            "lightrag.api.routers.document_routes.cleanup_deferred_delete_files",
            new=AsyncMock(),
        ) as cleanup,
    ):
        await _document_routes.background_delete_documents(
            rag, doc_manager, ["doc-a"], delete_file=True
        )

    cleanup.assert_awaited_once_with(rag, doc_manager, "initial-commit")


@pytest.mark.asyncio
async def test_resume_restores_deletion_pipeline_job_state_before_continuing(tmp_path):
    rag = _Rag(str(tmp_path))
    doc_manager = SimpleNamespace(input_dir=tmp_path / "input")
    pipeline_status = {
        "busy": False,
        "destructive_busy": False,
        "job_name": "",
        "history_messages": [],
    }

    async def resume_with_pipeline_guard(_rag, _batch_id):
        assert pipeline_status["busy"] is True
        assert pipeline_status["destructive_busy"] is True
        assert pipeline_status["job_name"] == "Deleting 1 Documents"
        return SimpleNamespace(stage=DeferredDeletionStage.COMMITTED, error_detail=None)

    with (
        patch(
            "lightrag.kg.shared_storage.get_namespace_data",
            new=AsyncMock(return_value=pipeline_status),
        ),
        patch(
            "lightrag.kg.shared_storage.get_namespace_lock",
            return_value=__import__("asyncio").Lock(),
        ),
        patch(
            "lightrag.api.routers.document_routes.resume_deferred_batch_delete",
            side_effect=resume_with_pipeline_guard,
        ),
        patch(
            "lightrag.api.routers.document_routes.cleanup_deferred_delete_files",
            new=AsyncMock(),
        ),
    ):
        await _document_routes.background_resume_deferred_delete(
            rag, doc_manager, "resume-after-restart"
        )
