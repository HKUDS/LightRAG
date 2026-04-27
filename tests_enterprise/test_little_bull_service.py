from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from lightrag_enterprise.little_bull.models import LittleBullQueryRequest
from lightrag_enterprise.little_bull.service import LittleBullService
from lightrag_enterprise.system import (
    AccessControlService,
    ApprovalService,
    AuditService,
    SystemAuthService,
    Workspace,
)
from lightrag_enterprise.system.repositories import (
    InMemorySystemRepository,
    membership_for_master,
)


class FakeDocStatus:
    async def get_docs_paginated(self, **_kwargs):
        doc = SimpleNamespace(
            file_path="manual.pdf",
            status="processed",
            content_summary="Manual",
            content_length=42,
            updated_at="2026-04-27T00:00:00Z",
            created_at="2026-04-27T00:00:00Z",
            track_id="trk",
            chunks_count=1,
            metadata={},
        )
        return [("doc-1", doc)], 1

    async def get_all_status_counts(self):
        return {"processed": 1}


class FakeRag:
    def __init__(self):
        self.doc_status = FakeDocStatus()
        self.workspace = "default"

    async def aquery_llm(self, query, param):
        return {
            "llm_response": {"content": f"Answer for {query} in {param.mode}"},
            "data": {"references": [{"reference_id": "1", "file_path": "manual.pdf"}]},
        }


class FakeDocManager:
    def __init__(self, tmp_path: Path):
        self.input_dir = tmp_path

    def is_supported_file(self, _filename: str) -> bool:
        return True


async def _principal_and_service(tmp_path: Path):
    repo = InMemorySystemRepository()
    auth = SystemAuthService(repo, secret="test-secret")
    user = await auth.bootstrap_master(username="master", password="secret123")
    await repo.create_membership(membership_for_master(user.user_id))
    principal = await auth.principal_for_user(user)
    service = LittleBullService(
        rag=FakeRag(),
        doc_manager=FakeDocManager(tmp_path),
        repository=repo,
        access=AccessControlService(),
        audit=AuditService(repo),
        approvals=ApprovalService(repo),
    )
    return principal, service


@pytest.mark.asyncio
async def test_little_bull_lists_documents_and_audits(tmp_path):
    principal, service = await _principal_and_service(tmp_path)

    response = await service.list_documents(principal, workspace_id="default")

    assert response.total_count == 1
    assert response.documents[0].title == "manual.pdf"
    assert (await service.list_activity(principal, workspace_id="default"))[0].result == "success"


@pytest.mark.asyncio
async def test_little_bull_query_blocks_hosted_profile_for_private_data(tmp_path):
    principal, service = await _principal_and_service(tmp_path)

    with pytest.raises(Exception) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                query="private question",
                confidentiality="privado",
                model_profile="equilibrado",
            ),
        )

    assert "Private/local" in str(exc.value)


@pytest.mark.asyncio
async def test_little_bull_query_blocks_hosted_profile_when_workspace_has_private_data(tmp_path):
    principal, service = await _principal_and_service(tmp_path)
    await service.repository.set_policy(
        "little_bull.workspace_contains_private_data",
        True,
        tenant_id="default",
        workspace_id="default",
    )

    with pytest.raises(HTTPException) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                query="normal looking question",
                confidentiality="normal",
                model_profile="equilibrado",
            ),
        )

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_little_bull_blocks_unbacked_workspace_data_plane(tmp_path):
    principal, service = await _principal_and_service(tmp_path)
    await service.repository.create_workspace(
        Workspace(
            workspace_id="other",
            tenant_id="default",
            name="Other",
            slug="other",
        )
    )
    principal = await SystemAuthService(service.repository, secret="test-secret").principal_for_user(
        await service.repository.get_user(principal.user_id)
    )

    with pytest.raises(HTTPException) as exc:
        await service.list_documents(principal, workspace_id="other")

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_little_bull_delete_creates_pending_approval(tmp_path):
    principal, service = await _principal_and_service(tmp_path)

    response = await service.delete_document(
        principal,
        workspace_id="default",
        document_id="doc-1",
    )

    assert response["status"] == "pending_approval"
    assert response["approval"]["status"] == "pending"
