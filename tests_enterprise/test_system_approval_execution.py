from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from lightrag_enterprise.system import (
    ACTIVITY_CORE_GRAPH_MUTATE,
    ACTIVITY_DOCUMENT_DELETE,
    ACTIVITY_DOCUMENT_REINDEX,
    AccessControlService,
    ApprovalService,
    AuditService,
    MANAGER_ROLE,
    Principal,
)
from lightrag_enterprise.system.approval_execution import ApprovalActionExecutor, ApprovalExecutionOutcome
from lightrag_enterprise.system.permissions import permissions_for_roles
from lightrag_enterprise.system.repositories import InMemorySystemRepository
from lightrag_enterprise.system.router import create_system_router


class FakeRag:
    def __init__(self) -> None:
        self.deleted_doc_ids: list[str] = []

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        self.deleted_doc_ids.append(doc_id)


def _manager_principal() -> Principal:
    roles = (MANAGER_ROLE,)
    return Principal(
        user_id="usr_manager",
        subject="manager",
        tenant_id="default",
        is_master_global=False,
        roles=roles,
        workspace_ids=("default",),
        permission_version=1,
        permissions=permissions_for_roles(roles),
    )


def _client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    repo: InMemorySystemRepository,
    principal: Principal,
    rag: FakeRag,
    action_handlers=None,
) -> TestClient:
    import lightrag_enterprise.system.router as system_router

    approvals = ApprovalService(repo)
    audit = AuditService(repo)
    monkeypatch.setattr(system_router, "get_access_service", lambda: AccessControlService())
    monkeypatch.setattr(system_router, "get_approval_service", lambda: approvals)
    monkeypatch.setattr(system_router, "get_audit_service", lambda: audit)
    app = FastAPI()
    app.include_router(create_system_router(ApprovalActionExecutor(rag, action_handlers=action_handlers)))
    app.dependency_overrides[system_router.require_principal] = lambda: principal
    return TestClient(app)


async def _approval_for_delete(
    repo: InMemorySystemRepository,
    principal: Principal,
    *,
    document_id: str = "doc-1",
):
    return await ApprovalService(repo).request(
        principal=principal,
        action=ACTIVITY_DOCUMENT_DELETE,
        tenant_id="default",
        workspace_id="default",
        reason="Document deletion requires human approval.",
        payload={"document_id": document_id},
    )


@pytest.mark.asyncio
async def test_approval_approve_executes_document_delete_and_audits(monkeypatch):
    repo = InMemorySystemRepository()
    principal = _manager_principal()
    rag = FakeRag()
    approval = await _approval_for_delete(repo, principal)
    client = _client(monkeypatch, repo=repo, principal=principal, rag=rag)

    response = client.post(f"/approvals/{approval.approval_id}/approve")

    assert response.status_code == 200
    assert response.json()["status"] == "executed"
    assert rag.deleted_doc_ids == ["doc-1"]
    events = await repo.list_audit_events(tenant_id="default")
    approval_events = [event for event in events if event.approval_id == approval.approval_id]
    assert {event.action for event in approval_events} >= {
        "little_bull.approvals.approve",
        ACTIVITY_DOCUMENT_DELETE,
    }
    assert any(event.result == "executed" for event in approval_events)
    assert any(event.metadata.get("document_id") == "doc-1" for event in approval_events)


@pytest.mark.asyncio
async def test_approval_approve_is_idempotent_after_execution(monkeypatch):
    repo = InMemorySystemRepository()
    principal = _manager_principal()
    rag = FakeRag()
    approval = await _approval_for_delete(repo, principal)
    client = _client(monkeypatch, repo=repo, principal=principal, rag=rag)

    first_response = client.post(f"/approvals/{approval.approval_id}/approve")
    second_response = client.post(f"/approvals/{approval.approval_id}/approve")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert second_response.json()["status"] == "executed"
    assert rag.deleted_doc_ids == ["doc-1"]
    events = await repo.list_audit_events(tenant_id="default")
    assert any(
        event.action == "little_bull.approvals.approve"
        and event.result == "already_executed"
        and event.approval_id == approval.approval_id
        for event in events
    )


@pytest.mark.asyncio
async def test_approval_approve_unsupported_action_does_not_execute(monkeypatch):
    repo = InMemorySystemRepository()
    principal = _manager_principal()
    rag = FakeRag()
    approval = await ApprovalService(repo).request(
        principal=principal,
        action=ACTIVITY_CORE_GRAPH_MUTATE,
        tenant_id="default",
        workspace_id="default",
        reason="Core graph mutation requires approval.",
        payload={"document_id": "doc-1"},
    )
    client = _client(monkeypatch, repo=repo, principal=principal, rag=rag)

    response = client.post(f"/approvals/{approval.approval_id}/approve")

    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert rag.deleted_doc_ids == []


@pytest.mark.asyncio
async def test_approval_approve_executes_registered_reindex_handler(monkeypatch):
    repo = InMemorySystemRepository()
    principal = _manager_principal()
    rag = FakeRag()
    approval = await ApprovalService(repo).request(
        principal=principal,
        action=ACTIVITY_DOCUMENT_REINDEX,
        tenant_id="default",
        workspace_id="default",
        reason="Knowledge base reindex requires approval.",
        payload={"workspace_id": "default", "include_archived": True, "include_input_root": True},
    )
    executed: list[str] = []

    async def execute_reindex(*, approval, approvals, principal):
        executing = await approvals.begin_execution(approval.approval_id, principal)
        assert executing is not None
        executed.append(executing.workspace_id)
        done = await approvals.mark_executed(executing.approval_id, principal)
        return ApprovalExecutionOutcome(
            approval=done,
            audit_result="executed",
            action_executed=True,
            metadata={"workspace_id": executing.workspace_id, "queued_count": 1},
        )

    client = _client(
        monkeypatch,
        repo=repo,
        principal=principal,
        rag=rag,
        action_handlers={ACTIVITY_DOCUMENT_REINDEX: execute_reindex},
    )

    response = client.post(f"/approvals/{approval.approval_id}/approve")

    assert response.status_code == 200
    assert response.json()["status"] == "executed"
    assert executed == ["default"]
