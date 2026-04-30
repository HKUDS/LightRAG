from fastapi.testclient import TestClient
import pytest

from lightrag_enterprise.system import (
    ACTIVITY_AUDIT_READ,
    AccessControlService,
    AuditService,
    MANAGER_ROLE,
    MASTER_ROLE,
    Principal,
)
from lightrag_enterprise.system.permissions import permissions_for_roles
from lightrag_enterprise.system.repositories import InMemorySystemRepository
from lightrag_enterprise.system.router import create_system_router


def _principal(role: str, *, workspace_ids: tuple[str, ...], master: bool = False) -> Principal:
    roles = (MASTER_ROLE,) if master else (role,)
    return Principal(
        user_id=f"usr_{role}",
        subject=role,
        tenant_id="default",
        is_master_global=master,
        roles=roles,
        workspace_ids=workspace_ids,
        permission_version=1,
        permissions=permissions_for_roles(roles),
    )


async def _seed_audit_events(repo: InMemorySystemRepository):
    audit = AuditService(repo)
    master = _principal(MASTER_ROLE, workspace_ids=("workspace-a", "workspace-b"), master=True)
    await audit.record(
        principal=master,
        action="little_bull.query",
        tenant_id="default",
        workspace_id="workspace-a",
        result="success",
        metadata={"marker": "visible"},
    )
    await audit.record(
        principal=master,
        action="little_bull.documents.delete",
        tenant_id="default",
        workspace_id="workspace-b",
        result="success",
        metadata={"marker": "leak"},
    )
    await audit.record(
        principal=master,
        action="little_bull.system",
        tenant_id="default",
        workspace_id=None,
        result="success",
        metadata={"marker": "tenant-wide"},
    )
    await audit.record(
        principal=master,
        action="little_bull.query",
        tenant_id="tenant-b",
        workspace_id="workspace-x",
        result="success",
        metadata={"marker": "other-tenant"},
    )


def _client_for_principal(repo: InMemorySystemRepository, principal: Principal) -> TestClient:
    from fastapi import FastAPI
    import lightrag_enterprise.system.router as system_router

    app = FastAPI()
    app.include_router(create_system_router())
    app.dependency_overrides[system_router.require_principal] = lambda: principal
    return TestClient(app)


@pytest.mark.asyncio
async def test_audit_events_manager_is_scoped_to_own_workspaces(monkeypatch):
    import lightrag_enterprise.system.router as system_router

    repo = InMemorySystemRepository()
    await _seed_audit_events(repo)
    monkeypatch.setattr(system_router, "get_access_service", lambda: AccessControlService())
    monkeypatch.setattr(system_router, "get_audit_service", lambda: AuditService(repo))
    manager = _principal(MANAGER_ROLE, workspace_ids=("workspace-a",))

    response = _client_for_principal(repo, manager).get("/audit/events")

    assert response.status_code == 200
    markers = [event["metadata"]["marker"] for event in response.json()["events"]]
    assert markers == ["visible"]


@pytest.mark.asyncio
async def test_audit_events_master_sees_all_workspaces(monkeypatch):
    import lightrag_enterprise.system.router as system_router

    repo = InMemorySystemRepository()
    await _seed_audit_events(repo)
    monkeypatch.setattr(system_router, "get_access_service", lambda: AccessControlService())
    monkeypatch.setattr(system_router, "get_audit_service", lambda: AuditService(repo))
    master = _principal(MASTER_ROLE, workspace_ids=("workspace-a", "workspace-b"), master=True)

    response = _client_for_principal(repo, master).get("/audit/events")

    assert response.status_code == 200
    markers = {event["metadata"]["marker"] for event in response.json()["events"]}
    assert {"visible", "leak", "tenant-wide", "other-tenant"} <= markers


@pytest.mark.asyncio
async def test_audit_events_manager_limit_is_applied_after_scope_filter(monkeypatch):
    import lightrag_enterprise.system.router as system_router

    repo = InMemorySystemRepository()
    audit = AuditService(repo)
    master = _principal(MASTER_ROLE, workspace_ids=("workspace-a", "workspace-b"), master=True)
    await audit.record(
        principal=master,
        action=ACTIVITY_AUDIT_READ,
        tenant_id="default",
        workspace_id="workspace-a",
        result="success",
        metadata={"marker": "visible-old"},
    )
    for index in range(3):
        await audit.record(
            principal=master,
            action=ACTIVITY_AUDIT_READ,
            tenant_id="default",
            workspace_id="workspace-b",
            result="success",
            metadata={"marker": f"foreign-new-{index}"},
        )
    monkeypatch.setattr(system_router, "get_access_service", lambda: AccessControlService())
    monkeypatch.setattr(system_router, "get_audit_service", lambda: audit)
    manager = _principal(MANAGER_ROLE, workspace_ids=("workspace-a",))

    response = _client_for_principal(repo, manager).get("/audit/events?limit=1")

    assert response.status_code == 200
    markers = [event["metadata"]["marker"] for event in response.json()["events"]]
    assert markers == ["visible-old"]
