from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from lightrag_enterprise.system import (
    ACTIVITY_CORE_GRAPH_CREATE,
    ACTIVITY_CORE_GRAPH_MUTATE,
    ACTIVITY_CORE_OLLAMA_USE,
    ACTIVITY_CORE_PIPELINE_MANAGE,
    ACTIVITY_CORE_QUERY_DATA,
    ACTIVITY_DOCUMENT_DELETE,
    ACTIVITY_DOCUMENT_UPLOAD,
    ACTIVITY_QUERY,
    AccessControlService,
    ApprovalService,
    AuditService,
    MASTER_ROLE,
    OPERATOR_ROLE,
    Principal,
)
from lightrag_enterprise.system.permissions import MANAGER_ROLE, permissions_for_roles
from lightrag_enterprise.system.repositories import InMemorySystemRepository


class FakeEnterpriseAuth:
    def __init__(self, principals):
        self.principals = principals

    async def has_users(self):
        return True

    async def principal_from_token(self, token):
        return self.principals[token]


def _principal(role: str, *, master: bool = False) -> Principal:
    roles = (MASTER_ROLE,) if master else (role,)
    return Principal(
        user_id=f"usr_{role}",
        subject=role,
        tenant_id="default",
        is_master_global=master,
        roles=roles,
        workspace_ids=("default",),
        permission_version=1,
        permissions=permissions_for_roles(roles),
    )


def _request(path: str = "/graph/entity/edit"):
    return SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path=path),
        headers={},
        state=SimpleNamespace(),
    )


def _response():
    return SimpleNamespace(headers={})


def _install_runtime(monkeypatch, repo, principals):
    import lightrag_enterprise.system.runtime as runtime

    monkeypatch.setattr(runtime, "little_bull_functional_enabled", lambda: True)
    monkeypatch.setattr(runtime, "get_system_auth_service", lambda: FakeEnterpriseAuth(principals))
    monkeypatch.setattr(runtime, "get_access_service", lambda: AccessControlService())
    monkeypatch.setattr(runtime, "get_audit_service", lambda: AuditService(repo))
    monkeypatch.setattr(runtime, "get_approval_service", lambda: ApprovalService(repo))
    monkeypatch.setattr(runtime, "approvals_enforced", lambda: True)


@pytest.mark.asyncio
async def test_core_mutable_route_blocks_enterprise_token_without_permission(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["pytest"])
    from lightrag.api.utils_api import get_combined_auth_dependency

    repo = InMemorySystemRepository()
    operator = _principal(OPERATOR_ROLE)
    _install_runtime(monkeypatch, repo, {"operator": operator})
    dependency = get_combined_auth_dependency(
        enterprise_activity=ACTIVITY_CORE_GRAPH_MUTATE,
        enterprise_requires_approval=True,
    )

    with pytest.raises(HTTPException) as exc:
        await dependency(request=_request(), response=_response(), token="operator")

    assert exc.value.status_code == 403
    events = await repo.list_audit_events(tenant_id="default")
    assert events[0].action == ACTIVITY_CORE_GRAPH_MUTATE
    assert events[0].result == "blocked"


@pytest.mark.asyncio
async def test_core_mutable_route_allows_master_token_and_audits(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["pytest"])
    from lightrag.api.utils_api import get_combined_auth_dependency

    repo = InMemorySystemRepository()
    master = _principal(MASTER_ROLE, master=True)
    _install_runtime(monkeypatch, repo, {"master": master})
    dependency = get_combined_auth_dependency(
        enterprise_activity=ACTIVITY_CORE_GRAPH_MUTATE,
        enterprise_requires_approval=True,
    )

    await dependency(request=_request(), response=_response(), token="master")

    events = await repo.list_audit_events(tenant_id="default")
    assert events[0].action == ACTIVITY_CORE_GRAPH_MUTATE
    assert events[0].result == "allowed"


@pytest.mark.asyncio
async def test_core_destructive_route_creates_pending_approval_for_manager(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["pytest"])
    from lightrag.api.utils_api import get_combined_auth_dependency

    repo = InMemorySystemRepository()
    manager = _principal(MANAGER_ROLE)
    _install_runtime(monkeypatch, repo, {"manager": manager})
    dependency = get_combined_auth_dependency(
        enterprise_activity=ACTIVITY_DOCUMENT_DELETE,
        enterprise_requires_approval=True,
    )

    with pytest.raises(HTTPException) as exc:
        await dependency(
            request=_request("/documents/delete_document"),
            response=_response(),
            token="manager",
        )

    assert exc.value.status_code == 409
    assert exc.value.detail["status"] == "pending_approval"
    approvals = await repo.list_approval_requests(tenant_id="default")
    events = await repo.list_audit_events(tenant_id="default")
    assert approvals[0].action == ACTIVITY_DOCUMENT_DELETE
    assert events[0].approval_id == approvals[0].approval_id
    assert events[0].result == "pending_approval"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("activity", "path", "token", "expected_result"),
    [
        (ACTIVITY_DOCUMENT_UPLOAD, "/documents/upload", "operator", "allowed"),
        (ACTIVITY_QUERY, "/query", "operator", "allowed"),
        (ACTIVITY_CORE_GRAPH_CREATE, "/graph/entity/create", "operator", "blocked"),
        (ACTIVITY_CORE_PIPELINE_MANAGE, "/documents/reprocess_failed", "operator", "blocked"),
        (ACTIVITY_CORE_QUERY_DATA, "/query/data", "operator", "blocked"),
        (ACTIVITY_CORE_OLLAMA_USE, "/api/chat", "operator", "blocked"),
        (ACTIVITY_CORE_GRAPH_CREATE, "/graph/entity/create", "manager", "allowed"),
        (ACTIVITY_CORE_PIPELINE_MANAGE, "/documents/reprocess_failed", "manager", "allowed"),
        (ACTIVITY_CORE_QUERY_DATA, "/query/data", "manager", "allowed"),
        (ACTIVITY_CORE_OLLAMA_USE, "/api/chat", "manager", "allowed"),
    ],
)
async def test_core_governance_for_remaining_activity_surface(
    monkeypatch,
    activity,
    path,
    token,
    expected_result,
):
    import sys
    monkeypatch.setattr(sys, "argv", ["pytest"])
    from lightrag.api.utils_api import get_combined_auth_dependency

    repo = InMemorySystemRepository()
    operator = _principal(OPERATOR_ROLE)
    manager = _principal(MANAGER_ROLE)
    _install_runtime(monkeypatch, repo, {"operator": operator, "manager": manager})
    dependency = get_combined_auth_dependency(enterprise_activity=activity)

    if expected_result == "blocked":
        with pytest.raises(HTTPException) as exc:
            await dependency(request=_request(path), response=_response(), token=token)
        assert exc.value.status_code == 403
    else:
        await dependency(request=_request(path), response=_response(), token=token)

    events = await repo.list_audit_events(tenant_id="default")
    assert events[0].action == activity
    assert events[0].result == expected_result


def test_core_router_wiring_uses_activity_specific_dependencies():
    repo_root = Path(__file__).resolve().parents[1]
    document_routes = (repo_root / "lightrag/api/routers/document_routes.py").read_text()
    graph_routes = (repo_root / "lightrag/api/routers/graph_routes.py").read_text()
    query_routes = (repo_root / "lightrag/api/routers/query_routes.py").read_text()
    ollama_api = (repo_root / "lightrag/api/routers/ollama_api.py").read_text()

    assert "enterprise_activity=ACTIVITY_DOCUMENT_UPLOAD" in document_routes
    assert "enterprise_activity=ACTIVITY_CORE_PIPELINE_MANAGE" in document_routes
    assert '"/scan", response_model=ScanResponse, dependencies=[Depends(document_upload_auth)]' in document_routes
    assert '"/reprocess_failed"' in document_routes
    assert "dependencies=[Depends(pipeline_manage_auth)]" in document_routes

    assert "enterprise_activity=ACTIVITY_CORE_GRAPH_CREATE" in graph_routes
    assert '@router.post("/graph/entity/create", dependencies=[Depends(graph_create_auth)])' in graph_routes
    assert '@router.post("/graph/relation/create", dependencies=[Depends(graph_create_auth)])' in graph_routes

    assert "enterprise_activity=ACTIVITY_QUERY" in query_routes
    assert "enterprise_activity=ACTIVITY_CORE_QUERY_DATA" in query_routes
    assert '"/query/data"' in query_routes
    assert "dependencies=[Depends(query_data_auth)]" in query_routes

    assert "enterprise_activity=ACTIVITY_CORE_OLLAMA_USE" in ollama_api
    assert '"/generate", dependencies=[Depends(ollama_use_auth)]' in ollama_api
    assert '"/chat", dependencies=[Depends(ollama_use_auth)]' in ollama_api
