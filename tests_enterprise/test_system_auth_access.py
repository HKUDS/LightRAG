import pytest
from fastapi import HTTPException

from lightrag_enterprise.system import (
    ACTIVITY_DOCUMENT_DELETE,
    ACTIVITY_QUERY,
    AccessControlService,
    ApprovalService,
    AuditService,
    Membership,
    OPERATOR_ROLE,
    SystemAuthService,
    SystemUser,
    Tenant,
    Workspace,
    hash_password,
)
from lightrag_enterprise.system.models import new_id
from lightrag_enterprise.system.repositories import (
    InMemorySystemRepository,
    membership_for_master,
)


@pytest.mark.asyncio
async def test_master_bootstrap_token_contains_enterprise_claims():
    repo = InMemorySystemRepository()
    auth = SystemAuthService(repo, secret="test-secret")
    user = await auth.bootstrap_master(username="master", password="secret123")
    await repo.create_membership(membership_for_master(user.user_id))

    principal = await auth.principal_for_user(user)
    token = auth.create_token(principal)
    decoded = auth.decode_token(token)

    assert decoded["sub"] == "master"
    assert decoded["user_id"] == user.user_id
    assert decoded["is_master_global"] is True
    assert "default" in decoded["workspace_ids"]
    assert decoded["permission_version"] == user.permission_version


@pytest.mark.asyncio
async def test_enterprise_token_secret_is_required(monkeypatch):
    monkeypatch.delenv("LIGHTRAG_SYSTEM_TOKEN_SECRET", raising=False)
    monkeypatch.delenv("TOKEN_SECRET", raising=False)
    monkeypatch.delenv("LIGHTRAG_SYSTEM_ALLOW_INSECURE_DEV_SECRET", raising=False)
    repo = InMemorySystemRepository()
    auth = SystemAuthService(repo)
    user = await auth.bootstrap_master(username="master", password="secret123")
    await repo.create_membership(membership_for_master(user.user_id))
    principal = await auth.principal_for_user(user)

    with pytest.raises(RuntimeError):
        auth.create_token(principal)


@pytest.mark.asyncio
async def test_operator_permission_and_workspace_scope_are_enforced():
    repo = InMemorySystemRepository()
    auth = SystemAuthService(repo, secret="test-secret")
    tenant = await repo.create_tenant(Tenant(tenant_id="tenant-a", name="Tenant A"))
    workspace = await repo.create_workspace(
        Workspace(
            workspace_id="workspace-a",
            tenant_id=tenant.tenant_id,
            name="Workspace A",
            slug="workspace-a",
        )
    )
    user = await auth.bootstrap_master(username="master", password="secret123")
    object.__setattr__(user, "is_master_global", False)
    await repo.create_membership(
        Membership(
            membership_id=new_id("mbr"),
            user_id=user.user_id,
            tenant_id=tenant.tenant_id,
            workspace_id=workspace.workspace_id,
            roles=(OPERATOR_ROLE,),
        )
    )
    principal = await auth.principal_for_user(user)
    access = AccessControlService()

    assert access.require(principal, activity=ACTIVITY_QUERY, workspace_id="workspace-a").allowed
    assert not access.require(principal, activity=ACTIVITY_QUERY, workspace_id="workspace-b").allowed
    assert not access.require(
        principal, activity=ACTIVITY_DOCUMENT_DELETE, workspace_id="workspace-a"
    ).allowed


@pytest.mark.asyncio
async def test_approval_and_audit_are_repository_backed():
    repo = InMemorySystemRepository()
    auth = SystemAuthService(repo, secret="test-secret")
    user = await auth.bootstrap_master(username="master", password="secret123")
    await repo.create_membership(membership_for_master(user.user_id))
    principal = await auth.principal_for_user(user)

    approvals = ApprovalService(repo)
    audit = AuditService(repo)
    approval = await approvals.request(
        principal=principal,
        action=ACTIVITY_DOCUMENT_DELETE,
        tenant_id="default",
        workspace_id="default",
        reason="delete needs approval",
        payload={"doc_id": "doc-1"},
    )
    approved = await approvals.approve(approval.approval_id, principal)
    event = await audit.record(
        principal=principal,
        action=ACTIVITY_DOCUMENT_DELETE,
        tenant_id="default",
        workspace_id="default",
        result="pending_approval",
        approval_id=approval.approval_id,
    )

    assert approved.status.value == "approved"
    assert (await audit.list(tenant_id="default"))[0].event_id == event.event_id


@pytest.mark.asyncio
async def test_approval_decision_requires_scope_and_pending_status():
    repo = InMemorySystemRepository()
    auth = SystemAuthService(repo, secret="test-secret")
    master = await auth.bootstrap_master(username="master", password="secret123")
    await repo.create_membership(membership_for_master(master.user_id))
    master_principal = await auth.principal_for_user(master)
    other_workspace = await repo.create_workspace(
        Workspace(
            workspace_id="other",
            tenant_id="default",
            name="Other",
            slug="other",
        )
    )
    manager = await repo.create_user(
        SystemUser(
            user_id=new_id("usr"),
            username="manager",
            password_hash=hash_password("secret123"),
            display_name="Manager",
        )
    )
    await repo.create_membership(
        Membership(
            membership_id=new_id("mbr"),
            user_id=manager.user_id,
            tenant_id="default",
            workspace_id=other_workspace.workspace_id,
            roles=("gerente",),
        )
    )
    manager_principal = await auth.principal_for_user(manager)
    approvals = ApprovalService(repo)
    approval = await approvals.request(
        principal=master_principal,
        action=ACTIVITY_DOCUMENT_DELETE,
        tenant_id="default",
        workspace_id="default",
        reason="delete needs approval",
        payload={"doc_id": "doc-1"},
    )

    with pytest.raises(PermissionError):
        await approvals.approve(approval.approval_id, manager_principal)

    await approvals.approve(approval.approval_id, master_principal)
    with pytest.raises(ValueError):
        await approvals.reject(approval.approval_id, master_principal)


@pytest.mark.asyncio
async def test_core_auth_dependency_requires_enterprise_token_when_users_exist(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["pytest"])
    from lightrag.api.utils_api import get_combined_auth_dependency
    import lightrag_enterprise.system.runtime as runtime

    class FakeEnterpriseAuth:
        async def has_users(self):
            return True

    monkeypatch.setattr(runtime, "get_system_auth_service", lambda: FakeEnterpriseAuth())
    monkeypatch.setattr(runtime, "little_bull_functional_enabled", lambda: True)

    dependency = get_combined_auth_dependency()
    request = type(
        "Request",
        (),
        {"url": type("Url", (), {"path": "/query"})(), "state": type("State", (), {})()},
    )()
    response = type("Response", (), {"headers": {}})()

    with pytest.raises(HTTPException) as exc:
        await dependency(request=request, response=response, token=None)

    assert exc.value.status_code == 401
