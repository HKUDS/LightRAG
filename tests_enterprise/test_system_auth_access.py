import pytest

from lightrag_enterprise.system import (
    ACTIVITY_DOCUMENT_DELETE,
    ACTIVITY_QUERY,
    AccessControlService,
    ApprovalService,
    AuditService,
    Membership,
    OPERATOR_ROLE,
    SystemAuthService,
    Tenant,
    Workspace,
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

