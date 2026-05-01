from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from .approval_execution import ApprovalActionExecutor, ApprovalExecutionError
from .models import Membership, Tenant, Workspace, new_id
from .permissions import ACTIVITY_APPROVAL_READ, MANAGER_ROLE, OPERATOR_ROLE
from .policy_keys import PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY, stable_policy_hash
from .repositories import default_tenant_and_workspace, membership_for_master
from .runtime import (
    get_access_service,
    get_approval_service,
    get_audit_service,
    get_system_auth_service,
    get_system_repository,
    require_principal,
)


class BootstrapMasterRequest(BaseModel):
    username: str = Field(min_length=3)
    password: str = Field(min_length=8)
    display_name: str | None = None
    tenant_name: str = "Default"
    workspace_name: str = "Default"


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3)
    password: str = Field(min_length=8)
    display_name: str | None = None
    tenant_id: str = "default"
    workspace_id: str = "default"
    role: str = OPERATOR_ROLE


class HostedPrivateLlmExceptionPolicyRequest(BaseModel):
    schema_version: int = 1
    enabled: bool = True
    tenant_id: str = "default"
    workspace_id: str = "default"
    provider: str = "openrouter"
    binding: str = "openai"
    binding_host: str = "https://openrouter.ai/api/v1"
    allowed_model_ids: list[str] = Field(default_factory=list)
    allowed_confidentiality: list[str] = Field(
        default_factory=lambda: ["sensivel", "privado"]
    )
    expires_at: str | None = None
    approval_id: str | None = None
    reason: str = Field(min_length=8)
    ticket_id: str | None = None


def create_system_router(
    approval_executor: ApprovalActionExecutor | None = None,
) -> APIRouter:
    router = APIRouter(tags=["little-bull-system"])

    @router.post("/auth/login")
    async def auth_login(form_data: OAuth2PasswordRequestForm = Depends()):
        auth = get_system_auth_service()
        try:
            _, principal = await auth.authenticate(
                form_data.username, form_data.password
            )
            auth.require_token_secret()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect credentials"
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
        return {
            "access_token": auth.create_token(principal),
            "token_type": "bearer",
            "auth_mode": "enterprise",
            "principal": principal.to_token_payload(),
        }

    @router.get("/auth/me")
    async def auth_me(principal=Depends(require_principal)):
        return principal.to_token_payload()

    @router.post("/system/bootstrap-master")
    async def bootstrap_master(
        raw_request: Request,
        request: BootstrapMasterRequest,
        x_little_bull_bootstrap_token: str | None = Header(default=None),
    ):
        expected = os.getenv("LITTLE_BULL_BOOTSTRAP_TOKEN")
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="LITTLE_BULL_BOOTSTRAP_TOKEN must be set for HTTP bootstrap; use the CLI for local bootstrap.",
            )
        if x_little_bull_bootstrap_token != expected:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Invalid bootstrap token"
            )

        repo = get_system_repository()
        auth = get_system_auth_service()
        try:
            auth.require_token_secret()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
        tenant = Tenant(tenant_id="default", name=request.tenant_name)
        workspace = Workspace(
            workspace_id="default",
            tenant_id=tenant.tenant_id,
            name=request.workspace_name,
            slug="default",
            description="Local-first Little Bull workspace",
        )
        await repo.create_tenant(tenant)
        await repo.create_workspace(workspace)
        try:
            user = await auth.bootstrap_master(
                username=request.username,
                password=request.password,
                display_name=request.display_name,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        await repo.create_membership(membership_for_master(user.user_id))
        principal = await auth.principal_for_user(user)
        await get_audit_service().record(
            principal=principal,
            action="little_bull.system.bootstrap_master",
            tenant_id=tenant.tenant_id,
            workspace_id=workspace.workspace_id,
            result="success",
            metadata={
                "client_host": raw_request.client.host if raw_request.client else None
            },
        )
        return {
            "user": user.public_dict(),
            "tenant": tenant.__dict__,
            "workspace": workspace.__dict__,
            "access_token": auth.create_token(principal),
            "token_type": "bearer",
        }

    @router.get("/system/tenants")
    async def list_tenants(principal=Depends(require_principal)):
        if not principal.is_master_global:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="MASTER required"
            )
        return {
            "tenants": [
                tenant.__dict__
                for tenant in await get_system_repository().list_tenants()
            ]
        }

    @router.get("/system/workspaces")
    async def list_workspaces(principal=Depends(require_principal)):
        repo = get_system_repository()
        workspaces = await repo.list_workspaces(
            None if principal.is_master_global else principal.tenant_id
        )
        if not principal.is_master_global:
            workspaces = [
                workspace
                for workspace in workspaces
                if workspace.workspace_id in principal.workspace_ids
            ]
        return {"workspaces": [workspace.__dict__ for workspace in workspaces]}

    @router.post("/system/users")
    async def create_user(
        request: CreateUserRequest, principal=Depends(require_principal)
    ):
        if not principal.is_master_global:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="MASTER required"
            )
        from .auth import hash_password
        from .models import SystemUser

        if request.role not in {OPERATOR_ROLE, MANAGER_ROLE}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported role preset",
            )
        repo = get_system_repository()
        user = await repo.create_user(
            SystemUser(
                user_id=new_id("usr"),
                username=request.username,
                password_hash=hash_password(request.password),
                display_name=request.display_name or request.username,
            )
        )
        await repo.create_membership(
            Membership(
                membership_id=new_id("mbr"),
                user_id=user.user_id,
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                roles=(request.role,),
            )
        )
        return {"user": user.public_dict()}

    @router.post("/system/policies/private-data/hosted-llm-exception")
    async def set_hosted_private_llm_exception_policy(
        request: HostedPrivateLlmExceptionPolicyRequest,
        principal=Depends(require_principal),
    ):
        if not principal.is_master_global:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="MASTER required"
            )
        if request.schema_version != 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported policy schema_version",
            )
        normalized_confidentiality = sorted(
            {
                item.strip().lower()
                for item in request.allowed_confidentiality
                if item.strip()
            }
        )
        if not set(normalized_confidentiality).issubset({"sensivel", "privado"}):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported confidentiality scope",
            )
        if request.enabled and not request.expires_at:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="expires_at is required"
            )
        try:
            expires_at = (
                datetime.fromisoformat(request.expires_at.replace("Z", "+00:00"))
                if request.expires_at
                else datetime.now(timezone.utc)
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid expires_at"
            ) from exc
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if request.enabled and expires_at.astimezone(timezone.utc) <= datetime.now(
            timezone.utc
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="expires_at must be in the future",
            )
        previous_policy = await get_system_repository().get_policy(
            PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY,
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
        )
        policy = {
            "schema_version": request.schema_version,
            "enabled": request.enabled,
            "provider": request.provider.strip().lower(),
            "binding": request.binding.strip().lower(),
            "binding_host": request.binding_host.strip().rstrip("/"),
            "allowed_model_ids": sorted(
                {model.strip() for model in request.allowed_model_ids if model.strip()}
            ),
            "allowed_confidentiality": normalized_confidentiality,
            "expires_at": expires_at.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "approved_by": principal.user_id,
            "approved_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "approval_id": request.approval_id,
            "reason": request.reason,
            "ticket_id": request.ticket_id,
        }
        if request.enabled and not policy["allowed_model_ids"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one model id is required",
            )
        await get_system_repository().set_policy(
            PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY,
            policy,
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
        )
        policy_hash = stable_policy_hash(policy)
        await get_audit_service().record(
            principal=principal,
            action="little_bull.policies.private_data.hosted_llm_exception.update",
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            result="enabled" if request.enabled else "disabled",
            approval_id=request.approval_id,
            metadata={
                "policy_key": PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY,
                "enabled": request.enabled,
                "binding_host": policy["binding_host"],
                "allowed_model_ids": policy["allowed_model_ids"],
                "allowed_confidentiality": policy["allowed_confidentiality"],
                "expires_at": policy["expires_at"],
                "reason": policy["reason"],
                "ticket_id": policy["ticket_id"],
                "policy_hash": policy_hash,
                "previous_policy_hash": stable_policy_hash(previous_policy)
                if previous_policy
                else None,
            },
        )
        return {"policy": policy, "policy_hash": policy_hash}

    @router.get("/approvals")
    async def list_approvals(principal=Depends(require_principal)):
        decision = get_access_service().require(
            principal,
            activity=ACTIVITY_APPROVAL_READ,
        )
        if not decision.allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=decision.reason
            )
        approvals = await get_approval_service().list(
            tenant_id=None if principal.is_master_global else principal.tenant_id,
            workspace_id=None,
        )
        if not principal.is_master_global:
            approvals = [
                approval
                for approval in approvals
                if approval.workspace_id is None
                or approval.workspace_id in principal.workspace_ids
            ]
        return {"approvals": [approval.to_dict() for approval in approvals]}

    @router.post("/approvals/{approval_id}/approve")
    async def approve(approval_id: str, principal=Depends(require_principal)):
        audit_metadata = {}
        audit_result = "approved"
        try:
            approval = await get_approval_service().approve(approval_id, principal)
            if approval_executor is not None:
                outcome = await approval_executor.execute_if_supported(
                    approval=approval,
                    approvals=get_approval_service(),
                    principal=principal,
                )
                approval = outcome.approval
                audit_result = outcome.audit_result
                audit_metadata = outcome.metadata
        except PermissionError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        except ApprovalExecutionError as exc:
            await get_audit_service().record(
                principal=principal,
                action=exc.approval.action,
                tenant_id=exc.approval.tenant_id,
                workspace_id=exc.approval.workspace_id,
                result="execution_failed",
                approval_id=exc.approval.approval_id,
                metadata=exc.metadata,
            )
            await get_audit_service().record(
                principal=principal,
                action="little_bull.approvals.approve",
                tenant_id=exc.approval.tenant_id,
                workspace_id=exc.approval.workspace_id,
                result="execution_failed",
                approval_id=exc.approval.approval_id,
                metadata=exc.metadata,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Approval not found"
            ) from exc
        if audit_result == "executed":
            await get_audit_service().record(
                principal=principal,
                action=approval.action,
                tenant_id=approval.tenant_id,
                workspace_id=approval.workspace_id,
                result="executed",
                approval_id=approval.approval_id,
                metadata=audit_metadata,
            )
        await get_audit_service().record(
            principal=principal,
            action="little_bull.approvals.approve",
            tenant_id=approval.tenant_id,
            workspace_id=approval.workspace_id,
            result=audit_result,
            approval_id=approval.approval_id,
            metadata=audit_metadata,
        )
        return approval.to_dict()

    @router.post("/approvals/{approval_id}/reject")
    async def reject(approval_id: str, principal=Depends(require_principal)):
        try:
            approval = await get_approval_service().reject(approval_id, principal)
        except PermissionError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Approval not found"
            ) from exc
        await get_audit_service().record(
            principal=principal,
            action="little_bull.approvals.reject",
            tenant_id=approval.tenant_id,
            workspace_id=approval.workspace_id,
            result="rejected",
            approval_id=approval.approval_id,
        )
        return approval.to_dict()

    @router.get("/audit/events")
    async def list_audit_events(limit: int = 100, principal=Depends(require_principal)):
        decision = get_access_service().require(
            principal,
            activity="little_bull.audit.read",
            workspace_id=None,
        )
        if not decision.allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=decision.reason
            )
        events = await get_audit_service().list(
            tenant_id=None if principal.is_master_global else principal.tenant_id,
            workspace_id=None,
            workspace_ids=None
            if principal.is_master_global
            else principal.workspace_ids,
            limit=min(max(limit, 1), 500),
        )
        return {"events": [event.to_dict() for event in events]}

    return router


async def ensure_default_scope() -> None:
    from .db import get_database_url, run_schema

    if get_database_url():
        await run_schema()
    repo = get_system_repository()
    tenant, workspace = default_tenant_and_workspace()
    await repo.create_tenant(tenant)
    await repo.create_workspace(workspace)
