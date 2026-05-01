from __future__ import annotations

import re
from typing import Any

from fastapi import HTTPException, Request, status

from .models import Principal
import lightrag_enterprise.system.runtime as runtime


def _workspace_from_request(request: Request, principal: Principal) -> str | None:
    workspace = request.headers.get("LIGHTRAG-WORKSPACE", "").strip()
    if workspace:
        return re.sub(r"[^a-zA-Z0-9_]", "_", workspace)
    if len(principal.workspace_ids) == 1:
        return principal.workspace_ids[0]
    return None


async def enforce_core_route_activity(
    request: Request,
    *,
    activity: str,
    require_approval: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    principal = getattr(request.state, "little_bull_principal", None)
    if principal is None:
        return

    workspace_id = _workspace_from_request(request, principal)
    tenant_id = principal.tenant_id
    audit_metadata = {
        "core_route": True,
        "method": request.method,
        "path": request.url.path,
        **(metadata or {}),
    }
    access = runtime.get_access_service().require(
        principal,
        activity=activity,
        workspace_id=workspace_id,
        require_approval=require_approval,
    )

    if not access.allowed:
        await runtime.get_audit_service().record(
            principal=principal,
            action=activity,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            result="blocked",
            metadata={**audit_metadata, "reason": access.reason},
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=access.reason)

    if (
        access.requires_approval
        and runtime.approvals_enforced()
        and not principal.is_master_global
    ):
        approval = await runtime.get_approval_service().request(
            principal=principal,
            action=activity,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            reason=f"{request.method} {request.url.path} requires approval.",
            payload=audit_metadata,
        )
        await runtime.get_audit_service().record(
            principal=principal,
            action=activity,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            result="pending_approval",
            approval_id=approval.approval_id,
            metadata=audit_metadata,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "status": "pending_approval",
                "message": "Approval is required before this core LightRAG action can run.",
                "approval": approval.to_dict(),
            },
        )

    await runtime.get_audit_service().record(
        principal=principal,
        action=activity,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        result="allowed",
        metadata=audit_metadata,
    )
