from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import ApprovalRequest, ApprovalStatus, Principal, new_id
from .permissions import ACTIVITY_APPROVAL_DECIDE
from .repositories import SystemRepository


def payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ApprovalService:
    def __init__(self, repository: SystemRepository) -> None:
        self.repository = repository

    async def request(
        self,
        *,
        principal: Principal,
        action: str,
        tenant_id: str | None,
        workspace_id: str | None,
        reason: str,
        payload: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        approval = ApprovalRequest(
            approval_id=new_id("apr"),
            action=action,
            actor_user_id=principal.user_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            payload_hash=payload_hash(payload or {}),
            reason=reason,
            metadata=payload or {},
        )
        return await self.repository.create_approval_request(approval)

    async def approve(self, approval_id: str, principal: Principal) -> ApprovalRequest:
        if not principal.is_master_global and ACTIVITY_APPROVAL_DECIDE not in principal.permissions:
            raise PermissionError("Principal cannot approve requests.")
        return await self.repository.update_approval_status(
            approval_id, ApprovalStatus.APPROVED, principal.user_id
        )

    async def reject(self, approval_id: str, principal: Principal) -> ApprovalRequest:
        if not principal.is_master_global and ACTIVITY_APPROVAL_DECIDE not in principal.permissions:
            raise PermissionError("Principal cannot reject requests.")
        return await self.repository.update_approval_status(
            approval_id, ApprovalStatus.REJECTED, principal.user_id
        )

    async def list(
        self,
        *,
        tenant_id: str | None = None,
        workspace_id: str | None = None,
        status: ApprovalStatus | None = None,
    ) -> list[ApprovalRequest]:
        return await self.repository.list_approval_requests(tenant_id, workspace_id, status)

