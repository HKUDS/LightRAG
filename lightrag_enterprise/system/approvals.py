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

    def _check_decider_scope(
        self, approval: ApprovalRequest, principal: Principal
    ) -> None:
        if (
            not principal.is_master_global
            and ACTIVITY_APPROVAL_DECIDE not in principal.permissions
        ):
            raise PermissionError("Principal cannot approve requests.")
        if not principal.is_master_global:
            if approval.tenant_id != principal.tenant_id:
                raise PermissionError("Approval tenant is outside principal scope.")
            if (
                approval.workspace_id
                and approval.workspace_id not in principal.workspace_ids
            ):
                raise PermissionError("Approval workspace is outside principal scope.")

    def _can_decide(self, approval: ApprovalRequest, principal: Principal) -> None:
        self._check_decider_scope(approval, principal)
        if approval.status != ApprovalStatus.PENDING:
            raise ValueError("Approval request is no longer pending.")

    async def get(self, approval_id: str) -> ApprovalRequest | None:
        return await self.repository.get_approval_request(approval_id)

    async def approve(self, approval_id: str, principal: Principal) -> ApprovalRequest:
        approval = await self.repository.get_approval_request(approval_id)
        if approval is None:
            raise KeyError(approval_id)
        self._check_decider_scope(approval, principal)
        if approval.status in {
            ApprovalStatus.APPROVED,
            ApprovalStatus.EXECUTING,
            ApprovalStatus.EXECUTED,
            ApprovalStatus.FAILED,
        }:
            return approval
        if approval.status != ApprovalStatus.PENDING:
            raise ValueError("Approval request is no longer pending.")
        return await self.repository.update_approval_status(
            approval_id, ApprovalStatus.APPROVED, principal.user_id
        )

    async def begin_execution(
        self, approval_id: str, principal: Principal
    ) -> ApprovalRequest | None:
        return await self.repository.transition_approval_status(
            approval_id,
            ApprovalStatus.APPROVED,
            ApprovalStatus.EXECUTING,
            principal.user_id,
        )

    async def mark_executed(
        self, approval_id: str, principal: Principal
    ) -> ApprovalRequest:
        return await self.repository.update_approval_status(
            approval_id, ApprovalStatus.EXECUTED, principal.user_id
        )

    async def mark_failed(
        self, approval_id: str, principal: Principal
    ) -> ApprovalRequest:
        return await self.repository.update_approval_status(
            approval_id, ApprovalStatus.FAILED, principal.user_id
        )

    async def reject(self, approval_id: str, principal: Principal) -> ApprovalRequest:
        approval = await self.repository.get_approval_request(approval_id)
        if approval is None:
            raise KeyError(approval_id)
        self._can_decide(approval, principal)
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
        return await self.repository.list_approval_requests(
            tenant_id, workspace_id, status
        )
