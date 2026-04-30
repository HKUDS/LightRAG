from __future__ import annotations

from typing import Any

from .models import AuditEvent, Principal, new_id
from .repositories import SystemRepository


class AuditService:
    def __init__(self, repository: SystemRepository) -> None:
        self.repository = repository

    async def record(
        self,
        *,
        principal: Principal,
        action: str,
        tenant_id: str | None,
        workspace_id: str | None,
        result: str,
        approval_id: str | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=new_id("aud"),
            actor_user_id=principal.user_id,
            action=action,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            result=result,
            approval_id=approval_id,
            model=model,
            metadata=metadata or {},
        )
        return await self.repository.write_audit_event(event)

    async def list(
        self,
        *,
        tenant_id: str | None = None,
        workspace_id: str | None = None,
        workspace_ids: tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        return await self.repository.list_audit_events(
            tenant_id,
            workspace_id,
            limit,
            workspace_ids,
        )
