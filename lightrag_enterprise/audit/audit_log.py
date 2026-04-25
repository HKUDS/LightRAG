from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    actor: str
    action: str
    tenant_id: str
    workspace: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data


class AuditSink(Protocol):
    async def write(
        self,
        *,
        actor: str,
        action: str,
        tenant_id: str,
        workspace: str,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        ...


@dataclass
class InMemoryAuditSink:
    events: list[AuditEvent] = field(default_factory=list)

    async def write(
        self,
        *,
        actor: str,
        action: str,
        tenant_id: str,
        workspace: str,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=str(uuid4()),
            actor=actor,
            action=action,
            tenant_id=tenant_id,
            workspace=workspace,
            metadata=metadata or {},
        )
        self.events.append(event)
        return event
