from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class CRMOrganization:
    organization_id: str
    name: str
    tenant_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CRMContact:
    contact_id: str
    tenant_id: str
    workspace: str
    name: str
    email: str | None = None
    organization_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CRMLead:
    lead_id: str
    tenant_id: str
    workspace: str
    title: str
    status: str = "new"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CRMOpportunity:
    opportunity_id: str
    tenant_id: str
    workspace: str
    title: str
    pipeline_stage: str
    contact_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CRMSLA:
    sla_id: str
    tenant_id: str
    name: str
    response_minutes: int
    resolution_minutes: int


@dataclass(frozen=True)
class CRMTicket:
    ticket_id: str
    tenant_id: str
    workspace: str
    title: str
    status: str = "open"
    priority: str = "normal"
    contact_id: str | None = None
    sla_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CRMTask:
    task_id: str
    tenant_id: str
    workspace: str
    title: str
    assignee: str | None = None
    due_at: datetime | None = None
    done: bool = False


@dataclass(frozen=True)
class CRMNote:
    note_id: str
    tenant_id: str
    workspace: str
    body: str
    target_type: str
    target_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
