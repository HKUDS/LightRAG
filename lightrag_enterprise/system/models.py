from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SystemUser:
    user_id: str
    username: str
    password_hash: str
    display_name: str
    is_master_global: bool = False
    is_active: bool = True
    permission_version: int = 1
    created_at: datetime = field(default_factory=utc_now)

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("password_hash", None)
        data["created_at"] = self.created_at.isoformat()
        return data


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    name: str
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class Workspace:
    workspace_id: str
    tenant_id: str
    name: str
    slug: str
    description: str = ""
    privacy: str = "team"
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class Membership:
    membership_id: str
    user_id: str
    tenant_id: str
    workspace_id: str
    roles: tuple[str, ...]
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class Principal:
    user_id: str
    subject: str
    tenant_id: str | None
    is_master_global: bool
    roles: tuple[str, ...]
    workspace_ids: tuple[str, ...]
    permission_version: int
    permissions: frozenset[str]

    def can_access_workspace(self, workspace_id: str) -> bool:
        return self.is_master_global or workspace_id in self.workspace_ids

    def to_token_payload(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "sub": self.subject,
            "tenant_id": self.tenant_id,
            "is_master_global": self.is_master_global,
            "roles": list(self.roles),
            "workspace_ids": list(self.workspace_ids),
            "permission_version": self.permission_version,
            "permissions": sorted(self.permissions),
        }


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str
    requires_approval: bool = False


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    action: str
    actor_user_id: str
    tenant_id: str | None
    workspace_id: str | None
    payload_hash: str
    reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = field(default_factory=utc_now)
    decided_at: datetime | None = None
    decided_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["requested_at"] = self.requested_at.isoformat()
        data["decided_at"] = self.decided_at.isoformat() if self.decided_at else None
        return data


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    actor_user_id: str
    action: str
    tenant_id: str | None
    workspace_id: str | None
    result: str
    approval_id: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

