from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ConnectorAction:
    name: str
    tenant_id: str
    workspace: str
    payload: dict[str, Any] = field(default_factory=dict)
    requires_human_approval: bool = False


@dataclass(frozen=True)
class ConnectorResult:
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class EnterpriseConnector(Protocol):
    name: str
    allowed_actions: set[str]

    async def execute(self, action: ConnectorAction) -> ConnectorResult: ...
