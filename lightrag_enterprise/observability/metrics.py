from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class MetricEvent:
    name: str
    value: float
    tenant_id: str
    workspace: str
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MetricsRecorder:
    events: list[MetricEvent] = field(default_factory=list)

    def record(
        self,
        name: str,
        value: float,
        *,
        tenant_id: str,
        workspace: str,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MetricEvent:
        event = MetricEvent(
            name=name,
            value=value,
            tenant_id=tenant_id,
            workspace=workspace,
            tags=tags or {},
            metadata=metadata or {},
        )
        self.events.append(event)
        return event
