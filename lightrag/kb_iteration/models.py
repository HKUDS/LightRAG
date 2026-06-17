from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SnapshotNode:
    id: str
    label: str
    entity_type: str
    description: str = ""
    source_id: str = ""
    file_path: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnapshotEdge:
    id: str
    source: str
    target: str
    keywords: str
    description: str = ""
    source_id: str = ""
    file_path: str = ""
    weight: float | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KGSnapshot:
    workspace: str
    generated_at: str
    source_files: list[str]
    nodes: list[SnapshotNode]
    edges: list[SnapshotEdge]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
