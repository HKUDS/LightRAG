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


@dataclass(frozen=True)
class QualityFinding:
    severity: str
    category: str
    message: str
    evidence: list[str] = field(default_factory=list)
    suggested_fix_type: str = "review"
    requires_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityScore:
    overall: int
    subscores: dict[str, int]
    metrics: dict[str, Any]
    details: dict[str, Any] = field(default_factory=dict)
    findings: list[QualityFinding] = field(default_factory=list)
    critical_blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "subscores": self.subscores,
            "metrics": self.metrics,
            "details": self.details,
            "findings": [finding.to_dict() for finding in self.findings],
            "critical_blockers": self.critical_blockers,
        }


@dataclass(frozen=True)
class ImprovementProposal:
    id: str
    type: str
    target: str
    proposed_change: str
    reason: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    risk: str = "medium"
    requires_approval: bool = True
    expected_metric_change: dict[str, int | float] = field(default_factory=dict)
    patch_candidate: str = ""
    judge: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
