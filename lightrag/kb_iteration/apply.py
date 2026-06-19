from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from lightrag.medical_kg.ontology import MedicalCategory, TOP_LEVEL_MEDICAL_CATEGORIES

from .models import ImprovementProposal
from .proposals import validate_proposal

APPLY_SOURCE = "kb_iteration_apply"
APPLY_RESULT_JSON = "accepted_changes_apply_result.json"
APPLY_RESULT_MARKDOWN = "accepted_changes_apply_result.md"
_PROPOSAL_FIELD_NAMES = {field_info.name for field_info in fields(ImprovementProposal)}
_REQUIRED_PROPOSAL_FIELD_NAMES = {
    "id",
    "type",
    "target",
    "proposed_change",
    "reason",
    "evidence",
    "confidence",
    "risk",
    "requires_approval",
    "expected_metric_change",
}


class ApplyChangeStatus(Enum):
    APPLIED = "applied"
    ALREADY_PRESENT = "already_present"
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ApplyChange:
    proposal_id: str
    proposal_type: str
    target: str
    status: ApplyChangeStatus
    action: str
    branch_key: str = ""
    branch_label: str = ""
    evidence: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
            "target": self.target,
            "status": self.status.value,
            "action": self.action,
            "branch_key": self.branch_key,
            "branch_label": self.branch_label,
            "evidence": self.evidence,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AcceptedApplyResult:
    workspace: str
    applied_at: str
    source_artifact: str
    proposal_ids: list[str]
    changes: list[ApplyChange] = field(default_factory=list)
    quality_before: dict[str, Any] = field(default_factory=dict)
    quality_after: dict[str, Any] = field(default_factory=dict)

    @property
    def applied_count(self) -> int:
        return sum(
            1 for change in self.changes if change.status == ApplyChangeStatus.APPLIED
        )

    @property
    def blocked_count(self) -> int:
        return sum(
            1 for change in self.changes if change.status == ApplyChangeStatus.BLOCKED
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "applied_at": self.applied_at,
            "source_artifact": self.source_artifact,
            "proposal_ids": self.proposal_ids,
            "applied_count": self.applied_count,
            "blocked_count": self.blocked_count,
            "changes": [change.to_dict() for change in self.changes],
            "quality_before": self.quality_before,
            "quality_after": self.quality_after,
        }


def load_proposals_by_id(package_dir: str | Path) -> dict[str, dict[str, Any]]:
    proposals_by_id: dict[str, dict[str, Any]] = {}
    for filename in ("approval_queue.md", "improvement_backlog.md"):
        for proposal in _load_proposals(Path(package_dir) / filename):
            proposal_id = proposal.get("id")
            if isinstance(proposal_id, str) and proposal_id not in proposals_by_id:
                proposals_by_id[proposal_id] = proposal
    return proposals_by_id


def branch_key_from_proposal(proposal: dict[str, Any]) -> str:
    text = "\n".join(
        str(proposal.get(field_name, ""))
        for field_name in ("target", "proposed_change", "reason")
    )
    for category in TOP_LEVEL_MEDICAL_CATEGORIES:
        pattern = (
            rf"(?<![A-Za-z0-9_.-]){re.escape(category.key)}"
            rf"(?!(?:[A-Za-z0-9_]|[.-][A-Za-z0-9_]))"
        )
        if re.search(pattern, text):
            return category.key
    return ""


def category_for_branch_key(branch_key: str) -> MedicalCategory | None:
    for category in TOP_LEVEL_MEDICAL_CATEGORIES:
        if category.key == branch_key:
            return category
    return None


def write_apply_result_artifacts(
    result: AcceptedApplyResult, package_dir: str | Path
) -> tuple[Path, Path]:
    target_dir = Path(package_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / APPLY_RESULT_JSON
    markdown_path = target_dir / APPLY_RESULT_MARKDOWN
    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_apply_result_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def render_apply_result_markdown(result: AcceptedApplyResult) -> str:
    lines = [
        "# Apply Result",
        "",
        f"- Applied: {result.applied_count}",
        f"- Blocked: {result.blocked_count}",
    ]
    before = _metric_value(result.quality_before, "hierarchy_missing_branch_count")
    after = _metric_value(result.quality_after, "hierarchy_missing_branch_count")
    if before is not None and after is not None:
        lines.append(f"- hierarchy_missing_branch_count: {before} -> {after}")

    lines.extend(["", "## Changes"])
    if not result.changes:
        lines.append("")
        lines.append("- No changes recorded.")
    for change in result.changes:
        branch = f" ({change.branch_key})" if change.branch_key else ""
        reason = f": {change.reason}" if change.reason else ""
        lines.append(f"- {change.proposal_id}: {change.status.value}{branch}{reason}")
    lines.append("")
    return "\n".join(lines)


def _load_proposals(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = _load_markdown_yaml(path.read_text(encoding="utf-8"))
    raw_proposals = payload.get("proposals", [])
    if not isinstance(raw_proposals, list):
        return []

    proposals: list[dict[str, Any]] = []
    for raw_proposal in raw_proposals:
        if not isinstance(raw_proposal, dict):
            continue
        proposal = dict(raw_proposal)
        if not isinstance(proposal.get("id"), str):
            continue
        if not _can_validate_proposal(proposal):
            continue
        proposal_fields = {
            key: value
            for key, value in proposal.items()
            if key in _PROPOSAL_FIELD_NAMES
        }
        try:
            validate_proposal(ImprovementProposal(**proposal_fields))
        except (TypeError, ValueError):
            continue
        proposals.append(proposal)
    return proposals


def _load_markdown_yaml(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    yaml_start = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^proposals\s*:", line)
        ),
        None,
    )
    if yaml_start is None:
        return {}
    try:
        payload = yaml.safe_load("\n".join(lines[yaml_start:])) or {}
    except yaml.YAMLError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _can_validate_proposal(proposal: dict[str, Any]) -> bool:
    return _REQUIRED_PROPOSAL_FIELD_NAMES.issubset(proposal)


def _metric_value(quality: dict[str, Any], metric_name: str) -> Any:
    metrics = quality.get("metrics")
    if not isinstance(metrics, dict) or metric_name not in metrics:
        return None
    return metrics[metric_name]
