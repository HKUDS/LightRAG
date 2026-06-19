from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kg.shared_storage import get_storage_keyed_lock
from lightrag.medical_kg.ontology import MedicalCategory, TOP_LEVEL_MEDICAL_CATEGORIES

from .models import ImprovementProposal
from .proposals import validate_proposal

APPLY_SOURCE = "kb_iteration_apply"
ACCEPTED_CHANGES_SOURCE_ARTIFACT = "accepted_changes.md"
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


async def apply_accepted_changes_to_graph(
    *,
    rag: Any,
    workspace: str,
    records: list[dict[str, Any]],
    proposals_by_id: dict[str, dict[str, Any]],
) -> AcceptedApplyResult:
    changes: list[ApplyChange] = []
    proposal_ids: list[str] = []
    graph = getattr(rag, "chunk_entity_relation_graph", None)
    graph_writes_happened = False

    for record in records:
        proposal_id = str(record.get("proposal_id", "")).strip()
        proposal_ids.append(proposal_id)
        proposal = proposals_by_id.get(proposal_id, {})
        proposal_type = str(
            proposal.get("type") or record.get("proposal_type") or ""
        ).strip()
        target = str(proposal.get("target") or record.get("proposal_target") or "").strip()
        evidence = [
            str(item)
            for item in proposal.get("evidence", [])
            if isinstance(item, str)
        ]

        if proposal_type != "add_hierarchy_branch":
            changes.append(
                ApplyChange(
                    proposal_id=proposal_id,
                    proposal_type=proposal_type,
                    target=target,
                    status=ApplyChangeStatus.UNSUPPORTED,
                    action=proposal_type,
                    evidence=evidence,
                    reason="Unsupported proposal type.",
                )
            )
            continue

        branch_key = branch_key_from_proposal({**record, **proposal})
        category = category_for_branch_key(branch_key)
        if category is None:
            changes.append(
                ApplyChange(
                    proposal_id=proposal_id,
                    proposal_type=proposal_type,
                    target=target,
                    status=ApplyChangeStatus.BLOCKED,
                    action="add_hierarchy_branch",
                    branch_key=branch_key,
                    evidence=evidence,
                    reason="Missing or unsupported branch key.",
                )
            )
            continue

        if graph is None:
            changes.append(
                ApplyChange(
                    proposal_id=proposal_id,
                    proposal_type=proposal_type,
                    target=target,
                    status=ApplyChangeStatus.BLOCKED,
                    action="add_hierarchy_branch",
                    branch_key=branch_key,
                    branch_label=category.label,
                    evidence=evidence,
                    reason="Graph storage is not available.",
                )
            )
            continue

        branch_exists = False
        branch_was_upserted = False
        async with get_storage_keyed_lock(
            branch_key,
            namespace=f"{workspace}:GraphDB",
            enable_logging=False,
        ):
            if await graph.has_node(branch_key):
                branch_exists = True
            else:
                await graph.upsert_nodes_batch(
                    [(branch_key, _branch_node_data(category, proposal_id))]
                )
                graph_writes_happened = True
                branch_was_upserted = True

        if branch_exists:
            changes.append(
                ApplyChange(
                    proposal_id=proposal_id,
                    proposal_type=proposal_type,
                    target=target,
                    status=ApplyChangeStatus.ALREADY_PRESENT,
                    action="add_hierarchy_branch",
                    branch_key=branch_key,
                    branch_label=category.label,
                    evidence=evidence,
                    reason="Branch node already exists.",
                )
            )
            continue

        if branch_was_upserted:
            changes.append(
                ApplyChange(
                    proposal_id=proposal_id,
                    proposal_type=proposal_type,
                    target=target,
                    status=ApplyChangeStatus.APPLIED,
                    action="add_hierarchy_branch",
                    branch_key=branch_key,
                    branch_label=category.label,
                    evidence=evidence,
                )
            )

    if graph_writes_happened:
        await graph.index_done_callback()

    return AcceptedApplyResult(
        workspace=workspace,
        applied_at=datetime.now(UTC).isoformat(),
        source_artifact=ACCEPTED_CHANGES_SOURCE_ARTIFACT,
        proposal_ids=proposal_ids,
        changes=changes,
    )


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


def _branch_node_data(
    category: MedicalCategory, proposal_id: str
) -> dict[str, Any]:
    return {
        "entity_id": category.key,
        "label": category.label,
        "entity_type": "MedicalGroup",
        "description": f"Top-level medical hierarchy branch: {category.label}",
        "source_id": APPLY_SOURCE,
        "file_path": ACCEPTED_CHANGES_SOURCE_ARTIFACT,
        "medical_group": category.key,
        "aliases": GRAPH_FIELD_SEP.join(category.aliases),
        "generated_by": APPLY_SOURCE,
        "accepted_proposal_ids": proposal_id,
    }


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
