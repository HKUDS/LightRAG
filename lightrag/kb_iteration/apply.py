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

from .medical_schema import CANONICAL_MEDICAL_RELATION_IDS
from .models import ImprovementProposal
from .proposals import validate_proposal

APPLY_SOURCE = "kb_iteration_apply"
ACCEPTED_CHANGES_SOURCE_ARTIFACT = "accepted_changes.md"
APPLY_RESULT_JSON = "accepted_changes_apply_result.json"
APPLY_RESULT_MARKDOWN = "accepted_changes_apply_result.md"
VALUE_LIKE_RELATION_KEYWORDS = frozenset(
    {
        "has_value",
        "value",
        "值",
        "剂量用法",
        "dosage_usage",
        "dose_usage",
        "has_result",
        "has_dosing_regimen",
    }
)
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
    for field_name in ("proposed_change", "target", "reason"):
        text = str(proposal.get(field_name, ""))
        matches: list[tuple[int, str]] = []
        for category in TOP_LEVEL_MEDICAL_CATEGORIES:
            pattern = (
                rf"(?<![A-Za-z0-9_.-]){re.escape(category.key)}"
                rf"(?!(?:[A-Za-z0-9_]|[.-][A-Za-z0-9_]))"
            )
            if match := re.search(pattern, text):
                matches.append((match.start(), category.key))
        if matches:
            return min(matches, key=lambda item: item[0])[1]
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
        if proposal_id not in proposals_by_id:
            changes.append(
                ApplyChange(
                    proposal_id=proposal_id,
                    proposal_type="",
                    target="",
                    status=ApplyChangeStatus.BLOCKED,
                    action="resolve_proposal_definition",
                    reason="Accepted proposal definition was not found.",
                )
            )
            continue

        proposal = proposals_by_id[proposal_id]
        proposal_type = str(
            proposal.get("type") or record.get("proposal_type") or ""
        ).strip()
        target = str(proposal.get("target") or record.get("proposal_target") or "").strip()
        evidence = [
            str(item)
            for item in proposal.get("evidence", [])
            if isinstance(item, str)
        ]

        if proposal_type == "medical_relation_schema_migration":
            if graph is None:
                changes.append(
                    _blocked_change(
                        proposal_id,
                        proposal,
                        "replace_relation",
                        "Graph storage is not available.",
                    )
                )
                continue
            change = await _apply_medical_relation_schema_migration(
                graph=graph,
                workspace=workspace,
                proposal_id=proposal_id,
                proposal=proposal,
            )
            if change.status == ApplyChangeStatus.APPLIED:
                graph_writes_happened = True
            changes.append(change)
            continue

        if proposal_type == "value_node_to_qualifier":
            if graph is None:
                changes.append(
                    _blocked_change(
                        proposal_id,
                        proposal,
                        "value_node_to_qualifier",
                        "Graph storage is not available.",
                    )
                )
                continue
            change = await _apply_value_node_to_qualifier(
                graph=graph,
                workspace=workspace,
                proposal_id=proposal_id,
                proposal=proposal,
            )
            if change.status == ApplyChangeStatus.APPLIED:
                graph_writes_happened = True
            changes.append(change)
            continue

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


async def _apply_medical_relation_schema_migration(
    *,
    graph: Any,
    workspace: str,
    proposal_id: str,
    proposal: dict[str, Any],
) -> ApplyChange:
    action_payload = proposal.get("action_payload")
    if not isinstance(action_payload, dict):
        return _blocked_change(
            proposal_id,
            proposal,
            "replace_relation",
            "Incomplete action_payload for replace_relation.",
        )

    action = action_payload.get("action")
    if action != "replace_relation":
        return _blocked_change(
            proposal_id,
            proposal,
            "replace_relation",
            "Unsupported medical relation schema migration action.",
        )

    required_fields = (
        "edge_id",
        "expected_source",
        "expected_target",
        "new_source",
        "new_target",
        "new_keywords",
        "current_keywords",
    )
    values = _required_string_payload_values(action_payload, required_fields)
    if values is None:
        return _blocked_change(
            proposal_id,
            proposal,
            "replace_relation",
            "Incomplete action_payload for replace_relation.",
        )

    expected_source = values["expected_source"]
    expected_target = values["expected_target"]
    new_source = values["new_source"]
    new_target = values["new_target"]
    new_keywords = values["new_keywords"]
    edge_id = values["edge_id"]
    if new_keywords not in CANONICAL_MEDICAL_RELATION_IDS:
        return _blocked_change(
            proposal_id,
            proposal,
            "replace_relation",
            "new_keywords must be a canonical medical relation id.",
        )

    async with get_storage_keyed_lock(
        sorted({expected_source, expected_target, new_source, new_target}),
        namespace=f"{workspace}:GraphDB",
        enable_logging=False,
    ):
        edge_data = await graph.get_edge(expected_source, expected_target)
        if edge_data is None:
            new_edge_data = await graph.get_edge(new_source, new_target)
            if _relation_replacement_already_present(
                new_edge_data,
                new_keywords,
                proposal_id,
            ):
                return _already_present_change(
                    proposal_id,
                    proposal,
                    "replace_relation",
                    "Relation replacement already applied.",
                )
            return _blocked_change(
                proposal_id,
                proposal,
                "replace_relation",
                "Expected edge was not found.",
            )

        if _relation_replacement_already_present(
            edge_data,
            new_keywords,
            proposal_id,
        ):
            return _already_present_change(
                proposal_id,
                proposal,
                "replace_relation",
                "Relation replacement already applied.",
            )

        if not _edge_identity_matches(edge_data, edge_id):
            return _blocked_change(
                proposal_id,
                proposal,
                "replace_relation",
                "Expected edge identity does not match proposal.",
            )

        if not _edge_orientation_matches(edge_data, expected_source, expected_target):
            return _blocked_change(
                proposal_id,
                proposal,
                "replace_relation",
                "Expected edge orientation does not match proposal.",
            )

        current_keywords = action_payload.get("current_keywords")
        if isinstance(current_keywords, str) and current_keywords.strip():
            edge_keywords = str(edge_data.get("keywords", "")).strip()
            if edge_keywords != current_keywords.strip():
                return _blocked_change(
                    proposal_id,
                    proposal,
                    "replace_relation",
                    "Current edge keywords no longer match proposal.",
                )

        if not await graph.has_node(new_source) or not await graph.has_node(new_target):
            return _blocked_change(
                proposal_id,
                proposal,
                "replace_relation",
                "Replacement relation endpoint was not found.",
            )

        target_edge_exists = await graph.has_edge(new_source, new_target)
        target_edge_data = (
            await graph.get_edge(new_source, new_target)
            if target_edge_exists
            else None
        )
        target_already_applied = _relation_replacement_already_present(
            target_edge_data,
            new_keywords,
            proposal_id,
        )
        if target_already_applied:
            await graph.remove_edges([(expected_source, expected_target)])
            return ApplyChange(
                proposal_id=proposal_id,
                proposal_type=str(proposal.get("type", "")),
                target=str(proposal.get("target", "")),
                status=ApplyChangeStatus.APPLIED,
                action="replace_relation",
                evidence=_proposal_evidence(proposal),
            )

        same_storage_edge = target_edge_exists and target_edge_data == edge_data
        if target_edge_exists and not same_storage_edge:
            return _blocked_change(
                proposal_id,
                proposal,
                "replace_relation",
                "Replacement relation target edge already exists.",
            )

        new_edge_data = dict(edge_data)
        new_edge_data["keywords"] = new_keywords
        new_edge_data["normalized_by"] = APPLY_SOURCE
        new_edge_data["accepted_proposal_ids"] = _append_graph_field(
            new_edge_data.get("accepted_proposal_ids"),
            proposal_id,
        )
        qualifiers = action_payload.get("qualifiers")
        if qualifiers:
            new_edge_data["qualifiers"] = json.dumps(
                qualifiers,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        await graph.upsert_edge(
            new_source,
            new_target,
            _scalar_graph_payload(new_edge_data),
        )
        if not same_storage_edge:
            await graph.remove_edges([(expected_source, expected_target)])

    return ApplyChange(
        proposal_id=proposal_id,
        proposal_type=str(proposal.get("type", "")),
        target=str(proposal.get("target", "")),
        status=ApplyChangeStatus.APPLIED,
        action="replace_relation",
        evidence=_proposal_evidence(proposal),
    )


async def _apply_value_node_to_qualifier(
    *,
    graph: Any,
    workspace: str,
    proposal_id: str,
    proposal: dict[str, Any],
) -> ApplyChange:
    action = "value_node_to_qualifier"
    action_payload = proposal.get("action_payload")
    if not isinstance(action_payload, dict):
        return _blocked_change(
            proposal_id,
            proposal,
            action,
            "Incomplete action_payload for value_node_to_qualifier.",
        )

    required_fields = (
        "value_node_id",
        "carrier_edge_source",
        "carrier_edge_target",
        "qualifier_key",
        "qualifier_value",
    )
    values = _required_string_payload_values(action_payload, required_fields)
    if values is None:
        return _blocked_change(
            proposal_id,
            proposal,
            action,
            "Incomplete action_payload for value_node_to_qualifier.",
        )

    value_node_id = values["value_node_id"]
    carrier_source = values["carrier_edge_source"]
    carrier_target = values["carrier_edge_target"]
    qualifier_key = values["qualifier_key"]
    qualifier_value = values["qualifier_value"]

    if value_node_id in {carrier_source, carrier_target}:
        return _blocked_change(
            proposal_id,
            proposal,
            action,
            "Value node overlaps carrier edge endpoint.",
        )

    async with get_storage_keyed_lock(
        sorted({value_node_id, carrier_source, carrier_target}),
        namespace=f"{workspace}:GraphDB",
        enable_logging=False,
    ):
        carrier_edge_data = await graph.get_edge(carrier_source, carrier_target)
        value_node_exists = await graph.has_node(value_node_id)
        already_applied = _value_node_qualifier_already_present(
            carrier_edge_data,
            qualifier_key,
            qualifier_value,
            proposal_id,
        )
        if already_applied and not value_node_exists:
            return _already_present_change(
                proposal_id,
                proposal,
                action,
                "Value node qualifier migration already applied.",
            )

        if not value_node_exists:
            return _blocked_change(
                proposal_id,
                proposal,
                action,
                "Value node was not found.",
            )

        value_node_degree = await graph.node_degree(value_node_id)
        if value_node_degree != 1:
            return _blocked_change(
                proposal_id,
                proposal,
                action,
                f"Value node degree must be 1, found {value_node_degree}.",
            )

        incident_edge = await _value_node_incident_edge(
            graph,
            value_node_id,
            carrier_source,
            carrier_target,
        )
        if incident_edge is None:
            return _blocked_change(
                proposal_id,
                proposal,
                action,
                "Value node incident edge does not match carrier edge endpoints.",
            )

        incident_edge_data = await graph.get_edge(*incident_edge)
        if not _edge_has_value_like_keyword(incident_edge_data):
            return _blocked_change(
                proposal_id,
                proposal,
                action,
                "Value node incident edge is not value-like.",
            )

        if carrier_edge_data is None:
            return _blocked_change(
                proposal_id,
                proposal,
                action,
                "Carrier edge was not found.",
            )

        new_edge_data = dict(carrier_edge_data)
        new_edge_data[f"qualifier_{qualifier_key}"] = qualifier_value
        new_edge_data["accepted_proposal_ids"] = _append_graph_field(
            new_edge_data.get("accepted_proposal_ids"),
            proposal_id,
        )
        await graph.upsert_edge(
            carrier_source,
            carrier_target,
            _scalar_graph_payload(new_edge_data),
        )
        await graph.remove_nodes([value_node_id])

    return ApplyChange(
        proposal_id=proposal_id,
        proposal_type=str(proposal.get("type", "")),
        target=str(proposal.get("target", "")),
        status=ApplyChangeStatus.APPLIED,
        action=action,
        evidence=_proposal_evidence(proposal),
    )


def _already_present_change(
    proposal_id: str,
    proposal: dict[str, Any],
    action: str,
    reason: str,
) -> ApplyChange:
    return ApplyChange(
        proposal_id=proposal_id,
        proposal_type=str(proposal.get("type", "")),
        target=str(proposal.get("target", "")),
        status=ApplyChangeStatus.ALREADY_PRESENT,
        action=action,
        evidence=_proposal_evidence(proposal),
        reason=reason,
    )


def _blocked_change(
    proposal_id: str,
    proposal: dict[str, Any],
    action: str,
    reason: str,
) -> ApplyChange:
    return ApplyChange(
        proposal_id=proposal_id,
        proposal_type=str(proposal.get("type", "")),
        target=str(proposal.get("target", "")),
        status=ApplyChangeStatus.BLOCKED,
        action=action,
        evidence=_proposal_evidence(proposal),
        reason=reason,
    )


def _proposal_evidence(proposal: dict[str, Any]) -> list[str]:
    return [
        str(item)
        for item in proposal.get("evidence", [])
        if isinstance(item, str)
    ]


def _edge_identity_matches(edge_data: dict[str, Any], edge_id: str) -> bool:
    for field_name in ("id", "edge_id", "relation_id"):
        if field_name in edge_data and str(edge_data[field_name]).strip() != edge_id:
            return False
    return True


def _edge_orientation_matches(
    edge_data: dict[str, Any],
    expected_source: str,
    expected_target: str,
) -> bool:
    for source_field, target_field in (
        ("source_node_id", "target_node_id"),
        ("source", "target"),
    ):
        if source_field in edge_data or target_field in edge_data:
            if str(edge_data.get(source_field, "")).strip() != expected_source:
                return False
            if str(edge_data.get(target_field, "")).strip() != expected_target:
                return False
    return True


def _relation_replacement_already_present(
    edge_data: dict[str, Any] | None,
    new_keywords: str,
    proposal_id: str,
) -> bool:
    if edge_data is None:
        return False
    return (
        str(edge_data.get("keywords", "")).strip() == new_keywords
        and _graph_field_contains(edge_data.get("accepted_proposal_ids"), proposal_id)
    )


def _value_node_qualifier_already_present(
    edge_data: dict[str, Any] | None,
    qualifier_key: str,
    qualifier_value: str,
    proposal_id: str,
) -> bool:
    if edge_data is None:
        return False
    return (
        str(edge_data.get(f"qualifier_{qualifier_key}", "")).strip()
        == qualifier_value
        and _graph_field_contains(edge_data.get("accepted_proposal_ids"), proposal_id)
    )


async def _value_node_incident_edge(
    graph: Any,
    value_node_id: str,
    carrier_source: str,
    carrier_target: str,
) -> tuple[str, str] | None:
    incident_edges = await graph.get_node_edges(value_node_id)
    if not incident_edges:
        return None
    if len(incident_edges) != 1:
        return None
    source_node_id, target_node_id = incident_edges[0]
    if {source_node_id, target_node_id} not in (
        {value_node_id, carrier_source},
        {value_node_id, carrier_target},
    ):
        return None
    return source_node_id, target_node_id


def _edge_has_value_like_keyword(edge_data: dict[str, Any] | None) -> bool:
    if edge_data is None:
        return False
    keywords = {
        keyword.strip()
        for keyword in str(edge_data.get("keywords", "")).split(GRAPH_FIELD_SEP)
        if keyword.strip()
    }
    return bool(keywords & VALUE_LIKE_RELATION_KEYWORDS)


def _required_string_payload_values(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
) -> dict[str, str] | None:
    values: dict[str, str] = {}
    for field_name in field_names:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            return None
        values[field_name] = value.strip()
    return values


def _append_graph_field(existing: Any, new_value: str) -> str:
    values = [
        value.strip()
        for value in str(existing or "").split(GRAPH_FIELD_SEP)
        if value.strip()
    ]
    if new_value not in values:
        values.append(new_value)
    return GRAPH_FIELD_SEP.join(values)


def _graph_field_contains(existing: Any, value: str) -> bool:
    return value in {
        item.strip()
        for item in str(existing or "").split(GRAPH_FIELD_SEP)
        if item.strip()
    }


def _scalar_graph_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scalar_payload: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, set):
            scalar_payload[key] = json.dumps(
                sorted(value),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        elif isinstance(value, (list, tuple, dict)):
            scalar_payload[key] = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        else:
            scalar_payload[key] = value
    return scalar_payload


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
