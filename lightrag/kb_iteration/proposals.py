from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import ImprovementProposal

MUTATION_PROPOSAL_TYPES = {
    "prompt_edit",
    "ontology_rule_change",
    "hierarchy_rule_change",
    "relation_rule_change",
    "workspace_rebuild",
    "kg_fact_correction",
    "web_display_change",
}

_REQUIRED_STRING_FIELDS = (
    "id",
    "type",
    "target",
    "proposed_change",
    "reason",
    "risk",
)


def validate_proposal(proposal: ImprovementProposal) -> None:
    for field_name in _REQUIRED_STRING_FIELDS:
        value = getattr(proposal, field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"proposal {field_name} must be a non-empty string")

    if not isinstance(proposal.evidence, list) or not all(
        isinstance(item, str) for item in proposal.evidence
    ):
        raise ValueError("proposal evidence must be a list of strings")

    if not isinstance(proposal.expected_metric_change, dict):
        raise ValueError("proposal expected_metric_change must be a dict")

    if not 0 <= proposal.confidence <= 1:
        raise ValueError("proposal confidence must be between 0 and 1")

    if proposal.type in MUTATION_PROPOSAL_TYPES and not proposal.requires_approval:
        raise ValueError(f"proposal type {proposal.type} requires approval")


def write_approval_queue(
    proposals: list[ImprovementProposal], output_dir: str | Path
) -> Path:
    valid_proposals = _validate_and_sort(proposals)
    queued = [proposal for proposal in valid_proposals if proposal.requires_approval]
    return _write_proposals(queued, output_dir, "approval_queue.md", "Approval Queue")


def write_improvement_backlog(
    proposals: list[ImprovementProposal], output_dir: str | Path
) -> Path:
    valid_proposals = _validate_and_sort(proposals)
    return _write_proposals(
        valid_proposals,
        output_dir,
        "improvement_backlog.md",
        "Improvement Backlog",
    )


def _validate_and_sort(
    proposals: list[ImprovementProposal],
) -> list[ImprovementProposal]:
    for proposal in proposals:
        validate_proposal(proposal)
    return sorted(proposals, key=lambda proposal: proposal.id)


def _write_proposals(
    proposals: list[ImprovementProposal],
    output_dir: str | Path,
    filename: str,
    title: str,
) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    path.write_text(_render_proposals(proposals, title), encoding="utf-8")
    return path


def _render_proposals(proposals: list[ImprovementProposal], title: str) -> str:
    lines = [f"# {title}", "", "proposals:"]
    if not proposals:
        lines[-1] = "proposals: []"
    else:
        for proposal in proposals:
            lines.extend(_render_proposal_block(proposal))
    lines.append("")
    return "\n".join(lines)


def _render_proposal_block(proposal: ImprovementProposal) -> list[str]:
    lines = [
        f"- id: {_render_scalar(proposal.id)}",
        f"  type: {_render_scalar(proposal.type)}",
        f"  target: {_render_scalar(proposal.target)}",
        f"  proposed_change: {_render_scalar(proposal.proposed_change)}",
        f"  reason: {_render_scalar(proposal.reason)}",
    ]
    lines.extend(_render_string_list("evidence", proposal.evidence))
    lines.extend(
        [
            f"  confidence: {_render_scalar(proposal.confidence)}",
            f"  risk: {_render_scalar(proposal.risk)}",
            f"  requires_approval: {_render_scalar(proposal.requires_approval)}",
        ]
    )
    lines.extend(
        _render_metric_change("expected_metric_change", proposal.expected_metric_change)
    )
    return lines


def _render_string_list(field_name: str, values: list[str]) -> list[str]:
    if not values:
        return [f"  {field_name}: []"]
    lines = [f"  {field_name}:"]
    lines.extend(f"  - {_render_scalar(value)}" for value in values)
    return lines


def _render_metric_change(
    field_name: str, values: dict[str, int | float]
) -> list[str]:
    if not values:
        return [f"  {field_name}: {{}}"]
    lines = [f"  {field_name}:"]
    for key in sorted(values):
        lines.append(f"  {key}: {_render_scalar(values[key])}")
    return lines


def _render_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
