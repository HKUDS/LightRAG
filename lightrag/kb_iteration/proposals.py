from __future__ import annotations

import re
from pathlib import Path

import yaml

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

NO_APPROVAL_PROPOSAL_TYPES = {"quality_report_note"}
_ALLOWED_RISKS = {"low", "medium", "high"}
_TYPE_PATTERN = re.compile(r"^[a-z0-9_]+$")
_METRIC_KEY_PATTERN = re.compile(r"^[a-z0-9_.-]+$")
_REPORT_NOTE_SAFE_PREFIXES = ("Record ", "Add note ", "Document ", "Summarize ")
_CONTROLLED_MUTATION_ACTIONS = (
    r"add|alter|apply|change|clear|correct|delete|drop|edit|fix|modify|move|patch|"
    r"rebuild|recreate|remove|replace|reset|rewrite|revise|update"
)
_CONTROLLED_MUTATION_TARGETS = (
    r"extraction\s+prompt|facts?|hierarchy(?:\s+rules?)?|kg\s+facts?|ontology"
    r"(?:\s+rules?)?|prompts?|relations?(?:\s+rules?)?|rules?|web\s+display|"
    r"workspace"
)
_REPORT_NOTE_MUTATION_INTENT_PATTERN = re.compile(
    rf"\b(?:{_CONTROLLED_MUTATION_ACTIONS})\w*\b"
    rf".{{0,80}}\b(?:{_CONTROLLED_MUTATION_TARGETS})\b"
    r"|"
    rf"\b(?:{_CONTROLLED_MUTATION_TARGETS})\b"
    rf".{{0,80}}\b(?:{_CONTROLLED_MUTATION_ACTIONS})\w*\b",
    re.IGNORECASE | re.DOTALL,
)
_REPORT_NOTE_DESTRUCTIVE_ACTION_PATTERN = re.compile(
    r"\b(?:clear|delete|drop|rebuild|recreate|reset)\w*\b",
    re.IGNORECASE,
)

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

    if not _is_canonical_type(proposal.type):
        raise ValueError("proposal type must be canonical lowercase snake_case")

    if proposal.risk not in _ALLOWED_RISKS:
        raise ValueError("proposal risk must be one of low, medium, high")

    if not isinstance(proposal.requires_approval, bool):
        raise ValueError("proposal requires_approval must be a bool")

    if not isinstance(proposal.evidence, list) or not all(
        isinstance(item, str) for item in proposal.evidence
    ):
        raise ValueError("proposal evidence must be a list of strings")

    if not isinstance(proposal.expected_metric_change, dict):
        raise ValueError("proposal expected_metric_change must be a dict")
    for key, value in proposal.expected_metric_change.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("proposal expected_metric_change keys must be non-empty strings")
        if not _METRIC_KEY_PATTERN.fullmatch(key):
            raise ValueError("proposal expected_metric_change keys must be metric identifiers")
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
        ):
            raise ValueError("proposal expected_metric_change values must be numbers")

    if (
        not isinstance(proposal.confidence, int | float)
        or isinstance(proposal.confidence, bool)
        or not 0 <= proposal.confidence <= 1
    ):
        raise ValueError("proposal confidence must be between 0 and 1")

    if proposal.type in MUTATION_PROPOSAL_TYPES and proposal.requires_approval is not True:
        raise ValueError(f"proposal type {proposal.type} requires approval")
    if (
        proposal.type == "quality_report_note"
        and proposal.requires_approval is not True
    ):
        _validate_no_approval_report_note(proposal)
    if (
        proposal.type not in MUTATION_PROPOSAL_TYPES
        and proposal.type not in NO_APPROVAL_PROPOSAL_TYPES
        and proposal.requires_approval is not True
    ):
        raise ValueError(f"unknown proposal type {proposal.type} requires approval")


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
    yaml_body = yaml.safe_dump(
        {"proposals": [_proposal_to_render_dict(proposal) for proposal in proposals]},
        allow_unicode=True,
        sort_keys=False,
    )
    return f"# {title}\n\n{yaml_body}"


def _proposal_to_render_dict(proposal: ImprovementProposal) -> dict[str, object]:
    return {
        "id": proposal.id,
        "type": proposal.type,
        "target": proposal.target,
        "proposed_change": proposal.proposed_change,
        "reason": proposal.reason,
        "evidence": proposal.evidence,
        "confidence": proposal.confidence,
        "risk": proposal.risk,
        "requires_approval": proposal.requires_approval,
        "expected_metric_change": {
            key: proposal.expected_metric_change[key]
            for key in sorted(proposal.expected_metric_change)
        },
    }


def _is_canonical_type(value: str) -> bool:
    return value == value.strip().casefold() and bool(_TYPE_PATTERN.fullmatch(value))


def _validate_no_approval_report_note(proposal: ImprovementProposal) -> None:
    if proposal.target != "quality_report.md":
        raise ValueError(
            "quality_report_note without approval must target quality_report.md"
        )
    proposed_change_body = _report_note_body(proposal.proposed_change)
    if proposed_change_body is None:
        raise ValueError(
            "quality_report_note without approval must be a report-only note "
            "or requires approval"
        )

    review_text = "\n".join(
        [
            proposed_change_body,
            proposal.reason,
            *proposal.evidence,
        ]
    )
    if (
        _REPORT_NOTE_MUTATION_INTENT_PATTERN.search(review_text)
        or _REPORT_NOTE_DESTRUCTIVE_ACTION_PATTERN.search(review_text)
    ):
        raise ValueError(
            "quality_report_note implies controlled mutation and requires approval"
        )


def _report_note_body(proposed_change: str) -> str | None:
    for prefix in _REPORT_NOTE_SAFE_PREFIXES:
        if proposed_change.startswith(prefix):
            return proposed_change.removeprefix(prefix)
    return None
