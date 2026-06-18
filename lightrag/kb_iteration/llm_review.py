from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from .markdown import _markdown_text
from .models import ImprovementProposal
from .proposals import validate_proposal


class LLMReviewClient(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass(frozen=True)
class LLMReviewOutput:
    confirmed_issues: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence: list[dict[str, Any]] = field(default_factory=list)
    out_of_scope: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[ImprovementProposal] = field(default_factory=list)


@dataclass(frozen=True)
class LLMJudgeResult:
    decision: str
    reason: str
    risk_override: str = ""
    required_human_checks: list[str] = field(default_factory=list)
    patch_consistency: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "risk_override": self.risk_override,
            "required_human_checks": self.required_human_checks,
            "patch_consistency": self.patch_consistency,
        }


def parse_llm_review_output(raw_text: str) -> LLMReviewOutput:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM review output must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("LLM review output must be a JSON object")

    proposals = []
    for proposal_dict in _list_of_dicts(payload.get("proposals")):
        try:
            proposal = ImprovementProposal(**proposal_dict)
        except TypeError as exc:
            raise ValueError("invalid proposal payload") from exc
        validate_proposal(proposal)
        proposals.append(proposal)

    return LLMReviewOutput(
        confirmed_issues=_list_of_dicts(payload.get("confirmed_issues")),
        hypotheses=_list_of_dicts(payload.get("hypotheses")),
        missing_evidence=_list_of_dicts(payload.get("missing_evidence")),
        out_of_scope=_list_of_dicts(payload.get("out_of_scope")),
        proposals=proposals,
    )


def write_llm_review_artifacts(
    output: LLMReviewOutput, output_dir: str | Path
) -> dict[str, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    report_path = target_dir / "llm_review_report.md"
    proposals_path = target_dir / "proposals.generated.yaml"

    report_path.write_text(_render_report(output), encoding="utf-8")
    proposals_path.write_text(
        "# Generated Proposals\n\n"
        + yaml.safe_dump(
            {"proposals": [proposal.to_dict() for proposal in output.proposals]},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return {
        "llm_review_report": report_path,
        "proposals_generated": proposals_path,
    }


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _render_report(output: LLMReviewOutput) -> str:
    lines = [
        "# LLM Review Report",
        "",
        "## Summary",
        "",
        f"- Confirmed issues: {len(output.confirmed_issues)}",
        f"- Hypotheses: {len(output.hypotheses)}",
        f"- Missing evidence: {len(output.missing_evidence)}",
        f"- Out of scope: {len(output.out_of_scope)}",
        f"- Generated proposals: {len(output.proposals)}",
        "",
    ]
    lines.extend(_render_dict_section("Confirmed Issues", output.confirmed_issues))
    lines.extend(_render_dict_section("Hypotheses", output.hypotheses))
    lines.extend(_render_dict_section("Missing Evidence", output.missing_evidence))
    lines.extend(_render_dict_section("Out Of Scope", output.out_of_scope))
    lines.extend(_render_proposal_section("Generated Proposals", output.proposals))
    lines.extend(_render_patch_candidates(output.proposals))
    lines.extend(["## Human Review Required", ""])
    gated_ids = [
        proposal.id for proposal in output.proposals if proposal.requires_approval
    ]
    if gated_ids:
        lines.extend(f"- {_markdown_text(proposal_id)}" for proposal_id in gated_ids)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _render_dict_section(title: str, items: list[dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", ""]
    if items:
        lines.extend(_render_dict_item(item) for item in items)
    else:
        lines.append("- none")
    lines.append("")
    return lines


def _render_proposal_section(
    title: str, proposals: list[ImprovementProposal]
) -> list[str]:
    lines = [f"## {title}", ""]
    if proposals:
        lines.extend(
            f"- {_markdown_text(proposal.id)}: {_markdown_text(proposal.proposed_change)}"
            for proposal in proposals
        )
    else:
        lines.append("- none")
    lines.append("")
    return lines


def _render_patch_candidates(proposals: list[ImprovementProposal]) -> list[str]:
    lines = ["## Patch Candidates", ""]
    patch_candidates = [
        proposal for proposal in proposals if proposal.patch_candidate.strip()
    ]
    if patch_candidates:
        lines.extend(
            f"- {_markdown_text(proposal.id)}: {_markdown_text(proposal.patch_candidate)}"
            for proposal in patch_candidates
        )
    else:
        lines.append("- none")
    lines.append("")
    return lines


def _render_dict_item(item: dict[str, Any]) -> str:
    suffixes = []
    for key in ("evidence", "source_ids", "sources", "supporting_ids"):
        value = item.get(key)
        if isinstance(value, list):
            rendered = [_markdown_text(entry) for entry in value if entry is not None]
            if rendered:
                suffixes.append(f"{key}: {', '.join(rendered)}")
        elif isinstance(value, dict):
            rendered = _render_one_level_dict(value)
            if rendered:
                suffixes.append(f"{key}: {rendered}")
        elif isinstance(value, str) and value.strip():
            suffixes.append(f"{key}: {_markdown_text(value)}")

    if suffixes:
        return f"- {_short_message(item)} ({'; '.join(suffixes)})"
    return f"- {_short_message(item)}"


def _render_one_level_dict(value: dict[str, Any]) -> str:
    rendered = []
    for nested_key, nested_value in value.items():
        if isinstance(nested_value, list):
            entries = [
                _markdown_text(entry) for entry in nested_value if entry is not None
            ]
            if entries:
                rendered.append(
                    f"{_markdown_text(nested_key)}: {', '.join(entries)}"
                )
        elif isinstance(nested_value, str) and nested_value.strip():
            rendered.append(
                f"{_markdown_text(nested_key)}: {_markdown_text(nested_value)}"
            )
    return "; ".join(rendered)


def _short_message(item: dict[str, Any]) -> str:
    for key in ("message", "reason", "proposed_change", "target", "id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return _markdown_text(value)
    return _markdown_text(json.dumps(item, ensure_ascii=False, sort_keys=True))
