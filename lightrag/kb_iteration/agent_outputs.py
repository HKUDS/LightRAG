from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm_review import parse_llm_review_output
from .markdown import _markdown_text
from .models import ImprovementProposal


@dataclass(frozen=True)
class AgentStageOutput:
    stage: str
    payload: dict[str, Any]
    proposals: list[ImprovementProposal] = field(default_factory=list)


_STAGE_ARTIFACTS = {
    "explain": (
        "llm_issue_analysis.json",
        "llm_issue_analysis.md",
        "llm_issue_analysis",
    ),
    "infer_branches": (
        "llm_missing_branch_inference.json",
        "llm_missing_branch_inference.md",
        "llm_missing_branch_inference",
    ),
    "locate_evidence": (
        "llm_evidence_map.json",
        "llm_evidence_map.md",
        "llm_evidence_map",
    ),
    "rank_repairs": (
        "llm_repair_plan.json",
        "llm_repair_plan.md",
        "llm_repair_plan",
    ),
    "judge": (
        "llm_judge_report.json",
        "llm_judge_report.md",
        "llm_judge_report",
    ),
}
_EVIDENCE_FIELD_ORDER = (
    "source_id",
    "file_path",
    "item_id",
    "entity_id",
    "relation_id",
    "metric",
)
_EVIDENCE_FIELD_NAMES = frozenset(_EVIDENCE_FIELD_ORDER)
_NUMERIC_STRING_PATTERN = re.compile(
    r"^[+-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$"
)


def parse_agent_stage_output(stage: str, raw_text: str) -> AgentStageOutput:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("agent stage output must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("agent stage output must be a JSON object")

    if stage != "propose":
        return AgentStageOutput(stage=stage, payload=payload)

    review_output = parse_llm_review_output(
        json.dumps(
            {"proposals": _normalize_agent_proposals(payload.get("proposals"))},
            ensure_ascii=False,
        )
    )
    for proposal in review_output.proposals:
        if proposal.type != "review_context_request" and not _has_evidence(proposal):
            raise ValueError("proposal evidence is required for non-context proposals")

    return AgentStageOutput(
        stage=stage,
        payload=payload,
        proposals=review_output.proposals,
    )


def _has_evidence(proposal: ImprovementProposal) -> bool:
    return any(item.strip() for item in proposal.evidence)


def _normalize_agent_proposals(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    proposals: list[dict[str, Any]] = []
    for proposal in value:
        if not isinstance(proposal, dict):
            continue
        normalized = dict(proposal)
        normalized["evidence"] = _normalize_agent_proposal_evidence(
            proposal.get("evidence")
        )
        normalized["expected_metric_change"] = _normalize_expected_metric_change(
            proposal.get("expected_metric_change")
        )
        proposals.append(normalized)
    return proposals


def _normalize_expected_metric_change(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    return {
        key: _normalize_metric_change_value(metric_value)
        for key, metric_value in value.items()
    }


def _normalize_metric_change_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not _NUMERIC_STRING_PATTERN.fullmatch(stripped):
        return value

    if "." not in stripped and "e" not in stripped.casefold():
        return int(stripped)

    numeric_value = float(stripped)
    if not math.isfinite(numeric_value):
        return value
    return numeric_value


def _normalize_agent_proposal_evidence(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    evidence = []
    for item in value:
        normalized = _normalize_agent_evidence_item(item)
        if normalized:
            evidence.append(normalized)
    return evidence


def _normalize_agent_evidence_item(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else value
    if not isinstance(value, dict):
        return ""

    unknown_fields = [
        key
        for key, item_value in value.items()
        if key not in _EVIDENCE_FIELD_NAMES and _render_evidence_value(item_value)
    ]
    if unknown_fields:
        raise ValueError("proposal evidence contains unknown structured field")

    parts = []
    for key in _EVIDENCE_FIELD_ORDER:
        rendered = _render_evidence_value(value.get(key))
        if rendered:
            parts.append(f"{key}: {rendered}")
    return "; ".join(parts)


def _render_evidence_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        rendered = [_render_evidence_value(item) for item in value]
        return ", ".join(item for item in rendered if item)
    return ""


def write_agent_stage_artifacts(
    output_dir: str | Path, output: AgentStageOutput
) -> dict[str, Path]:
    artifact = _STAGE_ARTIFACTS.get(output.stage)
    if artifact is None or output.stage == "propose":
        return {}

    json_filename, markdown_filename, key = artifact
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / json_filename
    markdown_path = target_dir / markdown_filename
    json_path.write_text(
        json.dumps(output.payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(stage_output_to_markdown(output), encoding="utf-8")

    return {
        f"{key}_json": json_path,
        key: markdown_path,
    }


def stage_output_to_markdown(output: AgentStageOutput) -> str:
    renderers = {
        "explain": _render_explain,
        "infer_branches": _render_infer_branches,
        "locate_evidence": _render_locate_evidence,
        "rank_repairs": _render_rank_repairs,
        "judge": _render_judge,
    }
    renderer = renderers.get(output.stage, _render_fallback)
    return _join_markdown(renderer(output.payload))


def _render_explain(payload: dict[str, Any]) -> list[str]:
    lines = ["# 问题解释", ""]
    issues = _dict_list(payload.get("issue_explanations"))
    if not issues:
        return lines + ["- none"]

    for issue in issues:
        lines.extend([f"## {_value(issue, 'id')}", ""])
        _append_field(lines, "Category", issue.get("category"))
        _append_field(lines, "Severity", issue.get("severity"))
        _append_field(lines, "Explanation", issue.get("explanation"))
        _append_field(lines, "Impact", issue.get("impact"))
        _append_list(lines, "Evidence refs", issue.get("evidence_refs"), _evidence_ref)
        lines.append("")
    return lines


def _render_infer_branches(payload: dict[str, Any]) -> list[str]:
    lines = ["# 缺失分支推断", ""]
    for key, title in (
        ("required", "Required"),
        ("present", "Present"),
        ("missing", "Missing"),
        ("missing_branches", "Missing branches"),
    ):
        _append_list(lines, title, payload.get(key))
        lines.append("")
    return lines


def _render_locate_evidence(payload: dict[str, Any]) -> list[str]:
    lines = ["# 证据定位", ""]
    evidence_items = _dict_list(payload.get("evidence_map"))
    if not evidence_items:
        return lines + ["- none"]

    for item in evidence_items:
        lines.extend([f"## {_value(item, 'issue_id')}", ""])
        _append_field(lines, "Target", item.get("target"))
        _append_field(lines, "Confidence", item.get("confidence"))
        _append_list(lines, "Missing evidence", item.get("missing_evidence"))
        supporting_items = _dict_list(item.get("supporting_items"))
        if supporting_items:
            lines.extend(["- Supporting items:", ""])
            for supporting in supporting_items:
                details = _compact_fields(
                    supporting,
                    ("source_id", "file_path", "evidence_status"),
                )
                lines.append(f"  - {details}")
        else:
            lines.append("- Supporting items: none")
        lines.append("")
    return lines


def _render_rank_repairs(payload: dict[str, Any]) -> list[str]:
    lines = ["# 修复方案排序", ""]
    plans = _dict_list(payload.get("repair_plan"))
    if not plans:
        return lines + ["- none"]

    for plan in plans:
        title = _value(plan, "proposal_id")
        rank = plan.get("rank")
        if rank is not None:
            title = f"{_markdown_text(rank)}. {title}"
        lines.extend([f"## {title}", ""])
        _append_field(lines, "Priority", plan.get("priority"))
        _append_field(lines, "Risk", plan.get("risk"))
        _append_field(lines, "Reason", plan.get("reason"))
        _append_list(lines, "Human checks", plan.get("human_checks"))
        lines.append("")
    return lines


def _render_judge(payload: dict[str, Any]) -> list[str]:
    lines = ["# Judge 评判", ""]
    results = _dict_list(payload.get("judge_results"))
    if not results and payload:
        results = [payload]
    if not results:
        return lines + ["- none"]

    for index, result in enumerate(results, start=1):
        title = _markdown_text(result.get("proposal_id") or f"Result {index}")
        lines.extend([f"## {title}", ""])
        for key in sorted(result):
            if key == "proposal_id":
                continue
            value = result[key]
            if isinstance(value, list):
                _append_list(lines, _title(key), value)
            elif isinstance(value, dict):
                lines.append(f"- {_title(key)}: {_markdown_text(_json_string(value))}")
            else:
                _append_field(lines, _title(key), value)
        lines.append("")
    return lines


def _render_fallback(payload: dict[str, Any]) -> list[str]:
    return [
        "# LLM Agent Stage",
        "",
        "```json",
        _json_string(payload).replace("```", "` ` `"),
        "```",
    ]


def _append_field(lines: list[str], title: str, value: Any) -> None:
    if value is None or value == "":
        return
    lines.append(f"- {title}: {_markdown_text(value)}")


def _append_list(
    lines: list[str],
    title: str,
    value: Any,
    formatter: Any = _markdown_text,
) -> None:
    items = _list_items(value)
    lines.append(f"- {title}:")
    if not items:
        lines.append("  - none")
        return
    lines.extend(f"  - {formatter(item)}" for item in items)


def _compact_fields(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    parts = []
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            parts.append(f"{key}: {_markdown_text(value)}")
    if parts:
        return "; ".join(parts)
    return _markdown_text(_json_string(item))


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _list_items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for item in value if item is not None]
    if value is None or value == "":
        return []
    return [value]


def _value(item: dict[str, Any], key: str) -> str:
    return _markdown_text(item.get(key) or "Unknown")


def _evidence_ref(value: Any) -> str:
    return _markdown_text(value).replace("-\\>", "->")


def _title(value: str) -> str:
    return value.replace("_", " ").capitalize()


def _json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _join_markdown(lines: list[str]) -> str:
    return "\n".join(lines).rstrip() + "\n"
