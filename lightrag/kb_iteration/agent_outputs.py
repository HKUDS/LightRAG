from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP

from .llm_review import parse_llm_review_output
from .markdown import _markdown_text
from .models import ImprovementProposal
from .proposals import REVIEW_ONLY_PROPOSAL_TYPES


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
_EVIDENCE_FIELD_ALIASES = {
    "edge_id": "relation_id",
    "node_id": "entity_id",
}
_EVIDENCE_IGNORED_FIELD_NAMES = frozenset({"item_type", "evidence_status"})
_PROPOSAL_FIELD_NAMES = frozenset(ImprovementProposal.__dataclass_fields__)
_NUMERIC_STRING_PATTERN = re.compile(
    r"^[+-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$"
)
_PROPOSAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_PROPOSAL_ID_PART_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_CANDIDATE_EDGE_REQUIRED_FIELDS = (
    "source",
    "target",
    "source_type",
    "target_type",
    "keywords",
    "source_id",
    "file_path",
)
_CANDIDATE_EDGE_TYPE_FIELDS = ("source_type", "target_type")
_EVIDENCE_TUPLE_FIELDS = ("source_id", "file_path", "evidence_quote")


def parse_agent_stage_output(
    stage: str,
    raw_text: str,
    *,
    allowed_evidence_spans: list[dict[str, Any]] | None = None,
) -> AgentStageOutput:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("agent stage output must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("agent stage output must be a JSON object")

    if stage != "propose":
        return AgentStageOutput(stage=stage, payload=payload)

    normalized_proposals = _normalize_agent_proposals(
        payload.get("proposals"),
        allowed_evidence_spans=allowed_evidence_spans,
    )
    review_output = parse_llm_review_output(
        json.dumps(
            {"proposals": normalized_proposals},
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


def _normalize_agent_proposals(
    value: Any,
    *,
    allowed_evidence_spans: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    proposals: list[dict[str, Any]] = []
    for proposal in value:
        if not isinstance(proposal, dict):
            continue
        normalized = {
            key: item_value
            for key, item_value in proposal.items()
            if key in _PROPOSAL_FIELD_NAMES
        }
        _copy_proposal_alias(proposal, normalized, "id", "proposal_id")
        _copy_proposal_alias(proposal, normalized, "type", "proposal_type")
        normalized["action_payload"] = _normalize_agent_action_payload(
            normalized.get("action_payload"),
            proposal=normalized,
        )
        normalized["type"] = _normalize_agent_proposal_type(normalized)
        if normalized["type"] in REVIEW_ONLY_PROPOSAL_TYPES:
            normalized["requires_approval"] = True
        _copy_proposal_alias(proposal, normalized, "reason", "rationale")
        _copy_proposal_alias(proposal, normalized, "reason", "description")
        normalized["id"] = _normalize_agent_proposal_id(normalized)
        normalized["evidence"] = _normalize_agent_proposal_evidence(
            normalized.get("evidence")
        )
        if not normalized["evidence"]:
            normalized["evidence"] = _derive_agent_proposal_evidence(normalized)
        if normalized["type"] == "candidate_kg_expansion":
            _validate_agent_candidate_kg_expansion_payload(
                normalized["action_payload"],
                allowed_evidence_spans=allowed_evidence_spans,
            )
        normalized["expected_metric_change"] = _normalize_expected_metric_change(
            normalized.get("expected_metric_change")
        )
        if not str(normalized.get("proposed_change", "")).strip():
            normalized["proposed_change"] = _default_agent_proposed_change(
                normalized,
                source=proposal,
            )
        proposals.append(normalized)
    return proposals


def _normalize_agent_proposal_type(proposal: dict[str, Any]) -> Any:
    proposal_type = proposal.get("type")
    if not isinstance(proposal_type, str):
        return proposal_type

    normalized_type = proposal_type.strip()
    if (
        normalized_type == "replace_relation"
        and _is_replace_relation_payload(proposal.get("action_payload"))
    ):
        return "medical_relation_schema_migration"
    return normalized_type


def _is_replace_relation_payload(value: Any) -> bool:
    return isinstance(value, dict) and value.get("action") == "replace_relation"


def _copy_proposal_alias(
    source: dict[str, Any], target: dict[str, Any], field: str, alias: str
) -> None:
    if field in target and target[field] not in (None, ""):
        return
    if alias in source:
        target[field] = source[alias]


def _normalize_agent_action_payload(
    value: Any,
    *,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    if value.get("action") == "replace_relation":
        return dict(value)

    edge_id = _relation_id_from_payload_or_target(value, proposal)
    new_relation = value.get("new_relation")
    if not isinstance(new_relation, dict):
        new_relation = value

    new_source = _first_string_value(
        new_relation.get("source"),
        value.get("new_source"),
    )
    new_target = _first_string_value(
        new_relation.get("target"),
        value.get("new_target"),
    )
    new_keywords = _first_string_value(
        new_relation.get("predicate"),
        new_relation.get("keywords"),
        value.get("new_predicate"),
        value.get("new_keywords"),
    )
    expected_source = _first_string_value(
        value.get("expected_source"),
        value.get("old_source"),
        value.get("current_source"),
    )
    expected_target = _first_string_value(
        value.get("expected_target"),
        value.get("old_target"),
        value.get("current_target"),
    )
    if (not expected_source or not expected_target) and edge_id:
        split_source, split_target = _split_relation_id(edge_id)
        expected_source = expected_source or split_source
        expected_target = expected_target or split_target

    if not all(
        (edge_id, expected_source, expected_target, new_source, new_target, new_keywords)
    ):
        return dict(value)

    return {
        "action": "replace_relation",
        "edge_id": edge_id,
        "expected_source": expected_source,
        "expected_target": expected_target,
        "current_keywords": _first_string_value(
            value.get("current_keywords"),
            value.get("old_keywords"),
            value.get("keywords"),
        ),
        "new_source": new_source,
        "new_target": new_target,
        "new_keywords": new_keywords,
        "qualifiers": (
            value.get("qualifiers") if isinstance(value.get("qualifiers"), dict) else {}
        ),
    }


def _relation_id_from_payload_or_target(
    payload: dict[str, Any], proposal: dict[str, Any]
) -> str:
    edge_id = _first_string_value(
        payload.get("edge_id"),
        payload.get("relation_id"),
        payload.get("old_relation_id"),
        payload.get("source_edge_id"),
        payload.get("item_id"),
    )
    if edge_id:
        return edge_id
    target = _first_string_value(proposal.get("target"))
    for prefix in ("kg:relation:", "relation:", "edge:"):
        if target.startswith(prefix):
            return target.removeprefix(prefix)
    return target


def _split_relation_id(value: str) -> tuple[str, str]:
    for separator in ("->", "=>", ">"):
        if separator in value:
            source, target = value.split(separator, 1)
            return source.strip(), target.strip()
    return "", ""


def _first_string_value(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_agent_proposal_id(proposal: dict[str, Any]) -> Any:
    raw_id = proposal.get("id")
    if not isinstance(raw_id, str):
        return raw_id

    stripped = raw_id.strip()
    if not stripped or _PROPOSAL_ID_PATTERN.fullmatch(stripped):
        return stripped

    proposal_type = str(proposal.get("type") or "proposal").strip().casefold()
    slug = _PROPOSAL_ID_PART_PATTERN.sub("-", proposal_type).strip(".-_")
    if not slug:
        slug = "proposal"
    digest_source = json.dumps(
        {
            "id": stripped,
            "type": proposal.get("type"),
            "target": proposal.get("target"),
            "proposed_change": proposal.get("proposed_change"),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:12]
    return f"prop-{slug}-{digest}"


def _default_agent_proposed_change(
    proposal: dict[str, Any],
    *,
    source: dict[str, Any] | None = None,
) -> str:
    for candidate in (source or {}, proposal):
        for key in ("change", "recommendation", "reason"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    payload = proposal.get("action_payload")
    if isinstance(payload, dict):
        action = str(payload.get("action") or "review").strip()
        target = str(payload.get("edge_id") or proposal.get("target") or "").strip()
        if target:
            return f"{action} {target}"
        return action
    target = str(proposal.get("target") or "proposal target").strip()
    return f"Review {target}"


def _normalize_expected_metric_change(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, str) and not value.strip():
        return {}
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
    if isinstance(value, dict):
        raise ValueError(
            "EVIDENCE_MUST_BE_STRING: proposal evidence must be a list of strings"
        )
    elif isinstance(value, list):
        evidence_items = value
    else:
        return []

    evidence = []
    for item in evidence_items:
        if isinstance(item, dict):
            raise ValueError(
                "EVIDENCE_MUST_BE_STRING: "
                "proposal evidence items must be strings"
            )
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
    raise ValueError(
        "EVIDENCE_MUST_BE_STRING: proposal evidence items must be strings"
    )


def _validate_agent_candidate_kg_expansion_payload(
    action_payload: dict[str, Any],
    *,
    allowed_evidence_spans: list[dict[str, Any]] | None,
) -> None:
    if not isinstance(action_payload, dict):
        return

    candidate_edges = action_payload.get("candidate_edges")
    if isinstance(candidate_edges, list):
        for index, edge in enumerate(candidate_edges):
            _validate_agent_candidate_edge(edge, index=index)

    _validate_agent_candidate_expansion_evidence_tuple(
        action_payload,
        allowed_evidence_spans=allowed_evidence_spans,
    )


def _validate_agent_candidate_edge(value: Any, *, index: int) -> None:
    if not isinstance(value, dict):
        return

    missing = [
        field_name
        for field_name in _CANDIDATE_EDGE_REQUIRED_FIELDS
        if not isinstance(value.get(field_name), str)
        or not str(value.get(field_name)).strip()
    ]
    missing_type_fields = [
        field_name for field_name in _CANDIDATE_EDGE_TYPE_FIELDS if field_name in missing
    ]
    if missing_type_fields:
        raise ValueError(
            "CANDIDATE_EDGE_TYPES_REQUIRED: "
            f"candidate_edges[{index}] missing_fields="
            + ",".join(missing_type_fields)
        )
    if missing:
        raise ValueError(
            "CANDIDATE_EDGE_REQUIRED_FIELDS: "
            f"candidate_edges[{index}] missing_fields="
            + ",".join(missing)
        )
    _reject_sep_joined_evidence_reference(
        value,
        fields=("source_id", "file_path"),
    )


def _validate_agent_candidate_expansion_evidence_tuple(
    action_payload: dict[str, Any],
    *,
    allowed_evidence_spans: list[dict[str, Any]] | None,
) -> None:
    _reject_sep_joined_evidence_reference(
        action_payload,
        fields=("source_id", "file_path"),
    )
    if allowed_evidence_spans is None:
        return

    evidence_tuple = _evidence_tuple(action_payload)
    allowed_tuples = {
        _evidence_tuple(span)
        for span in allowed_evidence_spans
        if all(str(span.get(field_name) or "").strip() for field_name in _EVIDENCE_TUPLE_FIELDS)
    }
    if evidence_tuple in allowed_tuples:
        return
    raise ValueError(
        "EVIDENCE_TUPLE_MUST_MATCH_ALLOWED_SPAN: "
        "candidate_kg_expansion action_payload must copy source_id, file_path, "
        "and evidence_quote from one allowed_evidence_spans record"
    )


def _evidence_tuple(value: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(
        str(value.get(field_name) or "").strip()
        for field_name in _EVIDENCE_TUPLE_FIELDS
    )


def _reject_sep_joined_evidence_reference(
    value: dict[str, Any],
    *,
    fields: tuple[str, ...],
) -> None:
    for field_name in fields:
        field_value = value.get(field_name)
        if isinstance(field_value, str) and GRAPH_FIELD_SEP in field_value:
            raise ValueError(
                "EVIDENCE_TUPLE_MUST_MATCH_ALLOWED_SPAN: "
                f"{field_name} must not join multiple references with "
                f"{GRAPH_FIELD_SEP}"
            )


def _normalize_agent_evidence_aliases(value: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, item_value in value.items():
        canonical_key = _EVIDENCE_FIELD_ALIASES.get(key, key)
        if canonical_key in normalized and _render_evidence_value(
            normalized[canonical_key]
        ):
            continue
        normalized[canonical_key] = item_value
    return normalized


def _derive_agent_proposal_evidence(proposal: dict[str, Any]) -> list[str]:
    payload = proposal.get("action_payload")
    if not isinstance(payload, dict):
        return []

    evidence_item = {
        "source_id": payload.get("source_id"),
        "file_path": payload.get("file_path"),
        "relation_id": payload.get("edge_id") or payload.get("relation_id"),
        "entity_id": payload.get("node_id") or payload.get("entity_id"),
        "item_id": payload.get("item_id"),
    }
    evidence = _render_internal_evidence_item(evidence_item)
    return [evidence] if evidence else []


def _render_internal_evidence_item(value: dict[str, Any]) -> str:
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
