from __future__ import annotations

import re
from typing import Any

from lightrag.kb_iteration.medical_schema import (
    normalize_medical_entity_type,
    relation_spec_by_id,
    validate_relation_instance,
)
from lightrag.kb_iteration.relation_tokens import split_relation_tokens

from .base import (
    CandidateGenerationResult,
    GenerationContext,
    candidate_rejection,
    combine_results,
    generate_with_builders,
)

_VALUE_LIKE_RELATION_KEYWORDS = {
    "has_value",
    "value",
    "dosage_usage",
    "dose_usage",
}
_ENTITY_MERGE_ISSUE_KINDS = {
    "synonym_duplicate",
    "alias_duplicate",
    "near_duplicate_entity",
}

_DOSE_PATTERN = re.compile(
    r"(?i)\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|μg|iu|u|unit|units|单位)"
)
_FREQUENCY_PATTERN = re.compile(
    r"(?i)(每日|每天|每周|每隔|每次|\b(?:qd|bid|tid|qid|q\d+h)\b|"
    r"\d+\s*次\s*/?\s*[日天周月]?)"
)
_DURATION_PATTERN = re.compile(
    r"(?i)(疗程|连续|共)?\s*\d+(?:[~-]\d+)?\s*"
    r"(?:小时|天|日|周|月|days?|weeks?|months?)"
)
_ROUTE_PATTERN = re.compile(
    r"(?i)(口服|静脉|肌内|皮下|鼻喷|吸入|雾化|oral|intravenous|"
    r"intramuscular|inhalation)"
)
_AGE_PATTERN = re.compile(
    r"(?i)(?:≥|<=|>=|<|>)?\s*\d+(?:\.\d+)?\s*"
    r"(?:周龄|月龄|岁|years?|months?)"
)
_TIME_WINDOW_PATTERN = re.compile(
    r"(?i)(发病|接种|治疗|用药)?\s*(?:前|后|内)?\s*"
    r"\d+(?:[~-]\d+)?\s*(?:小时|天|日|周|hours?|days?|weeks?)"
)


def generate(context: GenerationContext) -> CandidateGenerationResult:
    cleanup_candidates: list[dict[str, Any]] = []
    cleanup_rejections: list[dict[str, Any]] = []
    for issue in context.issues:
        if str(issue.get("issue_kind") or "") in _ENTITY_MERGE_ISSUE_KINDS:
            cleanup_rejections.append(
                candidate_rejection(
                    issue,
                    error_code="ENTITY_MERGE_APPLY_NOT_SUPPORTED",
                    error=(
                        "Entity merge cleanup is visible but deterministic apply "
                        "support is not available yet."
                    ),
                )
            )
            continue
        candidate, rejection = _value_node_to_qualifier_candidate(issue, context)
        if candidate is not None:
            cleanup_candidates.append(candidate)
        if rejection is not None:
            cleanup_rejections.append(rejection)
    return combine_results(
        generate_with_builders(context),
        CandidateGenerationResult(
            candidates=cleanup_candidates,
            rejections=cleanup_rejections,
        ),
    )


def _value_node_to_qualifier_candidate(
    issue: dict[str, Any],
    context: GenerationContext,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if str(issue.get("issue_kind") or "") != "value_node_to_qualifier":
        return None, None

    value_node_id = _string_value(issue.get("node_id"))
    qualifier_value = _string_value(issue.get("qualifier_value")) or _string_value(
        issue.get("label")
    )
    if not value_node_id or not qualifier_value:
        return None, _no_safe_repair(issue, "INCOMPLETE_VALUE_NODE_ISSUE")

    value_edge = _single_value_incident_edge(value_node_id, issue, context)
    if value_edge is None:
        return None, _no_safe_repair(issue, "NO_UNIQUE_VALUE_INCIDENT_EDGE")
    anchor_node_id = _other_endpoint(value_edge, value_node_id)
    if not anchor_node_id:
        return None, _no_safe_repair(issue, "NO_VALUE_NODE_ANCHOR")

    qualifier_key = _qualifier_key(issue)
    if not qualifier_key:
        return None, _no_safe_repair(issue, "NO_SUPPORTED_QUALIFIER_KEY")
    carrier_edge, carrier_error = _single_schema_compatible_carrier_edge(
        anchor_node_id,
        value_node_id,
        qualifier_key,
        qualifier_value,
        context,
    )
    if carrier_edge is None:
        return None, _no_safe_repair(issue, carrier_error)
    carrier_source = _edge_source(carrier_edge)
    carrier_target = _edge_target(carrier_edge)
    carrier_predicate = _single_canonical_predicate(carrier_edge)
    if not carrier_source or not carrier_target or not carrier_predicate:
        return None, _no_safe_repair(issue, "NO_SCHEMA_COMPATIBLE_CARRIER_EDGE")

    return {
        "proposal_type": "value_node_to_qualifier",
        "target": f"node:{value_node_id}",
        "issue_ref": _string_value(issue.get("issue_ref")),
        "issue_kind": "value_node_to_qualifier",
        "action_payload": {
            "value_node_id": value_node_id,
            "incident_edge_id": _edge_id(value_edge),
            "expected_incident_keywords": _edge_keywords(value_edge),
            "carrier_edge_id": _edge_id(carrier_edge),
            "carrier_edge_source": carrier_source,
            "carrier_edge_target": carrier_target,
            "expected_carrier_keywords": carrier_predicate,
            "carrier_source_type": _node_type(carrier_source, context),
            "carrier_target_type": _node_type(carrier_target, context),
            "qualifier_key": qualifier_key,
            "qualifier_value": qualifier_value,
        },
    }, None


def _single_value_incident_edge(
    value_node_id: str,
    issue: dict[str, Any],
    context: GenerationContext,
) -> dict[str, Any] | None:
    issue_edge_ids = [
        str(edge_id).strip()
        for edge_id in issue.get("connected_edge_ids", [])
        if isinstance(edge_id, str) and edge_id.strip()
    ]
    candidate_edges = [
        edge
        for edge_id in issue_edge_ids
        if (edge := context.edges_by_id.get(edge_id)) is not None
    ]
    if not candidate_edges:
        candidate_edges = [
            edge for edge in context.all_edges if _edge_touches(edge, value_node_id)
        ]
    value_edges = [
        edge
        for edge in candidate_edges
        if _edge_touches(edge, value_node_id) and _is_value_like_edge(edge)
    ]
    if len(value_edges) != 1:
        return None
    return value_edges[0]


def _single_schema_compatible_carrier_edge(
    anchor_node_id: str,
    value_node_id: str,
    qualifier_key: str,
    qualifier_value: str,
    context: GenerationContext,
) -> tuple[dict[str, Any] | None, str]:
    carrier_edges = [
        edge
        for edge in context.all_edges
        if _edge_touches(edge, anchor_node_id)
        and not _edge_touches(edge, value_node_id)
        and not _is_value_like_edge(edge)
    ]
    compatible_edges = [
        edge
        for edge in carrier_edges
        if _carrier_relation_allows_qualifier(
            edge,
            qualifier_key,
            qualifier_value,
            context,
        )
    ]
    if not compatible_edges:
        return None, "NO_SCHEMA_COMPATIBLE_CARRIER_EDGE"
    if len(compatible_edges) > 1:
        return None, "AMBIGUOUS_CARRIER_EDGE"
    return compatible_edges[0], ""


def _carrier_relation_allows_qualifier(
    edge: dict[str, Any],
    qualifier_key: str,
    qualifier_value: str,
    context: GenerationContext,
) -> bool:
    predicate = _single_canonical_predicate(edge)
    if not predicate:
        return False
    spec = relation_spec_by_id(predicate)
    if qualifier_key not in spec.allowed_qualifiers:
        return False
    return not validate_relation_instance(
        predicate=predicate,
        source_type=_node_type(_edge_source(edge), context),
        target_type=_node_type(_edge_target(edge), context),
        qualifiers={qualifier_key: qualifier_value},
    )


def _single_canonical_predicate(edge: dict[str, Any]) -> str:
    predicates: list[str] = []
    for token in _edge_keyword_tokens(edge):
        try:
            predicates.append(relation_spec_by_id(token).id)
        except KeyError:
            continue
    unique = sorted(set(predicates))
    return unique[0] if len(unique) == 1 else ""


def _edge_touches(edge: dict[str, Any], node_id: str) -> bool:
    return node_id in {_edge_source(edge), _edge_target(edge)}


def _other_endpoint(edge: dict[str, Any], node_id: str) -> str:
    source = _edge_source(edge)
    target = _edge_target(edge)
    if source == node_id:
        return target
    if target == node_id:
        return source
    return ""


def _edge_source(edge: dict[str, Any]) -> str:
    return _string_value(edge.get("source") or edge.get("source_node_id"))


def _edge_target(edge: dict[str, Any]) -> str:
    return _string_value(edge.get("target") or edge.get("target_node_id"))


def _edge_id(edge: dict[str, Any]) -> str:
    return _string_value(edge.get("id")) or f"{_edge_source(edge)}->{_edge_target(edge)}"


def _edge_keywords(edge: dict[str, Any]) -> str:
    return _string_value(edge.get("keywords"))


def _node_type(node_id: str, context: GenerationContext) -> str:
    node = context.nodes_by_id.get(node_id, {})
    return normalize_medical_entity_type(node.get("entity_type"))


def _is_value_like_edge(edge: dict[str, Any]) -> bool:
    return bool(_edge_keyword_tokens(edge) & _VALUE_LIKE_RELATION_KEYWORDS)


def _edge_keyword_tokens(edge: dict[str, Any]) -> set[str]:
    return set(split_relation_tokens(edge.get("keywords")))


def _qualifier_key(issue: dict[str, Any]) -> str:
    entity_type = _normalize_token(issue.get("entity_type"))
    label = str(issue.get("label") or "")
    if entity_type in {"dosage", "dose", "dosing"} or _DOSE_PATTERN.search(label):
        return "dose"
    if entity_type in {"frequency", "interval"} or _FREQUENCY_PATTERN.search(label):
        return "frequency"
    if entity_type in {"duration", "course"} or _DURATION_PATTERN.search(label):
        return "duration"
    if (
        entity_type in {"route", "administrationroute", "administration_route"}
        or _ROUTE_PATTERN.search(label)
    ):
        return "route"
    if entity_type in {"age", "agerange", "agegroup", "age_group"} or _AGE_PATTERN.search(
        label
    ):
        return "age"
    if entity_type in {"population", "patientpopulation", "patient_population"}:
        return "population"
    if (
        entity_type in {"timing", "time", "timewindow", "time_window"}
        or _TIME_WINDOW_PATTERN.search(label)
    ):
        return "time_window"
    return ""


def _no_safe_repair(issue: dict[str, Any], error_code: str) -> dict[str, Any]:
    return candidate_rejection(
        issue,
        error_code=error_code,
        error=f"No safe deterministic value-node qualifier repair: {error_code}",
    )


def _normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "", str(value or "").strip().casefold())


def _string_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
