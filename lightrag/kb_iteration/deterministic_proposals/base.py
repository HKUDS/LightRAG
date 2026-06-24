from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

IssueCandidateBuilder = Callable[[dict[str, Any]], dict[str, Any] | None]


@dataclass(frozen=True)
class GenerationContext:
    issue_family: str
    issues: Sequence[dict[str, Any]]
    builders: Sequence[IssueCandidateBuilder] = ()
    edges_by_id: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    nodes_by_id: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    all_edges: Sequence[dict[str, Any]] = ()


@dataclass(frozen=True)
class CandidateGenerationResult:
    candidates: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)


def generate_with_builders(context: GenerationContext) -> CandidateGenerationResult:
    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for issue in context.issues:
        if is_conflict_review_issue(issue):
            rejections.append(
                candidate_rejection(
                    issue,
                    error_code="REVIEW_CONFLICT_NOT_MUTATED",
                    error="Conflict review issues require human/LLM review, not a deterministic mutation.",
                )
            )
            continue
        for builder in context.builders:
            try:
                candidate = builder(issue)
            except ValueError as exc:
                rejections.append(
                    candidate_rejection(
                        issue,
                        error_code="DETERMINISTIC_BUILDER_ERROR",
                        error=str(exc),
                    )
                )
                continue
            if candidate is not None:
                candidates.append(_candidate_with_issue_ref(candidate, issue))
    return CandidateGenerationResult(
        candidates=_dedupe_candidates(candidates),
        rejections=rejections,
    )


def candidate_rejection(
    issue: Mapping[str, Any],
    *,
    error_code: str,
    error: str,
) -> dict[str, Any]:
    issue_ref = str(issue.get("issue_ref") or "")
    return {
        "candidate_id": _rejection_candidate_id(issue, error_code=error_code),
        "issue_ref": issue_ref,
        "issue_source": str(issue.get("issue_source") or ""),
        "issue_family": str(issue.get("issue_family") or ""),
        "issue_kind": str(issue.get("issue_kind") or ""),
        "stage": "deterministic_candidate",
        "error_code": error_code,
        "error": error,
    }


def _rejection_candidate_id(
    issue: Mapping[str, Any],
    *,
    error_code: str,
) -> str:
    basis = "|".join(
        str(issue.get(field_name) or "")
        for field_name in (
            "issue_ref",
            "issue_source",
            "issue_family",
            "issue_kind",
            "edge_id",
            "node_id",
        )
    )
    digest = hashlib.sha256(f"{basis}|{error_code}".encode("utf-8")).hexdigest()[:12]
    return f"rejection-{_safe_identifier_token(error_code)}-{digest}"


def _safe_identifier_token(value: str) -> str:
    token = "".join(
        char.casefold() if char.isalnum() else "-"
        for char in value.strip()
    ).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token[:48] or "candidate"


def replace_relation_candidate(
    issue: Mapping[str, Any],
    *,
    new_keywords: str,
    new_source: str | None = None,
    new_target: str | None = None,
    qualifiers: Mapping[str, Any] | None = None,
    issue_kind: str | None = None,
) -> dict[str, Any] | None:
    edge_id = string_value(issue.get("edge_id"))
    source = string_value(issue.get("source"))
    target = string_value(issue.get("target"))
    keywords = string_value(issue.get("keywords"))
    if not edge_id or not source or not target or not keywords or not new_keywords:
        return None

    resolved_source = new_source or source
    resolved_target = new_target or target
    if not resolved_source or not resolved_target:
        return None

    action_payload: dict[str, Any] = {
        "action": "replace_relation",
        "edge_id": edge_id,
        "expected_source": source,
        "expected_target": target,
        "current_keywords": keywords,
        "new_source": resolved_source,
        "new_target": resolved_target,
        "new_keywords": new_keywords,
        "qualifiers": dict(qualifiers if qualifiers is not None else issue_qualifiers(issue)),
    }
    source_type = string_value(issue.get("source_type"))
    target_type = string_value(issue.get("target_type"))
    new_source_type = _endpoint_type(
        resolved_source,
        source=source,
        target=target,
        source_type=source_type,
        target_type=target_type,
    )
    new_target_type = _endpoint_type(
        resolved_target,
        source=source,
        target=target,
        source_type=source_type,
        target_type=target_type,
    )
    if new_source_type:
        action_payload["new_source_type"] = new_source_type
    if new_target_type:
        action_payload["new_target_type"] = new_target_type

    return {
        "proposal_type": "medical_relation_schema_migration",
        "target": f"edge:{edge_id}",
        "issue_ref": string_value(issue.get("issue_ref")),
        "issue_kind": issue_kind or string_value(issue.get("issue_kind")),
        "action_payload": action_payload,
    }


def candidate_predicates(issue: Mapping[str, Any]) -> set[str]:
    return {
        str(predicate).strip()
        for predicate in issue.get("candidate_predicates", [])
        if isinstance(predicate, str) and predicate.strip()
    }


def issue_qualifiers(issue: Mapping[str, Any]) -> dict[str, Any]:
    qualifiers = issue.get("qualifiers")
    return dict(qualifiers) if isinstance(qualifiers, Mapping) else {}


def merged_qualifier_repair(issue: Mapping[str, Any]) -> dict[str, Any]:
    qualifiers = issue_qualifiers(issue)
    for key in ("suggested_qualifiers", "qualifier_repairs", "missing_qualifiers"):
        value = issue.get(key)
        if isinstance(value, Mapping):
            qualifiers.update(value)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            for qualifier_name in value:
                if not isinstance(qualifier_name, str) or not qualifier_name.strip():
                    continue
                qualifier_value = issue.get(qualifier_name)
                if qualifier_value not in (None, ""):
                    qualifiers[qualifier_name.strip()] = qualifier_value
    return qualifiers


def issue_text(issue: Mapping[str, Any]) -> str:
    return " ".join(
        string_value(issue.get(key)).casefold()
        for key in (
            "issue_kind",
            "keywords",
            "medical_subcase",
            "guidance",
            "suggested_action",
        )
    )


def is_conflict_review_issue(issue: Mapping[str, Any]) -> bool:
    return (
        string_value(issue.get("suggested_action")) == "review_conflict"
        or string_value(issue.get("issue_kind"))
        == "conflicting_recommendation_safety_scope"
    )


def type_is(value: Any, *expected: str) -> bool:
    normalized = normalize_type(value)
    return normalized in {normalize_type(item) for item in expected}


def normalize_type(value: Any) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


def string_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def combine_results(*results: CandidateGenerationResult) -> CandidateGenerationResult:
    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for result in results:
        candidates.extend(result.candidates)
        rejections.extend(result.rejections)
    return CandidateGenerationResult(
        candidates=_dedupe_candidates(candidates),
        rejections=rejections,
    )


def _candidate_with_issue_ref(
    candidate: dict[str, Any],
    issue: Mapping[str, Any],
) -> dict[str, Any]:
    if string_value(candidate.get("issue_ref")):
        return candidate
    return {
        **candidate,
        "issue_ref": string_value(issue.get("issue_ref")),
    }


def _endpoint_type(
    endpoint: str,
    *,
    source: str,
    target: str,
    source_type: str,
    target_type: str,
) -> str:
    if endpoint == source:
        return source_type
    if endpoint == target:
        return target_type
    return ""


def _dedupe_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for candidate in candidates:
        key = repr(
            (
                candidate.get("proposal_type"),
                candidate.get("target"),
                candidate.get("action_payload"),
            )
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(candidate))
    return deduped
