from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from .apply import proposal_apply_capability
from .proposal_fingerprints import (
    decision_fingerprints,
    decision_memory_suppresses_candidate,
)

IssueRouteState = Literal[
    "unrouted",
    "deterministic_covered",
    "llm_residual",
    "blocked_schema",
    "blocked_safety",
    "blocked_apply",
    "blocked_evidence",
    "deferred_budget",
    "duplicate",
    "stale",
]


@dataclass
class IssueRoute:
    issue_ref: str
    family: str
    issue_kind: str
    route_state: IssueRouteState = "unrouted"
    candidate_ids: list[str] = field(default_factory=list)
    proposal_ids: list[str] = field(default_factory=list)
    reason_code: str = ""
    reason: str = ""
    generation_disposition: str = ""
    queue_disposition: str = ""
    reason_codes: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.generation_disposition:
            self.generation_disposition = _generation_disposition_for_route(
                self.route_state,
                self.reason_code,
            )
        if self.reason_code and self.reason_code not in self.reason_codes:
            self.reason_codes.append(self.reason_code)

    def set_state(
        self,
        route_state: str,
        *,
        reason_code: str = "",
        reason: str = "",
        generation_disposition: str | None = None,
    ) -> None:
        self.route_state = route_state
        self.reason_code = reason_code
        self.reason = reason
        if reason_code and reason_code not in self.reason_codes:
            self.reason_codes.append(reason_code)
        self.generation_disposition = (
            generation_disposition
            or _generation_disposition_for_route(self.route_state, self.reason_code)
        )
        self.events.append(
            {
                "route_state": self.route_state,
                "generation_disposition": self.generation_disposition,
                "reason_code": self.reason_code,
                "reason": self.reason,
            }
        )


@dataclass
class DeterministicScanResult:
    issues: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    rejections: list[dict[str, Any]]
    issue_routes: list[IssueRoute]


MAX_DETERMINISTIC_FAMILY_CAP = 500
DEFAULT_DETERMINISTIC_FAMILY_CAPS: dict[str, int] = {
    "diagnosis": 40,
    "treatment": 40,
    "risk_safety": 35,
    "prevention": 30,
    "clinical_modeling": 30,
    "entity_cleanup": 25,
    "legacy_schema": 20,
}
DECISION_MEMORY_FILES = (
    "accepted_changes.md",
    "rejected_changes.md",
    "deferred_changes.md",
)
MODERN_DECISION_MEMORY_FIELDS = {
    "schema_version",
    "semantic_fingerprint",
    "execution_fingerprint",
    "evidence_fingerprint",
    "decision_reason_code",
    "rejection_scope",
    "semantic_rejection_class",
}
_FAMILY_PRIORITY = {
    family: index for index, family in enumerate(DEFAULT_DETERMINISTIC_FAMILY_CAPS)
}
_FAMILY_PRIORITY.update(
    {
        "direction": _FAMILY_PRIORITY["clinical_modeling"],
        "multi_predicate_split": _FAMILY_PRIORITY["treatment"],
        "alias_role_conflict": _FAMILY_PRIORITY["entity_cleanup"],
    }
)
_MUTATION_RISK_PRIORITY = {"low": 0, "medium": 1, "high": 2}


RAW_ISSUE_CONTRACT_DEFAULTS: dict[str, Any] = {
    "issue_ref": "",
    "issue_source": "",
    "issue_family": "legacy_schema",
    "issue_kind": "unknown_issue",
    "issue_order": 0,
    "issue_fingerprint": "",
    "edge_id": "",
    "source": "",
    "source_type": "",
    "target": "",
    "target_type": "",
    "keywords": "",
    "qualifiers": {},
    "candidate_predicates": [],
    "repair_options": [],
    "suggested_qualifiers": {},
    "source_id": "",
    "file_path": "",
    "evidence_quote": "",
    "evidence_spans": [],
    "auto_fixable": False,
    "blocked_reason": "",
}

STRUCTURED_ISSUE_SOURCES = (
    "medical_schema_issues",
    "entity_cleanup_issues",
    "evidence_issues",
    "generic_relation_issues",
    "hierarchy_issues",
)
ISSUE_FINGERPRINT_FIELDS = (
    "issue_source",
    "issue_family",
    "issue_kind",
    "edge_id",
    "node_id",
    "node_ids",
    "canonical_label",
    "source",
    "source_type",
    "target",
    "target_type",
    "keywords",
    "qualifiers",
    "candidate_predicates",
    "repair_options",
    "suggested_qualifiers",
    "source_id",
    "file_path",
    "evidence_quote",
    "evidence_spans",
    "blocked_reason",
    "suggested_action",
    "validation_errors",
    "category",
    "severity",
    "message",
    "evidence",
)


def collect_raw_issues(package_dir: str | Path) -> list[dict[str, Any]]:
    package = Path(package_dir)
    quality = _read_json(package / "snapshots" / "quality_score.json")
    return normalize_raw_issues(quality)


def scan_deterministic_candidates(
    package_dir: str | Path,
    *,
    prevalidate_action_candidates: bool = True,
    deterministic_family_caps: Mapping[str, Any] | None = None,
) -> DeterministicScanResult:
    from .proposal_orchestrator import (
        _action_candidates_for_issues,
        _edges_by_id,
        _issue_family,
        _issue_targets_previously_applied_edge,
        _issue_with_snapshot_context,
        _nodes_by_id,
        _preferred_role_for_issue_batch,
        _prevalidated_action_candidates,
        _stable_json_key,
        _stratified_issue_batches,
        _with_candidate_id,
    )

    package = Path(package_dir)
    raw_issues = collect_raw_issues(package)
    snapshot = _read_json(package / "snapshots" / "kg_snapshot.json")
    edges_by_id = _edges_by_id(snapshot)
    nodes_by_id = _nodes_by_id(snapshot)
    decision_memory = _decision_memory_records(package)
    legacy_rejected_fingerprints = _rejected_action_fingerprints(package)
    enriched_issues = [
        _issue_with_snapshot_context(issue, edges_by_id, nodes_by_id)
        for issue in raw_issues
    ]

    routes_by_ref: dict[str, IssueRoute] = {}
    issues_by_ref: dict[str, dict[str, Any]] = {}
    for issue in enriched_issues:
        issue_ref = _string_value(issue.get("issue_ref"))
        issues_by_ref[issue_ref] = issue
        routes_by_ref[issue_ref] = IssueRoute(
            issue_ref=issue_ref,
            family=_issue_family(issue),
            issue_kind=_string_value(issue.get("issue_kind")),
        )

    open_issues: list[dict[str, Any]] = []
    for issue in enriched_issues:
        route = routes_by_ref[_string_value(issue.get("issue_ref"))]
        if _issue_targets_previously_applied_edge(issue):
            _set_route_state(
                route,
                "stale",
                reason_code="PREVIOUSLY_APPLIED",
                reason="Issue targets an edge already normalized by an accepted change.",
            )
            continue
        if _issue_has_stale_expected_edge_keywords(issue, edges_by_id):
            _set_route_state(
                route,
                "stale",
                reason_code="STALE_EXPECTED_KEYWORDS",
                reason="Issue expected keywords no longer match the snapshot edge.",
            )
            continue
        open_issues.append(issue)

    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for family, chunk, _omitted in _stratified_issue_batches(
        open_issues,
        max_issues_per_pack=max(1, len(open_issues)),
        max_packs=max(1, len(open_issues)),
    ):
        role = _preferred_role_for_issue_batch(chunk, issue_family=family)
        generation = _action_candidates_for_issues(
            chunk,
            issue_family=family,
            edges_by_id=edges_by_id,
            nodes_by_id=nodes_by_id,
        )
        raw_candidates = generation.candidates
        if prevalidate_action_candidates:
            valid_candidates, validation_rejections = _prevalidated_action_candidates(
                raw_candidates,
                issues=chunk,
                role=role,
            )
        else:
            valid_candidates = [_with_candidate_id(candidate) for candidate in raw_candidates]
            validation_rejections = []
        candidates.extend(valid_candidates)
        rejections.extend([*generation.rejections, *validation_rejections])

    active_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        issue_ref = _string_value(candidate.get("issue_ref"))
        if _candidate_suppressed_by_decision_memory(
            candidate,
            decision_memory,
            legacy_rejected_fingerprints,
        ):
            rejections.append(
                {
                    "candidate_id": _string_value(candidate.get("candidate_id")),
                    "issue_ref": issue_ref,
                    "error_code": "REJECTED_DECISION_MEMORY",
                    "error": "Candidate action fingerprint was previously rejected.",
                    "generation_disposition": "blocked_decision_memory",
                    "candidate": candidate,
                }
            )
            continue
        active_candidates.append(candidate)

    supported_candidates: list[dict[str, Any]] = []
    for candidate in active_candidates:
        capability = proposal_apply_capability(candidate)
        if not capability.supported:
            issue_ref = _string_value(candidate.get("issue_ref"))
            candidate_id = _string_value(candidate.get("candidate_id"))
            rejections.append(
                {
                    "candidate_id": candidate_id,
                    "issue_ref": issue_ref,
                    "error_code": "APPLY_UNSUPPORTED",
                    "error": capability.reason,
                    "candidate": candidate,
                }
            )
            continue
        supported_candidates.append(candidate)

    active_candidates = _dedupe_candidate_actions(
        supported_candidates,
        rejections=rejections,
    )

    conflict_issue_refs = _same_edge_conflict_issue_refs(active_candidates, _stable_json_key)
    if conflict_issue_refs:
        conflict_candidates = [
            candidate
            for candidate in active_candidates
            if _string_value(candidate.get("issue_ref")) in conflict_issue_refs
        ]
        for candidate in conflict_candidates:
            issue_ref = _string_value(candidate.get("issue_ref"))
            route = routes_by_ref.get(issue_ref)
            if route is None:
                continue
            _set_route_state(
                route,
                "llm_residual",
                reason_code="SAME_EDGE_CONFLICT",
                reason="Multiple deterministic candidates conflict for the same edge.",
            )
            route.candidate_ids.append(_string_value(candidate.get("candidate_id")))
            rejections.append(
                {
                    "candidate_id": _string_value(candidate.get("candidate_id")),
                    "issue_ref": issue_ref,
                    "error_code": "SAME_EDGE_CONFLICT",
                    "error": "Multiple deterministic candidates conflict for the same edge.",
                    "candidate": candidate,
                }
            )
    valid_candidates = [
        candidate
        for candidate in active_candidates
        if _string_value(candidate.get("issue_ref")) not in conflict_issue_refs
    ]
    valid_candidates = _sorted_deterministic_candidates(
        valid_candidates,
        routes_by_ref=routes_by_ref,
        issues_by_ref=issues_by_ref,
    )
    valid_candidates, budget_rejections = _apply_deterministic_family_caps(
        valid_candidates,
        routes_by_ref=routes_by_ref,
        deterministic_family_caps=deterministic_family_caps,
    )
    rejections.extend(budget_rejections)

    for candidate in valid_candidates:
        issue_ref = _string_value(candidate.get("issue_ref"))
        route = routes_by_ref.get(issue_ref)
        if route is None or route.route_state not in {"unrouted"}:
            continue
        _set_route_state(
            route,
            "deterministic_covered",
            reason_code="DETERMINISTIC_CANDIDATE_VALID",
            reason="Valid deterministic candidate covers this issue.",
        )
        route.candidate_ids.append(_string_value(candidate.get("candidate_id")))

    _normalize_candidate_rejections(
        rejections,
        issues_by_ref=issues_by_ref,
        routes_by_ref=routes_by_ref,
    )

    rejection_codes_by_ref: dict[str, str] = {}
    rejection_candidate_ids_by_ref: dict[str, list[str]] = {}
    for rejection in rejections:
        issue_ref = _string_value(rejection.get("issue_ref"))
        error_code = _string_value(rejection.get("error_code"))
        if issue_ref and error_code:
            rejection_codes_by_ref.setdefault(issue_ref, error_code)
            rejection_candidate_ids_by_ref.setdefault(issue_ref, []).append(
                _string_value(rejection.get("candidate_id"))
            )

    for route in routes_by_ref.values():
        if route.route_state != "unrouted":
            continue
        rejection_code = rejection_codes_by_ref.get(route.issue_ref)
        if rejection_code in {
            "APPLY_UNSUPPORTED",
            "ENTITY_MERGE_APPLY_NOT_SUPPORTED",
        }:
            _set_route_state(
                route,
                "blocked_apply",
                reason_code=rejection_code,
                reason="No statically supported apply handler is available.",
            )
        elif rejection_code == "REJECTED_DECISION_MEMORY":
            _set_route_state(
                route,
                "duplicate",
                reason_code=rejection_code,
                reason="A previously rejected deterministic action suppresses this issue.",
                generation_disposition="blocked_decision_memory",
            )
        elif rejection_code == "DUPLICATE_ACTION_FINGERPRINT":
            _set_route_state(
                route,
                "duplicate",
                reason_code=rejection_code,
                reason="No non-duplicate deterministic candidate remains for this issue.",
            )
        elif blocked_disposition := _blocked_disposition_for_rejection_code(
            rejection_code
        ):
            route_state = (
                blocked_disposition
                if blocked_disposition
                in {"blocked_schema", "blocked_safety", "blocked_evidence"}
                else "unrouted"
            )
            _set_route_state(
                route,
                route_state,
                reason_code=rejection_code,
                reason="No valid deterministic candidate remains for this issue.",
                generation_disposition=blocked_disposition,
            )
        else:
            _set_route_state(
                route,
                "llm_residual",
                reason_code=rejection_code or "NO_VALID_DETERMINISTIC_CANDIDATE",
                reason="No valid deterministic candidate remains for this issue.",
            )
        route.candidate_ids.extend(
            candidate_id
            for candidate_id in rejection_candidate_ids_by_ref.get(route.issue_ref, [])
            if candidate_id
        )

    valid_candidate_ids = {
        _string_value(candidate.get("candidate_id")) for candidate in valid_candidates
    }
    return DeterministicScanResult(
        issues=enriched_issues,
        candidates=[
            candidate
            for candidate in valid_candidates
            if _string_value(candidate.get("candidate_id")) in valid_candidate_ids
        ],
        rejections=rejections,
        issue_routes=list(routes_by_ref.values()),
    )


def route_state_counts(issue_routes: list[IssueRoute]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in issue_routes:
        counts[route.route_state] = counts.get(route.route_state, 0) + 1
    return counts


def normalize_deterministic_family_caps(
    caps: Mapping[str, Any] | None,
) -> dict[str, int]:
    normalized = dict(DEFAULT_DETERMINISTIC_FAMILY_CAPS)
    if caps is None:
        return normalized
    for raw_family, raw_cap in caps.items():
        family = str(raw_family).strip()
        if not family:
            raise ValueError("deterministic_family_caps family names must be non-empty")
        if isinstance(raw_cap, bool) or not isinstance(raw_cap, int):
            raise ValueError("deterministic_family_caps values must be integers")
        if raw_cap < 1 or raw_cap > MAX_DETERMINISTIC_FAMILY_CAP:
            raise ValueError(
                "deterministic_family_caps values must be between 1 and "
                f"{MAX_DETERMINISTIC_FAMILY_CAP}"
            )
        normalized[family] = raw_cap
    return normalized


def _sorted_deterministic_candidates(
    candidates: list[dict[str, Any]],
    *,
    routes_by_ref: dict[str, IssueRoute],
    issues_by_ref: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda candidate: (
            _candidate_family_priority(candidate, routes_by_ref),
            _MUTATION_RISK_PRIORITY.get(_candidate_mutation_risk(candidate), 99),
            _candidate_issue_order(candidate, issues_by_ref),
            _string_value(candidate.get("candidate_id")),
        ),
    )


def _apply_deterministic_family_caps(
    candidates: list[dict[str, Any]],
    *,
    routes_by_ref: dict[str, IssueRoute],
    deterministic_family_caps: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    caps = normalize_deterministic_family_caps(deterministic_family_caps)
    kept: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    used_by_family: dict[str, int] = {}
    for candidate in candidates:
        issue_ref = _string_value(candidate.get("issue_ref"))
        route = routes_by_ref.get(issue_ref)
        family = route.family if route is not None else "legacy_schema"
        cap = caps.get(family, MAX_DETERMINISTIC_FAMILY_CAP)
        used = used_by_family.get(family, 0)
        if used < cap:
            kept.append(candidate)
            used_by_family[family] = used + 1
            continue

        candidate_id = _string_value(candidate.get("candidate_id"))
        if route is not None and route.route_state == "unrouted":
            _set_route_state(
                route,
                "deferred_budget",
                reason_code="FAMILY_CAP_REACHED",
                reason=f"Deterministic candidate family cap reached for {family}.",
            )
            if candidate_id:
                route.candidate_ids.append(candidate_id)
        rejections.append(
            {
                "candidate_id": candidate_id,
                "issue_ref": issue_ref,
                "error_code": "FAMILY_CAP_REACHED",
                "error": f"Deterministic candidate family cap reached for {family}.",
                "candidate": candidate,
            }
        )
    return kept, rejections


def _set_route_state(
    route: IssueRoute,
    route_state: str,
    *,
    reason_code: str = "",
    reason: str = "",
    generation_disposition: str | None = None,
) -> None:
    route.set_state(
        route_state,
        reason_code=reason_code,
        reason=reason,
        generation_disposition=generation_disposition,
    )


def _generation_disposition_for_route(
    route_state: str,
    reason_code: str = "",
) -> str:
    if route_state == "duplicate":
        if reason_code == "REJECTED_DECISION_MEMORY":
            return "blocked_decision_memory"
        return "duplicate_issue"
    if route_state == "stale":
        return "stale_issue"
    if route_state in {
        "deterministic_covered",
        "llm_residual",
        "blocked_schema",
        "blocked_safety",
        "blocked_apply",
        "blocked_evidence",
        "deferred_budget",
    }:
        return route_state
    return ""


def _blocked_disposition_for_rejection_code(error_code: str | None) -> str:
    normalized = error_code.strip().upper() if isinstance(error_code, str) else ""
    if not normalized:
        return ""
    if normalized == "KNOWN_BAD_MEDICAL_PATTERN":
        return "blocked_safety"
    if normalized in {
        "EVIDENCE_MUST_BE_STRING",
        "EVIDENCE_TUPLE_MUST_MATCH_ALLOWED_SPAN",
        "CANDIDATE_EVIDENCE_INVALID",
        "EVIDENCE_PAYLOAD_INVALID",
        "EVIDENCE_CONTRACT_INVALID",
        "PAYLOAD_EVIDENCE_CONTRACT_INVALID",
    }:
        return "blocked_evidence"
    if (
        normalized == "CANDIDATE_EDGE_TYPES_REQUIRED"
        or "SCHEMA" in normalized
        or "CONTRACT" in normalized
        or "PAYLOAD" in normalized
    ):
        return "blocked_schema"
    return ""


def _candidate_family_priority(
    candidate: dict[str, Any],
    routes_by_ref: dict[str, IssueRoute],
) -> int:
    issue_ref = _string_value(candidate.get("issue_ref"))
    route = routes_by_ref.get(issue_ref)
    family = route.family if route is not None else "legacy_schema"
    return _FAMILY_PRIORITY.get(family, len(_FAMILY_PRIORITY))


def _candidate_issue_order(
    candidate: dict[str, Any],
    issues_by_ref: dict[str, dict[str, Any]],
) -> int:
    issue = issues_by_ref.get(_string_value(candidate.get("issue_ref")), {})
    issue_order = issue.get("issue_order")
    return issue_order if isinstance(issue_order, int) else 0


def _candidate_mutation_risk(candidate: dict[str, Any]) -> str:
    proposal_type = _string_value(candidate.get("proposal_type"))
    action_payload = candidate.get("action_payload")
    action = ""
    if isinstance(action_payload, dict):
        action = _string_value(action_payload.get("action"))
    if proposal_type == "candidate_kg_expansion":
        return "high"
    if action in {"retire_relation", "entity_alias_merge"}:
        return "high"
    if action in {"split_relation", "value_node_to_qualifier"}:
        return "medium"
    if action == "replace_relation" and isinstance(action_payload, dict):
        expected_source = _string_value(action_payload.get("expected_source"))
        expected_target = _string_value(action_payload.get("expected_target"))
        new_source = _string_value(action_payload.get("new_source"))
        new_target = _string_value(action_payload.get("new_target"))
        if expected_source == new_source and expected_target == new_target:
            return "low"
        return "medium"
    return "medium"


def normalize_raw_issues(quality: dict[str, Any]) -> list[dict[str, Any]]:
    details = quality.get("details") if isinstance(quality, dict) else {}
    structured: list[dict[str, Any]] = []
    if isinstance(details, dict):
        for issue_source in STRUCTURED_ISSUE_SOURCES:
            for index, issue in enumerate(_dict_items(details.get(issue_source))):
                structured.append(
                    _normalize_issue(
                        issue,
                        issue_source=issue_source,
                        issue_order=index,
                    )
                )
    return _dedupe_issues(structured) if structured else _fallback_quality_findings(quality)


def _normalize_issue(
    issue: dict[str, Any],
    *,
    issue_source: str,
    issue_order: int,
) -> dict[str, Any]:
    normalized = _contract_defaults()
    normalized.update(deepcopy(issue))
    normalized["issue_source"] = issue_source
    normalized["issue_order"] = issue_order
    normalized["issue_kind"] = _string_value(
        normalized.get("issue_kind")
    ) or "unknown_issue"
    normalized["issue_family"] = _issue_family(
        {**normalized, "issue_family": issue.get("issue_family", "")},
        issue_source=issue_source,
    )
    normalized["edge_id"] = _string_value(normalized.get("edge_id"))
    normalized["source"] = _string_value(normalized.get("source"))
    normalized["source_type"] = _string_value(normalized.get("source_type"))
    normalized["target"] = _string_value(normalized.get("target"))
    normalized["target_type"] = _string_value(normalized.get("target_type"))
    normalized["keywords"] = _string_value(normalized.get("keywords"))
    normalized["qualifiers"] = _dict_value(normalized.get("qualifiers"))
    normalized["candidate_predicates"] = _list_value(
        normalized.get("candidate_predicates")
    )
    normalized["repair_options"] = _dict_list(normalized.get("repair_options"))
    normalized["suggested_qualifiers"] = _dict_value(
        normalized.get("suggested_qualifiers")
    )
    normalized["source_id"] = _string_value(normalized.get("source_id"))
    normalized["file_path"] = _string_value(normalized.get("file_path"))
    normalized["evidence_quote"] = _string_value(normalized.get("evidence_quote"))
    normalized["evidence_spans"] = _evidence_spans(normalized)
    normalized["auto_fixable"] = bool(normalized.get("auto_fixable", False))
    normalized["blocked_reason"] = _string_value(normalized.get("blocked_reason"))
    normalized["issue_fingerprint"] = _issue_fingerprint(normalized)
    normalized["issue_ref"] = _issue_ref(normalized)
    return normalized


def _fallback_quality_findings(quality: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(quality, dict):
        return []
    issues = []
    for index, finding in enumerate(_dict_items(quality.get("findings"))):
        issue = {
            "issue_kind": _string_value(finding.get("suggested_fix_type"))
            or _string_value(finding.get("category"))
            or "quality_finding",
            "severity": _string_value(finding.get("severity")),
            "category": _string_value(finding.get("category")),
            "blocked_reason": "fallback_quality_finding",
            "message": _string_value(finding.get("message"))[:240],
            "evidence_quote": _string_value(finding.get("message")),
        }
        evidence = finding.get("evidence")
        if isinstance(evidence, list):
            issue["evidence"] = [
                str(item).strip()[:240]
                for item in evidence[:10]
                if isinstance(item, str) and item.strip()
            ]
            if len(evidence) > 10:
                issue["evidence_omitted_count"] = len(evidence) - 10
            for item in evidence:
                if not isinstance(item, str):
                    continue
                if item.startswith("edge:"):
                    issue["edge_id"] = item.split(":", 1)[1]
                    break
                if item.startswith("node:"):
                    issue["node_id"] = item.split(":", 1)[1]
                    break
        issues.append(
            _normalize_issue(issue, issue_source="findings", issue_order=index)
        )
    return _dedupe_issues(issues)


def _issue_ref(issue: dict[str, Any]) -> str:
    issue_source = _string_value(issue.get("issue_source")) or "quality_issues"
    issue_family = _string_value(issue.get("issue_family")) or "legacy_schema"
    issue_kind = _string_value(issue.get("issue_kind")) or "unknown_issue"
    issue_fingerprint = (
        _string_value(issue.get("issue_fingerprint")) or _issue_fingerprint(issue)
    )
    fingerprint_prefix = issue_fingerprint[:12]
    primary_ref = _primary_issue_ref(issue)
    parts = [issue_source, issue_family, issue_kind, fingerprint_prefix]
    if primary_ref:
        parts.append(primary_ref)
    return ":".join(parts)


def _primary_issue_ref(issue: dict[str, Any]) -> str:
    for key in ("edge_id", "node_id", "canonical_label"):
        value = _string_value(issue.get(key))
        if value:
            return value
    node_ids = issue.get("node_ids")
    if isinstance(node_ids, list):
        values = sorted(
            str(node_id).strip()
            for node_id in node_ids
            if isinstance(node_id, str) and node_id.strip()
        )
        if values:
            return ",".join(values)
    return ""


def _issue_family(issue: dict[str, Any], *, issue_source: str) -> str:
    explicit = _string_value(issue.get("issue_family"))
    if issue_source == "findings" and explicit in {"", "legacy_schema"}:
        return "general"
    if explicit:
        return explicit
    if issue_source == "entity_cleanup_issues":
        return "entity_cleanup"
    kind = _string_value(issue.get("issue_kind")).casefold()
    predicates = " ".join(
        predicate.casefold() for predicate in _string_list(issue.get("candidate_predicates"))
    )
    source_type = _string_value(issue.get("source_type")).casefold()
    keywords = _string_value(issue.get("keywords")).casefold()
    text = f"{kind} {predicates} {keywords}"
    if kind.startswith("reverse_"):
        return "direction"
    if "multi_predicate" in text or "split" in text:
        return "multi_predicate_split"
    if any(
        token in text
        for token in (
            "diagnostic",
            "criterion",
            "test",
            "evidence",
            "supports_or_refutes",
        )
    ):
        return "diagnosis"
    if any(
        token in text for token in ("prevention", "vaccine", "reduces_risk")
    ):
        return "prevention"
    if "recommended_for" in predicates:
        if source_type in {"vaccine", "publichealthmeasure", "public_health_measure"}:
            return "prevention"
        return "treatment"
    if any(
        token in text
        for token in ("treatment", "indication", "recommends", "dosing", "drug")
    ):
        return "treatment"
    if any(token in text for token in ("risk", "complication", "adverse", "safety")):
        return "risk_safety"
    if any(token in text for token in ("manifestation", "causative_agent")):
        return "clinical_modeling"
    if any(token in text for token in ("alias", "role_conflict")):
        return "alias_role_conflict"
    if "cleanup" in text or "value_node" in text:
        return "entity_cleanup"
    if "direction" in text:
        return "direction"
    return "legacy_schema"


def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen: set[tuple[str, str]] = set()
    for issue in issues:
        issue_ref = _string_value(issue.get("issue_ref"))
        identity = _issue_identity(issue)
        dedupe_key = (issue_ref, identity)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(issue)
    return deduped


def _issue_identity(issue: dict[str, Any]) -> str:
    stable_issue = {
        key: issue[key]
        for key in ISSUE_FINGERPRINT_FIELDS
        if key in issue
    }
    return _stable_json(stable_issue)


def _issue_ref_disambiguator(issue: dict[str, Any]) -> str:
    return hashlib.sha256(_issue_identity(issue).encode("utf-8")).hexdigest()[:8]


def _issue_fingerprint(issue: dict[str, Any]) -> str:
    return hashlib.sha256(_issue_identity(issue).encode("utf-8")).hexdigest()


def _evidence_spans(issue: dict[str, Any]) -> list[dict[str, str]]:
    spans: list[dict[str, str]] = []
    raw_spans = issue.get("evidence_spans")
    if isinstance(raw_spans, list):
        for raw_span in raw_spans:
            if not isinstance(raw_span, dict):
                continue
            span = {
                "source_id": _string_value(raw_span.get("source_id")),
                "file_path": _string_value(raw_span.get("file_path")),
                "evidence_quote": _string_value(raw_span.get("evidence_quote")),
            }
            if any(span.values()):
                spans.append(span)

    legacy_span = {
        "source_id": _string_value(issue.get("source_id")),
        "file_path": _string_value(issue.get("file_path")),
        "evidence_quote": _string_value(issue.get("evidence_quote")),
    }
    if any(legacy_span.values()) and legacy_span not in spans:
        spans.append(legacy_span)
    return spans


def _contract_defaults() -> dict[str, Any]:
    return deepcopy(RAW_ISSUE_CONTRACT_DEFAULTS)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _issue_has_stale_expected_edge_keywords(
    issue: dict[str, Any],
    edges_by_id: dict[str, dict[str, Any]],
) -> bool:
    if _string_value(issue.get("suggested_action")) not in {
        "replace_relation",
        "retire_relation",
        "split_relation",
        "value_node_to_qualifier",
        "candidate_kg_expansion",
    }:
        return False
    edge_id = _string_value(issue.get("edge_id"))
    expected_keywords = _string_value(issue.get("keywords"))
    if not edge_id or not expected_keywords:
        return False
    edge = edges_by_id.get(edge_id)
    if not edge:
        return False
    current_keywords = _string_value(edge.get("keywords"))
    return bool(current_keywords and current_keywords != expected_keywords)


def _same_edge_conflict_issue_refs(
    candidates: list[dict[str, Any]],
    stable_json_key: Any,
) -> set[str]:
    by_edge: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        edge_id = _candidate_edge_id(candidate)
        if edge_id:
            by_edge.setdefault(edge_id, []).append(candidate)

    conflict_refs: set[str] = set()
    for edge_candidates in by_edge.values():
        if len(edge_candidates) <= 1:
            continue
        action_keys = {
            stable_json_key(candidate.get("action_payload", {}))
            for candidate in edge_candidates
        }
        if len(action_keys) <= 1:
            continue
        conflict_refs.update(
            _string_value(candidate.get("issue_ref"))
            for candidate in edge_candidates
            if _string_value(candidate.get("issue_ref"))
        )
    return conflict_refs


def _candidate_edge_id(candidate: dict[str, Any]) -> str:
    action_payload = candidate.get("action_payload")
    if isinstance(action_payload, dict):
        edge_id = _string_value(action_payload.get("edge_id"))
        if edge_id:
            return edge_id
    target = _string_value(candidate.get("target"))
    return target.removeprefix("edge:")


def _candidate_action_fingerprint(candidate: dict[str, Any]) -> str:
    return _stable_json(
        {
            "proposal_type": candidate.get("proposal_type", ""),
            "target": candidate.get("target", ""),
            "action_payload": candidate.get("action_payload", {}),
        }
    )


def _dedupe_candidate_actions(
    candidates: list[dict[str, Any]],
    *,
    rejections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    for candidate in candidates:
        fingerprint = _candidate_action_fingerprint(candidate)
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            deduped.append(candidate)
            continue

        issue_ref = _string_value(candidate.get("issue_ref"))
        candidate_id = _string_value(candidate.get("candidate_id"))
        rejections.append(
            {
                "candidate_id": candidate_id,
                "issue_ref": issue_ref,
                "error_code": "DUPLICATE_ACTION_FINGERPRINT",
                "error": "Candidate action duplicates an earlier deterministic candidate.",
                "candidate": candidate,
            }
        )
    return deduped


def _normalize_candidate_rejections(
    rejections: list[dict[str, Any]],
    *,
    issues_by_ref: dict[str, dict[str, Any]],
    routes_by_ref: dict[str, IssueRoute],
) -> None:
    for rejection in rejections:
        candidate = rejection.get("candidate")
        if not isinstance(candidate, dict):
            candidate = {}
        issue_ref = _string_value(rejection.get("issue_ref")) or _string_value(
            candidate.get("issue_ref")
        )
        error_code = (
            _string_value(rejection.get("error_code"))
            or "UNKNOWN_CANDIDATE_REJECTION"
        )
        stage = _string_value(rejection.get("stage")) or "deterministic_candidate"
        candidate_id = _string_value(rejection.get("candidate_id")) or _string_value(
            candidate.get("candidate_id")
        )
        rejection["issue_ref"] = issue_ref
        rejection["issue_family"] = _candidate_rejection_issue_family(
            rejection,
            candidate=candidate,
            issue_ref=issue_ref,
            issues_by_ref=issues_by_ref,
            routes_by_ref=routes_by_ref,
        )
        rejection["stage"] = stage
        rejection["error_code"] = error_code
        rejection["candidate_id"] = candidate_id or _synthetic_rejection_candidate_id(
            issue_ref=issue_ref,
            issue_family=rejection["issue_family"],
            stage=stage,
            error_code=error_code,
        )
        rejection["error"] = _string_value(rejection.get("error")) or rejection[
            "error_code"
        ]


def _candidate_rejection_issue_family(
    rejection: dict[str, Any],
    *,
    candidate: dict[str, Any],
    issue_ref: str,
    issues_by_ref: dict[str, dict[str, Any]],
    routes_by_ref: dict[str, IssueRoute],
) -> str:
    for value in (
        rejection.get("issue_family"),
        candidate.get("issue_family"),
        routes_by_ref.get(issue_ref).family if issue_ref in routes_by_ref else "",
        issues_by_ref.get(issue_ref, {}).get("issue_family"),
    ):
        family = _string_value(value)
        if family:
            return family
    return "legacy_schema"


def _synthetic_rejection_candidate_id(
    *,
    issue_ref: str,
    issue_family: str,
    stage: str,
    error_code: str,
) -> str:
    basis = f"{issue_ref}|{issue_family}|{stage}|{error_code}"
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]
    return f"rejection-{_safe_rejection_token(error_code)}-{digest}"


def _safe_rejection_token(value: str) -> str:
    token = "".join(
        char.casefold() if char.isalnum() else "-"
        for char in value.strip()
    ).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token[:48] or "candidate"


def _decision_memory_records(package: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_name in DECISION_MEMORY_FILES:
        path = package / file_name
        if not path.exists():
            continue
        records.extend(_json_payloads(path.read_text(encoding="utf-8")))
    return records


def _candidate_suppressed_by_decision_memory(
    candidate: dict[str, Any],
    decisions: list[dict[str, Any]],
    legacy_rejected_fingerprints: set[str],
) -> bool:
    if decision_memory_suppresses_candidate(candidate, decisions):
        return True
    return _candidate_action_fingerprint(candidate) in legacy_rejected_fingerprints


def _rejected_action_fingerprints(package: Path) -> set[str]:
    path = package / "rejected_changes.md"
    if not path.exists():
        return set()
    return {
        _decision_payload_fingerprint(payload)
        for payload in _json_payloads(path.read_text(encoding="utf-8"))
        if _string_value(payload.get("decision")).casefold() in {"reject", "rejected"}
        and _is_legacy_decision_payload(payload)
        and _decision_payload_fingerprint(payload)
    }


def _is_legacy_decision_payload(payload: dict[str, Any]) -> bool:
    if any(field in payload for field in MODERN_DECISION_MEMORY_FIELDS):
        return False
    return decision_fingerprints(payload) is None


def _decision_payload_fingerprint(payload: dict[str, Any]) -> str:
    action_payload = payload.get("action_payload")
    if not isinstance(action_payload, dict):
        return ""
    proposal_type = (
        _string_value(payload.get("proposal_type"))
        or _string_value(payload.get("type"))
    )
    target = (
        _string_value(payload.get("proposal_target"))
        or _string_value(payload.get("target"))
    )
    return _stable_json(
        {
            "proposal_type": proposal_type,
            "target": target,
            "action_payload": action_payload,
        }
    )


_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def _json_payloads(text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for block in _JSON_BLOCK_PATTERN.findall(text):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict_value(value: Any) -> dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [deepcopy(item) for item in value if isinstance(item, dict)]


def _list_value(value: Any) -> list[Any]:
    return deepcopy(value) if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _string_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
