from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .agent_context import safe_package_child_path
from .issue_ledger import DeterministicScanResult, IssueRoute
from .models import ImprovementProposal

ROUTE_STATES = (
    "deterministic_covered",
    "llm_residual",
    "blocked_schema",
    "blocked_safety",
    "blocked_apply",
    "blocked_evidence",
    "deferred_budget",
    "duplicate",
    "stale",
    "unrouted",
)

TERMINAL_GENERATION_DISPOSITIONS = {
    "deterministic_covered",
    "llm_residual",
    "blocked_safety",
    "blocked_schema",
    "blocked_evidence",
    "blocked_apply",
    "blocked_decision_memory",
    "conflict_requires_review",
    "deferred_budget",
    "duplicate_issue",
    "stale_issue",
    "unsupported_family",
    "conversion_failed",
    "llm_output_invalid",
}


@dataclass(frozen=True)
class ApprovalQueueCandidateValidation:
    allowed: bool
    error_code: str = ""
    reason: str = ""


def build_proposal_funnel_report(
    scan: DeterministicScanResult,
    *,
    selected_proposal_ids: list[str],
    dropped: list[dict[str, Any]],
    conflict_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected_ids = _string_set(selected_proposal_ids)
    routes = list(scan.issue_routes)
    raw_issue_count = len(scan.issues)
    routed_issue_count = len(routes)
    raw_issue_refs = _issue_refs(scan.issues)
    route_issue_refs = _route_issue_refs(routes)
    duplicate_route_issue_refs = _duplicate_strings(route_issue_refs)
    raw_issue_ref_set = set(raw_issue_refs)
    route_issue_ref_set = set(route_issue_refs)
    missing_route_issue_refs = sorted(raw_issue_ref_set - route_issue_ref_set)
    extra_route_issue_refs = sorted(route_issue_ref_set - raw_issue_ref_set)
    route_state_counts = _route_state_counts(routes)
    final_generation_disposition_counts = _final_generation_disposition_counts(routes)
    unrouted_issue_count = (
        route_state_counts.get("unrouted", 0) + len(missing_route_issue_refs)
    )
    selected_route_proposal_ids = _selected_route_proposal_ids(routes, selected_ids)

    dropped_by_family = _dropped_by_family(dropped)
    conflict_groups = _report_conflict_groups(routes, conflict_groups or [])

    families: dict[str, dict[str, Any]] = {}
    for issue in scan.issues:
        family = _issue_family(issue)
        metrics = families.setdefault(family, _empty_family_metrics())
        metrics["raw_issue_count"] += 1
        issue_ref = _issue_ref(issue)
        if issue_ref:
            metrics["_raw_issue_refs"].add(issue_ref)
    for route in routes:
        family = route.family or "unknown"
        metrics = families.setdefault(family, _empty_family_metrics())
        _add_route_to_metrics(
            metrics,
            route,
            selected_ids=selected_ids,
        )
    for issue in scan.issues:
        issue_ref = _issue_ref(issue)
        if issue_ref not in missing_route_issue_refs:
            continue
        metrics = families.setdefault(_issue_family(issue), _empty_family_metrics())
        metrics["missing_route_issue_count"] += 1
        metrics["unrouted_issue_count"] += 1
    for family, family_dropped in dropped_by_family.items():
        metrics = families.setdefault(family, _empty_family_metrics())
        metrics["_merge_drop_ids"].update(
            _deduped_strings(item.get("proposal_id") for item in family_dropped)
        )
    families = {
        family: _finalized_family_metrics(metrics)
        for family, metrics in sorted(families.items())
    }

    action_candidate_ids = _candidate_ids(scan.candidates)
    action_candidate_count = len(action_candidate_ids)
    selected_proposal_count = len(selected_route_proposal_ids)
    summary = {
        "raw_issue_count": raw_issue_count,
        "routed_issue_count": routed_issue_count,
        "unrouted_issue_count": unrouted_issue_count,
        "accounting_balanced": (
            raw_issue_ref_set == route_issue_ref_set
            and not duplicate_route_issue_refs
            and route_state_counts.get("unrouted", 0) == 0
        ),
        "missing_route_issue_refs": missing_route_issue_refs,
        "extra_route_issue_refs": extra_route_issue_refs,
        "duplicate_route_issue_refs": duplicate_route_issue_refs,
        "issue_with_candidate_count": sum(
            1 for route in routes if _deduped_strings(route.candidate_ids)
        ),
        "action_candidate_count": action_candidate_count,
        "selected_proposal_count": selected_proposal_count,
        "dropped_proposal_count": len(dropped),
        "route_state_counts": route_state_counts,
        "final_generation_disposition_counts": final_generation_disposition_counts,
        "issue_accounting_rate": _issue_accounting_rate(
            raw_issue_refs=raw_issue_refs,
            routes=routes,
        ),
        "candidate_validation_rate": _candidate_validation_rate(
            action_candidate_ids=action_candidate_ids,
            rejections=scan.rejections,
        ),
        "candidate_to_proposal_rate": _safe_rate(
            selected_proposal_count,
            action_candidate_count,
            empty_value=1.0,
        ),
        "queue_apply_support_rate": _queue_apply_support_rate(
            selected_proposal_count=selected_proposal_count,
            dropped=dropped,
        ),
        "hard_rejection_recurrence_count": _hard_rejection_recurrence_count(routes),
        "exact_duplicate_recurrence_count": (
            _exact_duplicate_recurrence_count(routes)
            + _duplicate_merge_drop_count(dropped)
        ),
        "known_bad_pattern_count": _known_bad_pattern_count(routes, scan.rejections),
        "reason_code_counts": _reason_code_counts(routes),
    }
    return {
        "summary": summary,
        "families": families,
        "conflict_groups": conflict_groups,
        "conflicts": conflict_groups,
    }


def validate_approval_queue_candidate(
    candidate: ImprovementProposal | dict[str, Any],
) -> ApprovalQueueCandidateValidation:
    evidence = _candidate_value(candidate, "evidence")
    if evidence is not None and not _evidence_is_string_list(evidence):
        return ApprovalQueueCandidateValidation(
            allowed=False,
            error_code="EVIDENCE_MUST_BE_STRING",
            reason="approval queue evidence must be a list of strings",
        )

    action_payload = _candidate_action_payload(candidate)
    if not isinstance(action_payload, dict):
        return ApprovalQueueCandidateValidation(allowed=True)

    proposal_type = _candidate_type(candidate)
    if proposal_type == "candidate_kg_expansion":
        missing_types = _candidate_expansion_edge_missing_endpoint_types(
            action_payload
        )
        if missing_types:
            return ApprovalQueueCandidateValidation(
                allowed=False,
                error_code="CANDIDATE_EDGE_TYPES_REQUIRED",
                reason="candidate KG expansion edges must provide endpoint types",
            )

    if _known_bad_medical_pattern(candidate, action_payload):
        return ApprovalQueueCandidateValidation(
            allowed=False,
            error_code="KNOWN_BAD_MEDICAL_PATTERN",
            reason="candidate matches a known unsafe medical relation pattern",
        )

    return ApprovalQueueCandidateValidation(allowed=True)


def write_proposal_funnel_artifacts(
    output_dir: Path,
    scan: DeterministicScanResult,
    *,
    selected_proposal_ids: list[str],
    dropped: list[dict[str, Any]],
    conflict_groups: list[dict[str, Any]] | None = None,
    task_packs: list[dict[str, Any]] | None = None,
    merge_payload: dict[str, Any] | None = None,
    subagent_index: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=selected_proposal_ids,
        dropped=dropped,
        conflict_groups=conflict_groups,
    )
    issue_ledger_payload = _issue_ledger_payload(scan)
    paths = {
        "issue_ledger": _write_json(
            output_dir,
            "issue_ledger.json",
            issue_ledger_payload,
        ),
        "proposal_issue_ledger": _write_json(
            output_dir,
            "proposal_issue_ledger.json",
            issue_ledger_payload,
        ),
        "deterministic_proposal_report": _write_json(
            output_dir,
            "deterministic_proposal_report.json",
            report,
        ),
        "proposal_funnel_report": _write_json(
            output_dir,
            "proposal_funnel_report.json",
            report,
        ),
        "proposal_conflict_groups": _write_json(
            output_dir,
            "proposal_conflict_groups.json",
            report["conflict_groups"],
        ),
        "proposal_task_packs": _write_json(
            output_dir,
            "proposal_task_packs.json",
            task_packs or [],
        ),
        "subagent_output_index": _write_json(
            output_dir,
            "subagent_outputs/index.json",
            subagent_index or [],
        ),
    }
    markdown_path = safe_package_child_path(
        output_dir,
        "deterministic_proposal_report.md",
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        _proposal_funnel_markdown(report),
        encoding="utf-8",
    )
    paths["deterministic_proposal_report_md"] = markdown_path
    funnel_markdown_path = safe_package_child_path(
        output_dir,
        "proposal_funnel_report.md",
    )
    funnel_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    funnel_markdown_path.write_text(
        _proposal_funnel_markdown(report),
        encoding="utf-8",
    )
    paths["proposal_funnel_report_md"] = funnel_markdown_path
    merge_report_path = safe_package_child_path(output_dir, "proposal_merge_report.md")
    merge_report_path.parent.mkdir(parents=True, exist_ok=True)
    merge_report_path.write_text(
        _proposal_merge_report_markdown(merge_payload or {}),
        encoding="utf-8",
    )
    paths["proposal_merge_report"] = merge_report_path
    return paths


def _candidate_value(
    candidate: ImprovementProposal | dict[str, Any],
    key: str,
) -> Any:
    if isinstance(candidate, ImprovementProposal):
        return getattr(candidate, key, None)
    return candidate.get(key)


def _candidate_type(candidate: ImprovementProposal | dict[str, Any]) -> str:
    if isinstance(candidate, ImprovementProposal):
        return candidate.type.strip()
    return str(candidate.get("type") or candidate.get("proposal_type") or "").strip()


def _candidate_action_payload(
    candidate: ImprovementProposal | dict[str, Any],
) -> Any:
    if isinstance(candidate, ImprovementProposal):
        return candidate.action_payload
    return candidate.get("action_payload")


def _evidence_is_string_list(evidence: Any) -> bool:
    return isinstance(evidence, list) and all(isinstance(item, str) for item in evidence)


def _candidate_expansion_edge_missing_endpoint_types(
    action_payload: dict[str, Any],
) -> bool:
    candidate_edges = action_payload.get("candidate_edges")
    if not isinstance(candidate_edges, list):
        return False

    node_types: dict[str, str] = {}
    candidate_nodes = action_payload.get("candidate_nodes")
    if isinstance(candidate_nodes, list):
        for raw_node in candidate_nodes:
            if not isinstance(raw_node, dict):
                continue
            node_id = str(raw_node.get("id") or raw_node.get("label") or "").strip()
            node_type = str(raw_node.get("entity_type") or "").strip()
            if node_id and node_type:
                node_types[node_id] = node_type

    for raw_edge in candidate_edges:
        if not isinstance(raw_edge, dict):
            continue
        source = str(raw_edge.get("source") or "").strip()
        target = str(raw_edge.get("target") or "").strip()
        source_type = str(
            raw_edge.get("source_type") or node_types.get(source) or ""
        ).strip()
        target_type = str(
            raw_edge.get("target_type") or node_types.get(target) or ""
        ).strip()
        if not source_type or not target_type:
            return True
    return False


def _known_bad_medical_pattern(
    candidate: ImprovementProposal | dict[str, Any],
    action_payload: dict[str, Any],
) -> bool:
    predicate = str(
        action_payload.get("new_keywords")
        or action_payload.get("predicate")
        or action_payload.get("keywords")
        or ""
    ).strip()
    source = str(action_payload.get("new_source") or action_payload.get("source") or "")
    target = str(action_payload.get("new_target") or action_payload.get("target") or "")
    source_type = _endpoint_type(candidate, action_payload, "source")
    target_type = _endpoint_type(candidate, action_payload, "target")

    if (
        predicate == "has_indication"
        and _is_type(source_type, "Vaccine")
        and _is_type(target_type, "Population")
    ):
        return True

    if (
        predicate == "has_indication"
        and _is_type(source_type, "Drug")
        and _looks_like_antibiotic(source)
        and _looks_like_influenza(target)
    ):
        return True

    if (
        predicate == "risk_factor_for"
        and _is_disease_like_type(source_type)
        and (
            _is_outcome_or_severity_type(target_type)
            or _contains_any(target, _OUTCOME_OR_SEVERITY_TERMS)
        )
    ):
        return True

    if predicate == "has_complication" and _is_disease_like_type(source_type):
        if _is_chronic_condition_or_population_type(target_type) or _contains_any(
            target, _CHRONIC_OR_POPULATION_TERMS
        ):
            return True

    return False


def _endpoint_type(
    candidate: ImprovementProposal | dict[str, Any],
    action_payload: dict[str, Any],
    endpoint: str,
) -> str:
    payload_fields = (
        f"new_{endpoint}_type",
        f"{endpoint}_type",
        f"expected_{endpoint}_type",
    )
    for field_name in payload_fields:
        value = action_payload.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = _candidate_value(candidate, f"{endpoint}_type")
    return value.strip() if isinstance(value, str) else ""


def _is_type(value: str, expected: str) -> bool:
    return value.strip().casefold() == expected.casefold()


def _is_disease_like_type(value: str) -> bool:
    normalized = value.strip().casefold()
    return normalized in {"disease", "clinicalcondition", "syndrome"}


def _is_outcome_or_severity_type(value: str) -> bool:
    normalized = value.strip().casefold()
    return normalized in {
        "outcome",
        "severity",
        "clinicalseverity",
        "severitygrade",
        "clinicalseveritygrade",
    }


def _is_chronic_condition_or_population_type(value: str) -> bool:
    normalized = value.strip().casefold()
    return normalized in {
        "population",
        "clinicalcondition",
        "chroniccondition",
        "chronicdisease",
        "chronicunderlyingcondition",
    }


def _looks_like_antibiotic(value: str) -> bool:
    return _contains_any(
        value,
        (
            "antibiotic",
            "antibacterial",
            "antimicrobial",
            "azithromycin",
            "amoxicillin",
            "macrolide",
        ),
    )


def _looks_like_influenza(value: str) -> bool:
    return _contains_any(value, ("influenza", "flu", "流感"))


def _contains_any(value: str, terms: tuple[str, ...]) -> bool:
    normalized = value.strip().casefold()
    return any(term.casefold() in normalized for term in terms)


_OUTCOME_OR_SEVERITY_TERMS = (
    "death",
    "mortality",
    "hospitalization",
    "severity",
    "severe",
    "outcome",
)
_CHRONIC_OR_POPULATION_TERMS = (
    "chronic",
    "population",
    "older adult",
    "elderly",
    "patient",
)


def _empty_family_metrics() -> dict[str, Any]:
    metrics = {
        "raw_issue_count": 0,
        "issue_with_candidate_count": 0,
        "deterministic_candidate_issue_count": 0,
        "action_candidate_count": 0,
        "schema_blocked_count": 0,
        "safety_blocked_count": 0,
        "evidence_blocked_count": 0,
        "apply_blocked_count": 0,
        "decision_memory_blocked_count": 0,
        "conflict_count": 0,
        "deferred_by_family_cap_count": 0,
        "deterministic_proposal_count": 0,
        "llm_residual_eligible_count": 0,
        "llm_residual_selected_count": 0,
        "valid_llm_proposal_count": 0,
        "conversion_failure_count": 0,
        "merge_drop_count": 0,
        "missing_route_issue_count": 0,
        "unrouted_issue_count": 0,
        "selected_approval_proposal_count": 0,
        "selected_proposal_count": 0,
        "unknown_route_state_count": 0,
        "reason_code_counts": {},
        "_raw_issue_refs": set(),
        "_candidate_ids": set(),
        "_selected_proposal_ids": set(),
        "_deterministic_proposal_ids": set(),
        "_valid_llm_proposal_ids": set(),
        "_merge_drop_ids": set(),
    }
    for state in ROUTE_STATES:
        if state == "unrouted":
            continue
        metrics[f"{state}_count"] = 0
    return metrics


def _add_route_to_metrics(
    metrics: dict[str, Any],
    route: IssueRoute,
    *,
    selected_ids: set[str],
) -> None:
    if route.issue_ref not in metrics["_raw_issue_refs"]:
        metrics["raw_issue_count"] += 1
        if route.issue_ref:
            metrics["_raw_issue_refs"].add(route.issue_ref)
    candidate_ids = _deduped_strings(route.candidate_ids)
    if candidate_ids:
        metrics["issue_with_candidate_count"] += 1
        metrics["_candidate_ids"].update(candidate_ids)
    state_key = f"{route.route_state}_count"
    if state_key in metrics:
        metrics[state_key] += 1
    elif route.route_state != "unrouted":
        metrics["unknown_route_state_count"] += 1
    else:
        metrics["unrouted_issue_count"] += 1
    route_selected = _deduped_strings(
        proposal_id
        for proposal_id in route.proposal_ids
        if proposal_id in selected_ids
    )
    metrics["_selected_proposal_ids"].update(route_selected)
    metrics["selected_approval_proposal_count"] += len(route_selected)
    disposition = _final_generation_disposition(route)
    dispositions = _route_dispositions(route)
    reason_codes = _route_reason_codes(route)
    if disposition == "deterministic_covered":
        metrics["deterministic_candidate_issue_count"] += 1
        metrics["_deterministic_proposal_ids"].update(_deduped_strings(route.proposal_ids))
    if "blocked_schema" in dispositions or _has_reason_prefix(
        reason_codes, ("SCHEMA_",)
    ):
        metrics["schema_blocked_count"] += 1
    if "blocked_safety" in dispositions or "KNOWN_BAD_MEDICAL_PATTERN" in reason_codes:
        metrics["safety_blocked_count"] += 1
    if "blocked_evidence" in dispositions or _has_reason_prefix(
        reason_codes, ("EVIDENCE_",)
    ):
        metrics["evidence_blocked_count"] += 1
    if "blocked_apply" in dispositions or _has_reason_prefix(reason_codes, ("APPLY_",)):
        metrics["apply_blocked_count"] += 1
    if (
        "blocked_decision_memory" in dispositions
        or "REJECTED_DECISION_MEMORY" in reason_codes
    ):
        metrics["decision_memory_blocked_count"] += 1
    if (
        "conflict_requires_review" in dispositions
        or "SAME_EDGE_CONFLICT" in reason_codes
    ):
        metrics["conflict_count"] += 1
    if "deferred_budget" in dispositions or "FAMILY_CAP_REACHED" in reason_codes:
        metrics["deferred_by_family_cap_count"] += 1
    if disposition == "llm_residual":
        metrics["llm_residual_eligible_count"] += 1
        if route_selected:
            metrics["llm_residual_selected_count"] += 1
            metrics["_valid_llm_proposal_ids"].update(route_selected)
    if "conversion_failed" in dispositions or "CONVERSION_FAILED" in reason_codes:
        metrics["conversion_failure_count"] += 1
    if route.reason_code:
        reason_counts = metrics["reason_code_counts"]
        reason_counts[route.reason_code] = reason_counts.get(route.reason_code, 0) + 1
    for reason_code in reason_codes:
        if reason_code == route.reason_code:
            continue
        reason_counts = metrics["reason_code_counts"]
        reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1


def _finalized_family_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(metrics)
    finalized.pop("_raw_issue_refs")
    finalized["action_candidate_count"] = len(finalized.pop("_candidate_ids"))
    finalized["selected_proposal_count"] = len(
        finalized.pop("_selected_proposal_ids")
    )
    finalized["deterministic_proposal_count"] = len(
        finalized.pop("_deterministic_proposal_ids")
    )
    finalized["valid_llm_proposal_count"] = len(
        finalized.pop("_valid_llm_proposal_ids")
    )
    finalized["merge_drop_count"] = len(finalized.pop("_merge_drop_ids"))
    finalized["blocked_safety_count"] = finalized["safety_blocked_count"]
    finalized["blocked_apply_count"] = finalized["apply_blocked_count"]
    finalized["blocked_evidence_count"] = finalized["evidence_blocked_count"]
    finalized["deferred_budget_count"] = finalized["deferred_by_family_cap_count"]
    return finalized


def _issue_ledger_payload(scan: DeterministicScanResult) -> dict[str, Any]:
    return {
        "issues": scan.issues,
        "candidates": scan.candidates,
        "rejections": scan.rejections,
        "issue_routes": [asdict(route) for route in scan.issue_routes],
    }


def _proposal_funnel_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Deterministic Proposal Funnel",
        "",
        "## Summary",
        "",
        f"- Raw issues: {summary['raw_issue_count']}",
        f"- Routed issues: {summary['routed_issue_count']}",
        f"- Unrouted issues: {summary['unrouted_issue_count']}",
        f"- Accounting balanced: {summary['accounting_balanced']}",
        f"- Issues with candidates: {summary['issue_with_candidate_count']}",
        f"- Action candidates: {summary['action_candidate_count']}",
        f"- Selected proposals: {summary['selected_proposal_count']}",
        f"- Dropped proposals: {summary['dropped_proposal_count']}",
        "",
        "## Families",
        "",
        (
            "| Family | Raw | With candidate | Candidates | Covered | Residual | "
            "Blocked | Deferred | Duplicate | Stale | Selected |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, metrics in report["families"].items():
        blocked = (
            metrics["blocked_safety_count"]
            + metrics["blocked_apply_count"]
            + metrics["blocked_evidence_count"]
        )
        lines.append(
            "| "
            f"{family} | "
            f"{metrics['raw_issue_count']} | "
            f"{metrics['issue_with_candidate_count']} | "
            f"{metrics['action_candidate_count']} | "
            f"{metrics['deterministic_covered_count']} | "
            f"{metrics['llm_residual_count']} | "
            f"{blocked} | "
            f"{metrics['deferred_budget_count']} | "
            f"{metrics['duplicate_count']} | "
            f"{metrics['stale_count']} | "
            f"{metrics['selected_proposal_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_json(output_dir: Path, relative_path: str, payload: Any) -> Path:
    path = safe_package_child_path(output_dir, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _route_state_counts(routes: list[IssueRoute]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        counts[route.route_state] = counts.get(route.route_state, 0) + 1
    return counts


def _final_generation_disposition_counts(routes: list[IssueRoute]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        disposition = _final_generation_disposition(route)
        if not disposition:
            continue
        counts[disposition] = counts.get(disposition, 0) + 1
    return counts


def _final_generation_disposition(route: IssueRoute) -> str:
    disposition = getattr(route, "generation_disposition", "")
    if isinstance(disposition, str) and disposition.strip():
        return disposition.strip()
    if route.route_state == "duplicate":
        if route.reason_code == "REJECTED_DECISION_MEMORY":
            return "blocked_decision_memory"
        return "duplicate_issue"
    if route.route_state == "stale":
        return "stale_issue"
    if route.route_state in TERMINAL_GENERATION_DISPOSITIONS:
        return route.route_state
    return ""


def _route_dispositions(route: IssueRoute) -> set[str]:
    dispositions = {_final_generation_disposition(route)}
    events = getattr(route, "events", [])
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            disposition = event.get("generation_disposition")
            if isinstance(disposition, str) and disposition.strip():
                dispositions.add(disposition.strip())
    return {disposition for disposition in dispositions if disposition}


def _route_reason_codes(route: IssueRoute) -> set[str]:
    codes = set(_deduped_strings(getattr(route, "reason_codes", [])))
    if route.reason_code:
        codes.add(route.reason_code)
    events = getattr(route, "events", [])
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            reason_code = event.get("reason_code")
            if isinstance(reason_code, str) and reason_code.strip():
                codes.add(reason_code.strip())
    return codes


def _has_reason_prefix(reason_codes: set[str], prefixes: tuple[str, ...]) -> bool:
    return any(
        reason_code.startswith(prefix)
        for reason_code in reason_codes
        for prefix in prefixes
    )


def _issue_accounting_rate(
    *,
    raw_issue_refs: list[str],
    routes: list[IssueRoute],
) -> float:
    raw_issue_ref_set = set(raw_issue_refs)

    routes_by_ref: dict[str, list[IssueRoute]] = {}
    for route in routes:
        issue_ref = route.issue_ref.strip() if isinstance(route.issue_ref, str) else ""
        if not issue_ref:
            continue
        routes_by_ref.setdefault(issue_ref, []).append(route)

    if not raw_issue_ref_set:
        return 1.0 if not routes_by_ref else 0.0

    accounted = 0
    for issue_ref in raw_issue_ref_set:
        issue_routes = routes_by_ref.get(issue_ref, [])
        if len(issue_routes) != 1:
            continue
        disposition = _final_generation_disposition(issue_routes[0])
        if disposition in TERMINAL_GENERATION_DISPOSITIONS:
            accounted += 1

    extra_route_ref_count = len(set(routes_by_ref) - raw_issue_ref_set)
    duplicate_route_penalty = sum(
        max(0, len(issue_routes) - 1)
        for issue_ref, issue_routes in routes_by_ref.items()
        if issue_ref in raw_issue_ref_set
    )
    denominator = (
        len(raw_issue_ref_set) + extra_route_ref_count + duplicate_route_penalty
    )
    return accounted / denominator if denominator else 1.0


def _candidate_validation_rate(
    *,
    action_candidate_ids: list[str],
    rejections: list[dict[str, Any]],
) -> float:
    validation_failure_ids = _candidate_validation_failure_ids(rejections)
    valid_candidate_ids = set(action_candidate_ids) - validation_failure_ids
    return _safe_rate(
        len(valid_candidate_ids),
        len(valid_candidate_ids) + len(validation_failure_ids),
        empty_value=1.0,
    )


def _candidate_validation_failure_ids(rejections: list[dict[str, Any]]) -> set[str]:
    validation_stages = {
        "candidate_validation",
        "deterministic_candidate",
        "prevalidation",
        "schema_validation",
        "safety_validation",
        "evidence_validation",
        "apply_validation",
        "apply_capability",
    }
    validation_error_codes = {
        "KNOWN_BAD_MEDICAL_PATTERN",
        "EVIDENCE_MUST_BE_STRING",
        "EVIDENCE_TUPLE_MUST_MATCH_ALLOWED_SPAN",
        "CANDIDATE_EVIDENCE_INVALID",
        "CANDIDATE_EDGE_TYPES_REQUIRED",
        "APPLY_UNSUPPORTED",
        "ENTITY_MERGE_APPLY_NOT_SUPPORTED",
    }
    failed: set[str] = set()
    for rejection in rejections:
        if not isinstance(rejection, dict):
            continue
        stage = str(rejection.get("stage") or "").strip()
        error_code = str(rejection.get("error_code") or "").strip()
        candidate_id = str(rejection.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        if stage not in validation_stages and error_code not in validation_error_codes:
            continue
        failed.add(candidate_id)
    return failed


def _queue_apply_support_rate(
    *,
    selected_proposal_count: int,
    dropped: list[dict[str, Any]],
) -> float:
    apply_drops = sum(
        1
        for item in dropped
        if isinstance(item, dict)
        and str(item.get("reason") or item.get("error_code") or "")
        in {"blocked_apply", "APPLY_UNSUPPORTED"}
    )
    return _safe_rate(
        selected_proposal_count,
        selected_proposal_count + apply_drops,
        empty_value=1.0,
    )


def _safe_rate(numerator: int, denominator: int, *, empty_value: float) -> float:
    if denominator <= 0:
        return empty_value
    return numerator / denominator


def _hard_rejection_recurrence_count(routes: list[IssueRoute]) -> int:
    return sum(
        1
        for route in routes
        if "HARD_REJECTION_RECURRED" in _route_reason_codes(route)
    )


def _exact_duplicate_recurrence_count(routes: list[IssueRoute]) -> int:
    return sum(
        1
        for route in routes
        if _route_reason_codes(route)
        & {"DUPLICATE_ACTION_FINGERPRINT", "EXACT_DUPLICATE_RECURRED"}
    )


def _duplicate_merge_drop_count(dropped: list[dict[str, Any]]) -> int:
    return sum(
        1
        for item in dropped
        if isinstance(item, dict)
        and "duplicate" in str(item.get("reason") or "").casefold()
    )


def _known_bad_pattern_count(
    routes: list[IssueRoute],
    rejections: list[dict[str, Any]],
) -> int:
    matches: set[str] = set()
    for route in routes:
        if "KNOWN_BAD_MEDICAL_PATTERN" in _route_reason_codes(route):
            matches.add(route.issue_ref)
    for rejection in rejections:
        if not isinstance(rejection, dict):
            continue
        if str(rejection.get("error_code") or "") != "KNOWN_BAD_MEDICAL_PATTERN":
            continue
        key = str(
            rejection.get("issue_ref")
            or rejection.get("candidate_id")
            or json.dumps(rejection, sort_keys=True)
        )
        matches.add(key)
    return len(matches)


def _reason_code_counts(routes: list[IssueRoute]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        for reason_code in _route_reason_codes(route):
            counts[reason_code] = counts.get(reason_code, 0) + 1
    return counts


def _dropped_by_family(dropped: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for item in dropped:
        if not isinstance(item, dict):
            continue
        family = str(item.get("issue_family") or "unknown").strip() or "unknown"
        by_family.setdefault(family, []).append(item)
    return by_family


def _report_conflict_groups(
    routes: list[IssueRoute],
    conflict_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = [
        dict(group) for group in conflict_groups if isinstance(group, dict)
    ]
    seen = {
        (
            str(group.get("target") or ""),
            ",".join(_deduped_strings(group.get("proposal_ids"))),
            str(group.get("issue_ref") or ""),
        )
        for group in groups
    }
    for route in routes:
        reason_codes = _route_reason_codes(route)
        if (
            "SAME_EDGE_CONFLICT" not in reason_codes
            and "conflict_requires_review" not in _route_dispositions(route)
        ):
            continue
        group = {
            "issue_ref": route.issue_ref,
            "family": route.family,
            "proposal_ids": _deduped_strings(route.proposal_ids),
            "candidate_ids": _deduped_strings(route.candidate_ids),
            "reason": route.reason or route.reason_code or "conflict_requires_review",
        }
        key = (
            str(group.get("target") or ""),
            ",".join(_deduped_strings(group.get("proposal_ids"))),
            str(group.get("issue_ref") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        groups.append(group)
    return groups


def _proposal_merge_report_markdown(merge_payload: dict[str, Any]) -> str:
    proposals = merge_payload.get("proposals")
    conflicts = merge_payload.get("conflicts")
    dropped = merge_payload.get("dropped")
    return "\n".join(
        [
            "# Proposal Merge Report",
            "",
            f"- Selected proposals: {len(proposals) if isinstance(proposals, list) else 0}",
            f"- Conflicts: {len(conflicts) if isinstance(conflicts, list) else 0}",
            f"- Dropped: {len(dropped) if isinstance(dropped, list) else 0}",
            "",
        ]
    )


def _selected_route_proposal_ids(
    routes: list[IssueRoute],
    selected_ids: set[str],
) -> set[str]:
    selected: set[str] = set()
    for route in routes:
        selected.update(
            proposal_id
            for proposal_id in _deduped_strings(route.proposal_ids)
            if proposal_id in selected_ids
        )
    return selected


def _issue_refs(issues: list[dict[str, Any]]) -> list[str]:
    return _deduped_strings(issue.get("issue_ref") for issue in issues)


def _issue_ref(issue: dict[str, Any]) -> str:
    value = issue.get("issue_ref")
    return value.strip() if isinstance(value, str) else ""


def _issue_family(issue: dict[str, Any]) -> str:
    value = issue.get("issue_family")
    return value.strip() if isinstance(value, str) and value.strip() else "unknown"


def _route_issue_refs(routes: list[IssueRoute]) -> list[str]:
    return [
        route.issue_ref.strip()
        for route in routes
        if isinstance(route.issue_ref, str) and route.issue_ref.strip()
    ]


def _candidate_ids(candidates: list[dict[str, Any]]) -> list[str]:
    return _deduped_strings(candidate.get("candidate_id") for candidate in candidates)


def _duplicate_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _string_set(values: list[str]) -> set[str]:
    return {value.strip() for value in values if isinstance(value, str) and value.strip()}


def _deduped_strings(values: Any) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    if values is None:
        values = []
    elif not isinstance(values, list):
        values = list(values)
    for value in values:
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
