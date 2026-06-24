from __future__ import annotations

import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from lightrag.kb_iteration.agent_pipeline import action_candidate_proposals_from_scan
from lightrag.kb_iteration.apply import proposal_apply_capability
from lightrag.kb_iteration.deterministic_proposals.base import (
    CandidateGenerationResult,
)
from lightrag.kb_iteration.issue_ledger import (
    DeterministicScanResult,
    IssueRoute,
    collect_raw_issues,
    scan_deterministic_candidates,
)
from lightrag.kb_iteration.models import ImprovementProposal
from lightrag.kb_iteration.proposal_funnel import (
    build_proposal_funnel_report,
    write_proposal_funnel_artifacts,
)
from lightrag.kb_iteration.proposal_orchestrator import (
    build_llm_residual_task_packs,
)
from lightrag.kb_iteration.proposals import write_approval_queue


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


def test_every_raw_issue_has_terminal_generation_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _make_v3_accounting_package(tmp_path)
    _patch_all_issues_as_supported_candidates(monkeypatch)

    scan = scan_deterministic_candidates(
        package,
        prevalidate_action_candidates=False,
    )
    raw_issues = collect_raw_issues(package)

    assert len(raw_issues) == 14
    assert Counter(issue["issue_family"] for issue in raw_issues) == {
        "diagnosis": 4,
        "treatment": 4,
        "risk_safety": 4,
        "entity_cleanup": 2,
    }

    dispositions = [
        getattr(route, "generation_disposition", None)
        for route in scan.issue_routes
    ]

    assert len(dispositions) == len(raw_issues)
    assert all(
        disposition in TERMINAL_GENERATION_DISPOSITIONS
        for disposition in dispositions
    ), (
        "V3 ledger rows must expose terminal generation_disposition values for "
        "every raw issue instead of relying only on legacy route_state fields."
    )


def test_raw_issue_accounting_matches_disposition_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _make_v3_accounting_package(tmp_path)
    _patch_all_issues_as_supported_candidates(monkeypatch)

    scan = scan_deterministic_candidates(
        package,
        prevalidate_action_candidates=False,
    )
    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=[],
        dropped=[],
    )
    summary = report["summary"]

    disposition_counts = summary.get("final_generation_disposition_counts")

    assert disposition_counts is not None, (
        "V3 funnel reports must expose final_generation_disposition_counts."
    )
    assert summary["raw_issue_count"] == sum(disposition_counts.values())
    assert summary.get("issue_accounting_rate") == 1.0


def test_proposal_funnel_report_exposes_v3_summary_rates_and_family_fields(
    tmp_path: Path,
) -> None:
    scan = _make_report_scan()

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=[
            "proposal-deterministic-diagnosis",
            "proposal-llm-treatment",
        ],
        dropped=[
            {
                "proposal_id": "proposal-llm-treatment-duplicate",
                "reason": "duplicate_action_payload",
                "issue_ref": "issue-treatment-llm-selected",
                "issue_family": "treatment",
            },
            {
                "proposal_id": "proposal-llm-treatment-conflict",
                "reason": "conflicting_with_action_candidate",
                "target": "edge-treatment",
                "issue_family": "treatment",
            },
        ],
    )

    summary = report["summary"]
    assert summary["issue_accounting_rate"] == 1.0
    assert summary["candidate_validation_rate"] == 0.75
    assert summary["candidate_to_proposal_rate"] == 0.5
    assert summary["queue_apply_support_rate"] == 1.0
    assert summary["hard_rejection_recurrence_count"] == 1
    assert summary["exact_duplicate_recurrence_count"] == 2
    assert summary["known_bad_pattern_count"] == 1

    diagnosis = report["families"]["diagnosis"]
    assert diagnosis["raw_issue_count"] == 2
    assert diagnosis["deterministic_candidate_issue_count"] == 1
    assert diagnosis["action_candidate_count"] == 1
    assert diagnosis["schema_blocked_count"] == 1
    assert diagnosis["deterministic_proposal_count"] == 1
    assert diagnosis["selected_approval_proposal_count"] == 1

    treatment = report["families"]["treatment"]
    assert treatment["raw_issue_count"] == 4
    assert treatment["safety_blocked_count"] == 1
    assert treatment["evidence_blocked_count"] == 1
    assert treatment["apply_blocked_count"] == 1
    assert treatment["decision_memory_blocked_count"] == 1
    assert treatment["conflict_count"] == 1
    assert treatment["deferred_by_family_cap_count"] == 1
    assert treatment["llm_residual_eligible_count"] == 1
    assert treatment["llm_residual_selected_count"] == 1
    assert treatment["valid_llm_proposal_count"] == 1
    assert treatment["conversion_failure_count"] == 1
    assert treatment["merge_drop_count"] == 2
    assert treatment["selected_approval_proposal_count"] == 1

    paths = write_proposal_funnel_artifacts(
        tmp_path,
        scan,
        selected_proposal_ids=[
            "proposal-deterministic-diagnosis",
            "proposal-llm-treatment",
        ],
        dropped=[],
    )
    assert {
        "issue_ledger",
        "proposal_issue_ledger",
        "deterministic_proposal_report",
        "deterministic_proposal_report_md",
        "proposal_funnel_report",
        "proposal_funnel_report_md",
        "proposal_conflict_groups",
        "proposal_task_packs",
        "proposal_merge_report",
        "subagent_output_index",
    } <= set(paths)
    for path in paths.values():
        assert path.exists()
    assert (tmp_path / "proposal_funnel_report.json").is_file()
    assert (tmp_path / "proposal_funnel_report.md").is_file()
    assert (tmp_path / "proposal_conflict_groups.json").is_file()
    assert (tmp_path / "proposal_task_packs.json").is_file()
    assert (tmp_path / "proposal_merge_report.md").is_file()
    assert (tmp_path / "subagent_outputs" / "index.json").is_file()


def test_candidate_validation_rate_counts_rejected_candidates_in_denominator() -> None:
    scan = DeterministicScanResult(
        issues=[
            _report_issue("issue-evidence-contract", "diagnosis", "evidence"),
            _report_issue("issue-schema-contract", "diagnosis", "schema"),
        ],
        candidates=[],
        rejections=[
            {
                "candidate_id": "candidate-evidence-contract",
                "issue_ref": "issue-evidence-contract",
                "issue_family": "diagnosis",
                "stage": "candidate_validation",
                "error_code": "EVIDENCE_MUST_BE_STRING",
            },
            {
                "candidate_id": "candidate-schema-contract",
                "issue_ref": "issue-schema-contract",
                "issue_family": "diagnosis",
                "stage": "candidate_validation",
                "error_code": "CANDIDATE_EDGE_TYPES_REQUIRED",
            },
        ],
        issue_routes=[],
    )

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=[],
        dropped=[],
    )

    assert report["summary"]["candidate_validation_rate"] == 0.0


def test_candidate_validation_rate_uses_survivors_plus_validation_failures() -> None:
    scan = DeterministicScanResult(
        issues=[
            _report_issue("issue-valid-a", "diagnosis", "valid"),
            _report_issue("issue-valid-b", "diagnosis", "valid"),
            _report_issue("issue-valid-c", "diagnosis", "valid"),
            _report_issue("issue-invalid", "diagnosis", "evidence"),
        ],
        candidates=[
            {"candidate_id": "candidate-valid-a", "issue_ref": "issue-valid-a"},
            {"candidate_id": "candidate-valid-b", "issue_ref": "issue-valid-b"},
            {"candidate_id": "candidate-valid-c", "issue_ref": "issue-valid-c"},
        ],
        rejections=[
            {
                "candidate_id": "candidate-invalid",
                "issue_ref": "issue-invalid",
                "issue_family": "diagnosis",
                "stage": "candidate_validation",
                "error_code": "CANDIDATE_EVIDENCE_INVALID",
            },
            {
                "candidate_id": "llm-output-invalid",
                "issue_ref": "issue-valid-a",
                "issue_family": "diagnosis",
                "stage": "llm_output_validation",
                "error_code": "LLM_OUTPUT_INVALID",
            },
        ],
        issue_routes=[],
    )

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=[],
        dropped=[],
    )

    assert report["summary"]["candidate_validation_rate"] == 0.75


def test_proposal_funnel_family_raw_counts_include_missing_routes() -> None:
    routed_issue = _report_issue("issue-diagnosis-routed", "diagnosis", "direction")
    missing_issue = _report_issue("issue-prevention-missing", "prevention", "vaccine")
    scan = DeterministicScanResult(
        issues=[routed_issue, missing_issue],
        candidates=[],
        rejections=[],
        issue_routes=[
            IssueRoute(
                issue_ref="issue-diagnosis-routed",
                family="diagnosis",
                issue_kind="direction",
                route_state="llm_residual",
            )
        ],
    )

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=[],
        dropped=[],
    )

    assert report["summary"]["missing_route_issue_refs"] == ["issue-prevention-missing"]
    assert report["families"]["diagnosis"]["raw_issue_count"] == 1
    assert report["families"]["prevention"]["raw_issue_count"] == 1
    assert report["families"]["prevention"]["missing_route_issue_count"] == 1
    assert report["families"]["prevention"]["unrouted_issue_count"] == 1


def test_deterministic_scan_is_not_limited_by_max_subagent_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _make_v3_accounting_package(tmp_path)
    _patch_all_issues_as_supported_candidates(monkeypatch)

    scan = scan_deterministic_candidates(
        package,
        prevalidate_action_candidates=False,
    )
    _ = build_llm_residual_task_packs(
        package,
        scan,
        max_issues_per_pack=1,
        max_packs=1,
    )

    assert len(scan.issues) == 14
    assert len(scan.candidates) == 14
    assert {
        route.issue_ref
        for route in scan.issue_routes
        if route.route_state == "deterministic_covered"
    } == {issue["issue_ref"] for issue in scan.issues}


def test_llm_residual_has_no_deterministic_covered_issue_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _make_v3_accounting_package(tmp_path)
    covered_edge_ids = {"edge-diagnosis-0", "edge-treatment-0", "edge-risk-0"}
    _patch_selected_issues_as_supported_candidates(monkeypatch, covered_edge_ids)

    scan = scan_deterministic_candidates(
        package,
        prevalidate_action_candidates=False,
    )
    packs = build_llm_residual_task_packs(
        package,
        scan,
        max_issues_per_pack=20,
        max_packs=1,
    )

    covered_refs = {
        route.issue_ref
        for route in scan.issue_routes
        if route.route_state == "deterministic_covered"
    }
    residual_refs = {
        issue["issue_ref"]
        for pack in packs
        for issue in pack.issues
    }

    assert covered_refs
    assert residual_refs
    assert residual_refs.isdisjoint(covered_refs)


def test_hard_rejected_suffix_variant_is_suppressed() -> None:
    candidate = _replace_relation_candidate(candidate_id="candidate-treatment-v2")
    fingerprints = _candidate_fingerprints(candidate)
    decision = _decision_record(
        proposal_id="prop-action-candidate-treatment-001",
        decision="reject",
        rejection_scope="semantic_until_schema_change",
        fingerprints=fingerprints,
    )

    suffix_variant = {
        **candidate,
        "candidate_id": "candidate-treatment-v2-rerun-003",
    }

    assert _decision_memory_suppresses(suffix_variant, [decision]) is True


def test_applied_exact_execution_fingerprint_is_suppressed() -> None:
    candidate = _replace_relation_candidate()
    fingerprints = _candidate_fingerprints(candidate)
    decision = _decision_record(
        proposal_id="prop-action-candidate-treatment-001",
        decision="accept",
        rejection_scope="exact_action",
        fingerprints=fingerprints,
        apply_status="applied",
    )

    assert _decision_memory_suppresses(candidate, [decision]) is True


def test_evidence_changed_can_reopen_evidence_scoped_rejection() -> None:
    original = _replace_relation_candidate(
        evidence_quote="Oseltamivir is indicated for confirmed influenza."
    )
    reopened = _replace_relation_candidate(
        evidence_quote=(
            "Oseltamivir is indicated for confirmed influenza when started early."
        )
    )
    decision = _decision_record(
        proposal_id="prop-action-candidate-treatment-001",
        decision="reject",
        rejection_scope="semantic_until_evidence_change",
        fingerprints=_candidate_fingerprints(original),
        decision_reason_code="insufficient_evidence",
    )

    assert _decision_memory_suppresses(reopened, [decision]) is False


def test_defer_only_rejection_does_not_block_future_candidate() -> None:
    candidate = _replace_relation_candidate()
    decision = _decision_record(
        proposal_id="prop-action-candidate-treatment-001",
        decision="defer",
        rejection_scope="defer_only",
        fingerprints=_candidate_fingerprints(candidate),
    )

    assert _decision_memory_suppresses(candidate, [decision]) is False


def test_candidate_conversion_errors_are_recorded_not_swallowed(
    tmp_path: Path,
) -> None:
    package = _make_v3_accounting_package(tmp_path)
    issue = collect_raw_issues(package)[0]
    scan = DeterministicScanResult(
        issues=[issue],
        candidates=[
            {
                "candidate_id": "candidate-malformed-conversion",
                "proposal_type": "medical_relation_schema_migration",
                "target": f"edge:{issue['edge_id']}",
                "issue_ref": issue["issue_ref"],
                "issue_family": issue["issue_family"],
            }
        ],
        rejections=[],
        issue_routes=[
            IssueRoute(
                issue_ref=issue["issue_ref"],
                family=issue["issue_family"],
                issue_kind=issue["issue_kind"],
                route_state="deterministic_covered",
                candidate_ids=["candidate-malformed-conversion"],
            )
        ],
    )

    result = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )

    assert result.proposals == []
    assert len(result.rejected) == 1
    rejection = result.rejected[0]
    assert rejection["candidate_id"] == "candidate-malformed-conversion"
    assert rejection["issue_ref"] == issue["issue_ref"]
    assert rejection.get("issue_family") == "diagnosis"
    assert rejection["stage"] == "proposal_conversion"
    assert rejection["error_code"] == "CONVERSION_FAILED"
    assert str(rejection.get("error", "")).strip()
    assert getattr(scan.issue_routes[0], "generation_disposition", None) == (
        "conversion_failed"
    )


def test_queue_contains_only_apply_supported_proposals(tmp_path: Path) -> None:
    supported = _proposal(
        proposal_id="proposal-supported-replace-relation",
        proposal_type="medical_relation_schema_migration",
        action_payload=_replace_relation_payload(),
    )
    unsupported = _proposal(
        proposal_id="proposal-unsupported-alias-merge",
        proposal_type="entity_alias_merge",
        target="node:flu-alias",
        action_payload={"action": "entity_alias_merge", "node_ids": ["flu", "flu-alias"]},
    )

    assert proposal_apply_capability(supported).supported is True
    assert proposal_apply_capability(unsupported).supported is False

    queue_path = write_approval_queue([unsupported, supported], tmp_path)
    queued_ids = _proposal_ids_from_queue(queue_path)

    assert supported.id in queued_ids
    assert unsupported.id not in queued_ids


def test_known_bad_medical_patterns_never_enter_queue() -> None:
    gate = _approval_queue_candidate_gate()
    bad_candidates = [
        _bad_candidate(
            "candidate-vaccine-indication",
            source="influenza-vaccine",
            source_type="Vaccine",
            target="older-adults",
            target_type="Population",
            new_keywords="has_indication",
        ),
        _bad_candidate(
            "candidate-disease-death-risk",
            source="influenza",
            source_type="Disease",
            target="death",
            target_type="Outcome",
            new_keywords="risk_factor_for",
        ),
        _bad_candidate(
            "candidate-disease-severity-risk-type-only",
            source="influenza",
            source_type="Disease",
            target="grade-3",
            target_type="ClinicalSeverity",
            new_keywords="risk_factor_for",
        ),
        _bad_candidate(
            "candidate-chronic-disease-complication",
            source="influenza",
            source_type="Disease",
            target="chronic-heart-disease",
            target_type="ClinicalCondition",
            new_keywords="has_complication",
        ),
        _bad_candidate(
            "candidate-copd-complication-type-only",
            source="influenza",
            source_type="Disease",
            target="COPD",
            target_type="ChronicCondition",
            new_keywords="has_complication",
        ),
        _bad_candidate(
            "candidate-antibiotic-indication",
            source="azithromycin",
            source_type="Drug",
            target="influenza",
            target_type="Disease",
            new_keywords="has_indication",
        ),
        {
            "candidate_id": "candidate-expansion-missing-endpoint-types",
            "proposal_type": "candidate_kg_expansion",
            "target": "kg:candidate:unsafe-expansion",
            "action_payload": {
                "action": "candidate_kg_expansion",
                "candidate_nodes": [],
                "candidate_edges": [
                    {
                        "source": "influenza",
                        "target": "body-ache",
                        "keywords": "has_manifestation",
                    }
                ],
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "evidence_quote": "Influenza can include body aches.",
            },
        },
        {
            "candidate_id": "candidate-evidence-object",
            "proposal_type": "medical_relation_schema_migration",
            "target": "edge:edge-evidence-object",
            "evidence": [{"source_id": "chunk-1", "file_path": "guide.md"}],
            "action_payload": _replace_relation_payload(edge_id="edge-evidence-object"),
        },
    ]

    for candidate in bad_candidates:
        result = gate(candidate)
        assert _gate_allows(result) is False
        assert _gate_error_code(result) in {
            "KNOWN_BAD_MEDICAL_PATTERN",
            "CANDIDATE_EDGE_TYPES_REQUIRED",
            "EVIDENCE_MUST_BE_STRING",
        }


def _make_report_scan() -> DeterministicScanResult:
    issues = [
        _report_issue("issue-diagnosis-covered", "diagnosis", "direction"),
        _report_issue("issue-diagnosis-schema", "diagnosis", "schema"),
        _report_issue("issue-treatment-llm-selected", "treatment", "residual"),
        _report_issue("issue-treatment-safety", "treatment", "safety"),
        _report_issue("issue-treatment-evidence", "treatment", "evidence"),
        _report_issue("issue-treatment-apply", "treatment", "apply"),
    ]
    candidates = [
        {
            "candidate_id": "candidate-diagnosis-covered",
            "issue_ref": "issue-diagnosis-covered",
            "issue_family": "diagnosis",
            "proposal_id": "proposal-deterministic-diagnosis",
            "proposal_source": "deterministic",
        },
        {
            "candidate_id": "candidate-treatment-llm-selected",
            "issue_ref": "issue-treatment-llm-selected",
            "issue_family": "treatment",
            "proposal_id": "proposal-llm-treatment",
            "proposal_source": "llm",
        },
        {
            "candidate_id": "candidate-treatment-conversion-failed",
            "issue_ref": "issue-treatment-llm-selected",
            "issue_family": "treatment",
            "proposal_source": "llm",
        },
        {
            "candidate_id": "candidate-treatment-known-bad",
            "issue_ref": "issue-treatment-safety",
            "issue_family": "treatment",
        },
    ]
    return DeterministicScanResult(
        issues=issues,
        candidates=candidates,
        rejections=[
            {
                "candidate_id": "candidate-treatment-conversion-failed",
                "issue_ref": "issue-treatment-llm-selected",
                "issue_family": "treatment",
                "stage": "proposal_conversion",
                "error_code": "CONVERSION_FAILED",
                "error": "payload missing action_payload",
            },
            {
                "candidate_id": "candidate-treatment-known-bad",
                "issue_ref": "issue-treatment-safety",
                "issue_family": "treatment",
                "stage": "candidate_validation",
                "error_code": "KNOWN_BAD_MEDICAL_PATTERN",
                "error": "unsafe medical predicate",
            },
        ],
        issue_routes=[
            IssueRoute(
                issue_ref="issue-diagnosis-covered",
                family="diagnosis",
                issue_kind="direction",
                route_state="deterministic_covered",
                candidate_ids=["candidate-diagnosis-covered"],
                proposal_ids=["proposal-deterministic-diagnosis"],
                queue_disposition="selected",
            ),
            IssueRoute(
                issue_ref="issue-diagnosis-schema",
                family="diagnosis",
                issue_kind="schema",
                route_state="unrouted",
                reason_code="SCHEMA_VALIDATION_FAILED",
                generation_disposition="blocked_schema",
            ),
            IssueRoute(
                issue_ref="issue-treatment-llm-selected",
                family="treatment",
                issue_kind="residual",
                route_state="llm_residual",
                candidate_ids=[
                    "candidate-treatment-llm-selected",
                    "candidate-treatment-conversion-failed",
                ],
                proposal_ids=["proposal-llm-treatment"],
                reason_codes=[
                    "SAME_EDGE_CONFLICT",
                    "CONVERSION_FAILED",
                    "HARD_REJECTION_RECURRED",
                ],
                events=[
                    {
                        "generation_disposition": "conflict_requires_review",
                        "reason_code": "SAME_EDGE_CONFLICT",
                    },
                    {
                        "generation_disposition": "conversion_failed",
                        "reason_code": "CONVERSION_FAILED",
                    },
                ],
                queue_disposition="selected",
            ),
            IssueRoute(
                issue_ref="issue-treatment-safety",
                family="treatment",
                issue_kind="safety",
                route_state="blocked_safety",
                candidate_ids=["candidate-treatment-known-bad"],
                reason_code="KNOWN_BAD_MEDICAL_PATTERN",
            ),
            IssueRoute(
                issue_ref="issue-treatment-evidence",
                family="treatment",
                issue_kind="evidence",
                route_state="blocked_evidence",
                reason_code="EVIDENCE_REQUIRED",
            ),
            IssueRoute(
                issue_ref="issue-treatment-apply",
                family="treatment",
                issue_kind="apply",
                route_state="blocked_apply",
                reason_code="APPLY_UNSUPPORTED",
                reason_codes=[
                    "APPLY_UNSUPPORTED",
                    "REJECTED_DECISION_MEMORY",
                    "FAMILY_CAP_REACHED",
                    "DUPLICATE_ACTION_FINGERPRINT",
                ],
                events=[
                    {
                        "generation_disposition": "blocked_decision_memory",
                        "reason_code": "REJECTED_DECISION_MEMORY",
                    },
                    {
                        "generation_disposition": "deferred_budget",
                        "reason_code": "FAMILY_CAP_REACHED",
                    },
                    {
                        "generation_disposition": "duplicate_issue",
                        "reason_code": "DUPLICATE_ACTION_FINGERPRINT",
                    },
                ],
            ),
        ],
    )


def _report_issue(issue_ref: str, issue_family: str, issue_kind: str) -> dict[str, Any]:
    return {
        "issue_ref": issue_ref,
        "issue_family": issue_family,
        "issue_kind": issue_kind,
        "edge_id": issue_ref.replace("issue-", "edge-"),
        "source": f"source-{issue_ref}",
        "target": f"target-{issue_ref}",
    }


def _make_v3_accounting_package(tmp_path: Path) -> Path:
    package = tmp_path / "kb-iteration-v3"
    diagnosis = [
        _issue(
            f"edge-diagnosis-{index}",
            issue_family="diagnosis",
            issue_kind="diagnostic_evidence_direction_mismatch",
            source=f"positive-test-{index}",
            source_type="TestResultPattern",
            target="influenza",
            target_type="Disease",
            keywords="diagnostic_basis",
            candidate_predicates=["supports_or_refutes"],
        )
        for index in range(4)
    ]
    treatment = [
        _issue(
            f"edge-treatment-{index}",
            issue_family="treatment",
            issue_kind="treatment_domain_range_mismatch",
            source=f"oseltamivir-{index}",
            source_type="Drug",
            target="influenza",
            target_type="Disease",
            keywords="recommended_treatment",
            candidate_predicates=["has_indication"],
        )
        for index in range(4)
    ]
    risk_safety = [
        _issue(
            f"edge-risk-{index}",
            issue_family="risk_safety",
            issue_kind="risk_safety_relation_mismatch",
            source=f"risk-factor-{index}",
            source_type="RiskFactor",
            target="hospitalization",
            target_type="Outcome",
            keywords="complication_risk",
            candidate_predicates=["risk_factor_for"],
        )
        for index in range(4)
    ]
    cleanup = [
        _issue(
            f"edge-cleanup-{index}",
            issue_family="entity_cleanup",
            issue_kind="value_node_to_qualifier",
            source=f"dose-node-{index}",
            source_type="Dosage",
            target="oseltamivir",
            target_type="Drug",
            keywords="has_value",
            candidate_predicates=["value_node_to_qualifier"],
        )
        for index in range(2)
    ]
    medical_schema_issues = [*diagnosis, *treatment, *risk_safety]
    entity_cleanup_issues = cleanup
    issues = [*medical_schema_issues, *entity_cleanup_issues]

    _write_json(
        package / "snapshots" / "kg_snapshot.json",
        {
            "workspace": "influenza_medical_v1",
            "generated_at": "2026-06-23T00:00:00+08:00",
            "source_files": ["guide.md"],
            "nodes": [],
            "edges": [
                _edge(
                    str(issue["edge_id"]),
                    source=str(issue["source"]),
                    target=str(issue["target"]),
                    keywords=str(issue["keywords"]),
                    source_type=str(issue["source_type"]),
                    target_type=str(issue["target_type"]),
                )
                for issue in issues
            ],
            "metadata": {"profile": "clinical_guideline_zh"},
        },
    )
    _write_json(
        package / "snapshots" / "quality_score.json",
        {
            "overall": 62,
            "subscores": {},
            "metrics": {"medical_schema_issue_count": len(medical_schema_issues)},
            "details": {
                "medical_schema_issues": medical_schema_issues,
                "entity_cleanup_issues": entity_cleanup_issues,
            },
            "findings": [],
            "critical_blockers": [],
        },
    )
    (package / "accepted_changes.md").write_text("# Accepted Changes\n", encoding="utf-8")
    (package / "rejected_changes.md").write_text("# Rejected Changes\n", encoding="utf-8")
    (package / "deferred_changes.md").write_text("# Deferred Changes\n", encoding="utf-8")
    return package


def _issue(
    edge_id: str,
    *,
    issue_family: str,
    issue_kind: str,
    source: str,
    source_type: str,
    target: str,
    target_type: str,
    keywords: str,
    candidate_predicates: list[str],
) -> dict[str, Any]:
    return {
        "issue_family": issue_family,
        "issue_kind": issue_kind,
        "edge_id": edge_id,
        "source": source,
        "source_type": source_type,
        "target": target,
        "target_type": target_type,
        "keywords": keywords,
        "qualifiers": {},
        "candidate_predicates": candidate_predicates,
        "repair_options": [{"action": "replace_relation"}],
        "suggested_qualifiers": {},
        "source_id": f"chunk-{edge_id}",
        "file_path": "guide.md",
        "evidence_quote": f"Grounded evidence for {edge_id}.",
        "auto_fixable": True,
        "blocked_reason": "",
    }


def _edge(
    edge_id: str,
    *,
    source: str,
    target: str,
    keywords: str,
    source_type: str,
    target_type: str,
) -> dict[str, Any]:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "source_type": source_type,
        "target_type": target_type,
        "keywords": keywords,
        "description": "",
        "source_id": f"chunk-{edge_id}",
        "file_path": "guide.md",
        "weight": None,
        "properties": {},
    }


def _patch_all_issues_as_supported_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_selected_issues_as_supported_candidates(monkeypatch, covered_edge_ids=None)


def _patch_selected_issues_as_supported_candidates(
    monkeypatch: pytest.MonkeyPatch,
    covered_edge_ids: set[str] | None,
) -> None:
    def candidates(
        chunk: list[dict[str, Any]],
        **_kwargs: object,
    ) -> CandidateGenerationResult:
        generated = [
            _candidate_from_issue(issue)
            for issue in chunk
            if covered_edge_ids is None or issue.get("edge_id") in covered_edge_ids
        ]
        return CandidateGenerationResult(candidates=generated, rejections=[])

    monkeypatch.setattr(
        # RED seam: synthesize full-scan candidates without changing production code.
        "lightrag.kb_iteration.proposal_orchestrator._action_candidates_for_issues",
        candidates,
    )


def _candidate_from_issue(issue: dict[str, Any]) -> dict[str, Any]:
    edge_id = str(issue["edge_id"])
    source = str(issue["source"])
    target = str(issue["target"])
    candidate_predicates = issue.get("candidate_predicates")
    new_keywords = (
        str(candidate_predicates[0])
        if isinstance(candidate_predicates, list) and candidate_predicates
        else "has_indication"
    )
    return {
        "candidate_id": f"candidate-{edge_id}",
        "proposal_type": "medical_relation_schema_migration",
        "target": f"edge:{edge_id}",
        "issue_ref": issue["issue_ref"],
        "issue_family": issue["issue_family"],
        "issue_kind": issue["issue_kind"],
        "evidence_spans": [
            {
                "source_id": issue["source_id"],
                "file_path": issue["file_path"],
                "evidence_quote": issue["evidence_quote"],
            }
        ],
        "action_payload": _replace_relation_payload(
            edge_id=edge_id,
            expected_source=source,
            expected_target=target,
            current_keywords=str(issue["keywords"]),
            new_source=source,
            new_target=target,
            new_keywords=new_keywords,
        ),
    }


def _replace_relation_candidate(
    *,
    candidate_id: str = "candidate-treatment-v1",
    evidence_quote: str = "Oseltamivir is indicated for confirmed influenza.",
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "proposal_type": "medical_relation_schema_migration",
        "target": "edge:edge-oseltamivir-flu",
        "issue_ref": "medical_schema_issues:treatment:domain:abc123",
        "issue_family": "treatment",
        "schema_version": "medical-schema-v3",
        "evidence_spans": [
            {
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "evidence_quote": evidence_quote,
            }
        ],
        "action_payload": _replace_relation_payload(
            edge_id="edge-oseltamivir-flu",
            expected_source="oseltamivir",
            expected_target="influenza",
            current_keywords="recommended_treatment",
            new_source="oseltamivir",
            new_target="influenza",
            new_keywords="has_indication",
            qualifiers={"purpose": "treatment"},
        ),
    }


def _replace_relation_payload(
    *,
    edge_id: str = "edge-oseltamivir-flu",
    expected_source: str = "oseltamivir",
    expected_target: str = "influenza",
    current_keywords: str = "recommended_treatment",
    new_source: str = "oseltamivir",
    new_target: str = "influenza",
    new_keywords: str = "has_indication",
    qualifiers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "action": "replace_relation",
        "edge_id": edge_id,
        "expected_source": expected_source,
        "expected_target": expected_target,
        "current_keywords": current_keywords,
        "new_source": new_source,
        "new_target": new_target,
        "new_keywords": new_keywords,
        "qualifiers": qualifiers or {},
    }


def _candidate_fingerprints(candidate: dict[str, Any]) -> Any:
    module = _proposal_fingerprints_module()
    try:
        candidate_fingerprints = getattr(module, "candidate_fingerprints")
    except AttributeError as exc:
        pytest.fail(
            "V3 requires lightrag.kb_iteration.proposal_fingerprints."
            "candidate_fingerprints(candidate)."
        )
        raise exc
    return candidate_fingerprints(candidate)


def _decision_memory_suppresses(
    candidate: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> bool:
    module = _proposal_fingerprints_module()
    try:
        suppresses = getattr(module, "decision_memory_suppresses_candidate")
    except AttributeError as exc:
        pytest.fail(
            "V3 requires a public decision_memory_suppresses_candidate helper "
            "that applies semantic/execution/evidence rejection scopes."
        )
        raise exc
    return bool(
        suppresses(
            candidate,
            decisions,
            schema_version="medical-schema-v3",
        )
    )


def _proposal_fingerprints_module() -> Any:
    try:
        return importlib.import_module("lightrag.kb_iteration.proposal_fingerprints")
    except ImportError as exc:
        pytest.fail(
            "V3 requires lightrag.kb_iteration.proposal_fingerprints with "
            "candidate_fingerprints(), proposal_fingerprints(), and "
            "decision_fingerprints() helpers."
        )
        raise exc


def _decision_record(
    *,
    proposal_id: str,
    decision: str,
    rejection_scope: str,
    fingerprints: Any,
    decision_reason_code: str = "unsafe_semantics",
    apply_status: str = "",
) -> dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "decision": decision,
        "decision_reason_code": decision_reason_code,
        "semantic_rejection_class": decision_reason_code,
        "rejection_scope": rejection_scope,
        "schema_version": "medical-schema-v3",
        "semantic_fingerprint": fingerprints.semantic,
        "execution_fingerprint": fingerprints.execution,
        "evidence_fingerprint": fingerprints.evidence,
        "apply_status": apply_status,
        "action_payload": _replace_relation_payload(qualifiers={"purpose": "treatment"}),
    }


def _proposal(
    *,
    proposal_id: str,
    proposal_type: str,
    action_payload: dict[str, Any],
    target: str = "edge:edge-oseltamivir-flu",
) -> ImprovementProposal:
    return ImprovementProposal(
        id=proposal_id,
        type=proposal_type,
        target=target,
        proposed_change="Normalize the medical relation.",
        reason="The candidate is grounded and apply-capability checked.",
        evidence=["source_id: chunk-1; file_path: guide.md; relation_id: edge-oseltamivir-flu"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={"medical_schema_issue_count": -1},
        action_payload=action_payload,
    )


def _approval_queue_candidate_gate() -> Any:
    module = importlib.import_module("lightrag.kb_iteration.proposal_funnel")
    for name in (
        "validate_approval_queue_candidate",
        "validate_queue_candidate",
        "validate_candidate_before_queue",
    ):
        gate = getattr(module, name, None)
        if gate is not None:
            return gate
    pytest.fail(
        "V3 requires a public pre-queue candidate/proposal gate before "
        "approval_queue.md is written."
    )


def _proposal_ids_from_queue(queue_path: Path) -> set[str]:
    payload = yaml.safe_load(queue_path.read_text(encoding="utf-8")) or {}
    proposals = payload.get("proposals") if isinstance(payload, dict) else []
    if not isinstance(proposals, list):
        return set()
    return {
        str(proposal.get("id", "")).strip()
        for proposal in proposals
        if isinstance(proposal, dict) and str(proposal.get("id", "")).strip()
    }


def _gate_allows(result: Any) -> bool:
    if isinstance(result, bool):
        return result
    if isinstance(result, dict):
        return bool(result.get("allowed"))
    return bool(getattr(result, "allowed", False))


def _gate_error_code(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("error_code", ""))
    return str(getattr(result, "error_code", ""))


def _bad_candidate(
    candidate_id: str,
    *,
    source: str,
    source_type: str,
    target: str,
    target_type: str,
    new_keywords: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "proposal_type": "medical_relation_schema_migration",
        "target": f"edge:{candidate_id}",
        "source_type": source_type,
        "target_type": target_type,
        "evidence_spans": [
            {
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "evidence_quote": f"Evidence for {candidate_id}.",
            }
        ],
        "action_payload": _replace_relation_payload(
            edge_id=candidate_id,
            expected_source=source,
            expected_target=target,
            new_source=source,
            new_target=target,
            new_keywords=new_keywords,
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
