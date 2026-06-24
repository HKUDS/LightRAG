import json
from pathlib import Path

from lightrag.kb_iteration.issue_ledger import (
    DeterministicScanResult,
    IssueRoute,
)
from lightrag.kb_iteration.proposal_funnel import (
    build_proposal_funnel_report,
    write_proposal_funnel_artifacts,
)


def test_funnel_report_balances_issue_accounting() -> None:
    scan = _scan_with_routes()

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=["proposal-a"],
        dropped=[],
    )

    assert report["summary"]["raw_issue_count"] == 8
    assert report["summary"]["routed_issue_count"] == 8
    assert report["summary"]["unrouted_issue_count"] == 0
    assert report["summary"]["accounting_balanced"] is True
    assert report["summary"]["missing_route_issue_refs"] == []
    assert report["summary"]["extra_route_issue_refs"] == []
    assert report["summary"]["duplicate_route_issue_refs"] == []


def test_funnel_report_flags_mismatched_issue_route_refs() -> None:
    scan = DeterministicScanResult(
        issues=[
            {"issue_ref": "issue-a", "issue_family": "diagnosis"},
            {"issue_ref": "issue-b", "issue_family": "diagnosis"},
        ],
        candidates=[],
        rejections=[],
        issue_routes=[
            IssueRoute(
                issue_ref="issue-a",
                family="diagnosis",
                issue_kind="covered",
                route_state="deterministic_covered",
            ),
            IssueRoute(
                issue_ref="issue-c",
                family="diagnosis",
                issue_kind="extra",
                route_state="llm_residual",
            ),
        ],
    )

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=[],
        dropped=[],
    )

    assert report["summary"]["accounting_balanced"] is False
    assert report["summary"]["missing_route_issue_refs"] == ["issue-b"]
    assert report["summary"]["extra_route_issue_refs"] == ["issue-c"]
    assert report["summary"]["duplicate_route_issue_refs"] == []
    assert report["summary"]["issue_accounting_rate"] < 1.0


def test_funnel_report_penalizes_extra_route_only_mismatch() -> None:
    scan = DeterministicScanResult(
        issues=[
            {"issue_ref": "issue-a", "issue_family": "diagnosis"},
        ],
        candidates=[],
        rejections=[],
        issue_routes=[
            IssueRoute(
                issue_ref="issue-a",
                family="diagnosis",
                issue_kind="covered",
                route_state="deterministic_covered",
            ),
            IssueRoute(
                issue_ref="issue-c",
                family="diagnosis",
                issue_kind="extra",
                route_state="llm_residual",
            ),
        ],
    )

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=[],
        dropped=[],
    )

    assert report["summary"]["accounting_balanced"] is False
    assert report["summary"]["missing_route_issue_refs"] == []
    assert report["summary"]["extra_route_issue_refs"] == ["issue-c"]
    assert report["summary"]["duplicate_route_issue_refs"] == []
    assert report["summary"]["issue_accounting_rate"] < 1.0


def test_funnel_report_flags_duplicate_route_refs() -> None:
    scan = DeterministicScanResult(
        issues=[
            {"issue_ref": "issue-a", "issue_family": "diagnosis"},
            {"issue_ref": "issue-b", "issue_family": "diagnosis"},
        ],
        candidates=[],
        rejections=[],
        issue_routes=[
            IssueRoute(
                issue_ref="issue-a",
                family="diagnosis",
                issue_kind="covered",
                route_state="deterministic_covered",
            ),
            IssueRoute(
                issue_ref="issue-a",
                family="diagnosis",
                issue_kind="duplicate",
                route_state="llm_residual",
            ),
        ],
    )

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=[],
        dropped=[],
    )

    assert report["summary"]["accounting_balanced"] is False
    assert report["summary"]["missing_route_issue_refs"] == ["issue-b"]
    assert report["summary"]["extra_route_issue_refs"] == []
    assert report["summary"]["duplicate_route_issue_refs"] == ["issue-a"]
    assert report["summary"]["issue_accounting_rate"] < 1.0


def test_funnel_report_counts_routes_and_reasons_per_family() -> None:
    scan = _scan_with_routes()

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=["proposal-a"],
        dropped=[],
    )

    assert report["summary"]["route_state_counts"] == {
        "deterministic_covered": 2,
        "llm_residual": 1,
        "blocked_safety": 1,
        "blocked_apply": 1,
        "blocked_evidence": 1,
        "deferred_budget": 1,
        "duplicate": 1,
    }
    assert report["summary"]["reason_code_counts"] == {
        "DETERMINISTIC_CANDIDATE_VALID": 2,
        "NO_VALID_DETERMINISTIC_CANDIDATE": 1,
        "UNSAFE_DOMAIN_RANGE": 1,
        "APPLY_UNSUPPORTED": 1,
        "MISSING_EVIDENCE": 1,
        "FAMILY_CAP_REACHED": 1,
        "DUPLICATE_ACTION_FINGERPRINT": 1,
    }

    diagnosis = report["families"]["diagnosis"]
    assert diagnosis["raw_issue_count"] == 4
    assert diagnosis["issue_with_candidate_count"] == 3
    assert diagnosis["action_candidate_count"] == 4
    assert diagnosis["deterministic_covered_count"] == 2
    assert diagnosis["llm_residual_count"] == 1
    assert diagnosis["blocked_safety_count"] == 1
    assert diagnosis["reason_code_counts"] == {
        "DETERMINISTIC_CANDIDATE_VALID": 2,
        "NO_VALID_DETERMINISTIC_CANDIDATE": 1,
        "UNSAFE_DOMAIN_RANGE": 1,
    }


def test_funnel_report_dedupes_action_candidate_counts() -> None:
    scan = DeterministicScanResult(
        issues=[
            {"issue_ref": "issue-a", "issue_family": "diagnosis"},
            {"issue_ref": "issue-b", "issue_family": "diagnosis"},
            {"issue_ref": "issue-c", "issue_family": "treatment"},
        ],
        candidates=[
            {"candidate_id": "candidate-a"},
            {"candidate_id": "candidate-a"},
            {"candidate_id": "candidate-b"},
            {"candidate_id": ""},
        ],
        rejections=[],
        issue_routes=[
            IssueRoute(
                issue_ref="issue-a",
                family="diagnosis",
                issue_kind="covered",
                route_state="deterministic_covered",
                candidate_ids=["candidate-a"],
            ),
            IssueRoute(
                issue_ref="issue-b",
                family="diagnosis",
                issue_kind="covered",
                route_state="deterministic_covered",
                candidate_ids=["candidate-a", "candidate-b"],
            ),
            IssueRoute(
                issue_ref="issue-c",
                family="treatment",
                issue_kind="covered",
                route_state="deterministic_covered",
                candidate_ids=["candidate-b", "candidate-b"],
            ),
        ],
    )

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=[],
        dropped=[],
    )

    assert report["summary"]["action_candidate_count"] == 2
    assert report["families"]["diagnosis"]["action_candidate_count"] == 2
    assert report["families"]["treatment"]["action_candidate_count"] == 1


def test_funnel_report_counts_selected_proposals_from_issue_routes() -> None:
    scan = _scan_with_routes()

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=["proposal-a", "proposal-outside-route"],
        dropped=[],
    )

    assert report["summary"]["selected_proposal_count"] == 1
    assert report["families"]["diagnosis"]["selected_proposal_count"] == 1


def test_funnel_report_dedupes_selected_proposals_per_family() -> None:
    scan = DeterministicScanResult(
        issues=[
            {"issue_ref": "issue-a", "issue_family": "diagnosis"},
            {"issue_ref": "issue-b", "issue_family": "diagnosis"},
        ],
        candidates=[],
        rejections=[],
        issue_routes=[
            IssueRoute(
                issue_ref="issue-a",
                family="diagnosis",
                issue_kind="covered",
                route_state="deterministic_covered",
                proposal_ids=["proposal-a"],
            ),
            IssueRoute(
                issue_ref="issue-b",
                family="diagnosis",
                issue_kind="covered",
                route_state="deterministic_covered",
                proposal_ids=["proposal-a"],
            ),
        ],
    )

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=["proposal-a"],
        dropped=[],
    )

    assert report["summary"]["selected_proposal_count"] == 1
    assert report["families"]["diagnosis"]["selected_proposal_count"] == 1


def test_funnel_report_tracks_unknown_route_state_without_breaking_markdown(
    tmp_path: Path,
) -> None:
    scan = DeterministicScanResult(
        issues=[{"issue_ref": "issue-a", "issue_family": "diagnosis"}],
        candidates=[],
        rejections=[],
        issue_routes=[
            IssueRoute(
                issue_ref="issue-a",
                family="diagnosis",
                issue_kind="covered",
                route_state="needs_review",
            ),
        ],
    )

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=[],
        dropped=[],
    )
    artifact_paths = write_proposal_funnel_artifacts(
        tmp_path,
        scan,
        selected_proposal_ids=[],
        dropped=[],
    )
    markdown = artifact_paths["deterministic_proposal_report_md"].read_text(
        encoding="utf-8"
    )

    assert report["summary"]["route_state_counts"]["needs_review"] == 1
    assert report["families"]["diagnosis"]["unknown_route_state_count"] == 1
    assert "diagnosis" in markdown


def test_funnel_report_includes_dropped_proposal_count() -> None:
    scan = _scan_with_routes()

    report = build_proposal_funnel_report(
        scan,
        selected_proposal_ids=[],
        dropped=[
            {"proposal_id": "proposal-x", "reason": "max_proposals"},
            {"proposal_id": "proposal-y", "reason": "duplicate_action_payload"},
        ],
    )

    assert report["summary"]["dropped_proposal_count"] == 2


def test_funnel_artifacts_write_json_ledger_report_and_markdown(tmp_path: Path) -> None:
    scan = _scan_with_routes()

    artifact_paths = write_proposal_funnel_artifacts(
        tmp_path,
        scan,
        selected_proposal_ids=["proposal-a"],
        dropped=[{"proposal_id": "proposal-x", "reason": "max_proposals"}],
    )

    assert {
        "issue_ledger",
        "proposal_issue_ledger",
        "deterministic_proposal_report",
        "deterministic_proposal_report_md",
    }.issubset(artifact_paths)
    assert {
        "proposal_funnel_report",
        "proposal_funnel_report_md",
        "proposal_conflict_groups",
        "proposal_task_packs",
        "proposal_merge_report",
        "subagent_output_index",
    }.issubset(artifact_paths)
    ledger = json.loads(artifact_paths["issue_ledger"].read_text(encoding="utf-8"))
    alias_ledger = json.loads(
        artifact_paths["proposal_issue_ledger"].read_text(encoding="utf-8")
    )
    report = json.loads(
        artifact_paths["deterministic_proposal_report"].read_text(encoding="utf-8")
    )
    markdown = artifact_paths["deterministic_proposal_report_md"].read_text(
        encoding="utf-8"
    )
    funnel_report = json.loads(
        artifact_paths["proposal_funnel_report"].read_text(encoding="utf-8")
    )

    assert ledger["issue_routes"][0]["issue_ref"] == "issue-1"
    assert alias_ledger == ledger
    assert funnel_report == report
    assert ledger["issue_routes"][0]["generation_disposition"] == (
        "deterministic_covered"
    )
    assert ledger["issue_routes"][0]["queue_disposition"] == ""
    assert ledger["issue_routes"][0]["reason_codes"] == [
        "DETERMINISTIC_CANDIDATE_VALID"
    ]
    assert ledger["issue_routes"][0]["events"] == []
    assert report["summary"]["accounting_balanced"] is True
    assert "Raw issues: 8" in markdown
    assert "Selected proposals: 1" in markdown
    assert "Dropped proposals: 1" in markdown
    assert "diagnosis" in markdown


def _scan_with_routes() -> DeterministicScanResult:
    routes = [
        IssueRoute(
            issue_ref="issue-1",
            family="diagnosis",
            issue_kind="reverse_diagnostic_test",
            route_state="deterministic_covered",
            candidate_ids=["candidate-1", "candidate-2"],
            proposal_ids=["proposal-a"],
            reason_code="DETERMINISTIC_CANDIDATE_VALID",
        ),
        IssueRoute(
            issue_ref="issue-2",
            family="diagnosis",
            issue_kind="reverse_diagnostic_test",
            route_state="deterministic_covered",
            candidate_ids=["candidate-3"],
            proposal_ids=["proposal-b"],
            reason_code="DETERMINISTIC_CANDIDATE_VALID",
        ),
        IssueRoute(
            issue_ref="issue-3",
            family="diagnosis",
            issue_kind="ambiguous_test_result",
            route_state="llm_residual",
            reason_code="NO_VALID_DETERMINISTIC_CANDIDATE",
        ),
        IssueRoute(
            issue_ref="issue-4",
            family="diagnosis",
            issue_kind="unsafe_relation",
            route_state="blocked_safety",
            candidate_ids=["candidate-4"],
            reason_code="UNSAFE_DOMAIN_RANGE",
        ),
        IssueRoute(
            issue_ref="issue-5",
            family="entity_cleanup",
            issue_kind="near_duplicate_entity",
            route_state="blocked_apply",
            candidate_ids=["candidate-5"],
            reason_code="APPLY_UNSUPPORTED",
        ),
        IssueRoute(
            issue_ref="issue-6",
            family="treatment",
            issue_kind="missing_evidence",
            route_state="blocked_evidence",
            reason_code="MISSING_EVIDENCE",
        ),
        IssueRoute(
            issue_ref="issue-7",
            family="treatment",
            issue_kind="budgeted_issue",
            route_state="deferred_budget",
            candidate_ids=["candidate-6"],
            reason_code="FAMILY_CAP_REACHED",
        ),
        IssueRoute(
            issue_ref="issue-8",
            family="risk_safety",
            issue_kind="duplicate_issue",
            route_state="duplicate",
            reason_code="DUPLICATE_ACTION_FINGERPRINT",
        ),
    ]
    issues = [
        {
            "issue_ref": route.issue_ref,
            "issue_family": route.family,
            "issue_kind": route.issue_kind,
        }
        for route in routes
    ]
    return DeterministicScanResult(
        issues=issues,
        candidates=[
            {"candidate_id": f"candidate-{index}"}
            for index in range(1, 7)
        ],
        rejections=[],
        issue_routes=routes,
    )
