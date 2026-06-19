from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.kb_iteration.apply import (
    APPLY_SOURCE,
    APPLY_RESULT_JSON,
    APPLY_RESULT_MARKDOWN,
    ApplyChangeStatus,
    AcceptedApplyResult,
    ApplyChange,
    branch_key_from_proposal,
    category_for_branch_key,
    load_proposals_by_id,
    render_apply_result_markdown,
    write_apply_result_artifacts,
)
from lightrag.kb_iteration.models import ImprovementProposal
from lightrag.kb_iteration.proposals import (
    validate_proposal,
    write_approval_queue,
    write_improvement_backlog,
)


def test_add_hierarchy_branch_is_a_known_mutation_type() -> None:
    proposal = ImprovementProposal(
        id="prop-add-branch-diagnosis",
        type="add_hierarchy_branch",
        target="hierarchy",
        proposed_change="Create branch diagnosis_testing.",
        reason="Missing required branch diagnosis_testing.",
        evidence=["item_id: influenza; source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=True,
        expected_metric_change={"hierarchy_missing_branch_count": -1},
    )

    validate_proposal(proposal)


def test_add_hierarchy_branch_without_approval_requires_known_mutation_type() -> None:
    proposal = ImprovementProposal(
        id="prop-add-branch-diagnosis",
        type="add_hierarchy_branch",
        target="hierarchy",
        proposed_change="Create branch diagnosis_testing.",
        reason="Missing required branch diagnosis_testing.",
        evidence=["item_id: influenza; source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=False,
        expected_metric_change={"hierarchy_missing_branch_count": -1},
    )

    with pytest.raises(
        ValueError, match="^proposal type add_hierarchy_branch requires approval$"
    ):
        validate_proposal(proposal)


def test_load_proposals_by_id_reads_queue_and_backlog_yaml_sections(
    tmp_path: Path,
) -> None:
    queued = ImprovementProposal(
        id="proposal-queue-diagnosis",
        type="add_hierarchy_branch",
        target="hierarchy",
        proposed_change="Create branch diagnosis_testing.",
        reason="Missing required branch diagnosis_testing.",
        evidence=["item_id: influenza; source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=True,
        expected_metric_change={"hierarchy_missing_branch_count": -1},
    )
    backlog = ImprovementProposal(
        id="proposal-backlog-note",
        type="quality_report_note",
        target="quality_report.md",
        proposed_change="Record a quality observation.",
        reason="Reviewer context should be retained.",
        evidence=[],
        confidence=0.4,
        risk="low",
        requires_approval=False,
        expected_metric_change={},
    )
    write_approval_queue([queued, backlog], tmp_path)
    write_improvement_backlog([queued, backlog], tmp_path)

    proposals = load_proposals_by_id(tmp_path)

    assert set(proposals) == {"proposal-queue-diagnosis", "proposal-backlog-note"}
    assert isinstance(proposals["proposal-queue-diagnosis"], dict)
    assert proposals["proposal-queue-diagnosis"]["proposed_change"] == (
        "Create branch diagnosis_testing."
    )
    assert proposals["proposal-backlog-note"]["target"] == "quality_report.md"


def test_load_proposals_by_id_preserves_unknown_future_keys(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - id: proposal-future-key
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record future metadata.
    reason: Future queue records may include extra fields.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
    future_reviewer: retained
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert proposals["proposal-future-key"]["future_reviewer"] == "retained"


def test_load_proposals_by_id_skips_invalid_malformed_optional_field(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - id: proposal-invalid-evidence
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record invalid evidence shape.
    reason: Enough fields are present for validation.
    evidence: not-a-list
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
  - id: proposal-valid-note
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record valid evidence shape.
    reason: Valid proposals should still load.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert set(proposals) == {"proposal-valid-note"}


def test_load_proposals_by_id_skips_incomplete_records_without_id(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - type: quality_report_note
    target: quality_report.md
    proposed_change: Record missing id.
    reason: Missing id records are incomplete.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
  - id: proposal-with-id
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record present id.
    reason: Complete records should still load.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert set(proposals) == {"proposal-with-id"}


def test_load_proposals_by_id_skips_records_missing_required_fields(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - id: proposal-incomplete
    type: quality_report_note
  - id: proposal-complete
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record complete proposal.
    reason: Complete records should load.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert set(proposals) == {"proposal-complete"}


def test_load_proposals_by_id_skips_records_missing_risk(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - id: proposal-missing-risk
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record missing risk.
    reason: Records missing renderer fields are incomplete.
    evidence: []
    confidence: 0.4
    requires_approval: false
    expected_metric_change: {}
  - id: proposal-complete-risk
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record complete risk.
    reason: Complete records should still load.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert set(proposals) == {"proposal-complete-risk"}


def test_load_proposals_by_id_keeps_queue_payload_for_duplicate_ids(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - id: proposal-duplicate
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record queue version.
    reason: Queue should win.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )
    (tmp_path / "improvement_backlog.md").write_text(
        """# Improvement Backlog

proposals:
  - id: proposal-duplicate
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record backlog version.
    reason: Backlog should only fill missing ids.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert proposals["proposal-duplicate"]["proposed_change"] == "Record queue version."


def test_load_proposals_by_id_finds_proposals_after_introductory_markdown(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

Review these proposals before applying accepted changes.

proposals:
  - id: proposal-after-intro
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record proposal after intro.
    reason: Markdown prose may precede YAML.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert set(proposals) == {"proposal-after-intro"}


def test_load_proposals_by_id_returns_empty_for_malformed_yaml(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - id: proposal-bad-yaml
    type: quality_report_note
    target: quality_report.md
    proposed_change: [unterminated
""",
        encoding="utf-8",
    )

    assert load_proposals_by_id(tmp_path) == {}


def test_branch_key_from_proposal_extracts_known_ontology_key() -> None:
    proposal = {
        "id": "proposal-diagnosis-branch",
        "type": "add_hierarchy_branch",
        "target": "hierarchy",
        "proposed_change": "Create branch diagnosis_testing.",
        "reason": "Missing required branch.",
    }

    assert branch_key_from_proposal(proposal) == "diagnosis_testing"


@pytest.mark.parametrize(
    "branch_reference",
    ["diagnosis_testing.child", "diagnosis_testing-v2"],
)
def test_branch_key_from_proposal_rejects_identifier_suffixes(
    branch_reference: str,
) -> None:
    proposal = {
        "id": "proposal-prefixed-branch",
        "type": "add_hierarchy_branch",
        "target": "hierarchy",
        "proposed_change": f"Create branch {branch_reference}.",
        "reason": "This is not the standalone branch key.",
    }

    assert branch_key_from_proposal(proposal) == ""


def test_branch_key_from_proposal_returns_empty_string_for_unknown_branch() -> None:
    proposal = {
        "id": "proposal-unknown-branch",
        "type": "add_hierarchy_branch",
        "target": "hierarchy",
        "proposed_change": "Create branch administrative_notes.",
        "reason": "Missing required branch.",
    }

    assert branch_key_from_proposal(proposal) == ""


def test_apply_change_status_blocked_value_is_stable() -> None:
    assert ApplyChangeStatus.BLOCKED.value == "blocked"


def test_category_for_branch_key_returns_only_top_level_category() -> None:
    assert category_for_branch_key("diagnosis_testing").key == "diagnosis_testing"
    assert category_for_branch_key("diagnosis_testing.child") is None


def test_apply_change_to_dict_matches_spec_shape() -> None:
    change = ApplyChange(
        proposal_id="proposal-diagnosis-branch",
        proposal_type="add_hierarchy_branch",
        target="hierarchy",
        status=ApplyChangeStatus.BLOCKED,
        action="add_hierarchy_branch",
        branch_key="diagnosis_testing",
        branch_label="Diagnosis Testing",
        evidence=["item_id: influenza; source_id: chunk-1"],
        reason="Requires graph mutation and Task 2 is artifact-only.",
    )

    assert change.to_dict() == {
        "proposal_id": "proposal-diagnosis-branch",
        "proposal_type": "add_hierarchy_branch",
        "target": "hierarchy",
        "status": "blocked",
        "action": "add_hierarchy_branch",
        "branch_key": "diagnosis_testing",
        "branch_label": "Diagnosis Testing",
        "evidence": ["item_id: influenza; source_id: chunk-1"],
        "reason": "Requires graph mutation and Task 2 is artifact-only.",
    }


def test_apply_result_to_dict_matches_spec_shape_and_computed_counts() -> None:
    result = AcceptedApplyResult(
        workspace="influenza_medical_v1",
        applied_at="2026-06-19T10:00:00+08:00",
        source_artifact=APPLY_SOURCE,
        proposal_ids=["proposal-applied", "proposal-blocked"],
        quality_before={"metrics": {"hierarchy_missing_branch_count": 2}},
        quality_after={"metrics": {"hierarchy_missing_branch_count": 1}},
        changes=[
            ApplyChange(
                proposal_id="proposal-applied",
                proposal_type="quality_report_note",
                target="quality_report.md",
                status=ApplyChangeStatus.APPLIED,
                action="record_note",
            ),
            ApplyChange(
                proposal_id="proposal-blocked",
                proposal_type="add_hierarchy_branch",
                target="hierarchy",
                status=ApplyChangeStatus.BLOCKED,
                action="add_hierarchy_branch",
                branch_key="diagnosis_testing",
            ),
        ],
    )

    payload = result.to_dict()

    assert result.applied_count == 1
    assert result.blocked_count == 1
    assert payload["workspace"] == "influenza_medical_v1"
    assert payload["applied_at"] == "2026-06-19T10:00:00+08:00"
    assert payload["source_artifact"] == "kb_iteration_apply"
    assert payload["proposal_ids"] == ["proposal-applied", "proposal-blocked"]
    assert payload["applied_count"] == 1
    assert payload["blocked_count"] == 1
    assert payload["quality_before"] == {
        "metrics": {"hierarchy_missing_branch_count": 2}
    }
    assert payload["quality_after"] == {
        "metrics": {"hierarchy_missing_branch_count": 1}
    }
    assert payload["changes"][0]["status"] == "applied"
    assert "quality_delta" not in payload


def test_apply_result_artifacts_use_spec_filenames_and_metric_transition(
    tmp_path: Path,
) -> None:
    result = AcceptedApplyResult(
        workspace="influenza_medical_v1",
        applied_at="2026-06-19T10:00:00+08:00",
        source_artifact=APPLY_SOURCE,
        proposal_ids=["proposal-diagnosis-branch"],
        quality_before={"metrics": {"hierarchy_missing_branch_count": 2}},
        quality_after={"metrics": {"hierarchy_missing_branch_count": 1}},
        changes=[
            ApplyChange(
                proposal_id="proposal-diagnosis-branch",
                proposal_type="add_hierarchy_branch",
                target="hierarchy",
                status=ApplyChangeStatus.BLOCKED,
                action="add_hierarchy_branch",
                branch_key="diagnosis_testing",
                branch_label="Diagnosis Testing",
                reason="Graph mutation is out of scope for Task 2.",
            )
        ],
    )

    json_path, markdown_path = write_apply_result_artifacts(result, tmp_path)
    markdown = render_apply_result_markdown(result)

    assert APPLY_SOURCE == "kb_iteration_apply"
    assert APPLY_RESULT_JSON == "accepted_changes_apply_result.json"
    assert APPLY_RESULT_MARKDOWN == "accepted_changes_apply_result.md"
    assert json_path == tmp_path / APPLY_RESULT_JSON
    assert markdown_path == tmp_path / APPLY_RESULT_MARKDOWN
    assert not (tmp_path / "accepted_changes_execution.json").exists()
    assert not (tmp_path / "accepted_changes_execution.md").exists()
    assert "hierarchy_missing_branch_count: 2 -> 1" in markdown
    assert "## Changes" in markdown
    assert "proposal-diagnosis-branch: blocked (diagnosis_testing)" in markdown
