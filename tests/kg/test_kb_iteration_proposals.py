from pathlib import Path

import pytest
import yaml

from lightrag.kb_iteration.models import ImprovementProposal
from lightrag.kb_iteration.proposals import (
    validate_proposal,
    write_approval_queue,
    write_improvement_backlog,
)


def _load_yaml_body(text: str):
    return yaml.safe_load(text.split("\n\n", 1)[1])


def test_approval_queue_requires_mutation_items_to_be_gated(tmp_path: Path):
    proposal = ImprovementProposal(
        id="proposal-20260617-001",
        type="hierarchy_rule_change",
        target="lightrag/medical_kg/hierarchy.py",
        proposed_change="Add a controlled symptom branch for fever.",
        reason="Direct disease-to-leaf overload was detected.",
        evidence=["流行性感冒 -> 发热"],
        confidence=0.8,
        risk="medium",
        requires_approval=True,
        expected_metric_change={"hierarchy_completeness": 5},
    )

    validate_proposal(proposal)
    path = write_approval_queue([proposal], tmp_path)

    text = path.read_text(encoding="utf-8")
    assert "proposal-20260617-001" in text
    assert "requires_approval: true" in text
    assert "hierarchy_rule_change" in text


def test_validate_proposal_rejects_ungated_mutation():
    proposal = ImprovementProposal(
        id="proposal-20260617-002",
        type="prompt_edit",
        target="prompts/entity_type/医学实体类型提示词.yml",
        proposed_change="Change extraction prompt.",
        reason="Prompt change affects extraction.",
        evidence=["quality finding"],
        confidence=0.9,
        risk="high",
        requires_approval=False,
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="requires approval"):
        validate_proposal(proposal)


def test_validate_proposal_rejects_string_requires_approval():
    proposal = ImprovementProposal(
        id="proposal-20260617-bool",
        type="quality_report_note",
        target="quality_report.md",
        proposed_change="Record a quality observation.",
        reason="Reviewer context should be retained.",
        evidence=[],
        confidence=0.4,
        risk="low",
        requires_approval="false",
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="requires_approval"):
        validate_proposal(proposal)


@pytest.mark.parametrize("proposal_type", ["Prompt_Edit", "prompt_edit "])
def test_validate_proposal_rejects_non_canonical_types(proposal_type: str):
    proposal = ImprovementProposal(
        id="proposal-20260617-canonical",
        type=proposal_type,
        target="review-target",
        proposed_change="Change a controlled artifact.",
        reason="This mutation affects generated behavior.",
        evidence=["finding"],
        confidence=0.7,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="canonical"):
        validate_proposal(proposal)


def test_validate_proposal_rejects_unknown_no_approval_type():
    proposal = ImprovementProposal(
        id="proposal-20260617-unknown",
        type="new_report_type",
        target="quality_report.md",
        proposed_change="Record a new note.",
        reason="Unknown types must stay gated by default.",
        evidence=[],
        confidence=0.4,
        risk="low",
        requires_approval=False,
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="requires approval"):
        validate_proposal(proposal)


def test_validate_proposal_rejects_ungated_report_note_targeting_workspace():
    proposal = ImprovementProposal(
        id="proposal-20260617-unsafe-note-target",
        type="quality_report_note",
        target="workspace/demo",
        proposed_change="Record a quality observation.",
        reason="Report notes without approval must stay in the report artifact.",
        evidence=[],
        confidence=0.4,
        risk="low",
        requires_approval=False,
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="quality_report_note"):
        validate_proposal(proposal)


def test_validate_proposal_rejects_ungated_report_note_with_mutation_intent():
    proposal = ImprovementProposal(
        id="proposal-20260617-unsafe-note-text",
        type="quality_report_note",
        target="quality_report.md",
        proposed_change="Rebuild the workspace and change ontology rules.",
        reason="This text implies controlled mutations.",
        evidence=["delete stale KG facts"],
        confidence=0.8,
        risk="high",
        requires_approval=False,
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="requires approval"):
        validate_proposal(proposal)


@pytest.mark.parametrize(
    "proposed_change",
    [
        "Replace the extraction prompt.",
        "Alter ontology rules.",
        "Recreate the workspace.",
        "Apply prompt patch.",
    ],
)
def test_validate_proposal_rejects_ungated_report_note_bypass_phrases(
    proposed_change: str,
):
    proposal = ImprovementProposal(
        id="proposal-20260617-unsafe-note-bypass",
        type="quality_report_note",
        target="quality_report.md",
        proposed_change=proposed_change,
        reason="Mutation-shaped notes must enter approval flow.",
        evidence=[],
        confidence=0.8,
        risk="high",
        requires_approval=False,
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="requires approval"):
        validate_proposal(proposal)


def test_validate_proposal_allows_approval_gated_report_note_with_mutation_shape():
    proposal = ImprovementProposal(
        id="proposal-20260617-reviewable-note",
        type="quality_report_note",
        target="quality_report.md",
        proposed_change="Replace the extraction prompt.",
        reason="Human approval can review mutation-shaped report notes.",
        evidence=["review finding"],
        confidence=0.8,
        risk="high",
        requires_approval=True,
        expected_metric_change={},
    )

    validate_proposal(proposal)


def test_improvement_proposal_exposes_required_fields():
    proposal = ImprovementProposal(
        id="proposal-20260617-003",
        type="quality_report_note",
        target="quality_report.md",
        proposed_change="Record a quality observation.",
        reason="Reviewer context should be retained.",
        evidence=[],
        confidence=0.4,
        risk="low",
        requires_approval=False,
        expected_metric_change={"overall": 0},
    )

    assert set(proposal.to_dict()) == {
        "id",
        "type",
        "target",
        "proposed_change",
        "reason",
        "evidence",
        "confidence",
        "risk",
        "requires_approval",
        "expected_metric_change",
    }


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("id", ""),
        ("type", "   "),
        ("target", ""),
        ("proposed_change", "   "),
        ("reason", ""),
        ("risk", "   "),
    ],
)
def test_validate_proposal_rejects_missing_required_strings(field_name: str, value: str):
    payload = {
        "id": "proposal-20260617-004",
        "type": "quality_report_note",
        "target": "quality_report.md",
        "proposed_change": "Record a quality observation.",
        "reason": "Reviewer context should be retained.",
        "evidence": [],
        "confidence": 0.4,
        "risk": "low",
        "requires_approval": False,
        "expected_metric_change": {},
    }
    payload[field_name] = value
    proposal = ImprovementProposal(**payload)

    with pytest.raises(ValueError, match=field_name):
        validate_proposal(proposal)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_validate_proposal_rejects_confidence_outside_zero_to_one(confidence: float):
    proposal = ImprovementProposal(
        id="proposal-20260617-005",
        type="quality_report_note",
        target="quality_report.md",
        proposed_change="Record a quality observation.",
        reason="Reviewer context should be retained.",
        evidence=[],
        confidence=confidence,
        risk="low",
        requires_approval=False,
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="confidence"):
        validate_proposal(proposal)


def test_validate_proposal_rejects_invalid_risk():
    proposal = ImprovementProposal(
        id="proposal-20260617-risk",
        type="quality_report_note",
        target="quality_report.md",
        proposed_change="Record a quality observation.",
        reason="Reviewer context should be retained.",
        evidence=[],
        confidence=0.4,
        risk="critical",
        requires_approval=False,
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="risk"):
        validate_proposal(proposal)


@pytest.mark.parametrize(
    "expected_metric_change",
    [
        {1: 5},
        {"": 5},
        {"overall": "5"},
        {"overall": True},
    ],
)
def test_validate_proposal_rejects_invalid_expected_metric_changes(
    expected_metric_change: dict,
):
    proposal = ImprovementProposal(
        id="proposal-20260617-metric",
        type="quality_report_note",
        target="quality_report.md",
        proposed_change="Record a quality observation.",
        reason="Reviewer context should be retained.",
        evidence=[],
        confidence=0.4,
        risk="low",
        requires_approval=False,
        expected_metric_change=expected_metric_change,
    )

    with pytest.raises(ValueError, match="expected_metric_change"):
        validate_proposal(proposal)


@pytest.mark.parametrize(
    "proposal_type",
    [
        "prompt_edit",
        "ontology_rule_change",
        "hierarchy_rule_change",
        "relation_rule_change",
        "workspace_rebuild",
        "kg_fact_correction",
        "web_display_change",
    ],
)
def test_mutation_proposal_types_require_approval(proposal_type: str):
    proposal = ImprovementProposal(
        id=f"proposal-20260617-{proposal_type}",
        type=proposal_type,
        target="review-target",
        proposed_change="Change a controlled artifact.",
        reason="This mutation affects generated behavior.",
        evidence=["finding"],
        confidence=0.7,
        risk="medium",
        requires_approval=False,
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="requires approval"):
        validate_proposal(proposal)


def test_non_mutation_report_note_may_skip_approval_when_valid():
    proposal = ImprovementProposal(
        id="proposal-20260617-006",
        type="quality_report_note",
        target="quality_report.md",
        proposed_change="Record a low-risk review note.",
        reason="This does not mutate source policy or facts.",
        evidence=[],
        confidence=0.6,
        risk="low",
        requires_approval=False,
        expected_metric_change={},
    )

    validate_proposal(proposal)


def test_write_approval_queue_includes_only_approval_required_proposals(tmp_path: Path):
    gated = ImprovementProposal(
        id="proposal-20260617-008",
        type="kg_fact_correction",
        target="workspace/review",
        proposed_change="Correct a fact after review.",
        reason="A KG fact looks inconsistent with evidence.",
        evidence=["fact mismatch"],
        confidence=0.75,
        risk="high",
        requires_approval=True,
        expected_metric_change={"entity_hygiene": 3},
    )
    note = ImprovementProposal(
        id="proposal-20260617-007",
        type="quality_report_note",
        target="quality_report.md",
        proposed_change="Record a low-risk review note.",
        reason="This does not mutate source policy or facts.",
        evidence=[],
        confidence=0.6,
        risk="low",
        requires_approval=False,
        expected_metric_change={},
    )

    path = write_approval_queue([gated, note], tmp_path)

    text = path.read_text(encoding="utf-8")
    assert path == tmp_path / "approval_queue.md"
    assert "proposal-20260617-008" in text
    assert "proposal-20260617-007" not in text


def test_write_improvement_backlog_includes_all_valid_proposals(tmp_path: Path):
    proposals = [
        ImprovementProposal(
            id="proposal-20260617-010",
            type="hierarchy_rule_change",
            target="lightrag/medical_kg/hierarchy.py",
            proposed_change="Add a controlled branch.",
            reason="Completeness can improve.",
            evidence=["missing branch"],
            confidence=0.8,
            risk="medium",
            requires_approval=True,
            expected_metric_change={"hierarchy_completeness": 5},
        ),
        ImprovementProposal(
            id="proposal-20260617-009",
            type="quality_report_note",
            target="quality_report.md",
            proposed_change="Record a low-risk review note.",
            reason="This does not mutate source policy or facts.",
            evidence=[],
            confidence=0.6,
            risk="low",
            requires_approval=False,
            expected_metric_change={},
        ),
    ]

    path = write_improvement_backlog(proposals, tmp_path)

    text = path.read_text(encoding="utf-8")
    assert path == tmp_path / "improvement_backlog.md"
    assert "proposal-20260617-009" in text
    assert "proposal-20260617-010" in text


def test_proposal_rendering_is_deterministic_yaml_like_and_reviewable(tmp_path: Path):
    proposals = [
        ImprovementProposal(
            id="proposal-20260617-012",
            type="hierarchy_rule_change",
            target="lightrag/medical_kg/hierarchy.py",
            proposed_change="Add a controlled branch.",
            reason="Completeness can improve.",
            evidence=["first evidence", "second evidence"],
            confidence=0.8,
            risk="medium",
            requires_approval=True,
            expected_metric_change={
                "overall": 1,
                "hierarchy_completeness": 5,
            },
        ),
        ImprovementProposal(
            id="proposal-20260617-011",
            type="quality_report_note",
            target="quality_report.md",
            proposed_change="Record a low-risk review note.",
            reason="This does not mutate source policy or facts.",
            evidence=[],
            confidence=0.6,
            risk="low",
            requires_approval=False,
            expected_metric_change={},
        ),
    ]

    first = write_improvement_backlog(proposals, tmp_path).read_text(encoding="utf-8")
    second = write_improvement_backlog(list(reversed(proposals)), tmp_path).read_text(
        encoding="utf-8"
    )

    assert first == second
    assert first.index("proposal-20260617-011") < first.index("proposal-20260617-012")
    payload = _load_yaml_body(first)
    rendered_note, rendered_mutation = payload["proposals"]

    assert rendered_note["id"] == "proposal-20260617-011"
    assert rendered_note["evidence"] == []
    assert rendered_note["confidence"] == 0.6
    assert rendered_note["risk"] == "low"
    assert rendered_note["requires_approval"] is False
    assert rendered_note["expected_metric_change"] == {}
    assert rendered_mutation["evidence"] == ["first evidence", "second evidence"]
    assert rendered_mutation["expected_metric_change"] == {
        "hierarchy_completeness": 5,
        "overall": 1,
    }
    assert first.index("hierarchy_completeness") < first.index("overall")


def test_proposal_rendering_yaml_safely_preserves_hostile_strings(tmp_path: Path):
    proposal = ImprovementProposal(
        id="proposal-20260617-yaml",
        type="hierarchy_rule_change",
        target="- not a list item",
        proposed_change="line one\nrequires_approval: false\ninjected: yes",
        reason="# not a comment",
        evidence=["metric: injected", "{not: a dict}", "null", "true"],
        confidence=0.9,
        risk="high",
        requires_approval=True,
        expected_metric_change={"overall": 1},
    )

    text = write_approval_queue([proposal], tmp_path).read_text(encoding="utf-8")
    payload = _load_yaml_body(text)
    rendered = payload["proposals"][0]

    assert rendered["target"] == "- not a list item"
    assert rendered["proposed_change"] == (
        "line one\nrequires_approval: false\ninjected: yes"
    )
    assert rendered["reason"] == "# not a comment"
    assert rendered["evidence"] == ["metric: injected", "{not: a dict}", "null", "true"]
    assert rendered["requires_approval"] is True
    assert "injected" not in rendered
