from pathlib import Path

import pytest

from lightrag.kb_iteration.models import ImprovementProposal
from lightrag.kb_iteration.proposals import (
    validate_proposal,
    write_approval_queue,
    write_improvement_backlog,
)


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
    assert "- id: proposal-20260617-011" in first
    assert "evidence: []" in first
    assert "confidence: 0.6" in first
    assert "risk: low" in first
    assert "requires_approval: false" in first
    assert "expected_metric_change: {}" in first
    assert "evidence:" in first
    assert "  - first evidence" in first
    assert "expected_metric_change:" in first
    assert "  hierarchy_completeness: 5" in first
    assert first.index("hierarchy_completeness") < first.index("overall")
