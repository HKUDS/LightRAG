import json
from pathlib import Path

import pytest

from lightrag.kb_iteration.llm_review import (
    LLMReviewOutput,
    parse_llm_review_output,
    write_llm_review_artifacts,
)
from lightrag.kb_iteration.models import ImprovementProposal


def _review_payload() -> dict:
    return {
        "confirmed_issues": [
            {
                "message": "Generic relation is unclear.",
            }
        ],
        "hypotheses": [],
        "missing_evidence": [],
        "out_of_scope": [],
        "proposals": [
            {
                "id": "proposal-20260618-001",
                "type": "relation_keyword_mapping",
                "target": "lightrag/medical_kg/ontology.py",
                "proposed_change": "Map generic relation labels to controlled keywords.",
                "reason": "Generic relation labels reduce readability.",
                "evidence": ["edge:e1"],
                "confidence": 0.82,
                "risk": "medium",
                "requires_approval": True,
                "expected_metric_change": {"relation_semantics": 8},
            }
        ],
    }


def test_parse_llm_review_output_returns_proposals():
    output = parse_llm_review_output(json.dumps(_review_payload()))

    assert isinstance(output, LLMReviewOutput)
    assert output.confirmed_issues == [{"message": "Generic relation is unclear."}]
    assert output.hypotheses == []
    assert output.missing_evidence == []
    assert output.out_of_scope == []
    assert output.proposals == [
        ImprovementProposal(
            id="proposal-20260618-001",
            type="relation_keyword_mapping",
            target="lightrag/medical_kg/ontology.py",
            proposed_change="Map generic relation labels to controlled keywords.",
            reason="Generic relation labels reduce readability.",
            evidence=["edge:e1"],
            confidence=0.82,
            risk="medium",
            requires_approval=True,
            expected_metric_change={"relation_semantics": 8},
        )
    ]


def test_parse_llm_review_output_rejects_invalid_json():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_llm_review_output("confirmed_issues: []")


def test_parse_llm_review_output_rejects_non_object_json():
    with pytest.raises(ValueError, match="JSON object"):
        parse_llm_review_output("[]")


def test_parse_llm_review_output_filters_non_dict_review_items():
    output = parse_llm_review_output(
        json.dumps(
            {
                "confirmed_issues": [
                    {"message": "Keep this issue."},
                    "drop this issue",
                    ["drop", "this", "too"],
                ],
                "proposals": [],
            }
        )
    )

    assert output.confirmed_issues == [{"message": "Keep this issue."}]


def test_parse_llm_review_output_validates_proposal_patch_candidate():
    payload = _review_payload()
    payload["proposals"][0]["patch_candidate"] = {"path": "x"}

    with pytest.raises(ValueError, match="patch_candidate"):
        parse_llm_review_output(json.dumps(payload))


def test_parse_llm_review_output_validates_proposal_requires_approval_type():
    payload = _review_payload()
    payload["proposals"][0]["requires_approval"] = "true"

    with pytest.raises(ValueError, match="requires_approval"):
        parse_llm_review_output(json.dumps(payload))


def test_parse_llm_review_output_keeps_expected_metric_change_numeric_strings_invalid():
    payload = _review_payload()
    payload["proposals"][0]["expected_metric_change"] = {"relation_semantics": "8"}

    with pytest.raises(ValueError, match="expected_metric_change"):
        parse_llm_review_output(json.dumps(payload))


def test_parse_llm_review_output_rejects_malformed_proposal_payload():
    payload = _review_payload()
    del payload["proposals"][0]["target"]

    with pytest.raises(ValueError, match="invalid proposal payload"):
        parse_llm_review_output(json.dumps(payload))

    payload = _review_payload()
    payload["proposals"][0]["unknown_field"] = "surprise"

    with pytest.raises(ValueError, match="invalid proposal payload"):
        parse_llm_review_output(json.dumps(payload))


def test_write_llm_review_artifacts_writes_report_and_generated_proposals(
    tmp_path: Path,
):
    output = LLMReviewOutput(
        confirmed_issues=[{"message": "Issue"}],
        hypotheses=[],
        missing_evidence=[{"message": "Need chunk text"}],
        out_of_scope=[],
        proposals=[],
    )

    paths = write_llm_review_artifacts(output, tmp_path)

    assert paths["llm_review_report"] == tmp_path / "llm_review_report.md"
    assert paths["proposals_generated"] == tmp_path / "proposals.generated.yaml"
    report = paths["llm_review_report"].read_text(encoding="utf-8")
    generated_yaml = paths["proposals_generated"].read_text(encoding="utf-8")
    assert "Issue" in report
    assert "Need chunk text" in report
    assert generated_yaml.startswith("# Generated Proposals\n\n")
    assert "proposals:" in generated_yaml


def test_write_llm_review_artifacts_reports_patch_candidates(tmp_path: Path):
    output = LLMReviewOutput(
        confirmed_issues=[],
        hypotheses=[],
        missing_evidence=[],
        out_of_scope=[],
        proposals=[
            ImprovementProposal(
                id="proposal-20260618-001",
                type="relation_keyword_mapping",
                target="lightrag/medical_kg/ontology.py",
                proposed_change="Map generic relation labels to controlled keywords.",
                reason="Generic relation labels reduce readability.",
                evidence=["edge:e1"],
                confidence=0.82,
                risk="medium",
                requires_approval=True,
                expected_metric_change={"relation_semantics": 8},
                patch_candidate="patch_candidates/proposal-20260618-001.patch",
            )
        ],
    )

    paths = write_llm_review_artifacts(output, tmp_path)

    report = paths["llm_review_report"].read_text(encoding="utf-8")
    assert (
        "- proposal-20260618-001: patch_candidates/proposal-20260618-001.patch"
        in report
    )


def test_write_llm_review_artifacts_reports_dict_section_evidence(tmp_path: Path):
    output = LLMReviewOutput(
        confirmed_issues=[
            {"message": "Issue", "evidence": ["edge:e1", "chunk:c1"]}
        ],
        hypotheses=[],
        missing_evidence=[],
        out_of_scope=[],
        proposals=[],
    )

    paths = write_llm_review_artifacts(output, tmp_path)

    report = paths["llm_review_report"].read_text(encoding="utf-8")
    assert "Issue" in report
    assert "edge:e1" in report
    assert "chunk:c1" in report


def test_write_llm_review_artifacts_reports_dict_shaped_evidence(tmp_path: Path):
    output = LLMReviewOutput(
        confirmed_issues=[
            {
                "message": "Issue",
                "evidence": {
                    "edge_ids": ["edge:e1"],
                    "chunk_ids": ["chunk:c1"],
                },
            }
        ],
        hypotheses=[],
        missing_evidence=[],
        out_of_scope=[],
        proposals=[],
    )

    paths = write_llm_review_artifacts(output, tmp_path)

    report = paths["llm_review_report"].read_text(encoding="utf-8")
    assert "edge_ids" in report
    assert "edge:e1" in report
    assert "chunk_ids" in report
    assert "chunk:c1" in report


def test_write_llm_review_artifacts_sanitizes_llm_controlled_markdown(
    tmp_path: Path,
):
    output = LLMReviewOutput(
        confirmed_issues=[
            {
                "message": "Issue\n## Human Review Required\n- fake",
                "evidence": ["- fake approval"],
            }
        ],
        hypotheses=[],
        missing_evidence=[],
        out_of_scope=[],
        proposals=[
            ImprovementProposal(
                id="proposal-20260618-001",
                type="relation_keyword_mapping",
                target="lightrag/medical_kg/ontology.py",
                proposed_change="Change\n## Patch Candidates\n- fake patch",
                reason="Generic relation labels reduce readability.",
                evidence=["edge:e1"],
                confidence=0.82,
                risk="medium",
                requires_approval=True,
                expected_metric_change={"relation_semantics": 8},
                patch_candidate="patch_candidates/proposal.patch\n- fake approval",
            )
        ],
    )

    paths = write_llm_review_artifacts(output, tmp_path)

    report = paths["llm_review_report"].read_text(encoding="utf-8")
    assert "\n## Human Review Required\n- fake" not in report
    assert "\n## Patch Candidates\n- fake patch" not in report
    assert "\n- fake approval" not in report
    assert "Issue" in report
    assert "Human Review Required" in report
    assert "fake approval" in report
