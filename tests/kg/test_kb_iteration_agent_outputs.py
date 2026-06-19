import json
from pathlib import Path

import pytest

from lightrag.kb_iteration.agent_outputs import (
    AgentStageOutput,
    parse_agent_stage_output,
    stage_output_to_markdown,
    write_agent_stage_artifacts,
)


def _proposal_payload(**overrides):
    payload = {
        "id": "proposal-hierarchy-symptom-001",
        "type": "hierarchy_rule_change",
        "target": "lightrag/medical_kg/hierarchy.py",
        "proposed_change": "Add a controlled symptom branch for fever.",
        "reason": "Direct disease-to-leaf hierarchy edges hide symptom grouping.",
        "evidence": ["edge:flu->fever"],
        "confidence": 0.82,
        "risk": "medium",
        "requires_approval": True,
        "expected_metric_change": {"hierarchy_completeness": 5},
    }
    payload.update(overrides)
    return payload


def test_stage_output_to_markdown_uses_readable_chinese_stage_headings():
    outputs = [
        AgentStageOutput(stage="explain", payload={"issue_explanations": []}),
        AgentStageOutput(stage="infer_branches", payload={"missing_branches": []}),
        AgentStageOutput(stage="locate_evidence", payload={"evidence_map": []}),
        AgentStageOutput(stage="rank_repairs", payload={"repair_plan": []}),
        AgentStageOutput(stage="judge", payload={"judge_results": []}),
    ]
    expected_headings = [
        "# 问题解释",
        "# 缺失分支推断",
        "# 证据定位",
        "# 修复方案排序",
        "# Judge 评判",
    ]

    for output, heading in zip(outputs, expected_headings, strict=True):
        markdown = stage_output_to_markdown(output)

        assert markdown.startswith(f"{heading}\n")


def test_parse_agent_stage_output_rejects_invalid_and_non_object_json():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_agent_stage_output("explain", "not json")

    with pytest.raises(ValueError, match="JSON object"):
        parse_agent_stage_output("explain", "[]")


def test_parse_agent_stage_output_validates_proposals_with_existing_schema():
    output = parse_agent_stage_output(
        "propose",
        json.dumps({"proposals": [_proposal_payload()]}, ensure_ascii=False),
    )

    assert isinstance(output, AgentStageOutput)
    assert output.stage == "propose"
    assert output.payload["proposals"][0]["id"] == "proposal-hierarchy-symptom-001"
    assert [proposal.id for proposal in output.proposals] == [
        "proposal-hierarchy-symptom-001"
    ]


def test_parse_agent_stage_output_normalizes_structured_proposal_evidence():
    output = parse_agent_stage_output(
        "propose",
        json.dumps(
            {
                "proposals": [
                    _proposal_payload(
                        evidence=[
                            {
                                "source_id": "chunk-1",
                                "file_path": "guide.md",
                                "item_id": "edge-flu-fever",
                                "comment": "do not render this free-form field",
                                "reason": "supports hierarchy gap",
                            },
                            "quality:hierarchy_missing_branch_count=1",
                        ]
                    )
                ]
            },
            ensure_ascii=False,
        ),
    )

    evidence = output.proposals[0].evidence
    assert all(isinstance(item, str) for item in evidence)
    assert evidence == [
        "source_id: chunk-1; file_path: guide.md; item_id: edge-flu-fever; reason: supports hierarchy gap",
        "quality:hierarchy_missing_branch_count=1",
    ]


def test_parse_agent_stage_output_rejects_unknown_only_structured_evidence():
    with pytest.raises(ValueError, match="evidence"):
        parse_agent_stage_output(
            "propose",
            json.dumps(
                {"proposals": [_proposal_payload(evidence=[{"comment": "chunk-1"}])]},
                ensure_ascii=False,
            ),
        )


def test_parse_agent_stage_output_rejects_mutation_proposals_with_empty_evidence():
    with pytest.raises(ValueError, match="evidence"):
        parse_agent_stage_output(
            "propose",
            json.dumps(
                {"proposals": [_proposal_payload(evidence=[])]},
                ensure_ascii=False,
            ),
        )

    output = parse_agent_stage_output(
        "propose",
        json.dumps(
            {
                "proposals": [
                    _proposal_payload(
                        id="proposal-context-request-001",
                        type="review_context_request",
                        target="review-context",
                        proposed_change="Request source snippets for review.",
                        reason="The reviewer needs more context before mutation.",
                        evidence=[],
                        expected_metric_change={},
                    )
                ]
            },
            ensure_ascii=False,
        ),
    )
    assert output.proposals[0].type == "review_context_request"
    assert output.proposals[0].evidence == []


@pytest.mark.parametrize("blank_evidence", [[""], ["   "], ["", "   "]])
def test_parse_agent_stage_output_rejects_mutation_proposals_with_blank_evidence(
    blank_evidence,
):
    with pytest.raises(ValueError, match="evidence"):
        parse_agent_stage_output(
            "propose",
            json.dumps(
                {"proposals": [_proposal_payload(evidence=blank_evidence)]},
                ensure_ascii=False,
            ),
        )


def test_parse_agent_stage_output_allows_context_request_with_blank_evidence():
    output = parse_agent_stage_output(
        "propose",
        json.dumps(
            {
                "proposals": [
                    _proposal_payload(
                        id="proposal-context-request-blank-001",
                        type="review_context_request",
                        target="review-context",
                        proposed_change="Request source snippets for review.",
                        reason="The reviewer needs more context before mutation.",
                        evidence=["   "],
                        expected_metric_change={},
                    )
                ]
            },
            ensure_ascii=False,
        ),
    )

    assert output.proposals[0].type == "review_context_request"
    assert output.proposals[0].evidence == ["   "]


def test_stage_output_to_markdown_renders_explain_report():
    markdown = stage_output_to_markdown(
        AgentStageOutput(
            stage="explain",
            payload={
                "issue_explanations": [
                    {
                        "id": "issue-001",
                        "category": "hierarchy",
                        "severity": "high",
                        "explanation": "疾病直接连接到症状叶子节点。",
                        "impact": "层级查询会缺少症状分组。",
                        "evidence_refs": ["edge:flu->fever", "chunk:12"],
                    }
                ]
            },
        )
    )

    assert markdown.startswith("# 问题解释\n")
    assert "疾病直接连接到症状叶子节点。" in markdown
    assert "edge:flu->fever" in markdown
    assert "chunk:12" in markdown
    assert markdown.endswith("\n")


def test_write_agent_stage_artifacts_writes_rank_repair_plan(tmp_path: Path):
    output = AgentStageOutput(
        stage="rank_repairs",
        payload={
            "repair_plan": [
                {
                    "rank": 1,
                    "proposal_id": "proposal-hierarchy-symptom-001",
                    "priority": "high",
                    "risk": "medium",
                    "reason": "It repairs the most visible hierarchy gap.",
                    "human_checks": ["Review hierarchy prompt impact."],
                }
            ]
        },
    )

    paths = write_agent_stage_artifacts(tmp_path, output)

    assert paths == {
        "llm_repair_plan_json": tmp_path / "llm_repair_plan.json",
        "llm_repair_plan": tmp_path / "llm_repair_plan.md",
    }
    assert paths["llm_repair_plan_json"].exists()
    assert paths["llm_repair_plan"].exists()
    written_payload = json.loads(
        paths["llm_repair_plan_json"].read_text(encoding="utf-8")
    )
    assert (
        written_payload["repair_plan"][0]["proposal_id"]
        == "proposal-hierarchy-symptom-001"
    )
    markdown = paths["llm_repair_plan"].read_text(encoding="utf-8")
    assert "# 修复方案排序" in markdown
    assert "proposal-hierarchy-symptom-001" in markdown
