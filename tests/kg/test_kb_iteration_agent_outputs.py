import json
from pathlib import Path

import pytest

from lightrag.kb_iteration.agent_outputs import (
    AgentStageOutput,
    parse_agent_stage_output,
    stage_output_to_markdown,
    write_agent_stage_artifacts,
)
from lightrag.kb_iteration.subagent_contracts import role_contract


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


def _candidate_expansion_payload(**overrides):
    evidence_quote = "Oseltamivir is indicated for influenza treatment in this guideline."
    payload = {
        "id": "proposal-candidate-expansion-001",
        "type": "candidate_kg_expansion",
        "target": "candidate:oseltamivir-influenza",
        "proposed_change": "Add a grounded treatment indication edge.",
        "reason": "The source states a direct treatment indication.",
        "evidence": ["source_id: chunk-1; file_path: guide.md"],
        "confidence": 0.8,
        "risk": "medium",
        "requires_approval": True,
        "expected_metric_change": {},
        "action_payload": {
            "candidate_nodes": [
                {"id": "oseltamivir", "label": "Oseltamivir", "entity_type": "Drug"},
                {"id": "influenza", "label": "Influenza", "entity_type": "Disease"},
            ],
            "candidate_edges": [
                {
                    "source": "oseltamivir",
                    "target": "influenza",
                    "keywords": "has_indication",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                }
            ],
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "evidence_quote": evidence_quote,
            "why_not_existing": "No existing edge captures this indication.",
        },
    }
    payload.update(overrides)
    return payload


def _parse_candidate_expansion_payload(payload, **kwargs):
    return parse_agent_stage_output(
        "propose",
        json.dumps({"proposals": [payload]}, ensure_ascii=False),
        **kwargs,
    )


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


def test_parse_agent_stage_output_normalizes_numeric_string_metric_changes():
    output = parse_agent_stage_output(
        "propose",
        json.dumps(
            {
                "proposals": [
                    _proposal_payload(
                        expected_metric_change={
                            "hierarchy_completeness": "6",
                            "precision_delta": "+0.25",
                            "error_rate_delta": "1e-2",
                        }
                    )
                ]
            },
            ensure_ascii=False,
        ),
    )

    assert output.proposals[0].expected_metric_change == {
        "hierarchy_completeness": 6,
        "precision_delta": 0.25,
        "error_rate_delta": 0.01,
    }


@pytest.mark.parametrize(
    "metric_value",
    ["about 6", "10%", "", "   ", True, False, None, {}, []],
)
def test_parse_agent_stage_output_rejects_non_numeric_metric_change_strings(
    metric_value,
):
    with pytest.raises(ValueError, match="expected_metric_change"):
        parse_agent_stage_output(
            "propose",
            json.dumps(
                {
                    "proposals": [
                        _proposal_payload(
                            expected_metric_change={
                                "hierarchy_completeness": metric_value,
                            }
                        )
                    ]
                },
                ensure_ascii=False,
            ),
        )


@pytest.mark.parametrize(
    "metric_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_parse_agent_stage_output_rejects_non_finite_numeric_metric_changes(
    metric_value,
):
    with pytest.raises(ValueError, match="expected_metric_change"):
        parse_agent_stage_output(
            "propose",
            json.dumps(
                {
                    "proposals": [
                        _proposal_payload(
                            expected_metric_change={
                                "hierarchy_completeness": metric_value,
                            }
                        )
                    ]
                },
                ensure_ascii=False,
            ),
        )


def test_parse_agent_stage_output_rejects_structured_proposal_evidence():
    with pytest.raises(ValueError, match="EVIDENCE_MUST_BE_STRING"):
        parse_agent_stage_output(
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
                                },
                                "quality:hierarchy_missing_branch_count=1",
                            ]
                        )
                    ]
                },
                ensure_ascii=False,
            ),
        )


def test_parse_agent_stage_output_requires_candidate_edge_endpoint_types():
    with pytest.raises(ValueError, match="CANDIDATE_EDGE_TYPES_REQUIRED"):
        _parse_candidate_expansion_payload(_candidate_expansion_payload())


def test_parse_agent_stage_output_accepts_exact_candidate_evidence_tuple():
    payload = _candidate_expansion_payload()
    payload["action_payload"]["candidate_edges"][0].update(
        {"source_type": "Drug", "target_type": "Disease"}
    )

    output = _parse_candidate_expansion_payload(
        payload,
        allowed_evidence_spans=[
            {
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "evidence_quote": payload["action_payload"]["evidence_quote"],
            }
        ],
    )

    assert output.proposals[0].id == "proposal-candidate-expansion-001"


def test_parse_agent_stage_output_rejects_cross_combined_evidence_tuple():
    payload = _candidate_expansion_payload()
    payload["action_payload"].update(
        {
            "source_id": "chunk-1",
            "file_path": "other-guide.md",
        }
    )
    payload["action_payload"]["candidate_edges"][0].update(
        {
            "source_type": "Drug",
            "target_type": "Disease",
            "source_id": "chunk-1",
            "file_path": "other-guide.md",
        }
    )

    with pytest.raises(ValueError, match="EVIDENCE_TUPLE_MUST_MATCH_ALLOWED_SPAN"):
        _parse_candidate_expansion_payload(
            payload,
            allowed_evidence_spans=[
                {
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                    "evidence_quote": payload["action_payload"]["evidence_quote"],
                },
                {
                    "source_id": "chunk-2",
                    "file_path": "other-guide.md",
                    "evidence_quote": payload["action_payload"]["evidence_quote"],
                },
            ],
        )


def test_parse_agent_stage_output_rejects_sep_joined_evidence_references():
    payload = _candidate_expansion_payload()
    payload["action_payload"].update(
        {
            "source_id": "chunk-1<SEP>chunk-2",
            "file_path": "guide.md",
        }
    )
    payload["action_payload"]["candidate_edges"][0].update(
        {
            "source_type": "Drug",
            "target_type": "Disease",
            "source_id": "chunk-1<SEP>chunk-2",
            "file_path": "guide.md",
        }
    )

    with pytest.raises(ValueError, match="EVIDENCE_TUPLE_MUST_MATCH_ALLOWED_SPAN"):
        _parse_candidate_expansion_payload(
            payload,
            allowed_evidence_spans=[
                {
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                    "evidence_quote": payload["action_payload"]["evidence_quote"],
                }
            ],
        )


def test_role_contract_exposes_structured_retry_error_codes_and_missing_fields():
    contract = role_contract("treatment").to_dict()

    assert "EVIDENCE_MUST_BE_STRING" in contract["retry_error_codes"]
    assert contract["candidate_edge_required_fields"] == (
        "source",
        "target",
        "source_type",
        "target_type",
        "keywords",
        "source_id",
        "file_path",
    )
    assert contract["retry_contract"]["CANDIDATE_EDGE_TYPES_REQUIRED"][
        "missing_fields"
    ] == ("source_type", "target_type")


def test_parse_agent_stage_output_normalizes_common_llm_proposal_aliases():
    output = parse_agent_stage_output(
        "propose",
        json.dumps(
            {
                "proposals": [
                    {
                        "proposal_id": "修复-咳嗽-流感",
                        "proposal_type": "medical_relation_schema_migration",
                        "target": "edge:咳嗽->流行性感冒",
                        "reason": "Reverse clinical manifestation edge direction.",
                        "requires_approval": True,
                        "expected_metric_change": {},
                        "action_payload": {
                            "action": "replace_relation",
                            "edge_id": "咳嗽->流行性感冒",
                            "expected_source": "咳嗽",
                            "expected_target": "流行性感冒",
                            "current_keywords": "临床表现",
                            "new_source": "流行性感冒",
                            "new_target": "咳嗽",
                            "new_keywords": "has_manifestation",
                            "source_id": "chunk-1",
                            "file_path": "guide.md",
                        },
                        "evidence": [
                            "source_id: chunk-1; file_path: guide.md; item_id: 咳嗽->流行性感冒"
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )

    proposal = output.proposals[0]
    assert proposal.id.startswith("prop-medical_relation_schema_migration-")
    assert proposal.type == "medical_relation_schema_migration"
    assert proposal.proposed_change == "Reverse clinical manifestation edge direction."
    assert proposal.evidence == [
        "source_id: chunk-1; file_path: guide.md; item_id: 咳嗽->流行性感冒"
    ]


def test_parse_agent_stage_output_derives_evidence_from_action_payload():
    output = parse_agent_stage_output(
        "propose",
        json.dumps(
            {
                "proposals": [
                    {
                        "proposal_id": "prop-fix-flu-cough",
                        "proposal_type": "medical_relation_schema_migration",
                        "target": "edge:flu->cough",
                        "reason": "Reverse clinical manifestation edge direction.",
                        "requires_approval": True,
                        "expected_metric_change": {},
                        "action_payload": {
                            "action": "replace_relation",
                            "edge_id": "flu->cough",
                            "expected_source": "cough",
                            "expected_target": "flu",
                            "current_keywords": "clinical_manifestation",
                            "new_source": "flu",
                            "new_target": "cough",
                            "new_keywords": "has_manifestation",
                            "source_id": "chunk-1",
                            "file_path": "guide.md",
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )

    proposal = output.proposals[0]
    assert proposal.id == "prop-fix-flu-cough"
    assert proposal.evidence == [
        "source_id: chunk-1; file_path: guide.md; relation_id: flu->cough"
    ]


def test_parse_agent_stage_output_uses_change_alias_for_default_proposed_change():
    output = parse_agent_stage_output(
        "propose",
        json.dumps(
            {
                "proposals": [
                    _proposal_payload(
                        proposed_change="",
                        change="Record the aliased recommendation.",
                    )
                ]
            },
            ensure_ascii=False,
        ),
    )

    assert output.proposals[0].proposed_change == (
        "Record the aliased recommendation."
    )


@pytest.mark.parametrize("unknown_key", ["comment", "reason"])
def test_parse_agent_stage_output_rejects_unknown_structured_evidence_fields(
    unknown_key,
):
    with pytest.raises(ValueError, match="evidence"):
        parse_agent_stage_output(
            "propose",
            json.dumps(
                {
                    "proposals": [
                        _proposal_payload(
                            evidence=[
                                {
                                    "source_id": "chunk-1",
                                    unknown_key: "free-form evidence laundering",
                                }
                            ]
                        )
                    ]
                },
                ensure_ascii=False,
            ),
        )


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


def test_parse_agent_stage_output_defaults_context_request_to_human_review():
    proposal = _proposal_payload(
        id="proposal-context-request-default-approval-001",
        type="review_context_request",
        target="review-context",
        proposed_change="Request source snippets for review.",
        reason="The reviewer needs more context before mutation.",
        evidence=[],
        expected_metric_change={},
    )
    proposal.pop("requires_approval")

    output = parse_agent_stage_output(
        "propose",
        json.dumps({"proposals": [proposal]}, ensure_ascii=False),
    )

    assert output.proposals[0].type == "review_context_request"
    assert output.proposals[0].requires_approval is True


def test_parse_agent_stage_output_coerces_context_request_false_approval():
    output = parse_agent_stage_output(
        "propose",
        json.dumps(
            {
                "proposals": [
                    _proposal_payload(
                        id="proposal-context-request-coerced-approval-001",
                        type="review_context_request",
                        target="review-context",
                        proposed_change="Request source snippets for review.",
                        reason="The reviewer needs more context before mutation.",
                        evidence=[],
                        requires_approval=False,
                        expected_metric_change={},
                    )
                ]
            },
            ensure_ascii=False,
        ),
    )

    assert output.proposals[0].type == "review_context_request"
    assert output.proposals[0].requires_approval is True


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
