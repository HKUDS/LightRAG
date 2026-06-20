import json
from pathlib import Path

import pytest

from lightrag.kb_iteration.agent_pipeline import (
    LLMAgentPipelineConfig,
    _stage_prompt,
    run_llm_agent_pipeline,
)


class SequencedAgentClient:
    model = "agent-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []
        self.outputs = [
            {
                "issue_explanations": [
                    {
                        "id": "issue-hierarchy-symptom-001",
                        "category": "hierarchy_completeness",
                        "severity": "high",
                        "explanation": "Disease nodes connect directly to symptom leaves.",
                        "impact": "Symptom retrieval misses the intermediate branch.",
                        "evidence_refs": ["entity:fever", "chunk-1"],
                    }
                ]
            },
            {
                "required": [{"key": "symptom", "label": "Symptom"}],
                "present": [{"key": "disease", "label": "Disease"}],
                "missing": [{"key": "symptom", "label": "Symptom"}],
                "missing_branches": [
                    {
                        "key": "symptom",
                        "label": "Symptom",
                        "reason": "Fever is represented without a hierarchy branch.",
                    }
                ],
            },
            {
                "evidence_map": [
                    {
                        "issue_id": "issue-hierarchy-symptom-001",
                        "target": "hierarchy:symptom",
                        "confidence": 0.9,
                        "missing_evidence": [],
                        "supporting_items": [
                            {
                                "item_type": "entity",
                                "item_id": "entity fever",
                                "source_id": "chunk-1",
                                "file_path": "guide.md",
                                "evidence_status": "grounded",
                            }
                        ],
                    }
                ]
            },
            {
                "proposals": [
                    {
                        "id": "proposal-hierarchy-symptom-001",
                        "type": "hierarchy_rule_change",
                        "target": "lightrag/medical_kg/hierarchy.py",
                        "proposed_change": "Add a controlled symptom hierarchy branch.",
                        "reason": "Fever has grounded evidence but no symptom branch.",
                        "evidence": [
                            "source_id: chunk-1; file_path: guide.md; item_id: entity fever"
                        ],
                        "confidence": 0.86,
                        "risk": "medium",
                        "requires_approval": True,
                        "expected_metric_change": {"hierarchy_completeness": 6},
                    }
                ]
            },
            {
                "repair_plan": [
                    {
                        "rank": 1,
                        "proposal_id": "proposal-hierarchy-symptom-001",
                        "priority": "high",
                        "risk": "medium",
                        "reason": "It repairs the grounded missing branch first.",
                        "human_checks": ["Review hierarchy rule impact."],
                    }
                ]
            },
            {
                "judge_results": [
                    {
                        "proposal_id": "proposal-hierarchy-symptom-001",
                        "decision": "needs_human",
                        "reason": "The mutation is evidence-backed but requires approval.",
                    }
                ]
            },
        ]

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return json.dumps(self.outputs[len(self.calls) - 1], ensure_ascii=False)


class NoEvidenceAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[2] = {
            "evidence_map": [
                {
                    "issue_id": "issue-hierarchy-symptom-001",
                    "target": "hierarchy:symptom",
                    "confidence": 0.2,
                    "missing_evidence": ["No grounded source item found."],
                    "supporting_items": [],
                }
            ]
        }
        self.outputs[3] = {"proposals": []}


class FailingRankAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[4] = "not json"

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        output = self.outputs[len(self.calls) - 1]
        if isinstance(output, str):
            return output
        return json.dumps(output, ensure_ascii=False)


class FailingJudgeAgentClient(FailingRankAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[4] = {
            "repair_plan": [
                {
                    "rank": 1,
                    "proposal_id": "proposal-hierarchy-symptom-001",
                    "priority": "high",
                    "risk": "medium",
                    "reason": "It repairs the grounded missing branch first.",
                    "human_checks": ["Review hierarchy rule impact."],
                }
            ]
        }
        self.outputs[5] = "not json"


class JudgeNeedsHumanReportNoteClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3] = {
            "proposals": [
                {
                    "id": "proposal-quality-note-001",
                    "type": "quality_report_note",
                    "target": "quality_report.md",
                    "proposed_change": "Record hierarchy observation for reviewer visibility.",
                    "reason": "Fever is grounded but the hierarchy score needs review.",
                    "evidence": [
                        "source_id: chunk-1; file_path: guide.md; item_id: entity fever"
                    ],
                    "confidence": 0.78,
                    "risk": "low",
                    "requires_approval": False,
                    "expected_metric_change": {"hierarchy_completeness": 0},
                }
            ]
        }
        self.outputs[4] = {
            "repair_plan": [
                {
                    "rank": 1,
                    "proposal_id": "proposal-quality-note-001",
                    "priority": "medium",
                    "risk": "low",
                    "reason": "Review the report note before accepting it.",
                    "human_checks": ["Confirm note wording."],
                }
            ]
        }
        self.outputs[5] = {
            "judge_results": [
                {
                    "proposal_id": "proposal-quality-note-001",
                    "decision": "needs_human",
                    "reason": "Human review should confirm the report note.",
                }
            ]
        }


class LegacyJudgeNeedsHumanReportNoteClient(JudgeNeedsHumanReportNoteClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[5] = {
            "decision": "needs_human",
            "reason": "Legacy judge response still needs human review.",
            "risk_override": "low",
            "required_human_checks": ["Confirm note wording."],
        }


class JudgeAcceptsReportNoteClient(JudgeNeedsHumanReportNoteClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[5] = {
            "judge_results": [
                {
                    "proposal_id": "proposal-quality-note-001",
                    "decision": "recommend_accept",
                    "reason": "Judge accepts the report note.",
                }
            ]
        }


class UnknownJudgeDecisionReportNoteClient(JudgeNeedsHumanReportNoteClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[5] = {
            "judge_results": [
                {
                    "proposal_id": "proposal-quality-note-001",
                    "decision": "needs_review",
                    "reason": "Typo decision should not be accepted.",
                }
            ]
        }


class InvalidJudgeResultsClient(JudgeNeedsHumanReportNoteClient):
    def __init__(self, judge_payload: dict) -> None:
        super().__init__()
        self.outputs[3]["proposals"].append(
            {
                "id": "proposal-quality-note-002",
                "type": "quality_report_note",
                "target": "quality_report.md",
                "proposed_change": "Record source coverage observation.",
                "reason": "The same grounded source should be reviewed for coverage.",
                "evidence": [
                    "source_id: chunk-1; file_path: guide.md; item_id: entity fever"
                ],
                "confidence": 0.73,
                "risk": "low",
                "requires_approval": False,
                "expected_metric_change": {"hierarchy_completeness": 0},
            }
        )
        self.outputs[5] = judge_payload


class FabricatedEvidenceAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3]["proposals"][0]["evidence"] = ["made-up-source"]


class FabricatedGroundedEvidenceAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[2] = {
            "evidence_map": [
                {
                    "issue_id": "issue-hierarchy-symptom-001",
                    "target": "hierarchy:symptom",
                    "confidence": 0.99,
                    "missing_evidence": [],
                    "supporting_items": [
                        {
                            "item_type": "source",
                            "item_id": "fake-source",
                            "source_id": "fake-source",
                            "file_path": "fake.md",
                            "evidence_status": "grounded",
                        }
                    ],
                }
            ]
        }
        self.outputs[3]["proposals"][0]["evidence"] = ["fake-source"]


class MixedGroundedAndFabricatedEvidenceAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3]["proposals"][0]["evidence"] = [
            "source_id: chunk-1; file_path: guide.md; item_id: entity fever",
            "fake-source",
        ]


class FabricatedActionPayloadAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3] = {
            "proposals": [
                {
                    "id": "proposal-medical-schema-001",
                    "type": "medical_relation_schema_migration",
                    "target": "kg:relation:flu-fever",
                    "proposed_change": "Normalize the relation direction.",
                    "reason": "The relation should use a canonical medical predicate.",
                    "evidence": [
                        "source_id: chunk-1; file_path: guide.md; relation_id: flu-fever"
                    ],
                    "confidence": 0.82,
                    "risk": "medium",
                    "requires_approval": True,
                    "expected_metric_change": {"medical_relation_schema_score": 4},
                    "action_payload": {
                        "action": "replace_relation",
                        "edge_id": "flu-fever",
                        "expected_source": "flu",
                        "expected_target": "entity fever",
                        "current_keywords": "clinical_manifestation",
                        "new_source": "invented-disease",
                        "new_target": "entity fever",
                        "new_keywords": "has_manifestation",
                        "qualifiers": {},
                    },
                }
            ]
        }


class MismatchedActionPayloadTypeAgentClient(SequencedAgentClient):
    def __init__(self, *, field_name: str, field_value: str) -> None:
        super().__init__()
        action_payload = {
            "action": "replace_relation",
            "edge_id": "flu-fever",
            "expected_source": "flu",
            "expected_target": "entity fever",
            "current_keywords": "clinical_manifestation",
            "new_source": "flu",
            "new_target": "entity fever",
            "new_keywords": "has_manifestation",
            "qualifiers": {},
        }
        action_payload[field_name] = field_value
        self.outputs[3] = {
            "proposals": [
                {
                    "id": "proposal-medical-schema-typed-001",
                    "type": "medical_relation_schema_migration",
                    "target": "kg:relation:flu-fever",
                    "proposed_change": "Normalize the relation direction.",
                    "reason": "The relation should use a canonical medical predicate.",
                    "evidence": [
                        "source_id: chunk-1; file_path: guide.md; relation_id: flu-fever"
                    ],
                    "confidence": 0.82,
                    "risk": "medium",
                    "requires_approval": True,
                    "expected_metric_change": {"medical_relation_schema_score": 4},
                    "action_payload": action_payload,
                }
            ]
        }


class MissingExecutableMedicalActionPayloadAgentClient(SequencedAgentClient):
    def __init__(self, proposal_type: str) -> None:
        super().__init__()
        self.outputs[3] = {
            "proposals": [
                {
                    "id": f"proposal-{proposal_type}-001",
                    "type": proposal_type,
                    "target": "kg:medical-action",
                    "proposed_change": "Prepare a deterministic medical KG action.",
                    "reason": "Executable medical changes need deterministic payloads.",
                    "evidence": [
                        "source_id: chunk-1; file_path: guide.md; item_id: entity fever"
                    ],
                    "confidence": 0.78,
                    "risk": "medium",
                    "requires_approval": True,
                    "expected_metric_change": {"medical_relation_schema_score": 1},
                }
            ]
        }


class LaunderedFreeTextEvidenceAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3]["proposals"][0]["evidence"] = [
            "entity fever chunk-1 guide.md fake-source"
        ]


class StructuredFabricatedItemEvidenceAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3]["proposals"][0]["evidence"] = [
            "source_id: chunk-1; item_id: fake-source"
        ]


class StructuredUnknownKeyEvidenceAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3]["proposals"][0]["evidence"] = [
            "source_id: chunk-1; note: guide.md"
        ]


class StructuredWrongKeyEvidenceAgentClient(SequencedAgentClient):
    def __init__(self, evidence: str) -> None:
        super().__init__()
        self.outputs[3]["proposals"][0]["evidence"] = [evidence]


class SubstringFabricatedEvidenceAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3]["proposals"][0]["evidence"] = [
            "flustered fabricated claim"
        ]


class InferOnlyEvidenceAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[1] = {
            "required": [{"key": "invented_alpha", "label": "Invented Branch Alpha"}],
            "present": [],
            "missing": [{"key": "invented_alpha", "label": "Invented Branch Alpha"}],
            "missing_branches": [
                {
                    "key": "invented_alpha",
                    "label": "Invented Branch Alpha",
                    "reason": "LLM-only inferred branch.",
                }
            ],
        }
        self.outputs[2] = {
            "evidence_map": [
                {
                    "issue_id": "issue-hierarchy-symptom-001",
                    "target": "hierarchy:invented_alpha",
                    "confidence": 0.2,
                    "missing_evidence": ["No grounded source item found."],
                    "supporting_items": [],
                }
            ]
        }
        self.outputs[3]["proposals"][0]["evidence"] = ["Invented Branch Alpha"]


class BranchKeyOrLabelOnlyEvidenceAgentClient(SequencedAgentClient):
    def __init__(self, evidence: str) -> None:
        super().__init__()
        self.outputs[3]["proposals"][0]["evidence"] = [evidence]


class RetryExplainAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs.insert(0, "not json")

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        output = self.outputs[len(self.calls) - 1]
        if isinstance(output, str):
            return output
        return json.dumps(output, ensure_ascii=False)


class RetryProposeMetricAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        invalid_proposal = dict(self.outputs[3]["proposals"][0])
        invalid_proposal["expected_metric_change"] = {
            "hierarchy_completeness": "about 6"
        }
        self.outputs.insert(3, {"proposals": [invalid_proposal]})


class RetryProposeTwiceAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        invalid_metric_proposal = dict(self.outputs[3]["proposals"][0])
        invalid_metric_proposal["expected_metric_change"] = {
            "hierarchy_completeness": "about 6"
        }
        invalid_evidence_proposal = dict(self.outputs[3]["proposals"][0])
        invalid_evidence_proposal["evidence"] = ["not-grounded-source"]
        self.outputs.insert(3, {"proposals": [invalid_metric_proposal]})
        self.outputs.insert(4, {"proposals": [invalid_evidence_proposal]})


class AlwaysInvalidProposeAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        invalid_proposal = dict(self.outputs[3]["proposals"][0])
        invalid_proposal["expected_metric_change"] = {
            "hierarchy_completeness": "about 6"
        }
        self.outputs[3] = {"proposals": [invalid_proposal]}

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        index = len(self.calls) - 1
        if index >= 3:
            index = 3
        output = self.outputs[index]
        if isinstance(output, str):
            return output
        return json.dumps(output, ensure_ascii=False)


SECRET_BEARING_CLIENT_ERROR = (
    "provider 401 while refreshing credentials: "
    "api_key=sk-review-secret-123 token=raw-token-456 "
    "Authorization: Bearer bearer-secret-789 secret=hunter2; retry later"
)
HEADER_STYLE_CLIENT_ERROR = (
    "transport failed: X-API-Key: header-secret-321 "
    "session_token=session-secret-654 while contacting provider"
)
SECRET_FRAGMENTS = (
    "sk-review-secret-123",
    "raw-token-456",
    "bearer-secret-789",
    "hunter2",
    "header-secret-321",
    "session-secret-654",
)


class ClientErrorOnceThenSuccessAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.output_index = 0

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if len(self.calls) == 1:
            raise RuntimeError(SECRET_BEARING_CLIENT_ERROR)
        output = self.outputs[self.output_index]
        self.output_index += 1
        return json.dumps(output, ensure_ascii=False)


class JudgeClientErrorAgentClient(SequencedAgentClient):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if len(self.calls) == 6:
            raise RuntimeError(SECRET_BEARING_CLIENT_ERROR)
        return json.dumps(self.outputs[len(self.calls) - 1], ensure_ascii=False)


class AlwaysClientErrorAgentClient:
    model = "agent-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        raise RuntimeError(HEADER_STYLE_CLIENT_ERROR)


def test_propose_prompt_requires_action_payload_for_medical_migrations():
    prompt = _stage_prompt("propose", "clinical_guideline_zh")

    assert "action_payload" in prompt
    assert "medical_relation_schema_migration" in prompt
    assert "replace_relation" in prompt
    assert "Do not invent medical facts" in prompt


def test_run_llm_agent_pipeline_writes_multistage_artifacts_and_queues_review(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = SequencedAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(),
        profile="medical_kg",
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-hierarchy-symptom-001"]
    assert len(client.calls) == 6
    assert (package / "llm_issue_analysis.md").exists()
    assert (package / "llm_missing_branch_inference.md").exists()
    assert (package / "llm_evidence_map.md").exists()
    assert (package / "llm_repair_plan.md").exists()
    assert (package / "llm_judge_report.md").exists()
    assert "proposal-hierarchy-symptom-001" in (
        package / "approval_queue.md"
    ).read_text(encoding="utf-8")

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    assert trace["mode"] == "agent_pipeline"
    stage_traces = trace["stages"]
    assert [stage["stage"] for stage in stage_traces] == [
        "explain",
        "infer_branches",
        "locate_evidence",
        "propose",
        "rank_repairs",
        "judge",
    ]
    artifact_keys_by_stage = {
        stage["stage"]: stage["artifact_keys"] for stage in stage_traces
    }
    assert "llm_issue_analysis" in artifact_keys_by_stage["explain"]
    assert (
        "llm_missing_branch_inference"
        in artifact_keys_by_stage["infer_branches"]
    )
    assert "llm_evidence_map" in artifact_keys_by_stage["locate_evidence"]
    assert "llm_review_report" in artifact_keys_by_stage["propose"]
    assert "proposals_generated" in artifact_keys_by_stage["propose"]
    assert "approval_queue" in artifact_keys_by_stage["propose"]
    assert "improvement_backlog" in artifact_keys_by_stage["propose"]
    assert "llm_repair_plan" in artifact_keys_by_stage["rank_repairs"]
    assert "llm_judge_report" in artifact_keys_by_stage["judge"]


def test_pipeline_does_not_queue_proposals_when_later_stage_fails(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = FailingRankAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    assert len(client.calls) == 5
    approval_queue = package / "approval_queue.md"
    improvement_backlog = package / "improvement_backlog.md"
    for artifact in (approval_queue, improvement_backlog):
        assert artifact.exists()
        text = artifact.read_text(encoding="utf-8")
        assert "proposals: []" in text
        assert "proposal-hierarchy-symptom-001" not in text


def test_pipeline_removes_stale_downstream_artifacts_when_later_stage_fails(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    (package / "llm_repair_plan.md").write_text("stale repair", encoding="utf-8")
    (package / "llm_judge_report.md").write_text("stale judge", encoding="utf-8")
    client = FailingRankAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    assert not (package / "llm_repair_plan.md").exists()
    assert not (package / "llm_judge_report.md").exists()
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    assert "proposals: []" in approval_queue
    assert "proposals: []" in proposals_yaml


def test_pipeline_failure_report_lists_rejected_attempts(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = AlwaysInvalidProposeAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=2),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    report = (package / "llm_review_report.md").read_text(encoding="utf-8")
    assert "## Rejected Attempts" in report
    assert "- Attempt 1: proposal expected_metric_change values must be numbers" in report
    assert "- Attempt 2: proposal expected_metric_change values must be numbers" in report
    assert "- Attempt 3: proposal expected_metric_change values must be numbers" in report
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert propose_trace["stage"] == "propose"
    assert propose_trace["attempts"] == 3
    assert len(propose_trace["attempt_logs"]) == 3


def test_pipeline_rejects_fabricated_proposal_evidence(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = FabricatedEvidenceAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    assert len(client.calls) == 4
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    report = (package / "llm_review_report.md").read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert "proposals: []" in approval_queue
    assert "proposal-hierarchy-symptom-001" not in approval_queue
    assert "evidence" in report
    assert "ground" in report
    assert propose_trace["stage"] == "propose"
    assert "evidence" in propose_trace["error"]
    assert "ground" in propose_trace["error"]


def test_pipeline_rejects_proposal_evidence_grounded_only_by_llm_evidence_map(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = FabricatedGroundedEvidenceAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    assert len(client.calls) == 4
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    report = (package / "llm_review_report.md").read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert "proposals: []" in approval_queue
    assert "proposal-hierarchy-symptom-001" not in approval_queue
    assert "evidence is not grounded in deterministic artifacts" in report
    assert propose_trace["stage"] == "propose"
    assert (
        "evidence is not grounded in deterministic artifacts"
        in propose_trace["error"]
    )


def test_pipeline_rejects_proposal_with_any_fabricated_evidence_item(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = MixedGroundedAndFabricatedEvidenceAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    assert len(client.calls) == 4
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    report = (package / "llm_review_report.md").read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert "proposals: []" in approval_queue
    assert "proposal-hierarchy-symptom-001" not in approval_queue
    assert "proposal proposal-hierarchy-symptom-001" in report
    assert "evidence is not grounded in deterministic artifacts" in report
    assert propose_trace["stage"] == "propose"
    assert (
        "evidence is not grounded in deterministic artifacts"
        in propose_trace["error"]
    )


def test_pipeline_rejects_fabricated_action_payload_ids(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = FabricatedActionPayloadAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
        profile="medical_kg",
    )

    assert result.stop_reason == "invalid_llm_output"
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert (
        "action_payload new_source is not grounded in deterministic artifacts"
        in propose_trace["error"]
    )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("edge_id", "flu"),
        ("expected_source", "flu-fever"),
        ("expected_target", "flu-fever"),
        ("new_source", "flu-fever"),
        ("new_target", "flu-fever"),
    ],
)
def test_pipeline_rejects_action_payload_ids_with_wrong_artifact_type(
    tmp_path: Path,
    field_name: str,
    field_value: str,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = MismatchedActionPayloadTypeAgentClient(
        field_name=field_name,
        field_value=field_value,
    )

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
        profile="medical_kg",
    )

    assert result.stop_reason == "invalid_llm_output"
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert (
        f"action_payload {field_name} is not grounded in deterministic artifacts"
        in propose_trace["error"]
    )


@pytest.mark.parametrize(
    "proposal_type",
    [
        "value_node_to_qualifier",
        "entity_alias_merge",
        "medical_fact_role_split",
    ],
)
def test_pipeline_requires_action_payload_for_executable_medical_proposal_types(
    tmp_path: Path,
    proposal_type: str,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = MissingExecutableMedicalActionPayloadAgentClient(proposal_type)

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
        profile="medical_kg",
    )

    assert result.stop_reason == "invalid_llm_output"
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    proposal_id = f"proposal-{proposal_type}-001"
    assert (
        f"proposal {proposal_id} action_payload is required for executable "
        "medical proposal type"
        in propose_trace["error"]
    )


def test_pipeline_rejects_laundered_fabricated_reference_in_single_evidence_item(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = LaunderedFreeTextEvidenceAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    assert len(client.calls) == 4
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert "proposals: []" in approval_queue
    assert "proposal-hierarchy-symptom-001" not in approval_queue
    assert propose_trace["stage"] == "propose"
    assert "evidence is not grounded in deterministic artifacts" in propose_trace["error"]


def test_pipeline_rejects_structured_evidence_with_fabricated_item_id(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = StructuredFabricatedItemEvidenceAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    assert len(client.calls) == 4
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert "proposals: []" in approval_queue
    assert "proposal-hierarchy-symptom-001" not in approval_queue
    assert propose_trace["stage"] == "propose"
    assert "evidence is not grounded in deterministic artifacts" in propose_trace["error"]


def test_pipeline_rejects_structured_evidence_with_unknown_key(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = StructuredUnknownKeyEvidenceAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    assert len(client.calls) == 4
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert "proposals: []" in approval_queue
    assert "proposal-hierarchy-symptom-001" not in approval_queue
    assert propose_trace["stage"] == "propose"
    assert "evidence is not grounded in deterministic artifacts" in propose_trace["error"]


@pytest.mark.parametrize(
    "evidence",
    [
        "source_id: guide.md",
        "file_path: chunk-1",
        "entity_id: guide.md",
    ],
)
def test_pipeline_rejects_structured_evidence_with_wrong_key_type(
    tmp_path: Path, evidence: str
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = StructuredWrongKeyEvidenceAgentClient(evidence)

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    assert len(client.calls) == 4
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert "proposals: []" in approval_queue
    assert "proposal-hierarchy-symptom-001" not in approval_queue
    assert propose_trace["stage"] == "propose"
    assert "evidence is not grounded in deterministic artifacts" in propose_trace["error"]


def test_pipeline_rejects_evidence_that_only_contains_reference_as_substring(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = SubstringFabricatedEvidenceAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    assert len(client.calls) == 4
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    assert "proposals: []" in approval_queue
    assert "proposal-hierarchy-symptom-001" not in approval_queue


def test_pipeline_rejects_evidence_grounded_only_by_llm_branch_inference(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = InferOnlyEvidenceAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    assert len(client.calls) == 4
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    assert "proposals: []" in approval_queue
    assert "proposal-hierarchy-symptom-001" not in approval_queue


@pytest.mark.parametrize("evidence", ["symptom", "Symptom"])
def test_pipeline_rejects_evidence_grounded_only_by_quality_branch_key_or_label(
    tmp_path: Path, evidence: str
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = BranchKeyOrLabelOnlyEvidenceAgentClient(evidence)

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    assert len(client.calls) == 4
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert "proposals: []" in approval_queue
    assert "proposal-hierarchy-symptom-001" not in approval_queue
    assert propose_trace["stage"] == "propose"
    assert "evidence is not grounded in deterministic artifacts" in propose_trace["error"]


def test_pipeline_queues_proposals_when_judge_is_unavailable(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = FailingJudgeAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-hierarchy-symptom-001"]
    assert len(client.calls) == 6
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    judge_trace = trace["stages"][-1]
    assert "proposal-hierarchy-symptom-001" in approval_queue
    assert "judge_unavailable" in approval_queue
    assert "judge_unavailable" in proposals_yaml
    assert judge_trace["stage"] == "judge"
    assert judge_trace["state"] == "judge_unavailable"
    assert "valid JSON" in judge_trace["error"]


def test_pipeline_attaches_judge_results_and_routes_needs_human(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = JudgeNeedsHumanReportNoteClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-quality-note-001"]

    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    assert "proposal-quality-note-001" in approval_queue
    assert "needs_human" in approval_queue
    assert "Human review should confirm the report note." in approval_queue
    assert "proposal-quality-note-001" in proposals_yaml
    assert "needs_human" in proposals_yaml
    assert "Human review should confirm the report note." in proposals_yaml
    assert "judge: {}" not in proposals_yaml


def test_pipeline_attaches_legacy_single_judge_result_for_single_proposal(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = LegacyJudgeNeedsHumanReportNoteClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-quality-note-001"]
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    assert "proposal-quality-note-001" in proposals_yaml
    assert "needs_human" in proposals_yaml
    assert "Legacy judge response still needs human review." in proposals_yaml
    assert "judge: {}" not in proposals_yaml
    assert "proposal-quality-note-001" in approval_queue
    assert "needs_human" in approval_queue


def test_pipeline_keeps_non_approval_proposals_pending_human_review(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = JudgeAcceptsReportNoteClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-quality-note-001"]
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    assert "proposal-quality-note-001" in proposals_yaml
    assert "recommend_accept" in proposals_yaml
    assert "Judge accepts the report note." in proposals_yaml
    assert "proposal-quality-note-001" in approval_queue


def test_pipeline_rejects_unknown_judge_decision_and_human_gates_proposal(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = UnknownJudgeDecisionReportNoteClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-quality-note-001"]
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    judge_trace = trace["stages"][-1]
    assert judge_trace["stage"] == "judge"
    assert judge_trace["state"] == "judge_unavailable"
    assert "decision" in judge_trace["error"]

    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    assert "proposal-quality-note-001" in approval_queue
    assert "proposal-quality-note-001" in proposals_yaml
    assert "requires_approval: true" in proposals_yaml
    assert "judge_unavailable" in proposals_yaml


@pytest.mark.parametrize(
    ("judge_payload", "expected_error"),
    [
        (
            {
                "judge_results": [
                    {
                        "proposal_id": "proposal-quality-note-001",
                        "decision": "recommend_accept",
                        "reason": "Only one proposal was judged.",
                    }
                ]
            },
            "missing",
        ),
        (
            {
                "judge_results": [
                    {
                        "proposal_id": "proposal-quality-note-001",
                        "decision": "recommend_accept",
                        "reason": "First judgment.",
                    },
                    {
                        "proposal_id": "proposal-quality-note-001",
                        "decision": "recommend_reject",
                        "reason": "Duplicate judgment.",
                    },
                ]
            },
            "duplicate",
        ),
        (
            {
                "judge_results": [
                    {
                        "proposal_id": "proposal-quality-note-001",
                        "decision": "recommend_accept",
                        "reason": "Known proposal.",
                    },
                    {
                        "proposal_id": "proposal-quality-note-unknown",
                        "decision": "recommend_accept",
                        "reason": "Unknown proposal.",
                    },
                ]
            },
            "match",
        ),
    ],
)
def test_pipeline_marks_incomplete_or_invalid_judge_results_unavailable(
    tmp_path: Path, judge_payload: dict, expected_error: str
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = InvalidJudgeResultsClient(judge_payload)

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == [
        "proposal-quality-note-001",
        "proposal-quality-note-002",
    ]
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    judge_trace = trace["stages"][-1]
    assert judge_trace["stage"] == "judge"
    assert judge_trace["state"] == "judge_unavailable"
    assert expected_error in judge_trace["error"]
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    assert "judge_unavailable" in proposals_yaml


def test_pipeline_sanitizes_judge_client_error_unavailable_reason(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = JudgeClientErrorAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-hierarchy-symptom-001"]
    trace_text = (package / "llm_review_trace.json").read_text(encoding="utf-8")
    trace = json.loads(trace_text)
    judge_trace = trace["stages"][-1]
    assert judge_trace["stage"] == "judge"
    assert judge_trace["state"] == "judge_unavailable"
    assert "provider 401 while refreshing credentials" in judge_trace["error"]
    assert "[REDACTED]" in judge_trace["error"]
    assert judge_trace["attempt_logs"] == [
        {
            "attempt": 1,
            "state": "client_error",
            "error": (
                "provider 401 while refreshing credentials: "
                "api_key=[REDACTED] token=[REDACTED] "
                "Authorization: Bearer [REDACTED] secret=[REDACTED]; retry later"
            ),
            "output_token_estimate": 0,
        }
    ]

    judge_report = (package / "llm_judge_report.md").read_text(encoding="utf-8")
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    assert "provider 401 while refreshing credentials" in judge_report
    assert "[REDACTED]" in judge_report
    _assert_text_excludes_secrets(trace_text)
    _assert_text_excludes_secrets(judge_report)
    _assert_text_excludes_secrets(proposals_yaml)
    _assert_text_excludes_secrets(approval_queue)


def test_pipeline_marks_judge_unavailable_when_judge_disabled(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = SequencedAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0, allow_llm_judge=False),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-hierarchy-symptom-001"]
    assert len(client.calls) == 5

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    judge_trace = trace["stages"][-1]
    assert judge_trace["stage"] == "judge"
    assert judge_trace["state"] == "judge_unavailable"
    assert judge_trace["attempt_logs"] == []
    assert "judge stage disabled by configuration" in judge_trace["error"]

    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    judge_report = (package / "llm_judge_report.md").read_text(encoding="utf-8")
    assert "judge_unavailable" in approval_queue
    assert "judge_unavailable" in proposals_yaml
    assert "judge_unavailable" in judge_report


def test_pipeline_marks_judge_unavailable_when_judge_context_too_large(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = SequencedAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_context_tokens_per_stage=800),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-hierarchy-symptom-001"]

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    judge_trace = trace["stages"][-1]
    assert judge_trace["stage"] == "judge"
    assert judge_trace["state"] == "judge_unavailable"
    assert "max_context_tokens_per_stage" in judge_trace["error"]

    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    judge_report = (package / "llm_judge_report.md").read_text(encoding="utf-8")
    assert "judge_unavailable" in approval_queue
    assert "judge_unavailable" in proposals_yaml
    assert "judge_unavailable" in judge_report


def test_pipeline_does_not_follow_symlinked_agent_context_dir(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    external_context = tmp_path / "external_context"
    external_context.mkdir()
    external_file = external_context / "explain-context.json"
    external_file.write_text("outside sentinel", encoding="utf-8")
    agent_context_link = package / "agent_context"
    try:
        agent_context_link.symlink_to(external_context, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=SequencedAgentClient(),
        config=LLMAgentPipelineConfig(),
    )

    assert result.stop_reason == "pending_human_review"
    assert external_file.read_text(encoding="utf-8") == "outside sentinel"
    assert agent_context_link.is_dir()
    assert not agent_context_link.is_symlink()
    assert (agent_context_link / "explain-context.json").exists()


def test_run_llm_agent_pipeline_stops_for_more_evidence_without_proposals(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = NoEvidenceAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(),
    )

    assert result.stop_reason == "needs_more_evidence"
    assert result.proposal_ids == []
    assert len(client.calls) == 4
    assert (package / "llm_issue_analysis.md").exists()
    assert (package / "llm_evidence_map.md").exists()
    assert not (package / "llm_repair_plan.md").exists()
    assert not (package / "llm_judge_report.md").exists()
    assert "proposals: []" in (
        package / "proposals.generated.yaml"
    ).read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    assert [stage["stage"] for stage in trace["stages"]] == [
        "explain",
        "infer_branches",
        "locate_evidence",
        "propose",
    ]


def test_pipeline_retries_invalid_stage_output(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = RetryExplainAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=1),
    )

    assert result.stop_reason == "pending_human_review"
    assert len(client.calls) == 7
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    explain_trace = trace["stages"][0]
    assert explain_trace["stage"] == "explain"
    assert explain_trace["attempts"] == 2


def test_pipeline_feeds_validation_errors_into_retry_prompt(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = RetryProposeMetricAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=1),
    )

    assert result.stop_reason == "pending_human_review"
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][3]
    assert propose_trace["stage"] == "propose"
    assert propose_trace["attempts"] == 2
    second_propose_prompt = client.calls[4]["user_prompt"]
    assert (
        "Previous output was rejected: proposal expected_metric_change values must be numbers."
        in second_propose_prompt
    )
    assert (
        'For "expected_metric_change", use finite JSON numbers only; '
        "use {} if no numeric estimate is available."
        in second_propose_prompt
    )
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    assert "proposal-hierarchy-symptom-001" in approval_queue
    assert "proposal-hierarchy-symptom-001" in proposals_yaml


def test_pipeline_retries_with_accumulated_rejection_history(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = RetryProposeTwiceAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=2),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-hierarchy-symptom-001"]
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][3]
    assert propose_trace["stage"] == "propose"
    assert propose_trace["attempts"] == 3
    attempt_logs = propose_trace["attempt_logs"]
    assert len(attempt_logs) == 2
    assert attempt_logs[0]["attempt"] == 1
    assert attempt_logs[0]["state"] == "invalid_llm_output"
    assert (
        attempt_logs[0]["error"]
        == "proposal expected_metric_change values must be numbers"
    )
    assert isinstance(attempt_logs[0]["output_token_estimate"], int)
    assert attempt_logs[0]["output_token_estimate"] > 0
    assert attempt_logs[1]["attempt"] == 2
    assert attempt_logs[1]["state"] == "invalid_llm_output"
    assert (
        attempt_logs[1]["error"]
        == "proposal proposal-hierarchy-symptom-001 evidence is not grounded in deterministic artifacts"
    )
    assert isinstance(attempt_logs[1]["output_token_estimate"], int)
    assert attempt_logs[1]["output_token_estimate"] > 0
    third_propose_prompt = client.calls[5]["user_prompt"]
    assert "Previous rejected attempts:" in third_propose_prompt
    assert (
        "Attempt 1: proposal expected_metric_change values must be numbers"
        in third_propose_prompt
    )
    assert (
        "Attempt 2: proposal proposal-hierarchy-symptom-001 evidence is not grounded in deterministic artifacts"
        in third_propose_prompt
    )
    assert "Return corrected JSON only." in third_propose_prompt
    assert (
        "Do not change evidence/human-approval requirements." in third_propose_prompt
    )


def test_pipeline_records_sanitized_client_error_attempt_logs_and_retry_prompt(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = ClientErrorOnceThenSuccessAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=1),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-hierarchy-symptom-001"]
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    explain_trace = trace["stages"][0]
    assert explain_trace["stage"] == "explain"
    assert explain_trace["state"] == "completed"
    assert explain_trace["attempts"] == 2
    assert explain_trace["attempt_logs"] == [
        {
            "attempt": 1,
            "state": "client_error",
            "error": (
                "provider 401 while refreshing credentials: "
                "api_key=[REDACTED] token=[REDACTED] "
                "Authorization: Bearer [REDACTED] secret=[REDACTED]; retry later"
            ),
            "output_token_estimate": 0,
        }
    ]

    retry_prompt = client.calls[1]["user_prompt"]
    assert "Previous output was rejected" not in retry_prompt
    assert "Return corrected JSON only." not in retry_prompt
    assert "Previous LLM client call failed" in retry_prompt
    assert "provider 401 while refreshing credentials" in retry_prompt
    assert "[REDACTED]" in retry_prompt
    _assert_text_excludes_secrets(retry_prompt)
    _assert_text_excludes_secrets((package / "llm_review_trace.json").read_text())


def test_pipeline_sanitizes_client_error_failure_report(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = AlwaysClientErrorAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=1),
    )

    assert result.stop_reason == "llm_client_error"
    assert result.proposal_ids == []
    trace_text = (package / "llm_review_trace.json").read_text(encoding="utf-8")
    trace = json.loads(trace_text)
    explain_trace = trace["stages"][0]
    assert explain_trace["stage"] == "explain"
    assert explain_trace["state"] == "client_error"
    assert explain_trace["attempts"] == 2
    assert len(explain_trace["attempt_logs"]) == 2
    assert all(
        attempt["state"] == "client_error"
        and attempt["output_token_estimate"] == 0
        and "[REDACTED]" in attempt["error"]
        for attempt in explain_trace["attempt_logs"]
    )

    report = (package / "llm_review_report.md").read_text(encoding="utf-8")
    assert "transport failed" in report
    assert "X-API-Key: [REDACTED]" in report
    assert "session_token=[REDACTED]" in report
    _assert_text_excludes_secrets(trace_text)
    _assert_text_excludes_secrets(report)


def test_pipeline_writes_failure_artifacts_for_missing_required_package_artifact(
    tmp_path: Path,
):
    package = tmp_path / "package"
    package.mkdir()
    client = SequencedAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(),
    )

    assert result.stop_reason == "invalid_kb_package"
    assert result.proposal_ids == []
    assert client.calls == []
    _assert_preflight_failure_artifacts(
        package,
        stop_reason="invalid_kb_package",
        expected_error="snapshots/kg_snapshot.json",
    )


def test_pipeline_writes_failure_artifacts_when_package_dir_missing(tmp_path: Path):
    package = tmp_path / "missing-package"
    client = SequencedAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(),
    )

    assert result.stop_reason == "invalid_kb_package"
    assert result.proposal_ids == []
    assert client.calls == []
    assert package.exists()
    _assert_preflight_failure_artifacts(
        package,
        stop_reason="invalid_kb_package",
        expected_error="snapshots/kg_snapshot.json",
    )


def test_pipeline_writes_failure_artifacts_for_mismatched_workspace(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = SequencedAgentClient()

    result = run_llm_agent_pipeline(
        workspace="other",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(),
    )

    assert result.stop_reason == "invalid_kb_package"
    assert result.proposal_ids == []
    assert client.calls == []
    _assert_preflight_failure_artifacts(
        package,
        stop_reason="invalid_kb_package",
        expected_error="workspace does not match",
    )


def test_pipeline_writes_failure_artifacts_for_invalid_config(tmp_path: Path):
    package = tmp_path / "package"
    package.mkdir()
    client = SequencedAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_context_tokens_per_stage=0),
    )

    assert result.stop_reason == "invalid_config"
    assert result.proposal_ids == []
    assert client.calls == []
    _assert_preflight_failure_artifacts(
        package,
        stop_reason="invalid_config",
        expected_error="max_context_tokens_per_stage",
    )


def test_pipeline_preflight_trace_includes_empty_attempt_logs(tmp_path: Path):
    package = tmp_path / "package"
    package.mkdir()
    client = SequencedAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_context_tokens_per_stage=0),
    )

    assert result.stop_reason == "invalid_config"
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    assert trace["stages"][0]["stage"] == "preflight"
    assert trace["stages"][0]["attempt_logs"] == []


def _write_agent_package(package: Path) -> None:
    snapshot_dir = package / "snapshots"
    snapshot_dir.mkdir(parents=True)
    _write_json(
        snapshot_dir / "kg_snapshot.json",
        {
            "workspace": "demo",
            "generated_at": "2026-06-18T00:00:00Z",
            "source_files": ["guide.md"],
            "nodes": [
                {
                    "id": "flu",
                    "label": "Flu",
                    "entity_type": "Disease",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                },
                {
                    "id": "entity fever",
                    "label": "Fever",
                    "entity_type": "Symptom",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                },
            ],
            "edges": [
                {
                    "id": "flu-fever",
                    "source": "flu",
                    "target": "entity fever",
                    "keywords": "clinical_manifestation",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                }
            ],
            "metadata": {"profile": "medical_kg"},
        },
    )
    _write_json(
        snapshot_dir / "quality_score.json",
        {
            "overall": 74,
            "subscores": {"hierarchy_completeness": 55},
            "metrics": {"missing_hierarchy_branch_count": 1},
            "details": {
                "hierarchy_branches": {
                    "required": [{"key": "symptom", "label": "Symptom"}],
                    "present": [{"key": "disease", "label": "Disease"}],
                    "missing": [{"key": "symptom", "label": "Symptom"}],
                }
            },
            "findings": [
                {
                    "severity": "high",
                    "category": "hierarchy_completeness",
                    "message": "Hierarchy is missing a symptom branch.",
                    "evidence": ["entity:entity fever"],
                    "suggested_fix_type": "add_hierarchy_branch",
                    "requires_approval": True,
                }
            ],
            "critical_blockers": [],
        },
    )
    (package / "kb_context.md").write_text(
        "# KB Context\n\nFever is a symptom mentioned in guide.md.\n",
        encoding="utf-8",
    )
    (package / "accepted_changes.md").write_text(
        "# Accepted Changes\n\n- Preserve symptom entities.\n",
        encoding="utf-8",
    )
    (package / "rejected_changes.md").write_text(
        "# Rejected Changes\n\n- Do not merge symptoms into diseases.\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _assert_text_excludes_secrets(text: str) -> None:
    for secret in SECRET_FRAGMENTS:
        assert secret not in text


def _assert_preflight_failure_artifacts(
    package: Path,
    *,
    stop_reason: str,
    expected_error: str,
) -> None:
    trace_path = package / "llm_review_trace.json"
    report_path = package / "llm_review_report.md"
    proposals_path = package / "proposals.generated.yaml"

    assert trace_path.exists()
    assert report_path.exists()
    assert proposals_path.exists()
    assert trace_path.read_text(encoding="utf-8").endswith("\n")
    assert report_path.read_text(encoding="utf-8").endswith("\n")
    assert proposals_path.read_text(encoding="utf-8").endswith("\n")

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["stop_reason"] == stop_reason
    assert trace["stages"][0]["stage"] == "preflight"
    assert expected_error in trace["stages"][0]["error"]

    report = report_path.read_text(encoding="utf-8")
    assert stop_reason in report
    assert expected_error in report
    assert "proposals: []" in proposals_path.read_text(encoding="utf-8")
