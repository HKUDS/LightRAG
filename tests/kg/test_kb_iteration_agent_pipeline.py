import json
import threading
import time
from pathlib import Path

import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kb_iteration.agent_pipeline import (
    _EvidenceReferenceTokens,
    LLMAgentPipelineConfig,
    MAX_SUBAGENT_ISSUES_PER_TASK,
    _action_candidate_proposals_from_task_packs,
    action_candidate_proposals_from_scan,
    _evidence_references_known_artifact,
    _parse_and_validate_propose_output,
    _run_orchestrated_propose_stage,
    _subagent_propose_system_prompt,
    _task_pack_with_issue_subset,
    _retry_user_prompt,
    _stage_prompt,
    run_llm_agent_pipeline,
)
from lightrag.kb_iteration.agent_outputs import parse_agent_stage_output
from lightrag.kb_iteration.issue_ledger import scan_deterministic_candidates
from lightrag.kb_iteration.proposal_orchestrator import (
    build_llm_residual_task_packs,
    build_proposal_task_packs,
)
from lightrag.kb_iteration.subagent_contracts import role_contract


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


class EmptySubagentProposalClient:
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
        return '{"proposals":[]}'


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


class NonAsciiProposalIdAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3]["proposals"][0]["id"] = "修复-临床表现-高热不退"


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
                "type": "hierarchy_rule_change",
                "target": "lightrag/medical_kg/source_coverage.py",
                "proposed_change": "Add a controlled source coverage rule.",
                "reason": "The same grounded source should be covered by a rule.",
                "evidence": [
                    "source_id: chunk-1; file_path: guide.md; item_id: entity fever"
                ],
                "confidence": 0.73,
                "risk": "low",
                "requires_approval": True,
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


class FabricatedCandidateKGExpansionPayloadAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3] = {
            "proposals": [
                {
                    "id": "proposal-candidate-kg-expansion-001",
                    "type": "candidate_kg_expansion",
                    "target": "kg:candidate:flu-manifestations",
                    "proposed_change": "Queue candidate symptom nodes and edges.",
                    "reason": "Grounded evidence suggests more review is needed.",
                    "evidence": [
                        "source_id: chunk-1; file_path: guide.md; item_id: entity fever"
                    ],
                    "confidence": 0.72,
                    "risk": "medium",
                    "requires_approval": True,
                    "expected_metric_change": {},
                    "action_payload": {
                        "candidate_nodes": [
                            {
                                "id": "symptom:invented",
                                "label": "Invented symptom",
                                "entity_type": "Symptom",
                            }
                        ],
                        "candidate_edges": [
                            {
                                "source": "flu",
                                "source_type": "Disease",
                                "target": "symptom:invented",
                                "target_type": "Symptom",
                                "keywords": "has_manifestation",
                                "source_id": "fake-source",
                                "file_path": "fake.md",
                                "qualifiers": {},
                            }
                        ],
                        "source_id": "fake-source",
                        "file_path": "fake.md",
                        "evidence_quote": "A fabricated quote not present in the package.",
                        "why_not_existing": "No equivalent candidate exists.",
                    },
                }
            ]
        }


class ValidCandidateKGExpansionPayloadAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3] = {
            "proposals": [
                {
                    "id": "proposal-candidate-kg-expansion-001",
                    "type": "candidate_kg_expansion",
                    "target": "kg:candidate:flu-manifestations",
                    "proposed_change": "Queue candidate symptom nodes and edges.",
                    "reason": "Grounded evidence suggests more review is needed.",
                    "evidence": [
                        "source_id: chunk-1; file_path: guide.md; item_id: entity fever"
                    ],
                    "confidence": 0.72,
                    "risk": "medium",
                    "requires_approval": True,
                    "expected_metric_change": {},
                    "action_payload": {
                        "candidate_nodes": [
                            {
                                "id": "symptom:body-aches",
                                "label": "Body aches",
                                "entity_type": "Symptom",
                            }
                        ],
                        "candidate_edges": [
                            {
                                "source": "flu",
                                "source_type": "Disease",
                                "target": "symptom:body-aches",
                                "target_type": "Symptom",
                                "keywords": "has_manifestation",
                                "source_id": "chunk-1",
                                "file_path": "guide.md",
                                "qualifiers": {},
                            }
                        ],
                        "source_id": "chunk-1",
                        "file_path": "guide.md",
                        "evidence_quote": "Fever is a symptom mentioned in guide.md.",
                        "why_not_existing": "No equivalent candidate exists.",
                    },
                }
            ]
        }
        self.outputs[4] = {
            "repair_plan": [
                {
                    "rank": 1,
                    "proposal_id": "proposal-candidate-kg-expansion-001",
                    "priority": "medium",
                    "risk": "medium",
                    "reason": "Candidate expansion remains human-reviewed.",
                    "human_checks": ["Verify source evidence before accepting."],
                }
            ]
        }
        self.outputs[5] = {
            "judge_results": [
                {
                    "proposal_id": "proposal-candidate-kg-expansion-001",
                    "decision": "needs_human",
                    "reason": "Candidate KG expansion requires human approval.",
                }
            ]
        }


class LLMOnlyCandidateQuoteAgentClient(ValidCandidateKGExpansionPayloadAgentClient):
    def __init__(self) -> None:
        super().__init__()
        llm_only_quote = "The LLM-only note says phantom aches need a KG candidate."
        self.outputs[0]["issue_explanations"][0]["explanation"] = llm_only_quote
        self.outputs[3]["proposals"][0]["action_payload"][
            "evidence_quote"
        ] = llm_only_quote


class GenericCandidateQuoteAgentClient(ValidCandidateKGExpansionPayloadAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3]["proposals"][0]["action_payload"]["evidence_quote"] = "flu"


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


class StaleCurrentKeywordsAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3] = {
            "proposals": [
                {
                    "id": "proposal-stale-relation-001",
                    "type": "medical_relation_schema_migration",
                    "target": "kg:relation:flu-fever",
                    "proposed_change": "Normalize a relation from an outdated issue.",
                    "reason": "The LLM is using an older issue snapshot.",
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
                        "current_keywords": "schema_relation",
                        "new_source": "flu",
                        "new_target": "entity fever",
                        "new_keywords": "has_manifestation",
                        "qualifiers": {},
                    },
                }
            ]
        }


class NonCandidateActionPayloadAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3] = {
            "proposals": [
                {
                    "id": "proposal-medical-schema-noncandidate-001",
                    "type": "medical_relation_schema_migration",
                    "target": "kg:relation:flu-fever",
                    "proposed_change": "Normalize the relation direction.",
                    "reason": "The relation should use a deterministic candidate payload.",
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
                        "expected_source": "entity fever",
                        "expected_target": "flu",
                        "current_keywords": "clinical_manifestation",
                        "new_source": "flu",
                        "new_target": "entity fever",
                        "new_keywords": "causative_agent",
                        "qualifiers": {},
                    },
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


class UnsafeStatefulNoProposalAgentClient:
    model = "agent-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []
        self._lock = threading.Lock()
        self._active_subagents = 0
        self.max_active_subagents = 0

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        with self._lock:
            self.calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
        if "Proposal Orchestrator Subagent" in system_prompt:
            with self._lock:
                self._active_subagents += 1
                self.max_active_subagents = max(
                    self.max_active_subagents, self._active_subagents
                )
            time.sleep(0.05)
            with self._lock:
                self._active_subagents -= 1
            return json.dumps({"proposals": []}, ensure_ascii=False)
        if "Explain Agent" in system_prompt:
            return json.dumps({"issue_explanations": []}, ensure_ascii=False)
        if "Infer Branches Agent" in system_prompt:
            return json.dumps(
                {
                    "required": [],
                    "present": [],
                    "missing": [],
                    "missing_branches": [],
                },
                ensure_ascii=False,
            )
        if "Locate Evidence Agent" in system_prompt:
            return json.dumps({"evidence_map": []}, ensure_ascii=False)
        raise AssertionError(f"unexpected prompt: {system_prompt}")


class ParallelNoProposalAgentClient:
    model = "agent-model"
    supports_parallel_subagent_calls = True

    def __init__(self, expected_parallel_subagents: int = 2) -> None:
        self.calls: list[dict[str, str]] = []
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(expected_parallel_subagents)
        self._active_subagents = 0
        self.max_active_subagents = 0

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        with self._lock:
            self.calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
        if "Proposal Orchestrator Subagent" in system_prompt:
            with self._lock:
                self._active_subagents += 1
                self.max_active_subagents = max(
                    self.max_active_subagents, self._active_subagents
                )
            try:
                self._barrier.wait(timeout=2)
            except threading.BrokenBarrierError:
                pass
            finally:
                with self._lock:
                    self._active_subagents -= 1
            return json.dumps({"proposals": []}, ensure_ascii=False)
        if "Explain Agent" in system_prompt:
            return json.dumps({"issue_explanations": []}, ensure_ascii=False)
        if "Infer Branches Agent" in system_prompt:
            return json.dumps(
                {
                    "required": [],
                    "present": [],
                    "missing": [],
                    "missing_branches": [],
                },
                ensure_ascii=False,
            )
        if "Locate Evidence Agent" in system_prompt:
            return json.dumps({"evidence_map": []}, ensure_ascii=False)
        raise AssertionError(f"unexpected prompt: {system_prompt}")


class CaptureNoProposalAgentClient:
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
        if "Proposal Orchestrator Subagent" in system_prompt:
            return json.dumps({"proposals": []}, ensure_ascii=False)
        if "Explain Agent" in system_prompt:
            return json.dumps({"issue_explanations": []}, ensure_ascii=False)
        if "Infer Branches Agent" in system_prompt:
            return json.dumps(
                {
                    "required": [],
                    "present": [],
                    "missing": [],
                    "missing_branches": [],
                },
                ensure_ascii=False,
            )
        if "Locate Evidence Agent" in system_prompt:
            return json.dumps({"evidence_map": []}, ensure_ascii=False)
        if "Rank Repairs Agent" in system_prompt:
            return json.dumps({"repair_plan": []}, ensure_ascii=False)
        if "Judge Agent" in system_prompt:
            return json.dumps(
                {
                    "judge_results": [
                        {
                            "proposal_id": proposal_id,
                            "decision": "needs_human",
                            "reason": "Deterministic candidate requires human approval.",
                        }
                        for proposal_id in _proposal_ids_from_prompt(user_prompt)
                    ]
                },
                ensure_ascii=False,
            )
        raise AssertionError(f"unexpected prompt: {system_prompt}")


class LargeProposeArtifactContextClient(CaptureNoProposalAgentClient):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        if "Locate Evidence Agent" in system_prompt:
            self.calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            return json.dumps(
                {
                    "evidence_map": [
                        {
                            "issue_id": f"issue-{index}",
                            "target": "edge:flu-fever",
                            "confidence": 0.9,
                            "missing_evidence": [],
                            "supporting_items": [
                                {
                                    "item_type": "relation",
                                    "item_id": "flu-fever",
                                    "source_id": "chunk-1",
                                    "file_path": "guide.md",
                                    "evidence_status": "grounded",
                                    "quote": "x" * 1600,
                                }
                            ],
                        }
                        for index in range(30)
                    ]
                },
                ensure_ascii=False,
            )
        return super().complete(system_prompt=system_prompt, user_prompt=user_prompt)


class InvalidJsonSubagentClient(CaptureNoProposalAgentClient):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if "Proposal Orchestrator Subagent" in system_prompt:
            return "{bad json"
        if "Explain Agent" in system_prompt:
            return json.dumps({"issue_explanations": []}, ensure_ascii=False)
        if "Infer Branches Agent" in system_prompt:
            return json.dumps(
                {
                    "required": [],
                    "present": [],
                    "missing": [],
                    "missing_branches": [],
                },
                ensure_ascii=False,
            )
        if "Locate Evidence Agent" in system_prompt:
            return json.dumps({"evidence_map": []}, ensure_ascii=False)
        raise AssertionError(f"unexpected prompt: {system_prompt}")


class InvalidJsonSubagentRankOkClient(InvalidJsonSubagentClient):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        if "Rank Repairs Agent" in system_prompt:
            self.calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            return json.dumps({"repair_plan": []}, ensure_ascii=False)
        return super().complete(system_prompt=system_prompt, user_prompt=user_prompt)


class PartialSubagentFailureAgentClient:
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
        if "Proposal Orchestrator Subagent" in system_prompt:
            task_id = json.loads(user_prompt)["task_id"]
            if task_id == "schema_repair-001":
                return json.dumps({"proposals": [{"id": ""}]}, ensure_ascii=False)
            return json.dumps(
                {
                    "proposals": [
                        {
                            "id": "proposal-quality-note-001",
                            "type": "quality_report_note",
                            "target": "quality_report.md",
                            "proposed_change": "Record surviving subagent proposal.",
                            "reason": "One role found a grounded note.",
                            "evidence": [
                                "source_id: chunk-1; file_path: guide.md; item_id: flu-fever"
                            ],
                            "confidence": 0.75,
                            "risk": "low",
                            "requires_approval": True,
                            "expected_metric_change": {},
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "Explain Agent" in system_prompt:
            return json.dumps({"issue_explanations": []}, ensure_ascii=False)
        if "Infer Branches Agent" in system_prompt:
            return json.dumps(
                {
                    "required": [],
                    "present": [],
                    "missing": [],
                    "missing_branches": [],
                },
                ensure_ascii=False,
            )
        if "Locate Evidence Agent" in system_prompt:
            return json.dumps({"evidence_map": []}, ensure_ascii=False)
        if "Rank Repairs Agent" in system_prompt:
            return json.dumps(
                {
                    "repair_plan": [
                        {
                            "rank": 1,
                            "proposal_id": "proposal-quality-note-001",
                            "priority": "medium",
                            "risk": "low",
                            "reason": "Keep valid proposal from surviving subagent.",
                            "human_checks": [],
                        }
                    ]
                },
                ensure_ascii=False,
            )
        raise AssertionError(f"unexpected prompt: {system_prompt}")


class MergeDropsProposalsAgentClient:
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
        if "Proposal Orchestrator Subagent" in system_prompt:
            return json.dumps(
                {
                    "proposals": [
                        _quality_report_note_proposal(
                            "proposal-merge-keep",
                            confidence=0.95,
                            proposed_change="Record merge-visible proposal.",
                        ),
                        _quality_report_note_proposal(
                            "proposal-merge-keep",
                            confidence=0.95,
                            proposed_change="Record merge-visible proposal.",
                        ),
                        _quality_report_note_proposal(
                            "proposal-merge-max-drop",
                            confidence=0.75,
                            proposed_change="Record lower-priority merge proposal.",
                        ),
                    ]
                },
                ensure_ascii=False,
            )
        if "Explain Agent" in system_prompt:
            return json.dumps({"issue_explanations": []}, ensure_ascii=False)
        if "Infer Branches Agent" in system_prompt:
            return json.dumps(
                {
                    "required": [],
                    "present": [],
                    "missing": [],
                    "missing_branches": [],
                },
                ensure_ascii=False,
            )
        if "Locate Evidence Agent" in system_prompt:
            return json.dumps({"evidence_map": []}, ensure_ascii=False)
        if "Rank Repairs Agent" in system_prompt:
            return json.dumps(
                {
                    "repair_plan": [
                        {
                            "rank": 1,
                            "proposal_id": "proposal-merge-keep",
                            "priority": "medium",
                            "risk": "low",
                            "reason": "Keep the highest confidence merge proposal.",
                            "human_checks": [],
                        }
                    ]
                },
                ensure_ascii=False,
            )
        raise AssertionError(f"unexpected prompt: {system_prompt}")


class RetryEachSubagentOnceClient:
    model = "agent-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []
        self._lock = threading.Lock()
        self._attempts_by_task: dict[str, int] = {}

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        with self._lock:
            self.calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
        if "Proposal Orchestrator Subagent" in system_prompt:
            prompt, _ = json.JSONDecoder().raw_decode(user_prompt)
            task_id = prompt["task_id"]
            role = prompt["task_pack"]["role"]
            with self._lock:
                attempt = self._attempts_by_task.get(task_id, 0) + 1
                self._attempts_by_task[task_id] = attempt
            if attempt == 1:
                return json.dumps(
                    {
                        "proposals": [
                            {
                                "id": f"proposal-{task_id}-invalid-metric",
                                "type": "quality_report_note",
                                "target": "quality_report.md",
                                "proposed_change": (
                                    f"Record {role} review context in the quality report."
                                ),
                                "reason": "First attempt has a non-numeric metric.",
                                "evidence": [
                                    "source_id: chunk-1; file_path: guide.md; item_id: entity fever"
                                ],
                                "confidence": 0.7,
                                "risk": "low",
                                "requires_approval": False,
                                "expected_metric_change": {
                                    "hierarchy_completeness": "about 1"
                                },
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            return json.dumps({"proposals": []}, ensure_ascii=False)
        if "Explain Agent" in system_prompt:
            return json.dumps({"issue_explanations": []}, ensure_ascii=False)
        if "Infer Branches Agent" in system_prompt:
            return json.dumps(
                {
                    "required": [],
                    "present": [],
                    "missing": [],
                    "missing_branches": [],
                },
                ensure_ascii=False,
            )
        if "Locate Evidence Agent" in system_prompt:
            return json.dumps({"evidence_map": []}, ensure_ascii=False)
        raise AssertionError(f"unexpected prompt: {system_prompt}")


def _proposal_ids_from_prompt(user_prompt: str) -> list[str]:
    try:
        payload = json.loads(user_prompt)
    except json.JSONDecodeError:
        return []

    proposal_ids: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, dict):
            proposal_id = value.get("id")
            if (
                isinstance(proposal_id, str)
                and "proposed_change" in value
                and proposal_id not in proposal_ids
            ):
                proposal_ids.append(proposal_id)
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(payload)
    return proposal_ids


class LargePreviousOutputAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[0]["issue_explanations"][0]["explanation"] = "x" * 30000


class LargeRankContextAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3]["proposals"] = [
            {
                **self.outputs[3]["proposals"][0],
                "id": f"proposal-rank-context-{index}",
                "target": f"hierarchy:rank-context-{index}",
                "proposed_change": f"Record rank context {index} " + "x" * 1000,
                "reason": "x" * 1000,
            }
            for index in range(20)
        ]


class LargeAuxiliaryContextAgentClient(LargeRankContextAgentClient):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        if "Rank Repairs Agent" in system_prompt:
            self.calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            return json.dumps(
                {
                    "repair_plan": [
                        {
                            "rank": index + 1,
                            "proposal_id": proposal_id,
                            "priority": "high",
                            "risk": "medium",
                            "reason": "Rank compact proposal context.",
                            "human_checks": ["Review bounded action and evidence."],
                        }
                        for index, proposal_id in enumerate(
                            _proposal_ids_from_prompt(user_prompt)
                        )
                    ]
                },
                ensure_ascii=False,
            )
        if "KG proposal judge" in system_prompt:
            self.calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            return json.dumps(
                {
                    "judge_results": [
                        {
                            "proposal_id": proposal_id,
                            "decision": "needs_human",
                            "reason": "Large proposal was judged from compact context.",
                        }
                        for proposal_id in _proposal_ids_from_prompt(user_prompt)
                    ]
                },
                ensure_ascii=False,
            )
        return super().complete(system_prompt=system_prompt, user_prompt=user_prompt)


class VeryLargeAuxiliaryContextAgentClient(LargeAuxiliaryContextAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[3]["proposals"] = [
            {
                **self.outputs[3]["proposals"][0],
                "id": f"proposal-rank-context-{index}",
                "target": f"hierarchy:rank-context-{index}",
                "proposed_change": f"Record rank context {index} " + "x" * 1000,
                "reason": "x" * 1000,
            }
            for index in range(120)
        ]


class LargeJudgeContextAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[4]["repair_plan"] = [
            {
                "rank": index + 1,
                "proposal_id": "proposal-hierarchy-symptom-001",
                "priority": "high",
                "risk": "medium",
                "reason": f"{'x' * 12000}{index}",
                "human_checks": ["y" * 12000, "z" * 12000],
                "extra_context": "w" * 12000,
            }
            for index in range(10)
        ]


def test_propose_prompt_requires_action_payload_for_medical_migrations():
    prompt = _stage_prompt("propose", "clinical_guideline_zh")

    assert "action_payload" in prompt
    assert "medical_relation_schema_migration" in prompt
    assert "replace_relation" in prompt
    assert "Do not invent medical facts" in prompt
    assert "流感临床表现" in prompt
    assert "肺炎链球菌" in prompt
    assert "supports_or_refutes" in prompt
    assert "泛化“流感病毒”" in prompt
    assert "甲型流感病毒" in prompt
    assert "必须复制该 `action_payload`" in prompt
    assert "扎那米韦" in prompt
    assert "低钾血症" in prompt
    assert "乙型流感病毒 -> is_a -> 流感病毒" in prompt
    assert "流感病毒 -> is_a -> 乙型流感病毒" in prompt


def test_judge_prompt_checks_medical_schema_edge_case_constraints():
    prompt = _stage_prompt("judge", "clinical_guideline_zh")

    assert "流感临床表现" in prompt
    assert "肺炎链球菌" in prompt
    assert "supports_or_refutes" in prompt
    assert "扎那米韦" in prompt
    assert "低钾血症" in prompt


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
        package / "improvement_backlog.md"
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


def test_default_pipeline_uses_orchestrated_subagent_propose_stage(
    tmp_path: Path,
) -> None:
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
    assert len(client.calls) == 6
    assert "Explain Agent" in client.calls[0]["system_prompt"]
    assert "Infer Branches Agent" in client.calls[1]["system_prompt"]
    assert "Locate Evidence Agent" in client.calls[2]["system_prompt"]
    assert "Rank Repairs Agent" in client.calls[4]["system_prompt"]
    assert "KG proposal judge" in client.calls[5]["system_prompt"]

    propose_call = client.calls[3]
    assert propose_call["system_prompt"] != _stage_prompt("propose", "medical_kg")
    assert "Proposal Orchestrator Subagent" in propose_call["system_prompt"]
    assert '"task_pack"' in propose_call["user_prompt"]
    assert '"role": "schema_repair"' in propose_call["user_prompt"]
    assert '"previous_outputs"' in propose_call["user_prompt"]

    task_packs_path = package / "proposal_task_packs.json"
    subagent_output_path = package / "subagent_outputs" / "schema_repair-001.json"
    merge_report_path = package / "proposal_merge_report.md"
    assert task_packs_path.exists()
    assert subagent_output_path.exists()
    assert merge_report_path.exists()

    task_packs = json.loads(task_packs_path.read_text(encoding="utf-8"))
    assert task_packs[0]["task_id"] == "schema_repair-001"
    assert task_packs[0]["role"] == "schema_repair"
    subagent_output = json.loads(subagent_output_path.read_text(encoding="utf-8"))
    assert subagent_output["task_id"] == "schema_repair-001"
    assert "raw_output" in subagent_output
    assert subagent_output["parsed_output"]["proposals"][0]["id"] == (
        "proposal-hierarchy-symptom-001"
    )

    merge_report = merge_report_path.read_text(encoding="utf-8")
    assert "proposal-hierarchy-symptom-001" in merge_report
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    assert "proposal-hierarchy-symptom-001" in proposals_yaml


def test_default_pipeline_runs_orchestrated_propose_for_hierarchy_only_package(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package, include_medical_schema_issues=False)
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
    propose_call = client.calls[3]
    assert "Proposal Orchestrator Subagent" in propose_call["system_prompt"]
    prompt = json.loads(propose_call["user_prompt"])
    assert prompt["task_pack"]["task_id"] == "general-001"
    assert prompt["task_pack"]["role"] == "general"
    assert prompt["task_pack"]["issues"][0]["category"] == "hierarchy_completeness"

    task_packs = json.loads(
        (package / "proposal_task_packs.json").read_text(encoding="utf-8")
    )
    assert task_packs[0]["role"] == "general"
    assert (package / "subagent_outputs" / "general-001.json").exists()
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    assert "proposal-hierarchy-symptom-001" in proposals_yaml


def test_pipeline_serializes_subagents_for_clients_without_parallel_contract(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"].append(
        {
            "issue_kind": "schema_review",
            "edge_id": "flu-treatment",
            "source": "flu",
            "target": "oseltamivir",
            "keywords": "recommended_treatment",
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "risk": "medium",
            "suggested_action": "review_relation_schema",
            "candidate_predicates": ["has_indication"],
        }
    )
    _write_json(quality_path, quality)
    client = UnsafeStatefulNoProposalAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_subagent_tasks=2,
            max_parallel_subagents=2,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "needs_more_evidence"
    assert client.max_active_subagents == 1
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][3]
    assert propose_trace["subagent_parallel_fallback"] == "client_not_thread_safe"
    assert [pack["task_id"] for pack in json.loads(
        (package / "proposal_task_packs.json").read_text(encoding="utf-8")
    )] == [
        "treatment-001",
        "schema_repair-001",
    ]


def test_pipeline_runs_subagent_tasks_in_parallel_when_configured(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"].append(
        {
            "issue_kind": "schema_review",
            "edge_id": "flu-treatment",
            "source": "flu",
            "target": "oseltamivir",
            "keywords": "recommended_treatment",
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "risk": "medium",
            "suggested_action": "review_relation_schema",
            "candidate_predicates": ["has_indication"],
        }
    )
    _write_json(quality_path, quality)
    client = ParallelNoProposalAgentClient(expected_parallel_subagents=2)

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_subagent_tasks=2,
            max_parallel_subagents=2,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "needs_more_evidence"
    assert client.max_active_subagents == 2
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][3]
    assert "subagent_parallel_fallback" not in propose_trace
    task_packs = json.loads(
        (package / "proposal_task_packs.json").read_text(encoding="utf-8")
    )
    assert [pack["task_id"] for pack in task_packs] == [
        "treatment-001",
        "schema_repair-001",
    ]


def test_task_pack_issue_subset_keeps_evidence_spans_only_allowlist():
    task_pack = {
        "role": "evidence_grounding",
        "issues": [
            {
                "issue_ref": "issue-a",
                "issue_family": "treatment",
                "evidence_spans": [
                    {
                        "source_id": "chunk-a",
                        "file_path": "guide-a.md",
                        "evidence_quote": "Grounded quote for issue A.",
                    }
                ],
            },
            {
                "issue_ref": "issue-b",
                "issue_family": "treatment",
                "evidence_spans": [
                    {
                        "source_id": "chunk-b",
                        "file_path": "guide-b.md",
                        "evidence_quote": "Grounded quote for issue B.",
                    }
                ],
            },
        ],
        "allowed_evidence_spans": [
            {
                "source_id": "chunk-a",
                "file_path": "guide-a.md",
                "evidence_quote": "Grounded quote for issue A.",
            },
            {
                "source_id": "chunk-b",
                "file_path": "guide-b.md",
                "evidence_quote": "Grounded quote for issue B.",
            },
        ],
    }

    subset = _task_pack_with_issue_subset(
        task_pack,
        [task_pack["issues"][1]],
        require_candidate_evidence_allowlist=True,
    )

    assert subset["allowed_evidence_spans"] == [
        {
            "source_id": "chunk-b",
            "file_path": "guide-b.md",
            "evidence_quote": "Grounded quote for issue B.",
        }
    ]
    assert subset["execution_mode"] == "grounded_expansion"
    assert subset["block_reason"] == ""


def test_pipeline_counts_total_subagent_attempts_and_labels_retries(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"].append(
        {
            "issue_kind": "schema_review",
            "edge_id": "flu-treatment",
            "source": "flu",
            "target": "oseltamivir",
            "keywords": "recommended_treatment",
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "risk": "medium",
            "suggested_action": "review_relation_schema",
            "candidate_predicates": ["has_indication"],
        }
    )
    _write_json(quality_path, quality)
    client = RetryEachSubagentOnceClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_stage_retries=1,
            max_subagent_tasks=2,
            max_parallel_subagents=2,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "needs_more_evidence"
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][3]
    assert propose_trace["stage"] == "propose"
    assert propose_trace["attempts"] == 4
    assert propose_trace["subagent_task_count"] == 2
    assert propose_trace["attempt_logs"] == [
        {
            "task_id": "schema_repair-001",
            "role": "schema_repair",
            "attempt": 1,
            "state": "invalid_llm_output",
            "error": "proposal expected_metric_change values must be numbers",
            "error_code": "EXPECTED_METRIC_CHANGE_INVALID",
            "output_token_estimate": propose_trace["attempt_logs"][0][
                "output_token_estimate"
            ],
        },
        {
            "task_id": "treatment-001",
            "role": "treatment",
            "attempt": 1,
            "state": "invalid_llm_output",
            "error": "proposal expected_metric_change values must be numbers",
            "error_code": "EXPECTED_METRIC_CHANGE_INVALID",
            "output_token_estimate": propose_trace["attempt_logs"][1][
                "output_token_estimate"
            ],
        },
    ]
    assert all(
        attempt["output_token_estimate"] > 0
        for attempt in propose_trace["attempt_logs"]
    )


def test_pipeline_merges_successful_subagents_when_one_subagent_fails(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"].append(
        {
            "issue_kind": "schema_review",
            "edge_id": "flu-treatment",
            "source": "flu",
            "target": "oseltamivir",
            "keywords": "recommended_treatment",
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "risk": "medium",
            "suggested_action": "review_relation_schema",
            "candidate_predicates": ["has_indication"],
        }
    )
    _write_json(quality_path, quality)
    client = PartialSubagentFailureAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            max_subagent_tasks=2,
            allow_llm_judge=False,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-quality-note-001"]
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    merge_report = (package / "proposal_merge_report.md").read_text(encoding="utf-8")
    assert "proposal-quality-note-001" in approval_queue
    assert "proposal-quality-note-001" in merge_report
    assert "- schema_repair-001: invalid_llm_output" in merge_report

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][3]
    assert propose_trace["stage"] == "propose"
    assert propose_trace["state"] == "completed"
    assert propose_trace["dropped_subagent_count"] == 1
    assert propose_trace["dropped_proposal_count"] == 1
    assert propose_trace["dropped_proposal_reasons"] == {
        "invalid_llm_output": 1
    }


def test_pipeline_trace_counts_selected_and_dropped_merge_proposals(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    client = MergeDropsProposalsAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            allow_llm_judge=False,
            max_proposals_per_run=1,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-merge-keep"]
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][3]
    assert propose_trace["stage"] == "propose"
    assert propose_trace["selected_proposal_count"] == 1
    assert propose_trace["dropped_proposal_count"] == 2
    assert propose_trace["dropped_proposal_reasons"] == {
        "duplicate_proposal_id": 1,
        "max_proposals": 1,
    }


def test_pipeline_stops_when_subagent_prompt_exceeds_context_budget(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        {
            "issue_kind": "schema_review",
            "edge_id": "flu-fever",
            "source": "flu",
            "target": "entity fever",
            "keywords": "clinical_manifestation",
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "risk": "medium",
            "suggested_action": "review_relation_schema",
            "candidate_predicates": ["schema_relation"],
            "guidance": "x" * 30000,
        }
    ]
    _write_json(quality_path, quality)
    client = CaptureNoProposalAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_context_tokens_per_stage=6000,
            allow_llm_judge=False,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "context_too_large"
    assert len(client.calls) == 3
    assert all(
        "Proposal Orchestrator Subagent" not in call["system_prompt"]
        for call in client.calls
    )
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = next(stage for stage in trace["stages"] if stage["stage"] == "propose")
    assert propose_trace["stage"] == "propose"
    assert propose_trace["state"] == "context_too_large"
    assert propose_trace["attempts"] == 0
    assert propose_trace["subagent_task_count"] == 1
    assert propose_trace["input_token_estimate"] > 6000
    assert "subagent prompt" in propose_trace["error"]
    assert "max_context_tokens_per_stage" in propose_trace["error"]
    output_path = package / "subagent_outputs" / "clinical_modeling-001.json"
    index_path = package / "subagent_outputs" / "index.json"
    merge_report_path = package / "proposal_merge_report.md"
    assert output_path.exists()
    assert index_path.exists()
    assert merge_report_path.exists()
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["task_id"] == "clinical_modeling-001"
    assert output["error"].startswith("subagent prompt input_token_estimate exceeds")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index == [
        {
            "task_id": "clinical_modeling-001",
            "role": "clinical_modeling",
            "proposal_ids": [],
            "path": "subagent_outputs/clinical_modeling-001.json",
            "state": "context_too_large",
        }
    ]
    merge_report = merge_report_path.read_text(encoding="utf-8")
    assert "- clinical_modeling-001: context_too_large" in merge_report


def test_pipeline_allows_large_propose_artifact_context_when_subagent_prompts_fit(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    client = LargeProposeArtifactContextClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_context_tokens_per_stage=6000,
            allow_llm_judge=False,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "needs_more_evidence"
    subagent_calls = [
        call
        for call in client.calls
        if "Proposal Orchestrator Subagent" in call["system_prompt"]
    ]
    assert subagent_calls
    assert all(len(call["user_prompt"]) // 4 <= 6000 for call in subagent_calls)

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = next(stage for stage in trace["stages"] if stage["stage"] == "propose")
    assert propose_trace["stage"] == "propose"
    assert propose_trace["state"] != "context_too_large"
    assert propose_trace["context_artifact_token_estimate"] > 6000
    assert propose_trace["input_token_estimate"] <= 6000


def test_pipeline_splits_large_proposal_task_packs_before_subagent_budget(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        {
            "issue_kind": "reverse_clinical_manifestation",
            "edge_id": f"flu-symptom-{index}",
            "source": "flu",
            "target": f"symptom-{index}",
            "keywords": "clinical_manifestation",
            "source_id": f"chunk-{index % 5}",
            "file_path": "guide.md",
            "risk": "medium",
            "suggested_action": "replace_relation",
            "candidate_predicates": ["has_manifestation"],
            "medical_subcase": "Symptom should be modeled as disease manifestation.",
            "guidance": "Reverse disease-as-symptom relation direction.",
            "qualifiers": {"clinical_setting": "outpatient", "certainty": "reported"},
        }
        for index in range(80)
    ]
    _write_json(quality_path, quality)
    client = CaptureNoProposalAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_context_tokens_per_stage=12000),
        profile="medical_kg",
    )

    assert result.stop_reason == "needs_more_evidence"
    subagent_calls = [
        call
        for call in client.calls
        if "Proposal Orchestrator Subagent" in call["system_prompt"]
    ]
    assert len(subagent_calls) > 1
    assert all(len(call["user_prompt"]) // 4 <= 12000 for call in subagent_calls)

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = next(stage for stage in trace["stages"] if stage["stage"] == "propose")
    assert propose_trace["stage"] == "propose"
    assert propose_trace["state"] != "context_too_large"
    assert propose_trace["subagent_task_count"] == len(subagent_calls)

    task_packs = json.loads(
        (package / "proposal_task_packs.json").read_text(encoding="utf-8")
    )
    assert len(task_packs) == len(subagent_calls)
    assert len(task_packs) > 1
    assert all(len(pack["issues"]) < 50 for pack in task_packs)


def test_pipeline_preserves_issue_family_stratification_when_fitting_splits(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = _stratified_schema_issues(
        count_per_family=50
    )
    _write_json(quality_path, quality)
    client = CaptureNoProposalAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_subagent_tasks=5,
            max_context_tokens_per_stage=12000,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "needs_more_evidence"
    task_packs = json.loads(
        (package / "proposal_task_packs.json").read_text(encoding="utf-8")
    )
    represented_families = [str(pack["issue_family"]) for pack in task_packs]
    assert represented_families == [
        "direction",
        "treatment",
        "diagnosis",
        "prevention",
        "multi_predicate_split",
    ]
    assert all(
        len(pack["issues"]) == MAX_SUBAGENT_ISSUES_PER_TASK
        for pack in task_packs
    )

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert propose_trace["subagent_issue_family_counts"] == {
        "direction": MAX_SUBAGENT_ISSUES_PER_TASK,
        "treatment": MAX_SUBAGENT_ISSUES_PER_TASK,
        "diagnosis": MAX_SUBAGENT_ISSUES_PER_TASK,
        "prevention": MAX_SUBAGENT_ISSUES_PER_TASK,
        "multi_predicate_split": MAX_SUBAGENT_ISSUES_PER_TASK,
    }


def test_pipeline_trace_records_subagent_issue_family_omissions(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        {
            "issue_kind": "reverse_clinical_manifestation",
            "edge_id": f"flu-symptom-{index}",
            "source": "flu",
            "target": f"symptom-{index}",
            "keywords": "needs direction review",
            "source_id": f"chunk-{index % 5}",
            "file_path": "guide.md",
            "risk": "medium",
            "suggested_action": "replace_relation",
            "candidate_predicates": ["direction_marker"],
        }
        for index in range(60)
    ]
    _write_json(quality_path, quality)
    client = CaptureNoProposalAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_subagent_tasks=1,
            max_context_tokens_per_stage=12000,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "needs_more_evidence"
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert propose_trace["stage"] == "propose"
    assert propose_trace["raw_task_count"] == 2
    assert propose_trace["raw_issue_count"] == 60
    assert propose_trace["fitted_task_count"] == 1
    assert propose_trace["fitted_issue_count"] == MAX_SUBAGENT_ISSUES_PER_TASK
    assert propose_trace["subagent_issue_family_counts"] == {
        "direction": MAX_SUBAGENT_ISSUES_PER_TASK
    }
    assert propose_trace["subagent_omitted_issue_family_counts"] == {
        "direction": 60 - MAX_SUBAGENT_ISSUES_PER_TASK
    }

    task_packs = json.loads(
        (package / "proposal_task_packs.json").read_text(encoding="utf-8")
    )
    assert task_packs[0]["issue_family"] == "direction"
    assert task_packs[0]["omitted_issue_count"] == 60 - MAX_SUBAGENT_ISSUES_PER_TASK


def test_pipeline_trace_counts_raw_fitted_and_invalid_subagent_outputs(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    total_issues = 12
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        _agent_schema_issue(
            f"edge-direction-{index}",
            issue_kind="reverse_clinical_manifestation",
            keywords="needs direction review",
            candidate_predicates=["has_manifestation"],
        )
        for index in range(total_issues)
    ]
    _write_json(quality_path, quality)
    client = InvalidJsonSubagentRankOkClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            max_subagent_tasks=2,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "invalid_llm_output"
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert propose_trace["stage"] == "propose"
    assert propose_trace["raw_task_count"] == 1
    assert propose_trace["raw_issue_count"] == total_issues
    assert propose_trace["fitted_task_count"] == 2
    assert propose_trace["fitted_issue_count"] <= total_issues
    assert propose_trace["invalid_subagent_output_count"] >= 1
    assert propose_trace["selected_proposal_count"] == 0
    assert propose_trace["dropped_proposal_count"] == 2
    assert propose_trace["dropped_proposal_reasons"] == {
        "invalid_llm_output": 2
    }
    assert propose_trace["subagent_issue_family_counts"] == {
        "direction": MAX_SUBAGENT_ISSUES_PER_TASK * 2
    }
    assert propose_trace["subagent_omitted_issue_family_counts"] == {
        "direction": total_issues - (MAX_SUBAGENT_ISSUES_PER_TASK * 2)
    }


def test_pipeline_queues_deterministic_action_candidate_proposals_when_subagents_fail(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    total_issues = 6
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        {
            "issue_kind": "reverse_clinical_manifestation",
            "edge_id": f"symptom-{index}->flu",
            "source": f"symptom-{index}",
            "target": "flu",
            "keywords": "has_manifestation",
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "risk": "medium",
            "suggested_action": "replace_relation",
            "candidate_predicates": ["has_manifestation"],
            "new_source": "flu",
            "new_target": f"symptom-{index}",
        }
        for index in range(total_issues)
    ]
    snapshot_path = package / "snapshots" / "kg_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["nodes"].extend(
        {
            "id": f"symptom-{index}",
            "label": f"Symptom {index}",
            "entity_type": "Symptom",
            "source_id": "chunk-1",
            "file_path": "guide.md",
        }
        for index in range(total_issues)
    )
    snapshot["edges"] = [
        {
            "id": f"symptom-{index}->flu",
            "source": f"symptom-{index}",
            "target": "flu",
            "keywords": "has_manifestation",
            "source_id": "chunk-1",
            "file_path": "guide.md",
        }
        for index in range(total_issues)
    ]
    _write_json(quality_path, quality)
    _write_json(snapshot_path, snapshot)
    client = InvalidJsonSubagentRankOkClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            max_subagent_tasks=1,
            max_proposals_per_run=5,
            allow_llm_judge=False,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "pending_human_review"
    assert len(result.proposal_ids) == 5
    assert all(
        proposal_id.startswith("prop-action-candidate-")
        for proposal_id in result.proposal_ids
    )

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = next(stage for stage in trace["stages"] if stage["stage"] == "propose")
    assert propose_trace["stage"] == "propose"
    assert propose_trace["invalid_subagent_output_count"] == 0
    assert propose_trace["action_candidate_count"] == total_issues
    assert propose_trace["action_candidate_proposal_count"] == 5
    assert propose_trace["llm_selected_proposal_count"] == 0
    assert propose_trace["selected_proposal_count"] == 5
    assert propose_trace["dropped_proposal_reasons"]["max_proposals"] == 1
    assert [
        call
        for call in client.calls
        if "Proposal Orchestrator Subagent" in call["system_prompt"]
    ] == []
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    assert "prop-action-candidate-" in approval_queue
    assert "medical_relation_schema_migration" in approval_queue


def test_pipeline_skips_llm_for_deterministic_only_action_candidate_tasks(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        {
            "issue_kind": "reverse_clinical_manifestation",
            "edge_id": "fever->flu",
            "source": "entity fever",
            "target": "flu",
            "keywords": "clinical_manifestation",
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "risk": "medium",
            "suggested_action": "replace_relation",
            "candidate_predicates": ["has_manifestation"],
            "new_source": "flu",
            "new_target": "entity fever",
        }
    ]
    snapshot_path = package / "snapshots" / "kg_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["edges"] = [
        {
            "id": "fever->flu",
            "source": "entity fever",
            "target": "flu",
            "keywords": "clinical_manifestation",
            "source_id": "chunk-1",
            "file_path": "guide.md",
        }
    ]
    _write_json(quality_path, quality)
    _write_json(snapshot_path, snapshot)
    client = InvalidJsonSubagentRankOkClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            max_subagent_tasks=1,
            allow_llm_judge=False,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "pending_human_review"
    assert len(result.proposal_ids) == 1
    subagent_calls = [
        call
        for call in client.calls
        if "Proposal Orchestrator Subagent" in call["system_prompt"]
    ]
    assert subagent_calls == []
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = next(stage for stage in trace["stages"] if stage["stage"] == "propose")
    assert propose_trace["invalid_subagent_output_count"] == 0
    assert propose_trace["action_candidate_proposal_count"] == 1
    assert json.loads((package / "proposal_task_packs.json").read_text()) == []
    assert json.loads((package / "subagent_outputs" / "index.json").read_text()) == []


def test_pipeline_sends_only_hybrid_residual_issues_to_subagent(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        {
            "issue_kind": "treatment_domain_range_mismatch",
            "edge_id": "edge-oseltamivir-flu",
            "source": "oseltamivir",
            "target": "flu",
            "keywords": "recommended_treatment",
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "risk": "medium",
            "suggested_action": "replace_relation",
            "candidate_predicates": ["has_indication"],
            "source_type": "Drug",
            "target_type": "Disease",
        },
        {
            "issue_kind": "missing_required_relation_qualifier",
            "edge_id": "edge-oseltamivir-children",
            "source": "oseltamivir",
            "target": "children",
            "keywords": "recommended_for",
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "risk": "medium",
            "suggested_action": "replace_relation",
            "candidate_predicates": ["recommended_for"],
            "source_type": "Drug",
            "target_type": "Population",
        },
    ]
    snapshot_path = package / "snapshots" / "kg_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["edges"] = [
        {
            "id": "edge-oseltamivir-flu",
            "source": "oseltamivir",
            "target": "flu",
            "keywords": "recommended_treatment",
            "source_id": "chunk-1",
            "file_path": "guide.md",
        },
        {
            "id": "edge-oseltamivir-children",
            "source": "oseltamivir",
            "target": "children",
            "keywords": "recommended_for",
            "source_id": "chunk-1",
            "file_path": "guide.md",
        },
    ]
    _write_json(quality_path, quality)
    _write_json(snapshot_path, snapshot)
    client = InvalidJsonSubagentRankOkClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            max_subagent_tasks=1,
            skip_deterministic_subagent_calls=True,
            allow_llm_judge=False,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "pending_human_review"
    subagent_calls = [
        call
        for call in client.calls
        if "Proposal Orchestrator Subagent" in call["system_prompt"]
    ]
    assert len(subagent_calls) == 1
    prompt = json.loads(subagent_calls[0]["user_prompt"])
    assert prompt["task_pack"]["execution_mode"] == "llm_assisted"
    assert [issue["edge_id"] for issue in prompt["task_pack"]["issues"]] == [
        "edge-oseltamivir-children"
    ]
    assert prompt["task_pack"]["action_candidates"] == []
    task_pack_artifact = json.loads(
        (package / "proposal_task_packs.json").read_text(encoding="utf-8")
    )
    assert all(task_pack["action_candidates"] == [] for task_pack in task_pack_artifact)
    artifact_issue_refs = {
        issue["issue_ref"]
        for task_pack in task_pack_artifact
        for issue in task_pack["issues"]
    }
    artifact_edge_ids = {
        issue["edge_id"]
        for task_pack in task_pack_artifact
        for issue in task_pack["issues"]
    }
    assert artifact_edge_ids == {"edge-oseltamivir-children"}
    assert "edge-oseltamivir-flu" not in artifact_edge_ids
    assert artifact_issue_refs == set(prompt["task_pack"]["residual_issue_refs"])
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = next(stage for stage in trace["stages"] if stage["stage"] == "propose")
    assert propose_trace["action_candidate_proposal_count"] == 1


def test_action_candidate_to_proposal_failure_is_reported(
    tmp_path: Path,
) -> None:
    result = _action_candidate_proposals_from_task_packs(
        [
            {
                "issues": [],
                "action_candidates": [
                    {
                        "candidate_id": "candidate-bad",
                        "issue_ref": "medical_schema_issues:bad:0",
                        "proposal_type": "medical_relation_schema_migration",
                        "target": "edge:bad",
                    }
                ],
            }
        ],
        output_dir=tmp_path,
        previous_outputs={},
    )

    assert result.proposals == []
    assert result.rejected == [
        {
            "candidate_id": "candidate-bad",
            "issue_ref": "medical_schema_issues:bad:0",
            "stage": "proposal_conversion",
            "error_code": "CONVERSION_FAILED",
            "error": "candidate could not be converted",
        }
    ]

    package = tmp_path / "package"
    _write_residual_route_agent_package(package)
    scan = scan_deterministic_candidates(package)
    candidate = scan.candidates[0]
    candidate.pop("action_payload")

    scan_result = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )

    assert scan_result.proposals == []
    route = next(
        route for route in scan.issue_routes if route.issue_ref == candidate["issue_ref"]
    )
    assert route.route_state == "blocked_safety"
    assert route.generation_disposition == "conversion_failed"
    assert route.reason_code == "CONVERSION_FAILED"
    assert route.reason_codes == [
        "DETERMINISTIC_CANDIDATE_VALID",
        "CONVERSION_FAILED",
    ]
    assert route.events[-1]["route_state"] == "blocked_safety"
    assert route.events[-1]["generation_disposition"] == "conversion_failed"
    assert route.events[-1]["reason_code"] == "CONVERSION_FAILED"


def test_action_candidate_proposals_from_scan_writes_route_proposal_ids(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_residual_route_agent_package(package)
    scan = scan_deterministic_candidates(package)

    result = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )

    assert len(result.proposals) == 1
    proposal_id = result.proposals[0].id
    covered_routes = [
        route
        for route in scan.issue_routes
        if route.route_state == "deterministic_covered"
    ]
    assert len(covered_routes) == 1
    assert covered_routes[0].proposal_ids == [proposal_id]


def test_action_candidate_proposals_from_scan_does_not_duplicate_route_proposal_ids(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_residual_route_agent_package(package)
    scan = scan_deterministic_candidates(package)

    first = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )
    second = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )

    assert len(first.proposals) == 1
    assert len(second.proposals) == 1
    proposal_id = first.proposals[0].id
    assert second.proposals[0].id == proposal_id
    route = next(
        route
        for route in scan.issue_routes
        if route.route_state == "deterministic_covered"
    )
    assert route.proposal_ids == [proposal_id]


def test_action_candidate_proposals_from_scan_blocks_evidence_failures(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_residual_route_agent_package(package)
    scan = scan_deterministic_candidates(package)
    candidate = scan.candidates[0]
    candidate["action_payload"] = {
        **candidate["action_payload"],
        "edge_id": "missing-edge",
    }

    result = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )

    assert result.proposals == []
    assert len(result.rejected) == 1
    assert result.rejected[0]["error_code"] == "CANDIDATE_EVIDENCE_INVALID"
    assert result.rejected[0] in scan.rejections
    route = next(route for route in scan.issue_routes if route.issue_ref == candidate["issue_ref"])
    assert route.route_state == "blocked_evidence"
    assert route.reason_code == "CANDIDATE_EVIDENCE_INVALID"
    assert route.proposal_ids == []


def test_action_candidate_proposals_from_scan_dedupes_malformed_rejections(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_residual_route_agent_package(package)
    scan = scan_deterministic_candidates(package)
    candidate = scan.candidates[0]
    candidate.pop("action_payload")

    first = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )
    second = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )

    assert first.proposals == []
    assert second.proposals == []
    assert first.rejected[0]["error_code"] == "CONVERSION_FAILED"
    assert second.rejected[0]["error_code"] == "CONVERSION_FAILED"
    scan_rejections = [
        rejection
        for rejection in scan.rejections
        if rejection.get("stage") == "proposal_conversion"
        and rejection.get("candidate_id") == candidate["candidate_id"]
    ]
    assert len(scan_rejections) == 1
    route = next(route for route in scan.issue_routes if route.issue_ref == candidate["issue_ref"])
    assert route.route_state == "blocked_safety"
    assert route.reason_code == "CONVERSION_FAILED"
    assert "converted" in route.reason


def test_action_candidate_proposals_from_scan_clears_route_proposal_ids_after_failure(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_residual_route_agent_package(package)
    scan = scan_deterministic_candidates(package)
    candidate = scan.candidates[0]

    success = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )
    assert len(success.proposals) == 1
    route = next(route for route in scan.issue_routes if route.issue_ref == candidate["issue_ref"])
    assert route.proposal_ids == [success.proposals[0].id]

    candidate.pop("action_payload")
    failure = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )

    assert failure.proposals == []
    assert failure.rejected[0]["error_code"] == "CONVERSION_FAILED"
    assert route.route_state == "blocked_safety"
    assert route.proposal_ids == []


def test_action_candidate_proposals_from_scan_clears_route_reason_after_success(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_residual_route_agent_package(package)
    scan = scan_deterministic_candidates(package)
    candidate = scan.candidates[0]
    action_payload = candidate.pop("action_payload")

    failure = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )
    assert failure.proposals == []
    route = next(route for route in scan.issue_routes if route.issue_ref == candidate["issue_ref"])
    assert route.reason_code == "CONVERSION_FAILED"
    assert route.reason

    candidate["action_payload"] = action_payload
    success = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )

    assert len(success.proposals) == 1
    assert route.route_state == "deterministic_covered"
    assert route.proposal_ids == [success.proposals[0].id]
    assert route.reason_code == ""
    assert route.reason == ""


def test_task_pack_rejects_unsafe_clinical_modeling_candidates_before_llm(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        {
            "issue_kind": "reverse_clinical_manifestation",
            "edge_id": "流感临床表现->流行性感冒",
            "source": "流感临床表现",
            "target": "流行性感冒",
            "keywords": "临床表现",
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "risk": "medium",
            "suggested_action": "replace_relation",
            "candidate_predicates": ["has_manifestation"],
            "new_source": "流行性感冒",
            "new_target": "流感临床表现",
            "source_type": "Symptom",
            "target_type": "Disease",
        }
    ]
    _write_json(quality_path, quality)

    task_packs = [task_pack.to_dict() for task_pack in build_proposal_task_packs(package)]

    assert len(task_packs) == 1
    task_pack = task_packs[0]
    assert task_pack["role"] == "clinical_modeling"
    assert task_pack["execution_mode"] == "blocked"
    assert task_pack["action_candidates"] == []
    assert task_pack["rejected_action_candidates"]
    assert task_pack["rejected_action_candidates"][0]["error_code"] == (
        "ACTION_CANDIDATE_INVALID"
    )
    assert "has_manifestation target" in task_pack["rejected_action_candidates"][0][
        "error"
    ]


def test_pipeline_blocks_incomplete_treatment_split_without_llm_call(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        {
            "issue_kind": "multi_predicate_edge_split_needed",
            "edge_id": "扎那米韦->儿童",
            "source": "扎那米韦",
            "target": "儿童",
            "keywords": "剂量用法,适用于",
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "risk": "medium",
            "suggested_action": "split_relation",
            "candidate_predicates": [
                "has_dosing_regimen",
                "recommended_for",
                "has_indication",
            ],
            "source_type": "Drug",
            "target_type": "Population",
        }
    ]
    _write_json(quality_path, quality)
    client = InvalidJsonSubagentRankOkClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            max_subagent_tasks=1,
            allow_llm_judge=False,
            skip_deterministic_subagent_calls=True,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "needs_more_evidence"
    subagent_calls = [
        call
        for call in client.calls
        if "Proposal Orchestrator Subagent" in call["system_prompt"]
    ]
    assert subagent_calls == []
    task_packs = json.loads(
        (package / "proposal_task_packs.json").read_text(encoding="utf-8")
    )
    assert task_packs[0]["role"] == "treatment_split"
    assert task_packs[0]["execution_mode"] == "blocked"
    index = json.loads(
        (package / "subagent_outputs" / "index.json").read_text(encoding="utf-8")
    )
    assert index[0]["state"] == "blocked_no_executable_candidate"


def test_pipeline_uses_typed_pathogen_candidate_without_subagent_fallback(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        {
            "issue_kind": "legacy_overloaded_relation",
            "edge_id": "流感病毒->乙型流感",
            "source": "流感病毒",
            "target": "乙型流感",
            "keywords": "病原分型",
            "source_id": "chunk-020",
            "file_path": "儿童流感指南.pdf",
            "risk": "medium",
            "suggested_action": "replace_relation",
            "candidate_predicates": ["is_a", "causative_agent"],
            "guidance": "病毒亚型层级用子类 -> is_a -> 父类；疾病指向致病病原体用 causative_agent。",
        }
    ]
    snapshot_path = package / "snapshots" / "kg_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["source_files"] = ["儿童流感指南.pdf"]
    snapshot["nodes"].extend(
        [
            {
                "id": "流感病毒",
                "label": "流感病毒",
                "entity_type": "Pathogen",
                "source_id": "chunk-020",
                "file_path": "儿童流感指南.pdf",
            },
            {
                "id": "乙型流感",
                "label": "乙型流感",
                "entity_type": "Disease",
                "source_id": "chunk-020",
                "file_path": "儿童流感指南.pdf",
            },
        ]
    )
    snapshot["edges"] = [
        {
            "id": "流感病毒->乙型流感",
            "source": "流感病毒",
            "target": "乙型流感",
            "keywords": "病原分型",
            "source_id": "chunk-020",
            "file_path": "儿童流感指南.pdf",
        }
    ]
    _write_json(quality_path, quality)
    _write_json(snapshot_path, snapshot)
    client = InvalidJsonSubagentRankOkClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            max_subagent_tasks=1,
            max_proposals_per_run=5,
            allow_llm_judge=False,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "pending_human_review"
    assert len(result.proposal_ids) == 1
    subagent_calls = [
        call
        for call in client.calls
        if "Proposal Orchestrator Subagent" in call["system_prompt"]
    ]
    assert subagent_calls == []
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = next(stage for stage in trace["stages"] if stage["stage"] == "propose")

    assert "candidate_kg_expansion" in approval_queue
    assert "typed_influenza_pathogen_expansion" in approval_queue
    assert propose_trace["invalid_subagent_output_count"] == 0
    assert propose_trace["action_candidate_count"] == 1
    assert propose_trace["action_candidate_proposal_count"] == 1
    assert json.loads((package / "proposal_task_packs.json").read_text()) == []
    assert json.loads((package / "subagent_outputs" / "index.json").read_text()) == []


def test_pipeline_clips_huge_task_pack_inputs_before_subagent_prompt_budget(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_agent_package(package, include_medical_schema_issues=False)
    snapshot_path = package / "snapshots" / "kg_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["source_files"] = [
        f"guide-{index}.md:" + ("x" * 1000) for index in range(60)
    ]
    snapshot["metadata"] = {
        f"metadata_key_{index}": "metadata-value-" + ("m" * 1000)
        for index in range(50)
    }
    _write_json(snapshot_path, snapshot)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["findings"][0]["message"] = "large finding " + ("f" * 1000)
    quality["findings"][0]["evidence"] = [
        f"source_id: chunk-{index}; " + ("e" * 1000) for index in range(30)
    ]
    _write_json(quality_path, quality)
    client = CaptureNoProposalAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_context_tokens_per_stage=20000),
        profile="medical_kg",
    )

    assert result.stop_reason == "needs_more_evidence"
    subagent_calls = [
        call
        for call in client.calls
        if "Proposal Orchestrator Subagent" in call["system_prompt"]
    ]
    assert len(subagent_calls) == 1
    assert len(subagent_calls[0]["user_prompt"]) // 4 <= 20000

    task_packs_path = package / "proposal_task_packs.json"
    task_packs_text = task_packs_path.read_text(encoding="utf-8")
    assert len(task_packs_text) < 20000
    task_pack = json.loads(task_packs_text)[0]
    issue = task_pack["issues"][0]
    assert len(issue["message"]) <= 240
    assert len(issue["evidence"]) == 10
    assert issue["evidence_omitted_count"] == 20
    assert all(len(item) <= 240 for item in issue["evidence"])

    snapshot_context = task_pack["snapshot_context"]
    assert len(snapshot_context["source_files"]) == 20
    assert snapshot_context["source_files_omitted_count"] == 40
    assert all(len(item) <= 240 for item in snapshot_context["source_files"])
    assert len(snapshot_context["metadata"]) == 20
    assert snapshot_context["metadata_omitted_key_count"] == 30
    assert all(
        not (isinstance(value, str) and len(value) > 240)
        for value in snapshot_context["metadata"].values()
    )


def test_pipeline_queues_proposals_when_rank_stage_fails(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = FailingRankAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-hierarchy-symptom-001"]
    assert len(client.calls) == 5
    improvement_backlog = package / "improvement_backlog.md"
    assert (package / "approval_queue.md").exists()
    assert improvement_backlog.exists()
    text = improvement_backlog.read_text(encoding="utf-8")
    assert "proposal-hierarchy-symptom-001" in text
    assert "judge_unavailable" in text
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    rank_trace = trace["stages"][-1]
    assert rank_trace["stage"] == "rank_repairs"
    assert rank_trace["state"] == "rank_repairs_unavailable"


def test_pipeline_compacts_auxiliary_context_for_many_large_proposals(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = LargeAuxiliaryContextAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_context_tokens_per_stage=12000),
    )

    assert result.stop_reason == "pending_human_review"
    assert len(result.proposal_ids) == 20
    assert len(client.calls) == 6
    rank_call = next(
        call for call in client.calls if "Rank Repairs Agent" in call["system_prompt"]
    )
    judge_call = next(
        call for call in client.calls if "KG proposal judge" in call["system_prompt"]
    )
    assert len(_proposal_ids_from_prompt(rank_call["user_prompt"])) == 20
    assert len(_proposal_ids_from_prompt(judge_call["user_prompt"])) == 20
    assert len(rank_call["user_prompt"]) // 4 <= 12000
    assert len(judge_call["user_prompt"]) // 4 <= 12000
    assert "x" * 500 not in rank_call["user_prompt"]
    assert "x" * 500 not in judge_call["user_prompt"]

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    rank_trace = next(
        stage for stage in trace["stages"] if stage["stage"] == "rank_repairs"
    )
    judge_trace = next(stage for stage in trace["stages"] if stage["stage"] == "judge")
    assert rank_trace["state"] == "completed"
    assert judge_trace["state"] == "completed"
    assert rank_trace["input_token_estimate"] <= 12000
    assert judge_trace["input_token_estimate"] <= 12000


def test_pipeline_limits_auxiliary_context_proposals_for_bulk_runs(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = VeryLargeAuxiliaryContextAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_context_tokens_per_stage=12000),
    )

    assert result.stop_reason == "pending_human_review"
    assert len(result.proposal_ids) == 120
    assert "proposal-rank-context-119" in (
        package / "proposals.generated.yaml"
    ).read_text(encoding="utf-8")

    rank_call = next(
        call for call in client.calls if "Rank Repairs Agent" in call["system_prompt"]
    )
    judge_call = next(
        call for call in client.calls if "KG proposal judge" in call["system_prompt"]
    )
    rank_payload = json.loads(rank_call["user_prompt"])
    judge_payload = json.loads(judge_call["user_prompt"])
    rank_ids = _proposal_ids_from_prompt(rank_call["user_prompt"])
    judge_ids = _proposal_ids_from_prompt(judge_call["user_prompt"])

    assert 0 < len(rank_ids) < 120
    assert len(rank_ids) == rank_payload["context_compaction"][
        "included_proposal_count"
    ]
    assert rank_payload["context_compaction"]["proposal_omitted_count"] > 0
    assert len(judge_ids) == judge_payload["context_compaction"][
        "included_proposal_count"
    ]
    assert len(rank_call["user_prompt"]) // 4 <= 12000
    assert len(judge_call["user_prompt"]) // 4 <= 12000

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    rank_trace = next(
        stage for stage in trace["stages"] if stage["stage"] == "rank_repairs"
    )
    judge_trace = next(stage for stage in trace["stages"] if stage["stage"] == "judge")
    assert rank_trace["state"] == "completed"
    assert judge_trace["state"] == "completed"


def test_pipeline_dynamically_reduces_auxiliary_proposals_to_fit_budget(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = VeryLargeAuxiliaryContextAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_context_tokens_per_stage=6000),
    )

    assert result.stop_reason == "pending_human_review"
    assert len(result.proposal_ids) == 120

    rank_call = next(
        call for call in client.calls if "Rank Repairs Agent" in call["system_prompt"]
    )
    rank_payload = json.loads(rank_call["user_prompt"])
    rank_ids = _proposal_ids_from_prompt(rank_call["user_prompt"])

    assert 0 < len(rank_ids) < 32
    assert rank_payload["context_compaction"]["included_proposal_count"] == len(
        rank_ids
    )
    assert rank_payload["context_compaction"]["proposal_omitted_count"] == (
        120 - len(rank_ids)
    )
    assert len(rank_call["user_prompt"]) // 4 <= 6000

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    rank_trace = next(
        stage for stage in trace["stages"] if stage["stage"] == "rank_repairs"
    )
    assert rank_trace["state"] == "completed"
    assert rank_trace["input_token_estimate"] <= 6000


def test_pipeline_replaces_stale_downstream_artifacts_when_rank_stage_fails(
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

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-hierarchy-symptom-001"]
    assert not (package / "llm_repair_plan.md").exists()
    assert "judge_unavailable" in (package / "llm_judge_report.md").read_text(
        encoding="utf-8"
    )
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    assert "proposal-hierarchy-symptom-001" in proposals_yaml


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
    assert (
        "- schema_repair-001 (schema_repair) attempt 1: "
        "proposal expected_metric_change values must be numbers"
    ) in report
    assert (
        "- schema_repair-001 (schema_repair) attempt 2: "
        "proposal expected_metric_change values must be numbers"
    ) in report
    assert (
        "- schema_repair-001 (schema_repair) attempt 3: "
        "proposal expected_metric_change values must be numbers"
    ) in report
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert propose_trace["stage"] == "propose"
    assert propose_trace["attempts"] == 3
    assert len(propose_trace["attempt_logs"]) == 3


def test_pipeline_writes_subagent_output_index_on_propose_failure(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = AlwaysInvalidProposeAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
    )

    assert result.stop_reason == "invalid_llm_output"
    output_path = package / "subagent_outputs" / "schema_repair-001.json"
    index_path = package / "subagent_outputs" / "index.json"
    merge_report_path = package / "proposal_merge_report.md"
    assert output_path.exists()
    assert index_path.exists()
    assert merge_report_path.exists()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index == [
        {
            "task_id": "schema_repair-001",
            "role": "schema_repair",
            "proposal_ids": [],
            "path": "subagent_outputs/schema_repair-001.json",
            "state": "invalid_llm_output",
            "error_code": "EXPECTED_METRIC_CHANGE_INVALID",
        }
    ]
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["error_code"] == "EXPECTED_METRIC_CHANGE_INVALID"
    assert output["attempts"][0]["error_code"] == "EXPECTED_METRIC_CHANGE_INVALID"
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert propose_trace["error_code"] == "EXPECTED_METRIC_CHANGE_INVALID"
    assert (
        propose_trace["dropped_subagents"][0]["error_code"]
        == "EXPECTED_METRIC_CHANGE_INVALID"
    )
    merge_report = merge_report_path.read_text(encoding="utf-8")
    assert "## Dropped" in merge_report
    assert "- schema_repair-001: invalid_llm_output" in merge_report
    assert "[EXPECTED_METRIC_CHANGE_INVALID]" in merge_report


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


def test_pipeline_drops_stale_relation_replacement_current_keywords(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = StaleCurrentKeywordsAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            require_candidate_evidence_allowlist=False,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "needs_more_evidence"
    assert result.proposal_ids == []
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    merge_report = (package / "proposal_merge_report.md").read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert "proposals: []" in approval_queue
    assert "proposal-stale-relation-001" not in approval_queue
    assert "- proposal-stale-relation-001: stale_current_keywords" in merge_report
    assert propose_trace["dropped_proposal_reasons"]["stale_current_keywords"] == 1


def test_llm_task_packs_are_built_only_from_residual_issue_routes(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_residual_route_agent_package(package)

    scan = scan_deterministic_candidates(package)
    packs = build_llm_residual_task_packs(package, scan)

    issue_refs_by_edge = {
        str(issue["edge_id"]): str(issue["issue_ref"])
        for issue in scan.issues
    }
    route_states_by_ref = {
        route.issue_ref: route.route_state for route in scan.issue_routes
    }
    route_states_by_edge = {
        edge_id: route_states_by_ref[issue_ref]
        for edge_id, issue_ref in issue_refs_by_edge.items()
    }
    assert route_states_by_edge["edge-deterministic"] == "deterministic_covered"
    assert route_states_by_edge["edge-residual"] == "llm_residual"

    residual_refs = {
        route.issue_ref
        for route in scan.issue_routes
        if route.route_state == "llm_residual"
    }
    deterministic_covered_refs = {
        route.issue_ref
        for route in scan.issue_routes
        if route.route_state == "deterministic_covered"
    }
    packed_refs = {
        issue["issue_ref"]
        for pack in packs
        for issue in pack.to_dict()["issues"]
    }
    assert packed_refs == residual_refs
    assert packed_refs.isdisjoint(deterministic_covered_refs)
    assert packed_refs


def test_orchestrated_propose_generates_action_candidates_from_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    package = tmp_path / "package"
    _write_bulk_complication_agent_package(package, issue_count=10)
    client = EmptySubagentProposalClient()
    monkeypatch.setattr(
        "lightrag.kb_iteration.agent_pipeline."
        "_action_candidate_proposals_from_task_packs",
        lambda *args, **kwargs: pytest.fail(
            "orchestrated propose should convert action candidates from scan"
        ),
    )

    result = _run_orchestrated_propose_stage(
        output_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            max_subagent_tasks=1,
            max_proposals_per_run=20,
        ),
        profile="medical_kg",
        context={},
        previous_outputs={},
        stage_trace={},
    )

    assert result.parsed is not None
    proposal_ids = [proposal.id for proposal in result.parsed.proposals]
    assert len(client.calls) == 0
    assert len(proposal_ids) == 10
    assert {
        proposal.action_payload["edge_id"] for proposal in result.parsed.proposals
    } == {f"{complication}->流行性感冒" for complication in _BULK_COMPLICATIONS[:10]}


def test_orchestrated_propose_syncs_disambiguated_action_candidate_route_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    _write_residual_route_agent_package(package)
    id_scan = scan_deterministic_candidates(package)
    original = action_candidate_proposals_from_scan(
        id_scan,
        output_dir=package,
        previous_outputs={},
    )
    original_proposal_id = original.proposals[0].id
    (package / "rejected_changes.md").write_text(
        f"# Rejected Changes\n\n## {original_proposal_id}\n\n"
        "- decision: reject\n",
        encoding="utf-8",
    )
    scan = scan_deterministic_candidates(package)
    monkeypatch.setattr(
        "lightrag.kb_iteration.agent_pipeline.scan_deterministic_candidates",
        lambda *args, **kwargs: scan,
    )
    stage_trace: dict[str, object] = {}

    result = _run_orchestrated_propose_stage(
        output_dir=package,
        client=EmptySubagentProposalClient(),
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            max_subagent_tasks=1,
            allow_llm_judge=False,
        ),
        profile="medical_kg",
        context={},
        previous_outputs={},
        stage_trace=stage_trace,
    )

    assert result.parsed is not None
    selected_id = result.parsed.proposals[0].id
    assert selected_id.startswith(f"{original_proposal_id}-")
    assert selected_id != original_proposal_id
    route = next(
        route
        for route in scan.issue_routes
        if route.route_state == "deterministic_covered"
    )
    assert route.proposal_ids == [selected_id]
    assert stage_trace["action_candidate_proposal_count"] == 1


def test_orchestrated_propose_writes_funnel_artifacts_and_trace(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_residual_route_agent_package(package)
    stage_trace: dict[str, object] = {}

    result = _run_orchestrated_propose_stage(
        output_dir=package,
        client=EmptySubagentProposalClient(),
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            max_subagent_tasks=1,
            allow_llm_judge=False,
        ),
        profile="medical_kg",
        context={},
        previous_outputs={},
        stage_trace=stage_trace,
    )

    assert result.parsed is not None
    assert {
        "issue_ledger",
        "deterministic_proposal_report",
        "deterministic_proposal_report_md",
    }.issubset(result.artifact_paths)
    assert result.artifact_paths["issue_ledger"].exists()
    assert result.artifact_paths["deterministic_proposal_report"].exists()
    assert result.artifact_paths["deterministic_proposal_report_md"].exists()

    report = json.loads(
        result.artifact_paths["deterministic_proposal_report"].read_text(
            encoding="utf-8"
        )
    )
    assert report["summary"]["accounting_balanced"] is True
    assert stage_trace["issue_accounting_balanced"] is True
    assert stage_trace["issue_route_state_counts"] == {
        "deterministic_covered": 1,
        "llm_residual": 1,
    }


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
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            require_candidate_evidence_allowlist=False,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "invalid_llm_output"
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert (
        "action_payload new_source is not grounded in deterministic artifacts"
        in propose_trace["error"]
    )


def test_pipeline_rejects_fabricated_candidate_kg_expansion_payload(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = FabricatedCandidateKGExpansionPayloadAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            require_candidate_evidence_allowlist=False,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    report = (package / "llm_review_report.md").read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    expected_error = (
        "action_payload source_id is not grounded in deterministic artifacts"
    )
    assert "proposals: []" in approval_queue
    assert "proposal-candidate-kg-expansion-001" not in approval_queue
    assert expected_error in report
    assert expected_error in propose_trace["error"]


def test_pipeline_accepts_grounded_candidate_kg_expansion_payload(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = ValidCandidateKGExpansionPayloadAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            require_candidate_evidence_allowlist=False,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-candidate-kg-expansion-001"]
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    assert "proposal-candidate-kg-expansion-001" in approval_queue
    assert "candidate_kg_expansion" in approval_queue
    assert "symptom:body-aches" in approval_queue
    assert "proposal-candidate-kg-expansion-001" in proposals_yaml


def test_pipeline_rejects_candidate_kg_expansion_quote_from_llm_outputs_only(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = LLMOnlyCandidateQuoteAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(
            max_stage_retries=0,
            require_candidate_evidence_allowlist=False,
        ),
        profile="medical_kg",
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    report = (package / "llm_review_report.md").read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    expected_error = (
        "action_payload evidence_quote is not grounded in deterministic artifacts"
    )
    assert "proposals: []" in approval_queue
    assert "proposal-candidate-kg-expansion-001" not in approval_queue
    assert expected_error in report
    assert expected_error in propose_trace["error"]


def test_pipeline_rejects_generic_candidate_kg_expansion_quote(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = GenericCandidateQuoteAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
        profile="medical_kg",
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    report = (package / "llm_review_report.md").read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    expected_error = (
        "action_payload evidence_quote is too short to ground a candidate KG expansion"
    )
    assert expected_error in report
    assert expected_error in propose_trace["error"]


def test_parse_propose_output_rejects_candidate_expansion_outside_allowlist(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    raw_output = json.dumps(
        {
            "proposals": [
                {
                    "id": "proposal-candidate-kg-expansion-allowlist",
                    "type": "candidate_kg_expansion",
                    "target": "kg:candidate:flu-manifestations",
                    "proposed_change": "Queue candidate symptom node and edge.",
                    "reason": "Grounded evidence suggests a missing symptom.",
                    "evidence": [
                        "source_id: chunk-1; file_path: guide.md; item_id: entity fever"
                    ],
                    "confidence": 0.72,
                    "risk": "medium",
                    "requires_approval": True,
                    "expected_metric_change": {},
                    "action_payload": {
                        "candidate_nodes": [
                            {
                                "id": "symptom:body-aches",
                                "label": "Body aches",
                                "entity_type": "Symptom",
                            }
                        ],
                        "candidate_edges": [
                            {
                                "source": "flu",
                                "source_type": "Disease",
                                "target": "symptom:body-aches",
                                "target_type": "Symptom",
                                "keywords": "has_manifestation",
                                "source_id": "chunk-1",
                                "file_path": "guide.md",
                                "qualifiers": {},
                            }
                        ],
                        "source_id": "chunk-1",
                        "file_path": "guide.md",
                        "evidence_quote": "Fever is a symptom mentioned in guide.md.",
                        "why_not_existing": "No equivalent candidate exists.",
                    },
                }
            ]
        },
        ensure_ascii=False,
    )

    with pytest.raises(ValueError, match="allowed_evidence_spans"):
        _parse_and_validate_propose_output(
            raw_output,
            output_dir=package,
            previous_outputs={},
            task_pack={
                "role": "evidence_grounding",
                "allowed_evidence_spans": [
                    {
                        "source_id": "chunk-1",
                        "file_path": "guide.md",
                        "evidence_quote": "A different exact quote.",
                    }
                ],
            },
        )


def test_parse_propose_output_rejects_candidate_expansion_sep_joined_allowlist(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    raw_output = json.dumps(
        {
            "proposals": [
                {
                    "id": "proposal-candidate-kg-expansion-sep",
                    "type": "candidate_kg_expansion",
                    "target": "kg:candidate:flu-manifestations",
                    "proposed_change": "Queue candidate symptom node and edge.",
                    "reason": "Grounded evidence suggests a missing symptom.",
                    "evidence": [
                        "source_id: chunk-1; file_path: guide.md; item_id: entity fever"
                    ],
                    "confidence": 0.72,
                    "risk": "medium",
                    "requires_approval": True,
                    "expected_metric_change": {},
                    "action_payload": {
                        "candidate_nodes": [
                            {
                                "id": "symptom:body-aches",
                                "label": "Body aches",
                                "entity_type": "Symptom",
                            }
                        ],
                        "candidate_edges": [
                            {
                                "source": "flu",
                                "source_type": "Disease",
                                "target": "symptom:body-aches",
                                "target_type": "Symptom",
                                "keywords": "has_manifestation",
                                "source_id": f"chunk-1{GRAPH_FIELD_SEP}chunk-2",
                                "file_path": "guide.md",
                                "qualifiers": {},
                            }
                        ],
                        "source_id": f"chunk-1{GRAPH_FIELD_SEP}chunk-2",
                        "file_path": "guide.md",
                        "evidence_quote": "Fever is a symptom mentioned in guide.md.",
                        "why_not_existing": "No equivalent candidate exists.",
                    },
                }
            ]
        },
        ensure_ascii=False,
    )

    with pytest.raises(ValueError, match="EVIDENCE_TUPLE_MUST_MATCH_ALLOWED_SPAN"):
        _parse_and_validate_propose_output(
            raw_output,
            output_dir=package,
            previous_outputs={},
            task_pack={
                "role": "evidence_grounding",
                "allowed_evidence_spans": [
                    {
                        "source_id": "chunk-1",
                        "file_path": "guide.md",
                        "evidence_quote": "Fever is a symptom mentioned in guide.md.",
                    }
                ],
            },
        )


def test_parse_propose_output_passes_allowed_evidence_spans_to_parser(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    raw_output = json.dumps(
        {
            "proposals": [
                {
                    "id": "proposal-candidate-kg-expansion-cross-combined",
                    "type": "candidate_kg_expansion",
                    "target": "kg:candidate:flu-manifestations",
                    "proposed_change": "Queue candidate symptom node and edge.",
                    "reason": "Grounded evidence suggests a missing symptom.",
                    "evidence": [
                        "source_id: chunk-1; file_path: guide.md; item_id: entity fever"
                    ],
                    "confidence": 0.72,
                    "risk": "medium",
                    "requires_approval": True,
                    "expected_metric_change": {},
                    "action_payload": {
                        "candidate_nodes": [
                            {
                                "id": "symptom:body-aches",
                                "label": "Body aches",
                                "entity_type": "Symptom",
                            }
                        ],
                        "candidate_edges": [
                            {
                                "source": "flu",
                                "source_type": "Disease",
                                "target": "symptom:body-aches",
                                "target_type": "Symptom",
                                "keywords": "has_manifestation",
                                "source_id": "chunk-1",
                                "file_path": "guide.md",
                                "qualifiers": {},
                            }
                        ],
                        "source_id": "chunk-1",
                        "file_path": "guide.md",
                        "evidence_quote": "Fever is a symptom mentioned in guide.md.",
                        "why_not_existing": "No equivalent candidate exists.",
                    },
                }
            ]
        },
        ensure_ascii=False,
    )

    with pytest.raises(ValueError, match="EVIDENCE_TUPLE_MUST_MATCH_ALLOWED_SPAN"):
        _parse_and_validate_propose_output(
            raw_output,
            output_dir=package,
            previous_outputs={},
            task_pack={
                "role": "evidence_grounding",
                "allowed_evidence_spans": [
                    {
                        "source_id": "chunk-1",
                        "file_path": "other-guide.md",
                        "evidence_quote": "Fever is a symptom mentioned in guide.md.",
                    },
                    {
                        "source_id": "chunk-2",
                        "file_path": "guide.md",
                        "evidence_quote": "Fever is a symptom mentioned in guide.md.",
                    },
                ],
            },
            require_candidate_evidence_allowlist=False,
        )


def test_parse_propose_output_rejects_structured_evidence_with_error_code(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    raw_output = json.dumps(
        {
            "proposals": [
                {
                    "id": "proposal-structured-evidence-001",
                    "type": "quality_report_note",
                    "target": "quality_report.md",
                    "proposed_change": "Record grounded relation review.",
                    "reason": "The source relation should be reviewed.",
                    "evidence": [
                        {
                            "source_id": "chunk-1",
                            "file_path": "guide.md",
                            "relation_id": "flu-fever",
                        }
                    ],
                    "confidence": 0.81,
                    "risk": "low",
                    "requires_approval": True,
                    "expected_metric_change": {},
                }
            ]
        },
        ensure_ascii=False,
    )

    with pytest.raises(ValueError, match="EVIDENCE_MUST_BE_STRING"):
        _parse_and_validate_propose_output(
            raw_output,
            output_dir=package,
            previous_outputs={},
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


def test_pipeline_accepts_grounded_short_relation_ids(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    snapshot_path = package / "snapshots" / "kg_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["edges"][0]["id"] = "0"
    _write_json(snapshot_path, snapshot)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"][0]["edge_id"] = "0"
    _write_json(quality_path, quality)
    client = MismatchedActionPayloadTypeAgentClient(
        field_name="edge_id",
        field_value="0",
    )
    client.outputs[3]["proposals"][0]["target"] = "kg:relation:0"
    client.outputs[3]["proposals"][0]["evidence"] = [
        "source_id: chunk-1; file_path: guide.md; relation_id: 0"
    ]
    client.outputs[3]["proposals"][0]["action_payload"]["edge_id"] = "0"

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
        profile="medical_kg",
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-medical-schema-typed-001"]
    approval_queue = (package / "approval_queue.md").read_text(encoding="utf-8")
    assert "relation_id: 0" in approval_queue


def test_pipeline_rejects_executable_payload_that_does_not_match_action_candidate(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    snapshot_path = package / "snapshots" / "kg_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["edges"][0]["source"] = "entity fever"
    snapshot["edges"][0]["target"] = "flu"
    snapshot["edges"][0]["keywords"] = "clinical_manifestation"
    _write_json(snapshot_path, snapshot)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        {
            "issue_kind": "reverse_clinical_manifestation",
            "edge_id": "flu-fever",
            "source": "entity fever",
            "target": "flu",
            "keywords": "clinical_manifestation",
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "suggested_action": "replace_relation",
            "candidate_predicates": ["has_manifestation"],
            "new_source": "flu",
            "new_target": "entity fever",
        }
    ]
    _write_json(quality_path, quality)
    client = NonCandidateActionPayloadAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0),
        profile="medical_kg",
    )

    assert result.stop_reason == "pending_human_review"
    assert len(result.proposal_ids) == 1
    assert result.proposal_ids[0].startswith("prop-action-candidate-")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = next(stage for stage in trace["stages"] if stage["stage"] == "propose")
    assert propose_trace["invalid_subagent_output_count"] == 0
    assert propose_trace["action_candidate_proposal_count"] == 1
    assert propose_trace["llm_selected_proposal_count"] == 0
    assert "dropped_subagents" not in propose_trace
    assert [
        call
        for call in client.calls
        if "Proposal Orchestrator Subagent" in call["system_prompt"]
    ] == []
    assert json.loads((package / "proposal_task_packs.json").read_text()) == []


@pytest.mark.parametrize(
    "proposal_type",
    [
        "value_node_to_qualifier",
        "entity_alias_merge",
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


def test_pipeline_rejects_medical_fact_role_split_without_split_relation_payload(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = MissingExecutableMedicalActionPayloadAgentClient("medical_fact_role_split")

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
        "medical_fact_role_split action_payload action must be split_relation"
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
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    judge_trace = trace["stages"][-1]
    assert "proposal-hierarchy-symptom-001" in proposals_yaml
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

    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    judge_report = (package / "llm_judge_report.md").read_text(encoding="utf-8")
    assert "judge_unavailable" in proposals_yaml
    assert "judge_unavailable" in judge_report


def test_pipeline_normalizes_non_ascii_proposal_ids_before_validation(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = NonAsciiProposalIdAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0, allow_llm_judge=False),
    )

    assert result.stop_reason == "pending_human_review"
    assert len(result.proposal_ids) == 1
    normalized_id = result.proposal_ids[0]
    assert normalized_id.startswith("prop-hierarchy_rule_change-")
    assert "修复" not in normalized_id

    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    assert normalized_id in proposals_yaml


def test_pipeline_disambiguates_proposal_ids_used_by_previous_decisions(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    (package / "rejected_changes.md").write_text(
        "# Rejected Changes\n\n## proposal-hierarchy-symptom-001\n\n"
        "- decision: reject\n",
        encoding="utf-8",
    )
    client = SequencedAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=0, allow_llm_judge=False),
    )

    assert result.stop_reason == "pending_human_review"
    assert len(result.proposal_ids) == 1
    disambiguated_id = result.proposal_ids[0]
    assert disambiguated_id.startswith("proposal-hierarchy-symptom-001-")
    assert disambiguated_id != "proposal-hierarchy-symptom-001"

    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    assert f"id: {disambiguated_id}" in proposals_yaml
    assert trace["proposal_ids"] == [disambiguated_id]


def test_pipeline_keeps_proposals_when_auxiliary_stage_unavailable(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = LargeRankContextAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_context_tokens_per_stage=1500),
    )

    assert result.stop_reason == "pending_human_review"
    assert len(result.proposal_ids) == 20
    assert result.proposal_ids[0] == "proposal-rank-context-0"

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    unavailable_trace = trace["stages"][-1]
    assert unavailable_trace["stage"] in {"rank_repairs", "judge"}
    assert unavailable_trace["state"] in {
        "rank_repairs_unavailable",
        "judge_unavailable",
    }
    assert unavailable_trace["error"]

    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    judge_report = (package / "llm_judge_report.md").read_text(encoding="utf-8")
    assert "proposal-rank-context-0" in proposals_yaml
    assert "judge_unavailable" in proposals_yaml
    assert "judge_unavailable" in judge_report


def test_pipeline_marks_judge_unavailable_when_judge_context_too_large(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = LargeJudgeContextAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_context_tokens_per_stage=1500),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-hierarchy-symptom-001"]

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    judge_trace = trace["stages"][-1]
    assert judge_trace["stage"] == "judge"
    assert judge_trace["state"] == "judge_unavailable"
    assert "max_context_tokens_per_stage" in judge_trace["error"]

    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    judge_report = (package / "llm_judge_report.md").read_text(encoding="utf-8")
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


def test_pipeline_replaces_stale_symlinked_subagent_outputs_dir(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    external_outputs = tmp_path / "external_outputs"
    external_outputs.mkdir()
    external_file = external_outputs / "schema_repair-001.json"
    external_file.write_text("outside sentinel", encoding="utf-8")
    subagent_outputs_link = package / "subagent_outputs"
    try:
        subagent_outputs_link.symlink_to(external_outputs, target_is_directory=True)
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
    assert subagent_outputs_link.is_dir()
    assert not subagent_outputs_link.is_symlink()
    assert (subagent_outputs_link / "index.json").exists()
    assert (subagent_outputs_link / "schema_repair-001.json").exists()


def test_pipeline_cleans_stale_orchestrator_display_artifacts(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    (package / "proposal_task_packs.zh.json").write_text(
        '{"stale": true}\n', encoding="utf-8"
    )
    (package / "proposal_merge_report.zh.md").write_text(
        "# Stale merge report\n", encoding="utf-8"
    )
    subagent_dir = package / "subagent_outputs"
    subagent_dir.mkdir()
    (subagent_dir / "index.zh.json").write_text('{"stale": true}\n', encoding="utf-8")

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=SequencedAgentClient(),
        config=LLMAgentPipelineConfig(),
    )

    assert result.stop_reason == "pending_human_review"
    assert (package / "proposal_task_packs.json").exists()
    assert (package / "proposal_merge_report.md").exists()
    assert not (package / "proposal_task_packs.zh.json").exists()
    assert not (package / "proposal_merge_report.zh.md").exists()
    assert not (subagent_dir / "index.zh.json").exists()


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
    proposals_yaml = (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )
    assert "proposal-hierarchy-symptom-001" in proposals_yaml


def test_retry_user_prompt_adds_specific_proposal_validation_hint():
    prompt = _retry_user_prompt(
        "{}",
        stage="propose",
        error="proposal expected_metric_change must be a dict",
    )

    assert (
        'Use expected_metric_change as an object, for example '
        '{"relation_semantic_issue_count": -1}.'
    ) in prompt


def test_retry_user_prompt_hints_for_action_payload_not_grounded_error():
    prompt = _retry_user_prompt(
        "{}",
        stage="propose",
        error=(
            "proposal X action_payload edge_id is not grounded in "
            "deterministic artifacts"
        ),
    )

    assert (
        "Copy action_payload from the provided action_candidates when available; "
        "do not invent edge_id/source/target values."
    ) in prompt


def test_subagent_role_contract_forbids_prevention_complication_predicate() -> None:
    contract = role_contract("prevention")

    assert contract.role == "prevention"
    assert "has_complication" in contract.forbidden_predicates
    assert "high_risk_for" in contract.allowed_predicates
    assert contract.require_action_candidate is True


def test_subagent_role_contract_lists_structured_retry_error_codes() -> None:
    contract = role_contract("diagnosis")

    for error_code in (
        "EVIDENCE_NOT_GROUNDED",
        "CANDIDATE_NODE_REQUIRED_FIELDS",
        "NO_OP_REPLACE_RELATION",
        "QUALITY_REPORT_NOTE_TARGET_INVALID",
        "RELATION_SCHEMA_VIOLATION",
        "EXPECTED_METRIC_CHANGE_INVALID",
        "CANDIDATE_EDGE_SCHEMA_VIOLATION",
    ):
        assert error_code in contract.retry_error_codes
        assert error_code in contract.retry_contract

    assert contract.retry_contract["CANDIDATE_NODE_REQUIRED_FIELDS"][
        "missing_fields"
    ] == ("id", "entity_type")
    assert contract.retry_contract["EVIDENCE_NOT_GROUNDED"]["missing_fields"] == (
        "source_id",
        "file_path",
        "evidence_quote",
    )


def test_retry_user_prompt_includes_machine_readable_role_constraints() -> None:
    prompt = _retry_user_prompt(
        "{}",
        stage="propose",
        error=(
            "proposal action_payload has_complication must not model chronic "
            "underlying conditions"
        ),
        task_pack={
            "role": "prevention",
            "role_contract": role_contract("prevention").to_dict(),
            "action_candidates": [
                {
                    "candidate_id": "ac-safe",
                    "action_payload": {"new_keywords": "high_risk_for"},
                }
            ],
        },
    )

    assert '"role": "prevention"' in prompt
    assert '"forbidden_predicates": ["has_complication"]' in prompt
    assert "不要再次使用 has_complication" in prompt
    assert '"allowed_candidate_ids": ["ac-safe"]' in prompt


@pytest.mark.parametrize(
    "error",
    [
        "candidate_kg_expansion action_payload candidate_nodes must be a list",
        "candidate_kg_expansion action_payload must include non-empty string source_id",
    ],
)
def test_retry_user_prompt_hints_for_candidate_kg_expansion_contract(error: str):
    prompt = _retry_user_prompt(
        "{}",
        stage="propose",
        error=error,
    )

    assert "candidate_kg_expansion action_payload" in prompt
    assert "candidate_nodes" in prompt
    assert "candidate_edges" in prompt
    assert "retire_edges" in prompt
    assert "source_id" in prompt
    assert "file_path" in prompt
    assert "evidence_quote" in prompt
    assert "why_not_existing" in prompt
    assert "top-level" in prompt


def test_parse_agent_stage_output_rejects_single_structured_evidence_object():
    with pytest.raises(ValueError, match="EVIDENCE_MUST_BE_STRING"):
        parse_agent_stage_output(
            "propose",
            json.dumps(
                {
                    "proposals": [
                        {
                            "proposal_id": "proposal-evidence-object-001",
                            "proposal_type": "quality_report_note",
                            "target": "quality_report.md",
                            "proposed_change": "Record grounded relation review.",
                            "reason": "The source relation should be reviewed.",
                            "evidence": {
                                "source_id": "chunk-1",
                                "file_path": "guide.md",
                                "edge_id": "flu-fever",
                                "item_type": "relation",
                                "evidence_status": "grounded",
                            },
                            "confidence": 0.81,
                            "risk": "low",
                            "requires_approval": True,
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        )


def test_parse_agent_stage_output_normalizes_replace_relation_alias_payload():
    output = parse_agent_stage_output(
        "propose",
        json.dumps(
            {
                "proposals": [
                    {
                        "proposal_id": "proposal-replace-relation-alias-001",
                        "proposal_type": "replace_relation",
                        "target": "edge:flu->entity fever",
                        "reason": "Normalize to the canonical medical relation.",
                        "evidence": [
                            "source_id: chunk-1; file_path: guide.md; relation_id: flu->entity fever"
                        ],
                        "confidence": 0.9,
                        "requires_approval": True,
                        "expected_metric_change": {},
                        "action_payload": {
                            "old_relation_id": "flu->entity fever",
                            "current_keywords": "clinical_manifestation",
                            "new_relation": {
                                "source": "flu",
                                "target": "entity fever",
                                "predicate": "has_manifestation",
                            },
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )

    proposal = output.proposals[0]
    assert proposal.type == "medical_relation_schema_migration"
    assert proposal.proposed_change == "Normalize to the canonical medical relation."
    assert proposal.evidence == [
        "source_id: chunk-1; file_path: guide.md; relation_id: flu->entity fever"
    ]
    assert proposal.action_payload == {
        "action": "replace_relation",
        "edge_id": "flu->entity fever",
        "expected_source": "flu",
        "expected_target": "entity fever",
        "current_keywords": "clinical_manifestation",
        "new_source": "flu",
        "new_target": "entity fever",
        "new_keywords": "has_manifestation",
        "qualifiers": {},
    }


def test_structured_evidence_accepts_graph_field_separator_values():
    tokens = _EvidenceReferenceTokens()
    tokens.by_key["source_id"].update({"chunk-1", "chunk-2"})
    tokens.by_key["file_path"].add("guide.md")
    tokens.by_key["relation_id"].add("flu-fever")

    assert _evidence_references_known_artifact(
        (
            f"source_id: chunk-1{GRAPH_FIELD_SEP}chunk-2; "
            "file_path: guide.md; relation_id: flu-fever"
        ),
        tokens,
    )


def test_subagent_propose_system_prompt_includes_exact_output_contract():
    prompt = _subagent_propose_system_prompt("medical_kg")

    assert "ImprovementProposal required fields" in prompt
    assert "medical_relation_schema_migration action_payload" in prompt
    assert "evidence: list of strings only" in prompt
    assert "structured items" not in prompt
    assert "EVIDENCE_MUST_BE_STRING" in prompt
    assert "Copy action_payload from task_pack.action_candidates" in prompt
    assert "流感临床表现" in prompt
    assert "肺炎链球菌" in prompt
    assert "supports_or_refutes" in prompt
    assert "扎那米韦" in prompt
    assert "低钾血症" in prompt
    assert "乙型流感病毒 -> is_a -> 流感病毒" in prompt
    assert "流感病毒 -> is_a -> 乙型流感病毒" in prompt
    assert "current_keywords" in prompt
    assert "exact full current keyword string" in prompt
    assert "血常规" in prompt
    assert "MRI" in prompt
    assert "Do not retire nonspecific test edges" in prompt
    assert "orders_test" in prompt
    assert "monitor_with" in prompt
    assert "Do not convert CT or MRI to generic 流行性感冒 -> orders_test" in prompt
    assert "扎那米韦->哮喘" in prompt
    assert "扎那米韦->儿童" in prompt
    assert "executable split" in prompt
    assert "TCM syndrome" in prompt
    assert "review-only" in prompt
    assert "candidate_kg_expansion action_payload" in prompt
    assert "candidate_nodes" in prompt
    assert "candidate_edges" in prompt
    assert "source_id" in prompt
    assert "file_path" in prompt
    assert "evidence_quote" in prompt
    assert "why_not_existing" in prompt
    assert "top-level" in prompt
    assert "candidate_edges[] item must include source, target, source_type" in prompt
    assert "same allowed_evidence_spans record" in prompt
    assert "<SEP>" in prompt
    assert "no-op" in prompt
    assert "慢性阻塞性肺疾病" in prompt
    assert "supports_or_refutes" in prompt
    assert "丙氨酸氨基转移酶" in prompt
    assert "死亡/住院/早产/低出生体重/重症/危重症" in prompt
    assert "do not use has_complication" in prompt
    assert "Do not emit review_context_request" in prompt
    assert "急性心衰患者" in prompt
    assert "reduces_risk_of" in prompt


def test_subagent_propose_system_prompt_loads_role_specific_prompt_file():
    prompt = _subagent_propose_system_prompt("medical_kg", role="treatment_split")

    assert "你是 treatment_split 子 Agent" in prompt
    assert "只能选择 task_pack.action_candidates 中已有候选" in prompt
    assert "不得自行创建或修改 action_payload" in prompt


def test_parse_and_validate_propose_output_rejects_review_context_request(
    tmp_path: Path,
):
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    (snapshot_dir / "kg_snapshot.json").write_text(
        json.dumps(
            {
                "workspace": "demo",
                "generated_at": "2026-06-22T00:00:00Z",
                "source_files": ["guide.md"],
                "nodes": [],
                "edges": [],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    (snapshot_dir / "quality_score.json").write_text(
        json.dumps({"overall": 80, "details": {}, "findings": []}),
        encoding="utf-8",
    )
    raw_output = json.dumps(
        {
            "proposals": [
                {
                    "id": "proposal-review-only-001",
                    "type": "review_context_request",
                    "target": "edge:流感病毒->实验室诊断",
                    "proposed_change": "Request a human review of this edge.",
                    "reason": "The edge is ambiguous.",
                    "evidence": ["source_id: chunk-1; file_path: guide.md"],
                    "confidence": 0.8,
                    "risk": "medium",
                    "requires_approval": True,
                    "expected_metric_change": {},
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="review_context_request.*executable"):
        _parse_and_validate_propose_output(
            raw_output,
            output_dir=tmp_path,
            previous_outputs={},
        )


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
    assert attempt_logs[0]["error_code"] == "EXPECTED_METRIC_CHANGE_INVALID"
    assert isinstance(attempt_logs[0]["output_token_estimate"], int)
    assert attempt_logs[0]["output_token_estimate"] > 0
    assert attempt_logs[1]["attempt"] == 2
    assert attempt_logs[1]["state"] == "invalid_llm_output"
    assert (
        attempt_logs[1]["error"]
        == "proposal proposal-hierarchy-symptom-001 evidence is not grounded in deterministic artifacts"
    )
    assert attempt_logs[1]["error_code"] == "EVIDENCE_NOT_GROUNDED"
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
    assert '"error_code": "EVIDENCE_NOT_GROUNDED"' in third_propose_prompt
    assert "Return corrected JSON only." in third_propose_prompt
    assert (
        "Do not change evidence/human-approval requirements." in third_propose_prompt
    )


@pytest.mark.parametrize(
    ("error", "error_code", "missing_fields"),
    [
        (
            "proposal p1 evidence is not grounded in deterministic artifacts",
            "EVIDENCE_NOT_GROUNDED",
            ["source_id", "file_path", "evidence_quote"],
        ),
        (
            "candidate_nodes[0] requires id and entity_type",
            "CANDIDATE_NODE_REQUIRED_FIELDS",
            ["id", "entity_type"],
        ),
        (
            "proposal action_payload replace_relation must not be a no-op: source, target, and relation keyword are unchanged",
            "NO_OP_REPLACE_RELATION",
            [],
        ),
        (
            "quality_report_note without approval must target quality_report.md",
            "QUALITY_REPORT_NOTE_TARGET_INVALID",
            ["target"],
        ),
        (
            "proposal action_payload orders_test source must be a disease, clinical condition, care process, or recommendation context, not a bare pathogen entity",
            "RELATION_SCHEMA_VIOLATION",
            ["new_source"],
        ),
        (
            "proposal action_payload orders_test must not connect influenza directly to a bare lab marker; order the parent test panel or model the marker as an observed lab finding",
            "RELATION_SCHEMA_VIOLATION",
            ["new_target"],
        ),
        (
            "proposal action_payload supports_or_refutes must not model nonspecific labs or complication-imaging findings as direct support/refutation for influenza diagnosis",
            "RELATION_SCHEMA_VIOLATION",
            ["new_target", "new_keywords"],
        ),
        (
            "CANDIDATE_EDGE_SCHEMA_VIOLATION: candidate_edges[0]: target_type Unknown is outside performed_by_method range Method | Technique | Instrument",
            "CANDIDATE_EDGE_SCHEMA_VIOLATION",
            ["source_type", "target_type", "keywords"],
        ),
    ],
)
def test_retry_user_prompt_includes_structured_validation_error_code(
    error: str, error_code: str, missing_fields: list[str]
) -> None:
    prompt = _retry_user_prompt(
        "{}",
        stage="propose",
        error=error,
        task_pack={
            "role": "diagnosis",
            "role_contract": role_contract("diagnosis").to_dict(),
            "allowed_evidence_spans": [
                {
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                    "evidence_quote": "confirmed grounded evidence",
                }
            ],
        },
    )

    assert f'"error_code": "{error_code}"' in prompt
    assert f'"missing_fields": {json.dumps(missing_fields)}' in prompt


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


def test_pipeline_cleanup_removes_stale_funnel_artifacts_on_preflight_failure(
    tmp_path: Path,
):
    package = tmp_path / "package"
    package.mkdir()
    stale_paths = [
        package / "issue_ledger.json",
        package / "deterministic_proposal_report.json",
        package / "deterministic_proposal_report.md",
    ]
    for path in stale_paths:
        path.write_text("stale artifact\n", encoding="utf-8")
    client = SequencedAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_context_tokens_per_stage=0),
    )

    assert result.stop_reason == "invalid_config"
    assert client.calls == []
    for path in stale_paths:
        assert not path.exists()


@pytest.mark.parametrize(
    ("config", "expected_error"),
    [
        (LLMAgentPipelineConfig(max_stage_retries=9), "max_stage_retries"),
        (LLMAgentPipelineConfig(max_subagent_tasks=51), "max_subagent_tasks"),
        (LLMAgentPipelineConfig(max_parallel_subagents=9), "max_parallel_subagents"),
        (LLMAgentPipelineConfig(max_proposals_per_run=201), "max_proposals_per_run"),
    ],
)
def test_pipeline_rejects_out_of_bounds_subagent_config(
    tmp_path: Path,
    config: LLMAgentPipelineConfig,
    expected_error: str,
):
    package = tmp_path / "package"
    package.mkdir()
    client = SequencedAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=config,
    )

    assert result.stop_reason == "invalid_config"
    assert result.proposal_ids == []
    assert client.calls == []
    _assert_preflight_failure_artifacts(
        package,
        stop_reason="invalid_config",
        expected_error=expected_error,
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


def _stratified_schema_issues(count_per_family: int) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for index in range(count_per_family):
        issues.append(
            _agent_schema_issue(
                f"edge-direction-{index}",
                issue_kind="reverse_clinical_manifestation",
                keywords="needs direction review",
                candidate_predicates=["has_manifestation"],
            )
        )
    for index in range(count_per_family):
        issues.append(
            _agent_schema_issue(
                f"edge-treatment-{index}",
                issue_kind="treatment_domain_range_mismatch",
                keywords="treatment domain review",
                candidate_predicates=["has_indication"],
            )
        )
    for index in range(count_per_family):
        issues.append(
            _agent_schema_issue(
                f"edge-diagnosis-{index}",
                issue_kind="diagnostic_evidence_direction_mismatch",
                keywords="diagnostic evidence review",
                candidate_predicates=["has_diagnostic_criterion"],
            )
        )
    for index in range(count_per_family):
        issues.append(
            _agent_schema_issue(
                f"edge-prevention-{index}",
                issue_kind="prevention_population_direction_mismatch",
                keywords="vaccine population review",
                candidate_predicates=["recommended_for"],
            )
        )
    for index in range(count_per_family):
        issues.append(
            _agent_schema_issue(
                f"edge-split-{index}",
                issue_kind="multi_predicate_edge_split_needed",
                keywords="split review",
                candidate_predicates=["has_indication", "recommended_for"],
            )
        )
    return issues


def _agent_schema_issue(
    edge_id: str,
    *,
    issue_kind: str,
    keywords: str,
    candidate_predicates: list[str],
) -> dict[str, object]:
    return {
        "issue_kind": issue_kind,
        "edge_id": edge_id,
        "source": "flu",
        "target": "oseltamivir",
        "keywords": keywords,
        "source_id": "chunk-1",
        "file_path": "guide.md",
        "risk": "medium",
        "suggested_action": "replace_relation",
        "candidate_predicates": candidate_predicates,
    }


def _quality_report_note_proposal(
    proposal_id: str,
    *,
    confidence: float,
    proposed_change: str,
) -> dict[str, object]:
    return {
        "id": proposal_id,
        "type": "quality_report_note",
        "target": "quality_report.md",
        "proposed_change": proposed_change,
        "reason": "Grounded proposal used to exercise merge accounting.",
        "evidence": [
            "source_id: chunk-1; file_path: guide.md; item_id: flu-fever"
        ],
        "confidence": confidence,
        "risk": "low",
        "requires_approval": False,
        "expected_metric_change": {},
    }


def _write_agent_package(
    package: Path,
    *,
    include_medical_schema_issues: bool = True,
) -> None:
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
    details = {
        "hierarchy_branches": {
            "required": [{"key": "symptom", "label": "Symptom"}],
            "present": [{"key": "disease", "label": "Disease"}],
            "missing": [{"key": "symptom", "label": "Symptom"}],
        },
    }
    if include_medical_schema_issues:
        details["medical_schema_issues"] = [
            {
                "issue_kind": "schema_review",
                "edge_id": "flu-fever",
                "source": "flu",
                "target": "entity fever",
                "keywords": "schema_relation",
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "risk": "medium",
                "suggested_action": "review_relation_schema",
                "candidate_predicates": ["schema_relation"],
            }
        ]
    _write_json(
        snapshot_dir / "quality_score.json",
        {
            "overall": 74,
            "subscores": {"hierarchy_completeness": 55},
            "metrics": {"missing_hierarchy_branch_count": 1},
            "details": details,
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


def _write_residual_route_agent_package(package: Path) -> None:
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
                    "id": "cough",
                    "label": "Cough",
                    "entity_type": "Symptom",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                },
            ],
            "edges": [
                {
                    "id": "edge-deterministic",
                    "source": "cough",
                    "target": "flu",
                    "keywords": "clinical_manifestation",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                },
                {
                    "id": "edge-residual",
                    "source": "oseltamivir",
                    "target": "children",
                    "keywords": "recommended_for",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                },
            ],
            "metadata": {"profile": "medical_kg"},
        },
    )
    _write_json(
        snapshot_dir / "quality_score.json",
        {
            "overall": 74,
            "subscores": {"relation_semantics": 30},
            "metrics": {"medical_schema_issue_count": 2},
            "details": {
                "medical_schema_issues": [
                    {
                        "issue_kind": "reverse_clinical_manifestation",
                        "edge_id": "edge-deterministic",
                        "source": "cough",
                        "target": "flu",
                        "keywords": "clinical_manifestation",
                        "source_id": "chunk-1",
                        "file_path": "guide.md",
                        "suggested_action": "replace_relation",
                        "candidate_predicates": ["has_manifestation"],
                    },
                    {
                        "issue_kind": "conflicting_recommendation_safety_scope",
                        "edge_id": "edge-residual",
                        "source": "oseltamivir",
                        "target": "children",
                        "keywords": "recommended_for",
                        "source_id": "chunk-1",
                        "file_path": "guide.md",
                        "suggested_action": "review_conflict",
                        "candidate_predicates": [
                            "recommended_for",
                            "contraindicated_for",
                        ],
                    },
                ]
            },
            "findings": [],
            "critical_blockers": [],
        },
    )
    (package / "kb_context.md").write_text("# KB Context\n", encoding="utf-8")
    (package / "accepted_changes.md").write_text("# Accepted Changes\n", encoding="utf-8")
    (package / "rejected_changes.md").write_text("# Rejected Changes\n", encoding="utf-8")


_BULK_COMPLICATIONS = (
    "流感病毒性肺炎",
    "流感相关性脑病",
    "急性坏死性脑病",
    "急性呼吸窘迫综合征",
    "病毒性心肌炎",
    "横纹肌溶解症",
    "急性肾损伤",
    "中耳炎",
    "鼻窦炎",
    "继发性细菌性肺炎",
    "急性播散性脑脊髓炎",
    "脓毒性休克",
)


def _write_bulk_complication_agent_package(package: Path, *, issue_count: int) -> None:
    snapshot_dir = package / "snapshots"
    snapshot_dir.mkdir(parents=True)
    nodes = [
        {
            "id": "流行性感冒",
            "label": "流行性感冒",
            "entity_type": "disease",
            "source_id": "chunk-1",
            "file_path": "guide.md",
        }
    ]
    edges = []
    issues = []
    for index in range(issue_count):
        complication = _BULK_COMPLICATIONS[index % len(_BULK_COMPLICATIONS)]
        edge_id = f"{complication}->流行性感冒"
        nodes.append(
            {
                "id": complication,
                "label": complication,
                "entity_type": "complication",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        )
        edges.append(
            {
                "id": edge_id,
                "source": complication,
                "target": "流行性感冒",
                "keywords": "并发风险",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        )
        issues.append(
            {
                "issue_kind": "legacy_overloaded_relation",
                "edge_id": edge_id,
                "source": complication,
                "target": "流行性感冒",
                "keywords": "并发风险",
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "risk": "medium",
                "suggested_action": "replace_relation",
                "candidate_predicates": [
                    "has_complication",
                    "may_cause_adverse_reaction",
                ],
            }
        )
    _write_json(
        snapshot_dir / "kg_snapshot.json",
        {
            "workspace": "demo",
            "generated_at": "2026-06-18T00:00:00Z",
            "source_files": ["guide.md"],
            "nodes": nodes,
            "edges": edges,
            "metadata": {"profile": "medical_kg"},
        },
    )
    _write_json(
        snapshot_dir / "quality_score.json",
        {
            "overall": 74,
            "subscores": {"relation_semantics": 30},
            "metrics": {"medical_schema_issue_count": issue_count},
            "details": {"medical_schema_issues": issues},
            "findings": [],
            "critical_blockers": [],
        },
    )
    (package / "kb_context.md").write_text("# KB Context\n", encoding="utf-8")
    (package / "accepted_changes.md").write_text("# Accepted Changes\n", encoding="utf-8")
    (package / "rejected_changes.md").write_text("# Rejected Changes\n", encoding="utf-8")


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
