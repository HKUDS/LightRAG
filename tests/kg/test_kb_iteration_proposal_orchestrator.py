from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.kb_iteration.deterministic_proposals.base import CandidateGenerationResult
from lightrag.kb_iteration.issue_ledger import scan_deterministic_candidates
from lightrag.kb_iteration.models import ImprovementProposal
from lightrag.kb_iteration.proposal_orchestrator import (
    build_llm_residual_task_packs,
    build_proposal_task_packs,
    merge_subagent_proposals,
)


def test_build_task_packs_reads_snapshots_and_builds_reversed_manifestation_action(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "reverse_clinical_manifestation",
                "edge_id": "edge-reversed",
                "suggested_action": "replace_relation",
                "candidate_predicates": ["has_manifestation"],
            }
        ],
        edges=[
            _edge(
                "edge-reversed",
                source="dry-cough",
                target="flu",
                keywords="clinical_manifestation",
            )
        ],
    )

    packs = build_proposal_task_packs(package)

    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    candidate = next(
        candidate
        for candidate in action_candidates
        if candidate["action_payload"]["edge_id"] == "edge-reversed"
    )
    assert candidate["proposal_type"] == "medical_relation_schema_migration"
    assert candidate["action_payload"] == {
        "action": "replace_relation",
        "edge_id": "edge-reversed",
        "expected_source": "dry-cough",
        "expected_target": "flu",
        "current_keywords": "clinical_manifestation",
        "new_source": "flu",
        "new_target": "dry-cough",
        "new_keywords": "has_manifestation",
        "qualifiers": {},
    }


def test_build_task_packs_builds_reversed_causative_agent_action_candidate(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "reverse_causative_agent",
                "edge_id": "edge-virus-flu",
                "source": "virus",
                "target": "flu",
                "keywords": "causes; etiologic_agent",
                "new_source": "flu",
                "new_target": "virus",
                "candidate_predicates": ["causative_agent"],
                "qualifiers": {"certainty": "known"},
            }
        ],
    )

    packs = build_proposal_task_packs(package)

    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    candidate = next(
        candidate
        for candidate in action_candidates
        if candidate["action_payload"]["edge_id"] == "edge-virus-flu"
    )
    assert candidate["proposal_type"] == "medical_relation_schema_migration"
    assert candidate["action_payload"] == {
        "action": "replace_relation",
        "edge_id": "edge-virus-flu",
        "expected_source": "virus",
        "expected_target": "flu",
        "current_keywords": "causes; etiologic_agent",
        "new_source": "flu",
        "new_target": "virus",
        "new_keywords": "causative_agent",
        "qualifiers": {"certainty": "known"},
    }


def test_build_task_packs_skips_previously_applied_edges(
    tmp_path: Path,
) -> None:
    edge = _edge(
        "edge-already-normalized",
        source="心电图",
        target="心脏损伤",
        keywords="supports_or_refutes",
    )
    edge["normalized_by"] = "kb_iteration_apply"
    edge["accepted_proposal_ids"] = "prop-diagnosis-002-ecg-cardiac-injury"
    package = _make_package(
        tmp_path,
        issues=[
            _issue(
                "edge-already-normalized",
                issue_kind="canonical_relation_domain_range_mismatch",
                source="心电图",
                target="心脏损伤",
                keywords="supports_or_refutes",
                candidate_predicates=["supports_or_refutes"],
            )
        ],
        edges=[edge],
    )

    packs = build_proposal_task_packs(package)

    assert packs == []


def test_full_scan_does_not_requeue_already_applied_action(
    tmp_path: Path,
) -> None:
    edge = _edge(
        "edge-already-normalized",
        source="ecg",
        target="cardiac-injury",
        keywords="supports_or_refutes",
    )
    edge["normalized_by"] = "kb_iteration_apply"
    edge["accepted_proposal_ids"] = "prop-diagnosis-002-ecg-cardiac-injury"
    package = _make_package(
        tmp_path,
        issues=[
            _issue(
                "edge-already-normalized",
                issue_kind="canonical_relation_domain_range_mismatch",
                source="ecg",
                target="cardiac-injury",
                keywords="supports_or_refutes",
                candidate_predicates=["supports_or_refutes"],
            )
        ],
        edges=[edge],
    )

    scan = scan_deterministic_candidates(package)

    route = _single_route_for_edge(scan, "edge-already-normalized")
    assert _route_state(route) in {"stale", "duplicate"}


def test_full_scan_does_not_repeat_rejected_action_fingerprint(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            _issue(
                "edge-rejected-before",
                issue_kind="reverse_clinical_manifestation",
                source="cough",
                target="flu",
                keywords="clinical_manifestation",
                candidate_predicates=["has_manifestation"],
            )
        ],
        edges=[
            _edge(
                "edge-rejected-before",
                source="cough",
                target="flu",
                keywords="clinical_manifestation",
            )
        ],
    )
    (package / "rejected_changes.md").write_text(
        "# Rejected Changes\n\n"
        "## prop-action-candidate-edge-rejected-before\n\n"
        "```json\n"
        + json.dumps(
            {
                "proposal_id": "prop-action-candidate-edge-rejected-before",
                "proposal_type": "medical_relation_schema_migration",
                "proposal_target": "edge:edge-rejected-before",
                "decision": "reject",
                "action_payload": {
                    "action": "replace_relation",
                    "edge_id": "edge-rejected-before",
                    "expected_source": "cough",
                    "expected_target": "flu",
                    "current_keywords": "clinical_manifestation",
                    "new_source": "flu",
                    "new_target": "cough",
                    "new_keywords": "has_manifestation",
                    "qualifiers": {},
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n```\n",
        encoding="utf-8",
    )

    scan = scan_deterministic_candidates(package)

    route = _single_route_for_edge(scan, "edge-rejected-before")
    assert _route_state(route) == "duplicate"


def test_full_scan_drops_stale_expected_keywords(tmp_path: Path) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            _issue(
                "edge-keywords-changed",
                issue_kind="treatment_domain_range_mismatch",
                keywords="old_recommended_treatment",
                candidate_predicates=["has_indication"],
            )
        ],
        edges=[
            _edge("edge-keywords-changed", keywords="recommended_treatment")
        ],
    )

    scan = scan_deterministic_candidates(package)

    route = _single_route_for_edge(scan, "edge-keywords-changed")
    assert _route_state(route) == "stale"


def test_same_edge_conflicting_candidates_enter_conflict_ledger(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            _issue(
                "edge-conflicting-deterministic",
                issue_kind="reverse_clinical_manifestation",
                source="cough",
                target="flu",
                keywords="clinical_manifestation",
                candidate_predicates=["has_manifestation"],
            ),
            _issue(
                "edge-conflicting-deterministic",
                issue_kind="reverse_causative_agent",
                source="cough",
                target="flu",
                keywords="clinical_manifestation",
                candidate_predicates=["causative_agent"],
            ),
        ],
        edges=[
            _edge(
                "edge-conflicting-deterministic",
                source="cough",
                target="flu",
                keywords="clinical_manifestation",
            )
        ],
    )

    scan = scan_deterministic_candidates(package)

    routes = _routes_for_edge(scan, "edge-conflicting-deterministic")
    assert len(routes) == 2
    assert {_route_state(route) for route in routes} == {"llm_residual"}
    assert {_route_reason_code(route) for route in routes} == {"SAME_EDGE_CONFLICT"}
    conflict_rejections = [
        rejection
        for rejection in scan.rejections
        if rejection.get("error_code") == "SAME_EDGE_CONFLICT"
    ]
    assert len(conflict_rejections) == 2
    for rejection in conflict_rejections:
        _assert_candidate_rejection_contract(rejection)


def test_same_edge_conflict_residual_packs_do_not_regenerate_action_candidates(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            _issue(
                "edge-conflicting-deterministic",
                issue_kind="reverse_clinical_manifestation",
                source="cough",
                target="flu",
                keywords="clinical_manifestation",
                candidate_predicates=["has_manifestation"],
            ),
            _issue(
                "edge-conflicting-deterministic",
                issue_kind="reverse_causative_agent",
                source="cough",
                target="flu",
                keywords="clinical_manifestation",
                candidate_predicates=["causative_agent"],
            ),
        ],
        edges=[
            _edge(
                "edge-conflicting-deterministic",
                source="cough",
                target="flu",
                keywords="clinical_manifestation",
            )
        ],
    )

    scan = scan_deterministic_candidates(package)
    packs = build_llm_residual_task_packs(package, scan)

    routes = _routes_for_edge(scan, "edge-conflicting-deterministic")
    assert len(routes) == 2
    assert {_route_state(route) for route in routes} == {"llm_residual"}
    assert {_route_reason_code(route) for route in routes} == {"SAME_EDGE_CONFLICT"}
    assert scan.candidates == []
    assert packs
    packed_refs = {
        issue["issue_ref"]
        for pack in packs
        for issue in pack.to_dict()["issues"]
    }
    assert packed_refs == {route.issue_ref for route in routes}
    for pack in packs:
        pack_dict = pack.to_dict()
        assert pack_dict["action_candidates"] == []
        assert pack_dict["covered_issue_refs"] == []
        assert pack_dict["execution_mode"] != "deterministic_only"


def test_build_task_packs_builds_typed_influenza_pathogen_expansion_candidate(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "reverse_causative_agent",
                "edge_id": "流感病毒->甲型流感",
                "source": "流感病毒",
                "target": "甲型流感",
                "keywords": "病原分型",
                "source_id": "chunk-020",
                "file_path": "儿童流感指南.pdf",
                "candidate_predicates": ["causative_agent"],
            }
        ],
    )

    packs = build_proposal_task_packs(package)

    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    assert all(
        candidate["proposal_type"] != "medical_relation_schema_migration"
        for candidate in action_candidates
    )
    candidate = action_candidates[0]
    assert candidate["proposal_type"] == "candidate_kg_expansion"
    assert candidate["target"] == "kg:candidate:typed-influenza-pathogen:甲型流感"
    assert candidate["action_payload"] == {
        "edge_id": "流感病毒->甲型流感",
        "candidate_nodes": [
            {
                "id": "甲型流感病毒",
                "label": "甲型流感病毒",
                "entity_type": "pathogen",
                "description": "甲型流感的精确病原体，属于流感病毒。",
            }
        ],
        "candidate_edges": [
            {
                "source": "甲型流感病毒",
                "target": "流感病毒",
                "source_type": "Pathogen",
                "target_type": "Pathogen",
                "keywords": "is_a",
                "source_id": "chunk-020",
                "file_path": "儿童流感指南.pdf",
                "qualifiers": {},
                "description": "甲型流感病毒是流感病毒的一个分型病原体。",
            },
            {
                "source": "甲型流感",
                "target": "甲型流感病毒",
                "source_type": "Disease",
                "target_type": "Pathogen",
                "keywords": "causative_agent",
                "source_id": "chunk-020",
                "file_path": "儿童流感指南.pdf",
                "qualifiers": {},
                "description": "甲型流感应指向精确的甲型流感病毒病原体。",
            },
        ],
        "retire_edges": [
            {
                "source": "流感病毒",
                "target": "甲型流感",
                "keywords": "病原分型",
                "reason": "用精确病原体节点和疾病 causative_agent 关系替代泛化分型边。",
            }
        ],
        "source_id": "chunk-020",
        "file_path": "儿童流感指南.pdf",
        "evidence_quote": '"id": "流感病毒->甲型流感", "keywords": "病原分型"',
        "why_not_existing": "甲型流感需要精确病原体节点，不能直接指向泛化流感病毒。",
    }


def test_build_task_packs_builds_typed_influenza_candidate_for_legacy_pathogen_split(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "legacy_overloaded_relation",
                "edge_id": "流感病毒->乙型流感",
                "source": "流感病毒",
                "target": "乙型流感",
                "keywords": "病原分型",
                "source_id": "chunk-020",
                "file_path": "儿童流感指南.pdf",
                "candidate_predicates": ["is_a", "causative_agent"],
                "guidance": "病毒亚型层级用子类 -> is_a -> 父类；疾病指向致病病原体用 causative_agent。",
            }
        ],
    )

    packs = build_proposal_task_packs(package)

    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    candidate = action_candidates[0]
    assert candidate["proposal_type"] == "candidate_kg_expansion"
    assert candidate["action_payload"]["candidate_nodes"][0]["id"] == "乙型流感病毒"
    assert candidate["action_payload"]["candidate_edges"] == [
        {
            "source": "乙型流感病毒",
            "target": "流感病毒",
            "source_type": "Pathogen",
            "target_type": "Pathogen",
            "keywords": "is_a",
            "source_id": "chunk-020",
            "file_path": "儿童流感指南.pdf",
            "qualifiers": {},
            "description": "乙型流感病毒是流感病毒的一个分型病原体。",
        },
        {
            "source": "乙型流感",
            "target": "乙型流感病毒",
            "source_type": "Disease",
            "target_type": "Pathogen",
            "keywords": "causative_agent",
            "source_id": "chunk-020",
            "file_path": "儿童流感指南.pdf",
            "qualifiers": {},
            "description": "乙型流感应指向精确的乙型流感病毒病原体。",
        },
    ]


def test_build_task_packs_builds_complication_candidate_from_node_types(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        nodes=[
            _node("流行性感冒", entity_type="disease"),
            _node("肺炎", entity_type="complication"),
        ],
        issues=[
            _issue(
                "肺炎->流行性感冒",
                source="肺炎",
                target="流行性感冒",
                keywords="并发风险",
                candidate_predicates=["has_complication", "may_cause_adverse_reaction"],
            )
        ],
        edges=[
            _edge(
                "肺炎->流行性感冒",
                source="肺炎",
                target="流行性感冒",
                keywords="并发风险",
            )
        ],
    )

    packs = build_proposal_task_packs(package)

    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    candidate = next(
        candidate
        for candidate in action_candidates
        if candidate["action_payload"]["edge_id"] == "肺炎->流行性感冒"
    )
    assert candidate["proposal_type"] == "medical_relation_schema_migration"
    assert candidate["action_payload"] == {
        "action": "replace_relation",
        "edge_id": "肺炎->流行性感冒",
        "expected_source": "肺炎",
        "expected_target": "流行性感冒",
        "current_keywords": "并发风险",
        "new_source": "流行性感冒",
        "new_target": "肺炎",
        "new_keywords": "has_complication",
        "qualifiers": {},
    }


def test_build_task_packs_skips_outcome_and_severity_as_complication_candidates(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        nodes=[
            _node("流行性感冒", entity_type="disease"),
            _node("流感相关死亡", entity_type="complication"),
            _node("流感相关住院", entity_type="complication"),
            _node("低出生体重", entity_type="complication"),
            _node("重症", entity_type="complication"),
            _node("缺血性中风", entity_type="complication"),
        ],
        issues=[
            _issue(
                f"流行性感冒->{target}",
                source="流行性感冒",
                target=target,
                keywords="并发风险",
                candidate_predicates=[
                    "has_complication",
                    "may_cause_adverse_reaction",
                ],
            )
            for target in (
                "流感相关死亡",
                "流感相关住院",
                "低出生体重",
                "重症",
                "缺血性中风",
            )
        ],
    )

    packs = build_proposal_task_packs(package)

    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    assert action_candidates == []


def test_build_task_packs_skips_non_influenza_disease_complication_candidates(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        nodes=[
            _node("新型冠状病毒感染", entity_type="disease"),
            _node("急性呼吸窘迫综合征", entity_type="complication"),
            _node("耐甲氧西林金黄色葡萄球菌感染", entity_type="disease"),
            _node("脓胸", entity_type="complication"),
        ],
        issues=[
            _issue(
                "新型冠状病毒感染->急性呼吸窘迫综合征",
                source="新型冠状病毒感染",
                target="急性呼吸窘迫综合征",
                keywords="并发风险",
                candidate_predicates=[
                    "has_complication",
                    "may_cause_adverse_reaction",
                ],
            ),
            _issue(
                "脓胸->耐甲氧西林金黄色葡萄球菌感染",
                source="脓胸",
                target="耐甲氧西林金黄色葡萄球菌感染",
                keywords="并发风险",
                candidate_predicates=[
                    "has_complication",
                    "may_cause_adverse_reaction",
                ],
            ),
        ],
    )

    packs = build_proposal_task_packs(package)

    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    assert action_candidates == []


def test_build_task_packs_builds_reversed_pathogen_subtype_action_candidate(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "reverse_pathogen_subtype_is_a",
                "edge_id": "edge-virus-subtype",
                "source": "virus",
                "target": "subtype",
                "keywords": "subtype; is-a",
                "new_source": "subtype",
                "new_target": "virus",
                "candidate_predicates": ["is_a"],
            }
        ],
    )

    packs = build_proposal_task_packs(package)

    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    candidate = next(
        candidate
        for candidate in action_candidates
        if candidate["action_payload"]["edge_id"] == "edge-virus-subtype"
    )
    assert candidate["proposal_type"] == "medical_relation_schema_migration"
    assert candidate["action_payload"] == {
        "action": "replace_relation",
        "edge_id": "edge-virus-subtype",
        "expected_source": "virus",
        "expected_target": "subtype",
        "current_keywords": "subtype; is-a",
        "new_source": "subtype",
        "new_target": "virus",
        "new_keywords": "is_a",
        "qualifiers": {},
    }


def test_build_task_packs_builds_orders_test_candidate_for_test_to_disease_edge(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "diagnostic_evidence_direction_mismatch",
                "edge_id": "edge-cbc-flu",
                "source": "血常规",
                "target": "流行性感冒",
                "keywords": "诊断依据",
                "source_type": "Test",
                "target_type": "Disease",
                "candidate_predicates": [
                    "has_diagnostic_criterion",
                    "supports_or_refutes",
                ],
            }
        ],
    )

    packs = build_proposal_task_packs(package)

    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    candidate = next(
        candidate
        for candidate in action_candidates
        if candidate["action_payload"]["edge_id"] == "edge-cbc-flu"
    )
    assert candidate["proposal_type"] == "medical_relation_schema_migration"
    assert candidate["action_payload"] == {
        "action": "replace_relation",
        "edge_id": "edge-cbc-flu",
        "expected_source": "血常规",
        "expected_target": "流行性感冒",
        "current_keywords": "诊断依据",
        "new_source": "流行性感冒",
        "new_target": "血常规",
        "new_keywords": "orders_test",
        "qualifiers": {"indication": "diagnosis_or_severity_assessment"},
        "new_source_type": "Disease",
        "new_target_type": "Test",
    }


def test_build_task_packs_backfills_json_string_qualifiers_from_snapshot(
    tmp_path: Path,
) -> None:
    edge = _edge(
        "edge-oseltamivir-children",
        source="oseltamivir",
        target="children",
        keywords="recommended_for",
    )
    edge["properties"] = {
        "qualifiers": json.dumps({"condition": "influenza"}, ensure_ascii=False)
    }
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "missing_required_relation_qualifier",
                "edge_id": "edge-oseltamivir-children",
                "source": "oseltamivir",
                "target": "children",
                "keywords": "recommended_for",
                "source_type": "Drug",
                "target_type": "Population",
                "candidate_predicates": ["recommended_for"],
                "suggested_action": "replace_relation",
            }
        ],
        edges=[edge],
    )

    candidate = _single_action_candidate(build_proposal_task_packs(package))

    assert candidate["issue_ref"].startswith(
        "medical_schema_issues:treatment:missing_required_relation_qualifier:"
    )
    assert candidate["action_payload"]["qualifiers"] == {
        "condition": "influenza",
        "purpose": "treatment",
    }


def test_build_task_packs_marks_mixed_candidate_coverage_as_hybrid(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "treatment_domain_range_mismatch",
                "edge_id": "edge-oseltamivir-flu",
                "source": "oseltamivir",
                "target": "flu",
                "keywords": "recommended_treatment",
                "source_type": "Drug",
                "target_type": "Disease",
                "candidate_predicates": ["has_indication"],
                "suggested_action": "replace_relation",
            },
            {
                "issue_kind": "missing_required_relation_qualifier",
                "edge_id": "edge-oseltamivir-children",
                "source": "oseltamivir",
                "target": "children",
                "keywords": "recommended_for",
                "source_type": "Drug",
                "target_type": "Population",
                "candidate_predicates": ["recommended_for"],
                "suggested_action": "replace_relation",
            },
        ],
    )

    packs = [pack.to_dict() for pack in build_proposal_task_packs(package)]

    treatment_pack = next(pack for pack in packs if pack["issue_family"] == "treatment")
    assert treatment_pack["execution_mode"] == "hybrid"
    assert len(treatment_pack["action_candidates"]) == 1
    assert len(treatment_pack["covered_issue_refs"]) == 1
    assert len(treatment_pack["residual_issue_refs"]) == 1
    assert treatment_pack["action_candidates"][0]["issue_ref"] in set(
        treatment_pack["covered_issue_refs"]
    )


def test_build_task_packs_records_generator_rejection_for_unknown_family(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "custom_schema_issue",
                "issue_family": "custom_family",
                "edge_id": "edge-custom",
                "source": "source",
                "target": "target",
                "keywords": "custom_relation",
                "source_type": "Disease",
                "target_type": "Symptom",
                "candidate_predicates": ["custom_relation"],
                "suggested_action": "replace_relation",
            }
        ],
    )

    packs = [pack.to_dict() for pack in build_proposal_task_packs(package)]

    pack = next(pack for pack in packs if pack["issue_family"] == "custom_family")
    assert pack["action_candidates"] == []
    rejection = pack["rejected_action_candidates"][0]
    assert rejection["error_code"] == "NO_GENERATOR_FOR_FAMILY"
    _assert_candidate_rejection_contract(
        rejection,
        issue_family="custom_family",
    )


def test_build_task_packs_prevalidation_rejections_include_candidate_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[_issue("edge-invalid-prevalidation")],
    )

    def candidates(
        chunk: list[dict[str, object]],
        **_kwargs: object,
    ) -> CandidateGenerationResult:
        issue = chunk[0]
        return CandidateGenerationResult(
            candidates=[
                {
                    "candidate_id": "candidate-invalid-prevalidation",
                    "proposal_type": "medical_relation_schema_migration",
                    "target": f"edge:{issue['edge_id']}",
                    "issue_ref": issue["issue_ref"],
                    "issue_kind": issue["issue_kind"],
                }
            ],
            rejections=[],
        )

    monkeypatch.setattr(
        "lightrag.kb_iteration.proposal_orchestrator._action_candidates_for_issues",
        candidates,
    )

    packs = [pack.to_dict() for pack in build_proposal_task_packs(package)]

    pack = next(pack for pack in packs if pack["rejected_action_candidates"])
    assert pack["action_candidates"] == []
    rejection = pack["rejected_action_candidates"][0]
    assert rejection["error_code"] == "ACTION_CANDIDATE_INVALID"
    _assert_candidate_rejection_contract(
        rejection,
        issue_family=pack["issue_family"],
    )


@pytest.mark.parametrize(
    "issue_source",
    ["evidence_issues", "generic_relation_issues", "hierarchy_issues"],
)
def test_build_task_packs_keeps_all_structured_issue_sources(
    tmp_path: Path,
    issue_source: str,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[],
        details_overrides={
            issue_source: [
                {
                    "issue_kind": f"{issue_source}_case",
                    "edge_id": f"edge-{issue_source}",
                    "source": "source",
                    "target": "target",
                    "keywords": "related_to",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                    "evidence_quote": f"Evidence for {issue_source}.",
                    "candidate_predicates": ["related_to"],
                    "suggested_action": "review",
                }
            ]
        },
        findings=[
            {
                "severity": "low",
                "category": "fallback",
                "message": "Fallback finding must not replace structured issues.",
                "evidence": ["fallback evidence"],
                "suggested_fix_type": "review",
            }
        ],
    )

    packs = [pack.to_dict() for pack in build_proposal_task_packs(package)]

    packed_issues = [issue for pack in packs for issue in pack["issues"]]
    assert [issue["issue_source"] for issue in packed_issues] == [issue_source]
    assert packed_issues[0]["issue_kind"] == f"{issue_source}_case"


def test_build_task_packs_does_not_orders_test_candidate_for_bare_lab_marker(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "diagnostic_evidence_direction_mismatch",
                "edge_id": "edge-alt-flu",
                "source": "丙氨酸氨基转移酶",
                "target": "流行性感冒",
                "keywords": "诊断依据",
                "source_type": "Test",
                "target_type": "Disease",
                "candidate_predicates": [
                    "has_diagnostic_criterion",
                    "supports_or_refutes",
                ],
            }
        ],
    )

    packs = build_proposal_task_packs(package)

    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    assert action_candidates == []


def test_build_task_packs_builds_diagnostic_criterion_candidate_for_severe_sign(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "legacy_overloaded_relation",
                "edge_id": "鼻翼扇动->流感重型",
                "source": "鼻翼扇动",
                "target": "流感重型",
                "keywords": "诊断依据",
                "source_type": "Symptom",
                "target_type": "Disease",
                "candidate_predicates": [
                    "has_diagnostic_criterion",
                    "criterion_requires",
                    "has_evidence",
                    "supports_or_refutes",
                ],
                "medical_subcase": "diagnostic_evidence_flattening",
            }
        ],
    )

    packs = build_proposal_task_packs(package)

    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    candidate = next(
        candidate
        for candidate in action_candidates
        if candidate["action_payload"]["edge_id"] == "鼻翼扇动->流感重型"
    )
    assert candidate["proposal_type"] == "medical_relation_schema_migration"
    assert candidate["action_payload"] == {
        "action": "replace_relation",
        "edge_id": "鼻翼扇动->流感重型",
        "expected_source": "鼻翼扇动",
        "expected_target": "流感重型",
        "current_keywords": "诊断依据",
        "new_source": "流感重型",
        "new_target": "鼻翼扇动",
        "new_keywords": "has_diagnostic_criterion",
        "qualifiers": {"context": "severe_influenza"},
    }


def test_build_task_packs_skips_replace_candidate_without_current_keywords(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "reverse_causative_agent",
                "edge_id": "edge-no-keywords",
                "source": "virus",
                "target": "flu",
                "new_source": "flu",
                "new_target": "virus",
                "candidate_predicates": ["causative_agent"],
            }
        ],
        edges=[
            _edge(
                "edge-no-keywords",
                source="virus",
                target="flu",
                keywords="",
            )
        ],
    )

    packs = build_proposal_task_packs(package)

    assert [
        candidate for pack in packs for candidate in pack.action_candidates
    ] == []


def test_build_task_packs_keeps_ambiguous_multi_predicate_split_for_subagent(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "multi_predicate_edge_split_needed",
                "edge_id": "edge-multi",
                "source": "flu",
                "target": "oseltamivir",
                "keywords": "recommended treatment; dosing regimen",
                "candidate_predicates": [
                    "has_dosing_regimen",
                    "",
                    "has_indication",
                    5,
                    "recommended_for",
                ],
            }
        ],
    )

    packs = build_proposal_task_packs(package)

    split_pack = next(
        pack
        for pack in packs
        if pack.issues[0]["issue_kind"] == "multi_predicate_edge_split_needed"
    )
    assert split_pack.action_candidates == []
    assert split_pack.issues[0]["candidate_predicates"] == [
        "has_dosing_regimen",
        "",
        "has_indication",
        5,
        "recommended_for",
    ]


def test_build_task_packs_builds_executable_multi_predicate_split_action_candidate(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "multi_predicate_edge_split_needed",
                "edge_id": "edge-multi",
                "source": "zanamivir",
                "target": "mixed-target",
                "keywords": "recommended_treatment,applies_to",
                "repair_options": [
                    {
                        "predicate": "has_indication",
                        "new_source": "zanamivir",
                        "new_target": "influenza",
                        "source_type": "Drug",
                        "target_type": "Disease",
                        "qualifiers": {"purpose": "treatment"},
                        "auto_fixable": True,
                    },
                    {
                        "predicate": "recommended_for",
                        "new_source": "zanamivir",
                        "new_target": "children",
                        "source_type": "Drug",
                        "target_type": "Population",
                        "qualifiers": {
                            "purpose": "treatment",
                            "age_min": 7,
                            "age_unit": "year",
                            "route": "inhalation",
                        },
                        "auto_fixable": True,
                    },
                ],
            }
        ],
    )

    packs = build_proposal_task_packs(package)

    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    assert len(action_candidates) == 1
    candidate = action_candidates[0]
    assert candidate["candidate_id"].startswith("ac-")
    assert candidate["proposal_type"] == "medical_fact_role_split"
    assert candidate["target"] == "edge:edge-multi"
    assert candidate["issue_kind"] == "multi_predicate_edge_split_needed"
    assert candidate["action_payload"] == {
        "action": "split_relation",
        "edge_id": "edge-multi",
        "expected_source": "zanamivir",
        "expected_target": "mixed-target",
        "current_keywords": "recommended_treatment,applies_to",
        "retire_original": True,
        "new_edges": [
            {
                "source": "zanamivir",
                "target": "influenza",
                "predicate": "has_indication",
                "source_type": "Drug",
                "target_type": "Disease",
                "qualifiers": {"purpose": "treatment"},
            },
            {
                "source": "zanamivir",
                "target": "children",
                "predicate": "recommended_for",
                "source_type": "Drug",
                "target_type": "Population",
                "qualifiers": {
                    "purpose": "treatment",
                    "age_min": 7,
                    "age_unit": "year",
                    "route": "inhalation",
                },
            },
        ],
    }


def test_build_task_packs_skips_split_candidate_without_valid_predicates(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "multi_predicate_edge_split_needed",
                "edge_id": "edge-empty-predicates",
                "source": "flu",
                "target": "oseltamivir",
                "keywords": "recommended treatment; dosing regimen",
                "candidate_predicates": ["", 5, None],
            },
            {
                "issue_kind": "multi_predicate_edge_split_needed",
                "edge_id": "edge-missing-predicates",
                "source": "flu",
                "target": "zanamivir",
                "keywords": "recommended treatment; dosing regimen",
            }
        ],
    )

    packs = build_proposal_task_packs(package)

    assert [
        candidate for pack in packs for candidate in pack.action_candidates
    ] == []


def test_build_task_packs_skips_split_candidate_without_current_keywords(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "multi_predicate_edge_split_needed",
                "edge_id": "edge-no-keywords",
                "source": "flu",
                "target": "oseltamivir",
                "candidate_predicates": [
                    "has_dosing_regimen",
                    "has_indication",
                ],
            }
        ],
        edges=[
            _edge(
                "edge-no-keywords",
                source="flu",
                target="oseltamivir",
                keywords="",
            )
        ],
    )

    packs = build_proposal_task_packs(package)

    assert [
        candidate for pack in packs for candidate in pack.action_candidates
    ] == []


def test_merge_deduplicates_replace_relation_by_new_edge_outcome() -> None:
    deterministic = _proposal(
        "prop-action-candidate-low-oxygen",
        target="edge:low-oxygen->flu",
        action_payload={
            "action": "replace_relation",
            "edge_id": "low-oxygen->flu",
            "expected_source": "low-oxygen",
            "expected_target": "flu",
            "current_keywords": "clinical manifestation",
            "new_source": "flu",
            "new_target": "low-oxygen",
            "new_keywords": "has_manifestation",
            "qualifiers": {},
        },
    )
    llm_duplicate = _proposal(
        "prop-medical_relation_schema_migration-duplicate",
        target="low-oxygen->flu",
        action_payload={
            "action": "replace_relation",
            "edge_id": "low-oxygen->flu",
            "expected_source": "flu",
            "expected_target": "low-oxygen",
            "current_keywords": "has_manifestation",
            "new_source": "flu",
            "new_target": "low-oxygen",
            "new_keywords": "has_manifestation",
            "qualifiers": {},
        },
    )

    merged = merge_subagent_proposals(
        [[llm_duplicate], [deterministic]],
        max_proposals=20,
    )

    assert [proposal.id for proposal in merged.proposals] == [
        "prop-action-candidate-low-oxygen"
    ]
    assert merged.dropped == [
        {
            "proposal_id": "prop-medical_relation_schema_migration-duplicate",
            "target": "low-oxygen->flu",
            "reason": "duplicate_action_payload",
        }
    ]


def test_build_task_packs_groups_medical_schema_issues_by_family(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            _issue(
                "edge-manifestation",
                issue_kind="reverse_clinical_manifestation",
                keywords="clinical_manifestation",
                candidate_predicates=["has_manifestation"],
            ),
            _issue(
                "edge-treatment",
                keywords="recommended_treatment",
                candidate_predicates=["has_indication", "recommends"],
            ),
            _issue(
                "edge-diagnosis",
                keywords="diagnostic_basis",
                candidate_predicates=[
                    "has_diagnostic_criterion",
                    "criterion_requires",
                ],
            ),
            _issue(
                "edge-evidence",
                keywords="diagnostic_basis",
                candidate_predicates=["has_evidence"],
                medical_subcase="diagnostic_evidence_flattening",
            ),
        ],
    )

    packs = [pack.to_dict() for pack in build_proposal_task_packs(package)]
    packs_by_family = {
        str(pack["issue_family"]): pack for pack in packs
    }

    assert set(packs_by_family) >= {
        "direction",
        "treatment",
        "diagnosis",
    }
    assert packs_by_family["direction"]["role"] == "clinical_modeling"
    assert packs_by_family["treatment"]["role"] == "treatment"
    assert packs_by_family["diagnosis"]["role"] == "diagnosis"
    assert _pack_edge_ids(packs_by_family["direction"]) == ["edge-manifestation"]
    assert _pack_edge_ids(packs_by_family["treatment"]) == ["edge-treatment"]
    assert _pack_edge_ids(packs_by_family["diagnosis"]) == [
        "edge-diagnosis",
        "edge-evidence",
    ]
    edge_ids = [edge_id for pack in packs for edge_id in _pack_edge_ids(pack)]
    assert len(edge_ids) == len(set(edge_ids))


def test_build_task_packs_includes_entity_cleanup_issues_with_schema_issues(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            _issue(
                "edge-treatment",
                keywords="recommended_treatment",
                candidate_predicates=["has_indication"],
            )
        ],
        entity_cleanup_issues=[
            {
                "issue_kind": "synonym_duplicate",
                "suggested_action": "merge_synonym_nodes",
                "canonical_label": "influenza",
                "entity_type": "disease",
                "node_ids": ["flu", "influenza"],
                "nodes": [
                    {"node_id": "flu", "label": "flu"},
                    {"node_id": "influenza", "label": "influenza"},
                ],
            }
        ],
    )

    packs = [pack.to_dict() for pack in build_proposal_task_packs(package)]

    packed_issues = [issue for pack in packs for issue in pack["issues"]]
    assert {
        (issue["issue_source"], issue["issue_kind"])
        for issue in packed_issues
    } >= {
        ("medical_schema_issues", "legacy_overloaded_relation"),
        ("entity_cleanup_issues", "synonym_duplicate"),
    }
    cleanup_pack = next(
        pack for pack in packs if pack["issue_family"] == "entity_cleanup"
    )
    assert cleanup_pack["role"] == "schema_repair"
    assert cleanup_pack["issues"][0]["issue_ref"].startswith(
        "entity_cleanup_issues:entity_cleanup:synonym_duplicate:"
    )


def test_build_task_packs_generates_value_node_to_qualifier_candidate_from_cleanup_issue(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[],
        nodes=[
            _node("oseltamivir", entity_type="DrugIngredient"),
            _node("flu", entity_type="Disease"),
            _node("dose-75mg", entity_type="Dosage"),
        ],
        edges=[
            _edge(
                "edge-dose",
                source="oseltamivir",
                target="dose-75mg",
                keywords="has_value",
            ),
            _edge(
                "edge-indication",
                source="oseltamivir",
                target="flu",
                keywords="has_indication",
            ),
        ],
        entity_cleanup_issues=[
            {
                "issue_kind": "value_node_to_qualifier",
                "suggested_action": "convert_to_qualifier",
                "node_id": "dose-75mg",
                "label": "75 mg",
                "entity_type": "Dosage",
                "qualifier_value": "75 mg",
                "connected_edge_ids": ["edge-dose"],
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
    )

    packs = build_proposal_task_packs(package)

    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    assert len(action_candidates) == 1
    candidate = action_candidates[0]
    assert candidate["proposal_type"] == "value_node_to_qualifier"
    assert candidate["target"] == "node:dose-75mg"
    assert candidate["issue_kind"] == "value_node_to_qualifier"
    assert candidate["action_payload"] == {
        "value_node_id": "dose-75mg",
        "incident_edge_id": "edge-dose",
        "expected_incident_keywords": "has_value",
        "carrier_edge_id": "edge-indication",
        "carrier_edge_source": "oseltamivir",
        "carrier_edge_target": "flu",
        "expected_carrier_keywords": "has_indication",
        "carrier_source_type": "Drug",
        "carrier_target_type": "Disease",
        "qualifier_key": "dose",
        "qualifier_value": "75 mg",
    }


def test_entity_cleanup_generator_detects_chinese_frequency_label(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[],
        nodes=[
            _node("oseltamivir", entity_type="Drug"),
            _node("flu", entity_type="Disease"),
            _node("freq-daily-twice", entity_type="Unknown"),
        ],
        edges=[
            _edge(
                "edge-frequency",
                source="oseltamivir",
                target="freq-daily-twice",
                keywords="has_value",
            ),
            _edge(
                "edge-indication",
                source="oseltamivir",
                target="flu",
                keywords="has_indication",
            ),
        ],
        entity_cleanup_issues=[
            {
                "issue_kind": "value_node_to_qualifier",
                "suggested_action": "convert_to_qualifier",
                "node_id": "freq-daily-twice",
                "label": "每日2次",
                "entity_type": "Unknown",
                "qualifier_value": "每日2次",
                "connected_edge_ids": ["edge-frequency"],
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
    )

    candidate = _single_action_candidate(build_proposal_task_packs(package))

    assert candidate["action_payload"]["qualifier_key"] == "frequency"
    assert candidate["action_payload"]["expected_carrier_keywords"] == "has_indication"


def test_entity_cleanup_generator_rejects_unsupported_carrier_relation(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[],
        nodes=[
            _node("oseltamivir", entity_type="Drug"),
            _node("drug-class", entity_type="DrugClass"),
            _node("dose-75mg", entity_type="Dosage"),
        ],
        edges=[
            _edge(
                "edge-dose",
                source="oseltamivir",
                target="dose-75mg",
                keywords="has_value",
            ),
            _edge(
                "edge-class",
                source="oseltamivir",
                target="drug-class",
                keywords="belongs_to_drug_class",
            ),
        ],
        entity_cleanup_issues=[
            {
                "issue_kind": "value_node_to_qualifier",
                "suggested_action": "convert_to_qualifier",
                "node_id": "dose-75mg",
                "label": "75 mg",
                "entity_type": "Dosage",
                "qualifier_value": "75 mg",
                "connected_edge_ids": ["edge-dose"],
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
    )

    packs = [pack.to_dict() for pack in build_proposal_task_packs(package)]

    cleanup_pack = next(pack for pack in packs if pack["issue_family"] == "entity_cleanup")
    assert cleanup_pack["action_candidates"] == []
    assert cleanup_pack["rejected_action_candidates"][0]["error_code"] == (
        "NO_SCHEMA_COMPATIBLE_CARRIER_EDGE"
    )


def test_entity_cleanup_generator_maps_age_value_node_to_qualifier(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[],
        nodes=[
            _node("oseltamivir", entity_type="Drug"),
            _node("flu", entity_type="Disease"),
            _node("age-12-plus", entity_type="Age"),
        ],
        edges=[
            _edge(
                "edge-age",
                source="oseltamivir",
                target="age-12-plus",
                keywords="has_value",
            ),
            _edge(
                "edge-indication",
                source="oseltamivir",
                target="flu",
                keywords="has_indication",
            ),
        ],
        entity_cleanup_issues=[
            {
                "issue_kind": "value_node_to_qualifier",
                "suggested_action": "convert_to_qualifier",
                "node_id": "age-12-plus",
                "label": "12 years and older",
                "entity_type": "Age",
                "qualifier_value": "12 years and older",
                "connected_edge_ids": ["edge-age"],
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
    )

    candidate = _single_action_candidate(build_proposal_task_packs(package))

    assert candidate["proposal_type"] == "value_node_to_qualifier"
    assert candidate["action_payload"]["qualifier_key"] == "age"
    assert candidate["action_payload"]["qualifier_value"] == "12 years and older"


def test_clinical_modeling_generator_repairs_symptom_to_disease_direction(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "clinical_manifestation_direction_mismatch",
                "edge_id": "edge-cough-flu",
                "source": "dry-cough",
                "target": "flu",
                "keywords": "clinical_manifestation",
                "source_type": "Symptom",
                "target_type": "Disease",
                "candidate_predicates": ["has_manifestation"],
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "suggested_action": "replace_relation",
            }
        ],
    )

    candidate = _single_action_candidate(build_proposal_task_packs(package))

    assert candidate["action_payload"] == {
        "action": "replace_relation",
        "edge_id": "edge-cough-flu",
        "expected_source": "dry-cough",
        "expected_target": "flu",
        "current_keywords": "clinical_manifestation",
        "new_source": "flu",
        "new_target": "dry-cough",
        "new_keywords": "has_manifestation",
        "qualifiers": {},
        "new_source_type": "Disease",
        "new_target_type": "Symptom",
    }


def test_diagnosis_generator_repairs_finding_to_diagnostic_criterion(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "diagnostic_criterion_direction_mismatch",
                "edge_id": "edge-tachypnea-severe-flu",
                "source": "tachypnea",
                "target": "severe-flu",
                "keywords": "diagnostic_basis",
                "source_type": "Symptom",
                "target_type": "Disease",
                "candidate_predicates": ["has_diagnostic_criterion"],
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "suggested_action": "replace_relation",
            }
        ],
    )

    candidate = _single_action_candidate(build_proposal_task_packs(package))

    assert candidate["action_payload"]["new_source"] == "severe-flu"
    assert candidate["action_payload"]["new_target"] == "tachypnea"
    assert candidate["action_payload"]["new_keywords"] == "has_diagnostic_criterion"
    assert candidate["action_payload"]["new_source_type"] == "Disease"
    assert candidate["action_payload"]["new_target_type"] == "Symptom"


def test_treatment_generator_repairs_drug_to_disease_indication(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "treatment_domain_range_mismatch",
                "edge_id": "edge-oseltamivir-flu",
                "source": "oseltamivir",
                "target": "flu",
                "keywords": "recommended_treatment",
                "source_type": "Drug",
                "target_type": "Disease",
                "candidate_predicates": ["has_indication", "recommends"],
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "suggested_action": "replace_relation",
            }
        ],
    )

    candidate = _single_action_candidate(build_proposal_task_packs(package))

    assert candidate["action_payload"]["new_source"] == "oseltamivir"
    assert candidate["action_payload"]["new_target"] == "flu"
    assert candidate["action_payload"]["new_keywords"] == "has_indication"


def test_legacy_schema_dispatches_to_treatment_generator_by_predicate(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "legacy_overloaded_relation",
                "issue_family": "legacy_schema",
                "edge_id": "edge-oseltamivir-flu",
                "source": "oseltamivir",
                "target": "flu",
                "keywords": "recommended_treatment",
                "source_type": "Drug",
                "target_type": "Disease",
                "candidate_predicates": ["has_indication"],
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "suggested_action": "replace_relation",
            }
        ],
    )

    candidate = _single_action_candidate(build_proposal_task_packs(package))

    assert candidate["action_payload"]["new_keywords"] == "has_indication"


def test_treatment_generator_rejects_ambiguous_safety_predicates(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "safety_scope_mismatch",
                "issue_family": "treatment",
                "edge_id": "edge-oseltamivir-pregnancy",
                "source": "oseltamivir",
                "target": "pregnancy",
                "keywords": "safety_scope",
                "source_type": "Drug",
                "target_type": "Population",
                "candidate_predicates": [
                    "contraindicated_for",
                    "precaution_for",
                ],
                "qualifiers": {"condition": "influenza"},
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "suggested_action": "replace_relation",
            }
        ],
    )

    packs = [pack.to_dict() for pack in build_proposal_task_packs(package)]

    pack = next(pack for pack in packs if pack["issue_family"] == "treatment")
    assert pack["action_candidates"] == []
    assert pack["rejected_action_candidates"][0]["error_code"] == (
        "AMBIGUOUS_SAFETY_PREDICATE"
    )


def test_risk_safety_generator_repairs_underlying_condition_risk_factor(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "risk_relation_mismatch",
                "edge_id": "edge-copd-flu",
                "source": "copd",
                "target": "flu",
                "keywords": "complication_risk",
                "source_type": "ClinicalCondition",
                "target_type": "Disease",
                "candidate_predicates": ["risk_factor_for"],
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "suggested_action": "replace_relation",
            }
        ],
    )

    candidate = _single_action_candidate(build_proposal_task_packs(package))

    assert candidate["action_payload"]["new_source"] == "copd"
    assert candidate["action_payload"]["new_target"] == "flu"
    assert candidate["action_payload"]["new_keywords"] == "risk_factor_for"


def test_risk_safety_generator_rejects_ambiguous_risk_predicates(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "risk_relation_mismatch",
                "edge_id": "edge-copd-flu",
                "source": "copd",
                "target": "flu",
                "keywords": "risk_relation",
                "source_type": "ClinicalCondition",
                "target_type": "Disease",
                "candidate_predicates": [
                    "risk_factor_for",
                    "increases_risk_of",
                ],
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "suggested_action": "replace_relation",
            }
        ],
    )

    packs = [pack.to_dict() for pack in build_proposal_task_packs(package)]

    pack = next(pack for pack in packs if pack["issue_family"] == "risk_safety")
    assert pack["action_candidates"] == []
    assert pack["rejected_action_candidates"][0]["error_code"] == (
        "AMBIGUOUS_RISK_PREDICATE"
    )


def test_conflict_review_issue_never_generates_mutation_candidate(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "conflicting_recommendation_safety_scope",
                "edge_id": "edge-conflict",
                "source": "oseltamivir",
                "target": "children",
                "keywords": "recommended_for",
                "source_type": "Drug",
                "target_type": "Population",
                "candidate_predicates": [
                    "recommended_for",
                    "contraindicated_for",
                ],
                "suggested_action": "review_conflict",
                "qualifiers": {"condition": "influenza", "purpose": "treatment"},
            }
        ],
    )

    packs = [pack.to_dict() for pack in build_proposal_task_packs(package)]

    assert [candidate for pack in packs for candidate in pack["action_candidates"]] == []


def test_prevention_generator_repairs_vaccine_to_target_disease(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "prevention_relation_mismatch",
                "edge_id": "edge-vaccine-flu",
                "source": "flu-vaccine",
                "target": "flu",
                "keywords": "prevention_measure",
                "source_type": "Vaccine",
                "target_type": "Disease",
                "candidate_predicates": ["targets_disease", "reduces_risk_of"],
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "suggested_action": "replace_relation",
            }
        ],
    )

    candidate = _single_action_candidate(build_proposal_task_packs(package))

    assert candidate["action_payload"]["new_source"] == "flu-vaccine"
    assert candidate["action_payload"]["new_target"] == "flu"
    assert candidate["action_payload"]["new_keywords"] == "targets_disease"


def test_build_task_packs_honors_issue_and_pack_limits(tmp_path: Path) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            _issue(
                f"edge-treatment-{index}",
                keywords="recommended_treatment",
                candidate_predicates=["has_indication", "recommends"],
            )
            for index in range(7)
        ],
    )

    packs = build_proposal_task_packs(
        package,
        max_issues_per_pack=2,
        max_packs=3,
    )

    assert len(packs) == 3
    assert all(len(pack.issues) <= 2 for pack in packs)


def test_build_task_packs_records_omitted_counts_for_repeated_family(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[
            _issue(
                f"edge-treatment-{index}",
                keywords="recommended_treatment",
                candidate_predicates=["has_indication", "recommends"],
            )
            for index in range(6)
        ],
    )

    packs = build_proposal_task_packs(
        package,
        max_issues_per_pack=1,
        max_packs=2,
    )

    assert [pack.role for pack in packs] == ["treatment", "treatment"]
    assert [pack.issue_family for pack in packs] == ["treatment", "treatment"]
    assert [pack.omitted_issue_count for pack in packs] == [5, 4]


def test_task_packs_stratify_issue_families_under_pack_limit(
    tmp_path: Path,
) -> None:
    issues = []
    for index in range(30):
        issues.append(
            _issue(
                f"edge-direction-{index}",
                issue_kind="reverse_clinical_manifestation",
                keywords="needs direction review",
                candidate_predicates=["has_manifestation"],
            )
        )
    for index in range(30):
        issues.append(
            _issue(
                f"edge-treatment-{index}",
                issue_kind="treatment_domain_range_mismatch",
                keywords="treatment domain review",
                candidate_predicates=["has_indication"],
            )
        )
    for index in range(30):
        issues.append(
            _issue(
                f"edge-diagnosis-{index}",
                issue_kind="diagnostic_evidence_direction_mismatch",
                keywords="diagnostic evidence review",
                candidate_predicates=["has_diagnostic_criterion"],
            )
        )
    for index in range(30):
        issues.append(
            _issue(
                f"edge-prevention-{index}",
                issue_kind="prevention_population_direction_mismatch",
                keywords="vaccine population review",
                candidate_predicates=["recommended_for"],
            )
        )
    for index in range(30):
        issues.append(
            _issue(
                f"edge-split-{index}",
                issue_kind="multi_predicate_edge_split_needed",
                keywords="split review",
                candidate_predicates=["has_indication", "recommended_for"],
            )
        )

    package = _make_package(tmp_path, issues=issues)

    packs = build_proposal_task_packs(
        package,
        max_issues_per_pack=4,
        max_packs=5,
    )

    represented = {
        issue["issue_kind"]
        for pack in packs
        for issue in pack.issues
    }
    assert represented == {
        "reverse_clinical_manifestation",
        "treatment_domain_range_mismatch",
        "diagnostic_evidence_direction_mismatch",
        "prevention_population_direction_mismatch",
        "multi_predicate_edge_split_needed",
    }
    assert sum(len(pack.issues) for pack in packs) <= 20


def test_build_task_packs_falls_back_to_quality_findings_without_schema_issues(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[],
        findings=[
            {
                "severity": "high",
                "category": "hierarchy_completeness",
                "message": "Hierarchy is missing a symptom branch.",
                "evidence": ["entity:entity fever", "source_id: chunk-1"],
                "suggested_fix_type": "add_hierarchy_branch",
                "requires_approval": True,
            }
        ],
    )

    packs = build_proposal_task_packs(package)

    assert len(packs) == 1
    pack = packs[0].to_dict()
    assert pack["role"] == "general"
    assert pack["action_candidates"] == []
    assert pack["issues"] == [
        {
            "issue_kind": "quality_finding",
            "severity": "high",
            "category": "hierarchy_completeness",
            "message": "Hierarchy is missing a symptom branch.",
            "evidence": ["entity:entity fever", "source_id: chunk-1"],
            "suggested_action": "add_hierarchy_branch",
            "requires_approval": True,
        }
    ]


def test_build_task_packs_clips_fallback_evidence_and_snapshot_context(
    tmp_path: Path,
) -> None:
    package = _make_package(
        tmp_path,
        issues=[],
        findings=[
            {
                "severity": "high",
                "category": "hierarchy_completeness",
                "message": "Hierarchy is missing a symptom branch. " + ("m" * 1000),
                "evidence": [
                    f"source_id: chunk-{index}; " + ("e" * 1000)
                    for index in range(30)
                ],
                "suggested_fix_type": "add_hierarchy_branch",
                "requires_approval": True,
            }
        ],
        source_files=[
            f"guide-{index}.md:" + ("s" * 1000) for index in range(60)
        ],
        metadata={
            f"metadata_key_{index}": "metadata-value-" + ("v" * 1000)
            for index in range(50)
        },
    )

    packs = build_proposal_task_packs(package)

    assert len(packs) == 1
    pack = packs[0].to_dict()
    issue = pack["issues"][0]
    assert len(issue["message"]) <= 240
    assert len(issue["evidence"]) == 10
    assert issue["evidence_omitted_count"] == 20
    assert all(len(item) <= 240 for item in issue["evidence"])

    snapshot_context = pack["snapshot_context"]
    assert len(snapshot_context["source_files"]) == 20
    assert snapshot_context["source_files_omitted_count"] == 40
    assert all(len(item) <= 240 for item in snapshot_context["source_files"])
    assert len(snapshot_context["metadata"]) == 20
    assert snapshot_context["metadata_omitted_key_count"] == 30
    assert all(
        not (isinstance(value, str) and len(value) > 240)
        for value in snapshot_context["metadata"].values()
    )


def test_merge_subagent_proposals_deduplicates_identical_action_payloads() -> None:
    payload = _replace_payload("edge-1")
    first = _proposal("proposal-a", target="edge:edge-1", action_payload=payload)
    duplicate = _proposal("proposal-b", target="edge:edge-1", action_payload=payload)

    merged = merge_subagent_proposals([[first], [duplicate]], max_proposals=10)

    assert merged.proposals == [first]
    assert merged.conflicts == []
    assert merged.dropped == [
        {
            "proposal_id": "proposal-b",
            "target": "edge:edge-1",
            "reason": "duplicate_action_payload",
        }
    ]


def test_merge_subagent_proposals_deduplicates_payloads_with_edge_target_prefix_mismatch() -> None:
    payload = _replace_payload("edge-1")
    first = _proposal("proposal-a", target="edge:edge-1", action_payload=payload)
    duplicate = _proposal("proposal-b", target="edge-1", action_payload=payload)

    merged = merge_subagent_proposals([[first], [duplicate]], max_proposals=10)

    assert merged.proposals == [first]
    assert merged.conflicts == []
    assert merged.dropped == [
        {
            "proposal_id": "proposal-b",
            "target": "edge-1",
            "reason": "duplicate_action_payload",
        }
    ]


def test_merge_subagent_proposals_drops_duplicate_ids_with_distinct_payloads() -> None:
    first = _proposal(
        "proposal-duplicate",
        target="edge:edge-1",
        action_payload=_replace_payload("edge-1", new_target="dry-cough"),
    )
    second = _proposal(
        "proposal-duplicate",
        target="edge:edge-2",
        action_payload=_replace_payload("edge-2", new_target="fatigue"),
    )

    merged = merge_subagent_proposals([[second], [first]], max_proposals=10)

    assert [proposal.id for proposal in merged.proposals] == ["proposal-duplicate"]
    assert len(merged.proposals) == 1
    assert merged.conflicts == []
    assert merged.dropped == [
        {
            "proposal_id": "proposal-duplicate",
            "target": "edge:edge-2",
            "reason": "duplicate_proposal_id_conflict",
        }
    ]


def test_merge_subagent_proposals_keeps_empty_payloads_for_different_semantics() -> None:
    first = _proposal(
        "proposal-a",
        target="edge:edge-1",
        proposal_type="medical_relation_schema_migration",
        proposed_change="Record the schema migration for edge 1.",
        action_payload={},
    )
    second = _proposal(
        "proposal-b",
        target="node:flu",
        proposal_type="entity_description_update",
        proposed_change="Clarify the flu entity description.",
        action_payload={},
    )

    merged = merge_subagent_proposals([[first], [second]], max_proposals=10)

    assert merged.proposals == [first, second]
    assert merged.conflicts == []
    assert merged.dropped == []


def test_merge_subagent_proposals_records_conflicts_and_excludes_them() -> None:
    first = _proposal(
        "proposal-a",
        target="edge:edge-1",
        action_payload=_replace_payload("edge-1", new_target="dry-cough"),
    )
    conflicting = _proposal(
        "proposal-b",
        target="edge:edge-1",
        action_payload=_replace_payload("edge-1", new_target="fatigue"),
    )

    merged = merge_subagent_proposals([[first], [conflicting]], max_proposals=10)

    assert merged.proposals == []
    assert merged.conflicts == [
        {
            "target": "edge-1",
            "proposal_ids": ["proposal-a", "proposal-b"],
            "reason": "different_action_payload_for_same_target",
        }
    ]
    assert merged.dropped == []


def test_merge_subagent_proposals_conflicts_edge_target_prefix_mismatch() -> None:
    first = _proposal(
        "proposal-a",
        target="edge:e1",
        action_payload=_replace_payload("e1", new_target="dry-cough"),
    )
    conflicting = _proposal(
        "proposal-b",
        target="e1",
        action_payload=_replace_payload("e1", new_target="fatigue"),
    )

    merged = merge_subagent_proposals([[first], [conflicting]], max_proposals=10)

    assert merged.proposals == []
    assert merged.conflicts == [
        {
            "target": "e1",
            "proposal_ids": ["proposal-a", "proposal-b"],
            "reason": "different_action_payload_for_same_target",
        }
    ]
    assert merged.dropped == []


def test_merge_subagent_proposals_prefers_action_candidate_over_llm_conflict() -> None:
    action_candidate = _proposal(
        "prop-action-candidate-cbc",
        target="edge:edge-cbc-flu",
        action_payload={
            "action": "replace_relation",
            "edge_id": "edge-cbc-flu",
            "expected_source": "cbc",
            "expected_target": "flu",
            "current_keywords": "诊断依据",
            "new_source": "flu",
            "new_target": "cbc",
            "new_keywords": "orders_test",
            "qualifiers": {},
        },
    )
    llm_retire = _proposal(
        "proposal-retire-cbc",
        target="edge:edge-cbc-flu",
        action_payload={
            "action": "retire_relation",
            "edge_id": "edge-cbc-flu",
            "expected_source": "cbc",
            "expected_target": "flu",
            "current_keywords": "诊断依据",
            "retirement_reason": "CBC is not a direct diagnostic criterion.",
        },
    )

    merged = merge_subagent_proposals(
        [[action_candidate], [llm_retire]], max_proposals=10
    )

    assert merged.proposals == [action_candidate]
    assert merged.conflicts == []
    assert merged.dropped == [
        {
            "proposal_id": "proposal-retire-cbc",
            "target": "edge:edge-cbc-flu",
            "reason": "conflicting_with_action_candidate",
        }
    ]


def test_merge_subagent_proposals_keeps_distinct_report_notes_for_same_target() -> None:
    first = _proposal(
        "proposal-note-a",
        target="quality_report.md",
        proposal_type="quality_report_note",
        proposed_change="Record the hierarchy finding in the quality report.",
        action_payload={},
    )
    second = _proposal(
        "proposal-note-b",
        target="quality_report.md",
        proposal_type="quality_report_note",
        proposed_change="Record the evidence coverage note in the quality report.",
        action_payload={},
    )

    merged = merge_subagent_proposals([[first], [second]], max_proposals=10)

    assert [proposal.id for proposal in merged.proposals] == [
        "proposal-note-a",
        "proposal-note-b",
    ]
    assert merged.conflicts == []
    assert merged.dropped == []


def test_merge_subagent_proposals_conflicts_empty_payload_same_target_changes() -> None:
    first = _proposal(
        "proposal-a",
        target="edge:edge-1",
        proposal_type="medical_relation_schema_migration",
        proposed_change="Replace the relation with canonical treatment schema.",
        action_payload={},
    )
    conflicting = _proposal(
        "proposal-b",
        target="edge:edge-1",
        proposal_type="entity_description_update",
        proposed_change="Rewrite the edge description without changing schema.",
        action_payload={},
    )

    merged = merge_subagent_proposals([[first], [conflicting]], max_proposals=10)

    assert merged.proposals == []
    assert merged.conflicts == [
        {
            "target": "edge-1",
            "proposal_ids": ["proposal-a", "proposal-b"],
            "reason": "different_semantic_action_for_same_target",
        }
    ]
    assert merged.dropped == []


def test_merge_subagent_proposals_respects_max_proposals_and_records_drops() -> None:
    proposals = [
        _proposal(
            f"proposal-{index}",
            target=f"edge:edge-{index}",
            action_payload=_replace_payload(
                f"edge-{index}",
                new_target=f"semantic-target-{index}",
            ),
        )
        for index in range(4)
    ]

    merged = merge_subagent_proposals([proposals], max_proposals=2)

    assert [proposal.id for proposal in merged.proposals] == [
        "proposal-0",
        "proposal-1",
    ]
    assert merged.dropped == [
        {
            "proposal_id": "proposal-2",
            "target": "edge:edge-2",
            "reason": "max_proposals",
        },
        {
            "proposal_id": "proposal-3",
            "target": "edge:edge-3",
            "reason": "max_proposals",
        },
    ]


def test_merge_subagent_proposals_applies_max_after_deterministic_sort() -> None:
    proposals = [
        _proposal(
            "proposal-low-high-confidence",
            target="edge:edge-1",
            action_payload=_replace_payload("edge-1", new_target="low-risk-target"),
            confidence=0.99,
            risk="low",
        ),
        _proposal(
            "proposal-high-lower-confidence",
            target="edge:edge-2",
            action_payload=_replace_payload("edge-2", new_target="high-lower-target"),
            confidence=0.7,
            risk="high",
        ),
        _proposal(
            "proposal-high-top-confidence",
            target="edge:edge-3",
            action_payload=_replace_payload("edge-3", new_target="high-top-target"),
            confidence=0.95,
            risk="high",
        ),
        _proposal(
            "proposal-medium-high-confidence",
            target="edge:edge-4",
            action_payload=_replace_payload("edge-4", new_target="medium-target"),
            confidence=0.98,
            risk="medium",
        ),
    ]
    ordered_batches = [[proposals[0], proposals[1]], [proposals[2], proposals[3]]]
    shuffled_batches = [[proposals[3], proposals[0]], [proposals[2], proposals[1]]]

    ordered = merge_subagent_proposals(ordered_batches, max_proposals=2)
    merged = merge_subagent_proposals(shuffled_batches, max_proposals=2)

    assert [proposal.id for proposal in ordered.proposals] == [
        proposal.id for proposal in merged.proposals
    ]
    assert [proposal.id for proposal in merged.proposals] == [
        "proposal-high-top-confidence",
        "proposal-high-lower-confidence",
    ]
    assert [drop["proposal_id"] for drop in merged.dropped] == [
        "proposal-medium-high-confidence",
        "proposal-low-high-confidence",
    ]


def _make_package(
    tmp_path: Path,
    *,
    issues: list[dict[str, object]],
    nodes: list[dict[str, object]] | None = None,
    edges: list[dict[str, object]] | None = None,
    entity_cleanup_issues: list[dict[str, object]] | None = None,
    details_overrides: dict[str, list[dict[str, object]]] | None = None,
    findings: list[dict[str, object]] | None = None,
    source_files: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    package = tmp_path / "kb-iteration"
    snapshot_edges = edges or [_edge(str(issue["edge_id"])) for issue in issues]
    _write_json(
        package / "snapshots" / "kg_snapshot.json",
        {
            "workspace": "medical-demo",
            "generated_at": "2026-06-21T00:00:00+08:00",
            "source_files": source_files or ["guide.md"],
            "nodes": nodes or [],
            "edges": snapshot_edges,
            "metadata": metadata or {"profile": "clinical_guideline_zh"},
        },
    )
    details: dict[str, object] = {
        "medical_schema_issues": issues,
        "entity_cleanup_issues": entity_cleanup_issues or [],
    }
    if details_overrides:
        details.update(details_overrides)

    _write_json(
        package / "snapshots" / "quality_score.json",
        {
            "overall": 70,
            "subscores": {},
            "metrics": {"medical_schema_issue_count": len(issues)},
            "details": details,
            "findings": findings or [],
            "critical_blockers": [],
        },
    )
    return package


def _issue(
    edge_id: str,
    *,
    issue_kind: str = "legacy_overloaded_relation",
    source: str = "flu",
    target: str = "oseltamivir",
    keywords: str = "recommended_treatment",
    candidate_predicates: list[str] | None = None,
    medical_subcase: str = "",
) -> dict[str, object]:
    issue = {
        "issue_kind": issue_kind,
        "edge_id": edge_id,
        "source": source,
        "target": target,
        "keywords": keywords,
        "source_id": "chunk-1",
        "file_path": "guide.md",
        "suggested_action": "replace_relation",
        "candidate_predicates": candidate_predicates or ["has_indication"],
        "new_source": "",
        "new_target": "",
        "guidance": "",
    }
    if medical_subcase:
        issue["medical_subcase"] = medical_subcase
    return issue


def _node(node_id: str, *, entity_type: str) -> dict[str, object]:
    return {
        "id": node_id,
        "label": node_id,
        "entity_type": entity_type,
        "description": "",
        "source_id": "chunk-1",
        "file_path": "guide.md",
    }


def _edge(
    edge_id: str,
    *,
    source: str = "flu",
    target: str = "oseltamivir",
    keywords: str = "recommended_treatment",
) -> dict[str, object]:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "keywords": keywords,
        "description": "",
        "source_id": "chunk-1",
        "file_path": "guide.md",
        "weight": None,
        "properties": {},
    }


def _proposal(
    proposal_id: str,
    *,
    target: str,
    action_payload: dict[str, object] | None = None,
    proposal_type: str = "medical_relation_schema_migration",
    proposed_change: str = "Replace the relation with the canonical schema action.",
    confidence: float = 0.9,
    risk: str = "medium",
) -> ImprovementProposal:
    edge_id = target.removeprefix("edge:")
    return ImprovementProposal(
        id=proposal_id,
        type=proposal_type,
        target=target,
        proposed_change=proposed_change,
        reason="The medical relation schema issue is deterministic.",
        evidence=[target],
        confidence=confidence,
        risk=risk,
        requires_approval=True,
        expected_metric_change={},
        action_payload=(
            _replace_payload(edge_id) if action_payload is None else action_payload
        ),
    )


def _replace_payload(edge_id: str, *, new_target: str = "dry-cough") -> dict[str, object]:
    return {
        "action": "replace_relation",
        "edge_id": edge_id,
        "expected_source": "dry-cough",
        "expected_target": "flu",
        "current_keywords": "clinical_manifestation",
        "new_source": "flu",
        "new_target": new_target,
        "new_keywords": "has_manifestation",
        "qualifiers": {},
    }


def _pack_edge_ids(pack: dict[str, object]) -> list[str]:
    return [str(issue["edge_id"]) for issue in pack["issues"]]  # type: ignore[index]


def _single_action_candidate(packs: list) -> dict[str, object]:
    action_candidates = [
        candidate for pack in packs for candidate in pack.action_candidates
    ]
    assert len(action_candidates) == 1
    return action_candidates[0]


def _single_route_for_edge(scan: object, edge_id: str) -> object:
    routes = _routes_for_edge(scan, edge_id)
    assert len(routes) == 1
    return routes[0]


def _routes_for_edge(scan: object, edge_id: str) -> list[object]:
    issues = getattr(scan, "issues")
    issue_refs = {
        str(issue["issue_ref"])
        for issue in issues
        if str(issue.get("edge_id", "")) == edge_id
    }
    return [
        route
        for route in getattr(scan, "issue_routes")
        if _route_issue_ref(route) in issue_refs
    ]


def _route_issue_ref(route: object) -> str:
    if isinstance(route, dict):
        return str(route.get("issue_ref", ""))
    return str(getattr(route, "issue_ref"))


def _route_state(route: object) -> str:
    if isinstance(route, dict):
        return str(route.get("route_state", ""))
    return str(getattr(route, "route_state"))


def _route_reason_code(route: object) -> str:
    if isinstance(route, dict):
        return str(route.get("reason_code", ""))
    return str(getattr(route, "reason_code"))


def _assert_candidate_rejection_contract(
    rejection: dict[str, object],
    *,
    issue_family: str | None = None,
) -> None:
    for field_name in (
        "candidate_id",
        "issue_ref",
        "issue_family",
        "stage",
        "error_code",
        "error",
    ):
        assert str(rejection.get(field_name) or "").strip(), field_name
    if issue_family is not None:
        assert rejection["issue_family"] == issue_family


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
