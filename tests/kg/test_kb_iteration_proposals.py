from pathlib import Path

import pytest
import yaml

from lightrag.kb_iteration.models import ImprovementProposal
from lightrag.kb_iteration.patches import PatchCandidate, validate_patch_candidate
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
        type="add_hierarchy_branch",
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
    assert "add_hierarchy_branch" in text


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


def test_validate_proposal_allows_route_compatible_proposal_id():
    proposal = ImprovementProposal(
        id="Proposal_20260618.001-A",
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

    validate_proposal(proposal)


def test_validate_value_node_to_qualifier_requires_full_payload() -> None:
    proposal = _value_node_to_qualifier_proposal(
        {
            "value_node_id": "dose-75mg",
            "carrier_edge_source": "oseltamivir",
            "carrier_edge_target": "flu",
            "qualifier_key": "dose",
            "qualifier_value": "75 mg",
        }
    )

    with pytest.raises(ValueError, match="incident_edge_id"):
        validate_proposal(proposal)


def test_validate_value_node_to_qualifier_rejects_unsupported_carrier() -> None:
    proposal = _value_node_to_qualifier_proposal(
        {
            "value_node_id": "dose-75mg",
            "incident_edge_id": "edge-dose",
            "expected_incident_keywords": "has_value",
            "carrier_edge_id": "edge-class",
            "carrier_edge_source": "oseltamivir",
            "carrier_edge_target": "drug-class",
            "expected_carrier_keywords": "belongs_to_drug_class",
            "carrier_source_type": "Drug",
            "carrier_target_type": "DrugClass",
            "qualifier_key": "dose",
            "qualifier_value": "75 mg",
        }
    )

    with pytest.raises(ValueError, match="does not allow qualifier dose"):
        validate_proposal(proposal)


@pytest.mark.parametrize(
    "proposal_id",
    [
        "proposal 20260618",
        "proposal\n20260618",
        "proposal/20260618",
        r"proposal\20260618",
        "../proposal",
        "proposal:20260618",
    ],
)
def test_validate_proposal_rejects_unsafe_proposal_ids(proposal_id: str):
    proposal = ImprovementProposal(
        id=proposal_id,
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

    with pytest.raises(ValueError, match="proposal id"):
        validate_proposal(proposal)


def test_validate_patch_candidate_allows_route_compatible_proposal_id():
    validate_patch_candidate(
        PatchCandidate(
            proposal_id="Proposal_20260618.001-A",
            target_path="docs/review.md",
            diff_text="--- a/docs/review.md\n+++ b/docs/review.md\n",
        )
    )


def _value_node_to_qualifier_proposal(
    action_payload: dict[str, object],
) -> ImprovementProposal:
    return ImprovementProposal(
        id="proposal-value-node-qualifier",
        type="value_node_to_qualifier",
        target="node:dose-75mg",
        proposed_change="Move value node into carrier edge qualifiers.",
        reason="Value-like nodes should qualify clinical facts.",
        evidence=["node:dose-75mg"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload=action_payload,
    )


@pytest.mark.parametrize(
    "proposal_id",
    [
        "proposal 20260618",
        "proposal\n20260618",
        "proposal/20260618",
        r"proposal\20260618",
        "../proposal",
        "proposal:20260618",
    ],
)
def test_validate_patch_candidate_rejects_unsafe_proposal_ids(proposal_id: str):
    with pytest.raises(ValueError, match="proposal id"):
        validate_patch_candidate(
            PatchCandidate(
                proposal_id=proposal_id,
                target_path="docs/review.md",
                diff_text="--- a/docs/review.md\n+++ b/docs/review.md\n",
            )
        )


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
        "patch_candidate",
        "judge",
    }


def test_improvement_proposal_preserves_patch_candidate_positional_order():
    proposal = ImprovementProposal(
        "proposal-20260620-positional",
        "relation_keyword_mapping",
        "lightrag/medical_kg/ontology.py",
        "Map generic relation keywords to controlled relation labels.",
        "Generic relation labels reduce KG readability.",
        ["edge:e1"],
        0.82,
        "medium",
        True,
        {"relation_semantics": 8},
        "patch_candidates/proposal-20260620-positional.patch",
        {"decision": "needs_human"},
    )

    assert proposal.patch_candidate == (
        "patch_candidates/proposal-20260620-positional.patch"
    )
    assert proposal.judge == {"decision": "needs_human"}
    assert proposal.action_payload == {}


def test_improvement_proposal_to_dict_includes_non_empty_action_payload():
    proposal = ImprovementProposal(
        id="proposal-20260620-action-payload",
        type="medical_relation_schema_migration",
        target="edge:e1",
        proposed_change="Normalize relation.",
        reason="Relation direction is invalid.",
        evidence=["relation_id:e1"],
        confidence=0.8,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "e1",
            "expected_source": "dry-cough",
            "expected_target": "flu",
            "new_source": "flu",
            "new_target": "dry-cough",
            "new_keywords": "has_manifestation",
        },
    )

    assert proposal.to_dict()["action_payload"]["action"] == "replace_relation"


def test_validate_medical_relation_schema_migration_rejects_indication_to_pathogen():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-indication-target",
        type="medical_relation_schema_migration",
        target="edge:flu-virus->neuraminidase-inhibitor",
        proposed_change="Move treatment relation to has_indication.",
        reason="The target is a pathogen rather than a disease or clinical condition.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "flu-virus->neuraminidase-inhibitor",
            "expected_source": "流感病毒",
            "expected_target": "神经氨酸酶抑制剂",
            "current_keywords": "推荐治疗",
            "new_source": "神经氨酸酶抑制剂",
            "new_target": "流感病毒",
            "new_keywords": "has_indication",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="has_indication target"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_allows_indication_to_infection_condition():
    proposal = ImprovementProposal(
        id="proposal-20260622-good-indication-target",
        type="medical_relation_schema_migration",
        target="edge:flu-virus-infection->neuraminidase-inhibitor",
        proposed_change="Move treatment relation to has_indication.",
        reason="The target is a disease condition rather than a bare pathogen.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "flu-virus-infection->neuraminidase-inhibitor",
            "expected_source": "流感病毒感染",
            "expected_target": "神经氨酸酶抑制剂",
            "current_keywords": "推荐治疗",
            "new_source": "神经氨酸酶抑制剂",
            "new_target": "流感病毒感染",
            "new_keywords": "has_indication",
            "qualifiers": {},
        },
    )

    validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_parent_to_subtype_is_a():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-is-a-direction",
        type="medical_relation_schema_migration",
        target="edge:influenza-virus->influenza-b-virus",
        proposed_change="Normalize pathogen subtype hierarchy.",
        reason="The proposal points from a parent pathogen class to a subtype.",
        evidence=["source_id: chunk-1; file_path: guide.md; relation_id: e1"],
        confidence=0.9,
        risk="low",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "流感病毒->乙型流感病毒",
            "expected_source": "流感病毒",
            "expected_target": "乙型流感病毒",
            "current_keywords": "属于,病原分型",
            "new_source": "流感病毒",
            "new_target": "乙型流感病毒",
            "new_keywords": "is_a",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="is_a direction"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_allows_subtype_to_parent_is_a():
    proposal = ImprovementProposal(
        id="proposal-20260622-good-is-a-direction",
        type="medical_relation_schema_migration",
        target="edge:influenza-b-virus->influenza-virus",
        proposed_change="Normalize pathogen subtype hierarchy.",
        reason="The subtype points to the parent pathogen class.",
        evidence=["source_id: chunk-1; file_path: guide.md; relation_id: e1"],
        confidence=0.9,
        risk="low",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "流感病毒->乙型流感病毒",
            "expected_source": "流感病毒",
            "expected_target": "乙型流感病毒",
            "current_keywords": "属于,病原分型",
            "new_source": "乙型流感病毒",
            "new_target": "流感病毒",
            "new_keywords": "is_a",
            "qualifiers": {},
        },
    )

    validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_manifestation_category_target():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-manifestation-category",
        type="medical_relation_schema_migration",
        target="edge:flu-clinical-manifestations->flu",
        proposed_change="Reverse a clinical manifestation edge.",
        reason="Clinical manifestation headings are not patient-observable symptoms.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "流感临床表现->流行性感冒",
            "expected_source": "流感临床表现",
            "expected_target": "流行性感冒",
            "current_keywords": "临床表现",
            "new_source": "流行性感冒",
            "new_target": "流感临床表现",
            "new_keywords": "has_manifestation",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="has_manifestation target"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_bacterial_agent_for_viral_influenza():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-influenza-causative-agent",
        type="medical_relation_schema_migration",
        target="edge:pneumococcus->influenza-a",
        proposed_change="Reverse a causative agent edge.",
        reason="Bacterial secondary infection pathogens do not cause viral influenza.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "肺炎链球菌->甲型流感",
            "expected_source": "肺炎链球菌",
            "expected_target": "甲型流感",
            "current_keywords": "病原导致",
            "new_source": "甲型流感",
            "new_target": "肺炎链球菌",
            "new_keywords": "causative_agent",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="causative_agent"):
        validate_proposal(proposal)


@pytest.mark.parametrize(
    ("proposal_suffix", "disease", "generic_target"),
    [
        ("a", "甲型流感", "流感病毒"),
        ("b", "乙型流感", "流感病毒"),
    ],
)
def test_validate_medical_relation_schema_migration_rejects_generic_influenza_virus_for_typed_flu(
    proposal_suffix: str,
    disease: str,
    generic_target: str,
):
    proposal = ImprovementProposal(
        id=f"proposal-20260622-generic-virus-{proposal_suffix}",
        type="medical_relation_schema_migration",
        target=f"edge:flu-virus->{disease}",
        proposed_change="Reverse a pathogen subtype edge.",
        reason="Typed influenza diseases need typed influenza virus agents.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": f"流感病毒->{disease}",
            "expected_source": "流感病毒",
            "expected_target": disease,
            "current_keywords": "病原分型",
            "new_source": disease,
            "new_target": generic_target,
            "new_keywords": "causative_agent",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="typed influenza"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_disease_to_diagnostic_evidence():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-diagnostic-direction",
        type="medical_relation_schema_migration",
        target="edge:pathogen-test->flu",
        proposed_change="Normalize diagnostic evidence relation.",
        reason="Diagnostic evidence should support or refute a disease, not the reverse.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="high",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "病原学检查->流行性感冒",
            "expected_source": "病原学检查",
            "expected_target": "流行性感冒",
            "current_keywords": "诊断依据",
            "new_source": "流行性感冒",
            "new_target": "病原学检查",
            "new_keywords": "supports_or_refutes",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="supports_or_refutes"):
        validate_proposal(proposal)


@pytest.mark.parametrize(
    "diagnostic_target",
    ["血常规", "血生化", "动脉血气分析"],
)
def test_validate_medical_relation_schema_migration_rejects_disease_to_nonspecific_diagnostic_evidence(
    diagnostic_target: str,
):
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-nonspecific-supports-direction",
        type="medical_relation_schema_migration",
        target=f"edge:flu->{diagnostic_target}",
        proposed_change="Normalize diagnostic evidence relation.",
        reason="Disease should not point to nonspecific diagnostic evidence.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="high",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": f"{diagnostic_target}->流行性感冒",
            "expected_source": diagnostic_target,
            "expected_target": "流行性感冒",
            "current_keywords": "诊断依据",
            "new_source": "流行性感冒",
            "new_target": diagnostic_target,
            "new_keywords": "supports_or_refutes",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="supports_or_refutes"):
        validate_proposal(proposal)


@pytest.mark.parametrize(
    "diagnostic_target",
    ["血常规", "丙氨酸氨基转移酶", "天门冬氨酸氨基转移酶", "MRI"],
)
def test_validate_medical_relation_schema_migration_rejects_nonspecific_influenza_diagnostic_criterion(
    diagnostic_target: str,
):
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-nonspecific-diagnostic-criterion",
        type="medical_relation_schema_migration",
        target=f"edge:{diagnostic_target}->flu",
        proposed_change="Normalize diagnostic relation.",
        reason="Nonspecific tests should not become influenza diagnostic criteria.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": f"{diagnostic_target}->流行性感冒",
            "expected_source": diagnostic_target,
            "expected_target": "流行性感冒",
            "current_keywords": "诊断依据",
            "new_source": "流行性感冒",
            "new_target": diagnostic_target,
            "new_keywords": "has_diagnostic_criterion",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="has_diagnostic_criterion"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_allows_pathogen_test_diagnostic_criterion():
    proposal = ImprovementProposal(
        id="proposal-20260622-good-pathogen-test-diagnostic-criterion",
        type="medical_relation_schema_migration",
        target="edge:pathogen-test->flu",
        proposed_change="Normalize diagnostic relation.",
        reason="Pathogen testing is a disease diagnostic criterion.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "病原学检查->流行性感冒",
            "expected_source": "病原学检查",
            "expected_target": "流行性感冒",
            "current_keywords": "诊断依据",
            "new_source": "流行性感冒",
            "new_target": "病原学检查",
            "new_keywords": "has_diagnostic_criterion",
            "qualifiers": {},
        },
    )

    validate_proposal(proposal)


@pytest.mark.parametrize(
    "supporting_source",
    ["丙氨酸氨基转移酶", "天门冬氨酸氨基转移酶", "MRI"],
)
def test_validate_medical_relation_schema_migration_rejects_nonspecific_evidence_supporting_influenza(
    supporting_source: str,
):
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-nonspecific-supports-or-refutes",
        type="medical_relation_schema_migration",
        target=f"edge:{supporting_source}->flu",
        proposed_change="Normalize diagnostic support relation.",
        reason="Nonspecific findings should not directly support or refute influenza.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": f"{supporting_source}->流行性感冒",
            "expected_source": supporting_source,
            "expected_target": "流行性感冒",
            "current_keywords": "诊断依据",
            "new_source": supporting_source,
            "new_target": "流行性感冒",
            "new_keywords": "supports_or_refutes",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="supports_or_refutes"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_chronic_disease_as_influenza_complication():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-copd-complication",
        type="medical_relation_schema_migration",
        target="edge:influenza-virus->copd",
        proposed_change="Normalize complication relation.",
        reason="COPD is a chronic underlying condition rather than a flu complication.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "流感病毒->慢性阻塞性肺疾病(COPD)",
            "expected_source": "流感病毒",
            "expected_target": "慢性阻塞性肺疾病(COPD)",
            "current_keywords": "并发风险",
            "new_source": "流行性感冒",
            "new_target": "慢性阻塞性肺疾病(COPD)",
            "new_keywords": "has_complication",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="has_complication"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_outcome_as_influenza_complication():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-death-complication",
        type="medical_relation_schema_migration",
        target="edge:influenza->death",
        proposed_change="Normalize complication relation.",
        reason="Death is an outcome rather than a direct flu complication.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "流行性感冒->死亡",
            "expected_source": "流行性感冒",
            "expected_target": "死亡",
            "current_keywords": "并发风险",
            "new_source": "流行性感冒",
            "new_target": "死亡",
            "new_keywords": "has_complication",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="has_complication"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_pathogen_ordering_test():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-pathogen-orders-test",
        type="medical_relation_schema_migration",
        target="edge:virus->antigen-test",
        proposed_change="Normalize test relation.",
        reason="A pathogen entity cannot order a diagnostic test.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "流感病毒->抗原检测",
            "expected_source": "流感病毒",
            "expected_target": "抗原检测",
            "current_keywords": "检测方法",
            "new_source": "流感病毒",
            "new_target": "抗原检测",
            "new_keywords": "orders_test",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="orders_test"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_flu_ordering_bare_lab_marker():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-flu-orders-creatinine",
        type="medical_relation_schema_migration",
        target="edge:creatinine->flu",
        proposed_change="Normalize creatinine diagnostic-basis edge.",
        reason=(
            "Creatinine is a blood chemistry analyte; influenza should order the "
            "blood chemistry panel or model creatinine as an observed lab finding."
        ),
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "肌酐->流行性感冒",
            "expected_source": "肌酐",
            "expected_target": "流行性感冒",
            "current_keywords": "诊断依据",
            "new_source": "流行性感冒",
            "new_target": "肌酐",
            "new_keywords": "orders_test",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="orders_test.*lab marker"):
        validate_proposal(proposal)


@pytest.mark.parametrize("imaging_test", ["CT", "MRI", "胸部影像学检查"])
def test_validate_medical_relation_schema_migration_rejects_generic_flu_ordering_complication_imaging(
    imaging_test: str,
):
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-flu-orders-complication-imaging",
        type="medical_relation_schema_migration",
        target=f"edge:{imaging_test}->flu",
        proposed_change="Normalize complication imaging diagnostic-basis edge.",
        reason=(
            "Complication imaging should attach to influenza pneumonia, acute "
            "necrotizing encephalopathy, or a severity/complication endpoint."
        ),
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": f"{imaging_test}->流行性感冒",
            "expected_source": imaging_test,
            "expected_target": "流行性感冒",
            "current_keywords": "诊断依据",
            "new_source": "流行性感冒",
            "new_target": imaging_test,
            "new_keywords": "orders_test",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="orders_test.*complication"):
        validate_proposal(proposal)


@pytest.mark.parametrize(
    ("source", "target"),
    [("流感肺炎", "CT"), ("急性坏死性脑病", "MRI")],
)
def test_validate_medical_relation_schema_migration_allows_complication_specific_imaging_orders(
    source: str,
    target: str,
):
    proposal = ImprovementProposal(
        id="proposal-20260622-good-complication-imaging-order",
        type="medical_relation_schema_migration",
        target=f"edge:{source}->{target}",
        proposed_change="Attach imaging to a specific complication endpoint.",
        reason="The imaging test is scoped to a concrete influenza complication.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": f"{target}->{source}",
            "expected_source": target,
            "expected_target": source,
            "current_keywords": "诊断依据",
            "new_source": source,
            "new_target": target,
            "new_keywords": "orders_test",
            "qualifiers": {},
        },
    )

    validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_vaccine_targeting_pathogen_as_disease():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-vaccine-targets-pathogen",
        type="medical_relation_schema_migration",
        target="edge:vaccine->virus",
        proposed_change="Normalize prevention relation.",
        reason="targets_disease should target a disease, not a bare pathogen.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "流感病毒->流感疫苗",
            "expected_source": "流感病毒",
            "expected_target": "流感疫苗",
            "current_keywords": "预防措施",
            "new_source": "流感疫苗",
            "new_target": "流感病毒",
            "new_keywords": "targets_disease",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="targets_disease"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_population_to_outcome_reduction_without_qualifier():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-vaccine-population-reduction",
        type="medical_relation_schema_migration",
        target="edge:vaccine->heart-failure-patients",
        proposed_change="Rewrite a population edge as a risk reduction edge.",
        reason=(
            "The source evidence is about acute heart failure patients, not "
            "preventing acute heart failure itself."
        ),
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "flu-vaccine->acute-heart-failure-patients",
            "expected_source": "flu vaccine",
            "expected_target": "acute heart failure patients",
            "current_keywords": "prevention",
            "new_source": "flu vaccine",
            "new_target": "acute heart failure",
            "new_keywords": "reduces_risk_of",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="reduces_risk_of.*population"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_allows_population_scoped_outcome_reduction():
    proposal = ImprovementProposal(
        id="proposal-20260622-good-vaccine-population-reduction",
        type="medical_relation_schema_migration",
        target="edge:vaccine->heart-failure-readmission",
        proposed_change="Model a population-scoped risk reduction outcome.",
        reason="The source evidence is scoped to acute heart failure patients.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "flu-vaccine->acute-heart-failure-patients",
            "expected_source": "flu vaccine",
            "expected_target": "acute heart failure patients",
            "current_keywords": "prevention",
            "new_source": "flu vaccine",
            "new_target": "death or readmission",
            "new_keywords": "reduces_risk_of",
            "qualifiers": {"population": "acute heart failure patients"},
        },
    )

    validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_noop_replacement():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-noop-replacement",
        type="medical_relation_schema_migration",
        target="edge:steroid->adem",
        proposed_change="Normalize treatment relation.",
        reason="No semantic migration occurs.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "糖皮质激素->急性播散性脑脊髓炎",
            "expected_source": "糖皮质激素",
            "expected_target": "急性播散性脑脊髓炎",
            "current_keywords": "has_indication",
            "new_source": "糖皮质激素",
            "new_target": "急性播散性脑脊髓炎",
            "new_keywords": "has_indication",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="no-op"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_allows_retire_relation():
    proposal = ImprovementProposal(
        id="prop-retire-manifestation-category-edge",
        type="medical_relation_schema_migration",
        target="edge:流感临床表现->流行性感冒",
        proposed_change="Retire invalid category-to-disease manifestation edge.",
        reason="A section/category node should not be kept as a clinical fact edge.",
        evidence=["source_id: chunk-1; relation_id: 流感临床表现->流行性感冒"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "retire_relation",
            "edge_id": "流感临床表现->流行性感冒",
            "expected_source": "流感临床表现",
            "expected_target": "流行性感冒",
            "current_keywords": "临床表现",
            "retirement_reason": "category node is not an atomic manifestation",
        },
    )

    validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_population_recommendation_without_qualifier():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-broad-population-recommendation",
        type="medical_relation_schema_migration",
        target="edge:zanamivir->children",
        proposed_change="Normalize broad population recommendation.",
        reason="Broad pediatric drug recommendations need age or condition qualifiers.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "扎那米韦->儿童",
            "expected_source": "扎那米韦",
            "expected_target": "儿童",
            "current_keywords": "剂量用法,适用于",
            "new_source": "扎那米韦",
            "new_target": "儿童",
            "new_keywords": "recommended_for",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="recommended_for"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_allows_population_recommendation_with_condition_qualifier():
    proposal = ImprovementProposal(
        id="proposal-20260622-good-population-recommendation",
        type="medical_relation_schema_migration",
        target="edge:oseltamivir->children",
        proposed_change="Normalize pediatric recommendation.",
        reason="The broad population target is constrained by condition.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "儿童->奥司他韦",
            "expected_source": "儿童",
            "expected_target": "奥司他韦",
            "current_keywords": "推荐治疗,适用于",
            "new_source": "奥司他韦",
            "new_target": "儿童",
            "new_keywords": "recommended_for",
            "qualifiers": {"condition": "流行性感冒", "purpose": "treatment"},
        },
    )

    validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_electrolyte_disorder_as_manifestation():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-electrolyte-manifestation",
        type="medical_relation_schema_migration",
        target="edge:hypokalemia->flu",
        proposed_change="Reverse manifestation edge.",
        reason="Electrolyte abnormalities need complication or finding semantics.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "低钾血症->流行性感冒",
            "expected_source": "低钾血症",
            "expected_target": "流行性感冒",
            "current_keywords": "临床表现,并发风险",
            "new_source": "流行性感冒",
            "new_target": "低钾血症",
            "new_keywords": "has_manifestation",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="has_manifestation target"):
        validate_proposal(proposal)


def test_validate_medical_relation_schema_migration_rejects_typed_domain_range_mismatch():
    proposal = ImprovementProposal(
        id="proposal-20260622-bad-typed-causative-agent",
        type="medical_relation_schema_migration",
        target="edge:flu->cough",
        proposed_change="Normalize a cause relation.",
        reason="The proposed target is a symptom, not a pathogen.",
        evidence=["source_id: chunk-1"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "flu->cough",
            "expected_source": "flu",
            "expected_target": "cough",
            "current_keywords": "causes",
            "new_source": "flu",
            "new_target": "cough",
            "new_keywords": "causative_agent",
            "new_source_type": "Disease",
            "new_target_type": "Symptom",
            "qualifiers": {},
        },
    )

    with pytest.raises(ValueError, match="causative_agent.*range"):
        validate_proposal(proposal)


def _candidate_kg_expansion_payload() -> dict[str, object]:
    return {
        "candidate_nodes": [
            {
                "id": "symptom:fever",
                "label": "fever",
                "entity_type": "symptom",
            }
        ],
        "candidate_edges": [
            {
                "source": "disease:flu",
                "source_type": "Disease",
                "target": "symptom:fever",
                "target_type": "Symptom",
                "keywords": "has_manifestation",
                "qualifiers": {},
            }
        ],
        "source_id": "chunk-001",
        "file_path": "inputs/flu.txt",
        "evidence_quote": "The patient presented with fever.",
        "why_not_existing": "No existing fever symptom node is attached to flu.",
    }


def _candidate_kg_expansion_proposal(**overrides: object) -> ImprovementProposal:
    payload = {
        "id": "proposal-20260621-candidate-kg-expansion",
        "type": "candidate_kg_expansion",
        "target": "workspace/candidate-kg",
        "proposed_change": "Record candidate KG nodes and edges for human review.",
        "reason": "The evidence suggests a missing KG expansion candidate.",
        "evidence": ["chunk-001"],
        "confidence": 0.72,
        "risk": "medium",
        "requires_approval": True,
        "expected_metric_change": {},
        "action_payload": _candidate_kg_expansion_payload(),
    }
    payload.update(overrides)
    return ImprovementProposal(**payload)


def test_candidate_kg_expansion_passes_with_required_candidate_payload():
    validate_proposal(_candidate_kg_expansion_proposal())


@pytest.mark.parametrize(
    ("field_present", "evidence_quote"),
    [
        (False, None),
        (True, ""),
        (True, "   "),
    ],
)
def test_candidate_kg_expansion_rejects_missing_or_blank_evidence_quote(
    field_present: bool, evidence_quote: str | None
):
    action_payload = _candidate_kg_expansion_payload()
    if field_present:
        action_payload["evidence_quote"] = evidence_quote
    else:
        action_payload.pop("evidence_quote")

    proposal = _candidate_kg_expansion_proposal(action_payload=action_payload)

    with pytest.raises(
        ValueError,
        match=r"candidate_kg_expansion.*evidence_quote",
    ):
        validate_proposal(proposal)


def test_candidate_kg_expansion_rejects_without_approval():
    proposal = _candidate_kg_expansion_proposal(requires_approval=False)

    with pytest.raises(
        ValueError,
        match=r"^proposal type candidate_kg_expansion requires approval$",
    ):
        validate_proposal(proposal)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("candidate_nodes", {"id": "symptom:fever"}),
        ("candidate_edges", {"source": "disease:flu", "target": "symptom:fever"}),
        ("candidate_nodes", "symptom:fever"),
        ("candidate_edges", None),
    ],
)
def test_candidate_kg_expansion_rejects_non_list_candidates(
    field_name: str, value: object
):
    action_payload = _candidate_kg_expansion_payload()
    action_payload[field_name] = value
    proposal = _candidate_kg_expansion_proposal(action_payload=action_payload)

    with pytest.raises(ValueError, match=field_name):
        validate_proposal(proposal)


def test_candidate_kg_expansion_rejects_noncanonical_candidate_edge_predicate():
    action_payload = _candidate_kg_expansion_payload()
    candidate_edges = action_payload["candidate_edges"]
    assert isinstance(candidate_edges, list)
    candidate_edges[0]["keywords"] = "clinical_manifestation"
    proposal = _candidate_kg_expansion_proposal(action_payload=action_payload)

    with pytest.raises(ValueError, match="CANDIDATE_EDGE_NON_CANONICAL_PREDICATE"):
        validate_proposal(proposal)


def test_candidate_kg_expansion_requires_candidate_edge_endpoint_types():
    action_payload = _candidate_kg_expansion_payload()
    candidate_edges = action_payload["candidate_edges"]
    assert isinstance(candidate_edges, list)
    candidate_edges[0].pop("source_type")
    candidate_edges[0].pop("target_type")
    proposal = _candidate_kg_expansion_proposal(action_payload=action_payload)

    with pytest.raises(ValueError, match="CANDIDATE_EDGE_TYPES_REQUIRED"):
        validate_proposal(proposal)


def test_candidate_kg_expansion_rejects_candidate_edge_schema_violation():
    action_payload = _candidate_kg_expansion_payload()
    candidate_edges = action_payload["candidate_edges"]
    assert isinstance(candidate_edges, list)
    candidate_edges[0]["keywords"] = "causative_agent"
    proposal = _candidate_kg_expansion_proposal(action_payload=action_payload)

    with pytest.raises(ValueError, match="CANDIDATE_EDGE_SCHEMA_VIOLATION"):
        validate_proposal(proposal)


def test_medical_fact_role_split_rejects_draft_split_payload():
    proposal = ImprovementProposal(
        id="proposal-20260622-draft-split",
        type="medical_fact_role_split",
        target="edge:drug->child",
        proposed_change="Draft a split of an overloaded drug-child relation.",
        reason="The edge contains multiple medical roles but no executable split.",
        evidence=["relation_id: drug->child"],
        confidence=0.8,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "draft_split_relation",
            "edge_id": "drug->child",
            "current_keywords": "recommended treatment,dosing regimen",
            "candidate_predicates": ["recommended_for", "has_dosing_regimen"],
        },
    )

    with pytest.raises(ValueError, match="medical_fact_role_split"):
        validate_proposal(proposal)


def test_medical_fact_role_split_accepts_executable_split_payload():
    proposal = ImprovementProposal(
        id="proposal-20260622-executable-split",
        type="medical_fact_role_split",
        target="edge:drug->mixed-target",
        proposed_change="Split an overloaded drug relation into indication and population edges.",
        reason="The source edge mixes disease indication and patient population semantics.",
        evidence=["relation_id: drug->mixed-target; source_id: chunk-1"],
        confidence=0.84,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "split_relation",
            "edge_id": "drug->mixed-target",
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
        },
    )

    validate_proposal(proposal)


def test_medical_fact_role_split_rejects_noncanonical_split_predicate():
    proposal = ImprovementProposal(
        id="proposal-20260622-invalid-split-predicate",
        type="medical_fact_role_split",
        target="edge:drug->mixed-target",
        proposed_change="Split an overloaded relation.",
        reason="The edge mixes roles.",
        evidence=["relation_id: drug->mixed-target; source_id: chunk-1"],
        confidence=0.84,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "split_relation",
            "edge_id": "drug->mixed-target",
            "expected_source": "zanamivir",
            "expected_target": "mixed-target",
            "current_keywords": "recommended_treatment,applies_to",
            "retire_original": True,
            "new_edges": [
                {
                    "source": "zanamivir",
                    "target": "influenza",
                    "predicate": "applies_to",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="canonical relation id"):
        validate_proposal(proposal)


def test_medical_fact_role_split_requires_split_edge_endpoint_types():
    proposal = ImprovementProposal(
        id="proposal-20260622-split-missing-types",
        type="medical_fact_role_split",
        target="edge:drug->mixed-target",
        proposed_change="Split an overloaded relation.",
        reason="The edge mixes roles.",
        evidence=["relation_id: drug->mixed-target; source_id: chunk-1"],
        confidence=0.84,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "split_relation",
            "edge_id": "drug->mixed-target",
            "expected_source": "zanamivir",
            "expected_target": "mixed-target",
            "current_keywords": "recommended_treatment,applies_to",
            "retire_original": True,
            "new_edges": [
                {
                    "source": "zanamivir",
                    "target": "influenza",
                    "predicate": "has_indication",
                    "qualifiers": {"purpose": "treatment"},
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="SPLIT_EDGE_TYPES_REQUIRED"):
        validate_proposal(proposal)


def test_improvement_proposal_renders_patch_candidate_and_judge(tmp_path: Path):
    proposal = ImprovementProposal(
        id="proposal-20260618-001",
        type="add_hierarchy_branch",
        target="lightrag/medical_kg/ontology.py",
        proposed_change="Map generic relation keywords to controlled relation labels.",
        reason="Generic relation labels reduce KG readability.",
        evidence=["edge:e1"],
        confidence=0.82,
        risk="medium",
        requires_approval=True,
        expected_metric_change={"relation_semantics": 8},
        patch_candidate="patch_candidates/proposal-20260618-001.patch",
        judge={
            "decision": "needs_human",
            "reason": "Rule change requires maintainer review.",
        },
    )

    text = write_approval_queue([proposal], tmp_path).read_text(encoding="utf-8")
    payload = _load_yaml_body(text)
    rendered = payload["proposals"][0]

    assert rendered["type"] == "add_hierarchy_branch"
    assert rendered["patch_candidate"] == "patch_candidates/proposal-20260618-001.patch"
    assert rendered["judge"]["decision"] == "needs_human"


def test_validate_proposal_rejects_invalid_patch_candidate_type():
    proposal = ImprovementProposal(
        id="proposal-20260618-invalid-patch",
        type="relation_keyword_mapping",
        target="lightrag/medical_kg/ontology.py",
        proposed_change="Map generic relation keywords to controlled relation labels.",
        reason="Generic relation labels reduce KG readability.",
        evidence=["edge:e1"],
        confidence=0.82,
        risk="medium",
        requires_approval=True,
        expected_metric_change={"relation_semantics": 8},
        patch_candidate={"path": "patch_candidates/proposal.patch"},
    )

    with pytest.raises(ValueError, match="patch_candidate"):
        validate_proposal(proposal)


def test_validate_proposal_rejects_invalid_judge_type():
    proposal = ImprovementProposal(
        id="proposal-20260618-invalid-judge",
        type="llm_judge_rejection",
        target="review-context",
        proposed_change="Record judge rejection details.",
        reason="The judge payload is LLM-originated.",
        evidence=["review-context:round-001"],
        confidence=0.7,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        judge="needs_human",
    )

    with pytest.raises(ValueError, match="judge"):
        validate_proposal(proposal)


@pytest.mark.parametrize(
    "action_payload",
    [
        {},
        {"action": "replace_relation"},
        {
            "action": "replace_relation",
            "edge_id": "e1",
            "expected_source": "dry-cough",
            "expected_target": "flu",
            "new_source": "flu",
            "new_target": "dry-cough",
            "new_keywords": "",
        },
        {
            "action": "merge_relation",
            "edge_id": "e1",
            "expected_source": "dry-cough",
            "expected_target": "flu",
            "new_source": "flu",
            "new_target": "dry-cough",
            "new_keywords": "has_manifestation",
        },
        {
            "action": "replace_relation",
            "edge_id": "e1",
            "expected_source": "dry-cough",
            "expected_target": "flu",
            "new_source": "flu",
            "new_target": "dry-cough",
            "new_keywords": "has_manifestation",
        },
    ],
)
def test_medical_relation_schema_migration_rejects_malformed_action_payload(
    action_payload: dict,
):
    proposal = ImprovementProposal(
        id="proposal-20260620-malformed-action",
        type="medical_relation_schema_migration",
        target="edge:e1",
        proposed_change="Normalize relation.",
        reason="Relation direction is invalid.",
        evidence=["relation_id:e1"],
        confidence=0.8,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload=action_payload,
    )

    with pytest.raises(ValueError, match="action_payload"):
        validate_proposal(proposal)


def test_medical_relation_schema_migration_rejects_non_canonical_new_keywords():
    proposal = ImprovementProposal(
        id="proposal-20260620-noncanonical-keyword",
        type="medical_relation_schema_migration",
        target="edge:e1",
        proposed_change="Normalize relation.",
        reason="Relation direction is invalid.",
        evidence=["relation_id:e1"],
        confidence=0.8,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload={
            "action": "replace_relation",
            "edge_id": "e1",
            "expected_source": "dry-cough",
            "expected_target": "flu",
            "current_keywords": "临床表现",
            "new_source": "flu",
            "new_target": "dry-cough",
            "new_keywords": "not_a_canonical_relation",
        },
    )

    with pytest.raises(ValueError, match="new_keywords|canonical relation"):
        validate_proposal(proposal)


def test_proposal_rendering_omits_empty_patch_candidate_and_judge(tmp_path: Path):
    proposal = ImprovementProposal(
        id="proposal-20260618-empty-extensions",
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

    text = write_improvement_backlog([proposal], tmp_path).read_text(encoding="utf-8")
    payload = _load_yaml_body(text)
    rendered = payload["proposals"][0]

    assert "patch_candidate" not in rendered
    assert "judge" not in rendered


@pytest.mark.parametrize(
    "proposal_type",
    [
        "source_evidence_repair",
        "synonym_merge_rule",
        "relation_keyword_mapping",
        "review_context_request",
        "llm_judge_rejection",
    ],
)
def test_new_llm_review_proposal_types_require_approval_by_default(
    proposal_type: str,
):
    proposal = ImprovementProposal(
        id=f"proposal-20260618-{proposal_type}",
        type=proposal_type,
        target="review-target",
        proposed_change="Record or prepare a review action.",
        reason="LLM review generated a structured proposal.",
        evidence=["review-context:round-001"],
        confidence=0.7,
        risk="medium",
        requires_approval=False,
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="requires approval"):
        validate_proposal(proposal)


@pytest.mark.parametrize(
    "target",
    ["edge:扎那米韦->哮喘", "edge:扎那米韦->儿童"],
)
def test_review_context_request_rejects_grounded_zanamivir_split_shortcut(
    target: str,
):
    proposal = ImprovementProposal(
        id="proposal-20260622-zanamivir-review-only",
        type="review_context_request",
        target=target,
        proposed_change="Request exact source text before splitting zanamivir relation.",
        reason=(
            "The edge may conflate dose, age group, and respiratory disease "
            "precaution semantics."
        ),
        evidence=[
            "source_id: doc-b29c711f27db9ad51c2851d9db562957-chunk-003; "
            "file_path: 流行性感冒诊疗方案（2025年版）.pdf"
        ],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="review_context_request.*扎那米韦"):
        validate_proposal(proposal)


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
        {"overall": float("nan")},
        {"overall": float("inf")},
        {"overall": float("-inf")},
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
        type="add_hierarchy_branch",
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
        type="add_hierarchy_branch",
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
