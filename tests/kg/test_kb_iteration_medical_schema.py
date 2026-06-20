from __future__ import annotations

from lightrag.kb_iteration.medical_schema import (
    CANONICAL_MEDICAL_RELATION_IDS,
    LEGACY_MEDICAL_RELATION_MIGRATIONS,
    MEDICAL_RELATION_SCHEMA_VERSION,
    migration_rule_for_legacy_keyword,
    relation_spec_by_id,
)


def test_medical_schema_registry_exposes_core_hospital_relations() -> None:
    assert MEDICAL_RELATION_SCHEMA_VERSION == "medical_relation_schema_v1"
    for relation_id in (
        "has_manifestation",
        "is_a",
        "has_indication",
        "recommends",
        "has_diagnostic_criterion",
        "performed_by_method",
        "supports_or_refutes",
        "has_dosing_regimen",
        "may_cause_adverse_reaction",
    ):
        assert relation_id in CANONICAL_MEDICAL_RELATION_IDS

    spec = relation_spec_by_id("has_manifestation")
    assert spec.id == "has_manifestation"
    assert "Disease" in spec.domain_types
    assert "Symptom" in spec.range_types
    assert "sign_or_symptom" in spec.range_types


def test_legacy_keyword_maps_to_safe_migration_options() -> None:
    treatment = migration_rule_for_legacy_keyword("推荐治疗")
    assert treatment is not None
    assert treatment.legacy_keywords
    assert set(treatment.canonical_options) >= {"has_indication", "recommends"}

    diagnostic = migration_rule_for_legacy_keyword("诊断依据")
    assert diagnostic is not None
    assert set(diagnostic.canonical_options) >= {
        "has_diagnostic_criterion",
        "criterion_requires",
        "supports_or_refutes",
    }


def test_legacy_migration_options_are_canonical_relation_ids() -> None:
    for rule in LEGACY_MEDICAL_RELATION_MIGRATIONS:
        for relation_id in rule.canonical_options:
            assert relation_id in CANONICAL_MEDICAL_RELATION_IDS


def test_schema_registry_covers_prompt_canonical_relation_families() -> None:
    prompt_relation_ids = (
        "is_a",
        "part_of",
        "names",
        "maps_to_code",
        "causative_agent",
        "has_manifestation",
        "has_complication",
        "has_risk_factor",
        "has_diagnostic_criterion",
        "criterion_requires",
        "differential_with",
        "has_severity_grade",
        "has_evidence",
        "ruled_out",
        "due_to",
        "orders_test",
        "has_result",
        "observes",
        "has_value",
        "has_interpretation",
        "uses_specimen",
        "performed_by_method",
        "supports_or_refutes",
        "has_active_ingredient",
        "belongs_to_drug_class",
        "has_indication",
        "recommends",
        "recommended_for",
        "not_recommended_for",
        "contraindicated_for",
        "precaution_for",
        "has_dosing_regimen",
        "may_cause_adverse_reaction",
        "interaction_with",
        "monitor_with",
        "targets_disease",
        "reduces_risk_of",
        "has_dose_schedule",
        "risk_group_for",
        "defined_by_characteristic",
        "evidenced_by",
        "asserted_by",
        "issued_by",
        "valid_during",
        "provided_by",
        "available_at",
    )

    assert set(prompt_relation_ids) <= CANONICAL_MEDICAL_RELATION_IDS
