from __future__ import annotations

from lightrag.kb_iteration.medical_schema import (
    CANONICAL_MEDICAL_RELATION_IDS,
    LEGACY_MEDICAL_RELATION_MIGRATIONS,
    MEDICAL_RELATION_SCHEMA_VERSION,
    medical_type_allowed,
    migration_rule_for_legacy_keyword,
    normalize_medical_entity_type,
    relation_spec_by_id,
    validate_relation_instance,
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
        "risk_factor_for",
        "high_risk_for",
        "increases_risk_of",
        "acute_exacerbation_of",
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


def test_normalize_medical_entity_type_handles_runtime_snapshot_types() -> None:
    cases = {
        "disease": "Disease",
        "syndrome": "Syndrome",
        "clinicalcondition": "ClinicalCondition",
        "clinical_condition": "ClinicalCondition",
        "clinicalfinding": "ClinicalFinding",
        "clinical_finding": "ClinicalFinding",
        "clinicalpathway": "ClinicalPathway",
        "clinical_pathway": "ClinicalPathway",
        "symptom": "Symptom",
        "sign": "Symptom",
        "diagnostictest": "Test",
        "treatmentregimen": "DosingRegimen",
        "dosingregimen": "DosingRegimen",
        "treatment": "Treatment",
        "procedure": "Procedure",
        "publichealthmeasure": "PublicHealthMeasure",
        "clinicaldepartment": "ClinicalDepartment",
        "pathogen": "Pathogen",
        "drug": "Drug",
        "drugingredient": "Drug",
        "drugclass": "DrugClass",
        "vaccine": "Vaccine",
        "diagnosticcriterion": "DiagnosticCriterion",
        "testresult": "TestResult",
        "testresultpattern": "TestResultPattern",
        "method": "Method",
        "observation": "Observation",
        "diagnosis": "Diagnosis",
        "evidence": "Evidence",
        "specimen": "Specimen",
        "population": "Population",
        "riskfactor": "RiskFactor",
        "complication": "Complication",
        "adversereaction": "AdverseReaction",
        "guideline": "Guideline",
        "recommendation": "Recommendation",
        "medicalgroup": "MedicalGroup",
        "anatomy": "Anatomy",
        "": "Unknown",
        "UNKNOWN": "Unknown",
        None: "Unknown",
    }

    for raw_type, expected in cases.items():
        assert normalize_medical_entity_type(raw_type) == expected

    assert normalize_medical_entity_type(" DiagnosticTest ") == "Test"
    assert normalize_medical_entity_type(object()) == "Unknown"


def test_medical_type_allowed_understands_clinical_supertypes() -> None:
    assert medical_type_allowed("Drug", ("Intervention",))
    assert medical_type_allowed("Procedure", ("Treatment",))
    assert medical_type_allowed("Symptom", ("ClinicalFinding",))
    assert medical_type_allowed("TestResultPattern", ("Evidence",))
    assert medical_type_allowed("DrugClass", ("MedicalConcept",))
    assert not medical_type_allowed("Disease", ("Pathogen",))


def test_core_medical_relation_specs_are_strict_not_generic() -> None:
    expectations = {
        "causative_agent": (("Disease", "ClinicalCondition"), ("Pathogen",)),
        "orders_test": (
            ("Recommendation", "ClinicalPathway", "Disease", "ClinicalCondition"),
            ("Test",),
        ),
        "has_result": (("Test", "Observation"), ("TestResult",)),
        "contraindicated_for": (
            ("Drug", "Treatment", "Procedure", "Vaccine", "PublicHealthMeasure"),
            ("Population", "ClinicalCondition", "RiskFactor"),
        ),
        "precaution_for": (
            ("Drug", "Treatment", "Procedure", "Vaccine", "PublicHealthMeasure"),
            ("Population", "ClinicalCondition", "RiskFactor"),
        ),
        "interaction_with": (("Drug", "Treatment", "Vaccine"), ("Drug", "Treatment")),
        "evidenced_by": (("MedicalConcept",), ("Evidence", "Guideline")),
    }

    for relation_id, (domain_types, range_types) in expectations.items():
        spec = relation_spec_by_id(relation_id)

        assert spec.domain_types == domain_types
        assert spec.range_types == range_types
        assert spec.domain_types != ("MedicalConcept",) or relation_id == "evidenced_by"


def test_validate_relation_instance_uses_shared_domain_range_rules() -> None:
    assert validate_relation_instance(
        predicate="causative_agent",
        source_type="Disease",
        target_type="Pathogen",
    ) == []

    errors = validate_relation_instance(
        predicate="causative_agent",
        source_type="Disease",
        target_type="Symptom",
    )

    assert errors == [
        "target_type Symptom is outside causative_agent range Pathogen"
    ]


def test_validate_relation_instance_checks_required_qualifiers() -> None:
    errors = validate_relation_instance(
        predicate="supports_or_refutes",
        source_type="TestResultPattern",
        target_type="Disease",
        qualifiers={},
    )

    assert errors == ["supports_or_refutes requires qualifier polarity"]


def test_validate_relation_instance_checks_required_qualifier_groups() -> None:
    errors = validate_relation_instance(
        predicate="recommends",
        source_type="Guideline",
        target_type="Test",
        qualifiers={"population": "children"},
    )

    assert errors == [
        "recommends requires one qualifier from strength | evidence_level | version"
    ]

    assert (
        validate_relation_instance(
            predicate="recommends",
            source_type="Guideline",
            target_type="Test",
            qualifiers={"version": "2025"},
        )
        == []
    )


def test_validate_relation_instance_checks_recommendation_scope_qualifiers() -> None:
    errors = validate_relation_instance(
        predicate="recommended_for",
        source_type="Drug",
        target_type="Population",
        qualifiers={},
    )

    assert errors == [
        "recommended_for requires qualifier purpose",
        (
            "recommended_for requires one qualifier from condition | age | "
            "age_min | age_max | population | route | timing | time_window"
        ),
    ]

    assert (
        validate_relation_instance(
            predicate="recommended_for",
            source_type="Drug",
            target_type="Population",
            qualifiers={"purpose": "treatment", "age_min": 7, "route": "inhalation"},
        )
        == []
    )


def test_validate_relation_instance_checks_qualifier_enum_values() -> None:
    assert validate_relation_instance(
        predicate="supports_or_refutes",
        source_type="TestResultPattern",
        target_type="Disease",
        qualifiers={"polarity": "maybe"},
    ) == ["supports_or_refutes qualifier polarity must be one of refutes | supports"]

    assert validate_relation_instance(
        predicate="recommended_for",
        source_type="Drug",
        target_type="Population",
        qualifiers={"purpose": "diagnosis", "age_min": 7},
    ) == ["recommended_for qualifier purpose must be one of prevention | treatment"]


def test_risk_relations_model_chronic_disease_risk_without_complication_semantics() -> None:
    assert validate_relation_instance(
        predicate="high_risk_for",
        source_type="Population",
        target_type="Disease",
        qualifiers={"condition": "流行性感冒"},
    ) == []
    assert validate_relation_instance(
        predicate="increases_risk_of",
        source_type="ClinicalCondition",
        target_type="Outcome",
        qualifiers={"population": "COPD患者"},
    ) == []
    assert validate_relation_instance(
        predicate="acute_exacerbation_of",
        source_type="ClinicalCondition",
        target_type="Disease",
        qualifiers={"trigger": "流感病毒感染"},
    ) == []


def test_schema_registry_includes_temporarily_deferred_for() -> None:
    assert "temporarily_deferred_for" in CANONICAL_MEDICAL_RELATION_IDS

    spec = relation_spec_by_id("temporarily_deferred_for")
    assert spec.domain_types == (
        "Drug",
        "Treatment",
        "Procedure",
        "Vaccine",
        "PublicHealthMeasure",
    )
    assert spec.range_types == ("Population", "ClinicalCondition", "RiskFactor")

    assert validate_relation_instance(
        predicate="temporarily_deferred_for",
        source_type="Vaccine",
        target_type="ClinicalCondition",
        qualifiers={},
    ) == ["temporarily_deferred_for requires qualifier reason"]
