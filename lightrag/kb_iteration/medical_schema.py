from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


MEDICAL_RELATION_SCHEMA_VERSION = "medical_relation_schema_v1"
_MEDICAL_PROFILE_HINTS = frozenset(
    {
        "clinical_guideline_zh",
        "medical_kg",
        "medical",
        "hospital",
        "clinical",
        "influenza",
    }
)
_NORMALIZED_MEDICAL_ENTITY_TYPES = {
    "disease": "Disease",
    "syndrome": "Syndrome",
    "clinicalcondition": "ClinicalCondition",
    "clinicalfinding": "ClinicalFinding",
    "clinicalpathway": "ClinicalPathway",
    "symptom": "Symptom",
    "sign": "Symptom",
    "clinicalmanifestation": "Symptom",
    "signorsymptom": "Symptom",
    "sign_or_symptom": "Symptom",
    "diagnostictest": "Test",
    "diagnosticmethod": "Test",
    "test": "Test",
    "treatmentregimen": "DosingRegimen",
    "dosingregimen": "DosingRegimen",
    "doseregimen": "DosingRegimen",
    "treatment": "Treatment",
    "procedure": "Procedure",
    "intervention": "Intervention",
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
    "labresult": "TestResult",
    "labresultpattern": "TestResultPattern",
    "method": "Method",
    "technique": "Technique",
    "instrument": "Instrument",
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
    "medicalcategory": "MedicalCategory",
    "medicalconcept": "MedicalConcept",
    "anatomy": "Anatomy",
    "outcome": "Outcome",
}
_MEDICAL_TYPE_SUPERTYPES = {
    "Disease": ("ClinicalCondition", "MedicalConcept"),
    "Syndrome": ("Disease", "ClinicalCondition", "MedicalConcept"),
    "ClinicalCondition": ("MedicalConcept",),
    "ClinicalFinding": ("Evidence", "MedicalConcept"),
    "Symptom": ("ClinicalFinding", "MedicalConcept"),
    "Test": ("Procedure", "Observation", "MedicalConcept"),
    "TestResult": ("ClinicalFinding", "Evidence", "MedicalConcept"),
    "TestResultPattern": (
        "TestResult",
        "ClinicalFinding",
        "Evidence",
        "MedicalConcept",
    ),
    "Treatment": ("Intervention", "MedicalConcept"),
    "Procedure": ("Treatment", "Intervention", "MedicalConcept"),
    "Drug": ("Intervention", "MedicalConcept"),
    "Vaccine": ("Drug", "Intervention", "PublicHealthMeasure", "MedicalConcept"),
    "PublicHealthMeasure": ("Intervention", "MedicalConcept"),
    "DosingRegimen": ("Treatment", "MedicalConcept"),
    "Pathogen": ("MedicalConcept",),
    "DrugClass": ("MedicalConcept",),
    "DiagnosticCriterion": ("MedicalConcept",),
    "Population": ("MedicalConcept",),
    "RiskFactor": ("MedicalConcept",),
    "Complication": ("Disease", "ClinicalCondition", "MedicalConcept"),
    "AdverseReaction": ("ClinicalFinding", "MedicalConcept"),
    "Guideline": ("Evidence", "MedicalConcept"),
    "Recommendation": ("Evidence", "MedicalConcept"),
    "ClinicalPathway": ("Recommendation", "MedicalConcept"),
    "Evidence": ("MedicalConcept",),
    "Method": ("MedicalConcept",),
    "Technique": ("Method", "MedicalConcept"),
    "Instrument": ("MedicalConcept",),
    "Observation": ("Evidence", "MedicalConcept"),
    "Diagnosis": ("MedicalConcept",),
    "Specimen": ("MedicalConcept",),
    "MedicalGroup": ("MedicalConcept",),
    "MedicalCategory": ("MedicalConcept",),
    "Anatomy": ("MedicalConcept",),
    "Outcome": ("MedicalConcept",),
    "ClinicalDepartment": ("MedicalConcept",),
}


@dataclass(frozen=True)
class MedicalRelationSpec:
    id: str
    zh_label: str
    family: str
    domain_types: tuple[str, ...]
    range_types: tuple[str, ...]
    allowed_qualifiers: tuple[str, ...] = ()
    required_qualifiers: tuple[str, ...] = ()
    required_any_qualifier_groups: tuple[tuple[str, ...], ...] = ()
    qualifier_enums: tuple[tuple[str, tuple[str, ...]], ...] = ()
    guidance: str = ""
    inverse_of: str = ""
    deprecated: bool = False
    canonical_replacement: str = ""


@dataclass(frozen=True)
class MedicalRelationMigrationRule:
    legacy_keywords: tuple[str, ...]
    canonical_options: tuple[str, ...]
    guidance: str


def _generic_relation_spec(relation_id: str, family: str) -> MedicalRelationSpec:
    return MedicalRelationSpec(
        id=relation_id,
        zh_label=relation_id,
        family=family,
        domain_types=("MedicalConcept",),
        range_types=("MedicalConcept",),
    )


MEDICAL_RELATION_SPECS: tuple[MedicalRelationSpec, ...] = (
    MedicalRelationSpec(
        id="is_a",
        zh_label="属于",
        family="ontology",
        domain_types=("MedicalConcept", "Disease", "Symptom", "Drug", "Test"),
        range_types=("MedicalConcept", "Disease", "Symptom", "Drug", "Test"),
        guidance="Use for subtype and taxonomy hierarchy only.",
    ),
    MedicalRelationSpec(
        id="has_manifestation",
        zh_label="临床表现",
        family="diagnosis",
        domain_types=("Disease", "Syndrome", "ClinicalCondition"),
        range_types=("Symptom", "sign_or_symptom", "ClinicalFinding"),
        allowed_qualifiers=("severity", "frequency", "stage", "population"),
        guidance="Disease or clinical condition points to signs and symptoms.",
    ),
    MedicalRelationSpec(
        id="has_indication",
        zh_label="适应证",
        family="treatment",
        domain_types=("Drug", "Treatment", "Procedure", "Vaccine"),
        range_types=("Disease", "ClinicalCondition", "Population"),
        allowed_qualifiers=(
            "dose",
            "frequency",
            "route",
            "duration",
            "age",
            "age_min",
            "age_max",
            "age_unit",
            "population",
            "severity",
            "timing",
            "time_window",
            "purpose",
            "contraindication",
            "evidence_level",
            "version",
        ),
        guidance="Therapy points to the disease or clinical context it applies to.",
    ),
    MedicalRelationSpec(
        id="recommends",
        zh_label="推荐",
        family="guideline",
        domain_types=("Guideline", "Recommendation", "ClinicalPathway"),
        range_types=("Drug", "Treatment", "Procedure", "Test", "Vaccine"),
        allowed_qualifiers=("strength", "evidence_level", "population", "version"),
        required_any_qualifier_groups=(("strength", "evidence_level", "version"),),
        guidance="Recommendation-bearing node points to the recommended action.",
    ),
    MedicalRelationSpec(
        id="has_diagnostic_criterion",
        zh_label="诊断标准",
        family="diagnosis",
        domain_types=("Disease", "ClinicalCondition"),
        range_types=("DiagnosticCriterion", "ClinicalFinding", "TestResultPattern"),
        allowed_qualifiers=("guideline_version", "population", "threshold"),
        guidance="Knowledge-level disease points to diagnostic criteria.",
    ),
    MedicalRelationSpec(
        id="criterion_requires",
        zh_label="标准要求",
        family="diagnosis",
        domain_types=("DiagnosticCriterion",),
        range_types=("Symptom", "sign_or_symptom", "Test", "TestResultPattern"),
        allowed_qualifiers=("threshold", "duration", "time_window"),
        guidance="Criterion points to required findings, tests, or result patterns.",
    ),
    MedicalRelationSpec(
        id="performed_by_method",
        zh_label="检测方法",
        family="tests",
        domain_types=("Test", "Procedure", "Observation"),
        range_types=("Method", "Technique", "Instrument"),
        allowed_qualifiers=("specimen", "timing", "site"),
        guidance="Test or procedure points to the method used to perform it.",
    ),
    MedicalRelationSpec(
        id="supports_or_refutes",
        zh_label="支持或排除",
        family="evidence",
        domain_types=("Evidence", "TestResult", "ClinicalFinding"),
        range_types=("Disease", "ClinicalCondition", "Diagnosis"),
        allowed_qualifiers=("polarity", "certainty", "threshold"),
        required_qualifiers=("polarity",),
        qualifier_enums=(("polarity", ("refutes", "supports")),),
        guidance="Evidence-bearing finding points to the diagnosis it supports/refutes.",
    ),
    MedicalRelationSpec(
        id="has_dosing_regimen",
        zh_label="剂量用法",
        family="treatment",
        domain_types=("Drug", "Treatment", "Vaccine"),
        range_types=("DosingRegimen",),
        allowed_qualifiers=("dose", "frequency", "route", "duration", "population"),
        guidance="Medication or treatment points to a dosing regimen node.",
    ),
    MedicalRelationSpec(
        id="may_cause_adverse_reaction",
        zh_label="可能导致不良反应",
        family="safety",
        domain_types=("Drug", "Treatment", "Vaccine", "Procedure"),
        range_types=("AdverseReaction", "Symptom", "sign_or_symptom"),
        allowed_qualifiers=("frequency", "severity", "population", "time_window"),
        guidance="Intervention points to possible adverse reaction.",
    ),
    _generic_relation_spec("part_of", "ontology"),
    _generic_relation_spec("names", "ontology"),
    _generic_relation_spec("maps_to_code", "ontology"),
    MedicalRelationSpec(
        id="causative_agent",
        zh_label="causative_agent",
        family="diagnosis",
        domain_types=("Disease", "ClinicalCondition"),
        range_types=("Pathogen",),
        guidance="Disease or clinical condition points to the causative pathogen.",
    ),
    MedicalRelationSpec(
        id="has_complication",
        zh_label="并发症",
        family="diagnosis",
        domain_types=("Disease", "ClinicalCondition"),
        range_types=("Complication", "Disease", "ClinicalCondition"),
        guidance="Disease or clinical condition points to complications.",
    ),
    MedicalRelationSpec(
        id="has_risk_factor",
        zh_label="风险因素",
        family="diagnosis",
        domain_types=("Disease", "ClinicalCondition"),
        range_types=("RiskFactor", "Population"),
        guidance="Disease or clinical condition points to risk factors.",
    ),
    _generic_relation_spec("differential_with", "diagnosis"),
    _generic_relation_spec("has_severity_grade", "diagnosis"),
    _generic_relation_spec("has_evidence", "diagnosis"),
    _generic_relation_spec("ruled_out", "diagnosis"),
    _generic_relation_spec("due_to", "diagnosis"),
    MedicalRelationSpec(
        id="orders_test",
        zh_label="orders_test",
        family="tests",
        domain_types=(
            "Recommendation",
            "ClinicalPathway",
            "Disease",
            "ClinicalCondition",
        ),
        range_types=("Test",),
        allowed_qualifiers=("timing", "population", "indication", "urgency", "setting"),
        guidance="Recommendation, pathway, or condition points to an ordered test.",
    ),
    MedicalRelationSpec(
        id="has_result",
        zh_label="has_result",
        family="tests",
        domain_types=("Test", "Observation"),
        range_types=("TestResult",),
        allowed_qualifiers=("threshold", "polarity", "unit", "value", "interpretation"),
        guidance="Test or observation points to a result value or result pattern.",
    ),
    _generic_relation_spec("observes", "tests"),
    _generic_relation_spec("has_value", "tests"),
    _generic_relation_spec("has_interpretation", "tests"),
    MedicalRelationSpec(
        id="uses_specimen",
        zh_label="uses_specimen",
        family="tests",
        domain_types=("Test",),
        range_types=("Specimen",),
        allowed_qualifiers=("site", "collection_method", "timing"),
        guidance="Test points to the specimen used for measurement.",
    ),
    _generic_relation_spec("has_active_ingredient", "treatment"),
    MedicalRelationSpec(
        id="belongs_to_drug_class",
        zh_label="belongs_to_drug_class",
        family="treatment",
        domain_types=("Drug", "Vaccine"),
        range_types=("DrugClass",),
        guidance="Drug points to its pharmacologic class.",
    ),
    MedicalRelationSpec(
        id="recommended_for",
        zh_label="推荐人群",
        family="treatment",
        domain_types=("Drug", "Treatment", "Vaccine", "PublicHealthMeasure"),
        range_types=("Population",),
        allowed_qualifiers=(
            "condition",
            "context",
            "purpose",
            "age",
            "age_min",
            "age_max",
            "age_unit",
            "population",
            "risk",
            "reason",
            "route",
            "timing",
            "time_window",
            "strength",
            "evidence_level",
            "version",
        ),
        required_qualifiers=("purpose",),
        required_any_qualifier_groups=(
            (
                "condition",
                "age",
                "age_min",
                "age_max",
                "population",
                "route",
                "timing",
                "time_window",
            ),
        ),
        qualifier_enums=(("purpose", ("prevention", "treatment")),),
        guidance="Drug, treatment, or vaccine points to the recommended population.",
    ),
    MedicalRelationSpec(
        id="not_recommended_for",
        zh_label="not_recommended_for",
        family="treatment",
        domain_types=("Drug", "Treatment", "Procedure", "Vaccine", "PublicHealthMeasure"),
        range_types=("Population", "ClinicalCondition", "RiskFactor"),
        allowed_qualifiers=(
            "condition",
            "context",
            "age",
            "age_min",
            "age_max",
            "age_unit",
            "population",
            "route",
            "reason",
            "risk",
            "time_window",
            "version",
        ),
        guidance="Intervention points to a population or condition where it is not recommended.",
    ),
    MedicalRelationSpec(
        id="contraindicated_for",
        zh_label="contraindicated_for",
        family="treatment",
        domain_types=("Drug", "Treatment", "Procedure", "Vaccine", "PublicHealthMeasure"),
        range_types=("Population", "ClinicalCondition", "RiskFactor"),
        allowed_qualifiers=(
            "condition",
            "context",
            "age",
            "age_min",
            "age_max",
            "age_unit",
            "population",
            "route",
            "reason",
            "risk",
            "time_window",
            "version",
        ),
        guidance="Intervention points to a population, condition, or risk factor where it must not be used.",
    ),
    MedicalRelationSpec(
        id="precaution_for",
        zh_label="precaution_for",
        family="treatment",
        domain_types=("Drug", "Treatment", "Procedure", "Vaccine", "PublicHealthMeasure"),
        range_types=("Population", "ClinicalCondition", "RiskFactor"),
        allowed_qualifiers=(
            "condition",
            "context",
            "age",
            "age_min",
            "age_max",
            "age_unit",
            "population",
            "route",
            "reason",
            "risk",
            "time_window",
            "version",
        ),
        guidance="Intervention points to a population, condition, or risk factor needing caution.",
    ),
    MedicalRelationSpec(
        id="temporarily_deferred_for",
        zh_label="temporarily_deferred_for",
        family="treatment",
        domain_types=("Drug", "Treatment", "Procedure", "Vaccine", "PublicHealthMeasure"),
        range_types=("Population", "ClinicalCondition", "RiskFactor"),
        allowed_qualifiers=(
            "condition",
            "context",
            "age",
            "age_min",
            "age_max",
            "age_unit",
            "population",
            "route",
            "reason",
            "risk",
            "time_window",
            "version",
        ),
        required_qualifiers=("reason",),
        guidance=(
            "Intervention points to a population, condition, or risk factor "
            "where use should be temporarily deferred rather than permanently "
            "contraindicated."
        ),
    ),
    MedicalRelationSpec(
        id="interaction_with",
        zh_label="interaction_with",
        family="treatment",
        domain_types=("Drug", "Treatment", "Vaccine"),
        range_types=("Drug", "Treatment"),
        allowed_qualifiers=("severity", "mechanism", "management"),
        guidance="Drug, vaccine, or treatment points to an interacting intervention.",
    ),
    MedicalRelationSpec(
        id="monitor_with",
        zh_label="monitor_with",
        family="treatment",
        domain_types=("Drug", "Treatment", "Procedure", "Vaccine"),
        range_types=("Test", "Observation"),
        allowed_qualifiers=("timing", "frequency", "reason", "threshold"),
        guidance="Intervention points to a test or observation used for monitoring.",
    ),
    MedicalRelationSpec(
        id="targets_disease",
        zh_label="目标疾病",
        family="prevention",
        domain_types=("Vaccine", "PublicHealthMeasure"),
        range_types=("Disease", "ClinicalCondition"),
        guidance="Vaccine or public-health measure points to the target disease.",
    ),
    MedicalRelationSpec(
        id="reduces_risk_of",
        zh_label="降低风险",
        family="prevention",
        domain_types=("Vaccine", "PublicHealthMeasure"),
        range_types=("Disease", "ClinicalCondition", "Outcome", "Complication"),
        guidance=(
            "Vaccine or public-health measure points to the disease, outcome, "
            "or complication whose risk is reduced."
        ),
    ),
    _generic_relation_spec("has_dose_schedule", "prevention"),
    MedicalRelationSpec(
        id="risk_factor_for",
        zh_label="risk factor for",
        family="risk",
        domain_types=("RiskFactor", "ClinicalCondition"),
        range_types=("Disease", "ClinicalCondition", "Outcome", "Complication"),
        allowed_qualifiers=(
            "population",
            "age",
            "severity",
            "time_window",
            "magnitude",
            "certainty",
            "evidence_level",
            "version",
        ),
        inverse_of="has_risk_factor",
        guidance=(
            "Risk factor or underlying clinical condition points to the disease "
            "or adverse outcome for which it is a risk factor."
        ),
    ),
    MedicalRelationSpec(
        id="high_risk_for",
        zh_label="high risk for",
        family="risk",
        domain_types=("Population",),
        range_types=("Disease", "ClinicalCondition", "Outcome", "Complication"),
        allowed_qualifiers=(
            "condition",
            "age",
            "severity",
            "time_window",
            "reason",
            "evidence_level",
            "version",
        ),
        guidance=(
            "A population points to the disease, severe state, or outcome for "
            "which it is considered high risk."
        ),
    ),
    MedicalRelationSpec(
        id="increases_risk_of",
        zh_label="increases risk of",
        family="risk",
        domain_types=("RiskFactor", "Disease", "ClinicalCondition"),
        range_types=("Disease", "ClinicalCondition", "Outcome", "Complication"),
        allowed_qualifiers=(
            "population",
            "time_window",
            "magnitude",
            "relative_risk",
            "odds_ratio",
            "hazard_ratio",
            "certainty",
            "evidence_level",
            "version",
        ),
        guidance=(
            "A disease, condition, or risk factor points to the clinical event "
            "whose probability is increased."
        ),
    ),
    MedicalRelationSpec(
        id="acute_exacerbation_of",
        zh_label="acute exacerbation of",
        family="risk",
        domain_types=("Disease", "ClinicalCondition"),
        range_types=("Disease", "ClinicalCondition"),
        allowed_qualifiers=("trigger", "severity", "time_window", "population"),
        guidance=(
            "An acute exacerbation state points to its underlying chronic disease."
        ),
    ),
    MedicalRelationSpec(
        id="risk_group_for",
        zh_label="风险人群",
        family="prevention",
        domain_types=("Population", "RiskFactor"),
        range_types=("Disease", "ClinicalCondition"),
        guidance="Population or risk factor points to the disease it is a risk group for.",
        deprecated=True,
        canonical_replacement="high_risk_for",
    ),
    _generic_relation_spec("defined_by_characteristic", "prevention"),
    MedicalRelationSpec(
        id="evidenced_by",
        zh_label="evidenced_by",
        family="evidence",
        domain_types=("MedicalConcept",),
        range_types=("Evidence", "Guideline"),
        allowed_qualifiers=("source_id", "file_path", "page", "span", "version"),
        guidance="Medical fact points to the evidence source that supports it.",
    ),
    _generic_relation_spec("asserted_by", "guideline"),
    _generic_relation_spec("issued_by", "guideline"),
    _generic_relation_spec("valid_during", "guideline"),
    _generic_relation_spec("provided_by", "hospital"),
    _generic_relation_spec("available_at", "hospital"),
)

CANONICAL_MEDICAL_RELATION_IDS = frozenset(
    spec.id for spec in MEDICAL_RELATION_SPECS
)
_RELATION_SPEC_BY_ID = {spec.id: spec for spec in MEDICAL_RELATION_SPECS}


LEGACY_MEDICAL_RELATION_MIGRATIONS: tuple[MedicalRelationMigrationRule, ...] = (
    MedicalRelationMigrationRule(
        legacy_keywords=("推荐治疗", "recommended_treatment", "treatment_recommendation"),
        canonical_options=("has_indication", "recommends"),
        guidance=(
            "药物/方案可用 has_indication 指向疾病或临床情境；指南/推荐意见"
            "用 recommends 指向药物或治疗方案。不要把适用人群、剂量、禁忌"
            "塞进同一条推荐治疗边。"
        ),
    ),
    MedicalRelationMigrationRule(
        legacy_keywords=("诊断依据", "diagnostic_basis", "diagnosis_basis"),
        canonical_options=(
            "has_diagnostic_criterion",
            "criterion_requires",
            "has_evidence",
            "supports_or_refutes",
        ),
        guidance=(
            "疾病知识层用 has_diagnostic_criterion 和 criterion_requires；"
            "患者实例层用 has_evidence；证据模式用 supports_or_refutes。"
            "样本、方法、阳性结果不能混成同一条诊断依据边。"
        ),
    ),
    MedicalRelationMigrationRule(
        legacy_keywords=("检测方法", "test_method", "diagnostic_method"),
        canonical_options=("performed_by_method",),
        guidance="方法只修饰检查/观察/程序，不直接连到疾病作为诊断证据。",
    ),
    MedicalRelationMigrationRule(
        legacy_keywords=("适用于", "applies_to", "applicable_to"),
        canonical_options=("recommended_for", "has_indication"),
        guidance=(
            "面向人群或场景时用 recommended_for；表达药物/疫苗/方案可用于"
            "某疾病或临床情境时用 has_indication。"
        ),
    ),
    MedicalRelationMigrationRule(
        legacy_keywords=("预防措施", "prevention_measure", "prevents"),
        canonical_options=("targets_disease", "reduces_risk_of"),
        guidance=(
            "疫苗或预防措施指向疾病/传播/并发症结局，不指向人群；"
            "人群推荐另用 recommended_for。"
        ),
    ),
    MedicalRelationMigrationRule(
        legacy_keywords=("并发风险", "complication_risk", "adverse_risk"),
        canonical_options=(
            "has_complication",
            "risk_factor_for",
            "high_risk_for",
            "increases_risk_of",
            "acute_exacerbation_of",
            "may_cause_adverse_reaction",
        ),
        guidance=(
            "疾病自然病程的并发症用 has_complication；药物、疫苗或操作的"
            "潜在负面结果用 may_cause_adverse_reaction；风险因素用 "
            "risk_factor_for；高危人群用 high_risk_for；概率增加用 "
            "increases_risk_of；急性加重状态指向基础慢病用 "
            "acute_exacerbation_of。"
        ),
    ),
    MedicalRelationMigrationRule(
        legacy_keywords=("剂量用法", "dosage_usage", "dose_usage"),
        canonical_options=("has_dosing_regimen",),
        guidance=(
            "剂量、频次、途径、疗程、时间窗放在给药方案或关系限定属性里，"
            "不要把 75 mg、每日2次、5天建成主实体。"
        ),
    ),
    MedicalRelationMigrationRule(
        legacy_keywords=("病原分型", "pathogen_subtype", "pathogen_typing"),
        canonical_options=("is_a", "causative_agent"),
        guidance=(
            "病毒亚型层级用子类 -> is_a -> 父类；疾病指向致病病原体用"
            " causative_agent。"
        ),
    ),
    MedicalRelationMigrationRule(
        legacy_keywords=("指南建议", "guideline_recommendation"),
        canonical_options=("recommends", "asserted_by"),
        guidance=(
            "推荐意见作为中间节点承载推荐强度、证据等级、适用条件和版本；"
            "医学事实通过 asserted_by 追到指南/推荐。"
        ),
    ),
)

_LEGACY_RULE_BY_TOKEN = {
    keyword.casefold(): rule
    for rule in LEGACY_MEDICAL_RELATION_MIGRATIONS
    for keyword in rule.legacy_keywords
}
_LEGACY_KEYWORD_OPTION_FALLBACKS = {
    "推荐治疗": frozenset({"has_indication", "recommends"}),
    "诊断依据": frozenset(
        {
            "has_diagnostic_criterion",
            "criterion_requires",
            "supports_or_refutes",
        }
    ),
}


def relation_spec_by_id(relation_id: str) -> MedicalRelationSpec:
    return _RELATION_SPEC_BY_ID[relation_id]


def normalize_medical_entity_type(entity_type: object) -> str:
    if not isinstance(entity_type, str):
        return "Unknown"
    normalized = (
        entity_type.strip()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .casefold()
    )
    if not normalized or normalized == "unknown":
        return "Unknown"
    return _NORMALIZED_MEDICAL_ENTITY_TYPES.get(normalized, "Unknown")


def medical_type_allowed(
    actual_type: object,
    expected_types: tuple[str, ...],
) -> bool:
    normalized_actual = normalize_medical_entity_type(actual_type)
    if normalized_actual == "Unknown":
        return False
    normalized_expected = _normalize_expected_types(expected_types)
    if normalized_actual in normalized_expected:
        return True
    actual_lineage = _medical_type_lineage(normalized_actual)
    return any(expected_type in actual_lineage for expected_type in normalized_expected)


def validate_relation_instance(
    *,
    predicate: str,
    source_type: object,
    target_type: object,
    qualifiers: Mapping[str, Any] | None = None,
) -> list[str]:
    spec = relation_spec_by_id(predicate)
    normalized_source_type = normalize_medical_entity_type(source_type)
    normalized_target_type = normalize_medical_entity_type(target_type)
    errors: list[str] = []

    if not medical_type_allowed(normalized_source_type, spec.domain_types):
        errors.append(
            "source_type "
            f"{normalized_source_type} is outside {predicate} domain "
            f"{_format_expected_types(spec.domain_types)}"
        )
    if not medical_type_allowed(normalized_target_type, spec.range_types):
        errors.append(
            "target_type "
            f"{normalized_target_type} is outside {predicate} range "
            f"{_format_expected_types(spec.range_types)}"
        )

    qualifier_keys = _non_empty_qualifier_keys(qualifiers)
    for required_key in spec.required_qualifiers:
        if required_key not in qualifier_keys:
            errors.append(f"{predicate} requires qualifier {required_key}")
    for required_group in spec.required_any_qualifier_groups:
        if not (set(required_group) & qualifier_keys):
            errors.append(
                f"{predicate} requires one qualifier from "
                + " | ".join(required_group)
            )
    if spec.allowed_qualifiers:
        unexpected_keys = sorted(qualifier_keys - set(spec.allowed_qualifiers))
        if unexpected_keys:
            errors.append(
                f"{predicate} does not allow qualifier(s) "
                + ", ".join(unexpected_keys)
            )
    qualifier_values = qualifiers if isinstance(qualifiers, Mapping) else {}
    for qualifier_name, allowed_values in spec.qualifier_enums:
        if qualifier_name not in qualifier_keys:
            continue
        value = str(qualifier_values.get(qualifier_name, "")).strip().casefold()
        allowed_normalized = tuple(item.casefold() for item in allowed_values)
        if value not in allowed_normalized:
            errors.append(
                f"{predicate} qualifier {qualifier_name} must be one of "
                + " | ".join(allowed_values)
            )

    return errors


def _normalize_expected_types(types: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for type_name in types:
        normalized_type = normalize_medical_entity_type(type_name)
        if normalized_type == "Unknown" and type_name:
            normalized_type = str(type_name)
        if normalized_type not in normalized:
            normalized.append(normalized_type)
    return tuple(normalized)


def _medical_type_lineage(entity_type: str) -> frozenset[str]:
    lineage = {entity_type}
    pending = list(_MEDICAL_TYPE_SUPERTYPES.get(entity_type, ()))
    while pending:
        parent = pending.pop()
        if parent in lineage:
            continue
        lineage.add(parent)
        pending.extend(_MEDICAL_TYPE_SUPERTYPES.get(parent, ()))
    return frozenset(lineage)


def _format_expected_types(types: tuple[str, ...]) -> str:
    return " | ".join(_normalize_expected_types(types))


def _non_empty_qualifier_keys(
    qualifiers: Mapping[str, Any] | None,
) -> set[str]:
    if qualifiers is None:
        return set()
    if not isinstance(qualifiers, Mapping):
        return {"<invalid_qualifiers>"}
    return {str(key) for key, value in qualifiers.items() if str(value).strip()}


def migration_rule_for_legacy_keyword(
    keyword: str,
) -> MedicalRelationMigrationRule | None:
    normalized = keyword.strip().casefold()
    rule = _LEGACY_RULE_BY_TOKEN.get(normalized)
    if rule is not None:
        return rule

    expected_options = _LEGACY_KEYWORD_OPTION_FALLBACKS.get(keyword.strip())
    if expected_options is None:
        return None
    for candidate in LEGACY_MEDICAL_RELATION_MIGRATIONS:
        if expected_options.issubset(candidate.canonical_options):
            return candidate
    return None


def is_medical_profile(profile: str | None) -> bool:
    if not profile:
        return False
    normalized = profile.strip().casefold()
    return any(hint in normalized for hint in _MEDICAL_PROFILE_HINTS)


def legacy_relation_migration_for_tokens(
    tokens: set[str],
) -> MedicalRelationMigrationRule | None:
    for token in tokens:
        rule = _LEGACY_RULE_BY_TOKEN.get(token.casefold())
        if rule is not None:
            return rule
    return None


def render_medical_relation_schema_prompt(profile: str | None) -> str:
    if not is_medical_profile(profile):
        return ""

    migrations = "\n".join(
        "- "
        + "/".join(rule.legacy_keywords[:2])
        + " -> "
        + " | ".join(rule.canonical_options)
        for rule in LEGACY_MEDICAL_RELATION_MIGRATIONS
    )
    return f"""## Medical Relationship Schema v1 ({MEDICAL_RELATION_SCHEMA_VERSION})

Scope: hospital-facing medical KG for patient education, clinician diagnosis support, treatment review, prevention, evidence tracing, and hospital service routing.

Hard rules:
- Separate knowledge-level facts from patient-instance facts.
- Use canonical predicate IDs internally and Chinese display labels in UI.
- Keep source_id, file_path, chunk/evidence span, guideline version, and recommendation strength whenever a medical fact is proposed.
- Put dose, frequency, route, duration, age range, time window, threshold, severity, season, and population constraints into qualifiers instead of standalone concept nodes.
- Do not invent medical facts from LLM inference. If deterministic evidence is missing, request more context instead of proposing a mutation.

Canonical relation families:
- Ontology/terminology: is_a, part_of, names, maps_to_code.
- Diagnosis/problem list: causative_agent, has_manifestation, has_complication, has_risk_factor, has_diagnostic_criterion, criterion_requires, differential_with, has_severity_grade, has_evidence, ruled_out, due_to.
- Tests/evidence: orders_test, has_result, observes, has_value, has_interpretation, uses_specimen, performed_by_method, supports_or_refutes.
- Drug/treatment: has_active_ingredient, belongs_to_drug_class, has_indication, recommends, recommended_for, not_recommended_for, contraindicated_for, precaution_for, temporarily_deferred_for, has_dosing_regimen, may_cause_adverse_reaction, interaction_with, monitor_with.
- Vaccine/prevention: targets_disease, reduces_risk_of, has_dose_schedule, high_risk_for, defined_by_characteristic.
- Evidence/guideline/hospital: evidenced_by, asserted_by, issued_by, valid_during, provided_by, available_at.

Direction examples:
- 流行性感冒 -> has_manifestation -> 干咳；干咳 -> is_a -> 呼吸道症状。
- 流感病毒核酸检测 -> performed_by_method -> RT-PCR；核酸检出结果模式 -> supports_or_refutes -> 流行性感冒。
- 推荐意见 -> recommends -> 奥司他韦；奥司他韦 -> has_indication -> 流行性感冒；奥司他韦 -> has_dosing_regimen -> 成人给药方案。
- 流感疫苗 -> targets_disease -> 流行性感冒；流感疫苗 -> recommended_for -> 适用人群；流感疫苗 -> contraindicated_for -> 禁忌条件。

Legacy relation migration rules:
{migrations}
"""
