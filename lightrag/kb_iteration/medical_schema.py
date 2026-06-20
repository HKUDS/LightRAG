from __future__ import annotations

from dataclasses import dataclass


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


@dataclass(frozen=True)
class MedicalRelationSpec:
    id: str
    zh_label: str
    family: str
    domain_types: tuple[str, ...]
    range_types: tuple[str, ...]
    allowed_qualifiers: tuple[str, ...] = ()
    guidance: str = ""


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
        allowed_qualifiers=("population", "severity", "timing", "contraindication"),
        guidance="Therapy points to the disease or clinical context it applies to.",
    ),
    MedicalRelationSpec(
        id="recommends",
        zh_label="推荐",
        family="guideline",
        domain_types=("Guideline", "Recommendation", "ClinicalPathway"),
        range_types=("Drug", "Treatment", "Procedure", "Test", "Vaccine"),
        allowed_qualifiers=("strength", "evidence_level", "population", "version"),
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
    _generic_relation_spec("causative_agent", "diagnosis"),
    _generic_relation_spec("has_complication", "diagnosis"),
    _generic_relation_spec("has_risk_factor", "diagnosis"),
    _generic_relation_spec("differential_with", "diagnosis"),
    _generic_relation_spec("has_severity_grade", "diagnosis"),
    _generic_relation_spec("has_evidence", "diagnosis"),
    _generic_relation_spec("ruled_out", "diagnosis"),
    _generic_relation_spec("due_to", "diagnosis"),
    _generic_relation_spec("orders_test", "tests"),
    _generic_relation_spec("has_result", "tests"),
    _generic_relation_spec("observes", "tests"),
    _generic_relation_spec("has_value", "tests"),
    _generic_relation_spec("has_interpretation", "tests"),
    _generic_relation_spec("uses_specimen", "tests"),
    _generic_relation_spec("has_active_ingredient", "treatment"),
    _generic_relation_spec("belongs_to_drug_class", "treatment"),
    _generic_relation_spec("recommended_for", "treatment"),
    _generic_relation_spec("not_recommended_for", "treatment"),
    _generic_relation_spec("contraindicated_for", "treatment"),
    _generic_relation_spec("precaution_for", "treatment"),
    _generic_relation_spec("interaction_with", "treatment"),
    _generic_relation_spec("monitor_with", "treatment"),
    _generic_relation_spec("targets_disease", "prevention"),
    _generic_relation_spec("reduces_risk_of", "prevention"),
    _generic_relation_spec("has_dose_schedule", "prevention"),
    _generic_relation_spec("risk_group_for", "prevention"),
    _generic_relation_spec("defined_by_characteristic", "prevention"),
    _generic_relation_spec("evidenced_by", "evidence"),
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
        canonical_options=("has_complication", "may_cause_adverse_reaction"),
        guidance=(
            "疾病自然病程的并发症用 has_complication；药物、疫苗或操作的"
            "潜在负面结果用 may_cause_adverse_reaction。"
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
- Drug/treatment: has_active_ingredient, belongs_to_drug_class, has_indication, recommends, recommended_for, not_recommended_for, contraindicated_for, precaution_for, has_dosing_regimen, may_cause_adverse_reaction, interaction_with, monitor_with.
- Vaccine/prevention: targets_disease, reduces_risk_of, has_dose_schedule, risk_group_for, defined_by_characteristic.
- Evidence/guideline/hospital: evidenced_by, asserted_by, issued_by, valid_during, provided_by, available_at.

Direction examples:
- 流行性感冒 -> has_manifestation -> 干咳；干咳 -> is_a -> 呼吸道症状。
- 流感病毒核酸检测 -> performed_by_method -> RT-PCR；核酸检出结果模式 -> supports_or_refutes -> 流行性感冒。
- 推荐意见 -> recommends -> 奥司他韦；奥司他韦 -> has_indication -> 流行性感冒；奥司他韦 -> has_dosing_regimen -> 成人给药方案。
- 流感疫苗 -> targets_disease -> 流行性感冒；流感疫苗 -> recommended_for -> 适用人群；流感疫苗 -> contraindicated_for -> 禁忌条件。

Legacy relation migration rules:
{migrations}
"""
