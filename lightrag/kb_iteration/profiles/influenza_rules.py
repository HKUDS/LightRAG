from __future__ import annotations


INFLUENZA_DISEASE_TERMS = (
    "流行性感冒",
    "甲型流感",
    "乙型流感",
    "重症流感",
    "流感",
    "influenza",
)
INFLUENZA_CONTEXT_TERMS = (
    "流感",
    "流行性感冒",
    "influenza",
)
GENERIC_INFLUENZA_VIRUS_TERMS = (
    "流感病毒",
    "influenza virus",
)
TYPED_INFLUENZA_DISEASE_TO_PATHOGEN = {
    "甲型流感": "甲型流感病毒",
    "乙型流感": "乙型流感病毒",
    "influenza a": "influenza A virus",
    "influenza b": "influenza B virus",
}
TYPED_INFLUENZA_DISEASE_TERMS = (
    "甲型流感",
    "乙型流感",
    "甲型h1n1流感",
    "甲型h3n2流感",
    "influenza a",
    "influenza b",
)
BACTERIAL_PATHOGEN_TERMS = (
    "肺炎链球菌",
    "金黄色葡萄球菌",
    "链球菌",
    "葡萄球菌",
    "细菌",
    "bacteria",
    "bacterium",
    "streptococcus",
    "staphylococcus",
)
DIRECT_ORDER_TEST_TERMS = (
    "血常规",
    "血生化",
    "动脉血气分析",
    "病原学检查",
    "核酸检测",
    "抗原检测",
    "rt-pcr",
    "反转录聚合酶链反应",
    "病毒分离培养",
)
BARE_LAB_MARKER_TERMS = (
    "肌酐",
    "乳酸脱氢酶",
    "丙氨酸氨基转移酶",
    "天门冬氨酸氨基转移酶",
    "转氨酶",
    "肌酸激酶",
    "肌红蛋白",
    "alt",
    "ast",
    "ldh",
    "creatinine",
)
NONSPECIFIC_DIAGNOSTIC_CRITERION_TERMS = (
    "血常规",
    "外周血常规",
    "丙氨酸氨基转移酶",
    "天门冬氨酸氨基转移酶",
    "转氨酶",
    "肝酶",
    "肌酐",
    "乳酸脱氢酶",
    "肌酸激酶",
    "肌红蛋白",
    "血生化",
    "血气",
    "动脉血气",
    "动脉血气分析",
    "mri",
)
COMPLICATION_IMAGING_TEST_TERMS = (
    "ct",
    "mri",
    "核磁",
    "磁共振",
    "影像学",
    "胸部影像",
    "胸部影像学检查",
)
COMPLICATION_OR_SEVERITY_CONTEXT_TERMS = (
    "肺炎",
    "脑炎",
    "脑病",
    "急性坏死性脑病",
    "ards",
    "呼吸衰竭",
    "重型",
    "危重",
    "重症",
    "并发",
    "严重",
    "pneumonia",
    "encephalitis",
    "encephalopathy",
    "complication",
    "severe",
    "severity",
)
CHRONIC_UNDERLYING_CONDITION_TERMS = (
    "慢性阻塞性肺疾病",
    "慢阻肺",
    "copd",
    "chronic obstructive pulmonary disease",
)
ACUTE_EXACERBATION_TERMS = (
    "急性加重",
    "急性恶化",
    "exacerbation",
)
NON_COMPLICATION_OUTCOME_OR_SEVERITY_TERMS = (
    "死亡",
    "住院",
    "不良妊娠结局",
    "低出生体重",
    "早产",
    "流产",
    "重症",
    "危重",
    "严重",
    "病例",
    "心肌梗死",
    "缺血性中风",
    "中风",
    "卒中",
    "心脏停搏",
    "肺气肿",
    "并发症",
    "呼吸系统疾病",
    "mortality",
    "death",
    "hospitalization",
    "pregnancy outcome",
    "low birth weight",
    "preterm",
    "miscarriage",
    "severe",
    "critical",
    "stroke",
    "myocardial infarction",
)
GROUNDED_ZANAMIVIR_REVIEW_ENDPOINT_TERMS = (
    "哮喘",
    "慢性呼吸道疾病",
    "儿童",
    "患儿",
    "7岁",
    "asthma",
    "chronic respiratory",
    "children",
    "child",
)
TAXONOMY_PARENT_TERMS = (
    "病原体",
    "病毒",
    "流感病毒",
    "甲型流感病毒",
    "乙型流感病毒",
    "influenza virus",
    "virus",
    "pathogen",
)
TAXONOMY_SUBTYPE_TERMS = (
    "亚型",
    "型流感病毒",
    "型",
    "系",
    "h1n1",
    "h3n2",
    "h5n1",
    "h7n9",
    "pdm09",
    "lineage",
    "subtype",
)


def typed_pathogen_for_disease(disease: str) -> str:
    normalized = disease.strip().casefold()
    for disease_name, pathogen in TYPED_INFLUENZA_DISEASE_TO_PATHOGEN.items():
        if normalized == disease_name.casefold():
            return pathogen
    return ""


def is_generic_influenza_virus(value: str) -> bool:
    return value.strip().casefold() in {
        term.casefold() for term in GENERIC_INFLUENZA_VIRUS_TERMS
    }


def looks_like_influenza_context(value: str) -> bool:
    return _contains_any(value, INFLUENZA_CONTEXT_TERMS)


def looks_like_severe_influenza_context(value: str) -> bool:
    normalized = value.strip().casefold()
    return looks_like_influenza_context(normalized) and any(
        term in normalized for term in ("重型", "危重", "重症", "severe", "critical")
    )


def looks_like_bare_lab_marker(value: str) -> bool:
    return _contains_any(value, BARE_LAB_MARKER_TERMS)


def looks_like_direct_order_test_source(value: str) -> bool:
    return _contains_any(value, DIRECT_ORDER_TEST_TERMS)


def looks_like_non_complication_outcome_or_severity(value: str) -> bool:
    return _contains_any(value, NON_COMPLICATION_OUTCOME_OR_SEVERITY_TERMS)


def looks_like_bare_lab_marker_order(source: str, target: str) -> bool:
    return looks_like_influenza_context(source) and looks_like_bare_lab_marker(target)


def looks_like_complication_imaging_order(source: str, target: str) -> bool:
    if not _contains_any(target, COMPLICATION_IMAGING_TEST_TERMS):
        return False
    if _contains_any(source, COMPLICATION_OR_SEVERITY_CONTEXT_TERMS):
        return False
    return looks_like_influenza_context(source)


def looks_like_bacterial_agent_for_influenza(source: str, target: str) -> bool:
    return looks_like_influenza_context(source) and _contains_any(
        target, BACTERIAL_PATHOGEN_TERMS
    )


def looks_like_generic_influenza_virus_for_typed_flu(
    source: str,
    target: str,
) -> bool:
    return _contains_any(source, TYPED_INFLUENZA_DISEASE_TERMS) and (
        target.strip().casefold()
        in {term.casefold() for term in GENERIC_INFLUENZA_VIRUS_TERMS}
    )


def looks_like_nonspecific_evidence_supporting_influenza(
    source: str,
    target: str,
) -> bool:
    return _contains_any(
        source, NONSPECIFIC_DIAGNOSTIC_CRITERION_TERMS
    ) and looks_like_influenza_context(target)


def looks_like_nonspecific_diagnostic_criterion(
    source: str,
    target: str,
) -> bool:
    return looks_like_influenza_context(source) and _contains_any(
        target, NONSPECIFIC_DIAGNOSTIC_CRITERION_TERMS
    )


def looks_like_chronic_condition_as_influenza_complication(
    source: str,
    target: str,
) -> bool:
    if _contains_any(target, ACUTE_EXACERBATION_TERMS):
        return False
    return looks_like_influenza_context(source) and _contains_any(
        target, CHRONIC_UNDERLYING_CONDITION_TERMS
    )


def looks_like_outcome_or_severity_as_influenza_complication(
    source: str,
    target: str,
) -> bool:
    return looks_like_influenza_context(source) and _contains_any(
        target, NON_COMPLICATION_OUTCOME_OR_SEVERITY_TERMS
    )


def looks_like_avoidable_zanamivir_review(text: str) -> bool:
    normalized = text.casefold()
    if "扎那米韦" not in normalized and "zanamivir" not in normalized:
        return False
    return _contains_any(normalized, GROUNDED_ZANAMIVIR_REVIEW_ENDPOINT_TERMS)


def looks_like_parent_to_subtype_is_a(source: str, target: str) -> bool:
    normalized_source = source.strip().casefold()
    normalized_target = target.strip().casefold()
    if not normalized_source or not normalized_target:
        return False
    if not any(
        term.casefold() == normalized_source for term in TAXONOMY_PARENT_TERMS
    ):
        return False
    return any(term.casefold() in normalized_target for term in TAXONOMY_SUBTYPE_TERMS)


def _contains_any(value: str, terms: tuple[str, ...]) -> bool:
    normalized = value.strip().casefold()
    if not normalized:
        return False
    return any(term.casefold() in normalized for term in terms)
