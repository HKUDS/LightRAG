from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern


CANONICAL_ALIASES: dict[str, str] = {
    "流感": "流行性感冒",
    "ARDS": "急性呼吸窘迫综合征（ARDS）",
    "急性呼吸窘迫综合征(ARDS)": "急性呼吸窘迫综合征（ARDS）",
    "oseltamivir": "奥司他韦（oseltamivir）",
}

VALUE_ENTITY_TYPES = {"Dosage", "TimeCourse", "Biomarker"}
_VALUE_ENTITY_TYPES_NORMALIZED = {entity_type.casefold() for entity_type in VALUE_ENTITY_TYPES}

VALUE_LIKE_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|kg|g|ml)\b", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?\s*%"),
    re.compile(r"\d+\s*次\s*/\s*日"),
    re.compile(r"\d+(?:\.\d+)?\s*(?:岁|天|小时)"),
    re.compile(r"(?:≥|≤|=|>|<)\s*\d+"),
    re.compile(r"(?:每日|每次|疗程|发病|暴露后)"),
    re.compile(r"(?:PaO2/FiO2|SpO2|RR|CRP|PCT)", re.IGNORECASE),
)

RELATION_KEYWORD_ALIASES: dict[str, str] = {
    "导致": "病原导致",
    "病原": "病原导致",
    "症状": "临床表现",
    "表现": "临床表现",
    "并发": "并发风险",
    "治疗": "推荐治疗",
    "用药": "剂量用法",
    "剂量": "剂量用法",
    "检查": "诊断依据",
    "诊断": "诊断依据",
    "预防": "预防措施",
    "建议": "指南建议",
}
CANONICAL_RELATION_KEYWORDS: frozenset[str] = frozenset(
    {
        "病原导致",
        "病原分型",
        "临床表现",
        "症状归类",
        "并发风险",
        "高危因素",
        "推荐治疗",
        "剂量用法",
        "诊断依据",
        "检测方法",
        "重症判定",
        "预防措施",
        "指南建议",
        "适用于",
        "属于",
    }
)


@dataclass(frozen=True)
class MedicalCategory:
    key: str
    label: str
    aliases: tuple[str, ...] = ()


TOP_LEVEL_MEDICAL_CATEGORIES: tuple[MedicalCategory, ...] = (
    MedicalCategory("pathogen", "病原体", ("病原", "病原学")),
    MedicalCategory("transmission_epidemiology", "传播/流行病学", ("传播", "流行病学")),
    MedicalCategory("clinical_manifestation", "临床表现", ("症状", "体征", "症状表现")),
    MedicalCategory("complication_severity", "并发症/重症", ("并发症", "重症", "危重症")),
    MedicalCategory("diagnosis_testing", "诊断/检查", ("诊断", "检查", "检测", "检验")),
    MedicalCategory("treatment", "治疗", ("用药", "治疗方案", "抗病毒治疗")),
    MedicalCategory("prevention", "预防", ("疫苗", "预防措施", "隔离")),
    MedicalCategory("high_risk_population", "高危人群", ("高危", "风险人群", "特殊人群")),
    MedicalCategory("guideline_evidence", "指南/证据来源", ("指南", "证据来源", "诊疗方案")),
)

EXTENSION_MEDICAL_CATEGORIES: tuple[MedicalCategory, ...] = (
    MedicalCategory("differential_diagnosis", "鉴别诊断", ("鉴别", "相似疾病")),
    MedicalCategory("nursing_care", "护理", ("照护", "居家护理")),
    MedicalCategory("follow_up", "随访", ("复诊", "复诊/随访", "随访观察")),
    MedicalCategory("rehabilitation", "康复", ("恢复期管理",)),
    MedicalCategory("contraindication", "禁忌证", ("用药禁忌", "禁忌", "不宜使用")),
    MedicalCategory("adverse_reaction", "不良反应", ("副作用", "药物不良反应")),
    MedicalCategory("public_health", "公共卫生处置", ("报告", "隔离管理", "学校防控")),
)

OTHER_MEDICAL_CATEGORY = MedicalCategory("other_medical", "其他医学")

_ALL_MEDICAL_CATEGORIES: tuple[MedicalCategory, ...] = (
    *TOP_LEVEL_MEDICAL_CATEGORIES,
    *EXTENSION_MEDICAL_CATEGORIES,
    OTHER_MEDICAL_CATEGORY,
)
CATEGORY_BY_KEY: dict[str, MedicalCategory] = {
    category.key: category for category in _ALL_MEDICAL_CATEGORIES
}
CATEGORY_ALIAS_MAP: dict[str, str] = {}
for _category in _ALL_MEDICAL_CATEGORIES:
    for _alias in (_category.key, _category.label, *_category.aliases):
        CATEGORY_ALIAS_MAP[_alias.strip().casefold()] = _category.key


@dataclass(frozen=True)
class DroppedNode:
    name: str
    reason: str
    replacement: str | None = None


def canonical_name(name: str) -> str:
    normalized = name.strip()
    return CANONICAL_ALIASES.get(normalized, normalized)


def is_value_like_entity(name: str, entity_type: str | None) -> bool:
    if entity_type and entity_type.casefold() in _VALUE_ENTITY_TYPES_NORMALIZED:
        return True

    normalized = name.strip()
    return any(pattern.search(normalized) for pattern in VALUE_LIKE_PATTERNS)


def normalize_relation_keyword(keywords: str) -> str:
    normalized = keywords.strip()
    if normalized in CANONICAL_RELATION_KEYWORDS:
        return normalized

    exact_match = RELATION_KEYWORD_ALIASES.get(normalized)
    if exact_match:
        return exact_match

    for alias, canonical in RELATION_KEYWORD_ALIASES.items():
        if alias in normalized:
            return canonical
    return normalized


def normalize_medical_category_key(category: str) -> str:
    normalized = category.strip().casefold()
    return CATEGORY_ALIAS_MAP.get(normalized, OTHER_MEDICAL_CATEGORY.key)


def medical_category_label(category: str) -> str:
    category_key = normalize_medical_category_key(category)
    return CATEGORY_BY_KEY[category_key].label
