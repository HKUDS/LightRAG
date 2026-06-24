from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

from .medical_schema import (
    CANONICAL_MEDICAL_RELATION_IDS,
    medical_type_allowed,
    relation_spec_by_id,
    validate_relation_instance,
)
from .models import ImprovementProposal
from .profiles import influenza_rules

MUTATION_PROPOSAL_TYPES = {
    "prompt_edit",
    "ontology_rule_change",
    "hierarchy_rule_change",
    "add_hierarchy_branch",
    "relation_rule_change",
    "workspace_rebuild",
    "kg_fact_correction",
    "web_display_change",
    "source_evidence_repair",
    "synonym_merge_rule",
    "relation_keyword_mapping",
    "medical_relation_schema_migration",
    "entity_alias_merge",
    "value_node_to_qualifier",
    "medical_fact_role_split",
    "candidate_kg_expansion",
}

NO_APPROVAL_PROPOSAL_TYPES = {"quality_report_note"}
REVIEW_ONLY_PROPOSAL_TYPES = {
    "review_context_request",
    "llm_judge_rejection",
}
EXECUTABLE_MEDICAL_PROPOSAL_TYPES_REQUIRING_NON_EMPTY_PAYLOAD = {
    "value_node_to_qualifier",
    "entity_alias_merge",
}
MIN_CANDIDATE_KG_EXPANSION_QUOTE_CHARS = 12
_ALLOWED_VALUE_QUALIFIER_KEYS = {
    "dose",
    "frequency",
    "duration",
    "route",
    "age",
    "population",
    "time_window",
}
_ALLOWED_RISKS = {"low", "medium", "high"}
PROPOSAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_TYPE_PATTERN = re.compile(r"^[a-z0-9_]+$")
_METRIC_KEY_PATTERN = re.compile(r"^[a-z0-9_.-]+$")
_REPORT_NOTE_SAFE_PREFIXES = ("Record ", "Add note ", "Document ", "Summarize ")
_CONTROLLED_MUTATION_ACTIONS = (
    r"add|alter|apply|change|clear|correct|delete|drop|edit|fix|modify|move|patch|"
    r"rebuild|recreate|remove|replace|reset|rewrite|revise|update"
)
_CONTROLLED_MUTATION_TARGETS = (
    r"extraction\s+prompt|facts?|hierarchy(?:\s+rules?)?|kg\s+facts?|ontology"
    r"(?:\s+rules?)?|prompts?|relations?(?:\s+rules?)?|rules?|web\s+display|"
    r"workspace"
)
_REPORT_NOTE_MUTATION_INTENT_PATTERN = re.compile(
    rf"\b(?:{_CONTROLLED_MUTATION_ACTIONS})\w*\b"
    rf".{{0,80}}\b(?:{_CONTROLLED_MUTATION_TARGETS})\b"
    r"|"
    rf"\b(?:{_CONTROLLED_MUTATION_TARGETS})\b"
    rf".{{0,80}}\b(?:{_CONTROLLED_MUTATION_ACTIONS})\w*\b",
    re.IGNORECASE | re.DOTALL,
)
_REPORT_NOTE_DESTRUCTIVE_ACTION_PATTERN = re.compile(
    r"\b(?:clear|delete|drop|rebuild|recreate|reset)\w*\b",
    re.IGNORECASE,
)

_REQUIRED_STRING_FIELDS = (
    "id",
    "type",
    "target",
    "proposed_change",
    "reason",
    "risk",
)
_BARE_PATHOGEN_TARGET_TERMS = (
    "病毒",
    "细菌",
    "真菌",
    "支原体",
    "衣原体",
    "病原体",
    "virus",
    "bacteria",
    "bacterium",
    "fungus",
    "pathogen",
)
_CLINICAL_CONDITION_TERMS = (
    "感染",
    "疾病",
    "病毒病",
    "炎",
    "症",
    "综合征",
    "condition",
    "disease",
    "infection",
    "syndrome",
)
_MANIFESTATION_CATEGORY_TARGET_TERMS = (
    "临床表现",
    "症状表现",
    "主要表现",
    "临床特征",
)
_DIAGNOSTIC_EVIDENCE_TERMS = (
    "检查",
    "检测",
    "检验",
    "试验",
    "结果",
    "血常规",
    "血生化",
    "血气",
    "动脉血气",
    "动脉血气分析",
    "病原学",
    "抗原",
    "核酸",
    "培养",
    "血清",
    "pcr",
    "test",
    "evidence",
)
_DISEASE_OR_CONDITION_TERMS = (
    "流行性感冒",
    "流感",
    "感冒",
    "感染",
    "肺炎",
    "疾病",
    "综合征",
    "病",
    "influenza",
    "infection",
    "pneumonia",
    "disease",
    "syndrome",
)
_BROAD_POPULATION_TERMS = (
    "儿童",
    "患儿",
    "婴儿",
    "青少年",
    "成人",
    "老年",
    "孕妇",
    "人群",
    "患者",
    "children",
    "child",
    "infant",
    "adult",
    "elderly",
    "pregnant",
    "population",
    "patient",
)
_POPULATION_QUALIFIER_KEYS = (
    "condition",
    "context",
    "indication",
    "age",
    "age_min",
    "age_max",
    "age_range",
    "route",
    "population",
    "risk",
    "reason",
)
_ELECTROLYTE_OR_LAB_ABNORMALITY_TERMS = (
    "低钾血症",
    "高钾血症",
    "低钠血症",
    "高钠血症",
    "低血钾",
    "高血钾",
    "低血钠",
    "高血钠",
    "电解质紊乱",
    "hypokalemia",
    "hyperkalemia",
    "hyponatremia",
    "hypernatremia",
    "electrolyte",
)
def validate_proposal_id(proposal_id: str) -> None:
    if not isinstance(proposal_id, str) or not PROPOSAL_ID_PATTERN.fullmatch(
        proposal_id
    ):
        raise ValueError("proposal id must match [A-Za-z0-9_.-]+")


def validate_proposal(proposal: ImprovementProposal) -> None:
    for field_name in _REQUIRED_STRING_FIELDS:
        value = getattr(proposal, field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"proposal {field_name} must be a non-empty string")

    validate_proposal_id(proposal.id)

    if not _is_canonical_type(proposal.type):
        raise ValueError("proposal type must be canonical lowercase snake_case")

    if proposal.risk not in _ALLOWED_RISKS:
        raise ValueError("proposal risk must be one of low, medium, high")

    if not isinstance(proposal.requires_approval, bool):
        raise ValueError("proposal requires_approval must be a bool")

    if not isinstance(proposal.evidence, list) or not all(
        isinstance(item, str) for item in proposal.evidence
    ):
        raise ValueError("proposal evidence must be a list of strings")

    if not isinstance(proposal.expected_metric_change, dict):
        raise ValueError("proposal expected_metric_change must be a dict")
    for key, value in proposal.expected_metric_change.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("proposal expected_metric_change keys must be non-empty strings")
        if not _METRIC_KEY_PATTERN.fullmatch(key):
            raise ValueError("proposal expected_metric_change keys must be metric identifiers")
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError("proposal expected_metric_change values must be numbers")

    if not isinstance(proposal.patch_candidate, str):
        raise ValueError("proposal patch_candidate must be a string")

    if not isinstance(proposal.action_payload, dict):
        raise ValueError("proposal action_payload must be a dict")
    if (
        proposal.type in EXECUTABLE_MEDICAL_PROPOSAL_TYPES_REQUIRING_NON_EMPTY_PAYLOAD
        and not proposal.action_payload
    ):
        raise ValueError(
            f"proposal {proposal.id} action_payload is required for "
            "executable medical proposal type"
        )

    if not isinstance(proposal.judge, dict):
        raise ValueError("proposal judge must be a dict")

    if (
        not isinstance(proposal.confidence, int | float)
        or isinstance(proposal.confidence, bool)
        or not 0 <= proposal.confidence <= 1
    ):
        raise ValueError("proposal confidence must be between 0 and 1")

    if proposal.type in MUTATION_PROPOSAL_TYPES and proposal.requires_approval is not True:
        raise ValueError(f"proposal type {proposal.type} requires approval")
    if (
        proposal.type == "quality_report_note"
        and proposal.requires_approval is not True
    ):
        _validate_no_approval_report_note(proposal)
    if (
        proposal.type not in MUTATION_PROPOSAL_TYPES
        and proposal.type not in NO_APPROVAL_PROPOSAL_TYPES
        and proposal.type not in REVIEW_ONLY_PROPOSAL_TYPES
        and proposal.requires_approval is not True
    ):
        raise ValueError(f"unknown proposal type {proposal.type} requires approval")
    if proposal.type in REVIEW_ONLY_PROPOSAL_TYPES and proposal.requires_approval is not True:
        raise ValueError(f"proposal type {proposal.type} requires approval")
    if proposal.type == "review_context_request":
        _validate_review_context_request(proposal)
    if proposal.type == "medical_relation_schema_migration":
        _validate_medical_relation_schema_migration_payload(proposal.action_payload)
    if proposal.type == "medical_fact_role_split":
        _validate_medical_fact_role_split_payload(proposal.action_payload)
    if proposal.type == "candidate_kg_expansion":
        _validate_candidate_kg_expansion_payload(proposal.action_payload)
    if proposal.type == "value_node_to_qualifier":
        _validate_value_node_to_qualifier_payload(proposal.action_payload)


def write_approval_queue(
    proposals: list[ImprovementProposal], output_dir: str | Path
) -> Path:
    valid_proposals = _validate_and_sort(proposals)
    queued = [
        proposal
        for proposal in valid_proposals
        if proposal.requires_approval and _approval_queue_accepts(proposal)
    ]
    return _write_proposals(queued, output_dir, "approval_queue.md", "Approval Queue")


def write_improvement_backlog(
    proposals: list[ImprovementProposal], output_dir: str | Path
) -> Path:
    valid_proposals = _validate_and_sort(proposals)
    return _write_proposals(
        valid_proposals,
        output_dir,
        "improvement_backlog.md",
        "Improvement Backlog",
    )


def _validate_and_sort(
    proposals: list[ImprovementProposal],
) -> list[ImprovementProposal]:
    for proposal in proposals:
        validate_proposal(proposal)
    return sorted(proposals, key=lambda proposal: proposal.id)


def _approval_queue_accepts(proposal: ImprovementProposal) -> bool:
    from .apply import proposal_apply_capability
    from .proposal_funnel import validate_approval_queue_candidate

    gate = validate_approval_queue_candidate(proposal)
    if not gate.allowed:
        return False
    if proposal.type not in MUTATION_PROPOSAL_TYPES:
        return True
    return proposal_apply_capability(proposal).supported


def _write_proposals(
    proposals: list[ImprovementProposal],
    output_dir: str | Path,
    filename: str,
    title: str,
) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    path.write_text(_render_proposals(proposals, title), encoding="utf-8")
    return path


def _render_proposals(proposals: list[ImprovementProposal], title: str) -> str:
    yaml_body = yaml.safe_dump(
        {"proposals": [_proposal_to_render_dict(proposal) for proposal in proposals]},
        allow_unicode=True,
        sort_keys=False,
    )
    return f"# {title}\n\n{yaml_body}"


def _proposal_to_render_dict(proposal: ImprovementProposal) -> dict[str, object]:
    rendered: dict[str, object] = {
        "id": proposal.id,
        "type": proposal.type,
        "target": proposal.target,
        "proposed_change": proposal.proposed_change,
        "reason": proposal.reason,
        "evidence": proposal.evidence,
        "confidence": proposal.confidence,
        "risk": proposal.risk,
        "requires_approval": proposal.requires_approval,
        "expected_metric_change": {
            key: proposal.expected_metric_change[key]
            for key in sorted(proposal.expected_metric_change)
        },
    }
    if proposal.patch_candidate:
        rendered["patch_candidate"] = proposal.patch_candidate
    if proposal.action_payload:
        rendered["action_payload"] = proposal.action_payload
    if proposal.judge:
        rendered["judge"] = proposal.judge
    return rendered


def _is_canonical_type(value: str) -> bool:
    return value == value.strip().casefold() and bool(_TYPE_PATTERN.fullmatch(value))


def _validate_medical_relation_schema_migration_payload(
    action_payload: dict[str, object],
) -> None:
    action = action_payload.get("action")
    if action == "retire_relation":
        required_string_fields = (
            "edge_id",
            "expected_source",
            "expected_target",
            "current_keywords",
            "retirement_reason",
        )
        for field_name in required_string_fields:
            value = action_payload.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "proposal action_payload must include non-empty string "
                    f"{field_name}"
                )
        return

    if action != "replace_relation":
        raise ValueError(
            "proposal action_payload action must be replace_relation or "
            "retire_relation "
            "for medical_relation_schema_migration"
        )

    required_string_fields = (
        "edge_id",
        "expected_source",
        "expected_target",
        "current_keywords",
        "new_source",
        "new_target",
        "new_keywords",
    )
    for field_name in required_string_fields:
        value = action_payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "proposal action_payload must include non-empty string "
                f"{field_name}"
            )

    if _looks_like_noop_relation_replacement(action_payload):
        raise ValueError(
            "proposal action_payload replace_relation must not be a no-op: "
            "source, target, and relation keyword are unchanged"
        )

    new_keywords = action_payload["new_keywords"]
    if new_keywords not in CANONICAL_MEDICAL_RELATION_IDS:
        raise ValueError(
            "proposal action_payload new_keywords must be a canonical relation id"
        )
    _validate_replace_relation_domain_range(action_payload, str(new_keywords))
    if new_keywords == "has_indication" and _looks_like_bare_pathogen_target(
        str(action_payload["new_target"])
    ):
        raise ValueError(
            "proposal action_payload has_indication target must be a disease "
            "or clinical condition, not a bare pathogen entity"
        )
    if new_keywords == "orders_test" and _looks_like_bare_pathogen_target(
        str(action_payload["new_source"])
    ):
        raise ValueError(
            "proposal action_payload orders_test source must be a disease, "
            "clinical condition, care process, or recommendation context, not a "
            "bare pathogen entity"
        )
    if new_keywords == "orders_test" and _looks_like_influenza_ordering_bare_lab_marker(
        str(action_payload["new_source"]),
        str(action_payload["new_target"]),
    ):
        raise ValueError(
            "proposal action_payload orders_test must not connect influenza "
            "directly to a bare lab marker; order the parent test panel or model "
            "the marker as an observed lab finding"
        )
    if new_keywords == "orders_test" and (
        _looks_like_generic_influenza_ordering_complication_imaging(
            str(action_payload["new_source"]),
            str(action_payload["new_target"]),
        )
    ):
        raise ValueError(
            "proposal action_payload orders_test must not connect generic "
            "influenza directly to complication-imaging tests such as CT or MRI; "
            "attach imaging to a specific complication or severity endpoint"
        )
    if new_keywords == "targets_disease" and _looks_like_bare_pathogen_target(
        str(action_payload["new_target"])
    ):
        raise ValueError(
            "proposal action_payload targets_disease target must be a disease "
            "or clinical condition, not a bare pathogen entity"
        )
    if (
        new_keywords == "reduces_risk_of"
        and _looks_like_broad_population(str(action_payload["expected_target"]))
        and not _has_population_scope_qualifier(action_payload.get("qualifiers"))
    ):
        raise ValueError(
            "proposal action_payload reduces_risk_of must preserve population "
            "scope qualifiers when migrating from a population or patient target"
        )
    if new_keywords == "has_manifestation" and _looks_like_manifestation_category(
        str(action_payload["new_target"])
    ):
        raise ValueError(
            "proposal action_payload has_manifestation target must be a patient "
            "symptom, sign, or clinical finding, not a clinical manifestation "
            "category label"
        )
    if new_keywords == "has_manifestation" and _looks_like_electrolyte_or_lab_abnormality(
        str(action_payload["new_target"])
    ):
        raise ValueError(
            "proposal action_payload has_manifestation target must not be an "
            "electrolyte/laboratory abnormality that needs complication or "
            "clinical finding semantics"
        )
    if new_keywords == "causative_agent" and _looks_like_bacterial_agent_for_influenza(
        str(action_payload["new_source"]),
        str(action_payload["new_target"]),
    ):
        raise ValueError(
            "proposal action_payload causative_agent must not assign bacterial "
            "secondary-infection pathogens as the cause of influenza disease"
        )
    if new_keywords == "causative_agent" and _looks_like_generic_influenza_virus_for_typed_flu(
        str(action_payload["new_source"]),
        str(action_payload["new_target"]),
    ):
        raise ValueError(
            "proposal action_payload causative_agent for typed influenza diseases "
            "must target a typed influenza virus, not generic influenza virus"
        )
    if new_keywords == "supports_or_refutes" and _looks_like_disease_to_diagnostic_evidence(
        str(action_payload["new_source"]),
        str(action_payload["new_target"]),
    ):
        raise ValueError(
            "proposal action_payload supports_or_refutes should point from "
            "diagnostic evidence or findings toward the disease being supported "
            "or refuted, not from disease to a generic test"
        )
    if new_keywords == "supports_or_refutes" and (
        _looks_like_nonspecific_evidence_directly_supporting_influenza(
            str(action_payload["new_source"]),
            str(action_payload["new_target"]),
        )
    ):
        raise ValueError(
            "proposal action_payload supports_or_refutes must not model "
            "nonspecific labs or complication-imaging findings as direct "
            "support/refutation for influenza diagnosis"
        )
    if new_keywords == "has_diagnostic_criterion" and (
        _looks_like_nonspecific_influenza_diagnostic_criterion(
            str(action_payload["new_source"]),
            str(action_payload["new_target"]),
        )
    ):
        raise ValueError(
            "proposal action_payload has_diagnostic_criterion must not model "
            "nonspecific labs or complication-imaging findings as direct "
            "influenza diagnostic criteria"
        )
    if new_keywords == "has_complication" and (
        _looks_like_chronic_underlying_condition_as_influenza_complication(
            str(action_payload["new_source"]),
            str(action_payload["new_target"]),
        )
        or _looks_like_outcome_or_severity_as_influenza_complication(
            str(action_payload["new_source"]),
            str(action_payload["new_target"]),
        )
    ):
        raise ValueError(
            "proposal action_payload has_complication must not model chronic "
            "underlying conditions, outcomes, or severity states as direct "
            "influenza complications"
        )
    if new_keywords == "is_a" and _looks_like_parent_to_subtype_is_a(
        str(action_payload["new_source"]),
        str(action_payload["new_target"]),
    ):
        raise ValueError(
            "proposal action_payload is_a direction must point from subtype/child "
            "to parent/supertype, not from parent to subtype"
        )
    if new_keywords in {"recommended_for", "contraindicated_for"} and (
        _looks_like_broad_population(str(action_payload["new_target"]))
        and not _has_population_scope_qualifier(action_payload.get("qualifiers"))
    ):
        raise ValueError(
            "proposal action_payload recommended_for/contraindicated_for broad "
            "population targets must include condition, age, route, context, "
            "or risk qualifiers"
        )


def _looks_like_noop_relation_replacement(action_payload: dict[str, object]) -> bool:
    qualifiers = action_payload.get("qualifiers")
    return (
        _same_relation_text(action_payload["expected_source"], action_payload["new_source"])
        and _same_relation_text(action_payload["expected_target"], action_payload["new_target"])
        and _same_relation_text(action_payload["current_keywords"], action_payload["new_keywords"])
        and not _has_non_empty_qualifier(qualifiers)
    )


def _same_relation_text(left: object, right: object) -> bool:
    return str(left).strip().casefold() == str(right).strip().casefold()


def _has_non_empty_qualifier(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return any(str(item).strip() for item in value.values() if item is not None)


def _looks_like_bare_pathogen_target(value: str) -> bool:
    target = value.strip().casefold()
    if not target:
        return False
    if any(term in target for term in _CLINICAL_CONDITION_TERMS):
        return False
    return any(term in target for term in _BARE_PATHOGEN_TARGET_TERMS)


def _looks_like_manifestation_category(value: str) -> bool:
    target = value.strip().casefold()
    if not target:
        return False
    return any(term.casefold() in target for term in _MANIFESTATION_CATEGORY_TARGET_TERMS)


def _looks_like_electrolyte_or_lab_abnormality(value: str) -> bool:
    target = value.strip().casefold()
    if not target:
        return False
    return target.endswith("血症") or any(
        term.casefold() in target for term in _ELECTROLYTE_OR_LAB_ABNORMALITY_TERMS
    )


def _looks_like_influenza_ordering_bare_lab_marker(source: str, target: str) -> bool:
    return influenza_rules.looks_like_bare_lab_marker_order(source, target)


def _looks_like_generic_influenza_ordering_complication_imaging(
    source: str, target: str
) -> bool:
    return influenza_rules.looks_like_complication_imaging_order(source, target)


def _looks_like_bacterial_agent_for_influenza(source: str, target: str) -> bool:
    return influenza_rules.looks_like_bacterial_agent_for_influenza(source, target)


def _looks_like_generic_influenza_virus_for_typed_flu(
    source: str, target: str
) -> bool:
    return influenza_rules.looks_like_generic_influenza_virus_for_typed_flu(
        source, target
    )


def _looks_like_disease_to_diagnostic_evidence(source: str, target: str) -> bool:
    normalized_source = source.strip().casefold()
    normalized_target = target.strip().casefold()
    if not normalized_source or not normalized_target:
        return False
    return (
        any(term.casefold() in normalized_source for term in _DISEASE_OR_CONDITION_TERMS)
        and any(term.casefold() in normalized_target for term in _DIAGNOSTIC_EVIDENCE_TERMS)
    )


def _looks_like_nonspecific_evidence_directly_supporting_influenza(
    source: str, target: str
) -> bool:
    return influenza_rules.looks_like_nonspecific_evidence_supporting_influenza(
        source, target
    )


def _looks_like_nonspecific_influenza_diagnostic_criterion(
    source: str, target: str
) -> bool:
    return influenza_rules.looks_like_nonspecific_diagnostic_criterion(
        source, target
    )


def _looks_like_chronic_underlying_condition_as_influenza_complication(
    source: str, target: str
) -> bool:
    return influenza_rules.looks_like_chronic_condition_as_influenza_complication(
        source, target
    )


def _looks_like_outcome_or_severity_as_influenza_complication(
    source: str, target: str
) -> bool:
    return influenza_rules.looks_like_outcome_or_severity_as_influenza_complication(
        source, target
    )


def _looks_like_parent_to_subtype_is_a(source: str, target: str) -> bool:
    return influenza_rules.looks_like_parent_to_subtype_is_a(source, target)


def _looks_like_broad_population(value: str) -> bool:
    target = value.strip().casefold()
    if not target:
        return False
    return any(term.casefold() in target for term in _BROAD_POPULATION_TERMS)


def _has_population_scope_qualifier(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    for key in _POPULATION_QUALIFIER_KEYS:
        qualifier_value = value.get(key)
        if qualifier_value is not None and str(qualifier_value).strip():
            return True
    return False


def _validate_candidate_kg_expansion_payload(
    action_payload: dict[str, object],
) -> None:
    for field_name in ("candidate_nodes", "candidate_edges"):
        value = action_payload.get(field_name)
        if not isinstance(value, list):
            raise ValueError(
                "candidate_kg_expansion action_payload "
                f"{field_name} must be a list"
            )

    required_string_fields = (
        "source_id",
        "file_path",
        "evidence_quote",
        "why_not_existing",
    )
    for field_name in required_string_fields:
        value = action_payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "candidate_kg_expansion action_payload must include "
                f"non-empty string {field_name}"
            )

    evidence_quote = str(action_payload["evidence_quote"]).strip()
    if len(evidence_quote) < MIN_CANDIDATE_KG_EXPANSION_QUOTE_CHARS:
        raise ValueError(
            "candidate_kg_expansion action_payload evidence_quote is too short "
            "to ground a candidate KG expansion"
        )
    _validate_candidate_expansion_edges(action_payload)


def _validate_candidate_expansion_edges(
    action_payload: dict[str, object],
) -> None:
    candidate_nodes = action_payload.get("candidate_nodes")
    candidate_edges = action_payload.get("candidate_edges")

    node_types: dict[str, str] = {}
    if isinstance(candidate_nodes, list):
        for index, raw_node in enumerate(candidate_nodes):
            if not isinstance(raw_node, dict):
                raise ValueError(f"candidate_nodes[{index}] must be an object")
            node_id = str(raw_node.get("id") or raw_node.get("label") or "").strip()
            node_type = str(raw_node.get("entity_type") or "").strip()
            if not node_id or not node_type:
                raise ValueError(
                    f"candidate_nodes[{index}] requires id and entity_type"
                )
            node_types[node_id] = node_type

    if not isinstance(candidate_edges, list):
        return

    for index, raw_edge in enumerate(candidate_edges):
        if not isinstance(raw_edge, dict):
            raise ValueError(f"candidate_edges[{index}] must be an object")

        source = str(raw_edge.get("source") or "").strip()
        target = str(raw_edge.get("target") or "").strip()
        predicate = str(
            raw_edge.get("predicate") or raw_edge.get("keywords") or ""
        ).strip()

        if predicate not in CANONICAL_MEDICAL_RELATION_IDS:
            raise ValueError(
                "CANDIDATE_EDGE_NON_CANONICAL_PREDICATE: "
                f"candidate_edges[{index}]={predicate}"
            )

        source_type = str(
            raw_edge.get("source_type") or node_types.get(source) or ""
        ).strip()
        target_type = str(
            raw_edge.get("target_type") or node_types.get(target) or ""
        ).strip()
        if not source_type or not target_type:
            raise ValueError(
                "CANDIDATE_EDGE_TYPES_REQUIRED: "
                f"candidate_edges[{index}] must provide endpoint types"
            )

        qualifiers = raw_edge.get("qualifiers", {})
        if not isinstance(qualifiers, dict):
            raise ValueError(
                f"candidate_edges[{index}].qualifiers must be an object"
            )

        errors = validate_relation_instance(
            predicate=predicate,
            source_type=source_type,
            target_type=target_type,
            qualifiers=qualifiers,
        )
        if errors:
            raise ValueError(
                "CANDIDATE_EDGE_SCHEMA_VIOLATION: "
                f"candidate_edges[{index}]: "
                + "; ".join(errors)
            )


def _validate_value_node_to_qualifier_payload(
    action_payload: dict[str, object],
) -> None:
    required_fields = (
        "value_node_id",
        "incident_edge_id",
        "expected_incident_keywords",
        "carrier_edge_id",
        "carrier_edge_source",
        "carrier_edge_target",
        "expected_carrier_keywords",
        "carrier_source_type",
        "carrier_target_type",
        "qualifier_key",
        "qualifier_value",
    )
    for field_name in required_fields:
        value = action_payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "value_node_to_qualifier requires "
                f"{field_name}"
            )

    qualifier_key = str(action_payload["qualifier_key"]).strip()
    if qualifier_key not in _ALLOWED_VALUE_QUALIFIER_KEYS:
        raise ValueError("unsupported value-node qualifier key")

    predicate = str(action_payload["expected_carrier_keywords"]).strip()
    spec = relation_spec_by_id(predicate)
    if qualifier_key not in spec.allowed_qualifiers:
        raise ValueError(f"{predicate} does not allow qualifier {qualifier_key}")

    errors = validate_relation_instance(
        predicate=predicate,
        source_type=action_payload["carrier_source_type"],
        target_type=action_payload["carrier_target_type"],
        qualifiers={qualifier_key: action_payload["qualifier_value"]},
    )
    if errors:
        raise ValueError("; ".join(errors))


def _validate_medical_fact_role_split_payload(
    action_payload: dict[str, object],
) -> None:
    action = action_payload.get("action")
    if action == "draft_split_relation":
        raise ValueError(
            "medical_fact_role_split proposals require an executable split payload; "
            "draft_split_relation is not accepted"
        )
    if action != "split_relation":
        raise ValueError(
            "medical_fact_role_split action_payload action must be split_relation"
        )

    for field_name in (
        "edge_id",
        "expected_source",
        "expected_target",
        "current_keywords",
    ):
        value = action_payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "medical_fact_role_split action_payload must include non-empty "
                f"string {field_name}"
            )

    retire_original = action_payload.get("retire_original", True)
    if not isinstance(retire_original, bool):
        raise ValueError(
            "medical_fact_role_split action_payload retire_original must be boolean"
        )

    new_edges = action_payload.get("new_edges")
    if not isinstance(new_edges, list) or not new_edges:
        raise ValueError(
            "medical_fact_role_split action_payload must include non-empty new_edges"
        )

    expected_pair = (
        str(action_payload["expected_source"]).strip(),
        str(action_payload["expected_target"]).strip(),
    )
    endpoint_pairs: set[tuple[str, str]] = set()
    semantic_edges: set[tuple[str, str, str]] = set()
    for index, raw_edge in enumerate(new_edges):
        if not isinstance(raw_edge, dict):
            raise ValueError(
                f"medical_fact_role_split new_edges[{index}] must be an object"
            )
        source = _required_split_edge_string(raw_edge, index, "source")
        target = _required_split_edge_string(raw_edge, index, "target")
        predicate = _required_split_edge_string(raw_edge, index, "predicate")
        if predicate not in CANONICAL_MEDICAL_RELATION_IDS:
            raise ValueError(
                "medical_fact_role_split new_edges predicate must be a canonical "
                "relation id"
            )
        endpoint_pair = (source, target)
        if retire_original and endpoint_pair == expected_pair:
            raise ValueError(
                "medical_fact_role_split cannot retire the original edge while "
                "creating a split edge with the same endpoints"
            )
        if endpoint_pair in endpoint_pairs:
            raise ValueError(
                "medical_fact_role_split new_edges must not reuse the same "
                "source/target pair because graph storage is not multi-edge"
            )
        endpoint_pairs.add(endpoint_pair)
        semantic_edge = (source, target, predicate)
        if semantic_edge in semantic_edges:
            raise ValueError("medical_fact_role_split new_edges contain duplicates")
        semantic_edges.add(semantic_edge)

        qualifiers = raw_edge.get("qualifiers", {})
        if qualifiers is not None and not isinstance(qualifiers, dict):
            raise ValueError(
                f"medical_fact_role_split new_edges[{index}].qualifiers must be an object"
            )
        _validate_split_edge_domain_range(raw_edge, predicate, index)


def _required_split_edge_string(
    raw_edge: dict[str, object],
    index: int,
    field_name: str,
) -> str:
    value = raw_edge.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "medical_fact_role_split "
            f"new_edges[{index}].{field_name} must be a non-empty string"
        )
    return value.strip()


def _validate_split_edge_domain_range(
    raw_edge: dict[str, object],
    predicate: str,
    index: int,
) -> None:
    source_type = raw_edge.get("source_type")
    target_type = raw_edge.get("target_type")
    if source_type in (None, "") or target_type in (None, ""):
        raise ValueError(
            "SPLIT_EDGE_TYPES_REQUIRED: "
            f"new_edges[{index}] must include source_type and target_type"
        )

    errors = validate_relation_instance(
        predicate=predicate,
        source_type=source_type,
        target_type=target_type,
        qualifiers=raw_edge.get("qualifiers")
        if isinstance(raw_edge.get("qualifiers"), dict)
        else None,
    )
    if errors:
        raise ValueError(
            "medical_fact_role_split "
            f"new_edges[{index}] violates relation schema: "
            + "; ".join(errors)
        )


def _validate_replace_relation_domain_range(
    action_payload: dict[str, object],
    predicate: str,
) -> None:
    source_type = action_payload.get("new_source_type") or action_payload.get(
        "source_type"
    )
    target_type = action_payload.get("new_target_type") or action_payload.get(
        "target_type"
    )
    if source_type in (None, "") and target_type in (None, ""):
        return

    spec = relation_spec_by_id(predicate)
    if source_type not in (None, "") and target_type not in (None, ""):
        qualifiers = action_payload.get("qualifiers")
        errors = validate_relation_instance(
            predicate=predicate,
            source_type=source_type,
            target_type=target_type,
            qualifiers=qualifiers if isinstance(qualifiers, dict) else None,
        )
        if errors:
            raise ValueError(
                "proposal action_payload replace_relation violates relation "
                "schema: "
                + "; ".join(errors)
            )
        return

    if source_type not in (None, "") and not medical_type_allowed(
        source_type, spec.domain_types
    ):
        raise ValueError(
            "proposal action_payload replace_relation source_type is outside "
            f"{predicate} domain"
        )
    if target_type not in (None, "") and not medical_type_allowed(
        target_type, spec.range_types
    ):
        raise ValueError(
            "proposal action_payload replace_relation target_type is outside "
            f"{predicate} range"
        )


def _validate_review_context_request(proposal: ImprovementProposal) -> None:
    if _looks_like_avoidable_zanamivir_review_context_request(proposal):
        raise ValueError(
            "review_context_request for 扎那米韦 grounded asthma/children edges "
            "must be replaced by an executable split proposal with age, route, "
            "dose, duration, or precaution qualifiers"
        )


def _looks_like_avoidable_zanamivir_review_context_request(
    proposal: ImprovementProposal,
) -> bool:
    text = "\n".join(
        [
            proposal.target,
            proposal.proposed_change,
            proposal.reason,
            *proposal.evidence,
        ]
    ).casefold()
    return influenza_rules.looks_like_avoidable_zanamivir_review(text)


def _validate_no_approval_report_note(proposal: ImprovementProposal) -> None:
    if proposal.target != "quality_report.md":
        raise ValueError(
            "quality_report_note without approval must target quality_report.md"
        )
    proposed_change_body = _report_note_body(proposal.proposed_change)
    if proposed_change_body is None:
        raise ValueError(
            "quality_report_note without approval must be a report-only note "
            "or requires approval"
        )

    review_text = "\n".join(
        [
            proposed_change_body,
            proposal.reason,
            *proposal.evidence,
        ]
    )
    if (
        _REPORT_NOTE_MUTATION_INTENT_PATTERN.search(review_text)
        or _REPORT_NOTE_DESTRUCTIVE_ACTION_PATTERN.search(review_text)
    ):
        raise ValueError(
            "quality_report_note implies controlled mutation and requires approval"
        )


def _report_note_body(proposed_change: str) -> str | None:
    for prefix in _REPORT_NOTE_SAFE_PREFIXES:
        if proposed_change.startswith(prefix):
            return proposed_change.removeprefix(prefix)
    return None
