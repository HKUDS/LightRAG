from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from .models import ImprovementProposal


PayloadMode = Literal["candidate_only", "grounded_expansion", "report_only"]

CANDIDATE_EDGE_REQUIRED_FIELDS = (
    "source",
    "target",
    "source_type",
    "target_type",
    "keywords",
    "source_id",
    "file_path",
)
EVIDENCE_TUPLE_FIELDS = ("source_id", "file_path", "evidence_quote")
SUBAGENT_RETRY_ERROR_CODES = (
    "EVIDENCE_MUST_BE_STRING",
    "CANDIDATE_EDGE_TYPES_REQUIRED",
    "CANDIDATE_EDGE_REQUIRED_FIELDS",
    "CANDIDATE_EDGE_SCHEMA_VIOLATION",
    "CANDIDATE_NODE_REQUIRED_FIELDS",
    "EVIDENCE_TUPLE_MUST_MATCH_ALLOWED_SPAN",
    "EVIDENCE_NOT_GROUNDED",
    "ACTION_PAYLOAD_NOT_GROUNDED",
    "EXPECTED_METRIC_CHANGE_INVALID",
    "NO_OP_REPLACE_RELATION",
    "QUALITY_REPORT_NOTE_TARGET_INVALID",
    "RELATION_SCHEMA_VIOLATION",
)


def _default_retry_contract() -> dict[str, dict[str, Any]]:
    return {
        "EVIDENCE_MUST_BE_STRING": {
            "missing_fields": (),
        },
        "CANDIDATE_EDGE_TYPES_REQUIRED": {
            "missing_fields": ("source_type", "target_type"),
        },
        "CANDIDATE_EDGE_REQUIRED_FIELDS": {
            "missing_fields": CANDIDATE_EDGE_REQUIRED_FIELDS,
        },
        "CANDIDATE_EDGE_SCHEMA_VIOLATION": {
            "missing_fields": ("source_type", "target_type", "keywords"),
        },
        "CANDIDATE_NODE_REQUIRED_FIELDS": {
            "missing_fields": ("id", "entity_type"),
        },
        "EVIDENCE_TUPLE_MUST_MATCH_ALLOWED_SPAN": {
            "missing_fields": EVIDENCE_TUPLE_FIELDS,
        },
        "EVIDENCE_NOT_GROUNDED": {
            "missing_fields": EVIDENCE_TUPLE_FIELDS,
        },
        "ACTION_PAYLOAD_NOT_GROUNDED": {
            "missing_fields": ("action_payload",),
        },
        "EXPECTED_METRIC_CHANGE_INVALID": {
            "missing_fields": ("expected_metric_change",),
        },
        "NO_OP_REPLACE_RELATION": {
            "missing_fields": (),
        },
        "QUALITY_REPORT_NOTE_TARGET_INVALID": {
            "missing_fields": ("target",),
        },
        "RELATION_SCHEMA_VIOLATION": {
            "missing_fields": ("new_source", "new_target", "new_keywords"),
        },
    }


@dataclass(frozen=True)
class SubagentRoleContract:
    role: str
    allowed_issue_families: tuple[str, ...]
    allowed_proposal_types: tuple[str, ...]
    allowed_predicates: tuple[str, ...]
    forbidden_predicates: tuple[str, ...] = ()
    payload_mode: PayloadMode = "candidate_only"
    require_action_candidate: bool = True
    max_proposals: int = 3
    instructions: tuple[str, ...] = ()
    candidate_edge_required_fields: tuple[str, ...] = CANDIDATE_EDGE_REQUIRED_FIELDS
    evidence_tuple_fields: tuple[str, ...] = EVIDENCE_TUPLE_FIELDS
    retry_error_codes: tuple[str, ...] = SUBAGENT_RETRY_ERROR_CODES
    retry_contract: dict[str, dict[str, Any]] = field(
        default_factory=_default_retry_contract
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ROLE_CONTRACTS: dict[str, SubagentRoleContract] = {
    "schema_repair": SubagentRoleContract(
        role="schema_repair",
        allowed_issue_families=("legacy_schema", "entity_cleanup", "alias_role_conflict"),
        allowed_proposal_types=(
            "prompt_edit",
            "ontology_rule_change",
            "hierarchy_rule_change",
            "add_hierarchy_branch",
            "relation_rule_change",
            "workspace_rebuild",
            "kg_fact_correction",
            "medical_relation_schema_migration",
            "value_node_to_qualifier",
            "entity_alias_merge",
            "candidate_kg_expansion",
            "quality_report_note",
            "source_evidence_repair",
            "synonym_merge_rule",
            "relation_keyword_mapping",
        ),
        allowed_predicates=(),
        payload_mode="report_only",
        require_action_candidate=False,
        instructions=("Repair generic schema issues only when the task pack is explicit.",),
    ),
    "treatment": SubagentRoleContract(
        role="treatment",
        allowed_issue_families=("treatment", "legacy_schema"),
        allowed_proposal_types=(
            "medical_relation_schema_migration",
            "medical_fact_role_split",
            "candidate_kg_expansion",
        ),
        allowed_predicates=(
            "has_indication",
            "recommends",
            "recommended_for",
            "not_recommended_for",
            "contraindicated_for",
            "precaution_for",
            "temporarily_deferred_for",
            "has_dosing_regimen",
        ),
        require_action_candidate=False,
        instructions=(
            "Keep treatment, population, dosing, and safety semantics separate.",
            "Do not create split payload fields unless task_pack.action_candidates provides them.",
        ),
    ),
    "treatment_split": SubagentRoleContract(
        role="treatment_split",
        allowed_issue_families=("multi_predicate_split",),
        allowed_proposal_types=("medical_fact_role_split",),
        allowed_predicates=(
            "has_dosing_regimen",
            "has_indication",
            "recommended_for",
            "not_recommended_for",
            "contraindicated_for",
            "precaution_for",
            "temporarily_deferred_for",
        ),
        require_action_candidate=True,
        max_proposals=2,
        instructions=(
            "Only choose existing split_relation action_candidates.",
            "Do not synthesize edge_id, expected_source, current_keywords, or new_edges.",
            "Return an empty proposals list when no executable candidate exists.",
        ),
    ),
    "prevention": SubagentRoleContract(
        role="prevention",
        allowed_issue_families=("prevention", "risk_safety"),
        allowed_proposal_types=(
            "medical_relation_schema_migration",
            "medical_fact_role_split",
            "candidate_kg_expansion",
        ),
        allowed_predicates=(
            "targets_disease",
            "reduces_risk_of",
            "recommended_for",
            "risk_factor_for",
            "high_risk_for",
            "increases_risk_of",
            "acute_exacerbation_of",
            "contraindicated_for",
            "precaution_for",
            "not_recommended_for",
            "temporarily_deferred_for",
        ),
        forbidden_predicates=("has_complication",),
        require_action_candidate=True,
        instructions=(
            "Do not force chronic disease, pregnancy, children, or older adults into complications.",
            "Population recommendations must preserve purpose and population qualifiers.",
            "Use separate predicates for contraindication, precaution, not-recommended, and temporary deferral.",
        ),
    ),
    "risk_safety": SubagentRoleContract(
        role="risk_safety",
        allowed_issue_families=("risk_safety", "prevention"),
        allowed_proposal_types=(
            "medical_relation_schema_migration",
            "candidate_kg_expansion",
        ),
        allowed_predicates=(
            "has_complication",
            "risk_factor_for",
            "high_risk_for",
            "increases_risk_of",
            "acute_exacerbation_of",
            "not_recommended_for",
            "contraindicated_for",
            "precaution_for",
            "temporarily_deferred_for",
            "may_cause_adverse_reaction",
        ),
        require_action_candidate=True,
        instructions=("Use risk and safety predicates instead of has_complication for risk states.",),
    ),
    "diagnosis": SubagentRoleContract(
        role="diagnosis",
        allowed_issue_families=("diagnosis", "legacy_schema"),
        allowed_proposal_types=(
            "medical_relation_schema_migration",
            "candidate_kg_expansion",
        ),
        allowed_predicates=(
            "has_diagnostic_criterion",
            "criterion_requires",
            "supports_or_refutes",
            "orders_test",
            "monitor_with",
        ),
        require_action_candidate=False,
        instructions=("Keep diagnostic evidence direction from evidence/finding toward disease.",),
    ),
    "clinical_modeling": SubagentRoleContract(
        role="clinical_modeling",
        allowed_issue_families=("direction", "legacy_schema"),
        allowed_proposal_types=(
            "medical_relation_schema_migration",
            "candidate_kg_expansion",
        ),
        allowed_predicates=("has_manifestation", "causative_agent", "is_a"),
        forbidden_predicates=(
            "has_complication",
            "risk_factor_for",
            "high_risk_for",
            "increases_risk_of",
            "recommended_for",
        ),
        require_action_candidate=True,
        instructions=(
            "Only handle disease-symptom direction, disease-pathogen direction, and pathogen taxonomy.",
            "Do not handle vaccine recommendations, chronic disease risk, or dosing.",
            "Do not model lab abnormalities, chronic diseases, or outcomes as ordinary symptoms.",
        ),
    ),
    "evidence_grounding": SubagentRoleContract(
        role="evidence_grounding",
        allowed_issue_families=(
            "treatment",
            "prevention",
            "risk_safety",
            "diagnosis",
            "legacy_schema",
        ),
        allowed_proposal_types=("candidate_kg_expansion",),
        allowed_predicates=(),
        payload_mode="grounded_expansion",
        require_action_candidate=False,
        max_proposals=2,
        instructions=(
            "Copy source_id, file_path, and evidence_quote from allowed_evidence_spans exactly.",
            "Return empty proposals when allowed_evidence_spans is empty.",
        ),
    ),
    "general": SubagentRoleContract(
        role="general",
        allowed_issue_families=("quality_finding", "legacy_schema"),
        allowed_proposal_types=(
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
            "candidate_kg_expansion",
            "quality_report_note",
            "review_context_request",
        ),
        allowed_predicates=(),
        payload_mode="report_only",
        require_action_candidate=False,
    ),
}


def role_contract(role: str) -> SubagentRoleContract:
    normalized = role.strip() or "schema_repair"
    try:
        return ROLE_CONTRACTS[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown subagent role: {role}") from exc


def action_payload_predicates(action_payload: Mapping[str, Any]) -> set[str]:
    predicates: set[str] = set()

    new_keywords = action_payload.get("new_keywords")
    if isinstance(new_keywords, str) and new_keywords.strip():
        predicates.add(new_keywords.strip())

    for list_field in ("new_edges", "candidate_edges"):
        raw_edges = action_payload.get(list_field)
        if not isinstance(raw_edges, list):
            continue
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, Mapping):
                continue
            predicate = raw_edge.get("predicate") or raw_edge.get("keywords")
            if isinstance(predicate, str) and predicate.strip():
                predicates.add(predicate.strip())

    return predicates


def validate_candidate_kg_expansion_contract(
    proposal: ImprovementProposal,
    contract: SubagentRoleContract,
) -> None:
    if proposal.type != "candidate_kg_expansion":
        return

    candidate_edges = proposal.action_payload.get("candidate_edges")
    if not isinstance(candidate_edges, list):
        return

    for index, raw_edge in enumerate(candidate_edges):
        if not isinstance(raw_edge, Mapping):
            continue
        missing = [
            field_name
            for field_name in contract.candidate_edge_required_fields
            if not isinstance(raw_edge.get(field_name), str)
            or not str(raw_edge.get(field_name)).strip()
        ]
        missing_type_fields = [
            field_name
            for field_name in ("source_type", "target_type")
            if field_name in missing
        ]
        if missing_type_fields:
            raise ValueError(
                "CANDIDATE_EDGE_TYPES_REQUIRED: "
                f"candidate_edges[{index}] missing_fields="
                + ",".join(missing_type_fields)
            )
        if missing:
            raise ValueError(
                "CANDIDATE_EDGE_REQUIRED_FIELDS: "
                f"candidate_edges[{index}] missing_fields="
                + ",".join(missing)
            )


def validate_proposal_role_contract(
    proposal: ImprovementProposal,
    contract: SubagentRoleContract,
) -> None:
    if proposal.type == "quality_report_note":
        return
    if proposal.type not in contract.allowed_proposal_types:
        raise ValueError(
            "ROLE_PROPOSAL_TYPE_NOT_ALLOWED: "
            f"{contract.role} cannot emit {proposal.type}"
        )

    predicates = action_payload_predicates(proposal.action_payload)

    forbidden = predicates & set(contract.forbidden_predicates)
    if forbidden:
        raise ValueError("ROLE_PREDICATE_FORBIDDEN: " + ",".join(sorted(forbidden)))

    if contract.allowed_predicates:
        unexpected = predicates - set(contract.allowed_predicates)
        if unexpected:
            raise ValueError(
                "ROLE_PREDICATE_NOT_ALLOWED: " + ",".join(sorted(unexpected))
            )
    validate_candidate_kg_expansion_contract(proposal, contract)
