from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from lightrag.constants import GRAPH_FIELD_SEP

from .deterministic_proposals.base import CandidateGenerationResult, GenerationContext
from .deterministic_proposals.registry import generate_candidates as generate_deterministic_candidates
from .issue_ledger import (
    STRUCTURED_ISSUE_SOURCES,
    DeterministicScanResult,
    normalize_raw_issues,
)
from .models import ImprovementProposal
from .profiles import influenza_rules
from .proposal_fingerprints import proposal_fingerprints
from .proposals import validate_proposal
from .subagent_contracts import role_contract, validate_proposal_role_contract

PROPOSAL_ORCHESTRATOR_ROLES = (
    "schema_repair",
    "treatment",
    "treatment_split",
    "prevention",
    "risk_safety",
    "diagnosis",
    "clinical_modeling",
    "evidence_grounding",
)
QUALITY_FINDING_FALLBACK_ROLE = "general"
REPORT_ONLY_PROPOSAL_TYPES = {
    "quality_report_note",
}
_ISSUE_FAMILY_PRIORITY = (
    "direction",
    "treatment",
    "diagnosis",
    "prevention",
    "risk_safety",
    "multi_predicate_split",
    "entity_cleanup",
    "alias_role_conflict",
    "legacy_schema",
)

_TREATMENT_PREDICATES = {
    "has_indication",
    "recommends",
    "recommended_for",
    "not_recommended_for",
    "contraindicated_for",
    "precaution_for",
    "temporarily_deferred_for",
}
_PREVENTION_PREDICATES = {
    "targets_disease",
    "reduces_risk_of",
    "recommended_for",
}
_RISK_SAFETY_PREDICATES = {
    "risk_factor_for",
    "high_risk_for",
    "increases_risk_of",
    "acute_exacerbation_of",
    "not_recommended_for",
    "contraindicated_for",
    "precaution_for",
    "temporarily_deferred_for",
    "may_cause_adverse_reaction",
}
_DIAGNOSIS_PREDICATES = {
    "has_diagnostic_criterion",
    "criterion_requires",
}
_CLINICAL_MODELING_PREDICATES = {
    "has_manifestation",
    "causative_agent",
    "is_a",
}
_EVIDENCE_PREDICATES = {
    "has_evidence",
    "supports_or_refutes",
    "evidenced_by",
}

_TREATMENT_HINTS = ("treatment", "indication", "therapy", "drug", "medication")
_DIAGNOSIS_HINTS = ("diagnos", "criterion", "test")
_CLINICAL_MODELING_HINTS = (
    "manifestation",
    "clinical",
    "symptom",
)
_EVIDENCE_HINTS = ("evidence", "source", "guideline")
_DETERMINISTIC_REPLACE_RELATION_ISSUES = {
    "reverse_clinical_manifestation": "has_manifestation",
    "reverse_causative_agent": "causative_agent",
    "reverse_pathogen_subtype_is_a": "is_a",
}

_RISK_PRIORITY = {
    "high": 0,
    "medium": 1,
    "low": 2,
}
MAX_TASK_PACK_STRING_CHARS = 240
MAX_TASK_PACK_FALLBACK_EVIDENCE_ITEMS = 10
MAX_TASK_PACK_SOURCE_FILES = 20
MAX_TASK_PACK_METADATA_KEYS = 20
MAX_TASK_PACK_METADATA_KEY_CHARS = 120


@dataclass(frozen=True)
class ProposalTaskPack:
    role: str
    issues: list[dict[str, Any]] = field(default_factory=list)
    action_candidates: list[dict[str, Any]] = field(default_factory=list)
    rejected_action_candidates: list[dict[str, Any]] = field(default_factory=list)
    covered_issue_refs: list[str] = field(default_factory=list)
    residual_issue_refs: list[str] = field(default_factory=list)
    role_contract: dict[str, Any] = field(default_factory=dict)
    allowed_evidence_spans: list[dict[str, str]] = field(default_factory=list)
    execution_mode: str = "blocked"
    block_reason: str = ""
    snapshot_context: dict[str, Any] = field(default_factory=dict)
    issue_family: str = ""
    omitted_issue_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        contract = payload.get("role_contract")
        if isinstance(contract, dict):
            payload["role_contract"] = _compact_role_contract(contract)
        return payload


@dataclass(frozen=True)
class MergedProposalSet:
    proposals: list[ImprovementProposal] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "conflicts": self.conflicts,
            "dropped": self.dropped,
        }


def _compact_role_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in contract.items()
        if key
        in {
            "role",
            "allowed_issue_families",
            "allowed_proposal_types",
            "allowed_predicates",
            "forbidden_predicates",
            "payload_mode",
            "require_action_candidate",
            "max_proposals",
            "instructions",
            "candidate_edge_required_fields",
            "retry_error_codes",
        }
    }


def build_proposal_task_packs(
    package_dir: str | Path,
    max_issues_per_pack: int = 50,
    max_packs: int = 20,
    *,
    prevalidate_action_candidates: bool = True,
    require_candidate_evidence_allowlist: bool = True,
) -> list[ProposalTaskPack]:
    if max_issues_per_pack <= 0 or max_packs <= 0:
        return []

    package_path = Path(package_dir)
    snapshot = _read_json(package_path / "snapshots" / "kg_snapshot.json")
    quality = _read_json(package_path / "snapshots" / "quality_score.json")
    issues = _collect_raw_issues(quality)
    edges_by_id = _edges_by_id(snapshot)
    nodes_by_id = _nodes_by_id(snapshot)
    snapshot_context = _snapshot_context(snapshot)

    if _has_structured_raw_issues(issues):
        enriched_issues = [
            _issue_with_snapshot_context(issue, edges_by_id, nodes_by_id)
            for issue in issues
        ]
        open_issues = [
            issue
            for issue in enriched_issues
            if not _issue_targets_previously_applied_edge(issue)
        ]
        return _structured_issue_task_packs(
            open_issues,
            edges_by_id=edges_by_id,
            nodes_by_id=nodes_by_id,
            snapshot_context=snapshot_context,
            max_issues_per_pack=max_issues_per_pack,
            max_packs=max_packs,
            prevalidate_action_candidates=prevalidate_action_candidates,
            require_candidate_evidence_allowlist=require_candidate_evidence_allowlist,
        )
    else:
        return _quality_finding_task_packs(
            issues,
            snapshot_context=snapshot_context,
            max_issues_per_pack=max_issues_per_pack,
            max_packs=max_packs,
            prevalidate_action_candidates=prevalidate_action_candidates,
            require_candidate_evidence_allowlist=require_candidate_evidence_allowlist,
        )


def build_llm_residual_task_packs(
    package_dir: str | Path,
    scan: DeterministicScanResult,
    *,
    max_issues_per_pack: int = 50,
    max_packs: int = 20,
    require_candidate_evidence_allowlist: bool = True,
) -> list[ProposalTaskPack]:
    if max_issues_per_pack <= 0 or max_packs <= 0:
        return []

    residual_refs = {
        route.issue_ref
        for route in scan.issue_routes
        if route.route_state == "llm_residual"
    }
    if not residual_refs:
        return []

    package_path = Path(package_dir)
    snapshot = _read_json(package_path / "snapshots" / "kg_snapshot.json")
    snapshot_context = _snapshot_context(snapshot)
    residual_issues = [
        issue
        for issue in scan.issues
        if _string_value(issue.get("issue_ref")) in residual_refs
    ]
    return _residual_issue_task_packs(
        residual_issues,
        snapshot_context=snapshot_context,
        max_issues_per_pack=max_issues_per_pack,
        max_packs=max_packs,
        require_candidate_evidence_allowlist=require_candidate_evidence_allowlist,
    )


def merge_subagent_proposals(
    proposal_batches: Iterable[Iterable[ImprovementProposal]],
    max_proposals: int,
) -> MergedProposalSet:
    proposals = [
        proposal for batch in proposal_batches for proposal in batch
    ]
    sorted_proposals, dropped = _drop_duplicate_proposal_ids(
        _sorted_proposals(proposals)
    )
    sorted_proposals, action_candidate_drops = (
        _prefer_action_candidates_over_target_conflicts(sorted_proposals)
    )
    dropped.extend(action_candidate_drops)
    conflicts = _proposal_conflicts(sorted_proposals)
    conflict_targets = {conflict["target"] for conflict in conflicts}

    merged: list[ImprovementProposal] = []
    seen_semantic_actions: set[str] = set()
    for proposal in sorted_proposals:
        if _proposal_conflict_target(proposal) in conflict_targets:
            continue

        semantic_key = _proposal_semantic_key(proposal)
        if semantic_key in seen_semantic_actions:
            dropped.append(_dropped_proposal(proposal, "duplicate_action_payload"))
            continue

        seen_semantic_actions.add(semantic_key)
        merged.append(proposal)

    limit = max(0, max_proposals)
    kept = merged[:limit]
    dropped.extend(
        _dropped_proposal(proposal, "max_proposals") for proposal in merged[limit:]
    )

    return MergedProposalSet(
        proposals=kept,
        conflicts=conflicts,
        dropped=dropped,
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _collect_raw_issues(quality: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = normalize_raw_issues(quality)
    if _has_structured_raw_issues(normalized):
        return normalized
    return _quality_finding_fallback_issues(quality)


def _structured_issue_task_packs(
    issues: list[dict[str, Any]],
    *,
    edges_by_id: dict[str, dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    snapshot_context: dict[str, Any],
    max_issues_per_pack: int,
    max_packs: int,
    prevalidate_action_candidates: bool,
    require_candidate_evidence_allowlist: bool,
) -> list[ProposalTaskPack]:
    if not issues:
        return []

    task_packs: list[ProposalTaskPack] = []
    for family, chunk, omitted in _stratified_issue_batches(
        issues,
        max_issues_per_pack=max_issues_per_pack,
        max_packs=max_packs,
    ):
        role = _preferred_role_for_issue_batch(chunk, issue_family=family)
        generation = _action_candidates_for_issues(
            chunk,
            issue_family=family,
            edges_by_id=edges_by_id,
            nodes_by_id=nodes_by_id,
        )
        raw_candidates = generation.candidates
        generator_rejections = generation.rejections
        if prevalidate_action_candidates:
            candidates, validation_rejections = _prevalidated_action_candidates(
                raw_candidates,
                issues=chunk,
                role=role,
            )
        else:
            candidates = [_with_candidate_id(candidate) for candidate in raw_candidates]
            validation_rejections = []
        rejected_candidates = [
            *generator_rejections,
            *validation_rejections,
        ]
        _normalize_rejected_action_candidates(rejected_candidates, issues=chunk)
        covered_issue_refs = _candidate_covered_issue_refs(candidates)
        residual_issue_refs = _residual_issue_refs(
            chunk,
            covered_issue_refs=covered_issue_refs,
        )
        contract = role_contract(role)
        allowed_evidence_spans = _allowed_evidence_spans_for_issues(chunk)
        execution_mode, block_reason = _task_pack_execution_status(
            action_candidates=candidates,
            residual_issue_refs=residual_issue_refs,
            rejected_action_candidates=rejected_candidates,
            contract=contract,
            allowed_evidence_spans=allowed_evidence_spans,
            require_candidate_evidence_allowlist=(
                require_candidate_evidence_allowlist
            ),
        )
        task_packs.append(
            ProposalTaskPack(
                role=role,
                issues=chunk,
                action_candidates=candidates,
                rejected_action_candidates=rejected_candidates,
                covered_issue_refs=covered_issue_refs,
                residual_issue_refs=residual_issue_refs,
                role_contract=contract.to_dict(),
                allowed_evidence_spans=allowed_evidence_spans,
                execution_mode=execution_mode,
                block_reason=block_reason,
                snapshot_context=snapshot_context,
                issue_family=family,
                omitted_issue_count=omitted,
            )
        )
    return task_packs


def _residual_issue_task_packs(
    issues: list[dict[str, Any]],
    *,
    snapshot_context: dict[str, Any],
    max_issues_per_pack: int,
    max_packs: int,
    require_candidate_evidence_allowlist: bool,
) -> list[ProposalTaskPack]:
    if not issues:
        return []

    task_packs: list[ProposalTaskPack] = []
    for family, chunk, omitted in _stratified_issue_batches(
        issues,
        max_issues_per_pack=max_issues_per_pack,
        max_packs=max_packs,
    ):
        role = _preferred_role_for_issue_batch(chunk, issue_family=family)
        residual_issue_refs = _residual_issue_refs(chunk, covered_issue_refs=[])
        contract = role_contract(role)
        allowed_evidence_spans = _allowed_evidence_spans_for_issues(chunk)
        execution_mode, block_reason = _task_pack_execution_status(
            action_candidates=[],
            residual_issue_refs=residual_issue_refs,
            rejected_action_candidates=[],
            contract=contract,
            allowed_evidence_spans=allowed_evidence_spans,
            require_candidate_evidence_allowlist=(
                require_candidate_evidence_allowlist
            ),
        )
        task_packs.append(
            ProposalTaskPack(
                role=role,
                issues=chunk,
                action_candidates=[],
                rejected_action_candidates=[],
                covered_issue_refs=[],
                residual_issue_refs=residual_issue_refs,
                role_contract=contract.to_dict(),
                allowed_evidence_spans=allowed_evidence_spans,
                execution_mode=execution_mode,
                block_reason=block_reason,
                snapshot_context=snapshot_context,
                issue_family=family,
                omitted_issue_count=omitted,
            )
        )
    return task_packs


def _has_structured_raw_issues(issues: list[dict[str, Any]]) -> bool:
    return any(
        _string_value(issue.get("issue_source")) in STRUCTURED_ISSUE_SOURCES
        for issue in issues
    )


def _quality_finding_fallback_issues(
    quality: dict[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for finding in _dict_items(quality.get("findings")):
        evidence, omitted_evidence_count = _bounded_string_list(
            finding.get("evidence"),
            max_items=MAX_TASK_PACK_FALLBACK_EVIDENCE_ITEMS,
            max_chars=MAX_TASK_PACK_STRING_CHARS,
        )
        issue = {
            "issue_kind": "quality_finding",
            "severity": _clip_string(
                _string_value(finding.get("severity")),
                MAX_TASK_PACK_STRING_CHARS,
            ),
            "category": _clip_string(
                _string_value(finding.get("category")),
                MAX_TASK_PACK_STRING_CHARS,
            ),
            "message": _clip_string(
                _string_value(finding.get("message")),
                MAX_TASK_PACK_STRING_CHARS,
            ),
            "evidence": evidence,
            "suggested_action": (
                _clip_string(
                    _string_value(finding.get("suggested_fix_type")),
                    MAX_TASK_PACK_STRING_CHARS,
                )
                or "review"
            ),
            "requires_approval": bool(finding.get("requires_approval")),
        }
        if omitted_evidence_count:
            issue["evidence_omitted_count"] = omitted_evidence_count
        issues.append(issue)
    return issues


def _quality_finding_task_packs(
    issues: list[dict[str, Any]],
    *,
    snapshot_context: dict[str, Any],
    max_issues_per_pack: int,
    max_packs: int,
    prevalidate_action_candidates: bool,
    require_candidate_evidence_allowlist: bool,
) -> list[ProposalTaskPack]:
    packs: list[ProposalTaskPack] = []
    for start in range(0, len(issues), max_issues_per_pack):
        if len(packs) >= max_packs:
            break
        chunk = issues[start : start + max_issues_per_pack]
        role = QUALITY_FINDING_FALLBACK_ROLE
        generation = _action_candidates_for_issues(chunk)
        raw_candidates = generation.candidates
        generator_rejections = generation.rejections
        if prevalidate_action_candidates:
            candidates, validation_rejections = _prevalidated_action_candidates(
                raw_candidates,
                issues=chunk,
                role=role,
            )
        else:
            candidates = [_with_candidate_id(candidate) for candidate in raw_candidates]
            validation_rejections = []
        rejected_candidates = [
            *generator_rejections,
            *validation_rejections,
        ]
        _normalize_rejected_action_candidates(rejected_candidates, issues=chunk)
        covered_issue_refs = _candidate_covered_issue_refs(candidates)
        residual_issue_refs = _residual_issue_refs(
            chunk,
            covered_issue_refs=covered_issue_refs,
        )
        contract = role_contract(role)
        allowed_evidence_spans = _allowed_evidence_spans_for_issues(chunk)
        execution_mode, block_reason = _task_pack_execution_status(
            action_candidates=candidates,
            residual_issue_refs=residual_issue_refs,
            rejected_action_candidates=rejected_candidates,
            contract=contract,
            allowed_evidence_spans=allowed_evidence_spans,
            require_candidate_evidence_allowlist=require_candidate_evidence_allowlist,
        )
        packs.append(
            ProposalTaskPack(
                role=role,
                issues=chunk,
                action_candidates=candidates,
                rejected_action_candidates=rejected_candidates,
                covered_issue_refs=covered_issue_refs,
                residual_issue_refs=residual_issue_refs,
                role_contract=contract.to_dict(),
                allowed_evidence_spans=allowed_evidence_spans,
                execution_mode=execution_mode,
                block_reason=block_reason,
                snapshot_context=snapshot_context,
                issue_family="quality_finding",
                omitted_issue_count=max(0, len(issues) - start - len(chunk)),
            )
        )
    return packs


def _edges_by_id(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    edges: dict[str, dict[str, Any]] = {}
    for edge in _dict_items(snapshot.get("edges")):
        edge_id = edge.get("id")
        if isinstance(edge_id, str) and edge_id:
            edges[edge_id] = edge
    return edges


def _nodes_by_id(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for node in _dict_items(snapshot.get("nodes")):
        node_id = node.get("id")
        if isinstance(node_id, str) and node_id:
            nodes[node_id] = node
    return nodes


def _snapshot_context(snapshot: dict[str, Any]) -> dict[str, Any]:
    source_files, omitted_source_file_count = _bounded_string_list(
        snapshot.get("source_files"),
        max_items=MAX_TASK_PACK_SOURCE_FILES,
        max_chars=MAX_TASK_PACK_STRING_CHARS,
    )
    metadata, omitted_metadata_key_count = _bounded_metadata(
        snapshot.get("metadata", {})
    )
    context = {
        "workspace": _clip_string(
            _string_value(snapshot.get("workspace")),
            MAX_TASK_PACK_STRING_CHARS,
        ),
        "generated_at": _clip_string(
            _string_value(snapshot.get("generated_at")),
            MAX_TASK_PACK_STRING_CHARS,
        ),
        "source_files": source_files,
        "node_count": len(_dict_items(snapshot.get("nodes"))),
        "edge_count": len(_dict_items(snapshot.get("edges"))),
        "metadata": metadata,
    }
    if omitted_source_file_count:
        context["source_files_omitted_count"] = omitted_source_file_count
    if omitted_metadata_key_count:
        context["metadata_omitted_key_count"] = omitted_metadata_key_count
    return context


def _bounded_string_list(
    value: Any,
    *,
    max_items: int,
    max_chars: int,
) -> tuple[list[str], int]:
    if not isinstance(value, list):
        return [], 0
    bounded = [_clip_string(str(item), max_chars) for item in value[:max_items]]
    return bounded, max(0, len(value) - max_items)


def _bounded_metadata(value: Any) -> tuple[dict[str, Any], int]:
    if not isinstance(value, dict):
        return {}, 0
    items = list(value.items())
    bounded: dict[str, Any] = {}
    for key, metadata_value in items[:MAX_TASK_PACK_METADATA_KEYS]:
        bounded_key = _clip_string(str(key), MAX_TASK_PACK_METADATA_KEY_CHARS)
        if not bounded_key:
            continue
        bounded[bounded_key] = _bounded_metadata_value(metadata_value)
    return bounded, max(0, len(items) - MAX_TASK_PACK_METADATA_KEYS)


def _bounded_metadata_value(value: Any) -> Any:
    if isinstance(value, str):
        return _clip_string(value, MAX_TASK_PACK_STRING_CHARS)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, dict):
        return {"type": "dict", "key_count": len(value)}
    if isinstance(value, list):
        return {"type": "list", "item_count": len(value)}
    return _clip_string(str(value), MAX_TASK_PACK_STRING_CHARS)


def _issue_with_snapshot_context(
    issue: dict[str, Any],
    edges_by_id: dict[str, dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    enriched = dict(issue)
    edge_id = _string_value(enriched.get("edge_id"))
    edge = edges_by_id.get(edge_id, {})
    for key in (
        "source",
        "target",
        "keywords",
        "source_id",
        "file_path",
        "normalized_by",
        "accepted_proposal_ids",
    ):
        if not _string_value(enriched.get(key)) and _string_value(edge.get(key)):
            enriched[key] = edge[key]
    qualifiers = enriched.get("qualifiers")
    if not isinstance(qualifiers, dict) or not qualifiers:
        qualifiers = _edge_qualifiers_from_snapshot(edge)
        if qualifiers:
            enriched["qualifiers"] = qualifiers
    source_node = nodes_by_id.get(_string_value(enriched.get("source")), {})
    target_node = nodes_by_id.get(_string_value(enriched.get("target")), {})
    if not _string_value(enriched.get("source_type")):
        source_type = _string_value(source_node.get("entity_type"))
        if source_type:
            enriched["source_type"] = source_type
    if not _string_value(enriched.get("target_type")):
        target_type = _string_value(target_node.get("entity_type"))
        if target_type:
            enriched["target_type"] = target_type
    return enriched


def _edge_qualifiers_from_snapshot(edge: dict[str, Any]) -> dict[str, Any]:
    properties = edge.get("properties")
    if not isinstance(properties, dict):
        properties = {}

    value = properties.get("qualifiers", edge.get("qualifiers"))
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _issue_targets_previously_applied_edge(issue: dict[str, Any]) -> bool:
    return (
        _string_value(issue.get("normalized_by")) == "kb_iteration_apply"
        or bool(_string_value(issue.get("accepted_proposal_ids")))
    )


def _roles_for_issue(issue: dict[str, Any]) -> list[str]:
    roles = ["schema_repair"]
    predicates = {
        str(predicate).casefold()
        for predicate in issue.get("candidate_predicates", [])
        if isinstance(predicate, str)
    }
    searchable_text = " ".join(
        _string_value(issue.get(key)).casefold()
        for key in (
            "issue_kind",
            "keywords",
            "medical_subcase",
            "guidance",
            "suggested_action",
        )
    )

    if predicates & _TREATMENT_PREDICATES or _contains_hint(
        searchable_text, _TREATMENT_HINTS
    ):
        roles.append("treatment")
    if predicates & _PREVENTION_PREDICATES or _contains_hint(
        searchable_text, ("prevention", "vaccine", "population")
    ):
        roles.append("prevention")
    if predicates & _RISK_SAFETY_PREDICATES or _contains_hint(
        searchable_text, ("risk", "contraindicat", "precaution", "defer", "safety")
    ):
        roles.append("risk_safety")
    if predicates & _DIAGNOSIS_PREDICATES or _contains_hint(
        searchable_text, _DIAGNOSIS_HINTS
    ):
        roles.append("diagnosis")
    if predicates & _CLINICAL_MODELING_PREDICATES or _contains_hint(
        searchable_text, _CLINICAL_MODELING_HINTS
    ):
        roles.append("clinical_modeling")
    if predicates & _EVIDENCE_PREDICATES or _contains_hint(
        searchable_text, _EVIDENCE_HINTS
    ):
        roles.append("evidence_grounding")
    return roles


_ROLE_PRIORITY = (
    "treatment_split",
    "clinical_modeling",
    "risk_safety",
    "prevention",
    "diagnosis",
    "treatment",
    "evidence_grounding",
    "schema_repair",
)

_ROLE_BY_FAMILY = {
    "general": "general",
    "direction": "clinical_modeling",
    "diagnosis": "diagnosis",
    "treatment": "treatment",
    "multi_predicate_split": "treatment_split",
    "risk_safety": "risk_safety",
    "prevention": "prevention",
    "entity_cleanup": "schema_repair",
    "alias_role_conflict": "schema_repair",
}


def _preferred_role_for_issue_batch(
    issues: list[dict[str, Any]],
    *,
    issue_family: str = "",
) -> str:
    if not issues:
        return "schema_repair"
    family_role = _ROLE_BY_FAMILY.get(issue_family)
    if family_role:
        return family_role
    if all(
        _string_value(issue.get("suggested_action"))
        in {"candidate_kg_expansion", "needs_source_evidence"}
        for issue in issues
    ):
        return "evidence_grounding"

    scores: dict[str, int] = {}
    for issue in issues:
        for role in _roles_for_issue(issue):
            if role == "schema_repair":
                continue
            scores[role] = scores.get(role, 0) + 1
    if not scores:
        return "schema_repair"

    return min(
        scores,
        key=lambda role: (
            -scores[role],
            _ROLE_PRIORITY.index(role)
            if role in _ROLE_PRIORITY
            else len(_ROLE_PRIORITY),
        ),
    )


def _issue_family(issue: dict[str, Any]) -> str:
    explicit_family = _string_value(issue.get("issue_family")).strip()
    if explicit_family:
        return explicit_family
    kind = _string_value(issue.get("issue_kind")).casefold()
    predicates = " ".join(
        str(predicate).casefold()
        for predicate in issue.get("candidate_predicates", [])
        if isinstance(predicate, str)
    )
    source_type = _string_value(issue.get("source_type")).casefold()
    text = f"{kind} {predicates} {_string_value(issue.get('keywords')).casefold()}"
    if kind.startswith("reverse_"):
        return "direction"
    if "multi_predicate" in text or "split" in text:
        return "multi_predicate_split"
    if "recommended_for" in predicates:
        if source_type in {"drug", "drugingredient", "treatment", "procedure"}:
            return "treatment"
        if source_type in {"vaccine", "publichealthmeasure", "public_health_measure"}:
            return "prevention"
    if any(
        token in text
        for token in ("treatment", "indication", "recommends", "dosing", "drug")
    ):
        return "treatment"
    if any(token in text for token in ("diagnostic", "criterion", "test", "evidence")):
        return "diagnosis"
    if any(
        token in text
        for token in ("prevention", "vaccine", "recommended_for", "population")
    ):
        return "prevention"
    if any(token in text for token in ("risk", "complication", "adverse")):
        return "risk_safety"
    if any(token in text for token in ("unknown", "entity_cleanup", "value_node")):
        return "entity_cleanup"
    if any(token in text for token in ("alias", "role_conflict")):
        return "alias_role_conflict"
    if "direction" in text:
        return "direction"
    return "legacy_schema"


def _stratified_issue_batches(
    issues: list[dict[str, Any]],
    *,
    max_issues_per_pack: int,
    max_packs: int,
) -> list[tuple[str, list[dict[str, Any]], int]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for issue in issues:
        by_family.setdefault(_issue_family(issue), []).append(issue)

    batches: list[tuple[str, list[dict[str, Any]], int]] = []
    family_positions = {family: 0 for family in by_family}
    priority_set = set(_ISSUE_FAMILY_PRIORITY)
    ordered_families = [
        *_ISSUE_FAMILY_PRIORITY,
        *(family for family in by_family if family not in priority_set),
    ]
    while len(batches) < max_packs:
        added = False
        for family in ordered_families:
            family_issues = by_family.get(family, [])
            start = family_positions.get(family, 0)
            if start >= len(family_issues):
                continue
            chunk = family_issues[start : start + max_issues_per_pack]
            family_positions[family] = start + len(chunk)
            omitted = max(0, len(family_issues) - family_positions[family])
            batches.append((family, chunk, omitted))
            added = True
            if len(batches) >= max_packs:
                break
        if not added:
            break
    return batches


def _contains_hint(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint.casefold() in text for hint in hints)


def _chunks(
    issues: list[dict[str, Any]], max_issues_per_pack: int
) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(issues), max_issues_per_pack):
        yield issues[index : index + max_issues_per_pack]


def _action_candidates_for_issues(
    issues: list[dict[str, Any]],
    *,
    issue_family: str = "",
    edges_by_id: dict[str, dict[str, Any]] | None = None,
    nodes_by_id: dict[str, dict[str, Any]] | None = None,
) -> CandidateGenerationResult:
    generation = generate_deterministic_candidates(
        GenerationContext(
            issue_family=issue_family or (_issue_family(issues[0]) if issues else ""),
            issues=issues,
            builders=(
            _typed_influenza_pathogen_expansion_candidate,
            _severe_sign_diagnostic_criterion_action_candidate,
            _disease_complication_action_candidate,
            _replace_relation_action_candidate,
            _diagnostic_test_order_action_candidate,
            _multi_predicate_split_action_candidate,
            ),
            edges_by_id=edges_by_id or {},
            nodes_by_id=nodes_by_id or {},
            all_edges=list((edges_by_id or {}).values()),
        )
    )
    return CandidateGenerationResult(
        candidates=[_with_candidate_id(candidate) for candidate in generation.candidates],
        rejections=generation.rejections,
    )


def _candidate_covered_issue_refs(candidates: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            _string_value(candidate.get("issue_ref"))
            for candidate in candidates
            if _string_value(candidate.get("issue_ref"))
        }
    )


def _residual_issue_refs(
    issues: list[dict[str, Any]],
    *,
    covered_issue_refs: list[str],
) -> list[str]:
    covered = set(covered_issue_refs)
    residual: list[str] = []
    for issue in issues:
        issue_ref = _string_value(issue.get("issue_ref"))
        if issue_ref and issue_ref not in covered:
            residual.append(issue_ref)
    return residual


def _prevalidated_action_candidates(
    candidates: list[dict[str, Any]],
    *,
    issues: list[dict[str, Any]],
    role: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contract = role_contract(role)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_with_id = _with_candidate_id(candidate)
        try:
            proposal = _candidate_validation_proposal(
                candidate_with_id,
                issues=issues,
            )
            validate_proposal(proposal)
            validate_proposal_role_contract(proposal, contract)
        except ValueError as exc:
            rejected.append(
                {
                    "candidate_id": candidate_with_id["candidate_id"],
                    "issue_ref": _string_value(candidate_with_id.get("issue_ref")),
                    "issue_family": _issue_family(
                        _issue_for_candidate(candidate_with_id, issues)
                    ),
                    "stage": "deterministic_candidate",
                    "error_code": "ACTION_CANDIDATE_INVALID",
                    "error": str(exc),
                    "candidate": candidate_with_id,
                }
            )
            continue
        accepted.append(candidate_with_id)
    return accepted, rejected


def _normalize_rejected_action_candidates(
    rejections: list[dict[str, Any]],
    *,
    issues: list[dict[str, Any]],
) -> None:
    issues_by_ref = {
        _string_value(issue.get("issue_ref")): issue
        for issue in issues
        if _string_value(issue.get("issue_ref"))
    }
    fallback_issue = issues[0] if len(issues) == 1 else {}
    for rejection in rejections:
        candidate = rejection.get("candidate")
        if not isinstance(candidate, dict):
            candidate = {}
        issue_ref = _string_value(rejection.get("issue_ref")) or _string_value(
            candidate.get("issue_ref")
        )
        issue = issues_by_ref.get(issue_ref, fallback_issue)
        error_code = (
            _string_value(rejection.get("error_code"))
            or "UNKNOWN_CANDIDATE_REJECTION"
        )
        stage = _string_value(rejection.get("stage")) or "deterministic_candidate"
        candidate_id = _string_value(rejection.get("candidate_id")) or _string_value(
            candidate.get("candidate_id")
        )
        rejection["issue_ref"] = issue_ref or _string_value(issue.get("issue_ref"))
        rejection["issue_family"] = _rejection_issue_family(
            rejection,
            candidate=candidate,
            issue=issue,
        )
        rejection["stage"] = stage
        rejection["error_code"] = error_code
        rejection["candidate_id"] = candidate_id or _synthetic_rejection_candidate_id(
            issue_ref=rejection["issue_ref"],
            issue_family=rejection["issue_family"],
            stage=stage,
            error_code=error_code,
        )
        rejection["error"] = _string_value(rejection.get("error")) or rejection[
            "error_code"
        ]


def _rejection_issue_family(
    rejection: dict[str, Any],
    *,
    candidate: dict[str, Any],
    issue: dict[str, Any],
) -> str:
    for value in (
        rejection.get("issue_family"),
        candidate.get("issue_family"),
        issue.get("issue_family"),
    ):
        family = _string_value(value)
        if family:
            return family
    return _issue_family(issue) if issue else "legacy_schema"


def _synthetic_rejection_candidate_id(
    *,
    issue_ref: str,
    issue_family: str,
    stage: str,
    error_code: str,
) -> str:
    basis = f"{issue_ref}|{issue_family}|{stage}|{error_code}"
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]
    return f"rejection-{_safe_rejection_token(error_code)}-{digest}"


def _safe_rejection_token(value: str) -> str:
    token = "".join(
        char.casefold() if char.isalnum() else "-"
        for char in value.strip()
    ).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token[:48] or "candidate"


def _candidate_validation_proposal(
    candidate: dict[str, Any],
    *,
    issues: list[dict[str, Any]],
) -> ImprovementProposal:
    proposal_type = _string_value(candidate.get("proposal_type"))
    target = _string_value(candidate.get("target"))
    action_payload = candidate.get("action_payload")
    issue = _issue_for_candidate(candidate, issues)
    issue_kind = _string_value(candidate.get("issue_kind") or issue.get("issue_kind"))
    edge_id = ""
    if isinstance(action_payload, dict):
        edge_id = _string_value(action_payload.get("edge_id"))
    return ImprovementProposal(
        id=f"prop-{candidate['candidate_id']}",
        type=proposal_type,
        target=target,
        proposed_change=f"Validate deterministic candidate for {target}.",
        reason=f"Candidate generated from {issue_kind or 'quality issue'}.",
        evidence=_candidate_validation_evidence(issue, edge_id),
        confidence=0.9,
        risk=_string_value(issue.get("risk")) or "medium",
        requires_approval=True,
        expected_metric_change={},
        action_payload=dict(action_payload) if isinstance(action_payload, dict) else {},
    )


def _issue_for_candidate(
    candidate: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    issue_ref = _string_value(candidate.get("issue_ref"))
    if issue_ref:
        for issue in issues:
            if _string_value(issue.get("issue_ref")) == issue_ref:
                return issue
    action_payload = candidate.get("action_payload")
    edge_id = ""
    if isinstance(action_payload, dict):
        edge_id = _string_value(action_payload.get("edge_id"))
    issue_kind = _string_value(candidate.get("issue_kind"))
    for issue in issues:
        if edge_id and _string_value(issue.get("edge_id")) != edge_id:
            continue
        if issue_kind and _string_value(issue.get("issue_kind")) != issue_kind:
            continue
        return issue
    for issue in issues:
        if edge_id and _string_value(issue.get("edge_id")) == edge_id:
            return issue
    return issues[0] if issues else {}


def _candidate_validation_evidence(issue: dict[str, Any], edge_id: str) -> list[str]:
    source_id = _first_graph_field(issue.get("source_id"))
    file_path = _first_graph_field(issue.get("file_path"))
    if source_id and file_path and edge_id:
        return [f"source_id: {source_id}; file_path: {file_path}; relation_id: {edge_id}"]
    if source_id and file_path:
        return [f"source_id: {source_id}; file_path: {file_path}"]
    return []


def _action_candidate_id(candidate: dict[str, Any]) -> str:
    stable = json.dumps(
        {
            "proposal_type": candidate.get("proposal_type"),
            "target": candidate.get("target"),
            "action_payload": candidate.get("action_payload"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:12]
    return f"ac-{digest}"


def _with_candidate_id(candidate: dict[str, Any]) -> dict[str, Any]:
    if _string_value(candidate.get("candidate_id")):
        return dict(candidate)
    return {
        "candidate_id": _action_candidate_id(candidate),
        **candidate,
    }


def _allowed_evidence_spans_for_issues(
    issues: list[dict[str, Any]],
) -> list[dict[str, str]]:
    spans: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in issues:
        source_id = _first_graph_field(issue.get("source_id"))
        file_path = _first_graph_field(issue.get("file_path"))
        evidence_quote = _evidence_quote_from_issue(issue)
        if not source_id or not file_path or not evidence_quote:
            continue
        key = (source_id, file_path, evidence_quote)
        if key in seen:
            continue
        seen.add(key)
        spans.append(
            {
                "source_id": source_id,
                "file_path": file_path,
                "evidence_quote": evidence_quote,
            }
        )
    return spans


def _evidence_quote_from_issue(issue: dict[str, Any]) -> str:
    for key in ("evidence_quote", "quote", "source_quote", "text_span"):
        value = _string_value(issue.get(key))
        if value:
            return value
    evidence_items = issue.get("evidence")
    if isinstance(evidence_items, list):
        for item in evidence_items:
            value = _string_value(item)
            if value and ":" not in value:
                return value
    return ""


def _task_pack_execution_status(
    *,
    action_candidates: list[dict[str, Any]],
    residual_issue_refs: list[str],
    rejected_action_candidates: list[dict[str, Any]],
    contract: Any,
    allowed_evidence_spans: list[dict[str, str]],
    require_candidate_evidence_allowlist: bool,
) -> tuple[str, str]:
    if action_candidates and not residual_issue_refs:
        return "deterministic_only", ""
    if action_candidates and residual_issue_refs:
        return "hybrid", ""
    if contract.payload_mode == "grounded_expansion":
        if require_candidate_evidence_allowlist and not allowed_evidence_spans:
            return "blocked", "no grounded evidence allowlist"
        return "grounded_expansion", ""
    if contract.require_action_candidate:
        if rejected_action_candidates:
            return "blocked", "all deterministic action candidates were rejected"
        return "blocked", "no executable action candidate"
    return "llm_assisted", ""


def _replace_relation_action_candidate(
    issue: dict[str, Any],
) -> dict[str, Any] | None:
    issue_kind = _string_value(issue.get("issue_kind"))
    new_keywords = _DETERMINISTIC_REPLACE_RELATION_ISSUES.get(issue_kind)
    if new_keywords is None:
        return None
    if _is_generic_to_typed_influenza_causative_issue(issue):
        return None

    edge_id = _string_value(issue.get("edge_id"))
    source = _string_value(issue.get("source"))
    target = _string_value(issue.get("target"))
    keywords = _string_value(issue.get("keywords"))
    new_source = _string_value(issue.get("new_source")) or target
    new_target = _string_value(issue.get("new_target")) or source
    source_type = _string_value(issue.get("source_type"))
    target_type = _string_value(issue.get("target_type"))
    new_source_type = source_type if new_source == source else target_type
    new_target_type = target_type if new_target == target else source_type
    if (
        not edge_id
        or not source
        or not target
        or not keywords
        or not new_source
        or not new_target
    ):
        return None

    action_payload = {
        "action": "replace_relation",
        "edge_id": edge_id,
        "expected_source": source,
        "expected_target": target,
        "current_keywords": keywords,
        "new_source": new_source,
        "new_target": new_target,
        "new_keywords": new_keywords,
        "qualifiers": _qualifiers(issue.get("qualifiers")),
    }
    if new_source_type:
        action_payload["new_source_type"] = new_source_type
    if new_target_type:
        action_payload["new_target_type"] = new_target_type

    return {
        "proposal_type": "medical_relation_schema_migration",
        "target": f"edge:{edge_id}",
        "issue_kind": issue_kind,
        "action_payload": action_payload,
    }


def _typed_influenza_pathogen_expansion_candidate(
    issue: dict[str, Any],
) -> dict[str, Any] | None:
    if not _is_generic_to_typed_influenza_causative_issue(issue):
        return None

    edge_id = _string_value(issue.get("edge_id"))
    source = _string_value(issue.get("source"))
    disease = _string_value(issue.get("target"))
    keywords = _string_value(issue.get("keywords"))
    source_id = _first_graph_field(issue.get("source_id"))
    file_path = _first_graph_field(issue.get("file_path"))
    pathogen = _typed_influenza_pathogen_for_disease(disease)
    if (
        not edge_id
        or not source
        or not disease
        or not keywords
        or not source_id
        or not file_path
        or not pathogen
    ):
        return None

    return {
        "proposal_type": "candidate_kg_expansion",
        "target": f"kg:candidate:typed-influenza-pathogen:{disease}",
        "issue_kind": "typed_influenza_pathogen_expansion",
        "action_payload": {
            "edge_id": edge_id,
            "candidate_nodes": [
                {
                    "id": pathogen,
                    "label": pathogen,
                    "entity_type": "pathogen",
                    "description": f"{disease}的精确病原体，属于{source}。",
                }
            ],
            "candidate_edges": [
                {
                    "source": pathogen,
                    "source_type": "Pathogen",
                    "target": source,
                    "target_type": "Pathogen",
                    "keywords": "is_a",
                    "source_id": source_id,
                    "file_path": file_path,
                    "qualifiers": {},
                    "description": f"{pathogen}是{source}的一个分型病原体。",
                },
                {
                    "source": disease,
                    "source_type": "Disease",
                    "target": pathogen,
                    "target_type": "Pathogen",
                    "keywords": "causative_agent",
                    "source_id": source_id,
                    "file_path": file_path,
                    "qualifiers": {},
                    "description": f"{disease}应指向精确的{pathogen}病原体。",
                },
            ],
            "retire_edges": [
                {
                    "source": source,
                    "target": disease,
                    "keywords": keywords,
                    "reason": (
                        "用精确病原体节点和疾病 causative_agent 关系替代"
                        "泛化分型边。"
                    ),
                }
            ],
            "source_id": source_id,
            "file_path": file_path,
            "evidence_quote": _edge_json_evidence_quote(edge_id, keywords),
            "why_not_existing": (
                f"{disease}需要精确病原体节点，不能直接指向泛化{source}。"
            ),
        },
    }


def _is_generic_to_typed_influenza_causative_issue(issue: dict[str, Any]) -> bool:
    issue_kind = _string_value(issue.get("issue_kind"))
    if issue_kind not in {
        "reverse_causative_agent",
        "legacy_overloaded_relation",
    }:
        return False
    source = _string_value(issue.get("source")).casefold()
    target = _string_value(issue.get("target"))
    if not influenza_rules.is_generic_influenza_virus(source):
        return False
    if issue_kind == "legacy_overloaded_relation":
        keywords = _string_value(issue.get("keywords"))
        predicates = {
            str(predicate).strip()
            for predicate in issue.get("candidate_predicates", [])
            if isinstance(predicate, str)
        }
        if "病原分型" not in keywords or not (
            {"is_a", "causative_agent"} <= predicates
        ):
            return False
    return _typed_influenza_pathogen_for_disease(target) != ""


def _typed_influenza_pathogen_for_disease(disease: str) -> str:
    return influenza_rules.typed_pathogen_for_disease(disease)


def _edge_json_evidence_quote(edge_id: str, keywords: str) -> str:
    return f'"id": "{edge_id}", "keywords": "{keywords}"'


def _severe_sign_diagnostic_criterion_action_candidate(
    issue: dict[str, Any],
) -> dict[str, Any] | None:
    if issue.get("issue_kind") != "legacy_overloaded_relation":
        return None
    if _string_value(issue.get("medical_subcase")) != "diagnostic_evidence_flattening":
        return None

    candidate_predicates = set(
        _ordered_non_empty_strings(issue.get("candidate_predicates", []))
    )
    if "has_diagnostic_criterion" not in candidate_predicates:
        return None

    edge_id = _string_value(issue.get("edge_id"))
    source = _string_value(issue.get("source"))
    target = _string_value(issue.get("target"))
    keywords = _string_value(issue.get("keywords"))
    if not edge_id or not source or not target or not keywords:
        return None
    if "诊断依据" not in keywords:
        return None
    if not _looks_like_severe_influenza_context(target):
        return None
    if _looks_like_manifestation_category(source) or _looks_like_bare_lab_marker(source):
        return None

    return {
        "proposal_type": "medical_relation_schema_migration",
        "target": f"edge:{edge_id}",
        "issue_kind": "diagnostic_evidence_flattening",
        "action_payload": {
            "action": "replace_relation",
            "edge_id": edge_id,
            "expected_source": source,
            "expected_target": target,
            "current_keywords": keywords,
            "new_source": target,
            "new_target": source,
            "new_keywords": "has_diagnostic_criterion",
            "qualifiers": {"context": "severe_influenza"},
        },
    }


def _looks_like_severe_influenza_context(value: str) -> bool:
    return influenza_rules.looks_like_severe_influenza_context(value)


def _looks_like_manifestation_category(value: str) -> bool:
    normalized = value.strip().casefold()
    return any(term in normalized for term in ("临床表现", "症状表现", "临床特征"))


def _looks_like_bare_lab_marker(value: str) -> bool:
    return influenza_rules.looks_like_bare_lab_marker(value)


def _disease_complication_action_candidate(
    issue: dict[str, Any],
) -> dict[str, Any] | None:
    if issue.get("issue_kind") != "legacy_overloaded_relation":
        return None
    candidate_predicates = set(
        _ordered_non_empty_strings(issue.get("candidate_predicates", []))
    )
    if "has_complication" not in candidate_predicates:
        return None

    edge_id = _string_value(issue.get("edge_id"))
    source = _string_value(issue.get("source"))
    target = _string_value(issue.get("target"))
    keywords = _string_value(issue.get("keywords"))
    source_type = _string_value(issue.get("source_type"))
    target_type = _string_value(issue.get("target_type"))
    if not edge_id or not source or not target or not keywords:
        return None

    if _is_disease_type(source_type) and _is_complication_type(target_type):
        new_source = source
        new_target = target
    elif _is_complication_type(source_type) and _is_disease_type(target_type):
        new_source = target
        new_target = source
    else:
        return None
    if not _looks_like_influenza_context(new_source):
        return None
    if _looks_like_non_complication_outcome_or_severity(new_target):
        return None

    return {
        "proposal_type": "medical_relation_schema_migration",
        "target": f"edge:{edge_id}",
        "issue_kind": "legacy_overloaded_relation",
        "action_payload": {
            "action": "replace_relation",
            "edge_id": edge_id,
            "expected_source": source,
            "expected_target": target,
            "current_keywords": keywords,
            "new_source": new_source,
            "new_target": new_target,
            "new_keywords": "has_complication",
            "qualifiers": _qualifiers(issue.get("qualifiers")),
        },
    }


def _is_disease_type(value: str) -> bool:
    return value.strip().casefold() in {
        "disease",
        "clinicalcondition",
        "clinical_condition",
    }


def _is_complication_type(value: str) -> bool:
    return value.strip().casefold() in {
        "complication",
        "adverseoutcome",
        "adverse_outcome",
    }


def _looks_like_influenza_context(value: str) -> bool:
    return influenza_rules.looks_like_influenza_context(value)


def _looks_like_non_complication_outcome_or_severity(value: str) -> bool:
    return influenza_rules.looks_like_non_complication_outcome_or_severity(value)


def _diagnostic_test_order_action_candidate(
    issue: dict[str, Any],
) -> dict[str, Any] | None:
    if issue.get("issue_kind") != "diagnostic_evidence_direction_mismatch":
        return None
    if _string_value(issue.get("source_type")) != "Test":
        return None

    edge_id = _string_value(issue.get("edge_id"))
    source = _string_value(issue.get("source"))
    target = _string_value(issue.get("target"))
    keywords = _string_value(issue.get("keywords"))
    if not _is_direct_order_test_candidate_source(source):
        return None
    new_source = _string_value(issue.get("new_source")) or target
    new_target = _string_value(issue.get("new_target")) or source
    if (
        not edge_id
        or not source
        or not target
        or not keywords
        or not new_source
        or not new_target
    ):
        return None

    qualifiers = _qualifiers(issue.get("qualifiers"))
    qualifiers.setdefault("indication", "diagnosis_or_severity_assessment")
    return {
        "proposal_type": "medical_relation_schema_migration",
        "target": f"edge:{edge_id}",
        "issue_kind": "diagnostic_evidence_direction_mismatch",
        "action_payload": {
            "action": "replace_relation",
            "edge_id": edge_id,
            "expected_source": source,
            "expected_target": target,
            "current_keywords": keywords,
            "new_source": new_source,
            "new_target": new_target,
            "new_keywords": "orders_test",
            "qualifiers": qualifiers,
            "new_source_type": "Disease",
            "new_target_type": "Test",
        },
    }


def _is_direct_order_test_candidate_source(source: str) -> bool:
    return influenza_rules.looks_like_direct_order_test_source(source)


def _multi_predicate_split_action_candidate(
    issue: dict[str, Any],
) -> dict[str, Any] | None:
    if issue.get("issue_kind") != "multi_predicate_edge_split_needed":
        return None
    edge_id = _string_value(issue.get("edge_id"))
    source = _string_value(issue.get("source"))
    target = _string_value(issue.get("target"))
    keywords = _string_value(issue.get("keywords"))
    repair_options = _executable_split_repair_options(issue.get("repair_options"))
    if not edge_id or not source or not target or not keywords or len(repair_options) < 2:
        return None
    return {
        "proposal_type": "medical_fact_role_split",
        "target": f"edge:{edge_id}",
        "issue_kind": "multi_predicate_edge_split_needed",
        "action_payload": {
            "action": "split_relation",
            "edge_id": edge_id,
            "expected_source": source,
            "expected_target": target,
            "current_keywords": keywords,
            "retire_original": True,
            "new_edges": repair_options,
        },
    }


def _executable_split_repair_options(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    new_edges: list[dict[str, Any]] = []
    endpoint_pairs: set[tuple[str, str]] = set()
    for option in value:
        if not isinstance(option, dict):
            return []
        if option.get("auto_fixable") is not True:
            return []
        predicate = _string_value(option.get("predicate"))
        source = _string_value(option.get("new_source") or option.get("source"))
        target = _string_value(option.get("new_target") or option.get("target"))
        if not predicate or not source or not target:
            return []
        endpoint_pair = (source, target)
        if endpoint_pair in endpoint_pairs:
            return []
        endpoint_pairs.add(endpoint_pair)
        edge = {
            "source": source,
            "target": target,
            "predicate": predicate,
        }
        source_type = _string_value(option.get("source_type"))
        target_type = _string_value(option.get("target_type"))
        if source_type:
            edge["source_type"] = source_type
        if target_type:
            edge["target_type"] = target_type
        qualifiers = option.get("qualifiers")
        if isinstance(qualifiers, dict):
            edge["qualifiers"] = dict(qualifiers)
        new_edges.append(edge)
    return new_edges


def _ordered_non_empty_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item:
            strings.append(item)
    return strings


def _qualifiers(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _first_graph_field(value: Any) -> str:
    values = _split_graph_field_values(value)
    return values[0] if values else ""


def _split_graph_field_values(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value)
    for separator in (GRAPH_FIELD_SEP, "<SEP>", "\x1d", "\n", "\r"):
        if separator:
            text = text.replace(separator, "\n")
    return [part.strip() for part in text.splitlines() if part.strip()]


def _proposal_conflicts(
    proposals: list[ImprovementProposal],
) -> list[dict[str, Any]]:
    targets: dict[str, list[ImprovementProposal]] = {}
    for proposal in proposals:
        if _is_report_only_proposal(proposal):
            continue
        targets.setdefault(_proposal_conflict_target(proposal), []).append(proposal)

    conflicts: list[dict[str, Any]] = []
    for target, target_proposals in targets.items():
        action_payloads = {
            _stable_json_key(proposal.action_payload)
            for proposal in target_proposals
        }
        semantic_actions = {
            _proposal_semantic_key(proposal) for proposal in target_proposals
        }
        if len(semantic_actions) <= 1:
            continue
        conflicts.append(
            {
                "target": target,
                "proposal_ids": [proposal.id for proposal in target_proposals],
                "reason": (
                    "different_action_payload_for_same_target"
                    if len(action_payloads) > 1
                    else "different_semantic_action_for_same_target"
                ),
            }
        )
    return conflicts


def _prefer_action_candidates_over_target_conflicts(
    proposals: list[ImprovementProposal],
) -> tuple[list[ImprovementProposal], list[dict[str, Any]]]:
    targets: dict[str, list[ImprovementProposal]] = {}
    for proposal in proposals:
        if _is_report_only_proposal(proposal):
            continue
        targets.setdefault(_proposal_conflict_target(proposal), []).append(proposal)

    dropped_ids: set[str] = set()
    dropped: list[dict[str, Any]] = []
    for target_proposals in targets.values():
        action_candidates = [
            proposal
            for proposal in target_proposals
            if _is_action_candidate_proposal(proposal)
        ]
        if not action_candidates:
            continue
        semantic_actions = {
            _proposal_semantic_key(proposal) for proposal in target_proposals
        }
        if len(semantic_actions) <= 1:
            continue
        for proposal in target_proposals:
            if _is_action_candidate_proposal(proposal):
                continue
            dropped_ids.add(proposal.id)
            dropped.append(_dropped_proposal(proposal, "conflicting_with_action_candidate"))

    if not dropped_ids:
        return proposals, []
    return [
        proposal for proposal in proposals if proposal.id not in dropped_ids
    ], dropped


def _drop_duplicate_proposal_ids(
    proposals: list[ImprovementProposal],
) -> tuple[list[ImprovementProposal], list[dict[str, Any]]]:
    seen_ids: dict[str, str] = {}
    unique: list[ImprovementProposal] = []
    dropped: list[dict[str, Any]] = []
    for proposal in proposals:
        semantic_key = _proposal_semantic_key(proposal)
        previous_semantic_key = seen_ids.get(proposal.id)
        if previous_semantic_key is not None:
            reason = (
                "duplicate_proposal_id"
                if previous_semantic_key == semantic_key
                else "duplicate_proposal_id_conflict"
            )
            dropped.append(_dropped_proposal(proposal, reason))
            continue
        seen_ids[proposal.id] = semantic_key
        unique.append(proposal)
    return unique, dropped


def _is_report_only_proposal(proposal: ImprovementProposal) -> bool:
    return proposal.type in REPORT_ONLY_PROPOSAL_TYPES and not proposal.action_payload


def _is_action_candidate_proposal(proposal: ImprovementProposal) -> bool:
    return proposal.id.startswith("prop-action-candidate-")


def _proposal_conflict_target(proposal: ImprovementProposal) -> str:
    edge_id = _string_value(proposal.action_payload.get("edge_id"))
    if edge_id:
        return edge_id
    return proposal.target.removeprefix("edge:")


def _sorted_proposals(
    proposals: list[ImprovementProposal],
) -> list[ImprovementProposal]:
    return sorted(
        proposals,
        key=lambda proposal: (
            _RISK_PRIORITY.get(proposal.risk, len(_RISK_PRIORITY)),
            -proposal.confidence,
            proposal.id,
            proposal.target,
        ),
    )


def _proposal_semantic_key(proposal: ImprovementProposal) -> str:
    if proposal.action_payload:
        return proposal_fingerprints(proposal.to_dict()).semantic

    action_identity: dict[str, Any] = {
        "action_payload": _semantic_action_payload(proposal.action_payload)
    }
    target = proposal.target
    if not proposal.action_payload:
        action_identity["proposed_change"] = proposal.proposed_change
    else:
        target = ""
    return _stable_json_key(
        {
            "type": proposal.type,
            "target": target,
            **action_identity,
        }
    )


def _semantic_action_payload(action_payload: dict[str, Any]) -> dict[str, Any]:
    if action_payload.get("action") == "replace_relation":
        return {
            "action": "replace_relation",
            "edge_id": action_payload.get("edge_id"),
            "new_source": action_payload.get("new_source"),
            "new_target": action_payload.get("new_target"),
            "new_keywords": action_payload.get("new_keywords"),
            "qualifiers": action_payload.get("qualifiers", {}),
        }
    return action_payload


def _dropped_proposal(
    proposal: ImprovementProposal, reason: str
) -> dict[str, str]:
    return {
        "proposal_id": proposal.id,
        "target": proposal.target,
        "reason": reason,
    }


def _stable_json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _clip_string(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    suffix = "...[truncated]"
    if max_chars <= len(suffix):
        return value[:max_chars]
    return value[: max_chars - len(suffix)] + suffix
