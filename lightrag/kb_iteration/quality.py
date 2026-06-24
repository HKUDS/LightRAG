from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.medical_kg.ontology import (
    TOP_LEVEL_MEDICAL_CATEGORIES,
    canonical_name,
    is_value_like_entity,
)

from .medical_schema import (
    MedicalRelationMigrationRule,
    is_medical_profile,
    medical_type_allowed,
    migration_rule_for_legacy_keyword,
    normalize_medical_entity_type,
    relation_spec_by_id,
    validate_relation_instance,
)
from .models import KGSnapshot, QualityFinding, QualityScore, SnapshotEdge, SnapshotNode
from .relation_tokens import split_relation_tokens

GENERIC_RELATION_KEYWORDS = frozenset({"\u76f8\u5173", "\u90bb\u63a5", "related"})
CLINICAL_MANIFESTATION_KEYWORDS = frozenset(
    {
        "clinical_manifestation",
        "clinical manifestation",
        "has_manifestation",
        "manifestation",
        "symptom_of",
        "has_symptom",
        "\u4e34\u5e8a\u8868\u73b0",
        "\u75c7\u72b6",
        "\u75c7\u72b6\u5f52\u7c7b",
        "\u8868\u73b0\u4e3a",
    }
)
CAUSATIVE_AGENT_KEYWORDS = frozenset(
    {"causative_agent", "\u75c5\u539f\u5bfc\u81f4"}
)
TREATMENT_RECOMMENDATION_KEYWORDS = frozenset(
    {
        "recommended_treatment",
        "treatment_recommendation",
        "has_indication",
        "recommends",
        "\u63a8\u8350\u6cbb\u7597",
    }
)
RECOMMENDS_RELATION_KEYWORDS = frozenset({"recommends"})
DIAGNOSTIC_BASIS_KEYWORDS = frozenset(
    {"diagnostic_basis", "diagnosis_basis", "\u8bca\u65ad\u4f9d\u636e"}
)
DOSING_USAGE_KEYWORDS = frozenset(
    {"dosage_usage", "dose_usage", "has_dosing_regimen", "\u5242\u91cf\u7528\u6cd5"}
)
APPLIES_TO_KEYWORDS = frozenset(
    {"applies_to", "applicable_to", "recommended_for", "\u9002\u7528\u4e8e"}
)
TAXONOMY_RELATION_KEYWORDS = frozenset(
    {"belongs_to", "is_a", "type_of", "\u5c5e\u4e8e"}
)
DISEASE_ENTITY_TYPES = frozenset({"Disease"})
SYMPTOM_ENTITY_TYPES = frozenset({"Symptom"})
MANIFESTATION_SOURCE_TYPES = frozenset({"Symptom", "ClinicalFinding"})
MANIFESTATION_TARGET_TYPES = frozenset(
    {"Disease", "Syndrome", "ClinicalCondition"}
)
INTERVENTION_ENTITY_TYPES = frozenset({"Drug", "Treatment", "Vaccine"})
RECOMMENDATION_SOURCE_TYPES = frozenset(
    {"Guideline", "Recommendation", "ClinicalPathway"}
)
INDICATION_TARGET_TYPES = frozenset({"Disease", "ClinicalCondition", "Population"})
RECOMMENDATION_TARGET_TYPES = frozenset({"Drug", "Treatment", "Vaccine", "Test"})
DIAGNOSTIC_TARGET_TYPES = frozenset({"Disease", "Syndrome", "ClinicalCondition"})
DIAGNOSTIC_SCHEMA_PREDICATES = frozenset(
    {
        "has_diagnostic_criterion",
        "criterion_requires",
        "has_evidence",
        "supports_or_refutes",
    }
)
ADVERSE_REACTION_SCHEMA_PREDICATES = frozenset({"may_cause_adverse_reaction"})
QUALITY_WEIGHTS = {
    "entity_hygiene": 20,
    "relation_semantics": 20,
    "hierarchy_completeness": 20,
    "evidence_grounding": 20,
    "web_readability": 10,
    "iteration_readiness": 10,
}


def evaluate_snapshot_quality(snapshot: KGSnapshot) -> QualityScore:
    value_like_nodes = [
        node
        for node in snapshot.nodes
        if is_value_like_entity(node.label or node.id, node.entity_type)
    ]
    generic_edges = [edge for edge in snapshot.edges if _is_generic_relation(edge)]
    node_by_id = {node.id: node for node in snapshot.nodes}
    clinical_direction_issues = _clinical_relation_direction_issues(
        snapshot.edges, node_by_id
    )
    taxonomy_relation_misuses = _taxonomy_relation_misuses(snapshot.edges, node_by_id)
    legacy_schema_relation_issues = (
        _medical_schema_legacy_relation_issues(snapshot.edges)
        if _needs_medical_schema(snapshot)
        else []
    )
    medical_domain_range_issues = (
        [
            *_medical_schema_domain_range_issues(snapshot.edges, node_by_id),
            *_medical_schema_cross_edge_issues(snapshot.edges),
        ]
        if _needs_medical_schema(snapshot)
        else []
    )
    medical_schema_issues = _medical_schema_issue_details(
        domain_range_issues=medical_domain_range_issues,
        clinical_direction_issues=clinical_direction_issues,
        taxonomy_relation_misuses=taxonomy_relation_misuses,
        legacy_schema_relation_issues=legacy_schema_relation_issues,
    )
    legacy_schema_edges = [edge for edge, _rule in legacy_schema_relation_issues]
    edge_by_id = {edge.id: edge for edge in snapshot.edges}
    relation_semantic_issue_edges = _unique_edges_by_id(
        [
            *generic_edges,
            *clinical_direction_issues,
            *taxonomy_relation_misuses,
            *legacy_schema_edges,
            *[
                edge_by_id[edge_id]
                for issue in medical_schema_issues
                if (edge_id := issue.get("edge_id")) in edge_by_id
            ],
        ]
    )

    missing_node_sources = [
        node for node in snapshot.nodes if not _has_provenance(node.source_id)
    ]
    missing_node_file_paths = [
        node for node in snapshot.nodes if not _has_provenance(node.file_path)
    ]
    missing_edge_sources = [
        edge for edge in snapshot.edges if not _has_provenance(edge.source_id)
    ]
    missing_edge_file_paths = [
        edge for edge in snapshot.edges if not _has_provenance(edge.file_path)
    ]

    edge_ids_by_node = _edge_ids_by_node(snapshot.edges)
    entity_cleanup_issues = _entity_cleanup_issue_details(
        value_like_nodes=value_like_nodes,
        synonym_duplicate_issues=_synonym_duplicate_issues(snapshot.nodes),
        edge_ids_by_node=edge_ids_by_node,
    )

    hierarchy_details = _hierarchy_details(snapshot)
    hierarchy = _hierarchy_metrics_from_details(hierarchy_details)
    disease_hub_overload_ratio = _disease_hub_overload_ratio(snapshot)

    metrics = {
        "value_like_node_count": len(value_like_nodes),
        "generic_relation_count": len(generic_edges),
        "clinical_relation_direction_issue_count": len(clinical_direction_issues),
        "taxonomy_relation_misuse_count": len(taxonomy_relation_misuses),
        "medical_schema_legacy_relation_count": len(legacy_schema_relation_issues),
        "medical_schema_issue_count": len(medical_schema_issues),
        "value_node_to_qualifier_candidate_count": sum(
            1
            for issue in entity_cleanup_issues
            if issue["issue_kind"] == "value_node_to_qualifier"
        ),
        "synonym_duplicate_count": sum(
            1
            for issue in entity_cleanup_issues
            if issue["issue_kind"] == "synonym_duplicate"
        ),
        "diagnostic_evidence_flattening_count": _diagnostic_evidence_flattening_count(
            legacy_schema_relation_issues
        ),
        "adverse_reaction_role_conflict_count": _adverse_reaction_role_conflict_count(
            legacy_schema_relation_issues
        ),
        "relation_semantic_issue_count": len(relation_semantic_issue_edges),
        "missing_node_source_count": len(missing_node_sources),
        "missing_node_file_path_count": len(missing_node_file_paths),
        "missing_edge_source_count": len(missing_edge_sources),
        "missing_edge_file_path_count": len(missing_edge_file_paths),
        "hierarchy_required_branch_count": hierarchy["required"],
        "hierarchy_present_branch_count": hierarchy["present"],
        "hierarchy_missing_branch_count": hierarchy["missing"],
        "disease_hub_overload_ratio": disease_hub_overload_ratio,
    }

    findings = _build_findings(
        value_like_nodes=value_like_nodes,
        generic_edges=generic_edges,
        clinical_direction_issues=clinical_direction_issues,
        taxonomy_relation_misuses=taxonomy_relation_misuses,
        legacy_schema_relation_issues=legacy_schema_relation_issues,
        medical_schema_issues=medical_schema_issues,
        entity_cleanup_issues=entity_cleanup_issues,
        metrics=metrics,
        missing_evidence_examples=_missing_evidence_examples(
            missing_node_sources=missing_node_sources,
            missing_node_file_paths=missing_node_file_paths,
            missing_edge_sources=missing_edge_sources,
            missing_edge_file_paths=missing_edge_file_paths,
        ),
    )
    critical_blockers = []
    if not snapshot.nodes:
        critical_blockers.append("Snapshot contains no nodes.")

    subscores = _score_subsections(snapshot, metrics)
    overall = _weighted_average(subscores)
    if critical_blockers:
        overall = 0

    return QualityScore(
        overall=overall,
        subscores=subscores,
        metrics=metrics,
        details={
            "hierarchy_branches": hierarchy_details,
            "relation_semantic_issues": _relation_semantic_issue_details(
                clinical_direction_issues=clinical_direction_issues,
                taxonomy_relation_misuses=taxonomy_relation_misuses,
                legacy_schema_relation_issues=legacy_schema_relation_issues,
            ),
            "medical_schema_issues": medical_schema_issues,
            "entity_cleanup_issues": entity_cleanup_issues,
        },
        findings=findings,
        critical_blockers=critical_blockers,
    )


def write_quality_artifacts(
    score: QualityScore, output_dir: str | Path
) -> dict[str, Path]:
    output_path = Path(output_dir)
    snapshot_dir = output_path / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "quality_score": snapshot_dir / "quality_score.json",
        "quality_report": output_path / "quality_report.md",
    }

    _write_json(artifacts["quality_score"], score.to_dict())
    artifacts["quality_report"].write_text(
        _quality_report(score), encoding="utf-8"
    )
    return artifacts


def _is_generic_relation(edge: SnapshotEdge) -> bool:
    tokens = _relation_tokens(edge.keywords)
    if not tokens:
        return True
    return any(token in GENERIC_RELATION_KEYWORDS for token in tokens)


def _relation_tokens(keywords: str) -> list[str]:
    return split_relation_tokens(keywords)


def _clinical_relation_direction_issues(
    edges: list[SnapshotEdge], node_by_id: dict[str, SnapshotNode]
) -> list[SnapshotEdge]:
    issues = []
    for edge in edges:
        tokens = set(_relation_tokens(edge.keywords))
        if not tokens & CLINICAL_MANIFESTATION_KEYWORDS:
            continue
        source_node = node_by_id.get(edge.source)
        target_node = node_by_id.get(edge.target)
        if _is_symptom_node(source_node) and _is_disease_node(target_node):
            issues.append(edge)
    return issues


def _taxonomy_relation_misuses(
    edges: list[SnapshotEdge], node_by_id: dict[str, SnapshotNode]
) -> list[SnapshotEdge]:
    issues = []
    for edge in edges:
        tokens = set(_relation_tokens(edge.keywords))
        if not tokens & TAXONOMY_RELATION_KEYWORDS:
            continue
        source_node = node_by_id.get(edge.source)
        target_node = node_by_id.get(edge.target)
        if _connects_disease_and_symptom(source_node, target_node):
            issues.append(edge)
    return issues


def _medical_schema_legacy_relation_issues(
    edges: list[SnapshotEdge],
) -> list[tuple[SnapshotEdge, MedicalRelationMigrationRule]]:
    issues = []
    for edge in edges:
        for token in _relation_tokens(edge.keywords):
            rule = migration_rule_for_legacy_keyword(token)
            if rule is not None:
                issues.append((edge, rule))
                break
            try:
                spec = relation_spec_by_id(token)
            except KeyError:
                continue
            if spec.deprecated and spec.canonical_replacement:
                issues.append(
                    (
                        edge,
                        MedicalRelationMigrationRule(
                            legacy_keywords=(token,),
                            canonical_options=(spec.canonical_replacement,),
                            guidance=(
                                f"{token} is deprecated; use "
                                f"{spec.canonical_replacement}."
                            ),
                        ),
                    )
                )
                break
    return issues


def _medical_schema_domain_range_issues(
    edges: list[SnapshotEdge], node_by_id: dict[str, SnapshotNode]
) -> list[dict[str, Any]]:
    issues = []
    for edge in edges:
        tokens = set(_relation_tokens(edge.keywords))
        source_type = _normalized_node_type(node_by_id.get(edge.source))
        target_type = _normalized_node_type(node_by_id.get(edge.target))

        if tokens & CLINICAL_MANIFESTATION_KEYWORDS:
            if (
                source_type in MANIFESTATION_SOURCE_TYPES
                and target_type in MANIFESTATION_TARGET_TYPES
            ):
                issues.append(
                    _schema_issue_payload(
                        edge,
                        issue_kind="reverse_clinical_manifestation",
                        candidate_predicates=["has_manifestation"],
                        new_source=edge.target,
                        new_target=edge.source,
                        guidance=(
                            "Clinical manifestation edges should point disease "
                            "to symptom."
                        ),
                        source_type=source_type,
                        target_type=target_type,
                    )
                )
            if (
                source_type in INTERVENTION_ENTITY_TYPES
                and target_type == "Symptom"
            ):
                issues.append(
                    _schema_issue_payload(
                        edge,
                        issue_kind="adverse_reaction_modeled_as_manifestation",
                        candidate_predicates=["may_cause_adverse_reaction"],
                        guidance=(
                            "Symptoms caused by drugs, vaccines, or treatments "
                            "should be modeled as adverse reactions."
                        ),
                        source_type=source_type,
                        target_type=target_type,
                    )
                )

        if (
            tokens & CAUSATIVE_AGENT_KEYWORDS
            and source_type == "Pathogen"
            and target_type == "Disease"
        ):
            issues.append(
                _schema_issue_payload(
                    edge,
                    issue_kind="reverse_causative_agent",
                    candidate_predicates=["causative_agent"],
                    new_source=edge.target,
                    new_target=edge.source,
                    guidance="Causative-agent edges should point disease to pathogen.",
                    source_type=source_type,
                    target_type=target_type,
                )
            )

        if (
            tokens & TREATMENT_RECOMMENDATION_KEYWORDS
            and not _is_valid_treatment_recommendation_edge(
                tokens, source_type, target_type
            )
        ):
            issues.append(
                _schema_issue_payload(
                    edge,
                    issue_kind="treatment_domain_range_mismatch",
                    candidate_predicates=["has_indication", "recommends"],
                    guidance=(
                        "Treatment recommendations should connect an intervention "
                        "to an indication or a recommendation to an intervention."
                    ),
                    source_type=source_type,
                    target_type=target_type,
                )
            )

        if (
            tokens & DIAGNOSTIC_BASIS_KEYWORDS
            and source_type == "Test"
            and target_type in DIAGNOSTIC_TARGET_TYPES
        ):
            issues.append(
                _schema_issue_payload(
                    edge,
                    issue_kind="diagnostic_evidence_direction_mismatch",
                    candidate_predicates=[
                        "has_diagnostic_criterion",
                        "supports_or_refutes",
                    ],
                    new_source=edge.target,
                    new_target=edge.source,
                    guidance=(
                        "Diagnosis-basis edges should separate disease criteria "
                        "from evidence that supports or refutes a diagnosis."
                    ),
                    source_type=source_type,
                    target_type=target_type,
                )
            )

        issues.extend(
            _canonical_relation_domain_range_issues(
                edge,
                tokens=tokens,
                source_type=source_type,
                target_type=target_type,
            )
        )

        predicate_tokens = _multi_predicate_tokens(tokens)
        if len(predicate_tokens) > 1:
            candidate_predicates = _candidate_predicates_for_tokens(predicate_tokens)
            repair_options = _split_repair_options(
                edge,
                candidate_predicates=candidate_predicates,
                source_type=source_type,
                target_type=target_type,
            )
            issues.append(
                _schema_issue_payload(
                    edge,
                    issue_kind="multi_predicate_edge_split_needed",
                    candidate_predicates=candidate_predicates,
                    suggested_action="split_relation",
                    guidance=(
                        "Edges with multiple medical predicate meanings should "
                        "be split into separate relations."
                    ),
                    source_type=source_type,
                    target_type=target_type,
                    predicate_tokens=predicate_tokens,
                    repair_options=repair_options,
                )
            )

    return issues


def _canonical_relation_domain_range_issues(
    edge: SnapshotEdge,
    *,
    tokens: set[str],
    source_type: str,
    target_type: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for token in sorted(tokens):
        try:
            spec = relation_spec_by_id(token)
        except KeyError:
            continue
        validation_errors = validate_relation_instance(
            predicate=token,
            source_type=source_type,
            target_type=target_type,
            qualifiers=_edge_qualifiers(edge),
        )
        if not validation_errors:
            continue
        issue_kind = _issue_kind_for_relation_errors(validation_errors)
        expected_domain_types = _normalized_expected_types(spec.domain_types)
        expected_range_types = _normalized_expected_types(spec.range_types)
        issue = _schema_issue_payload(
            edge,
            issue_kind=issue_kind,
            candidate_predicates=[spec.id],
            guidance=spec.guidance
            or "Canonical medical relation domain/range does not match node types.",
            source_type=source_type,
            target_type=target_type,
        )
        issue["expected_domain_types"] = list(expected_domain_types)
        issue["expected_range_types"] = list(expected_range_types)
        issue["validation_errors"] = validation_errors
        missing_qualifiers = _missing_relation_qualifier_names(validation_errors)
        if missing_qualifiers:
            issue["missing_qualifiers"] = missing_qualifiers
        missing_qualifier_groups = _missing_relation_qualifier_groups(
            validation_errors
        )
        if missing_qualifier_groups:
            issue["missing_qualifier_groups"] = missing_qualifier_groups
        unsupported_qualifiers = _unsupported_relation_qualifier_names(
            validation_errors
        )
        if unsupported_qualifiers:
            issue["unsupported_qualifiers"] = unsupported_qualifiers
        invalid_qualifier_values = _invalid_relation_qualifier_value_names(
            validation_errors
        )
        if invalid_qualifier_values:
            issue["invalid_qualifier_values"] = invalid_qualifier_values
        issues.append(issue)
    return issues


def _issue_kind_for_relation_errors(errors: list[str]) -> str:
    if any(" is outside " in error for error in errors):
        return "canonical_relation_domain_range_mismatch"
    if any("requires qualifier" in error for error in errors) or any(
        "requires one qualifier from" in error for error in errors
    ):
        return "missing_required_relation_qualifier"
    if any("does not allow qualifier" in error for error in errors):
        return "unsupported_relation_qualifier"
    if any(" must be one of " in error for error in errors):
        return "invalid_relation_qualifier_value"
    return "canonical_relation_domain_range_mismatch"


def _missing_relation_qualifier_names(errors: list[str]) -> list[str]:
    missing = []
    marker = " requires qualifier "
    for error in errors:
        if marker not in error:
            continue
        qualifier = error.split(marker, 1)[1].strip()
        if qualifier and qualifier not in missing:
            missing.append(qualifier)
    return missing


def _missing_relation_qualifier_groups(errors: list[str]) -> list[list[str]]:
    groups = []
    marker = " requires one qualifier from "
    for error in errors:
        if marker not in error:
            continue
        group = [
            qualifier.strip()
            for qualifier in error.split(marker, 1)[1].split("|")
            if qualifier.strip()
        ]
        if group and group not in groups:
            groups.append(group)
    return groups


def _unsupported_relation_qualifier_names(errors: list[str]) -> list[str]:
    unsupported = []
    marker = " does not allow qualifier(s) "
    for error in errors:
        if marker not in error:
            continue
        for qualifier in error.split(marker, 1)[1].split(","):
            qualifier = qualifier.strip()
            if qualifier and qualifier not in unsupported:
                unsupported.append(qualifier)
    return unsupported


def _invalid_relation_qualifier_value_names(errors: list[str]) -> list[str]:
    invalid = []
    qualifier_marker = " qualifier "
    value_marker = " must be one of "
    for error in errors:
        if qualifier_marker not in error or value_marker not in error:
            continue
        qualifier = error.split(qualifier_marker, 1)[1].split(value_marker, 1)[0]
        qualifier = qualifier.strip()
        if qualifier and qualifier not in invalid:
            invalid.append(qualifier)
    return invalid


def _medical_schema_cross_edge_issues(edges: list[SnapshotEdge]) -> list[dict[str, Any]]:
    by_scope: dict[
        tuple[str, str, str],
        dict[str, list[SnapshotEdge]],
    ] = {}
    scoped_predicates = {
        "recommended_for",
        "not_recommended_for",
        "contraindicated_for",
        "precaution_for",
        "temporarily_deferred_for",
    }
    for edge in edges:
        for predicate in set(_relation_tokens(edge.keywords)) & scoped_predicates:
            key = (
                edge.source.strip().casefold(),
                edge.target.strip().casefold(),
                _qualifier_scope_key(_edge_qualifiers(edge)),
            )
            by_scope.setdefault(key, {}).setdefault(predicate, []).append(edge)

    issues: list[dict[str, Any]] = []
    safety_predicates = (
        "contraindicated_for",
        "not_recommended_for",
        "precaution_for",
        "temporarily_deferred_for",
    )
    for scoped_edges in by_scope.values():
        for recommended_edge in scoped_edges.get("recommended_for", []):
            for safety_predicate in safety_predicates:
                for safety_edge in scoped_edges.get(safety_predicate, []):
                    issue = _schema_issue_payload(
                        recommended_edge,
                        issue_kind="conflicting_recommendation_safety_scope",
                        candidate_predicates=["recommended_for", safety_predicate],
                        suggested_action="review_conflict",
                        guidance=(
                            "The same intervention and target have both a "
                            "recommendation and a safety-limiting relation under "
                            "the same qualifier scope. Split the scope or preserve "
                            "the contradiction for human review."
                        ),
                    )
                    issue["conflict_edge_id"] = safety_edge.id
                    issue["conflict_keywords"] = safety_edge.keywords
                    issues.append(issue)
    return issues


def _qualifier_scope_key(qualifiers: dict[str, Any]) -> str:
    return json.dumps(qualifiers, ensure_ascii=False, sort_keys=True)


def _normalized_expected_types(types: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for type_name in types:
        normalized_type = normalize_medical_entity_type(type_name)
        if normalized_type == "Unknown" and type_name:
            normalized_type = str(type_name)
        if normalized_type not in normalized:
            normalized.append(normalized_type)
    return tuple(normalized)


def _type_allowed(actual_type: str, expected_types: tuple[str, ...]) -> bool:
    return medical_type_allowed(actual_type, expected_types)


def _edge_qualifiers(edge: SnapshotEdge) -> dict[str, Any]:
    qualifiers = edge.properties.get("qualifiers")
    if isinstance(qualifiers, dict):
        return deepcopy(qualifiers)
    return {}


def _is_valid_treatment_recommendation_edge(
    tokens: set[str], source_type: str, target_type: str
) -> bool:
    if tokens & RECOMMENDS_RELATION_KEYWORDS:
        return (
            source_type in RECOMMENDATION_SOURCE_TYPES
            and target_type in RECOMMENDATION_TARGET_TYPES
        )
    if source_type in INTERVENTION_ENTITY_TYPES and target_type in INDICATION_TARGET_TYPES:
        return True
    return False


def _normalized_node_type(node: SnapshotNode | None) -> str:
    if node is None:
        return "Unknown"
    return normalize_medical_entity_type(node.entity_type)


def _multi_predicate_tokens(tokens: set[str]) -> list[str]:
    semantic_tokens = []
    for token in sorted(tokens):
        if (
            token in DOSING_USAGE_KEYWORDS
            or token in TREATMENT_RECOMMENDATION_KEYWORDS
            or token in APPLIES_TO_KEYWORDS
            or token in DIAGNOSTIC_BASIS_KEYWORDS
        ):
            semantic_tokens.append(token)
    return semantic_tokens


def _candidate_predicates_for_tokens(tokens: list[str]) -> list[str]:
    candidates: list[str] = []
    for token in tokens:
        if token in DOSING_USAGE_KEYWORDS:
            candidates.append("has_dosing_regimen")
        if token in TREATMENT_RECOMMENDATION_KEYWORDS:
            candidates.extend(["has_indication", "recommends"])
        if token in APPLIES_TO_KEYWORDS:
            candidates.extend(["recommended_for", "has_indication"])
        if token in DIAGNOSTIC_BASIS_KEYWORDS:
            candidates.extend(["has_diagnostic_criterion", "supports_or_refutes"])
    return list(dict.fromkeys(candidates))


def _connects_disease_and_symptom(
    source_node: SnapshotNode | None, target_node: SnapshotNode | None
) -> bool:
    return (
        _is_disease_node(source_node)
        and _is_symptom_node(target_node)
        or _is_symptom_node(source_node)
        and _is_disease_node(target_node)
    )


def _is_disease_node(node: SnapshotNode | None) -> bool:
    return _normalized_node_type(node) in DISEASE_ENTITY_TYPES


def _is_symptom_node(node: SnapshotNode | None) -> bool:
    return _normalized_node_type(node) in SYMPTOM_ENTITY_TYPES


def _unique_edges_by_id(edges: list[SnapshotEdge]) -> list[SnapshotEdge]:
    seen = set()
    unique_edges = []
    for edge in edges:
        if edge.id in seen:
            continue
        seen.add(edge.id)
        unique_edges.append(edge)
    return unique_edges


def _has_provenance(value: str) -> bool:
    return any(part.strip() for part in value.split(GRAPH_FIELD_SEP))


def _hierarchy_details(snapshot: KGSnapshot) -> dict[str, list[dict[str, Any]]]:
    if not _needs_medical_hierarchy(snapshot):
        return {"required": [], "present": [], "missing": []}

    matched_by_key = _matched_hierarchy_nodes_by_key(snapshot)
    required = [
        _hierarchy_branch_payload(category)
        for category in TOP_LEVEL_MEDICAL_CATEGORIES
    ]
    present = [
        _hierarchy_branch_payload(
            category, matched_node_ids=matched_by_key[category.key]
        )
        for category in TOP_LEVEL_MEDICAL_CATEGORIES
        if matched_by_key[category.key]
    ]
    missing = [
        _hierarchy_branch_payload(category)
        for category in TOP_LEVEL_MEDICAL_CATEGORIES
        if not matched_by_key[category.key]
    ]
    return {"required": required, "present": present, "missing": missing}


def _hierarchy_branch_payload(
    category: Any, matched_node_ids: list[str] | None = None
) -> dict[str, Any]:
    payload = {
        "key": category.key,
        "label": category.label,
        "aliases": list(category.aliases),
    }
    if matched_node_ids is not None:
        payload["matched_node_ids"] = sorted(matched_node_ids)
    return payload


def _hierarchy_metrics_from_details(
    details: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    return {
        "required": len(details["required"]),
        "present": len(details["present"]),
        "missing": len(details["missing"]),
    }


def _matched_hierarchy_nodes_by_key(snapshot: KGSnapshot) -> dict[str, list[str]]:
    required_identifiers = _medical_category_identifiers()
    matched_by_key = {category.key: [] for category in TOP_LEVEL_MEDICAL_CATEGORIES}
    for node in snapshot.nodes:
        node_identifiers = _node_hierarchy_identifiers(node)
        for key, identifiers in required_identifiers.items():
            if node_identifiers & identifiers:
                matched_by_key[key].append(node.id)
    return matched_by_key


def _medical_category_identifiers() -> dict[str, set[str]]:
    identifiers_by_key = {}
    for category in TOP_LEVEL_MEDICAL_CATEGORIES:
        identifiers = set()
        for raw_identifier in (category.key, category.label, *category.aliases):
            normalized = _normalize_identifier(raw_identifier)
            if normalized:
                identifiers.add(normalized)
        identifiers_by_key[category.key] = identifiers
    return identifiers_by_key


def _node_hierarchy_identifiers(node: SnapshotNode) -> set[str]:
    identifiers = {node.id, node.label, str(node.properties.get("medical_group", ""))}
    return {
        normalized
        for identifier in identifiers
        if (normalized := _normalize_identifier(identifier))
    }


def _normalize_identifier(value: Any) -> str:
    return str(value).strip().casefold()


def _needs_medical_hierarchy(snapshot: KGSnapshot) -> bool:
    profile = str(snapshot.metadata.get("profile", "")).casefold()
    workspace = snapshot.workspace.casefold()
    return (
        profile == "clinical_guideline_zh"
        or "medical" in workspace
        or "influenza" in workspace
    )


def _needs_medical_schema(snapshot: KGSnapshot) -> bool:
    profile = str(snapshot.metadata.get("profile", ""))
    workspace = snapshot.workspace
    return is_medical_profile(profile) or is_medical_profile(workspace)


def _disease_hub_overload_ratio(snapshot: KGSnapshot) -> float:
    if not snapshot.edges:
        return 0.0

    disease_ids = {
        node.id
        for node in snapshot.nodes
        if node.entity_type.strip().casefold() == "disease"
    }
    if not disease_ids:
        return 0.0

    max_direct_edges = max(
        sum(1 for edge in snapshot.edges if disease_id in {edge.source, edge.target})
        for disease_id in disease_ids
    )
    return max_direct_edges / len(snapshot.edges)


def _score_subsections(
    snapshot: KGSnapshot, metrics: dict[str, int | float]
) -> dict[str, int]:
    node_count = max(len(snapshot.nodes), 1)
    edge_count = max(len(snapshot.edges), 1)
    evidence_fields = max((len(snapshot.nodes) + len(snapshot.edges)) * 2, 1)
    missing_evidence = (
        metrics["missing_node_source_count"]
        + metrics["missing_node_file_path_count"]
        + metrics["missing_edge_source_count"]
        + metrics["missing_edge_file_path_count"]
    )

    entity_hygiene = _clamp_score(
        100 - round(100 * metrics["value_like_node_count"] / node_count)
    )
    relation_semantic_issue_count = metrics["relation_semantic_issue_count"]
    relation_semantic_penalty = round(
        100 * relation_semantic_issue_count / edge_count
    )
    if relation_semantic_issue_count:
        relation_semantic_penalty = max(1, relation_semantic_penalty)
    relation_semantics = _clamp_score(100 - relation_semantic_penalty)
    if metrics["hierarchy_required_branch_count"]:
        hierarchy_completeness = _clamp_score(
            round(
                100
                * metrics["hierarchy_present_branch_count"]
                / metrics["hierarchy_required_branch_count"]
            )
        )
    else:
        hierarchy_completeness = 100
    evidence_grounding = _clamp_score(
        100 - round(100 * missing_evidence / evidence_fields)
    )
    web_readability = _clamp_score(
        100 - round(100 * float(metrics["disease_hub_overload_ratio"]))
    )
    iteration_readiness = _clamp_score(
        round((entity_hygiene + relation_semantics + evidence_grounding) / 3)
    )

    return {
        "entity_hygiene": entity_hygiene,
        "relation_semantics": relation_semantics,
        "hierarchy_completeness": hierarchy_completeness,
        "evidence_grounding": evidence_grounding,
        "web_readability": web_readability,
        "iteration_readiness": iteration_readiness,
    }


def _build_findings(
    *,
    value_like_nodes: list,
    generic_edges: list[SnapshotEdge],
    clinical_direction_issues: list[SnapshotEdge],
    taxonomy_relation_misuses: list[SnapshotEdge],
    legacy_schema_relation_issues: list[
        tuple[SnapshotEdge, MedicalRelationMigrationRule]
    ],
    medical_schema_issues: list[dict[str, Any]],
    entity_cleanup_issues: list[dict[str, Any]],
    metrics: dict[str, int | float],
    missing_evidence_examples: list[str],
) -> list[QualityFinding]:
    findings = []
    if value_like_nodes:
        findings.append(
            QualityFinding(
                severity="high",
                category="entity_hygiene",
                message="Value-like entities should be represented as attributes.",
                evidence=[node.id for node in value_like_nodes],
                suggested_fix_type="convert_value_node",
                requires_approval=True,
            )
        )

    if clinical_direction_issues:
        findings.append(
            QualityFinding(
                severity="medium",
                category="relation_semantics",
                message=(
                    "Clinical manifestation edges should use a consistent "
                    "disease-to-symptom direction."
                ),
                evidence=_edge_references(clinical_direction_issues),
                suggested_fix_type="normalize_relation_direction",
                requires_approval=True,
            )
        )

    if taxonomy_relation_misuses:
        findings.append(
            QualityFinding(
                severity="medium",
                category="relation_semantics",
                message=(
                    "Taxonomy keywords such as belongs_to/属于 should be "
                    "reserved for category or type hierarchies."
                ),
                evidence=_edge_references(taxonomy_relation_misuses),
                suggested_fix_type="replace_taxonomy_relation_keyword",
                requires_approval=True,
            )
        )
    if generic_edges:
        findings.append(
            QualityFinding(
                severity="high",
                category="relation_semantics",
                message="Generic relation keywords should be replaced with specific semantics.",
                evidence=[edge.id for edge in generic_edges],
                suggested_fix_type="replace_relation_keyword",
                requires_approval=True,
            )
        )
    if legacy_schema_relation_issues:
        findings.append(
            QualityFinding(
                severity="high",
                category="relation_semantics",
                message=(
                    "Legacy overloaded medical relation keywords should be "
                    "migrated to Medical Relationship Schema v1 predicates."
                ),
                evidence=_edge_references(
                    [edge for edge, _rule in legacy_schema_relation_issues]
                ),
                suggested_fix_type="normalize_medical_relation_schema",
                requires_approval=True,
            )
        )

    if missing_evidence_examples:
        findings.append(
            QualityFinding(
                severity="high",
                category="evidence_grounding",
                message="Snapshot items should retain source_id and file_path evidence.",
                evidence=missing_evidence_examples,
                suggested_fix_type="restore_evidence",
                requires_approval=True,
            )
        )

    if metrics["hierarchy_missing_branch_count"]:
        findings.append(
            QualityFinding(
                severity="medium",
                category="hierarchy_completeness",
                message="Medical hierarchy is missing expected first-level branches.",
                evidence=[str(metrics["hierarchy_missing_branch_count"])],
                suggested_fix_type="add_hierarchy_branch",
                requires_approval=True,
            )
        )

    if metrics["disease_hub_overload_ratio"] > 0.5:
        findings.append(
            QualityFinding(
                severity="medium",
                category="web_readability",
                message="A disease node owns a high share of direct edges.",
                evidence=[f"{metrics['disease_hub_overload_ratio']:.2f}"],
                suggested_fix_type="split_hub_edges",
                requires_approval=True,
            )
        )

    if medical_schema_issues:
        findings.append(
            QualityFinding(
                severity="high",
                category="relation_semantics",
                message="Medical relation schema issues need predicate or direction migration.",
                evidence=_issue_edge_references(medical_schema_issues),
                suggested_fix_type="medical_relation_schema_migration",
                requires_approval=True,
            )
        )

    if entity_cleanup_issues:
        findings.append(
            QualityFinding(
                severity="high",
                category="entity_hygiene",
                message="Entity cleanup candidates should be reviewed before mutation.",
                evidence=_entity_cleanup_references(entity_cleanup_issues),
                suggested_fix_type="entity_cleanup",
                requires_approval=True,
            )
        )

    return findings


def _edge_references(edges: list[SnapshotEdge], limit: int = 25) -> list[str]:
    return [f"edge:{edge.id}" for edge in edges[:limit]]


def _issue_edge_references(
    issues: list[dict[str, Any]], limit: int = 25
) -> list[str]:
    references = []
    for issue in issues:
        edge_id = issue.get("edge_id")
        if isinstance(edge_id, str) and edge_id:
            references.append(f"edge:{edge_id}")
        if len(references) >= limit:
            break
    return references


def _entity_cleanup_references(
    issues: list[dict[str, Any]], limit: int = 25
) -> list[str]:
    references = []
    for issue in issues:
        node_id = issue.get("node_id")
        if isinstance(node_id, str) and node_id:
            references.append(f"node:{node_id}")
            if len(references) >= limit:
                break
            continue
        for duplicate_node_id in issue.get("node_ids", []):
            if not isinstance(duplicate_node_id, str) or not duplicate_node_id:
                continue
            references.append(f"node:{duplicate_node_id}")
            if len(references) >= limit:
                break
        if len(references) >= limit:
            break
    return references


def _relation_semantic_issue_details(
    *,
    clinical_direction_issues: list[SnapshotEdge],
    taxonomy_relation_misuses: list[SnapshotEdge],
    legacy_schema_relation_issues: list[
        tuple[SnapshotEdge, MedicalRelationMigrationRule]
    ],
) -> dict[str, list[dict[str, str]]]:
    return {
        "clinical_direction": [
            _relation_issue_payload(edge) for edge in clinical_direction_issues
        ],
        "taxonomy_misuse": [
            _relation_issue_payload(edge) for edge in taxonomy_relation_misuses
        ],
        "legacy_schema": [
            _legacy_relation_issue_payload(edge, rule)
            for edge, rule in legacy_schema_relation_issues
        ],
    }


def _medical_schema_issue_details(
    *,
    domain_range_issues: list[dict[str, Any]],
    clinical_direction_issues: list[SnapshotEdge],
    taxonomy_relation_misuses: list[SnapshotEdge],
    legacy_schema_relation_issues: list[
        tuple[SnapshotEdge, MedicalRelationMigrationRule]
    ],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    issues.extend(domain_range_issues)
    domain_range_issue_keys = _schema_issue_keys(domain_range_issues)
    issues.extend(
        _schema_issue_payload(
            edge,
            issue_kind="reverse_clinical_manifestation",
            candidate_predicates=["has_manifestation"],
            new_source=edge.target,
            new_target=edge.source,
            guidance="Clinical manifestation edges should point disease to symptom.",
        )
        for edge in clinical_direction_issues
        if (edge.id, "reverse_clinical_manifestation") not in domain_range_issue_keys
    )
    issues.extend(
        _schema_issue_payload(
            edge,
            issue_kind="disease_symptom_taxonomy_misuse",
            candidate_predicates=["has_manifestation"],
        )
        for edge in taxonomy_relation_misuses
    )
    issues.extend(
        _schema_issue_payload(
            edge,
            issue_kind="legacy_overloaded_relation",
            candidate_predicates=list(rule.canonical_options),
            guidance=rule.guidance,
            medical_subcase=_legacy_medical_subcase(rule),
        )
        for edge, rule in legacy_schema_relation_issues
    )
    return _dedupe_schema_issues(issues)


def _schema_issue_keys(issues: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(issue.get("edge_id", "")), str(issue.get("issue_kind", "")))
        for issue in issues
    }


def _schema_issue_payload(
    edge: SnapshotEdge,
    *,
    issue_kind: str,
    candidate_predicates: list[str],
    suggested_action: str = "replace_relation",
    new_source: str = "",
    new_target: str = "",
    guidance: str = "",
    medical_subcase: str = "",
    source_type: str = "",
    target_type: str = "",
    predicate_tokens: list[str] | None = None,
    repair_options: list[dict[str, Any]] | None = None,
    validation_errors: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "issue_kind": issue_kind,
        "edge_id": edge.id,
        "source": edge.source,
        "source_type": source_type,
        "target": edge.target,
        "target_type": target_type,
        "keywords": edge.keywords,
        "source_id": edge.source_id,
        "file_path": edge.file_path,
        "evidence_quote": _evidence_quote(edge),
        "suggested_action": suggested_action,
        "candidate_predicates": candidate_predicates,
        "new_source": new_source,
        "new_target": new_target,
        "guidance": guidance,
        "qualifiers": _edge_qualifiers(edge),
        "repair_options": repair_options or [],
    }
    if medical_subcase:
        payload["medical_subcase"] = medical_subcase
    if predicate_tokens is not None:
        payload["predicate_tokens"] = predicate_tokens
    if validation_errors:
        payload["validation_errors"] = validation_errors
    return payload


def _evidence_quote(edge: SnapshotEdge) -> str:
    for key in ("evidence_quote", "quote", "source_quote", "text_span"):
        value = edge.properties.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return edge.description.strip()


def _split_repair_options(
    edge: SnapshotEdge,
    *,
    candidate_predicates: list[str],
    source_type: str,
    target_type: str,
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    qualifiers = _edge_qualifiers(edge)
    for predicate in candidate_predicates:
        validation_errors = validate_relation_instance(
            predicate=predicate,
            source_type=source_type,
            target_type=target_type,
            qualifiers=qualifiers,
        )
        options.append(
            {
                "predicate": predicate,
                "new_source": edge.source,
                "new_target": edge.target,
                "source_type": source_type,
                "target_type": target_type,
                "qualifiers": deepcopy(qualifiers),
                "auto_fixable": not validation_errors,
                "validation_errors": validation_errors,
            }
        )
    return options


def _dedupe_schema_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for issue in issues:
        key = _schema_issue_identity(issue)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def _schema_issue_identity(issue: dict[str, Any]) -> str:
    stable_issue = {
        key: value
        for key, value in issue.items()
        if key not in {"issue_ref", "issue_order", "issue_source_index"}
    }
    return json.dumps(
        stable_issue,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _legacy_medical_subcase(rule: MedicalRelationMigrationRule) -> str:
    canonical_options = set(rule.canonical_options)
    if DIAGNOSTIC_SCHEMA_PREDICATES & canonical_options:
        return "diagnostic_evidence_flattening"
    if ADVERSE_REACTION_SCHEMA_PREDICATES & canonical_options:
        return "adverse_reaction_role_conflict"
    return ""


def _entity_cleanup_issue_details(
    *,
    value_like_nodes: list[SnapshotNode],
    synonym_duplicate_issues: list[dict[str, Any]],
    edge_ids_by_node: dict[str, list[str]],
) -> list[dict[str, Any]]:
    issues = [
        *[
            {
                "issue_kind": "value_node_to_qualifier",
                "suggested_action": "convert_to_qualifier",
                "node_id": node.id,
                "label": node.label,
                "entity_type": node.entity_type,
                "qualifier_value": node.label or node.id,
                "connected_edge_ids": edge_ids_by_node.get(node.id, []),
                "source_id": node.source_id,
                "file_path": node.file_path,
                "evidence_quote": _node_evidence_quote(node),
                "candidate_predicates": [],
                "repair_options": [],
            }
            for node in value_like_nodes
        ],
        *synonym_duplicate_issues,
    ]
    return [_entity_cleanup_contract_issue(issue) for issue in issues]


def _node_evidence_quote(node: SnapshotNode) -> str:
    for key in ("evidence_quote", "quote", "source_quote", "text_span"):
        value = node.properties.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return node.description.strip()


def _entity_cleanup_contract_issue(issue: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(issue)
    normalized.setdefault("candidate_predicates", [])
    normalized.setdefault("repair_options", [])
    normalized.setdefault("evidence_quote", "")
    if "source_id" not in normalized or "file_path" not in normalized:
        for node in normalized.get("nodes", []):
            if not isinstance(node, dict):
                continue
            normalized.setdefault("source_id", node.get("source_id", ""))
            normalized.setdefault("file_path", node.get("file_path", ""))
            break
    normalized.setdefault("source_id", "")
    normalized.setdefault("file_path", "")
    return normalized


def _edge_ids_by_node(edges: list[SnapshotEdge]) -> dict[str, list[str]]:
    edge_ids_by_node: dict[str, list[str]] = {}
    for edge in edges:
        edge_ids_by_node.setdefault(edge.source, []).append(edge.id)
        edge_ids_by_node.setdefault(edge.target, []).append(edge.id)
    return edge_ids_by_node


def _synonym_duplicate_issues(nodes: list[SnapshotNode]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[SnapshotNode]] = {}
    for node in nodes:
        canonical_label = canonical_name(node.label or node.id)
        if not canonical_label:
            continue
        key = (_normalize_identifier(node.entity_type), canonical_label)
        grouped.setdefault(key, []).append(node)

    issues: list[dict[str, Any]] = []
    for (entity_type, canonical_label), duplicate_nodes in sorted(grouped.items()):
        node_ids = sorted(node.id for node in duplicate_nodes)
        if len(node_ids) < 2:
            continue
        issues.append(
            {
                "issue_kind": "synonym_duplicate",
                "suggested_action": "merge_synonym_nodes",
                "canonical_label": canonical_label,
                "entity_type": entity_type,
                "node_ids": node_ids,
                "nodes": [
                    {
                        "node_id": node.id,
                        "label": node.label,
                        "source_id": node.source_id,
                        "file_path": node.file_path,
                    }
                    for node in sorted(
                        duplicate_nodes, key=lambda duplicate_node: duplicate_node.id
                    )
                ],
            }
        )
    return issues


def _diagnostic_evidence_flattening_count(
    legacy_schema_relation_issues: list[
        tuple[SnapshotEdge, MedicalRelationMigrationRule]
    ],
) -> int:
    return sum(
        1
        for _edge, rule in legacy_schema_relation_issues
        if DIAGNOSTIC_SCHEMA_PREDICATES & set(rule.canonical_options)
    )


def _adverse_reaction_role_conflict_count(
    legacy_schema_relation_issues: list[
        tuple[SnapshotEdge, MedicalRelationMigrationRule]
    ],
) -> int:
    return sum(
        1
        for _edge, rule in legacy_schema_relation_issues
        if ADVERSE_REACTION_SCHEMA_PREDICATES & set(rule.canonical_options)
    )


def _relation_issue_payload(edge: SnapshotEdge) -> dict[str, str]:
    return {
        "edge_id": edge.id,
        "source": edge.source,
        "target": edge.target,
        "keywords": edge.keywords,
    }


def _legacy_relation_issue_payload(
    edge: SnapshotEdge, rule: MedicalRelationMigrationRule
) -> dict[str, str]:
    return {
        **_relation_issue_payload(edge),
        "canonical_options": " | ".join(rule.canonical_options),
        "guidance": rule.guidance,
    }


def _missing_evidence_examples(
    *,
    missing_node_sources: list[SnapshotNode],
    missing_node_file_paths: list[SnapshotNode],
    missing_edge_sources: list[SnapshotEdge],
    missing_edge_file_paths: list[SnapshotEdge],
    limit_per_field: int = 3,
) -> list[str]:
    examples: list[str] = []
    examples.extend(
        f"node:{node.id} missing source_id"
        for node in missing_node_sources[:limit_per_field]
    )
    examples.extend(
        f"node:{node.id} missing file_path"
        for node in missing_node_file_paths[:limit_per_field]
    )
    examples.extend(
        f"edge:{edge.id} missing source_id"
        for edge in missing_edge_sources[:limit_per_field]
    )
    examples.extend(
        f"edge:{edge.id} missing file_path"
        for edge in missing_edge_file_paths[:limit_per_field]
    )
    return examples


def _weighted_average(subscores: dict[str, int]) -> int:
    total = sum(subscores[name] * weight for name, weight in QUALITY_WEIGHTS.items())
    return _clamp_score(round(total / sum(QUALITY_WEIGHTS.values())))


def _clamp_score(value: int) -> int:
    return max(0, min(100, int(value)))


def _quality_report(score: QualityScore) -> str:
    lines = [
        "# KB Iteration Quality Report",
        "",
        f"Overall score: {score.overall}",
        "",
        "## Subscores",
    ]
    for name in sorted(score.subscores):
        lines.append(f"- {name}: {score.subscores[name]}")

    lines.extend(["", "## Metrics"])
    for name in sorted(score.metrics):
        lines.append(f"- {name}: {score.metrics[name]}")

    hierarchy_branches = score.details.get("hierarchy_branches")
    if hierarchy_branches:
        lines.extend(["", "## Hierarchy Branches"])
        for status in ("required", "present", "missing"):
            branches = hierarchy_branches[status]
            lines.append(f"- {status}: {len(branches)}")
            for branch in branches:
                line = f"  - {branch['key']} | {branch['label']}"
                matched_node_ids = branch.get("matched_node_ids")
                if matched_node_ids:
                    line += f" | matched: {', '.join(matched_node_ids)}"
                lines.append(line)

    lines.extend(["", "## Findings"])
    if score.findings:
        for finding in score.findings:
            evidence = ", ".join(finding.evidence)
            lines.append(
                "- "
                f"{finding.severity} | {finding.category} | {finding.message} "
                f"| evidence: {evidence} "
                f"| suggested_fix_type: {finding.suggested_fix_type} "
                f"| requires_approval: {finding.requires_approval}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Critical Blockers"])
    if score.critical_blockers:
        for blocker in score.critical_blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
