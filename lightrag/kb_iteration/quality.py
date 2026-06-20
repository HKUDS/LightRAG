from __future__ import annotations

import json
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
    migration_rule_for_legacy_keyword,
)
from .models import KGSnapshot, QualityFinding, QualityScore, SnapshotEdge, SnapshotNode

GENERIC_RELATION_KEYWORDS = frozenset({"\u76f8\u5173", "\u90bb\u63a5", "related"})
CLINICAL_MANIFESTATION_KEYWORDS = frozenset(
    {
        "clinical_manifestation",
        "clinical manifestation",
        "manifestation",
        "symptom_of",
        "has_symptom",
        "\u4e34\u5e8a\u8868\u73b0",
        "\u75c7\u72b6",
        "\u75c7\u72b6\u5f52\u7c7b",
        "\u8868\u73b0\u4e3a",
    }
)
TAXONOMY_RELATION_KEYWORDS = frozenset(
    {"belongs_to", "is_a", "type_of", "\u5c5e\u4e8e"}
)
DISEASE_ENTITY_TYPES = frozenset({"disease", "\u75be\u75c5"})
SYMPTOM_ENTITY_TYPES = frozenset(
    {"symptom", "sign", "clinical_manifestation", "\u75c7\u72b6", "\u4f53\u5f81"}
)
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
    medical_schema_issues = _medical_schema_issue_details(
        clinical_direction_issues=clinical_direction_issues,
        taxonomy_relation_misuses=taxonomy_relation_misuses,
        legacy_schema_relation_issues=legacy_schema_relation_issues,
    )
    legacy_schema_edges = [edge for edge, _rule in legacy_schema_relation_issues]
    relation_semantic_issue_edges = _unique_edges_by_id(
        [
            *generic_edges,
            *clinical_direction_issues,
            *taxonomy_relation_misuses,
            *legacy_schema_edges,
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
    normalized = keywords.replace("\uff0c", ",").replace(GRAPH_FIELD_SEP, ",")
    return [token.strip().casefold() for token in normalized.split(",") if token.strip()]


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
    return issues


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
    return node is not None and _normalize_identifier(node.entity_type) in DISEASE_ENTITY_TYPES


def _is_symptom_node(node: SnapshotNode | None) -> bool:
    return node is not None and _normalize_identifier(node.entity_type) in SYMPTOM_ENTITY_TYPES


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
    relation_semantics = _clamp_score(
        100 - round(100 * metrics["generic_relation_count"] / edge_count)
    )
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
    clinical_direction_issues: list[SnapshotEdge],
    taxonomy_relation_misuses: list[SnapshotEdge],
    legacy_schema_relation_issues: list[
        tuple[SnapshotEdge, MedicalRelationMigrationRule]
    ],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
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
    return issues


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
) -> dict[str, Any]:
    payload = {
        "issue_kind": issue_kind,
        "edge_id": edge.id,
        "source": edge.source,
        "target": edge.target,
        "keywords": edge.keywords,
        "source_id": edge.source_id,
        "file_path": edge.file_path,
        "suggested_action": suggested_action,
        "candidate_predicates": candidate_predicates,
        "new_source": new_source,
        "new_target": new_target,
        "guidance": guidance,
    }
    if medical_subcase:
        payload["medical_subcase"] = medical_subcase
    return payload


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
    return [
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
            }
            for node in value_like_nodes
        ],
        *synonym_duplicate_issues,
    ]


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
