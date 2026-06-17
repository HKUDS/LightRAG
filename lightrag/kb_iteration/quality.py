from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.medical_kg.ontology import (
    TOP_LEVEL_MEDICAL_CATEGORIES,
    is_value_like_entity,
)

from .models import KGSnapshot, QualityFinding, QualityScore, SnapshotEdge, SnapshotNode

GENERIC_RELATION_KEYWORDS = frozenset({"\u76f8\u5173", "\u90bb\u63a5", "related"})
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

    hierarchy = _hierarchy_metrics(snapshot)
    disease_hub_overload_ratio = _disease_hub_overload_ratio(snapshot)

    metrics = {
        "value_like_node_count": len(value_like_nodes),
        "generic_relation_count": len(generic_edges),
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


def _has_provenance(value: str) -> bool:
    return any(part.strip() for part in value.split(GRAPH_FIELD_SEP))


def _hierarchy_metrics(snapshot: KGSnapshot) -> dict[str, int]:
    required_identifiers = _medical_category_identifiers()
    if not _needs_medical_hierarchy(snapshot):
        return {"required": 0, "present": 0, "missing": 0}

    present = sum(
        1
        for identifiers in required_identifiers.values()
        if any(
            node_identifier in identifiers
            for node in snapshot.nodes
            for node_identifier in _node_hierarchy_identifiers(node)
        )
    )
    required = len(required_identifiers)
    return {"required": required, "present": present, "missing": required - present}


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

    return findings


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
