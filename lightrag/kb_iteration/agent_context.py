from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP

from .memory import AGENT_MEMORY_SUMMARY_FILE, refresh_agent_memory_summary

AGENT_CONTEXT_DIR = "agent_context"
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
MAX_AGENT_SCHEMA_ISSUES = 24
MAX_AGENT_CLEANUP_ISSUES = 40
MAX_AGENT_QUALITY_FINDINGS = 12
MAX_AGENT_FINDING_EVIDENCE = 4
MAX_AGENT_FINDING_MESSAGE_CHARS = 240
MAX_AGENT_FINDING_EVIDENCE_CHARS = 200
MAX_AGENT_CANDIDATE_RELATIONS = 20
MAX_AGENT_CANDIDATE_ENTITIES = 40
MAX_AGENT_EVIDENCE_WINDOWS = 40
MAX_AGENT_EVIDENCE_RELATION_WINDOWS = 20
MAX_AGENT_HIERARCHY_BRANCHES = 40
MAX_STAGE_RULES_MEMORY_LEGACY_CHARS = 1200
MAX_STAGE_RULES_MEMORY_SUMMARY_CHARS = 2400
MAX_PROPOSAL_REVISION_REQUESTS = 20
MAX_PROPOSAL_REVISION_REQUEST_CHARS = 240
MAX_PREVIOUS_OUTPUT_ITEMS = 20
MAX_PREVIOUS_OUTPUT_STRING_CHARS = 1000
MAX_COMPACT_STRING_CHARS = 600
MAX_OBSERVATION_SOURCE_FILES = 40
MAX_OBSERVATION_KB_CONTEXT_CHARS = 5000
MAX_OBSERVATION_MEMORY_CHARS = 1200
MAX_OBSERVATION_MEMORY_SUMMARY_CHARS = 2400
MAX_OBSERVATION_ITEMS = 8
MAX_OBSERVATION_STRING_CHARS = 240
MAX_OBSERVATION_QUALITY_FINDINGS = 8
MAX_OBSERVATION_CRITICAL_BLOCKERS = 8
_AGENT_ISSUE_CONTEXT_FIELDS = (
    "issue_kind",
    "edge_id",
    "node_id",
    "canonical_label",
    "entity_type",
    "node_ids",
    "nodes",
    "connected_edge_ids",
    "source",
    "target",
    "keywords",
    "source_id",
    "file_path",
    "suggested_action",
    "candidate_predicates",
    "medical_subcase",
    "new_source",
    "new_target",
    "value_node_id",
    "carrier_edge_source",
    "carrier_edge_target",
    "qualifier_key",
    "qualifier_value",
)


def safe_package_child_path(package_dir: str | Path, relative_path: str) -> Path:
    package_path = Path(package_dir)
    package_root = package_path.resolve(strict=False)
    candidate = package_path / relative_path
    resolved_candidate = candidate.resolve(strict=False)
    if not _is_relative_to(resolved_candidate, package_root):
        raise ValueError("generated artifact path escapes KB iteration package")
    return candidate


def remove_agent_context_link(package_dir: str | Path) -> None:
    context_dir = Path(package_dir) / AGENT_CONTEXT_DIR
    if _is_link_or_junction(context_dir):
        _remove_link(context_dir)


def ensure_safe_agent_context_dir(package_dir: str | Path) -> Path:
    remove_agent_context_link(package_dir)
    context_dir = safe_package_child_path(package_dir, AGENT_CONTEXT_DIR)
    context_dir.mkdir(parents=True, exist_ok=True)
    return context_dir


def build_agent_observation(
    package_dir: str | Path, *, workspace: str
) -> dict[str, Any]:
    package_path = Path(package_dir)
    snapshot = _read_json(package_path / "snapshots" / "kg_snapshot.json", default={})
    quality = _read_json(package_path / "snapshots" / "quality_score.json", default={})
    nodes = _dict_items(snapshot.get("nodes"))
    edges = _dict_items(snapshot.get("edges"))
    rules_memory = _observation_rules_memory(package_path)

    return {
        "workspace": workspace,
        "generated_at": snapshot.get("generated_at", ""),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "source_files": _compact_source_files(snapshot.get("source_files")),
        "metadata": _compact_observation_value(snapshot.get("metadata", {})),
        "quality": _quality_summary(quality),
        "hierarchy_branches": _hierarchy_branches(quality),
        "artifact_status": _artifact_status(package_path),
        "kb_context": _clip_text(
            _read_text(package_path / "kb_context.md"),
            MAX_OBSERVATION_KB_CONTEXT_CHARS,
        ),
        "rules_memory": rules_memory,
    }


def build_stage_context(
    package_dir: str | Path,
    *,
    workspace: str,
    stage: str,
    previous_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    package_path = Path(package_dir)
    snapshot = _read_json(package_path / "snapshots" / "kg_snapshot.json", default={})
    quality = _read_json(package_path / "snapshots" / "quality_score.json", default={})
    nodes = _dict_items(snapshot.get("nodes"))
    edges = _dict_items(snapshot.get("edges"))
    quality_findings = _dict_items(quality.get("findings"))
    compact_previous_outputs = _compact_previous_outputs(previous_outputs or {})
    medical_schema_issues, medical_schema_issues_summary = (
        _agent_issue_context(
            quality,
            "medical_schema_issues",
            limit=MAX_AGENT_SCHEMA_ISSUES,
        )
    )
    entity_cleanup_issues, entity_cleanup_issues_summary = _agent_issue_context(
        quality,
        "entity_cleanup_issues",
        limit=MAX_AGENT_CLEANUP_ISSUES,
    )
    entity_cleanup_issue_details = _quality_detail_items(
        quality, "entity_cleanup_issues"
    )
    candidate_entities, candidate_relations = _select_candidates(
        nodes,
        edges,
        previous_outputs=previous_outputs,
        quality_findings=quality_findings,
        entity_cleanup_issues=entity_cleanup_issue_details,
    )

    return {
        "workspace": workspace,
        "stage": stage,
        "quality_findings": _compact_quality_findings(quality_findings),
        "hierarchy_branches": _hierarchy_branches(quality),
        "medical_schema_issues": medical_schema_issues,
        "medical_schema_issues_summary": medical_schema_issues_summary,
        "entity_cleanup_issues": entity_cleanup_issues,
        "entity_cleanup_issues_summary": entity_cleanup_issues_summary,
        "proposal_revision_requests": _proposal_revision_requests(package_path),
        "previous_outputs": compact_previous_outputs,
        "candidate_entities": candidate_entities,
        "candidate_relations": candidate_relations,
        "evidence_windows": _evidence_windows(candidate_entities, candidate_relations),
        "rules_memory": _stage_rules_memory(package_path),
    }


def write_agent_context(
    package_dir: str | Path, stage: str, context: dict[str, Any]
) -> Path:
    context_dir = ensure_safe_agent_context_dir(package_dir)
    output_path = safe_package_child_path(
        package_dir, f"{AGENT_CONTEXT_DIR}/{stage}-context.json"
    )
    if output_path.parent != context_dir:
        raise ValueError("agent context path escapes KB iteration package")
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(context, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path


def _quality_summary(quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": quality.get("overall"),
        "subscores": _compact_observation_value(quality.get("subscores", {})),
        "metrics": _compact_observation_value(quality.get("metrics", {})),
        "findings": _compact_quality_findings(
            _dict_items(quality.get("findings")),
            limit=MAX_OBSERVATION_QUALITY_FINDINGS,
        ),
        "critical_blockers": _compact_observation_value(
            quality.get("critical_blockers", [])
        ),
    }


def _artifact_status(package_path: Path) -> dict[str, dict[str, Any]]:
    return {
        name: _file_status(package_path / name)
        for name in (
            "kb_context.md",
            "accepted_changes.md",
            "rejected_changes.md",
            AGENT_MEMORY_SUMMARY_FILE,
            "snapshots/kg_snapshot.json",
            "snapshots/quality_score.json",
        )
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", lambda: False)
    return (
        path.is_symlink()
        or bool(is_junction())
        or _is_windows_reparse_point(path)
    )


def _remove_link(path: Path) -> None:
    try:
        path.unlink()
    except (IsADirectoryError, PermissionError):
        path.rmdir()


def _is_windows_reparse_point(path: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        file_attributes = os.stat(path, follow_symlinks=False).st_file_attributes
    except (FileNotFoundError, OSError, AttributeError):
        return False
    return bool(file_attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _file_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "size_bytes": 0}
    return {"exists": True, "size_bytes": path.stat().st_size}


def _compact_source_files(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        _clip_text(str(item), MAX_OBSERVATION_STRING_CHARS)
        for item in value[:MAX_OBSERVATION_SOURCE_FILES]
    ]


def _rules_memory(package_path: Path) -> dict[str, str]:
    summary_path = refresh_agent_memory_summary(package_path)
    return {
        "accepted_changes": _read_text(package_path / "accepted_changes.md"),
        "rejected_changes": _read_text(package_path / "rejected_changes.md"),
        "proposal_revision_requests": _read_text(
            package_path / "proposal_revision_requests.md"
        ),
        "agent_memory_summary": _read_text(summary_path),
    }


def _observation_rules_memory(package_path: Path) -> dict[str, str]:
    summary_path = refresh_agent_memory_summary(package_path)
    return {
        "accepted_changes": _clip_text(
            _read_text(package_path / "accepted_changes.md"),
            MAX_OBSERVATION_MEMORY_CHARS,
        ),
        "rejected_changes": _clip_text(
            _read_text(package_path / "rejected_changes.md"),
            MAX_OBSERVATION_MEMORY_CHARS,
        ),
        "proposal_revision_requests": _clip_text(
            _read_text(package_path / "proposal_revision_requests.md"),
            MAX_OBSERVATION_MEMORY_CHARS,
        ),
        "agent_memory_summary": _clip_text(
            _read_text(summary_path),
            MAX_OBSERVATION_MEMORY_SUMMARY_CHARS,
        ),
    }


def _stage_rules_memory(package_path: Path) -> dict[str, str]:
    summary_path = refresh_agent_memory_summary(package_path)
    return {
        "accepted_changes": _clip_text(
            _read_text(package_path / "accepted_changes.md"),
            MAX_STAGE_RULES_MEMORY_LEGACY_CHARS,
        ),
        "rejected_changes": _clip_text(
            _read_text(package_path / "rejected_changes.md"),
            MAX_STAGE_RULES_MEMORY_LEGACY_CHARS,
        ),
        "proposal_revision_requests": _clip_text(
            _read_text(package_path / "proposal_revision_requests.md"),
            MAX_STAGE_RULES_MEMORY_LEGACY_CHARS,
        ),
        "agent_memory_summary": _clip_text(
            _read_text(summary_path),
            MAX_STAGE_RULES_MEMORY_SUMMARY_CHARS,
        ),
    }


def _select_candidates(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    previous_outputs: dict[str, Any] | None,
    quality_findings: list[dict[str, Any]] | None = None,
    entity_cleanup_issues: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_edges = _candidate_edges_for_issue_context(
        edges,
        quality_findings=quality_findings or [],
        entity_cleanup_issues=entity_cleanup_issues or [],
    )
    if selected_edges:
        selected_edges = selected_edges[:MAX_AGENT_CANDIDATE_RELATIONS]
        return (
            _compact_candidate_nodes(_nodes_for_edges(nodes, selected_edges)),
            _compact_candidate_edges(selected_edges),
        )

    selected_nodes = _nodes_for_missing_branches(
        nodes, _missing_branches(previous_outputs or {})
    )
    selected_nodes = selected_nodes[:MAX_AGENT_CANDIDATE_ENTITIES]
    selected_node_ids = {str(node.get("id", "")) for node in selected_nodes}
    selected_edges = [
        edge
        for edge in edges
        if str(edge.get("source", "")) in selected_node_ids
        or str(edge.get("target", "")) in selected_node_ids
    ]
    if not selected_nodes and not selected_edges:
        return (
            _compact_candidate_nodes(nodes[:MAX_AGENT_CANDIDATE_ENTITIES]),
            _compact_candidate_edges(edges[:MAX_AGENT_CANDIDATE_RELATIONS]),
        )
    return (
        _compact_candidate_nodes(selected_nodes),
        _compact_candidate_edges(selected_edges[:MAX_AGENT_CANDIDATE_RELATIONS]),
    )


def _compact_candidate_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _compact_fields(
            node,
            ("id", "node_id", "label", "entity_type", "source_id", "file_path"),
        )
        for node in nodes[:MAX_AGENT_CANDIDATE_ENTITIES]
    ]


def _compact_candidate_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _compact_fields(
            edge,
            ("id", "source", "target", "keywords", "source_id", "file_path"),
        )
        for edge in edges[:MAX_AGENT_CANDIDATE_RELATIONS]
    ]


def _compact_fields(
    item: dict[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    return {
        field: _compact_value(item[field])
        for field in fields
        if field in item
    }


def _edges_for_quality_findings(
    edges: list[dict[str, Any]], quality_findings: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    edge_ids = {str(edge.get("id", "")) for edge in edges if edge.get("id") is not None}
    selected_edge_ids = _quality_finding_edge_ids(quality_findings, edge_ids)
    if not selected_edge_ids:
        return []
    return [
        edge
        for edge in edges
        if str(edge.get("id", "")) in selected_edge_ids
    ]


def _candidate_edges_for_issue_context(
    edges: list[dict[str, Any]],
    *,
    quality_findings: list[dict[str, Any]],
    entity_cleanup_issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    edge_ids = {str(edge.get("id", "")) for edge in edges if edge.get("id") is not None}
    selected_edge_ids = _quality_finding_edge_ids(quality_findings, edge_ids)
    selected_edge_ids.update(
        _entity_cleanup_connected_edge_ids(entity_cleanup_issues, edge_ids)
    )
    if not selected_edge_ids:
        return []
    return [
        edge
        for edge in edges
        if str(edge.get("id", "")) in selected_edge_ids
    ]


def _quality_finding_edge_ids(
    quality_findings: list[dict[str, Any]], edge_ids: set[str]
) -> set[str]:
    selected_edge_ids = set()
    for finding in quality_findings:
        evidence = finding.get("evidence", [])
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            evidence_text = str(item).strip()
            if evidence_text.startswith("edge:"):
                evidence_text = evidence_text.removeprefix("edge:")
            evidence_id = evidence_text.split(maxsplit=1)[0]
            if evidence_id in edge_ids:
                selected_edge_ids.add(evidence_id)
    return selected_edge_ids


def _entity_cleanup_connected_edge_ids(
    entity_cleanup_issues: list[dict[str, Any]], edge_ids: set[str]
) -> set[str]:
    selected_edge_ids = set()
    for issue in entity_cleanup_issues:
        connected_edge_ids = issue.get("connected_edge_ids", [])
        if not isinstance(connected_edge_ids, list):
            continue
        for item in connected_edge_ids[:MAX_PREVIOUS_OUTPUT_ITEMS]:
            edge_id = str(item)
            if edge_id in edge_ids:
                selected_edge_ids.add(edge_id)
    return selected_edge_ids


def _nodes_for_edges(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected_node_ids = {
        str(edge.get(endpoint, ""))
        for edge in edges
        for endpoint in ("source", "target")
        if edge.get(endpoint) is not None
    }
    return [node for node in nodes if str(node.get("id", "")) in selected_node_ids]


def _nodes_for_missing_branches(
    nodes: list[dict[str, Any]], branches: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not branches:
        return []
    branch_keys = {
        str(branch.get("key", "")).casefold() for branch in branches if branch.get("key")
    }
    branch_labels = {
        str(branch.get("label", "")).casefold()
        for branch in branches
        if branch.get("label")
    }
    return [
        node
        for node in nodes
        if str(node.get("entity_type", "")).casefold() in branch_keys
        or str(node.get("label", "")).casefold() in branch_labels
    ]


def _missing_branches(previous_outputs: dict[str, Any]) -> list[dict[str, Any]]:
    return _dict_items(previous_outputs.get("missing_branches"))


def _evidence_windows(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    windows = []
    for item_type, items in (("entity", nodes), ("relation", edges)):
        for item in items:
            source_id = item.get("source_id", "")
            file_path = item.get("file_path", "")
            windows.append(
                {
                    "item_type": item_type,
                    "item_id": item.get("id", ""),
                    "evidence_status": (
                        "grounded"
                        if _has_provenance(source_id) and _has_provenance(file_path)
                        else "missing"
                    ),
                }
            )
    if len(windows) <= MAX_AGENT_EVIDENCE_WINDOWS:
        return windows

    relation_windows = [
        window for window in windows if window["item_type"] == "relation"
    ][:MAX_AGENT_EVIDENCE_RELATION_WINDOWS]
    entity_window_limit = MAX_AGENT_EVIDENCE_WINDOWS - len(relation_windows)
    entity_windows = [
        window for window in windows if window["item_type"] == "entity"
    ][:entity_window_limit]
    return relation_windows + entity_windows


def _has_provenance(value: Any) -> bool:
    if value is None:
        return False
    return any(part.strip() for part in str(value).split(GRAPH_FIELD_SEP))


def _hierarchy_branches(quality: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    details = quality.get("details", {})
    if not isinstance(details, dict):
        details = {}
    branches = details.get("hierarchy_branches", {})
    if not isinstance(branches, dict):
        branches = {}
    return {
        key: _compact_hierarchy_branches(_dict_items(branches.get(key)))
        for key in ("required", "present", "missing")
    }


def _compact_hierarchy_branches(branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _compact_fields(
            branch,
            ("key", "label", "aliases", "matched_node_ids", "missing"),
        )
        for branch in branches[:MAX_AGENT_HIERARCHY_BRANCHES]
    ]


def _quality_detail_items(quality: dict[str, Any], key: str) -> list[dict[str, Any]]:
    details = quality.get("details", {})
    if not isinstance(details, dict):
        return []
    return _dict_items(details.get(key))


def _agent_issue_context(
    quality: dict[str, Any],
    key: str,
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues = _quality_detail_items(quality, key)
    visible = [_compact_agent_issue(item) for item in issues[:limit]]
    return visible, _issue_summary(issues, visible)


def _compact_agent_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        field: _compact_agent_issue_value(field, issue[field])
        for field in _AGENT_ISSUE_CONTEXT_FIELDS
        if field in issue
    }


def _compact_agent_issue_value(field: str, value: Any) -> Any:
    if field == "nodes" and isinstance(value, list):
        return [
            _compact_candidate_nodes([node])[0]
            for node in value[:10]
            if isinstance(node, dict)
        ]
    if field == "node_ids" and isinstance(value, list):
        return [
            _clip_text(str(item), MAX_COMPACT_STRING_CHARS)
            for item in value[:MAX_PREVIOUS_OUTPUT_ITEMS]
        ]
    return _compact_value(value)


def _issue_summary(
    issues: list[dict[str, Any]],
    visible: list[dict[str, Any]],
) -> dict[str, Any]:
    total_by_kind: dict[str, int] = {}
    visible_by_kind: dict[str, int] = {}
    for issue in issues:
        kind = _issue_kind(issue)
        total_by_kind[kind] = total_by_kind.get(kind, 0) + 1
    for issue in visible:
        kind = _issue_kind(issue)
        visible_by_kind[kind] = visible_by_kind.get(kind, 0) + 1

    return {
        "total": len(issues),
        "visible": len(visible),
        "omitted": max(len(issues) - len(visible), 0),
        "by_issue_kind": {
            kind: {
                "total": total,
                "visible": visible_by_kind.get(kind, 0),
                "omitted": max(total - visible_by_kind.get(kind, 0), 0),
            }
            for kind, total in sorted(total_by_kind.items())
        },
    }


def _issue_kind(issue: dict[str, Any]) -> str:
    kind = issue.get("issue_kind")
    if kind is None:
        return "unknown"
    return str(kind)


def _proposal_revision_requests(package_path: Path) -> list[str]:
    text = _read_text(package_path / "proposal_revision_requests.md")
    if not text.strip():
        return []
    return [
        _clip_text(line.strip(), MAX_PROPOSAL_REVISION_REQUEST_CHARS)
        for line in text.splitlines()
        if line.strip()
    ][:MAX_PROPOSAL_REVISION_REQUESTS]


def _compact_quality_findings(
    findings: list[dict[str, Any]],
    *,
    limit: int = MAX_AGENT_QUALITY_FINDINGS,
) -> list[dict[str, Any]]:
    return [
        _compact_quality_finding(finding)
        for finding in findings[:limit]
    ]


def _compact_quality_finding(finding: dict[str, Any]) -> dict[str, Any]:
    compact = {
        field: _clip_text(str(finding[field]), MAX_AGENT_FINDING_MESSAGE_CHARS)
        for field in (
            "severity",
            "category",
            "message",
            "suggested_fix_type",
        )
        if field in finding
    }
    if "requires_approval" in finding:
        compact["requires_approval"] = bool(finding["requires_approval"])
    evidence = finding.get("evidence")
    if isinstance(evidence, list):
        compact["evidence"] = [
            _clip_text(str(item), MAX_AGENT_FINDING_EVIDENCE_CHARS)
            for item in evidence[:MAX_AGENT_FINDING_EVIDENCE]
        ]
    return compact


def _compact_previous_outputs(previous_outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _compact_value(value)
        for key, value in list(previous_outputs.items())[:MAX_PREVIOUS_OUTPUT_ITEMS]
    }


def _compact_value(value: Any) -> Any:
    if isinstance(value, str):
        return _clip_text(value, MAX_PREVIOUS_OUTPUT_STRING_CHARS)
    if isinstance(value, int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [
            _compact_value(item)
            for item in value[:MAX_PREVIOUS_OUTPUT_ITEMS]
        ]
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item)
            for key, item in list(value.items())[:MAX_PREVIOUS_OUTPUT_ITEMS]
        }
    return _clip_text(str(value), MAX_PREVIOUS_OUTPUT_STRING_CHARS)


def _compact_observation_value(value: Any) -> Any:
    if isinstance(value, str):
        return _clip_text(value, MAX_OBSERVATION_STRING_CHARS)
    if isinstance(value, int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [
            _compact_observation_value(item)
            for item in value[:MAX_OBSERVATION_ITEMS]
        ]
    if isinstance(value, dict):
        return {
            str(key): _compact_observation_value(item)
            for key, item in list(value.items())[:MAX_OBSERVATION_ITEMS]
        }
    return _clip_text(str(value), MAX_OBSERVATION_STRING_CHARS)


def _clip_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3].rstrip() + "..."


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
