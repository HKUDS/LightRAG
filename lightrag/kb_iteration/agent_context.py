from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP

AGENT_CONTEXT_DIR = "agent_context"


def build_agent_observation(
    package_dir: str | Path, *, workspace: str
) -> dict[str, Any]:
    package_path = Path(package_dir)
    snapshot = _read_json(package_path / "snapshots" / "kg_snapshot.json", default={})
    quality = _read_json(package_path / "snapshots" / "quality_score.json", default={})
    nodes = _dict_items(snapshot.get("nodes"))
    edges = _dict_items(snapshot.get("edges"))

    return {
        "workspace": workspace,
        "generated_at": snapshot.get("generated_at", ""),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "source_files": snapshot.get("source_files", []),
        "metadata": snapshot.get("metadata", {}),
        "quality": _quality_summary(quality),
        "hierarchy_branches": _hierarchy_branches(quality),
        "artifact_status": _artifact_status(package_path),
        "kb_context": _read_text(package_path / "kb_context.md"),
        "rules_memory": _rules_memory(package_path),
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
    candidate_entities, candidate_relations = _select_candidates(
        nodes, edges, previous_outputs=previous_outputs
    )

    return {
        "workspace": workspace,
        "stage": stage,
        "quality_findings": _dict_items(quality.get("findings")),
        "hierarchy_branches": _hierarchy_branches(quality),
        "previous_outputs": previous_outputs or {},
        "candidate_entities": candidate_entities,
        "candidate_relations": candidate_relations,
        "evidence_windows": _evidence_windows(candidate_entities, candidate_relations),
        "rules_memory": _rules_memory(package_path),
    }


def write_agent_context(
    package_dir: str | Path, stage: str, context: dict[str, Any]
) -> Path:
    output_path = Path(package_dir) / AGENT_CONTEXT_DIR / f"{stage}-context.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(context, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path


def _quality_summary(quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": quality.get("overall"),
        "subscores": quality.get("subscores", {}),
        "metrics": quality.get("metrics", {}),
        "findings": _dict_items(quality.get("findings")),
        "critical_blockers": quality.get("critical_blockers", []),
    }


def _artifact_status(package_path: Path) -> dict[str, dict[str, Any]]:
    return {
        name: _file_status(package_path / name)
        for name in (
            "kb_context.md",
            "accepted_changes.md",
            "rejected_changes.md",
            "snapshots/kg_snapshot.json",
            "snapshots/quality_score.json",
        )
    }


def _file_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "size_bytes": 0}
    return {"exists": True, "size_bytes": path.stat().st_size}


def _rules_memory(package_path: Path) -> dict[str, str]:
    return {
        "accepted_changes": _read_text(package_path / "accepted_changes.md"),
        "rejected_changes": _read_text(package_path / "rejected_changes.md"),
    }


def _select_candidates(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    previous_outputs: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_nodes = _nodes_for_missing_branches(
        nodes, _missing_branches(previous_outputs or {})
    )
    selected_node_ids = {str(node.get("id", "")) for node in selected_nodes}
    selected_edges = [
        edge
        for edge in edges
        if str(edge.get("source", "")) in selected_node_ids
        or str(edge.get("target", "")) in selected_node_ids
    ]
    if not selected_nodes and not selected_edges:
        return nodes[:20], edges[:20]
    return selected_nodes[:20], selected_edges[:20]


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
                    "source_id": source_id,
                    "file_path": file_path,
                    "evidence_status": (
                        "grounded"
                        if _has_provenance(source_id) and _has_provenance(file_path)
                        else "missing"
                    ),
                }
            )
    return windows


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
        key: _dict_items(branches.get(key))
        for key in ("required", "present", "missing")
    }


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
