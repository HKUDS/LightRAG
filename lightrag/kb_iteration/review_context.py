from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP

GENERIC_RELATION_KEYWORDS = {"", "相关", "邻接", "related", "neighbor", "adjacent"}


def build_review_context(
    package_dir: str | Path, *, round_id: str, focus: list[str]
) -> dict[str, Any]:
    package_path = Path(package_dir)
    snapshot = _read_json(package_path / "snapshots" / "kg_snapshot.json", default={})
    quality = _read_json(package_path / "snapshots" / "quality_score.json", default={})

    nodes = _dict_items(snapshot.get("nodes"))
    edges = _dict_items(snapshot.get("edges"))
    findings = _quality_findings_for_focus(_dict_items(quality.get("findings")), focus)
    selected_edges = _select_edges(edges, focus=focus, quality_findings=findings)
    selected_nodes = _select_nodes(nodes, selected_edges)

    return {
        "round_id": round_id,
        "focus": focus,
        "quality_findings": findings,
        "entities": selected_nodes,
        "relations": selected_edges,
        "evidence_windows": _evidence_windows(selected_nodes, selected_edges),
        "rules_memory": {
            "accepted_changes": _read_text(package_path / "accepted_changes.md"),
            "rejected_changes": _read_text(package_path / "rejected_changes.md"),
        },
    }


def write_review_context(
    package_dir: str | Path, round_id: str, context: dict[str, Any]
) -> Path:
    output_path = Path(package_dir) / "review_context" / f"{round_id}-context.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(context, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path


def _select_edges(
    edges: list[dict[str, Any]],
    *,
    focus: list[str],
    quality_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_focus = {_normalize_text(item) for item in focus}
    if "generic_relation" in normalized_focus:
        return [edge for edge in edges if _is_generic_relation(edge)]

    evidence_edge_ids = _evidence_edge_ids(quality_findings)
    if evidence_edge_ids:
        return [
            edge
            for edge in edges
            if str(edge.get("id", "")) in evidence_edge_ids
        ]

    return edges[:10]


def _quality_findings_for_focus(
    findings: list[dict[str, Any]], focus: list[str]
) -> list[dict[str, Any]]:
    if not focus:
        return findings

    focus_terms = {_normalize_text(item) for item in focus}
    focus_terms.update({term.replace("_", " ") for term in focus_terms})
    return [
        finding
        for finding in findings
        if any(
            term in _normalize_text(finding.get(field, ""))
            for term in focus_terms
            for field in ("category", "suggested_fix_type", "message")
        )
    ]


def _select_nodes(
    nodes: list[dict[str, Any]], selected_edges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected_ids = {
        str(edge.get(endpoint, ""))
        for edge in selected_edges
        for endpoint in ("source", "target")
        if edge.get(endpoint) is not None
    }
    return [node for node in nodes if str(node.get("id", "")) in selected_ids]


def _evidence_windows(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    windows = []
    seen = set()
    for item_type, items in (("entity", nodes), ("relation", edges)):
        for item in items:
            key = (
                item.get("source_id", ""),
                item.get("file_path", ""),
                item_type,
                item.get("id", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            windows.append(
                {
                    "item_type": item_type,
                    "item_id": item.get("id", ""),
                    "source_id": item.get("source_id", ""),
                    "file_path": item.get("file_path", ""),
                }
            )
    return windows


def _evidence_edge_ids(findings: list[dict[str, Any]]) -> set[str]:
    edge_ids = set()
    for finding in findings:
        evidence = finding.get("evidence", [])
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            evidence_id = str(item).strip()
            if not evidence_id.startswith("edge:"):
                continue
            evidence_id = evidence_id.removeprefix("edge:")
            evidence_id = evidence_id.split(maxsplit=1)[0]
            if evidence_id:
                edge_ids.add(evidence_id)
    return edge_ids


def _is_generic_relation(edge: dict[str, Any]) -> bool:
    tokens = _relation_tokens(edge.get("keywords", ""))
    if not tokens:
        return True
    return any(token in GENERIC_RELATION_KEYWORDS for token in tokens)


def _relation_tokens(keywords: Any) -> list[str]:
    normalized = str(keywords).replace("\uff0c", ",").replace(GRAPH_FIELD_SEP, ",")
    return [token.strip().casefold() for token in normalized.split(",") if token.strip()]


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


def _normalize_text(value: Any) -> str:
    return str(value).strip().casefold()
