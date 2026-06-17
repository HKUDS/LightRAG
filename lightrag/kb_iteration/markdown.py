from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP

from .models import KGSnapshot, SnapshotEdge, SnapshotNode

HIERARCHY_KEYWORDS = {
    "属于",
    "症状归类",
    "belongs_to",
    "category",
    "hierarchy",
}
_WHITESPACE_PATTERN = re.compile(r"\s+")
_MARKDOWN_ESCAPE_PATTERN = re.compile(r"([\\`*{}\[\]()#!>|])")
_MARKDOWN_LIST_MARKER_PATTERN = re.compile(r"(^|\s)([+-])(?=\s)")

RULE_MEMORY_TEMPLATES = {
    "quality_rules.md": "# Quality Rules\n\n- Add durable KB quality rules here.\n",
    "known_issues.md": "# Known Issues\n\n- Track unresolved KB issues here.\n",
    "accepted_changes.md": "# Accepted Changes\n\n- Record approved KB changes here.\n",
    "rejected_changes.md": "# Rejected Changes\n\n- Record rejected KB changes here.\n",
    "approval_queue.md": "# Approval Queue\n\n- Queue proposed KB changes for human review here.\n",
    "improvement_backlog.md": "# Improvement Backlog\n\n- Track future KB improvements here.\n",
    "iteration_log.md": "# Iteration Log\n\n- Record KB iteration runs here.\n",
    "diff_report.md": "# Diff Report\n\n- Summarize snapshot-to-snapshot changes here.\n",
}


def write_markdown_memory(
    snapshot: KGSnapshot,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "kb_context": output_path / "kb_context.md",
        "entity_catalog": output_path / "entity_catalog.md",
        "relation_catalog": output_path / "relation_catalog.md",
        "kg_structure": output_path / "kg_structure.md",
    }

    _write_text(paths["kb_context"], _render_kb_context(snapshot))
    _write_text(paths["entity_catalog"], _render_entity_catalog(snapshot))
    _write_text(paths["relation_catalog"], _render_relation_catalog(snapshot))
    _write_text(paths["kg_structure"], _render_kg_structure(snapshot))
    _initialize_rule_memory_files(output_path)

    return paths


def _render_kb_context(snapshot: KGSnapshot) -> str:
    profile = _markdown_text(snapshot.metadata.get("profile") or "Not specified")
    node_count = snapshot.metadata.get("graph_node_count", len(snapshot.nodes))
    edge_count = snapshot.metadata.get("graph_edge_count", len(snapshot.edges))
    score_summary = _markdown_text(
        snapshot.metadata.get("latest_score_summary") or "Not generated yet"
    )

    lines = [
        "# KB Context",
        "",
        "## Snapshot",
        "",
        f"- Workspace: {_markdown_text(snapshot.workspace)}",
        f"- Generated At: {_markdown_text(snapshot.generated_at)}",
        f"- Profile: {profile}",
        "",
        "## Source Files",
        "",
    ]
    lines.extend(_bullet_list(snapshot.source_files, empty="No source files recorded."))
    lines.extend(
        [
            "",
            "## Graph Scale",
            "",
            f"- Nodes: {node_count}",
            f"- Edges: {edge_count}",
            "",
            "## Entity Type Distribution",
            "",
        ]
    )
    lines.extend(_stats_lines(_count_values(node.entity_type for node in snapshot.nodes)))
    lines.extend(["", "## Relation Keyword Distribution", ""])
    lines.extend(_stats_lines(_count_values(edge.keywords for edge in snapshot.edges)))
    lines.extend(
        [
            "",
            "## Latest Score Summary",
            "",
            str(score_summary),
            "",
            "## Detail Files",
            "",
            "- [Entity Catalog](entity_catalog.md)",
            "- [Relation Catalog](relation_catalog.md)",
            "- [KG Structure](kg_structure.md)",
            "- [Quality Rules](quality_rules.md)",
            "- [Known Issues](known_issues.md)",
            "- [Approval Queue](approval_queue.md)",
        ]
    )
    return _join_markdown(lines)


def _render_entity_catalog(snapshot: KGSnapshot) -> str:
    groups: dict[str, list[SnapshotNode]] = defaultdict(list)
    for node in snapshot.nodes:
        groups[_markdown_label_or_unknown(node.entity_type)].append(node)

    lines = ["# Entity Catalog", ""]
    if not groups:
        lines.append("No entities found.")
        return _join_markdown(lines)

    for entity_type, nodes in _sorted_groups(groups):
        lines.extend([f"## {entity_type} ({len(nodes)})", ""])
        for node in sorted(nodes, key=lambda item: (_display_node(item), item.id)):
            lines.append(f"- {_display_node(node)}")
            details = _node_details(node)
            if details:
                lines.extend(f"  - {detail}" for detail in details)
        lines.append("")

    return _join_markdown(lines)


def _render_relation_catalog(snapshot: KGSnapshot) -> str:
    groups: dict[str, list[SnapshotEdge]] = defaultdict(list)
    for edge in snapshot.edges:
        groups[_markdown_label_or_unknown(edge.keywords)].append(edge)

    lines = ["# Relation Catalog", ""]
    if not groups:
        lines.append("No relations found.")
        return _join_markdown(lines)

    for keyword, edges in _sorted_groups(groups):
        lines.extend([f"## {keyword} ({len(edges)})", ""])
        for edge in sorted(
            edges, key=lambda item: (item.source, item.target, item.id)
        ):
            lines.append(
                f"- {_markdown_text(edge.source)} -> {_markdown_text(edge.target)}"
            )
            details = _edge_details(edge)
            if details:
                lines.extend(f"  - {detail}" for detail in details)
        lines.append("")

    return _join_markdown(lines)


def _render_kg_structure(snapshot: KGSnapshot) -> str:
    node_by_id = {node.id: node for node in snapshot.nodes}
    hierarchy_edges = [
        edge
        for edge in snapshot.edges
        if _is_hierarchy_like(edge, node_by_id.get(edge.source), node_by_id.get(edge.target))
    ]

    lines = ["# KG Structure", "", "## Hierarchy-Like Edges", ""]
    if not hierarchy_edges:
        lines.append("No hierarchy-like edges detected.")
        return _join_markdown(lines)

    for edge in sorted(hierarchy_edges, key=lambda item: (item.source, item.target, item.id)):
        lines.append(f"- {_markdown_text(edge.source)} -> {_markdown_text(edge.target)}")
        lines.append(f"  - Keyword: {_markdown_label_or_unknown(edge.keywords)}")
        if edge.description:
            lines.append(f"  - Description: {_markdown_text(edge.description)}")
        source = _source_detail(edge.file_path, edge.source_id)
        if source:
            lines.append(f"  - Source: {source}")

    return _join_markdown(lines)


def _is_hierarchy_like(
    edge: SnapshotEdge,
    source: SnapshotNode | None,
    target: SnapshotNode | None,
) -> bool:
    return (
        any(keyword in HIERARCHY_KEYWORDS for keyword in _keyword_tokens(edge.keywords))
        or _is_medical_group(source)
        or _is_medical_group(target)
    )


def _is_medical_group(node: SnapshotNode | None) -> bool:
    if node is None:
        return False
    return node.entity_type == "MedicalGroup" or bool(node.properties.get("medical_group"))


def _node_details(node: SnapshotNode) -> list[str]:
    details = []
    aliases = node.properties.get("aliases")
    if aliases:
        details.append(f"Aliases: {_format_aliases(aliases)}")
    if node.description:
        details.append(f"Description: {_markdown_text(node.description)}")
    source = _source_detail(node.file_path, node.source_id)
    if source:
        details.append(f"Source: {source}")
    return details


def _edge_details(edge: SnapshotEdge) -> list[str]:
    details = []
    if edge.description:
        details.append(f"Description: {_markdown_text(edge.description)}")
    source = _source_detail(edge.file_path, edge.source_id)
    if source:
        details.append(f"Source: {source}")
    return details


def _initialize_rule_memory_files(output_dir: Path) -> None:
    for filename, content in sorted(RULE_MEMORY_TEMPLATES.items()):
        path = output_dir / filename
        if not path.exists():
            _write_text(path, content)


def _count_values(values: Any) -> Counter[str]:
    return Counter(_markdown_label_or_unknown(value) for value in values)


def _stats_lines(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["No data recorded."]
    return [f"- {label}: {count}" for label, count in _sorted_counter_items(counter)]


def _bullet_list(values: list[str], *, empty: str) -> list[str]:
    if not values:
        return [empty]
    return [f"- {_markdown_text(value)}" for value in sorted(values)]


def _sorted_groups(groups: dict[str, list[Any]]) -> list[tuple[str, list[Any]]]:
    return sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))


def _sorted_counter_items(counter: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def _display_node(node: SnapshotNode) -> str:
    return _markdown_text(node.label or node.id)


def _label_or_unknown(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return text or "Unknown"


def _markdown_label_or_unknown(value: Any) -> str:
    return _markdown_text(_label_or_unknown(value))


def _markdown_text(value: Any, *, empty: str = "Unknown") -> str:
    text = "" if value is None else str(value)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    if not text:
        return empty
    text = _MARKDOWN_ESCAPE_PATTERN.sub(r"\\\1", text)
    return _MARKDOWN_LIST_MARKER_PATTERN.sub(r"\1\\\2", text)


def _format_aliases(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_markdown_text(item) for item in sorted(value, key=str))
    return _markdown_text(value)


def _source_detail(file_path: str, source_id: str) -> str:
    files = _normalized_joined_values(file_path)
    sources = _normalized_joined_values(source_id)
    if files and sources:
        return f"{files} / {sources}"
    return files or sources


def _keyword_tokens(keywords: str) -> list[str]:
    return [keyword.strip() for keyword in keywords.split(",") if keyword.strip()]


def _normalized_joined_values(value: str) -> str:
    values = {part.strip() for part in value.split(GRAPH_FIELD_SEP) if part.strip()}
    return ", ".join(_markdown_text(part) for part in sorted(values))


def _join_markdown(lines: list[str]) -> str:
    return "\n".join(lines).rstrip() + "\n"


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
