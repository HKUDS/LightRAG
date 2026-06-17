from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import KGSnapshot, QualityScore

EdgePair = tuple[str, str]


@dataclass(frozen=True)
class SnapshotDiff:
    workspace: dict[str, str]
    added_nodes: list[str] = field(default_factory=list)
    removed_nodes: list[str] = field(default_factory=list)
    changed_entity_types: dict[str, dict[str, str]] = field(default_factory=dict)
    added_edge_pairs: list[EdgePair] = field(default_factory=list)
    removed_edge_pairs: list[EdgePair] = field(default_factory=list)
    changed_relation_keywords: dict[EdgePair, dict[str, str]] = field(
        default_factory=dict
    )
    dangerous_regression_flags: list[str] = field(default_factory=list)
    quality_delta: dict[str, int | float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_edge_pairs": _edge_pairs_to_dicts(self.added_edge_pairs),
            "added_nodes": self.added_nodes,
            "changed_entity_types": self.changed_entity_types,
            "changed_relation_keywords": _relation_keyword_changes_to_dicts(
                self.changed_relation_keywords
            ),
            "dangerous_regression_flags": self.dangerous_regression_flags,
            "quality_delta": self.quality_delta,
            "removed_edge_pairs": _edge_pairs_to_dicts(self.removed_edge_pairs),
            "removed_nodes": self.removed_nodes,
            "workspace": self.workspace,
        }


def compare_snapshots(
    before: KGSnapshot,
    after: KGSnapshot,
    *,
    before_quality: QualityScore | None = None,
    after_quality: QualityScore | None = None,
    core_disease_node_ids: list[str] | set[str] | tuple[str, ...] | None = None,
) -> SnapshotDiff:
    before_nodes = {node.id: node for node in before.nodes}
    after_nodes = {node.id: node for node in after.nodes}
    before_node_ids = set(before_nodes)
    after_node_ids = set(after_nodes)

    added_nodes = sorted(after_node_ids - before_node_ids)
    removed_nodes = sorted(before_node_ids - after_node_ids)
    common_nodes = before_node_ids & after_node_ids
    changed_entity_types = {
        node_id: {
            "before": before_nodes[node_id].entity_type,
            "after": after_nodes[node_id].entity_type,
        }
        for node_id in sorted(common_nodes)
        if before_nodes[node_id].entity_type != after_nodes[node_id].entity_type
    }

    before_edges = _edge_keywords_by_pair(before)
    after_edges = _edge_keywords_by_pair(after)
    before_pairs = set(before_edges)
    after_pairs = set(after_edges)
    common_pairs = before_pairs & after_pairs

    changed_relation_keywords = {
        pair: {"before": before_edges[pair], "after": after_edges[pair]}
        for pair in sorted(common_pairs)
        if before_edges[pair] != after_edges[pair]
    }

    quality_delta = _quality_delta(before_quality, after_quality)

    return SnapshotDiff(
        workspace={"before": before.workspace, "after": after.workspace},
        added_nodes=added_nodes,
        removed_nodes=removed_nodes,
        changed_entity_types=changed_entity_types,
        added_edge_pairs=sorted(after_pairs - before_pairs),
        removed_edge_pairs=sorted(before_pairs - after_pairs),
        changed_relation_keywords=changed_relation_keywords,
        dangerous_regression_flags=_dangerous_regression_flags(
            before_nodes=before_nodes,
            removed_nodes=removed_nodes,
            before_quality=before_quality,
            after_quality=after_quality,
            core_disease_node_ids=core_disease_node_ids,
        ),
        quality_delta=quality_delta,
    )


def write_diff_report(diff: SnapshotDiff, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    snapshot_dir = output_path / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_path / "diff_report.md"
    summary_path = snapshot_dir / "diff_summary.json"

    _write_json(summary_path, diff.to_dict())
    report_path.write_text(_diff_report(diff), encoding="utf-8")
    return report_path


def _edge_keywords_by_pair(snapshot: KGSnapshot) -> dict[EdgePair, str]:
    keywords_by_pair: dict[EdgePair, set[str]] = {}
    for edge in snapshot.edges:
        pair = (edge.source, edge.target)
        keywords_by_pair.setdefault(pair, set()).add(edge.keywords)
    return {
        pair: ", ".join(sorted(keyword for keyword in keywords if keyword))
        for pair, keywords in sorted(keywords_by_pair.items())
    }


def _quality_delta(
    before_quality: QualityScore | None,
    after_quality: QualityScore | None,
) -> dict[str, int | float]:
    if before_quality is None or after_quality is None:
        return {}

    delta: dict[str, int | float] = {}
    for metric in ("generic_relation_count",):
        before_value = before_quality.metrics.get(metric)
        after_value = after_quality.metrics.get(metric)
        if _is_number(before_value) and _is_number(after_value):
            delta[metric] = after_value - before_value

    before_evidence = _evidence_coverage(before_quality)
    after_evidence = _evidence_coverage(after_quality)
    if before_evidence is not None and after_evidence is not None:
        delta["evidence_coverage"] = after_evidence - before_evidence

    return delta


def _dangerous_regression_flags(
    *,
    before_nodes: dict[str, Any],
    removed_nodes: list[str],
    before_quality: QualityScore | None,
    after_quality: QualityScore | None,
    core_disease_node_ids: list[str] | set[str] | tuple[str, ...] | None,
) -> list[str]:
    flags: list[str] = []
    if core_disease_node_ids is None:
        core_nodes = _disease_node_ids(before_nodes)
    else:
        core_nodes = set(core_disease_node_ids)
    for node_id in sorted(core_nodes & set(removed_nodes)):
        flags.append(f"core_disease_node_removed:{node_id}")

    if before_quality is not None and after_quality is not None:
        before_generic = before_quality.metrics.get("generic_relation_count")
        after_generic = after_quality.metrics.get("generic_relation_count")
        if (
            _is_number(before_generic)
            and _is_number(after_generic)
            and after_generic > before_generic
        ):
            flags.append(
                "generic_relation_count_increased:"
                f"{before_generic}->{after_generic}"
            )

        before_evidence = _evidence_coverage(before_quality)
        after_evidence = _evidence_coverage(after_quality)
        if (
            before_evidence is not None
            and after_evidence is not None
            and after_evidence < before_evidence
        ):
            flags.append(
                "evidence_coverage_decreased:"
                f"{before_evidence}->{after_evidence}"
            )

    return sorted(flags)


def _disease_node_ids(nodes: dict[str, Any]) -> set[str]:
    return {
        node_id
        for node_id, node in nodes.items()
        if node.entity_type.strip().casefold() == "disease"
    }


def _evidence_coverage(score: QualityScore) -> int | float | None:
    value = score.metrics.get("evidence_coverage")
    if _is_number(value):
        return value

    value = score.subscores.get("evidence_grounding")
    if _is_number(value):
        return value
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _diff_report(diff: SnapshotDiff) -> str:
    lines = [
        "# KB Iteration Snapshot Diff Report",
        "",
        "## Workspace",
        f"- before: {diff.workspace['before']}",
        f"- after: {diff.workspace['after']}",
        "",
        "## Added nodes",
        *_list_lines(diff.added_nodes),
        "",
        "## Removed nodes",
        *_list_lines(diff.removed_nodes),
        "",
        "## Changed entity types",
        *_change_lines(diff.changed_entity_types),
        "",
        "## Added edge pairs",
        *_edge_pair_lines(diff.added_edge_pairs),
        "",
        "## Removed edge pairs",
        *_edge_pair_lines(diff.removed_edge_pairs),
        "",
        "## Changed relation keywords",
        *_edge_change_lines(diff.changed_relation_keywords),
        "",
        "## Dangerous regression flags",
        *_list_lines(diff.dangerous_regression_flags),
        "",
        "## Quality delta",
        *_quality_delta_lines(diff.quality_delta),
    ]
    return "\n".join(lines) + "\n"


def _list_lines(values: list[str]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- {value}" for value in values]


def _change_lines(changes: dict[str, dict[str, str]]) -> list[str]:
    if not changes:
        return ["- none"]
    return [
        f"- {node_id}: {change['before']} -> {change['after']}"
        for node_id, change in sorted(changes.items())
    ]


def _edge_pair_lines(pairs: list[EdgePair]) -> list[str]:
    if not pairs:
        return ["- none"]
    return [f"- {_format_edge_pair(pair)}" for pair in pairs]


def _edge_change_lines(changes: dict[EdgePair, dict[str, str]]) -> list[str]:
    if not changes:
        return ["- none"]
    return [
        f"- {_format_edge_pair(pair)}: {change['before']} -> {change['after']}"
        for pair, change in sorted(changes.items())
    ]


def _quality_delta_lines(delta: dict[str, int | float]) -> list[str]:
    if not delta:
        return ["- none"]
    return [f"- {name}: {delta[name]}" for name in sorted(delta)]


def _edge_pairs_to_dicts(pairs: list[EdgePair]) -> list[dict[str, str]]:
    return [{"source": source, "target": target} for source, target in pairs]


def _relation_keyword_changes_to_dicts(
    changes: dict[EdgePair, dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "source": source,
            "target": target,
            "before": change["before"],
            "after": change["after"],
        }
        for (source, target), change in sorted(changes.items())
    ]


def _format_edge_pair(pair: EdgePair) -> str:
    return f"{pair[0]} -> {pair[1]}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
