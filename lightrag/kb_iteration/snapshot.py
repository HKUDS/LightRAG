from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx

from .models import KGSnapshot, SnapshotEdge, SnapshotNode

NODE_FIELDS = {"label", "entity_type", "description", "source_id", "file_path"}
EDGE_FIELDS = {
    "id",
    "keywords",
    "description",
    "source_id",
    "file_path",
    "weight",
}


def build_snapshot_from_graphml(
    graph_path: str | Path,
    *,
    workspace: str,
    source_files: list[str] | None = None,
    profile: str | None = None,
) -> KGSnapshot:
    graph = nx.read_graphml(graph_path)
    nodes = [_snapshot_node(node_id, data) for node_id, data in graph.nodes(data=True)]
    edges = [_snapshot_edge(edge) for edge in _iter_edges(graph)]

    metadata: dict[str, Any] = {
        "graph_node_count": graph.number_of_nodes(),
        "graph_edge_count": graph.number_of_edges(),
    }
    if profile is not None:
        metadata["profile"] = profile

    return KGSnapshot(
        workspace=workspace,
        generated_at=_utc_timestamp(),
        source_files=sorted(source_files or []),
        nodes=nodes,
        edges=edges,
        metadata=metadata,
    )


def write_snapshot_artifacts(
    snapshot: KGSnapshot,
    output_dir: str | Path,
) -> dict[str, Path]:
    snapshot_dir = Path(output_dir) / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "kg_snapshot": snapshot_dir / "kg_snapshot.json",
        "entity_stats": snapshot_dir / "entity_stats.json",
        "relation_stats": snapshot_dir / "relation_stats.json",
        "hierarchy_paths": snapshot_dir / "hierarchy_paths.json",
        "source_coverage": snapshot_dir / "source_coverage.json",
    }

    _write_json(artifacts["kg_snapshot"], snapshot.to_dict())
    _write_json(artifacts["entity_stats"], _entity_stats(snapshot))
    _write_json(artifacts["relation_stats"], _relation_stats(snapshot))
    _write_json(artifacts["hierarchy_paths"], {"paths": []})
    _write_json(artifacts["source_coverage"], _source_coverage(snapshot))

    return artifacts


def _snapshot_node(node_id: str, data: dict[str, Any]) -> SnapshotNode:
    return SnapshotNode(
        id=str(node_id),
        label=_as_text(data.get("label")),
        entity_type=_as_text(data.get("entity_type")),
        description=_as_text(data.get("description")),
        source_id=_as_text(data.get("source_id")),
        file_path=_as_text(data.get("file_path")),
        properties=_extra_properties(data, NODE_FIELDS),
    )


def _snapshot_edge(edge: tuple[str, str, str, dict[str, Any]]) -> SnapshotEdge:
    source, target, edge_id, data = edge
    return SnapshotEdge(
        id=str(edge_id),
        source=str(source),
        target=str(target),
        keywords=_as_text(data.get("keywords")),
        description=_as_text(data.get("description")),
        source_id=_as_text(data.get("source_id")),
        file_path=_as_text(data.get("file_path")),
        weight=_as_float(data.get("weight")),
        properties=_extra_properties(data, EDGE_FIELDS),
    )


def _iter_edges(graph: nx.Graph) -> list[tuple[str, str, str, dict[str, Any]]]:
    if graph.is_multigraph():
        return [
            (str(source), str(target), str(key), dict(data))
            for source, target, key, data in graph.edges(keys=True, data=True)
        ]

    edges = []
    for source, target, data in graph.edges(data=True):
        edge_id = data.get("id") or f"{source}->{target}"
        edges.append((str(source), str(target), str(edge_id), dict(data)))
    return edges


def _entity_stats(snapshot: KGSnapshot) -> list[dict[str, Any]]:
    counts = Counter(node.entity_type for node in snapshot.nodes)
    return _sorted_stats(counts)


def _relation_stats(snapshot: KGSnapshot) -> list[dict[str, Any]]:
    counts = Counter(edge.keywords for edge in snapshot.edges)
    return _sorted_stats(counts)


def _source_coverage(snapshot: KGSnapshot) -> dict[str, Any]:
    node_files = Counter(node.file_path for node in snapshot.nodes if node.file_path)
    edge_files = Counter(edge.file_path for edge in snapshot.edges if edge.file_path)
    node_sources = Counter(node.source_id for node in snapshot.nodes if node.source_id)
    edge_sources = Counter(edge.source_id for edge in snapshot.edges if edge.source_id)

    return {
        "source_files": snapshot.source_files,
        "file_path_counts": {
            "nodes": dict(_sorted_counter_items(node_files)),
            "edges": dict(_sorted_counter_items(edge_files)),
        },
        "source_id_counts": {
            "nodes": dict(_sorted_counter_items(node_sources)),
            "edges": dict(_sorted_counter_items(edge_sources)),
        },
    }


def _sorted_stats(counts: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"label": label, "count": count}
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _sorted_counter_items(counter: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def _extra_properties(data: dict[str, Any], known_fields: set[str]) -> dict[str, Any]:
    return {str(key): value for key, value in data.items() if key not in known_fields}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
