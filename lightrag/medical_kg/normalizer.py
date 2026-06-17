from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ontology import (
    DroppedNode,
    canonical_name,
    is_value_like_entity,
    normalize_relation_keyword,
)


@dataclass
class NormalizedExtraction:
    nodes: dict[str, list[dict[str, Any]]]
    edges: dict[tuple[str, str], list[dict[str, Any]]]
    dropped_nodes: dict[str, DroppedNode]


def normalize_medical_extraction(
    nodes: dict[str, list[dict[str, Any]]],
    edges: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    enabled: bool,
) -> NormalizedExtraction:
    if not enabled:
        return NormalizedExtraction(nodes=nodes, edges=edges, dropped_nodes={})

    normalized_nodes: dict[str, list[dict[str, Any]]] = {}
    dropped_nodes: dict[str, DroppedNode] = {}
    dropped_node_data: dict[str, list[dict[str, Any]]] = {}

    for name, node_list in nodes.items():
        canonical = canonical_name(name)
        for node in node_list:
            normalized_node = dict(node)
            normalized_node["entity_name"] = canonical
            entity_type = normalized_node.get("entity_type")

            if is_value_like_entity(canonical, entity_type):
                dropped_nodes[canonical] = DroppedNode(
                    name=canonical,
                    reason="value_like_entity",
                    replacement=None,
                )
                dropped_node_data.setdefault(canonical, []).append(normalized_node)
                continue

            if canonical not in normalized_nodes:
                normalized_nodes[canonical] = [normalized_node]
                continue

            normalized_nodes[canonical].append(normalized_node)

    normalized_edges: dict[tuple[str, str], list[dict[str, Any]]] = {}
    value_value_edges: list[tuple[dict[str, Any], DroppedNode, DroppedNode]] = []
    for edge_key, edge_list in edges.items():
        for edge in edge_list:
            normalized_edge = dict(edge)
            source = _canonical_endpoint(normalized_edge.get("src_id") or edge_key[0])
            target = _canonical_endpoint(normalized_edge.get("tgt_id") or edge_key[1])
            if not source or not target:
                continue
            if source == target:
                continue

            dropped_source = _dropped_value_endpoint(source, dropped_nodes)
            dropped_target = _dropped_value_endpoint(target, dropped_nodes)

            if dropped_source and dropped_target:
                value_value_edges.append(
                    (normalized_edge, dropped_source, dropped_target)
                )
                continue

            retained_endpoint = target if dropped_source else source
            if dropped_source or dropped_target:
                dropped = dropped_source or dropped_target
                if dropped:
                    dropped = _set_dropped_replacement(
                        dropped_nodes,
                        dropped,
                        retained_endpoint,
                    )
                _append_value_entity_annotations(
                    normalized_nodes,
                    retained_endpoint,
                    normalized_edge,
                    dropped,
                    dropped_node_data.get(dropped.name, []) if dropped else [],
                )
                continue
            else:
                normalized_edge["src_id"] = source
                normalized_edge["tgt_id"] = target

            keywords = normalized_edge.get("keywords")
            if isinstance(keywords, str):
                normalized_edge["keywords"] = normalize_relation_keyword(keywords)

            key = (normalized_edge["src_id"], normalized_edge["tgt_id"])
            normalized_edges.setdefault(key, []).append(normalized_edge)

    _fold_value_value_edges(
        normalized_nodes,
        dropped_nodes,
        dropped_node_data,
        value_value_edges,
    )

    return NormalizedExtraction(
        nodes=normalized_nodes,
        edges=normalized_edges,
        dropped_nodes=dropped_nodes,
    )


def _canonical_endpoint(name: Any) -> str:
    if name is None:
        return ""
    return canonical_name(str(name)).strip()


def _dropped_value_endpoint(
    name: str,
    dropped_nodes: dict[str, DroppedNode],
) -> DroppedNode | None:
    dropped = dropped_nodes.get(name)
    if dropped:
        return dropped

    if not is_value_like_entity(name, None):
        return None

    dropped = DroppedNode(name=name, reason="value_like_entity", replacement=None)
    dropped_nodes[name] = dropped
    return dropped


def _set_dropped_replacement(
    dropped_nodes: dict[str, DroppedNode],
    dropped: DroppedNode,
    replacement: str,
) -> DroppedNode:
    existing = dropped_nodes.get(dropped.name, dropped)
    if existing.replacement == replacement:
        return existing

    updated = DroppedNode(
        name=existing.name,
        reason=existing.reason,
        replacement=existing.replacement or replacement,
    )
    dropped_nodes[updated.name] = updated
    return updated


def _fold_value_value_edges(
    normalized_nodes: dict[str, list[dict[str, Any]]],
    dropped_nodes: dict[str, DroppedNode],
    dropped_node_data: dict[str, list[dict[str, Any]]],
    value_value_edges: list[tuple[dict[str, Any], DroppedNode, DroppedNode]],
) -> None:
    pending = list(value_value_edges)
    changed = True

    while pending and changed:
        changed = False
        next_pending: list[tuple[dict[str, Any], DroppedNode, DroppedNode]] = []

        for edge, source, target in pending:
            current_source = dropped_nodes.get(source.name, source)
            current_target = dropped_nodes.get(target.name, target)
            source_replacement = current_source.replacement
            target_replacement = current_target.replacement

            if source_replacement and not target_replacement:
                current_target = _set_dropped_replacement(
                    dropped_nodes,
                    current_target,
                    source_replacement,
                )
                _append_value_entity_annotations(
                    normalized_nodes,
                    source_replacement,
                    edge,
                    current_target,
                    dropped_node_data.get(current_target.name, []),
                )
                changed = True
                continue

            if target_replacement and not source_replacement:
                current_source = _set_dropped_replacement(
                    dropped_nodes,
                    current_source,
                    target_replacement,
                )
                _append_value_entity_annotations(
                    normalized_nodes,
                    target_replacement,
                    edge,
                    current_source,
                    dropped_node_data.get(current_source.name, []),
                )
                changed = True
                continue

            if source_replacement and target_replacement:
                if source_replacement == target_replacement:
                    _append_value_entity_annotations(
                        normalized_nodes,
                        source_replacement,
                        edge,
                        current_target,
                        [],
                    )
                changed = True
                continue

            next_pending.append((edge, current_source, current_target))

        pending = next_pending


def _merge_sep_values(left: Any, right: Any) -> str:
    values: list[str] = []
    for raw_value in (left, right):
        if raw_value is None:
            continue
        for value in str(raw_value).split("<SEP>"):
            stripped = value.strip()
            if stripped and stripped not in values:
                values.append(stripped)
    return "<SEP>".join(values)


def _append_value_entity_annotations(
    normalized_nodes: dict[str, list[dict[str, Any]]],
    retained_endpoint: str,
    edge: dict[str, Any],
    dropped: DroppedNode | None,
    dropped_fragments: list[dict[str, Any]],
) -> None:
    if dropped is None:
        return

    node_list = normalized_nodes.setdefault(retained_endpoint, [])
    entity_type = _retained_entity_type(node_list)
    edge_source_ids = _split_sep_values(edge.get("source_id"))
    edge_file_paths = _split_sep_values(edge.get("file_path"))
    for index, edge_source_id in enumerate(edge_source_ids):
        _append_unique_fragment(
            node_list,
            {
                "entity_name": retained_endpoint,
                "entity_type": entity_type,
                "description": _edge_annotation_description(edge, dropped),
                "source_id": edge_source_id,
                "file_path": _indexed_value(edge_file_paths, index),
                "timestamp": edge.get("timestamp")
                or _first_present(dropped_fragments, "timestamp")
                or 0,
            }
        )

    for fragment in dropped_fragments:
        dropped_source_ids = _split_sep_values(fragment.get("source_id"))
        dropped_file_paths = _split_sep_values(fragment.get("file_path"))
        for index, dropped_source_id in enumerate(dropped_source_ids):
            _append_unique_fragment(
                node_list,
                {
                    "entity_name": retained_endpoint,
                    "entity_type": entity_type,
                    "description": _value_annotation_description(dropped, fragment),
                    "source_id": dropped_source_id,
                    "file_path": _indexed_value(dropped_file_paths, index),
                    "timestamp": fragment.get("timestamp") or 0,
                }
            )


def _retained_entity_type(node_list: list[dict[str, Any]]) -> str:
    if not node_list:
        return "UNKNOWN"
    return str(node_list[0].get("entity_type") or "UNKNOWN")


def _append_unique_fragment(
    node_list: list[dict[str, Any]],
    fragment: dict[str, Any],
) -> None:
    fragment_key = _fragment_dedupe_key(fragment)
    if any(_fragment_dedupe_key(existing) == fragment_key for existing in node_list):
        return
    node_list.append(fragment)


def _fragment_dedupe_key(fragment: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(fragment.get("source_id") or ""),
        str(fragment.get("description") or ""),
        str(fragment.get("file_path") or ""),
    )


def _edge_annotation_description(
    edge: dict[str, Any],
    dropped: DroppedNode,
) -> str:
    return _merge_sep_values(edge.get("description"), dropped.name)


def _value_annotation_description(
    dropped: DroppedNode,
    fragment: dict[str, Any],
) -> str:
    return _merge_sep_values(dropped.name, fragment.get("description"))


def _first_present(fragments: list[dict[str, Any]], field: str) -> Any:
    for fragment in fragments:
        value = fragment.get(field)
        if value:
            return value
    return None


def _split_sep_values(value: Any) -> list[str]:
    if value is None:
        return []

    values: list[str] = []
    for raw_value in str(value).split("<SEP>"):
        stripped = raw_value.strip()
        if stripped and stripped not in values:
            values.append(stripped)
    return values


def _indexed_value(values: list[str], index: int) -> str:
    if index < len(values):
        return values[index]
    return "unknown_source"
