from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any


GraphRequestPayload = Mapping[str, Any] | dict[str, Any]
GraphBackendQueryHook = Callable[
    [Any, dict[str, Any], int], Awaitable[Any] | Any
]
LOW_WEIGHT_EDGE_THRESHOLD = 0.1
IGNORED_HIGHLIGHT_MATCHES = "view_options.highlight_matches"


def _model_dump_or_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _normalize_item(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _normalize_graph_data(raw_graph: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    payload = _model_dump_or_dict(raw_graph)
    raw_nodes = payload.get("nodes", [])
    raw_edges = payload.get("edges", [])
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    if isinstance(raw_nodes, Sequence) and not isinstance(raw_nodes, (str, bytes)):
        nodes = [_normalize_item(node) for node in raw_nodes]
    if isinstance(raw_edges, Sequence) and not isinstance(raw_edges, (str, bytes)):
        edges = [_normalize_item(edge) for edge in raw_edges]

    return nodes, edges, bool(payload.get("is_truncated", False))


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, Sequence):
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                normalized = item.strip()
                if normalized:
                    result.append(normalized)
        return result
    return []


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _to_int(value: Any, default: int, minimum: int = 1) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(value, minimum)
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return default
    return max(converted, minimum)


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_runtime_max_nodes(rag: Any) -> int | None:
    candidates: list[int] = []

    runtime_limit = getattr(rag, "max_graph_nodes", None)
    if isinstance(runtime_limit, int) and runtime_limit > 0:
        candidates.append(runtime_limit)

    global_config = getattr(rag, "global_config", None)
    if isinstance(global_config, Mapping):
        config_limit = global_config.get("max_graph_nodes")
        if isinstance(config_limit, int) and config_limit > 0:
            candidates.append(config_limit)

    backend = getattr(rag, "chunk_entity_relation_graph", None)
    if backend is not None:
        backend_limit = getattr(backend, "max_graph_nodes", None)
        if isinstance(backend_limit, int) and backend_limit > 0:
            candidates.append(backend_limit)
        backend_config = getattr(backend, "global_config", None)
        if isinstance(backend_config, Mapping):
            backend_config_limit = backend_config.get("max_graph_nodes")
            if isinstance(backend_config_limit, int) and backend_config_limit > 0:
                candidates.append(backend_config_limit)

    if not candidates:
        return None
    return min(candidates)


def _safe_lower(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    return ""


def _extract_entity_types(node: dict[str, Any]) -> set[str]:
    entity_types: set[str] = set()
    labels = _normalize_list(node.get("labels"))
    entity_types.update(label.lower() for label in labels)

    properties = _model_dump_or_dict(node.get("properties"))
    property_entity_types = _normalize_list(properties.get("entity_type"))
    entity_types.update(entity_type.lower() for entity_type in property_entity_types)
    return entity_types


def _extract_source_values(record: dict[str, Any]) -> tuple[list[str], list[str], list[datetime]]:
    properties = _model_dump_or_dict(record.get("properties"))

    source_ids = _normalize_list(properties.get("source_id"))
    file_paths = _normalize_list(properties.get("file_path")) + _normalize_list(
        properties.get("file_paths")
    )

    timestamps: list[datetime] = []
    for key in ("time", "timestamp", "created_at", "updated_at"):
        for value in _normalize_list(properties.get(key)):
            parsed = _parse_iso_datetime(value)
            if parsed is not None:
                timestamps.append(parsed)
    return source_ids, file_paths, timestamps


def _parse_iso_datetime(value: str) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source_filter_active(source_filters: dict[str, Any]) -> bool:
    return bool(
        _normalize_text(source_filters.get("source_id_query"))
        or _normalize_list(source_filters.get("file_paths"))
        or _normalize_text(source_filters.get("time_from"))
        or _normalize_text(source_filters.get("time_to"))
    )


def _node_filter_active(node_filters: dict[str, Any]) -> bool:
    return bool(
        _normalize_list(node_filters.get("entity_types"))
        or _normalize_text(node_filters.get("name_query"))
        or _normalize_text(node_filters.get("description_query"))
        or node_filters.get("degree_min") is not None
        or node_filters.get("degree_max") is not None
        or bool(node_filters.get("isolated_only"))
    )


def _edge_filter_active(edge_filters: dict[str, Any]) -> bool:
    return bool(
        _normalize_list(edge_filters.get("relation_types"))
        or _normalize_text(edge_filters.get("keyword_query"))
        or edge_filters.get("weight_min") is not None
        or edge_filters.get("weight_max") is not None
        or _normalize_list(edge_filters.get("source_entity_types"))
        or _normalize_list(edge_filters.get("target_entity_types"))
    )


def _view_options_active(view_options: dict[str, Any]) -> bool:
    return any(
        bool(view_options.get(key))
        for key in (
            "show_nodes_only",
            "show_edges_only",
            "hide_low_weight_edges",
            "hide_empty_description",
        )
    )


def _matches_source_filters(record: dict[str, Any], source_filters: dict[str, Any]) -> bool:
    source_id_query = _safe_lower(source_filters.get("source_id_query"))
    requested_paths = {path.lower() for path in _normalize_list(source_filters.get("file_paths"))}

    time_from_raw = _normalize_text(source_filters.get("time_from"))
    time_to_raw = _normalize_text(source_filters.get("time_to"))
    time_from = _parse_iso_datetime(time_from_raw) if time_from_raw else None
    time_to = _parse_iso_datetime(time_to_raw) if time_to_raw else None

    source_ids, file_paths, timestamps = _extract_source_values(record)

    if source_id_query:
        if not any(source_id_query in source_id.lower() for source_id in source_ids):
            return False

    if requested_paths:
        lowered_paths = {path.lower() for path in file_paths}
        if not lowered_paths.intersection(requested_paths):
            return False

    if time_from is not None or time_to is not None:
        if not timestamps:
            return False
        in_range = False
        for timestamp in timestamps:
            if time_from is not None and timestamp < time_from:
                continue
            if time_to is not None and timestamp > time_to:
                continue
            in_range = True
            break
        if not in_range:
            return False

    return True


def _build_degree_map(
    node_ids: list[str], edges: list[dict[str, Any]]
) -> dict[str, int]:
    degree_map = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source in degree_map:
            degree_map[source] += 1
        if target in degree_map:
            degree_map[target] += 1
    return degree_map


def _matches_node_filters(
    node: dict[str, Any],
    node_filters: dict[str, Any],
    degree_map: dict[str, int],
) -> bool:
    node_id = str(node.get("id", "")).strip()
    properties = _model_dump_or_dict(node.get("properties"))

    entity_types = [item.lower() for item in _normalize_list(node_filters.get("entity_types"))]
    if entity_types:
        node_entity_types = _extract_entity_types(node)
        if not any(entity_type in node_entity_types for entity_type in entity_types):
            return False

    name_query = _safe_lower(node_filters.get("name_query"))
    if name_query:
        candidates = [node_id, _normalize_text(properties.get("name"))]
        if not any(name_query in candidate.lower() for candidate in candidates if candidate):
            return False

    description_query = _safe_lower(node_filters.get("description_query"))
    if description_query:
        description = _normalize_text(properties.get("description"))
        if description_query not in description.lower():
            return False

    degree = degree_map.get(node_id, 0)
    degree_min = _to_int(node_filters.get("degree_min"), 0, 0) if node_filters.get("degree_min") is not None else None
    degree_max = _to_int(node_filters.get("degree_max"), 0, 0) if node_filters.get("degree_max") is not None else None
    if degree_min is not None and degree < degree_min:
        return False
    if degree_max is not None and degree > degree_max:
        return False
    if bool(node_filters.get("isolated_only")) and degree != 0:
        return False

    return True


def _extract_edge_weight(edge: dict[str, Any]) -> float | None:
    properties = _model_dump_or_dict(edge.get("properties"))
    weight = properties.get("weight", edge.get("weight"))
    return _to_float_or_none(weight)


def _matches_edge_filters(
    edge: dict[str, Any],
    edge_filters: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
) -> bool:
    properties = _model_dump_or_dict(edge.get("properties"))

    relation_types = [item.lower() for item in _normalize_list(edge_filters.get("relation_types"))]
    if relation_types:
        edge_types = {
            _safe_lower(edge.get("type")),
            _safe_lower(properties.get("relation_type")),
        }
        if not any(edge_type in edge_types for edge_type in relation_types):
            return False

    keyword_query = _safe_lower(edge_filters.get("keyword_query"))
    if keyword_query:
        keyword_fields = [
            _normalize_text(properties.get("keywords")),
            _normalize_text(properties.get("description")),
        ]
        if not any(keyword_query in field.lower() for field in keyword_fields if field):
            return False

    weight = _extract_edge_weight(edge)
    weight_min = _to_float_or_none(edge_filters.get("weight_min"))
    weight_max = _to_float_or_none(edge_filters.get("weight_max"))
    if weight_min is not None:
        if weight is None or weight < weight_min:
            return False
    if weight_max is not None:
        if weight is None or weight > weight_max:
            return False

    source_entity_types = [
        item.lower() for item in _normalize_list(edge_filters.get("source_entity_types"))
    ]
    target_entity_types = [
        item.lower() for item in _normalize_list(edge_filters.get("target_entity_types"))
    ]
    source_node = node_by_id.get(str(edge.get("source", "")).strip(), {})
    target_node = node_by_id.get(str(edge.get("target", "")).strip(), {})
    if source_entity_types:
        source_types = _extract_entity_types(source_node)
        if not any(entity_type in source_types for entity_type in source_entity_types):
            return False
    if target_entity_types:
        target_types = _extract_entity_types(target_node)
        if not any(entity_type in target_types for entity_type in target_entity_types):
            return False

    return True


def _apply_view_options(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    view_options: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    next_nodes = list(nodes)
    next_edges = list(edges)

    if bool(view_options.get("hide_low_weight_edges")):
        next_edges = [
            edge
            for edge in next_edges
            if (_extract_edge_weight(edge) or 0.0) > LOW_WEIGHT_EDGE_THRESHOLD
        ]

    if bool(view_options.get("hide_empty_description")):
        next_nodes = [
            node
            for node in next_nodes
            if _normalize_text(_model_dump_or_dict(node.get("properties")).get("description"))
        ]
        allowed_ids = {str(node.get("id", "")).strip() for node in next_nodes}
        next_edges = [
            edge
            for edge in next_edges
            if str(edge.get("source", "")).strip() in allowed_ids
            and str(edge.get("target", "")).strip() in allowed_ids
            and _normalize_text(_model_dump_or_dict(edge.get("properties")).get("description"))
        ]

    if bool(view_options.get("show_nodes_only")):
        next_edges = []
    if bool(view_options.get("show_edges_only")):
        next_nodes = []

    return next_nodes, next_edges


def _apply_post_filter_guardrail(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    effective_max_nodes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    if len(nodes) <= effective_max_nodes:
        return nodes, edges, False

    limited_nodes = nodes[:effective_max_nodes]
    allowed_ids = {str(node.get("id", "")).strip() for node in limited_nodes}
    limited_edges = [
        edge
        for edge in edges
        if str(edge.get("source", "")).strip() in allowed_ids
        and str(edge.get("target", "")).strip() in allowed_ids
    ]
    return limited_nodes, limited_edges, True


async def _fetch_base_graph(
    rag: Any,
    request_payload: dict[str, Any],
    scope: dict[str, Any],
    effective_max_nodes: int,
    backend_query_hook: GraphBackendQueryHook | None,
) -> Any:
    if backend_query_hook is not None:
        maybe_graph = backend_query_hook(rag, request_payload, effective_max_nodes)
        if isinstance(maybe_graph, Awaitable):
            maybe_graph = await maybe_graph
        if maybe_graph is not None:
            return maybe_graph

    return await rag.get_knowledge_graph(
        node_label=_normalize_text(scope.get("label")) or "*",
        max_depth=_to_int(scope.get("max_depth"), 3, 1),
        max_nodes=effective_max_nodes,
    )


async def query_graph_workbench(
    rag: Any,
    request: GraphRequestPayload | Any,
    backend_query_hook: GraphBackendQueryHook | None = None,
) -> dict[str, Any]:
    request_payload = _model_dump_or_dict(request)
    scope = _model_dump_or_dict(request_payload.get("scope"))
    node_filters = _model_dump_or_dict(request_payload.get("node_filters"))
    edge_filters = _model_dump_or_dict(request_payload.get("edge_filters"))
    source_filters = _model_dump_or_dict(request_payload.get("source_filters"))
    view_options = _model_dump_or_dict(request_payload.get("view_options"))

    requested_max_nodes = _to_int(scope.get("max_nodes"), 1000, 1)
    runtime_limit = _extract_runtime_max_nodes(rag)
    effective_max_nodes = (
        min(requested_max_nodes, runtime_limit)
        if runtime_limit is not None
        else requested_max_nodes
    )

    raw_graph = await _fetch_base_graph(
        rag=rag,
        request_payload=request_payload,
        scope=scope,
        effective_max_nodes=effective_max_nodes,
        backend_query_hook=backend_query_hook,
    )
    nodes, edges, was_truncated_before = _normalize_graph_data(raw_graph)

    normalized_nodes: list[dict[str, Any]] = []
    node_by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        if node_id in node_by_id:
            continue
        normalized_node = dict(node)
        normalized_node["id"] = node_id
        normalized_nodes.append(normalized_node)
        node_by_id[node_id] = normalized_node

    normalized_edges: list[dict[str, Any]] = []
    for index, edge in enumerate(edges):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target:
            continue
        normalized_edge = dict(edge)
        normalized_edge["source"] = source
        normalized_edge["target"] = target
        if not _normalize_text(normalized_edge.get("id")):
            normalized_edge["id"] = f"edge-{index}"
        normalized_edges.append(normalized_edge)

    degree_map = _build_degree_map(
        [str(node.get("id", "")).strip() for node in normalized_nodes], normalized_edges
    )

    node_filter_active = _node_filter_active(node_filters)
    edge_filter_active = _edge_filter_active(edge_filters)
    source_filter_active = _source_filter_active(source_filters)
    view_options_active = _view_options_active(view_options)
    highlight_matches_requested = bool(view_options.get("highlight_matches"))

    filtered_nodes: list[dict[str, Any]] = []
    for node in normalized_nodes:
        if node_filter_active and not _matches_node_filters(node, node_filters, degree_map):
            continue
        filtered_nodes.append(node)

    filtered_node_ids = {str(node.get("id", "")).strip() for node in filtered_nodes}

    if bool(scope.get("only_matched_neighborhood")) and node_filter_active:
        neighborhood_ids = set(filtered_node_ids)
        for edge in normalized_edges:
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if source in filtered_node_ids or target in filtered_node_ids:
                neighborhood_ids.add(source)
                neighborhood_ids.add(target)
        filtered_nodes = [
            node
            for node in normalized_nodes
            if str(node.get("id", "")).strip() in neighborhood_ids
        ]
        filtered_node_ids = {
            str(node.get("id", "")).strip() for node in filtered_nodes
        }

    filtered_edges: list[dict[str, Any]] = []
    for edge in normalized_edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source not in filtered_node_ids or target not in filtered_node_ids:
            continue
        if edge_filter_active and not _matches_edge_filters(edge, edge_filters, node_by_id):
            continue
        if source_filter_active and not _matches_source_filters(edge, source_filters):
            continue
        filtered_edges.append(edge)

    filtered_nodes, filtered_edges = _apply_view_options(
        nodes=filtered_nodes,
        edges=filtered_edges,
        view_options=view_options,
    )
    filtered_nodes, filtered_edges, was_truncated_after_cap = _apply_post_filter_guardrail(
        nodes=filtered_nodes,
        edges=filtered_edges,
        effective_max_nodes=effective_max_nodes,
    )

    was_truncated_after = bool(was_truncated_before or was_truncated_after_cap)
    filtering_applied = bool(
        node_filter_active
        or edge_filter_active
        or source_filter_active
        or view_options_active
        or bool(scope.get("only_matched_neighborhood"))
    )
    ignored_filter_groups: list[str] = []
    if highlight_matches_requested:
        ignored_filter_groups.append(IGNORED_HIGHLIGHT_MATCHES)

    return {
        "data": {
            "nodes": filtered_nodes,
            "edges": filtered_edges,
            "is_truncated": was_truncated_after,
        },
        "truncation": {
            "requested_max_nodes": requested_max_nodes,
            "effective_max_nodes": effective_max_nodes,
            "was_truncated_before_filtering": bool(was_truncated_before),
            "was_truncated_after_filtering": was_truncated_after,
        },
        "meta": {
            "filter_semantics": {
                "group_operator": "AND",
                "field_operator": "AND",
                "array_operator": "OR",
                "version": "v1",
            },
            "execution_mode": "base_graph_only_placeholder",
            "filtering_applied": filtering_applied,
            "ignored_filter_groups": ignored_filter_groups,
        },
    }


async def get_legacy_graph_payload(
    rag: Any,
    *,
    label: str,
    max_depth: int,
    max_nodes: int,
    backend_query_hook: GraphBackendQueryHook | None = None,
) -> dict[str, Any]:
    result = await query_graph_workbench(
        rag=rag,
        request={
            "scope": {
                "label": label,
                "max_depth": max_depth,
                "max_nodes": max_nodes,
                "only_matched_neighborhood": False,
            },
        },
        backend_query_hook=backend_query_hook,
    )
    return result["data"]
