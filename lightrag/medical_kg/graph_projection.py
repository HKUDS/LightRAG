from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from lightrag.medical_kg.ontology import (
    TOP_LEVEL_MEDICAL_CATEGORIES,
    normalize_relation_keyword,
)


MEDICAL_GROUPS: tuple[tuple[str, str], ...] = (
    ("pathogen", "病原体"),
    ("symptom", "临床表现"),
    ("complication", "并发症"),
    ("diagnosis", "诊断"),
    ("treatment", "治疗"),
    ("prevention", "预防"),
    ("population", "人群"),
    ("guideline", "指南建议"),
    ("other", "其他"),
)
_MEDICAL_GROUP_KEYS = {key for key, _label in MEDICAL_GROUPS}
_TOP_LEVEL_CATEGORY_LABELS = {
    category.label for category in TOP_LEVEL_MEDICAL_CATEGORIES
}

_ENTITY_TYPE_GROUPS = {
    "pathogen": "pathogen",
    "symptom": "symptom",
    "complication": "complication",
    "diagnostictest": "diagnosis",
    "diagnosticcriterion": "diagnosis",
    "drug": "treatment",
    "treatmentregimen": "treatment",
    "vaccine": "prevention",
    "publichealthmeasure": "prevention",
    "population": "population",
    "riskfactor": "population",
    "guideline": "guideline",
    "recommendation": "guideline",
}

_KEYWORD_ENDPOINT_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("pathogen", ("病原导致", "病原分型"), "target"),
    ("symptom", ("临床表现", "症状归类"), "target"),
    ("complication", ("并发风险",), "target"),
    ("diagnosis", ("诊断依据", "检测方法", "重症判定"), "target"),
    ("treatment", ("推荐治疗", "剂量用法"), "source"),
    ("prevention", ("预防措施",), "target"),
    ("population", ("适用于", "高危因素"), "target"),
    ("guideline", ("指南建议",), "source"),
)

_SYMPTOM_NAME_HINTS = (
    "症状",
    "临床表现",
    "发热",
    "咳嗽",
    "乏力",
    "疼痛",
    "腹泻",
    "呕吐",
    "头痛",
    "肌痛",
)

_MEDICAL_GROUP_NAME_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pathogen", ("病毒", "病原体", "毒株", "亚型")),
    ("symptom", ("症状", "临床表现")),
    ("complication", ("并发症", "并发", "综合征", "衰竭", "损伤", "休克", "坏死", "血栓")),
    ("diagnosis", ("诊断", "检测", "检查", "检验")),
    ("treatment", ("治疗", "药物", "用药", "方案")),
    ("prevention", ("预防", "疫苗", "隔离", "监测")),
    ("population", ("人群", "患者", "儿童", "孕妇", "老年")),
    ("guideline", ("指南", "诊疗方案", "共识", "建议")),
)


def project_medical_graph(
    graph_payload: Any,
    *,
    include_browse: bool = False,
    root_id: str | None = None,
) -> dict[str, Any]:
    """Return graph payload plus medical grouping metadata.

    The projection is intentionally additive: it serializes the public
    KnowledgeGraph shape to a dict, then appends metadata without changing
    the underlying node or edge lists.
    """
    payload = _payload_to_dict(graph_payload)
    node_order = [str(node.get("id", "")) for node in payload.get("nodes", [])]
    edge_suggestions = _edge_group_suggestions(
        payload.get("edges", []), payload.get("nodes", [])
    )

    grouped_node_ids: dict[str, list[str]] = {key: [] for key, _label in MEDICAL_GROUPS}
    seen_by_group: dict[str, set[str]] = {key: set() for key, _label in MEDICAL_GROUPS}

    for node in payload.get("nodes", []):
        node_id = str(node.get("id", ""))
        if not node_id:
            continue
        group_key = _node_group(node, edge_suggestions.get(node_id, []))
        if node_id not in seen_by_group[group_key]:
            grouped_node_ids[group_key].append(node_id)
            seen_by_group[group_key].add(node_id)

    medical_groups = [
        {
            "key": key,
            "label": label,
            "node_ids": grouped_node_ids[key],
            "count": len(grouped_node_ids[key]),
        }
        for key, label in MEDICAL_GROUPS
        if grouped_node_ids[key]
    ]

    metadata = payload.get("metadata")
    payload["metadata"] = dict(metadata) if isinstance(metadata, Mapping) else {}
    payload["metadata"]["medical_groups"] = medical_groups
    if include_browse:
        payload["metadata"]["medical_browse"] = _build_medical_browse_metadata(
            payload,
            preferred_root_id=root_id,
        )

    # Defensive check documents the stability contract for future edits.
    if [str(node.get("id", "")) for node in payload.get("nodes", [])] != node_order:
        raise RuntimeError("Medical graph projection must preserve node order")

    return payload


def _build_medical_browse_metadata(
    payload: Mapping[str, Any],
    *,
    preferred_root_id: str | None = None,
) -> dict[str, Any]:
    nodes = [
        node for node in payload.get("nodes", []) or [] if isinstance(node, Mapping)
    ]
    edges = [
        edge for edge in payload.get("edges", []) or [] if isinstance(edge, Mapping)
    ]
    node_order = [str(node.get("id", "")) for node in nodes if node.get("id")]
    root_id = _browse_root_id(nodes, preferred_root_id)
    node_roles = _browse_node_roles(nodes, root_id)

    return {
        "root_id": root_id,
        "default_depth": "medium",
        "category_order": [
            category.key for category in TOP_LEVEL_MEDICAL_CATEGORIES
        ],
        "node_roles": node_roles,
        "collapsed_groups": _browse_collapsed_groups(
            edges, node_roles, node_order
        ),
        "relation_details": _browse_relation_details(edges),
    }


def _browse_root_id(
    nodes: list[Mapping[str, Any]], preferred_root_id: str | None = None
) -> str | None:
    if preferred_root_id:
        for node in nodes:
            node_id = str(node.get("id", ""))
            if node_id == preferred_root_id:
                return node_id
    for node in nodes:
        node_id = str(node.get("id", ""))
        if node_id and _node_has_type_or_label(node, "disease"):
            return node_id
    for node in nodes:
        node_id = str(node.get("id", ""))
        if node_id:
            return node_id
    return None


def _browse_node_roles(
    nodes: list[Mapping[str, Any]], root_id: str | None
) -> dict[str, str]:
    node_roles: dict[str, str] = {}
    for node in nodes:
        node_id = str(node.get("id", ""))
        if not node_id:
            continue
        if node_id == root_id:
            node_roles[node_id] = "root"
        elif node_id in _TOP_LEVEL_CATEGORY_LABELS:
            node_roles[node_id] = "category"
        elif _node_has_type_or_label(node, "medicalgroup"):
            node_roles[node_id] = "subgroup"
        else:
            node_roles[node_id] = "leaf"
    return node_roles


def _browse_collapsed_groups(
    edges: list[Mapping[str, Any]],
    node_roles: Mapping[str, str],
    node_order: list[str],
) -> list[dict[str, Any]]:
    children_by_parent: dict[str, list[str]] = {}
    seen_by_parent: dict[str, set[str]] = {}
    node_index = {node_id: index for index, node_id in enumerate(node_order)}

    for edge in edges:
        if _edge_relation(edge) != "属于":
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        source_role = node_roles.get(source)
        target_role = node_roles.get(target)
        if source_role == "leaf" and target_role == "subgroup":
            parent_id, child_id = target, source
        elif source_role == "subgroup" and target_role == "leaf":
            parent_id, child_id = source, target
        else:
            continue

        seen_children = seen_by_parent.setdefault(parent_id, set())
        if child_id in seen_children:
            continue
        seen_children.add(child_id)
        children_by_parent.setdefault(parent_id, []).append(child_id)

    collapsed_groups: list[dict[str, Any]] = []
    for parent_id in node_order:
        child_ids = children_by_parent.get(parent_id, [])
        if len(child_ids) <= 1:
            continue
        ordered_child_ids = sorted(
            child_ids, key=lambda child_id: node_index.get(child_id, len(node_order))
        )
        examples = ordered_child_ids[:3]
        collapsed_groups.append(
            {
                "id": f"medical-collapsed:{parent_id}",
                "label": f"{parent_id} ({len(ordered_child_ids)}): {'、'.join(examples)}",
                "parent_id": parent_id,
                "child_ids": ordered_child_ids,
                "count": len(ordered_child_ids),
                "examples": examples,
            }
        )
    return collapsed_groups


def _browse_relation_details(
    edges: list[Mapping[str, Any]]
) -> dict[str, dict[str, str]]:
    relation_details: dict[str, dict[str, str]] = {}
    for index, edge in enumerate(edges):
        edge_id = str(edge.get("id") or f"edge:{index}")
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        relation = _edge_relation(edge)
        relation_details[edge_id] = {
            "source": source,
            "target": target,
            "relation": relation,
            "display": f"{relation}：{target}",
            "triple": f"{source} - {relation} -> {target}",
        }
    return relation_details


def _edge_relation(edge: Mapping[str, Any]) -> str:
    properties = _mapping_or_empty(edge.get("properties"))
    relation = normalize_relation_keyword(
        str(properties.get("keywords") or edge.get("type") or "相关")
    )
    return relation or "相关"


def _node_has_type_or_label(node: Mapping[str, Any], expected: str) -> bool:
    properties = _mapping_or_empty(node.get("properties"))
    entity_type = str(properties.get("entity_type") or "")
    if _normalize_entity_type(entity_type) == expected:
        return True
    return any(_normalize_entity_type(label) == expected for label in _node_labels(node))


def _node_labels(node: Mapping[str, Any]) -> list[str]:
    labels = node.get("labels") or []
    if isinstance(labels, str):
        return [labels]
    return [str(label) for label in labels]


def _payload_to_dict(graph_payload: Any) -> dict[str, Any]:
    if hasattr(graph_payload, "model_dump"):
        return graph_payload.model_dump()
    if isinstance(graph_payload, Mapping):
        return deepcopy(dict(graph_payload))
    return {
        "nodes": deepcopy(getattr(graph_payload, "nodes", [])),
        "edges": deepcopy(getattr(graph_payload, "edges", [])),
        "is_truncated": bool(getattr(graph_payload, "is_truncated", False)),
    }


def _node_group(node: Mapping[str, Any], suggested_groups: list[str]) -> str:
    properties = _mapping_or_empty(node.get("properties"))
    entity_type = str(properties.get("entity_type") or "").strip()
    node_name = str(properties.get("entity_name") or node.get("id") or "")
    hinted_group = _medical_group_name_hint(node_name)
    if _normalize_entity_type(entity_type) == "disease" and hinted_group == "complication":
        return hinted_group
    explicit_group = _explicit_node_group(node)
    if explicit_group:
        return explicit_group
    if _normalize_entity_type(entity_type) == "medicalgroup":
        if hinted_group:
            return hinted_group

    if suggested_groups:
        return suggested_groups[0]
    if hinted_group == "complication":
        return hinted_group
    return "other"


def _explicit_node_group(node: Mapping[str, Any]) -> str | None:
    properties = _mapping_or_empty(node.get("properties"))
    medical_group = _normalize_group_key(str(properties.get("medical_group") or ""))
    if medical_group:
        return medical_group

    entity_type = str(properties.get("entity_type") or "").strip()
    normalized_type = _normalize_entity_type(entity_type)

    if normalized_type in _ENTITY_TYPE_GROUPS:
        return _ENTITY_TYPE_GROUPS[normalized_type]
    if normalized_type == "disease":
        return "other"

    labels = node.get("labels") or []
    if isinstance(labels, str):
        labels = [labels]
    for label in labels:
        normalized_label = _normalize_entity_type(str(label))
        label_group = _ENTITY_TYPE_GROUPS.get(normalized_label)
        if label_group:
            return label_group
        if normalized_label == "disease":
            return "other"
    return None


def _edge_group_suggestions(edges: Any, nodes: Any) -> dict[str, list[str]]:
    suggestions: dict[str, list[str]] = {}
    node_groups = _explicit_groups_by_node_id(nodes)
    for edge in edges or []:
        if not isinstance(edge, Mapping):
            continue
        properties = _mapping_or_empty(edge.get("properties"))
        keywords = str(properties.get("keywords") or "")
        if not keywords:
            continue

        for group_key, keyword_hints, endpoint in _KEYWORD_ENDPOINT_RULES:
            if not any(keyword in keywords for keyword in keyword_hints):
                continue
            for node_id in _candidate_edge_node_ids(edge, endpoint, group_key, node_groups):
                node_suggestions = suggestions.setdefault(node_id, [])
                if group_key not in node_suggestions:
                    node_suggestions.append(group_key)
    return suggestions


def _candidate_edge_node_ids(
    edge: Mapping[str, Any],
    fallback_endpoint: str,
    group_key: str,
    node_groups: Mapping[str, str],
) -> list[str]:
    endpoint_ids = [
        str(edge.get("source") or ""),
        str(edge.get("target") or ""),
    ]
    typed_matches = [
        node_id for node_id in endpoint_ids if node_id and node_groups.get(node_id) == group_key
    ]
    if typed_matches:
        return typed_matches

    fallback_id = str(edge.get(fallback_endpoint) or "")
    return [fallback_id] if fallback_id else []


def _explicit_groups_by_node_id(nodes: Any) -> dict[str, str]:
    groups: dict[str, str] = {}
    for node in nodes or []:
        if not isinstance(node, Mapping):
            continue
        node_id = str(node.get("id", ""))
        group_key = _explicit_node_group(node)
        if node_id and group_key:
            groups[node_id] = group_key
    return groups


def _normalize_entity_type(entity_type: str) -> str:
    return entity_type.replace("_", "").replace(" ", "").lower()


def _normalize_group_key(group_key: str) -> str | None:
    normalized = group_key.strip().lower()
    return normalized if normalized in _MEDICAL_GROUP_KEYS else None


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _has_symptom_name_hint(name: str) -> bool:
    return any(hint in name for hint in _SYMPTOM_NAME_HINTS)


def _medical_group_name_hint(name: str) -> str | None:
    if _has_symptom_name_hint(name):
        return "symptom"
    for group_key, hints in _MEDICAL_GROUP_NAME_HINTS:
        if any(hint in name for hint in hints):
            return group_key
    return None
