from __future__ import annotations

from typing import Any


GROUP_NODE_TYPE = "MedicalGroup"
PROFILE_SOURCE = "medical_kg_profile"
PROFILE_TIMESTAMP = 0

PARENT_BY_CHILD: dict[str, str] = {
    "甲型流感病毒": "流感病毒",
    "乙型流感病毒": "流感病毒",
    "A(H1N1)pdm09": "甲型流感病毒",
    "A(H3N2)": "甲型流感病毒",
    "Victoria系": "乙型流感病毒",
    "发热": "全身症状",
    "头痛": "全身症状",
    "肌肉关节酸痛": "全身症状",
    "咳嗽": "呼吸道症状",
    "咽痛": "呼吸道症状",
    "流涕": "呼吸道症状",
    "呼吸困难": "呼吸道症状",
    "恶心": "消化道症状",
    "呕吐": "消化道症状",
    "腹泻": "消化道症状",
    "急性呼吸窘迫综合征（ARDS）": "呼吸系统并发症",
    "病毒性肺炎": "呼吸系统并发症",
    "心肌炎": "心脏并发症",
    "脑炎": "神经系统并发症",
}

TOP_LEVEL_PARENT_BY_CHILD: dict[str, str] = {
    "流感病毒": "病原体",
    "全身症状": "临床表现",
    "呼吸道症状": "临床表现",
    "消化道症状": "临床表现",
    "呼吸系统并发症": "并发症/重症",
    "心脏并发症": "并发症/重症",
    "神经系统并发症": "并发症/重症",
}

HIERARCHY_PARENT_BY_CHILD: dict[str, str] = {
    **PARENT_BY_CHILD,
    **TOP_LEVEL_PARENT_BY_CHILD,
}

GROUP_BY_PARENT: dict[str, str] = {
    "流感病毒": "pathogen",
    "甲型流感病毒": "pathogen",
    "乙型流感病毒": "pathogen",
    "病原体": "pathogen",
    "全身症状": "symptom",
    "呼吸道症状": "symptom",
    "消化道症状": "symptom",
    "临床表现": "clinical_manifestation",
    "呼吸系统并发症": "complication",
    "心脏并发症": "complication",
    "神经系统并发症": "complication",
    "并发症/重症": "complication_severity",
}


def build_medical_hierarchy_edges(
    nodes: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    next_nodes = {name: list(fragments) for name, fragments in nodes.items()}
    hierarchy_edges: dict[tuple[str, str], list[dict[str, Any]]] = {}

    changed = True
    while changed:
        changed = False
        for child, parent in HIERARCHY_PARENT_BY_CHILD.items():
            edge_key = (child, parent)
            if child not in next_nodes or edge_key in hierarchy_edges:
                continue

            if parent not in next_nodes:
                next_nodes[parent] = [_group_node(parent)]
                changed = True

            hierarchy_edges[edge_key] = [
                {
                    "src_id": child,
                    "tgt_id": parent,
                    "weight": 0.0,
                    "keywords": "属于",
                    "description": f"{child}属于{parent}。",
                    "source_id": PROFILE_SOURCE,
                    "file_path": PROFILE_SOURCE,
                    "timestamp": PROFILE_TIMESTAMP,
                    "generated_by": PROFILE_SOURCE,
                }
            ]
            changed = True

    return next_nodes, hierarchy_edges


def _group_node(
    parent: str,
) -> dict[str, Any]:
    return {
        "entity_name": parent,
        "entity_type": GROUP_NODE_TYPE,
        "medical_group": GROUP_BY_PARENT.get(parent, "other"),
        "description": parent,
        "source_id": PROFILE_SOURCE,
        "file_path": PROFILE_SOURCE,
        "timestamp": PROFILE_TIMESTAMP,
        "generated_by": PROFILE_SOURCE,
    }
