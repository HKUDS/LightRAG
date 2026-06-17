import importlib
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode


_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_graph_routes = importlib.import_module("lightrag.api.routers.graph_routes")
sys.argv = _original_argv

create_graph_routes = _graph_routes.create_graph_routes

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_HEADERS = {"X-API-Key": _API_KEY}


def _medical_graph_payload() -> dict:
    return {
        "nodes": [
            {
                "id": "流行性感冒",
                "labels": ["Disease"],
                "properties": {"entity_type": "Disease", "entity_name": "流行性感冒"},
            },
            {
                "id": "甲型流感病毒",
                "labels": ["Pathogen"],
                "properties": {
                    "entity_type": "Pathogen",
                    "entity_name": "甲型流感病毒",
                },
            },
            {
                "id": "发热",
                "labels": ["Symptom"],
                "properties": {"entity_type": "Symptom", "entity_name": "发热"},
            },
            {
                "id": "奥司他韦",
                "labels": ["Drug"],
                "properties": {"entity_type": "Drug", "entity_name": "奥司他韦"},
            },
            {
                "id": "核酸检测",
                "labels": ["DiagnosticTest"],
                "properties": {
                    "entity_type": "DiagnosticTest",
                    "entity_name": "核酸检测",
                },
            },
            {
                "id": "儿童",
                "labels": ["Population"],
                "properties": {"entity_type": "Population", "entity_name": "儿童"},
            },
            {
                "id": "流感诊疗方案",
                "labels": ["Guideline"],
                "properties": {
                    "entity_type": "Guideline",
                    "entity_name": "流感诊疗方案",
                },
            },
        ],
        "edges": [
            {
                "id": "e1",
                "type": "related",
                "source": "流行性感冒",
                "target": "甲型流感病毒",
                "properties": {"keywords": "病原导致"},
            },
            {
                "id": "e2",
                "type": "related",
                "source": "流行性感冒",
                "target": "发热",
                "properties": {"keywords": "临床表现"},
            },
            {
                "id": "e3",
                "type": "related",
                "source": "奥司他韦",
                "target": "流行性感冒",
                "properties": {"keywords": "推荐治疗"},
            },
            {
                "id": "e4",
                "type": "related",
                "source": "流行性感冒",
                "target": "核酸检测",
                "properties": {"keywords": "诊断依据"},
            },
            {
                "id": "e5",
                "type": "related",
                "source": "流感诊疗方案",
                "target": "儿童",
                "properties": {"keywords": "适用于"},
            },
            {
                "id": "e6",
                "type": "related",
                "source": "流感诊疗方案",
                "target": "流行性感冒",
                "properties": {"keywords": "指南建议"},
            },
        ],
        "is_truncated": False,
    }


def _knowledge_graph() -> KnowledgeGraph:
    payload = _medical_graph_payload()
    return KnowledgeGraph(
        nodes=[KnowledgeGraphNode(**node) for node in payload["nodes"]],
        edges=[KnowledgeGraphEdge(**edge) for edge in payload["edges"]],
        is_truncated=payload["is_truncated"],
    )


def test_project_medical_graph_returns_ordered_non_empty_groups():
    from lightrag.medical_kg.graph_projection import project_medical_graph

    projected = project_medical_graph(_medical_graph_payload())

    groups = projected["metadata"]["medical_groups"]
    assert [group["key"] for group in groups] == [
        "pathogen",
        "symptom",
        "diagnosis",
        "treatment",
        "population",
        "guideline",
        "other",
    ]
    assert [(group["key"], group["label"], group["node_ids"], group["count"]) for group in groups] == [
        ("pathogen", "病原体", ["甲型流感病毒"], 1),
        ("symptom", "临床表现", ["发热"], 1),
        ("diagnosis", "诊断", ["核酸检测"], 1),
        ("treatment", "治疗", ["奥司他韦"], 1),
        ("population", "人群", ["儿童"], 1),
        ("guideline", "指南建议", ["流感诊疗方案"], 1),
        ("other", "其他", ["流行性感冒"], 1),
    ]


def test_project_medical_graph_preserves_graph_payload_without_mutating_input():
    from lightrag.medical_kg.graph_projection import project_medical_graph

    payload = _medical_graph_payload()
    original_nodes = [dict(node) for node in payload["nodes"]]
    original_edges = [dict(edge) for edge in payload["edges"]]

    projected = project_medical_graph(payload)

    assert projected["nodes"] == original_nodes
    assert projected["edges"] == original_edges
    assert projected["is_truncated"] is False
    assert payload["nodes"] == original_nodes
    assert payload["edges"] == original_edges
    assert "metadata" not in payload


def test_project_medical_graph_groups_prevention_boundary_types_and_keeps_metadata():
    from lightrag.medical_kg.graph_projection import project_medical_graph

    payload = {
        "nodes": [
            {
                "id": "流感疫苗",
                "labels": ["Vaccine"],
                "properties": {"entity_type": "Vaccine", "entity_name": "流感疫苗"},
            },
            {
                "id": "居家隔离",
                "labels": ["PublicHealthMeasure"],
                "properties": {
                    "entity_type": "PublicHealthMeasure",
                    "entity_name": "居家隔离",
                },
            },
            {
                "id": "急性呼吸窘迫综合征（ARDS）",
                "labels": ["Complication"],
                "properties": {
                    "entity_type": "Complication",
                    "entity_name": "急性呼吸窘迫综合征（ARDS）",
                },
            },
        ],
        "edges": [],
        "is_truncated": True,
        "metadata": {"source": "unit-test"},
    }

    projected = project_medical_graph(payload)

    assert projected["metadata"]["source"] == "unit-test"
    assert [(group["key"], group["node_ids"]) for group in projected["metadata"]["medical_groups"]] == [
        ("complication", ["急性呼吸窘迫综合征（ARDS）"]),
        ("prevention", ["流感疫苗", "居家隔离"]),
    ]


def test_project_medical_graph_uses_endpoint_types_when_edge_order_is_canonicalized():
    from lightrag.medical_kg.graph_projection import project_medical_graph

    payload = {
        "nodes": [
            {
                "id": "流行性感冒",
                "labels": ["Disease"],
                "properties": {"entity_type": "Disease", "entity_name": "流行性感冒"},
            },
            {
                "id": "甲型流感病毒",
                "labels": ["Pathogen"],
                "properties": {
                    "entity_type": "Pathogen",
                    "entity_name": "甲型流感病毒",
                },
            },
            {
                "id": "奥司他韦",
                "labels": ["Drug"],
                "properties": {"entity_type": "Drug", "entity_name": "奥司他韦"},
            },
        ],
        "edges": [
            {
                "id": "e1",
                "type": "related",
                "source": "甲型流感病毒",
                "target": "流行性感冒",
                "properties": {"keywords": "病原导致"},
            },
            {
                "id": "e2",
                "type": "related",
                "source": "流行性感冒",
                "target": "奥司他韦",
                "properties": {"keywords": "推荐治疗"},
            },
        ],
        "is_truncated": False,
    }

    projected = project_medical_graph(payload)

    assert [(group["key"], group["node_ids"]) for group in projected["metadata"]["medical_groups"]] == [
        ("pathogen", ["甲型流感病毒"]),
        ("treatment", ["奥司他韦"]),
        ("other", ["流行性感冒"]),
    ]


def test_project_medical_graph_never_uses_keyword_fallback_to_reclassify_disease():
    from lightrag.medical_kg.graph_projection import project_medical_graph

    payload = {
        "nodes": [
            {
                "id": "流行性感冒",
                "labels": ["Disease"],
                "properties": {"entity_type": "Disease", "entity_name": "流行性感冒"},
            },
            {"id": "未知病原", "labels": [], "properties": {}},
            {"id": "未知药物", "labels": [], "properties": {}},
        ],
        "edges": [
            {
                "id": "e1",
                "type": "related",
                "source": "未知病原",
                "target": "流行性感冒",
                "properties": {"keywords": "病原导致"},
            },
            {
                "id": "e2",
                "type": "related",
                "source": "流行性感冒",
                "target": "未知药物",
                "properties": {"keywords": "推荐治疗"},
            },
        ],
        "is_truncated": False,
    }

    projected = project_medical_graph(payload)

    assert [(group["key"], group["node_ids"]) for group in projected["metadata"]["medical_groups"]] == [
        ("other", ["流行性感冒", "未知病原", "未知药物"]),
    ]


def test_project_medical_graph_groups_synthetic_medical_hierarchy_parents():
    from lightrag.medical_kg.graph_projection import project_medical_graph

    payload = {
        "nodes": [
            {
                "id": "流感病毒",
                "labels": ["MedicalGroup"],
                "properties": {"entity_type": "MedicalGroup", "entity_name": "流感病毒"},
            },
            {
                "id": "全身症状",
                "labels": ["MedicalGroup"],
                "properties": {"entity_type": "MedicalGroup", "entity_name": "全身症状"},
            },
            {
                "id": "呼吸系统并发症",
                "labels": ["MedicalGroup"],
                "properties": {
                    "entity_type": "MedicalGroup",
                    "entity_name": "呼吸系统并发症",
                },
            },
        ],
        "edges": [],
        "is_truncated": False,
    }

    projected = project_medical_graph(payload)

    assert [(group["key"], group["node_ids"]) for group in projected["metadata"]["medical_groups"]] == [
        ("pathogen", ["流感病毒"]),
        ("symptom", ["全身症状"]),
        ("complication", ["呼吸系统并发症"]),
    ]


def test_project_medical_graph_uses_conservative_name_hints_for_untyped_complications():
    from lightrag.medical_kg.graph_projection import project_medical_graph

    payload = {
        "nodes": [
            {
                "id": "急性呼吸窘迫综合征（ARDS）",
                "labels": [],
                "properties": {
                    "entity_name": "急性呼吸窘迫综合征（ARDS）",
                },
            },
            {
                "id": "呼吸衰竭",
                "labels": ["Disease"],
                "properties": {
                    "entity_type": "Disease",
                    "entity_name": "呼吸衰竭",
                },
            },
            {
                "id": "流行性感冒",
                "labels": ["Disease"],
                "properties": {"entity_type": "Disease", "entity_name": "流行性感冒"},
            },
        ],
        "edges": [],
        "is_truncated": False,
    }

    projected = project_medical_graph(payload)

    assert [(group["key"], group["node_ids"]) for group in projected["metadata"]["medical_groups"]] == [
        ("complication", ["急性呼吸窘迫综合征（ARDS）", "呼吸衰竭"]),
        ("other", ["流行性感冒"]),
    ]


def test_project_medical_graph_adds_medium_expanded_browse_metadata():
    from lightrag.medical_kg.graph_projection import project_medical_graph

    payload = {
        "nodes": [
            {
                "id": "流行性感冒",
                "labels": ["Disease"],
                "properties": {"entity_type": "Disease", "entity_name": "流行性感冒"},
            },
            {
                "id": "临床表现",
                "labels": ["MedicalGroup"],
                "properties": {
                    "entity_type": "MedicalGroup",
                    "medical_group": "clinical_manifestation",
                },
            },
            {
                "id": "全身症状",
                "labels": ["MedicalGroup"],
                "properties": {
                    "entity_type": "MedicalGroup",
                    "medical_group": "clinical_manifestation",
                },
            },
            {
                "id": "呼吸道症状",
                "labels": ["MedicalGroup"],
                "properties": {
                    "entity_type": "MedicalGroup",
                    "medical_group": "clinical_manifestation",
                },
            },
            {
                "id": "发热",
                "labels": ["Symptom"],
                "properties": {"entity_type": "Symptom", "entity_name": "发热"},
            },
            {
                "id": "咳嗽",
                "labels": ["Symptom"],
                "properties": {"entity_type": "Symptom", "entity_name": "咳嗽"},
            },
            {
                "id": "咽痛",
                "labels": ["Symptom"],
                "properties": {"entity_type": "Symptom", "entity_name": "咽痛"},
            },
            {
                "id": "流涕",
                "labels": ["Symptom"],
                "properties": {"entity_type": "Symptom", "entity_name": "流涕"},
            },
        ],
        "edges": [
            {
                "id": "e0",
                "source": "流行性感冒",
                "target": "临床表现",
                "type": "related",
                "properties": {"keywords": "临床表现"},
            },
            {
                "id": "e1",
                "source": "临床表现",
                "target": "全身症状",
                "type": "related",
                "properties": {"keywords": "属于"},
            },
            {
                "id": "e2",
                "source": "临床表现",
                "target": "呼吸道症状",
                "type": "related",
                "properties": {"keywords": "属于"},
            },
            {
                "id": "e3",
                "source": "发热",
                "target": "全身症状",
                "type": "related",
                "properties": {"keywords": "属于"},
            },
            {
                "id": "e4",
                "source": "咳嗽",
                "target": "呼吸道症状",
                "type": "related",
                "properties": {"keywords": "属于"},
            },
            {
                "id": "e5",
                "source": "咽痛",
                "target": "呼吸道症状",
                "type": "related",
                "properties": {"keywords": "属于"},
            },
            {
                "id": "e6",
                "source": "流涕",
                "target": "呼吸道症状",
                "type": "related",
                "properties": {"keywords": "属于"},
            },
        ],
        "is_truncated": False,
    }
    original_payload = deepcopy(payload)

    projected = project_medical_graph(payload, include_browse=True)

    assert payload == original_payload
    assert "medical_groups" in projected["metadata"]
    browse = projected["metadata"]["medical_browse"]
    assert browse["root_id"] == "流行性感冒"
    assert browse["default_depth"] == "medium"
    assert browse["category_order"][:3] == [
        "pathogen",
        "transmission_epidemiology",
        "clinical_manifestation",
    ]
    assert browse["node_roles"]["流行性感冒"] == "root"
    assert browse["node_roles"]["临床表现"] == "category"
    assert browse["node_roles"]["呼吸道症状"] == "subgroup"
    assert browse["node_roles"]["发热"] == "leaf"
    assert browse["collapsed_groups"][0] == {
        "id": "medical-collapsed:呼吸道症状",
        "label": "呼吸道症状 (3): 咳嗽、咽痛、流涕",
        "parent_id": "呼吸道症状",
        "child_ids": ["咳嗽", "咽痛", "流涕"],
        "count": 3,
        "examples": ["咳嗽", "咽痛", "流涕"],
    }
    assert browse["relation_details"]["e0"] == {
        "source": "流行性感冒",
        "target": "临床表现",
        "relation": "临床表现",
        "display": "临床表现：临床表现",
        "triple": "流行性感冒 - 临床表现 -> 临床表现",
    }


def test_project_medical_graph_prefers_requested_root_for_browse_metadata():
    from lightrag.medical_kg.graph_projection import project_medical_graph

    payload = {
        "nodes": [
            {
                "id": "急性呼吸窘迫综合征（ARDS）",
                "labels": ["Disease"],
                "properties": {
                    "entity_type": "Disease",
                    "entity_name": "急性呼吸窘迫综合征（ARDS）",
                },
            },
            {
                "id": "流行性感冒",
                "labels": ["Disease"],
                "properties": {"entity_type": "Disease", "entity_name": "流行性感冒"},
            },
        ],
        "edges": [],
        "is_truncated": False,
    }

    projected = project_medical_graph(
        payload,
        include_browse=True,
        root_id="流行性感冒",
    )

    assert projected["metadata"]["medical_browse"]["root_id"] == "流行性感冒"
    assert projected["metadata"]["medical_browse"]["node_roles"]["流行性感冒"] == "root"
    assert (
        projected["metadata"]["medical_browse"]["node_roles"]["急性呼吸窘迫综合征（ARDS）"]
        == "leaf"
    )


def test_graphs_route_adds_medical_metadata_only_when_requested():
    rag = SimpleNamespace(get_knowledge_graph=AsyncMock(return_value=_knowledge_graph()))
    app = FastAPI()
    app.include_router(create_graph_routes(rag, api_key=_API_KEY))
    client = TestClient(app)

    medical_response = client.get(
        "/graphs?label=流行性感冒&max_depth=2&max_nodes=20&medical_view=true",
        headers=_HEADERS,
    )
    default_response = client.get(
        "/graphs?label=流行性感冒&max_depth=2&max_nodes=20",
        headers=_HEADERS,
    )
    browse_response = client.get(
        "/graphs?label=流行性感冒&max_depth=2&max_nodes=20&medical_browse=true",
        headers=_HEADERS,
    )
    combined_response = client.get(
        "/graphs?label=流行性感冒&max_depth=2&max_nodes=20&medical_view=true&medical_browse=true",
        headers=_HEADERS,
    )

    assert medical_response.status_code == 200, medical_response.text
    assert "medical_groups" in medical_response.json()["metadata"]
    assert default_response.status_code == 200, default_response.text
    assert "metadata" not in default_response.json()
    assert browse_response.status_code == 200, browse_response.text
    assert "medical_browse" in browse_response.json()["metadata"]
    assert combined_response.status_code == 200, combined_response.text
    assert "medical_browse" in combined_response.json()["metadata"]
