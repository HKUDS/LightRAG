from lightrag.medical_kg.hierarchy import (
    GROUP_NODE_TYPE,
    build_medical_hierarchy_edges,
)


def _node(name: str, entity_type: str = "Disease") -> dict[str, object]:
    return {
        "entity_name": name,
        "entity_type": entity_type,
        "description": f"{name} fragment",
        "source_id": "chunk-source",
        "file_path": "guide.md",
        "timestamp": 123,
    }


def test_build_medical_hierarchy_edges_adds_required_parent_edges() -> None:
    nodes = {
        "甲型流感病毒": [_node("甲型流感病毒", "Pathogen")],
        "A(H1N1)pdm09": [_node("A(H1N1)pdm09", "Pathogen")],
        "发热": [_node("发热", "Symptom")],
        "咳嗽": [_node("咳嗽", "Symptom")],
        "急性呼吸窘迫综合征（ARDS）": [
            _node("急性呼吸窘迫综合征（ARDS）", "Complication")
        ],
    }

    next_nodes, hierarchy_edges = build_medical_hierarchy_edges(nodes)

    expected_pairs = {
        ("甲型流感病毒", "流感病毒"),
        ("A(H1N1)pdm09", "甲型流感病毒"),
        ("发热", "全身症状"),
        ("咳嗽", "呼吸道症状"),
        ("急性呼吸窘迫综合征（ARDS）", "呼吸系统并发症"),
    }
    assert expected_pairs.issubset(set(hierarchy_edges))
    assert next_nodes is not nodes
    assert next_nodes["甲型流感病毒"] == nodes["甲型流感病毒"]

    for child, parent in expected_pairs:
        assert parent in next_nodes
        edge = hierarchy_edges[(child, parent)][0]
        assert edge["src_id"] == child
        assert edge["tgt_id"] == parent
        assert edge["weight"] == 0.0
        assert edge["keywords"] == "属于"
        assert edge["description"] == f"{child}属于{parent}。"
        assert edge["source_id"] == "medical_kg_profile"
        assert edge["file_path"] == "medical_kg_profile"
        assert edge["timestamp"] == 0
        assert edge["generated_by"] == "medical_kg_profile"


def test_build_medical_hierarchy_edges_creates_missing_group_nodes() -> None:
    nodes = {
        "发热": [_node("发热", "Symptom")],
    }

    next_nodes, hierarchy_edges = build_medical_hierarchy_edges(nodes)

    assert ("发热", "全身症状") in hierarchy_edges
    assert next_nodes["全身症状"] == [
        {
            "entity_name": "全身症状",
            "entity_type": GROUP_NODE_TYPE,
            "medical_group": "symptom",
            "description": "全身症状",
            "source_id": "medical_kg_profile",
            "file_path": "medical_kg_profile",
            "timestamp": 0,
            "generated_by": "medical_kg_profile",
        }
    ]


def test_build_medical_hierarchy_edges_completes_transitive_parent_chain() -> None:
    nodes = {
        "A(H1N1)pdm09": [_node("A(H1N1)pdm09", "Pathogen")],
    }

    next_nodes, hierarchy_edges = build_medical_hierarchy_edges(nodes)

    assert ("A(H1N1)pdm09", "甲型流感病毒") in hierarchy_edges
    assert ("甲型流感病毒", "流感病毒") in hierarchy_edges
    assert next_nodes["甲型流感病毒"][0]["entity_type"] == GROUP_NODE_TYPE
    assert next_nodes["流感病毒"][0]["entity_type"] == GROUP_NODE_TYPE


def test_build_medical_hierarchy_edges_preserves_existing_parent_node() -> None:
    parent_fragments = [_node("流感病毒", "PathogenGroup")]
    nodes = {
        "甲型流感病毒": [_node("甲型流感病毒", "Pathogen")],
        "流感病毒": parent_fragments,
    }

    next_nodes, hierarchy_edges = build_medical_hierarchy_edges(nodes)

    assert hierarchy_edges[("甲型流感病毒", "流感病毒")]
    assert next_nodes["流感病毒"] == parent_fragments
    assert next_nodes["流感病毒"] is not parent_fragments


def test_build_medical_hierarchy_edges_skips_missing_child_nodes() -> None:
    nodes = {
        "流感病毒": [_node("流感病毒", "PathogenGroup")],
        "全身症状": [_node("全身症状", GROUP_NODE_TYPE)],
    }

    next_nodes, hierarchy_edges = build_medical_hierarchy_edges(nodes)

    assert hierarchy_edges == {
        ("流感病毒", "病原体"): [
            {
                "src_id": "流感病毒",
                "tgt_id": "病原体",
                "weight": 0.0,
                "keywords": "属于",
                "description": "流感病毒属于病原体。",
                "source_id": "medical_kg_profile",
                "file_path": "medical_kg_profile",
                "timestamp": 0,
                "generated_by": "medical_kg_profile",
            }
        ],
        ("全身症状", "临床表现"): [
            {
                "src_id": "全身症状",
                "tgt_id": "临床表现",
                "weight": 0.0,
                "keywords": "属于",
                "description": "全身症状属于临床表现。",
                "source_id": "medical_kg_profile",
                "file_path": "medical_kg_profile",
                "timestamp": 0,
                "generated_by": "medical_kg_profile",
            }
        ],
    }
    assert "甲型流感病毒" not in next_nodes
    assert "病原体" in next_nodes
    assert "临床表现" in next_nodes


def test_build_medical_hierarchy_edges_adds_extended_medical_pairs() -> None:
    expected_pairs = {
        ("乙型流感病毒", "流感病毒"),
        ("A(H3N2)", "甲型流感病毒"),
        ("Victoria系", "乙型流感病毒"),
        ("头痛", "全身症状"),
        ("肌肉关节酸痛", "全身症状"),
        ("咽痛", "呼吸道症状"),
        ("流涕", "呼吸道症状"),
        ("呼吸困难", "呼吸道症状"),
        ("恶心", "消化道症状"),
        ("呕吐", "消化道症状"),
        ("腹泻", "消化道症状"),
        ("病毒性肺炎", "呼吸系统并发症"),
        ("心肌炎", "心脏并发症"),
        ("脑炎", "神经系统并发症"),
    }
    nodes = {
        child: [_node(child)]
        for child, _parent in expected_pairs
    }

    next_nodes, hierarchy_edges = build_medical_hierarchy_edges(nodes)

    assert expected_pairs.issubset(set(hierarchy_edges))
    for parent in {"消化道症状", "心脏并发症", "神经系统并发症"}:
        assert next_nodes[parent][0]["entity_type"] == GROUP_NODE_TYPE
        assert next_nodes[parent][0]["entity_name"] == parent
        assert next_nodes[parent][0]["medical_group"] in {
            "symptom",
            "complication",
        }
        assert next_nodes[parent][0]["source_id"] == "medical_kg_profile"
        assert next_nodes[parent][0]["timestamp"] == 0


def test_build_medical_hierarchy_edges_adds_influenza_top_level_category_spine() -> None:
    nodes = {
        "流感病毒": [_node("流感病毒", "MedicalGroup")],
        "全身症状": [_node("全身症状", "MedicalGroup")],
        "呼吸道症状": [_node("呼吸道症状", "MedicalGroup")],
        "消化道症状": [_node("消化道症状", "MedicalGroup")],
        "呼吸系统并发症": [_node("呼吸系统并发症", "MedicalGroup")],
        "心脏并发症": [_node("心脏并发症", "MedicalGroup")],
        "神经系统并发症": [_node("神经系统并发症", "MedicalGroup")],
    }

    next_nodes, hierarchy_edges = build_medical_hierarchy_edges(nodes)

    assert ("流感病毒", "病原体") in hierarchy_edges
    assert ("全身症状", "临床表现") in hierarchy_edges
    assert ("呼吸道症状", "临床表现") in hierarchy_edges
    assert ("消化道症状", "临床表现") in hierarchy_edges
    assert ("呼吸系统并发症", "并发症/重症") in hierarchy_edges
    assert ("心脏并发症", "并发症/重症") in hierarchy_edges
    assert ("神经系统并发症", "并发症/重症") in hierarchy_edges
    assert next_nodes["病原体"][0]["medical_group"] == "pathogen"
    assert next_nodes["临床表现"][0]["medical_group"] == "clinical_manifestation"
    assert next_nodes["并发症/重症"][0]["medical_group"] == "complication_severity"


def test_normalize_medical_category_key_maps_controlled_extensions() -> None:
    from lightrag.medical_kg.ontology import normalize_medical_category_key

    assert normalize_medical_category_key("复诊/随访") == "follow_up"
    assert normalize_medical_category_key("照护") == "nursing_care"
    assert normalize_medical_category_key("用药禁忌") == "contraindication"
    assert normalize_medical_category_key("未收录结果") == "other_medical"
