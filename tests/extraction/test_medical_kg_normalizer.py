from lightrag.medical_kg.normalizer import normalize_medical_extraction
from lightrag.medical_kg.ontology import (
    is_value_like_entity,
    normalize_relation_keyword,
)


def test_normalize_merges_medical_synonyms_with_real_extraction_shape() -> None:
    nodes = {
        "流感": [
            {
                "entity_name": "流感",
                "entity_type": "disease",
                "description": "甲型、乙型流感均可引起急性呼吸道感染。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
        "流行性感冒": [
            {
                "entity_name": "流行性感冒",
                "entity_type": "disease",
                "description": "流行性感冒是由流感病毒引起的急性呼吸道传染病。",
                "source_id": "chunk-2",
                "file_path": "guide.md",
            }
        ],
        "ARDS": [
            {
                "entity_name": "ARDS",
                "entity_type": "complication",
                "description": "严重病例可出现ARDS。",
                "source_id": "chunk-3",
                "file_path": "icu.md",
            }
        ],
    }
    edges = {
        ("流感", "ARDS"): [
            {
                "src_id": "流感",
                "tgt_id": "ARDS",
                "keywords": "并发",
                "description": "重症流感有并发ARDS风险。",
            }
        ]
    }

    result = normalize_medical_extraction(nodes, edges, enabled=True)

    assert "流感" not in result.nodes
    assert "ARDS" not in result.nodes
    assert "流行性感冒" in result.nodes
    assert "急性呼吸窘迫综合征（ARDS）" in result.nodes
    assert isinstance(result.nodes["流行性感冒"], list)
    assert len(result.nodes["流行性感冒"]) >= 2
    assert {
        node["entity_name"] for node in result.nodes["流行性感冒"]
    } == {"流行性感冒"}
    assert {
        node["entity_name"] for node in result.nodes["急性呼吸窘迫综合征（ARDS）"]
    } == {"急性呼吸窘迫综合征（ARDS）"}
    assert ("流行性感冒", "急性呼吸窘迫综合征（ARDS）") in result.edges
    normalized_edge = result.edges[("流行性感冒", "急性呼吸窘迫综合征（ARDS）")][0]
    assert normalized_edge["src_id"] == "流行性感冒"
    assert normalized_edge["tgt_id"] == "急性呼吸窘迫综合征（ARDS）"
    assert normalized_edge["keywords"] == "并发风险"


def test_normalize_drops_value_like_entities_and_preserves_dosage() -> None:
    nodes = {
        "奥司他韦": [
            {
                "entity_name": "奥司他韦",
                "entity_type": "drug",
                "description": "神经氨酸酶抑制剂。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
        "75 mg": [
            {
                "entity_name": "75 mg",
                "entity_type": "dosage",
                "description": "成人推荐剂量。",
                "source_id": "chunk-2",
                "file_path": "dosage.md",
                "timestamp": 20,
            }
        ],
    }
    edges = {
        ("奥司他韦", "75 mg"): [
            {
                "src_id": "奥司他韦",
                "tgt_id": "75 mg",
                "keywords": "剂量",
                "description": "成人每次一粒，每日2次，疗程5天。",
                "weight": 1.0,
                "source_id": "chunk-edge",
                "file_path": "guide.md",
                "timestamp": 30,
            }
        ]
    }

    result = normalize_medical_extraction(nodes, edges, enabled=True)

    assert "75 mg" not in result.nodes
    assert result.dropped_nodes["75 mg"].reason == "value_like_entity"
    assert result.dropped_nodes["75 mg"].replacement == "奥司他韦"
    assert ("奥司他韦", "奥司他韦") not in result.edges
    assert ("奥司他韦", "75 mg") not in result.edges
    fragments = result.nodes["奥司他韦"]
    assert fragments[0]["description"] == "神经氨酸酶抑制剂。"
    assert {fragment["source_id"] for fragment in fragments} == {
        "chunk-1",
        "chunk-edge",
        "chunk-2",
    }
    assert all("<SEP>" not in fragment["source_id"] for fragment in fragments)
    edge_annotation = next(
        fragment for fragment in fragments if fragment["source_id"] == "chunk-edge"
    )
    value_annotation = next(
        fragment for fragment in fragments if fragment["source_id"] == "chunk-2"
    )
    assert "成人每次一粒，每日2次，疗程5天。" in edge_annotation["description"]
    assert "75 mg" in edge_annotation["description"]
    assert "成人推荐剂量" not in edge_annotation["description"]
    assert edge_annotation["file_path"] == "guide.md"
    assert "75 mg" in value_annotation["description"]
    assert "成人推荐剂量" in value_annotation["description"]
    assert value_annotation["file_path"] == "dosage.md"


def test_normalize_preserves_value_annotation_with_same_source_as_edge() -> None:
    nodes = {
        "奥司他韦": [
            {
                "entity_name": "奥司他韦",
                "entity_type": "drug",
                "description": "神经氨酸酶抑制剂。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
        "75 mg": [
            {
                "entity_name": "75 mg",
                "entity_type": "dosage",
                "description": "成人推荐剂量。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "timestamp": 20,
            }
        ],
    }
    edges = {
        ("奥司他韦", "75 mg"): [
            {
                "src_id": "奥司他韦",
                "tgt_id": "75 mg",
                "keywords": "剂量",
                "description": "成人每日2次。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "timestamp": 30,
            }
        ]
    }

    result = normalize_medical_extraction(nodes, edges, enabled=True)

    descriptions = [
        fragment["description"] for fragment in result.nodes["奥司他韦"]
    ]
    assert descriptions.count("神经氨酸酶抑制剂。") == 1
    assert any(
        "成人每日2次。" in description
        and "75 mg" in description
        and "成人推荐剂量" not in description
        for description in descriptions
    )
    assert any(
        "75 mg" in description and "成人推荐剂量" in description
        for description in descriptions
    )
    assert [fragment["source_id"] for fragment in result.nodes["奥司他韦"]] == [
        "chunk-1",
        "chunk-1",
        "chunk-1",
    ]
    assert all(
        "<SEP>" not in fragment["source_id"]
        for fragment in result.nodes["奥司他韦"]
    )
    assert ("奥司他韦", "75 mg") not in result.edges


def test_normalize_drops_value_endpoint_omitted_from_nodes() -> None:
    nodes = {
        "奥司他韦": [
            {
                "entity_name": "奥司他韦",
                "entity_type": "drug",
                "description": "神经氨酸酶抑制剂。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
    }
    edges = {
        ("奥司他韦", "75 mg"): [
            {
                "src_id": "奥司他韦",
                "tgt_id": "75 mg",
                "keywords": "剂量",
                "description": "成人每日2次。",
                "source_id": "chunk-edge",
                "file_path": "guide.md",
                "timestamp": 30,
            }
        ]
    }

    result = normalize_medical_extraction(nodes, edges, enabled=True)

    assert "75 mg" not in result.nodes
    assert ("奥司他韦", "75 mg") not in result.edges
    assert ("奥司他韦", "奥司他韦") not in result.edges
    assert result.dropped_nodes["75 mg"].reason == "value_like_entity"
    assert result.dropped_nodes["75 mg"].replacement == "奥司他韦"
    fragments = result.nodes["奥司他韦"]
    assert {fragment["source_id"] for fragment in fragments} == {
        "chunk-1",
        "chunk-edge",
    }
    assert all("<SEP>" not in fragment["source_id"] for fragment in fragments)
    annotation = next(
        fragment for fragment in fragments if fragment["source_id"] == "chunk-edge"
    )
    assert "成人每日2次。" in annotation["description"]
    assert "75 mg" in annotation["description"]


def test_normalize_drops_omitted_value_endpoints_with_extended_units() -> None:
    nodes = {
        "奥司他韦": [
            {
                "entity_name": "奥司他韦",
                "entity_type": "drug",
                "description": "神经氨酸酶抑制剂。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
    }
    edges = {
        ("奥司他韦", "0.5 g"): [
            {
                "src_id": "奥司他韦",
                "tgt_id": "0.5 g",
                "keywords": "剂量",
                "description": "单次用量。",
                "source_id": "chunk-g",
                "file_path": "dose-g.md",
            }
        ],
        ("奥司他韦", "5 ml"): [
            {
                "src_id": "奥司他韦",
                "tgt_id": "5 ml",
                "keywords": "剂量",
                "description": "液体剂量。",
                "source_id": "chunk-ml",
                "file_path": "dose-ml.md",
            }
        ],
        ("奥司他韦", "2次/日"): [
            {
                "src_id": "奥司他韦",
                "tgt_id": "2次/日",
                "keywords": "剂量",
                "description": "给药频率。",
                "source_id": "chunk-frequency",
                "file_path": "frequency.md",
            }
        ],
    }

    result = normalize_medical_extraction(nodes, edges, enabled=True)

    assert "0.5 g" not in result.nodes
    assert "5 ml" not in result.nodes
    assert "2次/日" not in result.nodes
    assert not result.edges
    descriptions = [
        fragment["description"] for fragment in result.nodes["奥司他韦"]
    ]
    assert any("0.5 g" in description for description in descriptions)
    assert any("5 ml" in description for description in descriptions)
    assert any("2次/日" in description for description in descriptions)


def test_normalize_preserves_value_chain_descriptions_on_retained_entity() -> None:
    nodes = {
        "奥司他韦": [
            {
                "entity_name": "奥司他韦",
                "entity_type": "Drug",
                "description": "抗病毒药物。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
        "75 mg": [
            {
                "entity_name": "75 mg",
                "entity_type": "Dosage",
                "description": "单次剂量。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
        "每日2次": [
            {
                "entity_name": "每日2次",
                "entity_type": "Dosage",
                "description": "给药频次。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
    }
    edges = {
        ("奥司他韦", "75 mg"): [
            {
                "src_id": "奥司他韦",
                "tgt_id": "75 mg",
                "keywords": "剂量",
                "description": "奥司他韦每次75 mg。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
        ("75 mg", "每日2次"): [
            {
                "src_id": "75 mg",
                "tgt_id": "每日2次",
                "keywords": "剂量",
                "description": "75 mg每日2次。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
    }

    result = normalize_medical_extraction(nodes, edges, enabled=True)

    descriptions = "<SEP>".join(
        str(fragment.get("description") or "")
        for fragment in result.nodes["奥司他韦"]
    )
    assert "75 mg" in descriptions
    assert "每日2次" in descriptions
    assert "给药频次" in descriptions
    assert result.dropped_nodes["75 mg"].replacement == "奥司他韦"
    assert result.dropped_nodes["每日2次"].replacement == "奥司他韦"


def test_normalize_splits_sep_sources_for_omitted_value_endpoint_edge() -> None:
    nodes = {
        "奥司他韦": [
            {
                "entity_name": "奥司他韦",
                "entity_type": "drug",
                "description": "神经氨酸酶抑制剂。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
    }
    edges = {
        ("奥司他韦", "75 mg"): [
            {
                "src_id": "奥司他韦",
                "tgt_id": "75 mg",
                "keywords": "剂量",
                "description": "分散来源剂量。",
                "source_id": "chunk-a<SEP>chunk-b",
                "file_path": "a.md<SEP>b.md",
                "timestamp": 30,
            }
        ]
    }

    result = normalize_medical_extraction(nodes, edges, enabled=True)

    fragments = result.nodes["奥司他韦"]
    by_source = {fragment["source_id"]: fragment for fragment in fragments}
    assert "chunk-a" in by_source
    assert "chunk-b" in by_source
    assert all("<SEP>" not in fragment["source_id"] for fragment in fragments)
    assert by_source["chunk-a"]["file_path"] == "a.md"
    assert by_source["chunk-b"]["file_path"] == "b.md"
    assert "75 mg" in by_source["chunk-a"]["description"]
    assert "75 mg" in by_source["chunk-b"]["description"]
    assert not result.edges


def test_value_like_patterns_cover_common_biomarkers_and_thresholds() -> None:
    assert is_value_like_entity("CRP", None)
    assert is_value_like_entity("PCT", None)
    assert is_value_like_entity("= 38", None)


def test_normalize_creates_missing_retained_endpoint_for_value_relation() -> None:
    nodes = {
        "75 mg": [
            {
                "entity_name": "75 mg",
                "entity_type": "dosage",
                "description": "成人推荐剂量。",
                "source_id": "chunk-2",
                "file_path": "dosage.md",
                "timestamp": 20,
            }
        ],
    }
    edges = {
        ("奥司他韦", "75 mg"): [
            {
                "src_id": "奥司他韦",
                "tgt_id": "75 mg",
                "keywords": "剂量",
                "description": "成人每日2次。",
                "weight": 1.0,
                "source_id": "chunk-edge",
                "file_path": "guide.md",
                "timestamp": 30,
            }
        ]
    }

    result = normalize_medical_extraction(nodes, edges, enabled=True)

    assert "奥司他韦" in result.nodes
    fragments = result.nodes["奥司他韦"]
    assert {fragment["source_id"] for fragment in fragments} == {
        "chunk-edge",
        "chunk-2",
    }
    assert all("<SEP>" not in fragment["source_id"] for fragment in fragments)
    edge_annotation = next(
        fragment for fragment in fragments if fragment["source_id"] == "chunk-edge"
    )
    value_annotation = next(
        fragment for fragment in fragments if fragment["source_id"] == "chunk-2"
    )
    assert edge_annotation["entity_name"] == "奥司他韦"
    assert edge_annotation["entity_type"] == "UNKNOWN"
    assert "成人每日2次。" in edge_annotation["description"]
    assert "75 mg" in edge_annotation["description"]
    assert "成人推荐剂量" not in edge_annotation["description"]
    assert edge_annotation["file_path"] == "guide.md"
    assert edge_annotation["timestamp"] == 30
    assert "75 mg" in value_annotation["description"]
    assert "成人推荐剂量" in value_annotation["description"]
    assert value_annotation["file_path"] == "dosage.md"
    assert value_annotation["timestamp"] == 20
    assert ("奥司他韦", "奥司他韦") not in result.edges
    assert ("奥司他韦", "75 mg") not in result.edges
    assert result.dropped_nodes["75 mg"].replacement == "奥司他韦"


def test_normalize_splits_sep_sources_for_dropped_value_node() -> None:
    nodes = {
        "奥司他韦": [
            {
                "entity_name": "奥司他韦",
                "entity_type": "drug",
                "description": "神经氨酸酶抑制剂。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
        "75 mg": [
            {
                "entity_name": "75 mg",
                "entity_type": "dosage",
                "description": "成人推荐剂量。",
                "source_id": "chunk-2<SEP>chunk-3",
                "file_path": "dosage2.md<SEP>dosage3.md",
                "timestamp": 20,
            }
        ],
    }
    edges = {
        ("奥司他韦", "75 mg"): [
            {
                "src_id": "奥司他韦",
                "tgt_id": "75 mg",
                "keywords": "剂量",
                "description": "成人每日2次。",
                "source_id": "chunk-edge",
                "file_path": "guide.md",
                "timestamp": 30,
            }
        ]
    }

    result = normalize_medical_extraction(nodes, edges, enabled=True)

    fragments = result.nodes["奥司他韦"]
    by_source = {fragment["source_id"]: fragment for fragment in fragments}
    assert "chunk-2" in by_source
    assert "chunk-3" in by_source
    assert all("<SEP>" not in fragment["source_id"] for fragment in fragments)
    assert by_source["chunk-2"]["file_path"] == "dosage2.md"
    assert by_source["chunk-3"]["file_path"] == "dosage3.md"
    assert "成人推荐剂量" in by_source["chunk-2"]["description"]
    assert "成人推荐剂量" in by_source["chunk-3"]["description"]
    assert not result.edges


def test_normalize_skips_value_annotations_with_empty_source_id() -> None:
    nodes = {
        "奥司他韦": [
            {
                "entity_name": "奥司他韦",
                "entity_type": "drug",
                "description": "神经氨酸酶抑制剂。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
        "75 mg": [
            {
                "entity_name": "75 mg",
                "entity_type": "dosage",
                "description": "成人推荐剂量。",
                "source_id": "",
                "file_path": "dosage.md",
            }
        ],
    }
    edges = {
        ("奥司他韦", "75 mg"): [
            {
                "src_id": "奥司他韦",
                "tgt_id": "75 mg",
                "keywords": "剂量",
                "description": "成人每日2次。",
                "file_path": "guide.md",
            }
        ]
    }

    result = normalize_medical_extraction(nodes, edges, enabled=True)

    assert ("奥司他韦", "75 mg") not in result.edges
    assert {fragment["source_id"] for fragment in result.nodes["奥司他韦"]} == {
        "chunk-1"
    }
    assert all(fragment["source_id"] for fragment in result.nodes["奥司他韦"])


def test_normalize_skips_original_and_canonical_self_loop_edges() -> None:
    nodes = {
        "流感": [
            {
                "entity_name": "流感",
                "entity_type": "disease",
                "description": "流感。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
        "流行性感冒": [
            {
                "entity_name": "流行性感冒",
                "entity_type": "disease",
                "description": "流行性感冒。",
                "source_id": "chunk-2",
                "file_path": "guide.md",
            }
        ],
    }
    edges = {
        ("流感", "流行性感冒"): [
            {
                "src_id": "流感",
                "tgt_id": "流行性感冒",
                "keywords": "建议",
                "description": "alias 折叠后会变成自环。",
            }
        ],
        ("流行性感冒", "流行性感冒"): [
            {
                "src_id": "流行性感冒",
                "tgt_id": "流行性感冒",
                "keywords": "建议",
                "description": "原始自环也应跳过。",
            }
        ],
    }

    result = normalize_medical_extraction(nodes, edges, enabled=True)

    assert result.edges == {}


def test_normalize_falls_back_to_edge_key_and_skips_empty_endpoints() -> None:
    nodes = {
        "流感": [
            {
                "entity_name": "流感",
                "entity_type": "disease",
                "description": "流感。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ],
        "ARDS": [
            {
                "entity_name": "ARDS",
                "entity_type": "complication",
                "description": "急性呼吸窘迫综合征。",
                "source_id": "chunk-2",
                "file_path": "guide.md",
            }
        ],
    }
    edges = {
        ("流感", "ARDS"): [
            {
                "src_id": "",
                "tgt_id": None,
                "keywords": "并发",
                "description": "端点缺失时回退到 edge key。",
            }
        ],
        ("", ""): [
            {
                "src_id": "",
                "tgt_id": None,
                "keywords": "并发",
                "description": "非法空端点应被跳过。",
            }
        ],
    }

    result = normalize_medical_extraction(nodes, edges, enabled=True)

    assert ("流行性感冒", "急性呼吸窘迫综合征（ARDS）") in result.edges
    assert ("", "") not in result.edges


def test_normalize_disabled_returns_original_content_unchanged() -> None:
    nodes = {
        "流感": [
            {
                "entity_name": "流感",
                "entity_type": "disease",
                "description": "流感。",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            }
        ]
    }
    edges = {
        ("流感", "ARDS"): [
            {
                "src_id": "流感",
                "tgt_id": "ARDS",
                "keywords": "并发",
                "description": "重症流感可并发ARDS。",
            }
        ]
    }

    result = normalize_medical_extraction(nodes, edges, enabled=False)

    assert result.nodes == nodes
    assert result.edges == edges
    assert result.nodes is nodes
    assert result.edges is edges
    assert result.dropped_nodes == {}


def test_normalize_relation_keyword_preserves_canonical_specific_keywords() -> None:
    assert normalize_relation_keyword("病原分型") == "病原分型"
    assert normalize_relation_keyword("症状归类") == "症状归类"
    assert normalize_relation_keyword("病原导致") == "病原导致"
