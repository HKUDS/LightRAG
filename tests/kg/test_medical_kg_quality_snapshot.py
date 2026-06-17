from pathlib import Path

from lightrag.medical_kg.hierarchy import build_medical_hierarchy_edges
from lightrag.medical_kg.normalizer import normalize_medical_extraction


def _node(name: str, entity_type: str, description: str = "") -> dict[str, object]:
    return {
        "entity_name": name,
        "entity_type": entity_type,
        "description": description or name,
        "source_id": "fixture-chunk",
        "file_path": "fixture.md",
        "timestamp": 0,
    }


def _edge(
    source: str,
    target: str,
    keywords: str,
    description: str = "",
) -> dict[str, object]:
    return {
        "src_id": source,
        "tgt_id": target,
        "weight": 1.0,
        "keywords": keywords,
        "description": description or f"{source} {keywords} {target}",
        "source_id": "fixture-chunk",
        "file_path": "fixture.md",
        "timestamp": 0,
    }


def test_medical_kg_profile_quality_snapshot_removes_values_and_preserves_hierarchy():
    nodes = {
        "流感": [_node("流感", "Disease")],
        "流行性感冒": [_node("流行性感冒", "Disease")],
        "ARDS": [_node("ARDS", "Complication")],
        "急性呼吸窘迫综合征（ARDS）": [
            _node("急性呼吸窘迫综合征（ARDS）", "Complication")
        ],
        "甲型流感病毒": [_node("甲型流感病毒", "Pathogen")],
        "发热": [_node("发热", "Symptom")],
        "75 mg": [_node("75 mg", "Dosage")],
        "每日2次": [_node("每日2次", "Dosage")],
        "发病48小时内": [_node("发病48小时内", "TimeCourse")],
    }
    edges = {
        ("流感", "ARDS"): [_edge("流感", "ARDS", "并发")],
        ("甲型流感病毒", "流感"): [_edge("甲型流感病毒", "流感", "病原")],
        ("流感", "发热"): [_edge("流感", "发热", "表现为")],
        ("奥司他韦", "75 mg"): [
            _edge("奥司他韦", "75 mg", "剂量", "奥司他韦每次75 mg。")
        ],
        ("奥司他韦", "每日2次"): [
            _edge("奥司他韦", "每日2次", "剂量", "每日2次给药。")
        ],
        ("奥司他韦", "发病48小时内"): [
            _edge("奥司他韦", "发病48小时内", "用药", "发病48小时内用药。")
        ],
    }

    normalized = normalize_medical_extraction(nodes, edges, enabled=True)
    hierarchy_nodes, hierarchy_edges = build_medical_hierarchy_edges(normalized.nodes)

    assert "75 mg" not in hierarchy_nodes
    assert "每日2次" not in hierarchy_nodes
    assert "发病48小时内" not in hierarchy_nodes
    assert not {"流感", "流行性感冒"}.issubset(hierarchy_nodes)
    assert not {"ARDS", "急性呼吸窘迫综合征（ARDS）"}.issubset(hierarchy_nodes)
    assert ("甲型流感病毒", "流感病毒") in hierarchy_edges
    assert ("发热", "全身症状") in hierarchy_edges
    assert ("急性呼吸窘迫综合征（ARDS）", "呼吸系统并发症") in hierarchy_edges


def test_medical_kg_profile_documentation_states_scope_and_quality_thresholds():
    documentation = Path("docs/MedicalKGProfile-zh.md").read_text(encoding="utf-8")

    required_fragments = [
        "## 不适用范围",
        "药品说明书全量结构化",
        "病历时间线",
        "真实世界队列数据",
        "## 质量阈值",
        "值型节点数量为 0",
        "明确同义词重复数量为 0",
        "关键病原体层级存在",
        "关键症状分组存在",
        "关键并发症分组存在",
        "流行性感冒",
        "孤立剂量、阈值、页码、表格碎片",
        "MedicalKGCleanWorkspace-zh.md",
        "medical_view=true",
        "medical_browse=true",
    ]

    for fragment in required_fragments:
        assert fragment in documentation


def test_clean_workspace_documentation_states_delete_and_rebuild_policy():
    doc = Path("docs/MedicalKGCleanWorkspace-zh.md").read_text(encoding="utf-8")

    required = [
        "旧 workspace 可以删除",
        "新 workspace",
        "MEDICAL_KG_PROFILE=clinical_guideline_zh",
        "ENTITY_EXTRACTION_USE_JSON=true",
        "不要删除配置目录之外的路径",
        "重新导入原始医学文档",
    ]
    for fragment in required:
        assert fragment in doc
