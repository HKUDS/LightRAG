import json
from pathlib import Path

import networkx as nx

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kb_iteration.snapshot import (
    build_snapshot_from_graphml,
    write_snapshot_artifacts,
)


def test_build_snapshot_from_graphml_preserves_medical_fields(tmp_path: Path):
    graph = nx.MultiDiGraph()
    graph.add_node(
        "流行性感冒",
        entity_type="Disease",
        description="急性呼吸道传染病",
        source_id="chunk-1",
        file_path="guideline.md",
    )
    graph.add_node(
        "发热",
        entity_type="Symptom",
        description="常见临床表现",
        source_id="chunk-1",
        file_path="guideline.md",
    )
    graph.add_edge(
        "流行性感冒",
        "发热",
        keywords="临床表现",
        description="流行性感冒可表现为发热",
        source_id="chunk-1",
        file_path="guideline.md",
        weight=1.0,
    )
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)

    snapshot = build_snapshot_from_graphml(
        graph_path,
        workspace="influenza_medical_v1",
        source_files=["guideline.md"],
        profile="clinical_guideline_zh",
    )

    assert snapshot.workspace == "influenza_medical_v1"
    assert snapshot.metadata["profile"] == "clinical_guideline_zh"
    assert snapshot.metadata["graph_node_count"] == 2
    assert snapshot.metadata["graph_edge_count"] == 1
    assert snapshot.nodes[0].id == "流行性感冒"
    assert snapshot.nodes[0].entity_type == "Disease"
    assert snapshot.edges[0].source == "流行性感冒"
    assert snapshot.edges[0].target == "发热"
    assert snapshot.edges[0].keywords == "临床表现"


def test_write_snapshot_artifacts_creates_machine_readable_stats(tmp_path: Path):
    graph = nx.MultiDiGraph()
    graph.add_node(
        "流行性感冒", entity_type="Disease", source_id="chunk-1", file_path="a.md"
    )
    graph.add_node("发热", entity_type="Symptom", source_id="chunk-1", file_path="a.md")
    graph.add_node("咳嗽", entity_type="Symptom", source_id="chunk-2", file_path="b.md")
    graph.add_node("抗病毒治疗", entity_type="Treatment", source_id="chunk-2", file_path="b.md")
    graph.add_edge(
        "流行性感冒",
        "发热",
        keywords="临床表现",
        source_id="chunk-1",
        file_path="a.md",
    )
    graph.add_edge(
        "流行性感冒",
        "咳嗽",
        keywords="临床表现",
        source_id="chunk-2",
        file_path="b.md",
    )
    graph.add_edge(
        "流行性感冒",
        "抗病毒治疗",
        keywords="治疗建议",
        source_id="chunk-2",
        file_path="b.md",
    )
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)
    snapshot = build_snapshot_from_graphml(graph_path, workspace="demo")

    output_dir = tmp_path / "work" / "kb-iteration" / "demo"
    written = write_snapshot_artifacts(snapshot, output_dir)

    assert (output_dir / "snapshots" / "kg_snapshot.json").exists()
    assert (output_dir / "snapshots" / "entity_stats.json").exists()
    assert (output_dir / "snapshots" / "relation_stats.json").exists()
    assert (output_dir / "snapshots" / "hierarchy_paths.json").exists()
    assert (output_dir / "snapshots" / "source_coverage.json").exists()
    assert written["kg_snapshot"].name == "kg_snapshot.json"

    entity_stats = json.loads(
        (output_dir / "snapshots" / "entity_stats.json").read_text(encoding="utf-8")
    )
    relation_stats = json.loads(
        (output_dir / "snapshots" / "relation_stats.json").read_text(encoding="utf-8")
    )
    source_coverage = json.loads(
        (output_dir / "snapshots" / "source_coverage.json").read_text(
            encoding="utf-8"
        )
    )

    assert entity_stats == [
        {"label": "Symptom", "count": 2},
        {"label": "Disease", "count": 1},
        {"label": "Treatment", "count": 1},
    ]
    assert relation_stats == [
        {"label": "临床表现", "count": 2},
        {"label": "治疗建议", "count": 1},
    ]
    assert source_coverage["file_path_counts"]["nodes"] == {"a.md": 2, "b.md": 2}
    assert source_coverage["file_path_counts"]["edges"] == {"b.md": 2, "a.md": 1}
    assert source_coverage["source_id_counts"]["nodes"] == {
        "chunk-1": 2,
        "chunk-2": 2,
    }
    assert source_coverage["source_id_counts"]["edges"] == {
        "chunk-2": 2,
        "chunk-1": 1,
    }


def test_write_snapshot_artifacts_splits_joined_provenance_fields(tmp_path: Path):
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Disease",
        entity_type="Disease",
        source_id=GRAPH_FIELD_SEP.join(["chunk-1", "", "chunk-2"]),
        file_path=GRAPH_FIELD_SEP.join(["a.md", "b.md"]),
    )
    graph.add_node(
        "Symptom",
        entity_type="Symptom",
        source_id="chunk-2",
        file_path="b.md",
    )
    graph.add_edge(
        "Disease",
        "Symptom",
        keywords="manifestation",
        source_id=GRAPH_FIELD_SEP.join(["chunk-2", "chunk-3"]),
        file_path=GRAPH_FIELD_SEP.join(["b.md", "", "c.md"]),
    )
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)
    snapshot = build_snapshot_from_graphml(graph_path, workspace="demo")

    output_dir = tmp_path / "work" / "kb-iteration" / "demo"
    write_snapshot_artifacts(snapshot, output_dir)

    source_coverage = json.loads(
        (output_dir / "snapshots" / "source_coverage.json").read_text(
            encoding="utf-8"
        )
    )

    assert source_coverage["file_path_counts"]["nodes"] == {"b.md": 2, "a.md": 1}
    assert source_coverage["file_path_counts"]["edges"] == {"b.md": 1, "c.md": 1}
    assert source_coverage["source_id_counts"]["nodes"] == {
        "chunk-2": 2,
        "chunk-1": 1,
    }
    assert source_coverage["source_id_counts"]["edges"] == {
        "chunk-2": 1,
        "chunk-3": 1,
    }


def test_build_snapshot_from_graphml_preserves_edge_ids_defaults_and_weight(
    tmp_path: Path,
):
    graph = nx.DiGraph()
    graph.add_node("A")
    graph.add_node("B", entity_type="Target")
    graph.add_edge("A", "B", id="rel-2", weight="2.5")
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)

    loaded = nx.read_graphml(graph_path)
    assert not loaded.is_multigraph()

    snapshot = build_snapshot_from_graphml(graph_path, workspace="demo")

    assert snapshot.nodes[0].id == "A"
    assert snapshot.nodes[0].label == ""
    assert snapshot.nodes[0].entity_type == ""
    assert snapshot.nodes[0].description == ""
    assert snapshot.nodes[0].source_id == ""
    assert snapshot.nodes[0].file_path == ""
    assert snapshot.edges[0].id == "rel-2"
    assert snapshot.edges[0].keywords == ""
    assert snapshot.edges[0].description == ""
    assert snapshot.edges[0].source_id == ""
    assert snapshot.edges[0].file_path == ""
    assert snapshot.edges[0].weight == 2.5


def test_build_snapshot_from_graphml_uses_directional_edge_metadata_for_undirected_graph(
    tmp_path: Path,
):
    graph = nx.Graph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_edge(
        "A",
        "B",
        id="B->A",
        source_node_id="B",
        target_node_id="A",
        keywords="has_manifestation",
    )
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)

    snapshot = build_snapshot_from_graphml(graph_path, workspace="demo")

    assert snapshot.edges[0].id == "B->A"
    assert snapshot.edges[0].source == "B"
    assert snapshot.edges[0].target == "A"
