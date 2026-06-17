from pathlib import Path

import networkx as nx

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
    graph.add_edge(
        "流行性感冒",
        "发热",
        keywords="临床表现",
        source_id="chunk-1",
        file_path="a.md",
    )
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)
    snapshot = build_snapshot_from_graphml(graph_path, workspace="demo")

    output_dir = tmp_path / "work" / "kb-iteration" / "demo"
    written = write_snapshot_artifacts(snapshot, output_dir)

    assert (output_dir / "snapshots" / "kg_snapshot.json").exists()
    assert (output_dir / "snapshots" / "entity_stats.json").exists()
    assert (output_dir / "snapshots" / "relation_stats.json").exists()
    assert written["kg_snapshot"].name == "kg_snapshot.json"
