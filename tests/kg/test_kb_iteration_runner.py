from pathlib import Path

import networkx as nx

from lightrag.kb_iteration.runner import run_iteration


def test_run_iteration_writes_core_artifacts(tmp_path: Path):
    storage_dir = tmp_path / "rag_storage" / "demo"
    input_dir = tmp_path / "inputs" / "demo"
    storage_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)
    (input_dir / "guideline.md").write_text("source", encoding="utf-8")

    graph = nx.MultiDiGraph()
    graph.add_node(
        "流行性感冒",
        entity_type="Disease",
        source_id="chunk-1",
        file_path="guideline.md",
    )
    graph.add_node(
        "发热",
        entity_type="Symptom",
        source_id="chunk-1",
        file_path="guideline.md",
    )
    graph.add_edge(
        "流行性感冒",
        "发热",
        keywords="临床表现",
        source_id="chunk-1",
        file_path="guideline.md",
    )
    nx.write_graphml(graph, storage_dir / "graph_chunk_entity_relation.graphml")

    result = run_iteration(
        workspace="demo",
        storage_root=tmp_path / "rag_storage",
        input_root=tmp_path / "inputs",
        output_root=tmp_path / "work" / "kb-iteration",
        profile="clinical_guideline_zh",
    )

    assert result.output_dir == tmp_path / "work" / "kb-iteration" / "demo"
    assert (result.output_dir / "snapshots" / "kg_snapshot.json").exists()
    assert (result.output_dir / "kb_context.md").exists()
    assert (result.output_dir / "quality_report.md").exists()
    assert (result.output_dir / "iteration_log.md").exists()
