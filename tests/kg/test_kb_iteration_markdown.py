from pathlib import Path

from lightrag.kb_iteration.markdown import write_markdown_memory
from lightrag.kb_iteration.models import KGSnapshot, SnapshotEdge, SnapshotNode


RULE_MEMORY_FILES = {
    "quality_rules.md",
    "known_issues.md",
    "accepted_changes.md",
    "rejected_changes.md",
    "approval_queue.md",
    "improvement_backlog.md",
    "iteration_log.md",
    "diff_report.md",
}


def _snapshot() -> KGSnapshot:
    return KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=["guideline.md"],
        nodes=[
            SnapshotNode(
                "流行性感冒",
                "流行性感冒",
                "Disease",
                "疾病",
                "chunk-1",
                "guideline.md",
            ),
            SnapshotNode("发热", "发热", "Symptom", "症状", "chunk-1", "guideline.md"),
            SnapshotNode(
                "症状",
                "症状",
                "MedicalGroup",
                "医学分组",
                "chunk-2",
                "taxonomy.md",
                {"aliases": ["临床表现"]},
            ),
        ],
        edges=[
            SnapshotEdge(
                "e1",
                "流行性感冒",
                "发热",
                "临床表现",
                "表现为发热",
                "chunk-1",
                "guideline.md",
            ),
            SnapshotEdge(
                "e2",
                "发热",
                "症状",
                "症状归类",
                "发热属于症状",
                "chunk-2",
                "taxonomy.md",
            ),
        ],
        metadata={
            "profile": "clinical_guideline_zh",
            "graph_node_count": 3,
            "graph_edge_count": 2,
        },
    )


def test_write_markdown_memory_creates_llm_entrypoint_and_catalogs(
    tmp_path: Path,
):
    paths = write_markdown_memory(_snapshot(), tmp_path)

    assert (tmp_path / "kb_context.md").exists()
    assert (tmp_path / "entity_catalog.md").exists()
    assert (tmp_path / "relation_catalog.md").exists()
    assert (tmp_path / "kg_structure.md").exists()
    assert "influenza_medical_v1" in (
        tmp_path / "kb_context.md"
    ).read_text(encoding="utf-8")
    assert "Disease" in (tmp_path / "entity_catalog.md").read_text(
        encoding="utf-8"
    )
    assert "临床表现" in (tmp_path / "relation_catalog.md").read_text(
        encoding="utf-8"
    )
    assert paths["kb_context"].name == "kb_context.md"


def test_kb_context_summarizes_snapshot_and_links_detail_files(tmp_path: Path):
    write_markdown_memory(_snapshot(), tmp_path)

    content = (tmp_path / "kb_context.md").read_text(encoding="utf-8")

    assert "Workspace: influenza_medical_v1" in content
    assert "Profile: clinical_guideline_zh" in content
    assert "guideline.md" in content
    assert "Nodes: 3" in content
    assert "Edges: 2" in content
    assert "Disease: 1" in content
    assert "Symptom: 1" in content
    assert "临床表现: 1" in content
    assert "症状归类: 1" in content
    assert "Latest Score Summary" in content
    assert "Not generated yet" in content
    assert "[Entity Catalog](entity_catalog.md)" in content
    assert "[Relation Catalog](relation_catalog.md)" in content
    assert "[KG Structure](kg_structure.md)" in content


def test_entity_catalog_groups_by_type_with_source_metadata(tmp_path: Path):
    write_markdown_memory(_snapshot(), tmp_path)

    content = (tmp_path / "entity_catalog.md").read_text(encoding="utf-8")

    assert "## Disease (1)" in content
    assert "## MedicalGroup (1)" in content
    assert "- 流行性感冒" in content
    assert "Description: 疾病" in content
    assert "Source: guideline.md / chunk-1" in content
    assert "Aliases: 临床表现" in content


def test_entity_catalog_sorts_set_aliases_deterministically(tmp_path: Path):
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode(
                "entity-1",
                "Entity One",
                "Finding",
                properties={"aliases": {"beta", "alpha"}},
            ),
        ],
        edges=[],
        metadata={},
    )

    write_markdown_memory(snapshot, tmp_path)

    content = (tmp_path / "entity_catalog.md").read_text(encoding="utf-8")
    assert "Aliases: alpha, beta" in content


def test_relation_catalog_groups_by_keyword_and_preserves_direction(
    tmp_path: Path,
):
    write_markdown_memory(_snapshot(), tmp_path)

    content = (tmp_path / "relation_catalog.md").read_text(encoding="utf-8")

    assert "## 临床表现 (1)" in content
    assert "## 症状归类 (1)" in content
    assert "流行性感冒 -> 发热" in content
    assert "发热 -> 症状" in content
    assert "Description: 表现为发热" in content
    assert "Source: guideline.md / chunk-1" in content


def test_kg_structure_lists_hierarchy_like_edges(tmp_path: Path):
    write_markdown_memory(_snapshot(), tmp_path)

    content = (tmp_path / "kg_structure.md").read_text(encoding="utf-8")

    assert "发热 -> 症状" in content
    assert "症状归类" in content
    assert "No hierarchy-like edges detected." not in content


def test_kg_structure_reports_when_no_hierarchy_edges(tmp_path: Path):
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[SnapshotNode("A", "A", "Disease"), SnapshotNode("B", "B", "Symptom")],
        edges=[SnapshotEdge("e1", "A", "B", "clinical_manifestation")],
        metadata={},
    )

    write_markdown_memory(snapshot, tmp_path)

    assert "No hierarchy-like edges detected." in (
        tmp_path / "kg_structure.md"
    ).read_text(encoding="utf-8")


def test_rule_memory_files_are_initialized_without_overwriting_existing(
    tmp_path: Path,
):
    existing = tmp_path / "quality_rules.md"
    existing.write_text("# Existing Rules\n\nKeep me.\n", encoding="utf-8")

    write_markdown_memory(_snapshot(), tmp_path)

    for filename in RULE_MEMORY_FILES:
        assert (tmp_path / filename).exists()
    assert existing.read_text(encoding="utf-8") == "# Existing Rules\n\nKeep me.\n"
    assert "# Known Issues" in (tmp_path / "known_issues.md").read_text(
        encoding="utf-8"
    )
