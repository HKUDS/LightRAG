import json
from pathlib import Path

from lightrag.kb_iteration.models import KGSnapshot, SnapshotEdge, SnapshotNode
from lightrag.kb_iteration.quality import (
    evaluate_snapshot_quality,
    write_quality_artifacts,
)
from lightrag.medical_kg.ontology import TOP_LEVEL_MEDICAL_CATEGORIES


def test_quality_flags_value_nodes_generic_relations_and_missing_evidence():
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode("流行性感冒", "流行性感冒", "Disease"),
            SnapshotNode("75 mg", "75 mg", "Dosage"),
            SnapshotNode("发热", "发热", "Symptom", source_id="chunk-1", file_path="guide.md"),
        ],
        edges=[
            SnapshotEdge("e1", "流行性感冒", "75 mg", "邻接"),
            SnapshotEdge(
                "e2",
                "流行性感冒",
                "发热",
                "临床表现",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["value_like_node_count"] == 1
    assert score.metrics["generic_relation_count"] == 1
    assert score.metrics["missing_edge_source_count"] == 1
    assert score.subscores["entity_hygiene"] < 100
    assert score.subscores["relation_semantics"] < 100
    assert score.overall < 100
    assert any(f.severity == "high" for f in score.findings)


def test_write_quality_artifacts_creates_score_json_and_report(tmp_path: Path):
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[SnapshotNode("Disease", "Disease", "Disease", "desc", "chunk-1", "guide.md")],
        edges=[],
        metadata={},
    )
    score = evaluate_snapshot_quality(snapshot)

    written = write_quality_artifacts(score, tmp_path)

    assert written == {
        "quality_score": tmp_path / "snapshots" / "quality_score.json",
        "quality_report": tmp_path / "quality_report.md",
    }
    assert written["quality_score"].exists()
    assert written["quality_report"].exists()

    payload = json.loads(written["quality_score"].read_text(encoding="utf-8"))
    assert set(payload) == {
        "overall",
        "subscores",
        "metrics",
        "findings",
        "critical_blockers",
    }
    assert payload["overall"] == score.overall
    assert payload["subscores"] == score.subscores
    assert payload["metrics"] == score.metrics
    assert payload["critical_blockers"] == score.critical_blockers

    report = written["quality_report"].read_text(encoding="utf-8")
    assert f"Overall score: {score.overall}" in report
    assert "entity_hygiene" in report
    assert "value_like_node_count" in report


def test_generic_relation_detection_handles_empty_english_chinese_and_merged_tokens():
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode("a", "A", "Disease", source_id="chunk-1", file_path="a.md"),
            SnapshotNode("b", "B", "Symptom", source_id="chunk-1", file_path="a.md"),
        ],
        edges=[
            SnapshotEdge("e1", "a", "b", "相关", source_id="chunk-1", file_path="a.md"),
            SnapshotEdge("e2", "a", "b", "邻接", source_id="chunk-1", file_path="a.md"),
            SnapshotEdge("e3", "a", "b", "", source_id="chunk-1", file_path="a.md"),
            SnapshotEdge("e4", "a", "b", "related", source_id="chunk-1", file_path="a.md"),
            SnapshotEdge(
                "e5",
                "a",
                "b",
                "clinical manifestation, related",
                source_id="chunk-1",
                file_path="a.md",
            ),
            SnapshotEdge(
                "e6",
                "a",
                "b",
                "clinical manifestation",
                source_id="chunk-1",
                file_path="a.md",
            ),
        ],
        metadata={},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["generic_relation_count"] == 5


def test_missing_evidence_counts_node_and_edge_source_fields_separately():
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode("n1", "N1", "Disease", source_id="", file_path="a.md"),
            SnapshotNode("n2", "N2", "Symptom", source_id="chunk-1", file_path=""),
            SnapshotNode("n3", "N3", "Treatment", source_id="", file_path=""),
        ],
        edges=[
            SnapshotEdge("e1", "n1", "n2", "causes", source_id="", file_path="a.md"),
            SnapshotEdge("e2", "n1", "n3", "treats", source_id="chunk-1", file_path=""),
            SnapshotEdge("e3", "n2", "n3", "prevents", source_id="", file_path=""),
        ],
        metadata={},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["missing_node_source_count"] == 2
    assert score.metrics["missing_node_file_path_count"] == 2
    assert score.metrics["missing_edge_source_count"] == 2
    assert score.metrics["missing_edge_file_path_count"] == 2


def test_medical_hierarchy_coverage_improves_when_required_labels_are_present():
    labels = [category.label for category in TOP_LEVEL_MEDICAL_CATEGORIES]
    sparse = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode(
                "flu",
                "流行性感冒",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        edges=[],
        metadata={"profile": "clinical_guideline_zh"},
    )
    covered = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode(
                label,
                label,
                "MedicalCategory",
                source_id="chunk-1",
                file_path="guide.md",
            )
            for label in labels
        ],
        edges=[],
        metadata={"profile": "clinical_guideline_zh"},
    )

    sparse_score = evaluate_snapshot_quality(sparse)
    covered_score = evaluate_snapshot_quality(covered)

    assert sparse_score.metrics["hierarchy_required_branch_count"] == len(labels)
    assert sparse_score.metrics["hierarchy_present_branch_count"] == 0
    assert sparse_score.metrics["hierarchy_missing_branch_count"] == len(labels)
    assert covered_score.metrics["hierarchy_present_branch_count"] == len(labels)
    assert covered_score.metrics["hierarchy_missing_branch_count"] == 0
    assert (
        covered_score.subscores["hierarchy_completeness"]
        > sparse_score.subscores["hierarchy_completeness"]
    )


def test_disease_hub_overload_ratio_uses_max_disease_direct_edges():
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode("flu", "Flu", "Disease", source_id="chunk-1", file_path="guide.md"),
            SnapshotNode("cold", "Cold", "Disease", source_id="chunk-1", file_path="guide.md"),
            SnapshotNode("fever", "Fever", "Symptom", source_id="chunk-1", file_path="guide.md"),
            SnapshotNode("cough", "Cough", "Symptom", source_id="chunk-1", file_path="guide.md"),
            SnapshotNode("rest", "Rest", "Treatment", source_id="chunk-1", file_path="guide.md"),
        ],
        edges=[
            SnapshotEdge("e1", "flu", "fever", "manifestation", source_id="chunk-1", file_path="guide.md"),
            SnapshotEdge("e2", "flu", "cough", "manifestation", source_id="chunk-1", file_path="guide.md"),
            SnapshotEdge("e3", "flu", "rest", "treatment", source_id="chunk-1", file_path="guide.md"),
            SnapshotEdge("e4", "cold", "cough", "manifestation", source_id="chunk-1", file_path="guide.md"),
        ],
        metadata={},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["disease_hub_overload_ratio"] == 0.75
    assert score.subscores["web_readability"] < 100


def test_findings_expose_required_review_fields():
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[SnapshotNode("75 mg", "75 mg", "Dosage")],
        edges=[SnapshotEdge("e1", "75 mg", "missing", "related")],
        metadata={},
    )

    score = evaluate_snapshot_quality(snapshot)

    finding = score.findings[0]
    payload = finding.to_dict()
    assert set(payload) == {
        "severity",
        "category",
        "message",
        "evidence",
        "suggested_fix_type",
        "requires_approval",
    }
    assert finding.severity
    assert finding.category
    assert finding.message
    assert finding.evidence
    assert finding.suggested_fix_type
    assert isinstance(finding.requires_approval, bool)
