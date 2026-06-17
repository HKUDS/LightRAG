import json
from pathlib import Path

from lightrag.kb_iteration.diff import compare_snapshots, write_diff_report
from lightrag.kb_iteration.models import (
    KGSnapshot,
    QualityScore,
    SnapshotEdge,
    SnapshotNode,
)


def _snapshot(nodes, edges):
    return KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=nodes,
        edges=edges,
        metadata={},
    )


def _quality(generic_relation_count: int, evidence_grounding: int) -> QualityScore:
    return QualityScore(
        overall=80,
        subscores={"evidence_grounding": evidence_grounding},
        metrics={"generic_relation_count": generic_relation_count},
    )


def test_compare_snapshots_reports_added_removed_and_changed_relations(
    tmp_path: Path,
):
    before = _snapshot(
        [SnapshotNode("流行性感冒", "流行性感冒", "Disease")],
        [SnapshotEdge("e1", "流行性感冒", "发热", "邻接")],
    )
    after = _snapshot(
        [
            SnapshotNode("流行性感冒", "流行性感冒", "Disease"),
            SnapshotNode("发热", "发热", "Symptom"),
        ],
        [SnapshotEdge("e1", "流行性感冒", "发热", "临床表现")],
    )

    diff = compare_snapshots(before, after)
    path = write_diff_report(diff, tmp_path)

    assert diff.added_nodes == ["发热"]
    assert ("流行性感冒", "发热") in diff.changed_relation_keywords
    assert "Changed relation keywords" in path.read_text(encoding="utf-8")


def test_compare_snapshots_reports_removed_nodes_changed_types_and_edge_pairs():
    before = _snapshot(
        [
            SnapshotNode("A", "A", "Disease"),
            SnapshotNode("B", "B", "Symptom"),
            SnapshotNode("C", "C", "Treatment"),
        ],
        [
            SnapshotEdge("e1", "A", "B", "clinical"),
            SnapshotEdge("e2", "B", "C", "therapy"),
        ],
    )
    after = _snapshot(
        [
            SnapshotNode("A", "A", "Disease"),
            SnapshotNode("B", "B", "Sign"),
            SnapshotNode("D", "D", "Finding"),
        ],
        [
            SnapshotEdge("e1", "A", "B", "clinical"),
            SnapshotEdge("e3", "A", "D", "evidence"),
        ],
    )

    diff = compare_snapshots(before, after)

    assert diff.added_nodes == ["D"]
    assert diff.removed_nodes == ["C"]
    assert diff.changed_entity_types == {"B": {"before": "Symptom", "after": "Sign"}}
    assert diff.added_edge_pairs == [("A", "D")]
    assert diff.removed_edge_pairs == [("B", "C")]


def test_compare_snapshots_reports_dangerous_regression_flags_from_quality_scores():
    before = _snapshot(
        [
            SnapshotNode("流行性感冒", "流行性感冒", "Disease"),
            SnapshotNode("发热", "发热", "Symptom"),
        ],
        [SnapshotEdge("e1", "流行性感冒", "发热", "临床表现")],
    )
    after = _snapshot(
        [SnapshotNode("发热", "发热", "Symptom")],
        [SnapshotEdge("e1", "发热", "流行性感冒", "邻接")],
    )

    diff = compare_snapshots(
        before,
        after,
        before_quality=_quality(generic_relation_count=0, evidence_grounding=90),
        after_quality=_quality(generic_relation_count=1, evidence_grounding=70),
        core_disease_node_ids=["流行性感冒"],
    )

    assert diff.dangerous_regression_flags == [
        "core_disease_node_removed:流行性感冒",
        "evidence_coverage_decreased:90->70",
        "generic_relation_count_increased:0->1",
    ]


def test_explicit_empty_core_disease_list_disables_removed_disease_regression_flag():
    before = _snapshot(
        [SnapshotNode("Influenza", "Influenza", "Disease")],
        [],
    )
    after = _snapshot([], [])

    diff = compare_snapshots(before, after, core_disease_node_ids=[])

    assert diff.removed_nodes == ["Influenza"]
    assert diff.dangerous_regression_flags == []


def test_write_diff_report_creates_deterministic_json_summary(tmp_path: Path):
    before = _snapshot(
        [
            SnapshotNode("B", "B", "Symptom"),
            SnapshotNode("A", "A", "Disease"),
        ],
        [
            SnapshotEdge("e2", "B", "A", "related"),
            SnapshotEdge("e1", "A", "B", "related"),
        ],
    )
    after = _snapshot(
        [
            SnapshotNode("C", "C", "Treatment"),
            SnapshotNode("A", "A", "Condition"),
        ],
        [
            SnapshotEdge("e3", "A", "C", "therapy"),
            SnapshotEdge("e1", "A", "B", "clinical"),
        ],
    )

    diff = compare_snapshots(before, after)
    report_path = write_diff_report(diff, tmp_path)
    summary_path = tmp_path / "snapshots" / "diff_summary.json"

    assert report_path == tmp_path / "diff_report.md"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary == {
        "added_edge_pairs": [{"source": "A", "target": "C"}],
        "added_nodes": ["C"],
        "changed_entity_types": {"A": {"after": "Condition", "before": "Disease"}},
        "changed_relation_keywords": [
            {
                "after": "clinical",
                "before": "related",
                "source": "A",
                "target": "B",
            }
        ],
        "dangerous_regression_flags": [],
        "quality_delta": {},
        "removed_edge_pairs": [{"source": "B", "target": "A"}],
        "removed_nodes": ["B"],
        "workspace": {"after": "demo", "before": "demo"},
    }


def test_write_diff_report_keeps_changed_relation_edge_pairs_lossless(
    tmp_path: Path,
):
    before = _snapshot(
        [
            SnapshotNode("A -> B", "A -> B", "Entity"),
            SnapshotNode("A", "A", "Entity"),
            SnapshotNode("B -> C", "B -> C", "Entity"),
            SnapshotNode("C", "C", "Entity"),
        ],
        [
            SnapshotEdge("e1", "A -> B", "C", "old-left"),
            SnapshotEdge("e2", "A", "B -> C", "old-right"),
        ],
    )
    after = _snapshot(
        before.nodes,
        [
            SnapshotEdge("e1", "A -> B", "C", "new-left"),
            SnapshotEdge("e2", "A", "B -> C", "new-right"),
        ],
    )

    diff = compare_snapshots(before, after)
    write_diff_report(diff, tmp_path)

    summary = json.loads(
        (tmp_path / "snapshots" / "diff_summary.json").read_text(encoding="utf-8")
    )

    assert len(diff.changed_relation_keywords) == 2
    assert summary["changed_relation_keywords"] == [
        {
            "after": "new-right",
            "before": "old-right",
            "source": "A",
            "target": "B -> C",
        },
        {
            "after": "new-left",
            "before": "old-left",
            "source": "A -> B",
            "target": "C",
        },
    ]
