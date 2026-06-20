import json
from pathlib import Path

from lightrag.constants import GRAPH_FIELD_SEP
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
        "details",
        "findings",
        "critical_blockers",
    }
    assert payload["overall"] == score.overall
    assert payload["subscores"] == score.subscores
    assert payload["metrics"] == score.metrics
    assert payload["details"] == score.details
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


def test_empty_snapshot_has_critical_blocker_and_zero_overall():
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[],
        edges=[],
        metadata={},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.critical_blockers
    assert score.overall == 0


def test_generic_relation_detection_handles_lightrag_and_chinese_separators():
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode("flu", "Flu", "Disease", source_id="chunk-1", file_path="a.md"),
            SnapshotNode("fever", "Fever", "Symptom", source_id="chunk-1", file_path="a.md"),
        ],
        edges=[
            SnapshotEdge(
                "e1",
                "flu",
                "fever",
                f"相关{GRAPH_FIELD_SEP}临床表现",
                source_id="chunk-1",
                file_path="a.md",
            ),
            SnapshotEdge(
                "e2",
                "flu",
                "fever",
                "相关，临床表现",
                source_id="chunk-1",
                file_path="a.md",
            ),
            SnapshotEdge(
                "e3",
                "flu",
                "fever",
                f"manifestation{GRAPH_FIELD_SEP}treatment",
                source_id="chunk-1",
                file_path="a.md",
            ),
        ],
        metadata={},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["generic_relation_count"] == 2


def test_quality_flags_clinical_manifestation_edges_with_reverse_direction():
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode("flu", "流行性感冒", "Disease", source_id="chunk-1", file_path="guide.md"),
            SnapshotNode("dry-cough", "干咳", "Symptom", source_id="chunk-1", file_path="guide.md"),
            SnapshotNode("malaise", "全身不适", "Symptom", source_id="chunk-2", file_path="guide.md"),
        ],
        edges=[
            SnapshotEdge(
                "edge-dry-cough-flu",
                "dry-cough",
                "flu",
                "临床表现",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "edge-flu-malaise",
                "flu",
                "malaise",
                "临床表现",
                source_id="chunk-2",
                file_path="guide.md",
            ),
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["clinical_relation_direction_issue_count"] == 1
    assert score.metrics["relation_semantic_issue_count"] == 1
    finding = next(
        finding
        for finding in score.findings
        if finding.suggested_fix_type == "normalize_relation_direction"
    )
    assert finding.category == "relation_semantics"
    assert finding.evidence == ["edge:edge-dry-cough-flu"]
    assert finding.requires_approval is True
    assert score.subscores["relation_semantics"] == 100


def test_quality_flags_taxonomy_keyword_on_direct_disease_symptom_relation():
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode("flu", "流行性感冒", "Disease", source_id="chunk-1", file_path="guide.md"),
            SnapshotNode("dry-cough", "干咳", "Symptom", source_id="chunk-1", file_path="guide.md"),
            SnapshotNode(
                "respiratory-symptom",
                "呼吸道症状",
                "MedicalGroup",
                source_id="medical_kg_profile",
                file_path="medical_kg_profile",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-dry-cough-flu",
                "dry-cough",
                "flu",
                "属于",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "edge-dry-cough-respiratory",
                "dry-cough",
                "respiratory-symptom",
                "属于",
                source_id="medical_kg_profile",
                file_path="medical_kg_profile",
            ),
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["taxonomy_relation_misuse_count"] == 1
    assert score.metrics["relation_semantic_issue_count"] == 1
    finding = next(
        finding
        for finding in score.findings
        if finding.suggested_fix_type == "replace_taxonomy_relation_keyword"
    )
    assert finding.evidence == ["edge:edge-dry-cough-flu"]


def test_quality_details_include_medical_schema_issues_for_disease_symptom_taxonomy() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "flu",
                "\u6d41\u884c\u6027\u611f\u5192",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "dry-cough",
                "\u5e72\u54b3",
                "Symptom",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-dry-cough-flu",
                "dry-cough",
                "flu",
                "\u5c5e\u4e8e",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["medical_schema_issue_count"] == 1
    issue = score.details["medical_schema_issues"][0]
    assert issue["issue_kind"] == "disease_symptom_taxonomy_misuse"
    assert issue["edge_id"] == "edge-dry-cough-flu"
    assert issue["suggested_action"] == "replace_relation"
    assert issue["candidate_predicates"] == ["has_manifestation"]


def test_quality_flags_reverse_clinical_manifestation_schema_issue_details() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "flu",
                "\u6d41\u884c\u6027\u611f\u5192",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "dry-cough",
                "\u5e72\u54b3",
                "Symptom",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-dry-cough-flu",
                "dry-cough",
                "flu",
                "\u4e34\u5e8a\u8868\u73b0",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issue = score.details["medical_schema_issues"][0]
    assert issue["issue_kind"] == "reverse_clinical_manifestation"
    assert issue["suggested_action"] == "replace_relation"
    assert issue["candidate_predicates"] == ["has_manifestation"]
    assert issue["new_source"] == "flu"
    assert issue["new_target"] == "dry-cough"
    assert any(
        finding.suggested_fix_type == "medical_relation_schema_migration"
        and finding.requires_approval is True
        for finding in score.findings
    )


def test_medical_schema_issue_does_not_reduce_relation_semantics_score() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "flu",
                "\u6d41\u884c\u6027\u611f\u5192",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "dry-cough",
                "\u5e72\u54b3",
                "Symptom",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-dry-cough-flu",
                "dry-cough",
                "flu",
                "\u4e34\u5e8a\u8868\u73b0",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["generic_relation_count"] == 0
    assert score.metrics["medical_schema_issue_count"] == 1
    assert score.details["medical_schema_issues"][0]["issue_kind"] == (
        "reverse_clinical_manifestation"
    )
    assert any(
        finding.suggested_fix_type == "medical_relation_schema_migration"
        for finding in score.findings
    )
    assert score.subscores["relation_semantics"] == 100


def test_quality_flags_value_nodes_as_qualifier_candidates() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "oseltamivir",
                "\u5965\u53f8\u4ed6\u97e6",
                "DrugIngredient",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "dose-75mg",
                "75 mg",
                "Dosage",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-dose",
                "oseltamivir",
                "dose-75mg",
                "\u5242\u91cf\u7528\u6cd5",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["value_node_to_qualifier_candidate_count"] == 1
    cleanup = score.details["entity_cleanup_issues"][0]
    assert cleanup["issue_kind"] == "value_node_to_qualifier"
    assert cleanup["suggested_action"] == "convert_to_qualifier"
    assert cleanup["node_id"] == "dose-75mg"
    assert cleanup["label"] == "75 mg"
    assert cleanup["source_id"] == "chunk-1"
    assert cleanup["file_path"] == "guide.md"
    assert cleanup["qualifier_value"] == "75 mg"


def test_quality_flags_synonym_duplicates_from_medical_alias_map() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "flu-short",
                "\u6d41\u611f",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "flu-full",
                "\u6d41\u884c\u6027\u611f\u5192",
                "Disease",
                source_id="chunk-2",
                file_path="guide.md",
            ),
        ],
        edges=[],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["synonym_duplicate_count"] == 1
    duplicate = score.details["entity_cleanup_issues"][0]
    assert duplicate["issue_kind"] == "synonym_duplicate"
    assert duplicate["suggested_action"] == "merge_synonym_nodes"
    assert duplicate["canonical_label"] == "\u6d41\u884c\u6027\u611f\u5192"
    assert set(duplicate["node_ids"]) == {"flu-short", "flu-full"}
    assert sorted(duplicate["nodes"], key=lambda node: node["node_id"]) == [
        {
            "node_id": "flu-full",
            "label": "\u6d41\u884c\u6027\u611f\u5192",
            "source_id": "chunk-2",
            "file_path": "guide.md",
        },
        {
            "node_id": "flu-short",
            "label": "\u6d41\u611f",
            "source_id": "chunk-1",
            "file_path": "guide.md",
        },
    ]


def test_missing_evidence_treats_whitespace_and_separator_only_values_as_missing():
    separator_only = f" {GRAPH_FIELD_SEP}  {GRAPH_FIELD_SEP}"
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode("n1", "N1", "Disease", source_id="   ", file_path="a.md"),
            SnapshotNode("n2", "N2", "Symptom", source_id="chunk-1", file_path=separator_only),
        ],
        edges=[
            SnapshotEdge(
                "e1",
                "n1",
                "n2",
                "causes",
                source_id=separator_only,
                file_path="   ",
            ),
        ],
        metadata={},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["missing_node_source_count"] == 1
    assert score.metrics["missing_node_file_path_count"] == 1
    assert score.metrics["missing_edge_source_count"] == 1
    assert score.metrics["missing_edge_file_path_count"] == 1


def test_missing_evidence_findings_include_bounded_concrete_examples():
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode("n1", "N1", "Disease", source_id="", file_path="a.md"),
            SnapshotNode("n2", "N2", "Symptom", source_id="chunk-1", file_path=""),
        ],
        edges=[
            SnapshotEdge("e1", "n1", "n2", "causes", source_id="", file_path="a.md"),
            SnapshotEdge("e2", "n2", "n1", "causes", source_id="chunk-1", file_path=""),
        ],
        metadata={},
    )

    score = evaluate_snapshot_quality(snapshot)
    evidence_finding = next(
        finding
        for finding in score.findings
        if finding.category == "evidence_grounding"
    )

    assert "node:n1 missing source_id" in evidence_finding.evidence
    assert "node:n2 missing file_path" in evidence_finding.evidence
    assert "edge:e1 missing source_id" in evidence_finding.evidence
    assert "edge:e2 missing file_path" in evidence_finding.evidence


def test_hierarchy_and_hub_structural_findings_require_approval():
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode("flu", "Flu", "Disease", source_id="chunk-1", file_path="a.md"),
            SnapshotNode("fever", "Fever", "Symptom", source_id="chunk-1", file_path="a.md"),
            SnapshotNode("cough", "Cough", "Symptom", source_id="chunk-1", file_path="a.md"),
        ],
        edges=[
            SnapshotEdge("e1", "flu", "fever", "manifestation", source_id="chunk-1", file_path="a.md"),
            SnapshotEdge("e2", "flu", "cough", "manifestation", source_id="chunk-1", file_path="a.md"),
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)
    by_fix_type = {finding.suggested_fix_type: finding for finding in score.findings}

    assert by_fix_type["add_hierarchy_branch"].requires_approval is True
    assert by_fix_type["split_hub_edges"].requires_approval is True


def test_medical_hierarchy_coverage_accepts_keys_aliases_and_medical_group():
    categories = TOP_LEVEL_MEDICAL_CATEGORIES[:3]
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode(
                categories[0].key,
                "Category by key",
                "MedicalCategory",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "alias-node",
                categories[1].aliases[0],
                "MedicalCategory",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "group-node",
                "Category by property",
                "MedicalCategory",
                source_id="chunk-1",
                file_path="guide.md",
                properties={"medical_group": categories[2].key},
            ),
        ],
        edges=[],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["hierarchy_present_branch_count"] == 3


def test_disease_hub_detection_strips_entity_type_whitespace():
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode("flu", "Flu", " Disease ", source_id="chunk-1", file_path="a.md"),
            SnapshotNode("fever", "Fever", "Symptom", source_id="chunk-1", file_path="a.md"),
        ],
        edges=[
            SnapshotEdge("e1", "flu", "fever", "manifestation", source_id="chunk-1", file_path="a.md"),
        ],
        metadata={},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["disease_hub_overload_ratio"] == 1.0


def test_quality_score_exposes_hierarchy_branch_details_for_medical_profile():
    categories = TOP_LEVEL_MEDICAL_CATEGORIES
    present = categories[:2]
    missing = categories[2:]
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-19T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode(
                present[0].key,
                "Category by key",
                "MedicalCategory",
                source_id="medical_kg_profile",
                file_path="medical_kg_profile",
            ),
            SnapshotNode(
                "alias-node",
                present[1].aliases[0],
                "MedicalCategory",
                source_id="medical_kg_profile",
                file_path="medical_kg_profile",
            ),
        ],
        edges=[],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    branches = score.details["hierarchy_branches"]
    assert [branch["key"] for branch in branches["required"]] == [
        category.key for category in categories
    ]
    assert [branch["key"] for branch in branches["present"]] == [
        category.key for category in present
    ]
    assert [branch["key"] for branch in branches["missing"]] == [
        category.key for category in missing
    ]
    assert branches["present"][0]["matched_node_ids"] == [present[0].key]
    assert branches["present"][1]["matched_node_ids"] == ["alias-node"]


def test_quality_artifacts_include_hierarchy_branch_details(tmp_path: Path):
    category = TOP_LEVEL_MEDICAL_CATEGORIES[0]
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-19T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode(
                category.label,
                category.label,
                "MedicalCategory",
                source_id="medical_kg_profile",
                file_path="medical_kg_profile",
            )
        ],
        edges=[],
        metadata={"profile": "clinical_guideline_zh"},
    )
    score = evaluate_snapshot_quality(snapshot)

    written = write_quality_artifacts(score, tmp_path)

    payload = json.loads(written["quality_score"].read_text(encoding="utf-8"))
    assert "details" in payload
    assert "hierarchy_branches" in payload["details"]
    report = written["quality_report"].read_text(encoding="utf-8")
    assert "## Hierarchy Branches" in report
    assert category.key in report


def test_hierarchy_branch_matched_node_ids_are_sorted():
    category = TOP_LEVEL_MEDICAL_CATEGORIES[0]
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-19T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode(
                "z-category-match",
                category.label,
                "MedicalCategory",
                source_id="medical_kg_profile",
                file_path="medical_kg_profile",
            ),
            SnapshotNode(
                "a-category-match",
                category.key,
                "MedicalCategory",
                source_id="medical_kg_profile",
                file_path="medical_kg_profile",
            ),
        ],
        edges=[],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    branches = score.details["hierarchy_branches"]
    assert branches["present"][0]["matched_node_ids"] == [
        "a-category-match",
        "z-category-match",
    ]


def test_quality_flags_legacy_overloaded_medical_relation_keywords():
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "flu",
                "流行性感冒",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "oseltamivir",
                "奥司他韦",
                "DrugIngredient",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "rt-pcr",
                "RT-PCR",
                "DiagnosticMethod",
                source_id="chunk-2",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-treatment",
                "oseltamivir",
                "flu",
                "推荐治疗",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "edge-diagnostic-basis",
                "flu",
                "rt-pcr",
                "诊断依据",
                source_id="chunk-2",
                file_path="guide.md",
            ),
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["medical_schema_legacy_relation_count"] == 2
    finding = next(
        finding
        for finding in score.findings
        if finding.suggested_fix_type == "normalize_medical_relation_schema"
    )
    assert finding.category == "relation_semantics"
    assert finding.evidence == [
        "edge:edge-treatment",
        "edge:edge-diagnostic-basis",
    ]
    details = score.details["relation_semantic_issues"]["legacy_schema"]
    assert details[0]["canonical_options"] == "has_indication | recommends"
    assert details[1]["canonical_options"] == (
        "has_diagnostic_criterion | criterion_requires | has_evidence | "
        "supports_or_refutes"
    )
    assert score.metrics["medical_schema_issue_count"] == 2
    schema_issue = score.details["medical_schema_issues"][0]
    assert schema_issue["issue_kind"] == "legacy_overloaded_relation"
    assert schema_issue["suggested_action"] == "replace_relation"
    assert schema_issue["candidate_predicates"] == ["has_indication", "recommends"]
    assert schema_issue["guidance"]


def test_legacy_relation_detection_uses_first_matching_keyword_order() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "flu",
                "Flu",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "rt-pcr",
                "RT-PCR",
                "DiagnosticMethod",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "oseltamivir",
                "Oseltamivir",
                "DrugIngredient",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-diagnostic-first",
                "flu",
                "rt-pcr",
                "diagnostic_basis, recommended_treatment",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "edge-treatment-first",
                "oseltamivir",
                "flu",
                "recommended_treatment, diagnostic_basis",
                source_id="chunk-2",
                file_path="guide.md",
            ),
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issues_by_edge = {
        issue["edge_id"]: issue for issue in score.details["medical_schema_issues"]
    }
    assert issues_by_edge["edge-diagnostic-first"]["candidate_predicates"] == [
        "has_diagnostic_criterion",
        "criterion_requires",
        "has_evidence",
        "supports_or_refutes",
    ]
    assert issues_by_edge["edge-treatment-first"]["candidate_predicates"] == [
        "has_indication",
        "recommends",
    ]


def test_medical_schema_detection_uses_medical_profile_hints_beyond_hierarchy() -> None:
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "guideline",
                "Guideline",
                "Guideline",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "oseltamivir",
                "Oseltamivir",
                "DrugIngredient",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-hospital-treatment",
                "guideline",
                "oseltamivir",
                "recommended_treatment",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "hospital"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["medical_schema_legacy_relation_count"] == 1
    assert score.details["medical_schema_issues"][0]["edge_id"] == (
        "edge-hospital-treatment"
    )


def test_legacy_schema_subcase_metrics_have_grounded_issue_details() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "flu",
                "Flu",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "rt-pcr",
                "RT-PCR",
                "DiagnosticMethod",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "rash",
                "Rash",
                "AdverseReaction",
                source_id="chunk-2",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-diagnostic-flat",
                "flu",
                "rt-pcr",
                "diagnostic_basis",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "edge-adverse-conflict",
                "flu",
                "rash",
                "adverse_risk",
                source_id="chunk-2",
                file_path="guide.md",
            ),
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["diagnostic_evidence_flattening_count"] == 1
    assert score.metrics["adverse_reaction_role_conflict_count"] == 1
    issues_by_edge = {
        issue["edge_id"]: issue for issue in score.details["medical_schema_issues"]
    }
    assert issues_by_edge["edge-diagnostic-flat"]["medical_subcase"] == (
        "diagnostic_evidence_flattening"
    )
    assert issues_by_edge["edge-diagnostic-flat"]["source_id"] == "chunk-1"
    assert issues_by_edge["edge-diagnostic-flat"]["file_path"] == "guide.md"
    assert issues_by_edge["edge-adverse-conflict"]["medical_subcase"] == (
        "adverse_reaction_role_conflict"
    )
    assert issues_by_edge["edge-adverse-conflict"]["source_id"] == "chunk-2"
    assert issues_by_edge["edge-adverse-conflict"]["file_path"] == "guide.md"


def test_quality_does_not_apply_medical_schema_to_non_medical_workspace():
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "topic",
                "Topic",
                "Concept",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "action",
                "Action",
                "Concept",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-action",
                "topic",
                "action",
                "推荐治疗",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["medical_schema_legacy_relation_count"] == 0
    assert not any(
        finding.suggested_fix_type == "normalize_medical_relation_schema"
        for finding in score.findings
    )
