import json
from pathlib import Path

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kb_iteration.models import KGSnapshot, SnapshotEdge, SnapshotNode
from lightrag.kb_iteration.quality import (
    evaluate_snapshot_quality,
    write_quality_artifacts,
)
from lightrag.medical_kg.ontology import TOP_LEVEL_MEDICAL_CATEGORIES


def _snapshot_with_reversed_diagnostic_test_edge() -> KGSnapshot:
    return KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-23T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "positive-pcr",
                "Positive PCR",
                "TestResultPattern",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "flu",
                "Influenza",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-evidence-without-polarity",
                "positive-pcr",
                "flu",
                "supports_or_refutes",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )


def _snapshot_with_value_node_cleanup_issue() -> KGSnapshot:
    return KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-23T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "oseltamivir",
                "Oseltamivir",
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
                "dosage_usage",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )


def test_quality_schema_issue_includes_raw_issue_contract_fields() -> None:
    snapshot = _snapshot_with_reversed_diagnostic_test_edge()

    score = evaluate_snapshot_quality(snapshot)

    issue = score.details["medical_schema_issues"][0]
    for key in (
        "issue_kind",
        "edge_id",
        "source",
        "source_type",
        "target",
        "target_type",
        "keywords",
        "qualifiers",
        "candidate_predicates",
        "repair_options",
        "source_id",
        "file_path",
        "evidence_quote",
    ):
        assert key in issue


def test_quality_entity_cleanup_issue_includes_contract_fields() -> None:
    snapshot = _snapshot_with_value_node_cleanup_issue()

    score = evaluate_snapshot_quality(snapshot)

    issue = score.details["entity_cleanup_issues"][0]
    for key in (
        "issue_kind",
        "node_id",
        "candidate_predicates",
        "repair_options",
        "source_id",
        "file_path",
        "evidence_quote",
    ):
        assert key in issue


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


def test_quality_schema_issue_preserves_existing_relation_qualifiers() -> None:
    edge_qualifiers = {"age_min": 1, "age_unit": "year"}
    snapshot = KGSnapshot(
        workspace="demo",
        generated_at="2026-06-22T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "oseltamivir",
                "Oseltamivir",
                "Drug",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "children",
                "Children",
                "Population",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-recommended-for",
                "oseltamivir",
                "children",
                "recommended_for",
                source_id="chunk-1",
                file_path="guide.md",
                properties={"qualifiers": edge_qualifiers},
            )
        ],
        metadata={"profile": "medical_kg"},
    )

    score = evaluate_snapshot_quality(snapshot)

    schema_issues = score.details["medical_schema_issues"]
    issue = next(
        issue
        for issue in schema_issues
        if issue["edge_id"] == "edge-recommended-for"
    )
    assert issue["qualifiers"] == {"age_min": 1, "age_unit": "year"}

    issue["qualifiers"]["age_min"] = 2

    assert edge_qualifiers == {"age_min": 1, "age_unit": "year"}
    assert snapshot.edges[0].properties["qualifiers"] == {
        "age_min": 1,
        "age_unit": "year",
    }


def test_quality_schema_issue_deep_copies_nested_relation_qualifiers() -> None:
    edge_qualifiers = {"dose": {"amount": 75, "unit": "mg"}}
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-23T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "flu",
                "Influenza",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "fever",
                "Fever",
                "Symptom",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-manifestation-with-dose",
                "flu",
                "fever",
                "has_manifestation",
                source_id="chunk-1",
                file_path="guide.md",
                properties={"qualifiers": edge_qualifiers},
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issue = next(
        issue
        for issue in score.details["medical_schema_issues"]
        if issue["edge_id"] == "edge-manifestation-with-dose"
    )
    issue["qualifiers"]["dose"]["amount"] = 150

    assert edge_qualifiers == {"dose": {"amount": 75, "unit": "mg"}}
    assert snapshot.edges[0].properties["qualifiers"] == {
        "dose": {"amount": 75, "unit": "mg"}
    }


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
    assert score.subscores["relation_semantics"] < 100


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


def test_medical_domain_range_detector_emits_precise_issue_kinds() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "flu",
                "\u6d41\u884c\u6027\u611f\u5192",
                "disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "fever",
                "\u53d1\u70ed",
                "sign",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "virus",
                "\u6d41\u611f\u75c5\u6bd2",
                "pathogen",
                source_id="chunk-2",
                file_path="guide.md",
            ),
            SnapshotNode(
                "oseltamivir",
                "\u5965\u53f8\u4ed6\u97e6",
                "drugingredient",
                source_id="chunk-3",
                file_path="guide.md",
            ),
            SnapshotNode(
                "rt-pcr",
                "RT-PCR",
                "diagnostictest",
                source_id="chunk-4",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-manifestation",
                "fever",
                "flu",
                "\u4e34\u5e8a\u8868\u73b0",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "edge-causative",
                "virus",
                "flu",
                "\u75c5\u539f\u5bfc\u81f4",
                source_id="chunk-2",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "edge-treatment",
                "virus",
                "oseltamivir",
                "\u63a8\u8350\u6cbb\u7597",
                source_id="chunk-3",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "edge-adverse",
                "oseltamivir",
                "fever",
                "\u4e34\u5e8a\u8868\u73b0",
                source_id="chunk-3",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "edge-diagnostic",
                "rt-pcr",
                "flu",
                "\u8bca\u65ad\u4f9d\u636e",
                source_id="chunk-4",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "edge-multi",
                "oseltamivir",
                "flu",
                "\u5242\u91cf\u7528\u6cd5,\u63a8\u8350\u6cbb\u7597,\u9002\u7528\u4e8e",
                source_id="chunk-5",
                file_path="guide.md",
            ),
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issues = score.details["medical_schema_issues"]
    issues_by_kind = {issue["issue_kind"]: issue for issue in issues}
    assert set(issues_by_kind) >= {
        "reverse_clinical_manifestation",
        "reverse_causative_agent",
        "treatment_domain_range_mismatch",
        "adverse_reaction_modeled_as_manifestation",
        "diagnostic_evidence_direction_mismatch",
        "multi_predicate_edge_split_needed",
    }

    reverse_manifestation = issues_by_kind["reverse_clinical_manifestation"]
    assert reverse_manifestation["edge_id"] == "edge-manifestation"
    assert reverse_manifestation["source_type"] == "Symptom"
    assert reverse_manifestation["target_type"] == "Disease"
    assert reverse_manifestation["new_source"] == "flu"
    assert reverse_manifestation["new_target"] == "fever"
    assert reverse_manifestation["candidate_predicates"] == ["has_manifestation"]

    reverse_causative = issues_by_kind["reverse_causative_agent"]
    assert reverse_causative["edge_id"] == "edge-causative"
    assert reverse_causative["source_type"] == "Pathogen"
    assert reverse_causative["target_type"] == "Disease"
    assert reverse_causative["new_source"] == "flu"
    assert reverse_causative["new_target"] == "virus"
    assert reverse_causative["candidate_predicates"] == ["causative_agent"]

    treatment = issues_by_kind["treatment_domain_range_mismatch"]
    assert treatment["edge_id"] == "edge-treatment"
    assert treatment["source_type"] == "Pathogen"
    assert treatment["target_type"] == "Drug"
    assert set(treatment["candidate_predicates"]) >= {"has_indication", "recommends"}

    adverse = issues_by_kind["adverse_reaction_modeled_as_manifestation"]
    assert adverse["edge_id"] == "edge-adverse"
    assert adverse["source_type"] == "Drug"
    assert adverse["target_type"] == "Symptom"
    assert adverse["candidate_predicates"] == ["may_cause_adverse_reaction"]

    diagnostic = issues_by_kind["diagnostic_evidence_direction_mismatch"]
    assert diagnostic["edge_id"] == "edge-diagnostic"
    assert diagnostic["source_type"] == "Test"
    assert diagnostic["target_type"] == "Disease"
    assert set(diagnostic["candidate_predicates"]) >= {
        "has_diagnostic_criterion",
        "supports_or_refutes",
    }

    multi = issues_by_kind["multi_predicate_edge_split_needed"]
    assert multi["edge_id"] == "edge-multi"
    assert set(multi["predicate_tokens"]) >= {
        "\u5242\u91cf\u7528\u6cd5",
        "\u63a8\u8350\u6cbb\u7597",
        "\u9002\u7528\u4e8e",
    }
    assert multi["repair_options"]
    assert {
        option["predicate"] for option in multi["repair_options"]
    } >= {"has_dosing_regimen", "recommended_for"}
    assert all("validation_errors" in option for option in multi["repair_options"])
    assert any(option["auto_fixable"] is False for option in multi["repair_options"])
    assert len(
        [
            issue
            for issue in issues
            if issue["edge_id"] == "edge-manifestation"
            and issue["issue_kind"] == "reverse_clinical_manifestation"
        ]
    ) == 1
    assert score.metrics["relation_semantic_issue_count"] > 0
    assert score.subscores["relation_semantics"] < 100


def test_medical_domain_range_detector_validates_canonical_relation_specs() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-21T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode("flu", "流行性感冒", "Disease", source_id="chunk-1", file_path="guide.md"),
            SnapshotNode("fever", "发热", "Symptom", source_id="chunk-1", file_path="guide.md"),
            SnapshotNode("vaccine", "流感疫苗", "Vaccine", source_id="chunk-2", file_path="guide.md"),
            SnapshotNode("children", "儿童", "Population", source_id="chunk-2", file_path="guide.md"),
            SnapshotNode("pneumonia", "肺炎", "Complication", source_id="chunk-3", file_path="guide.md"),
            SnapshotNode("elderly", "老年人", "RiskFactor", source_id="chunk-4", file_path="guide.md"),
            SnapshotNode("guideline", "指南", "Guideline", source_id="chunk-5", file_path="guide.md"),
        ],
        edges=[
            SnapshotEdge(
                "valid-recommended-for",
                "vaccine",
                "children",
                "recommended_for",
                source_id="chunk-2",
                file_path="guide.md",
                properties={
                    "qualifiers": {
                        "purpose": "prevention",
                        "age_min": 6,
                        "age_unit": "month",
                    }
                },
            ),
            SnapshotEdge(
                "invalid-recommended-for",
                "children",
                "vaccine",
                "recommended_for",
                source_id="chunk-2",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "valid-targets-disease",
                "vaccine",
                "flu",
                "targets_disease",
                source_id="chunk-2",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "invalid-targets-disease",
                "flu",
                "vaccine",
                "targets_disease",
                source_id="chunk-2",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "valid-complication",
                "flu",
                "pneumonia",
                "has_complication",
                source_id="chunk-3",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "invalid-complication",
                "fever",
                "flu",
                "has_complication",
                source_id="chunk-3",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "valid-risk-group",
                "children",
                "flu",
                "risk_group_for",
                source_id="chunk-4",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "invalid-risk-group",
                "guideline",
                "children",
                "risk_group_for",
                source_id="chunk-4",
                file_path="guide.md",
            ),
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issue_by_edge = {
        issue["edge_id"]: issue
        for issue in score.details["medical_schema_issues"]
        if issue["issue_kind"] == "canonical_relation_domain_range_mismatch"
    }
    assert set(issue_by_edge) == {
        "invalid-recommended-for",
        "invalid-targets-disease",
        "invalid-complication",
        "invalid-risk-group",
    }
    recommended_for_issue = issue_by_edge["invalid-recommended-for"]
    assert recommended_for_issue["candidate_predicates"] == ["recommended_for"]
    assert recommended_for_issue["expected_domain_types"] == [
        "Drug",
        "Treatment",
        "Vaccine",
        "PublicHealthMeasure",
    ]
    assert recommended_for_issue["expected_range_types"] == ["Population"]
    assert recommended_for_issue["source_type"] == "Population"
    assert recommended_for_issue["target_type"] == "Vaccine"


def test_medical_schema_issue_reduces_relation_semantics_score() -> None:
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
    assert score.subscores["relation_semantics"] < 100


def test_single_relation_semantic_issue_reduces_score_in_large_graph() -> None:
    symptom_nodes = [
        SnapshotNode(
            f"symptom-{index}",
            f"Symptom {index}",
            "Symptom",
            source_id="chunk-1",
            file_path="guide.md",
        )
        for index in range(250)
    ]
    good_edges = [
        SnapshotEdge(
            f"edge-good-{index}",
            "flu",
            f"symptom-{index}",
            "\u4e34\u5e8a\u8868\u73b0",
            source_id="chunk-1",
            file_path="guide.md",
        )
        for index in range(1, 250)
    ]
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
            *symptom_nodes,
        ],
        edges=[
            SnapshotEdge(
                "edge-bad-direction",
                "symptom-0",
                "flu",
                "\u4e34\u5e8a\u8868\u73b0",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            *good_edges,
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["relation_semantic_issue_count"] == 1
    assert score.subscores["relation_semantics"] < 100


def test_drug_to_clinical_condition_treatment_edge_is_valid_indication() -> None:
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
                "severe-case",
                "\u91cd\u75c7\u75c5\u4f8b",
                "clinical_condition",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-treatment-condition",
                "oseltamivir",
                "severe-case",
                "\u63a8\u8350\u6cbb\u7597",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert not any(
        issue["edge_id"] == "edge-treatment-condition"
        and issue["issue_kind"] == "treatment_domain_range_mismatch"
        for issue in score.details["medical_schema_issues"]
    )


def test_drug_to_disease_recommends_requires_indication_predicate() -> None:
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
                "flu",
                "\u6d41\u884c\u6027\u611f\u5192",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-drug-recommends-disease",
                "oseltamivir",
                "flu",
                "recommends",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issue = next(
        issue
        for issue in score.details["medical_schema_issues"]
        if issue["edge_id"] == "edge-drug-recommends-disease"
        and issue["issue_kind"] == "treatment_domain_range_mismatch"
    )
    assert "has_indication" in issue["candidate_predicates"]


def test_guideline_recommends_diagnostic_test_is_valid_recommendation() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "guideline",
                "\u6d41\u611f\u6307\u5357",
                "Guideline",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "rt-pcr",
                "RT-PCR",
                "DiagnosticTest",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-guideline-test",
                "guideline",
                "rt-pcr",
                "recommends",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert not any(
        issue["edge_id"] == "edge-guideline-test"
        and issue["issue_kind"] == "treatment_domain_range_mismatch"
        for issue in score.details["medical_schema_issues"]
    )


def test_clinical_pathway_recommends_test_is_valid_recommendation() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["pathway.md"],
        nodes=[
            SnapshotNode(
                "pathway",
                "\u53d1\u70ed\u95e8\u8bca\u8def\u5f84",
                "clinical_pathway",
                source_id="chunk-1",
                file_path="pathway.md",
            ),
            SnapshotNode(
                "rapid-test",
                "\u6297\u539f\u5feb\u68c0",
                "Test",
                source_id="chunk-1",
                file_path="pathway.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-pathway-test",
                "pathway",
                "rapid-test",
                "recommends",
                source_id="chunk-1",
                file_path="pathway.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert not any(
        issue["edge_id"] == "edge-pathway-test"
        and issue["issue_kind"] == "treatment_domain_range_mismatch"
        for issue in score.details["medical_schema_issues"]
    )


def test_clinical_finding_to_clinical_condition_manifestation_is_reversed() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "hypoxemia",
                "\u4f4e\u6c27\u8840\u75c7",
                "clinical_finding",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "severe-case",
                "\u91cd\u75c7\u75c5\u4f8b",
                "clinical_condition",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-finding-condition",
                "hypoxemia",
                "severe-case",
                "\u4e34\u5e8a\u8868\u73b0",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issue = next(
        issue
        for issue in score.details["medical_schema_issues"]
        if issue["edge_id"] == "edge-finding-condition"
    )
    assert issue["issue_kind"] == "reverse_clinical_manifestation"
    assert issue["source_type"] == "ClinicalFinding"
    assert issue["target_type"] == "ClinicalCondition"
    assert issue["new_source"] == "severe-case"
    assert issue["new_target"] == "hypoxemia"


def test_symptom_to_syndrome_has_manifestation_is_reversed() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "fever",
                "\u53d1\u70ed",
                "Symptom",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "flu-like-syndrome",
                "\u6d41\u611f\u6837\u7efc\u5408\u5f81",
                "Syndrome",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-symptom-syndrome",
                "fever",
                "flu-like-syndrome",
                "has_manifestation",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issue = next(
        issue
        for issue in score.details["medical_schema_issues"]
        if issue["edge_id"] == "edge-symptom-syndrome"
    )
    assert issue["issue_kind"] == "reverse_clinical_manifestation"
    assert issue["source_type"] == "Symptom"
    assert issue["target_type"] == "Syndrome"
    assert issue["new_source"] == "flu-like-syndrome"
    assert issue["new_target"] == "fever"


def test_test_to_clinical_condition_diagnostic_basis_direction_mismatch() -> None:
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-20T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "rt-pcr",
                "RT-PCR",
                "DiagnosticTest",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "confirmed-case",
                "\u786e\u8bca\u75c5\u4f8b",
                "ClinicalCondition",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-test-condition",
                "rt-pcr",
                "confirmed-case",
                "\u8bca\u65ad\u4f9d\u636e",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issue = next(
        issue
        for issue in score.details["medical_schema_issues"]
        if issue["edge_id"] == "edge-test-condition"
        and issue["issue_kind"] == "diagnostic_evidence_direction_mismatch"
    )
    assert issue["source_type"] == "Test"
    assert issue["target_type"] == "ClinicalCondition"
    assert issue["new_source"] == "confirmed-case"
    assert issue["new_target"] == "rt-pcr"


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


def test_quality_flags_strict_canonical_relation_domain_range_mismatches() -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-22T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "flu",
                "Influenza",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "cough",
                "Cough",
                "Symptom",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "cbc",
                "Complete blood count",
                "Test",
                source_id="chunk-2",
                file_path="guide.md",
            ),
            SnapshotNode(
                "result",
                "Leukopenia",
                "ClinicalFinding",
                source_id="chunk-2",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-cause-symptom",
                "flu",
                "cough",
                "causative_agent",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotEdge(
                "edge-result-finding",
                "cbc",
                "result",
                "has_result",
                source_id="chunk-2",
                file_path="guide.md",
            ),
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issues = {
        issue["edge_id"]: issue
        for issue in score.details["medical_schema_issues"]
        if issue["issue_kind"] == "canonical_relation_domain_range_mismatch"
    }
    assert issues["edge-cause-symptom"]["candidate_predicates"] == [
        "causative_agent"
    ]
    assert issues["edge-cause-symptom"]["expected_range_types"] == ["Pathogen"]
    assert issues["edge-result-finding"]["candidate_predicates"] == ["has_result"]
    assert issues["edge-result-finding"]["expected_range_types"] == ["TestResult"]


def test_quality_flags_missing_required_relation_qualifier() -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-22T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "positive-pcr",
                "Positive PCR",
                "TestResultPattern",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "flu",
                "Influenza",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-evidence-without-polarity",
                "positive-pcr",
                "flu",
                "supports_or_refutes",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issue = next(
        issue
        for issue in score.details["medical_schema_issues"]
        if issue["edge_id"] == "edge-evidence-without-polarity"
    )
    assert issue["issue_kind"] == "missing_required_relation_qualifier"
    assert issue["missing_qualifiers"] == ["polarity"]
    assert issue["candidate_predicates"] == ["supports_or_refutes"]


def test_quality_flags_unsupported_relation_qualifier() -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-22T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "flu",
                "Influenza",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "fever",
                "Fever",
                "Symptom",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-manifestation-with-dose",
                "flu",
                "fever",
                "has_manifestation",
                source_id="chunk-1",
                file_path="guide.md",
                properties={"qualifiers": {"dose": "75 mg"}},
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issue = next(
        issue
        for issue in score.details["medical_schema_issues"]
        if issue["edge_id"] == "edge-manifestation-with-dose"
    )
    assert issue["issue_kind"] == "unsupported_relation_qualifier"
    assert issue["unsupported_qualifiers"] == ["dose"]


def test_quality_flags_invalid_relation_qualifier_value() -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-22T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "zanamivir",
                "Zanamivir",
                "Drug",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "children",
                "Children",
                "Population",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-recommendation-invalid-purpose",
                "zanamivir",
                "children",
                "recommended_for",
                source_id="chunk-1",
                file_path="guide.md",
                properties={"qualifiers": {"purpose": "diagnosis", "age_min": 7}},
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issue = next(
        issue
        for issue in score.details["medical_schema_issues"]
        if issue["edge_id"] == "edge-recommendation-invalid-purpose"
    )
    assert issue["issue_kind"] == "invalid_relation_qualifier_value"
    assert issue["invalid_qualifier_values"] == ["purpose"]


def test_quality_flags_conflicting_recommendation_and_contraindication_scope() -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-22T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "zanamivir",
                "Zanamivir",
                "Drug",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "children",
                "Children",
                "Population",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-zanamivir-recommended",
                "zanamivir",
                "children",
                "recommended_for",
                source_id="chunk-1",
                file_path="guide.md",
                properties={
                    "qualifiers": {
                        "purpose": "treatment",
                        "age_min": 7,
                        "age_unit": "year",
                    }
                },
            ),
            SnapshotEdge(
                "edge-zanamivir-contraindicated",
                "zanamivir",
                "children",
                "contraindicated_for",
                source_id="chunk-2",
                file_path="guide.md",
                properties={
                    "qualifiers": {
                        "purpose": "treatment",
                        "age_min": 7,
                        "age_unit": "year",
                    }
                },
            ),
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issue = next(
        issue
        for issue in score.details["medical_schema_issues"]
        if issue["issue_kind"] == "conflicting_recommendation_safety_scope"
    )
    assert issue["edge_id"] == "edge-zanamivir-recommended"
    assert issue["conflict_edge_id"] == "edge-zanamivir-contraindicated"
    assert issue["candidate_predicates"] == ["recommended_for", "contraindicated_for"]


def test_quality_preserves_multiple_conflicting_safety_edges_for_same_recommendation() -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-23T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "zanamivir",
                "Zanamivir",
                "Drug",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "children",
                "Children",
                "Population",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-zanamivir-recommended",
                "zanamivir",
                "children",
                "recommended_for",
                source_id="chunk-1",
                file_path="guide.md",
                properties={
                    "qualifiers": {
                        "purpose": "treatment",
                        "age_min": 7,
                        "age_unit": "year",
                    }
                },
            ),
            SnapshotEdge(
                "edge-zanamivir-contraindicated-asthma",
                "zanamivir",
                "children",
                "contraindicated_for",
                source_id="chunk-2",
                file_path="guide.md",
                properties={
                    "qualifiers": {
                        "purpose": "treatment",
                        "age_min": 7,
                        "age_unit": "year",
                    }
                },
            ),
            SnapshotEdge(
                "edge-zanamivir-contraindicated-allergy",
                "zanamivir",
                "children",
                "contraindicated_for",
                source_id="chunk-3",
                file_path="guide.md",
                properties={
                    "qualifiers": {
                        "purpose": "treatment",
                        "age_min": 7,
                        "age_unit": "year",
                    }
                },
            ),
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    issues = [
        issue
        for issue in score.details["medical_schema_issues"]
        if issue["issue_kind"] == "conflicting_recommendation_safety_scope"
        and issue["edge_id"] == "edge-zanamivir-recommended"
    ]
    assert [issue["conflict_edge_id"] for issue in issues] == [
        "edge-zanamivir-contraindicated-asthma",
        "edge-zanamivir-contraindicated-allergy",
    ]


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
