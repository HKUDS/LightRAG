import json
from pathlib import Path

from lightrag.kb_iteration.review_context import (
    build_review_context,
    write_review_context,
)


def test_build_review_context_focuses_generic_relation_package(tmp_path: Path):
    package = tmp_path / "package"
    snapshot_dir = package / "snapshots"
    snapshot_dir.mkdir(parents=True)
    _write_json(
        snapshot_dir / "kg_snapshot.json",
        {
            "workspace": "demo",
            "generated_at": "2026-06-18T00:00:00+08:00",
            "source_files": ["guide.md"],
            "nodes": [
                {
                    "id": "flu",
                    "label": "Flu",
                    "entity_type": "Disease",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                },
                {
                    "id": "fever",
                    "label": "Fever",
                    "entity_type": "Symptom",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                },
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "flu",
                    "target": "fever",
                    "keywords": "相关",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                },
                {
                    "id": "e2",
                    "source": "flu",
                    "target": "fever",
                    "keywords": "clinical_manifestation",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                },
            ],
            "metadata": {},
        },
    )
    _write_json(
        snapshot_dir / "quality_score.json",
        {
            "overall": 80,
            "subscores": {"relation_semantics": 60},
            "metrics": {"generic_relation_count": 1},
            "findings": [
                {
                    "severity": "high",
                    "category": "relation_semantics",
                    "message": "Generic relation keywords should be reviewed.",
                    "evidence": ["e1"],
                    "suggested_fix_type": "replace_relation_keyword",
                    "requires_approval": True,
                }
            ],
            "critical_blockers": [],
        },
    )
    (package / "accepted_changes.md").write_text(
        "# Accepted Changes\n\n- Keep specific symptom edge.\n", encoding="utf-8"
    )
    (package / "rejected_changes.md").write_text(
        "# Rejected Changes\n\n- Do not merge flu and fever.\n", encoding="utf-8"
    )

    context = build_review_context(
        package, round_id="round-001", focus=["generic_relation"]
    )

    assert context["round_id"] == "round-001"
    assert context["focus"] == ["generic_relation"]
    assert [relation["id"] for relation in context["relations"]] == ["e1"]
    assert {entity["id"] for entity in context["entities"]} == {"flu", "fever"}
    assert context["quality_findings"] == [
        {
            "severity": "high",
            "category": "relation_semantics",
            "message": "Generic relation keywords should be reviewed.",
            "evidence": ["e1"],
            "suggested_fix_type": "replace_relation_keyword",
            "requires_approval": True,
        }
    ]
    assert "Do not merge flu and fever" in context["rules_memory"]["rejected_changes"]


def test_build_review_context_selects_edges_with_generic_relation_token(
    tmp_path: Path,
):
    package = tmp_path / "package"
    snapshot_dir = package / "snapshots"
    snapshot_dir.mkdir(parents=True)
    _write_json(
        snapshot_dir / "kg_snapshot.json",
        {
            "nodes": [
                {"id": "flu", "label": "Flu", "entity_type": "Disease"},
                {"id": "fever", "label": "Fever", "entity_type": "Symptom"},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "flu",
                    "target": "fever",
                    "keywords": "相关, clinical_manifestation",
                },
                {
                    "id": "e2",
                    "source": "flu",
                    "target": "fever",
                    "keywords": "clinical_manifestation",
                },
            ],
        },
    )
    _write_json(
        snapshot_dir / "quality_score.json",
        {
            "findings": [
                {
                    "category": "relation_semantics",
                    "message": "Generic relation keywords should be reviewed.",
                    "evidence": ["e1"],
                    "suggested_fix_type": "replace_relation_keyword",
                }
            ]
        },
    )

    context = build_review_context(
        package, round_id="round-001", focus=["generic_relation"]
    )

    assert [relation["id"] for relation in context["relations"]] == ["e1"]


def test_build_review_context_selects_edge_from_quality_evidence_example(
    tmp_path: Path,
):
    package = tmp_path / "package"
    snapshot_dir = package / "snapshots"
    snapshot_dir.mkdir(parents=True)
    _write_json(
        snapshot_dir / "kg_snapshot.json",
        {
            "nodes": [
                {"id": "flu", "label": "Flu", "entity_type": "Disease"},
                {"id": "fever", "label": "Fever", "entity_type": "Symptom"},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "flu",
                    "target": "fever",
                    "keywords": "clinical_manifestation",
                },
                {
                    "id": "e2",
                    "source": "fever",
                    "target": "flu",
                    "keywords": "diagnostic_context",
                },
            ],
        },
    )
    _write_json(
        snapshot_dir / "quality_score.json",
        {
            "findings": [
                {
                    "category": "evidence_grounding",
                    "message": "Snapshot items should retain source_id evidence.",
                    "evidence": ["edge:e1 missing source_id", "node:flu missing file_path"],
                    "suggested_fix_type": "restore_evidence",
                }
            ]
        },
    )

    context = build_review_context(
        package, round_id="round-001", focus=["evidence_grounding"]
    )

    assert [relation["id"] for relation in context["relations"]] == ["e1"]


def test_build_review_context_ignores_unprefixed_metric_evidence_for_edges(
    tmp_path: Path,
):
    package = tmp_path / "package"
    snapshot_dir = package / "snapshots"
    snapshot_dir.mkdir(parents=True)
    _write_json(
        snapshot_dir / "kg_snapshot.json",
        {
            "nodes": [
                {"id": "flu", "label": "Flu", "entity_type": "Disease"},
                {"id": "fever", "label": "Fever", "entity_type": "Symptom"},
                {"id": "cough", "label": "Cough", "entity_type": "Symptom"},
            ],
            "edges": [
                {
                    "id": "3",
                    "source": "flu",
                    "target": "fever",
                    "keywords": "clinical_manifestation",
                },
                {
                    "id": "e2",
                    "source": "flu",
                    "target": "cough",
                    "keywords": "clinical_manifestation",
                },
            ],
        },
    )
    _write_json(
        snapshot_dir / "quality_score.json",
        {
            "findings": [
                {
                    "category": "hierarchy_completeness",
                    "message": "Medical hierarchy is missing expected branches.",
                    "evidence": ["3"],
                    "suggested_fix_type": "add_hierarchy_branch",
                }
            ]
        },
    )

    context = build_review_context(
        package, round_id="round-001", focus=["hierarchy_completeness"]
    )

    assert [relation["id"] for relation in context["relations"]] == ["3", "e2"]


def test_write_review_context_writes_round_context_json(tmp_path: Path):
    package = tmp_path / "package"
    context = {
        "round_id": "round-001",
        "focus": ["generic_relation"],
        "quality_findings": [],
        "entities": [],
        "relations": [],
        "evidence_windows": [],
        "rules_memory": {"accepted_changes": "保留相关证据", "rejected_changes": ""},
    }

    written = write_review_context(package, "round-001", context)

    assert written == package / "review_context" / "round-001-context.json"
    raw_context = written.read_text(encoding="utf-8")
    payload = json.loads(raw_context)
    assert payload["focus"] == ["generic_relation"]
    assert "generic_relation" in raw_context
    assert "保留相关证据" in raw_context
    assert raw_context.endswith("\n")


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
