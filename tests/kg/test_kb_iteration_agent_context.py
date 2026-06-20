import json
import os
from pathlib import Path
import subprocess

import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kb_iteration.agent_context import (
    build_agent_observation,
    build_stage_context,
    write_agent_context,
)


def test_build_agent_observation_includes_package_artifacts(tmp_path: Path):
    package = _make_agent_package(tmp_path)

    observation = build_agent_observation(package, workspace="influenza_medical_v1")

    assert observation["workspace"] == "influenza_medical_v1"
    assert observation["quality"]["overall"] == 88
    assert observation["hierarchy_branches"]["missing"] == [
        {"key": "symptom", "label": "Symptoms"}
    ]
    assert observation["artifact_status"]["kb_context.md"]["exists"] is True
    assert observation["artifact_status"]["agent_memory_summary.md"]["exists"] is True
    assert "Current KG context" in observation["kb_context"]
    assert "Accepted rule" in observation["rules_memory"]["accepted_changes"]
    assert "Rejected rule" in observation["rules_memory"]["rejected_changes"]
    assert "Accepted rule" in observation["rules_memory"]["agent_memory_summary"]
    assert "Rejected rule" in observation["rules_memory"]["agent_memory_summary"]
    assert set(observation["rules_memory"]) == {
        "accepted_changes",
        "rejected_changes",
        "proposal_revision_requests",
        "agent_memory_summary",
    }


def test_revision_request_memory_is_visible_to_agent_context(tmp_path: Path):
    package = _make_agent_package(tmp_path)
    (package / "proposal_revision_requests.md").write_text(
        "# Proposal Revision Requests\n\n"
        "## p1\n\n"
        "```json\n"
        '{"proposal_id": "p1", "reason": "Evidence missing source_id"}\n'
        "```\n",
        encoding="utf-8",
    )

    observation = build_agent_observation(package, workspace="influenza_medical_v1")
    stage_context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="propose_changes",
    )

    for context in (observation, stage_context):
        assert (
            "Evidence missing source_id"
            in context["rules_memory"]["proposal_revision_requests"]
        )
        assert (
            "Evidence missing source_id"
            in context["rules_memory"]["agent_memory_summary"]
        )


def test_stage_context_keeps_top_level_revision_requests_capped_and_stripped(
    tmp_path: Path,
):
    package = _make_agent_package(tmp_path)
    request_lines = [
        f"  Revision request {index}  "
        for index in range(1, 63)
    ]
    (package / "proposal_revision_requests.md").write_text(
        "\n".join(["", *request_lines[:30], "   ", *request_lines[30:], ""]),
        encoding="utf-8",
    )

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="propose",
    )

    assert len(context["proposal_revision_requests"]) == 20
    assert context["proposal_revision_requests"][0] == "Revision request 1"
    assert context["proposal_revision_requests"][19] == "Revision request 20"
    assert all(line == line.strip() for line in context["proposal_revision_requests"])
    assert "Revision request 21" not in context["proposal_revision_requests"]


def test_build_agent_observation_budgets_large_package_inputs(tmp_path: Path):
    package = _make_agent_package(tmp_path)
    snapshot_path = package / "snapshots" / "kg_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["source_files"] = [f"source-{index}.md" for index in range(500)]
    snapshot["metadata"] = {
        f"metadata_{index}": {
            "long_value": "verbose metadata value " * 200,
            "items": [f"item-{index}-{item}" for item in range(100)],
        }
        for index in range(100)
    }
    _write_json(snapshot_path, snapshot)

    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["subscores"] = {
        f"subscore_{index}": "verbose subscore value " * 100
        for index in range(100)
    }
    quality["metrics"] = {
        f"metric_{index}": "verbose metric value " * 100
        for index in range(100)
    }
    quality["findings"] = [
        {
            "severity": "high",
            "category": f"category-{index}",
            "message": "finding message " * 200,
            "evidence": [
                f"evidence-{index}-{item} " + ("x" * 1000)
                for item in range(50)
            ],
            "suggested_fix_type": "normalize_relation_direction",
            "requires_approval": True,
        }
        for index in range(200)
    ]
    quality["critical_blockers"] = [
        "critical blocker " * 100
        for _ in range(100)
    ]
    _write_json(quality_path, quality)

    (package / "kb_context.md").write_text(
        "# Current KG context\n\n" + ("Large context paragraph. " * 5000),
        encoding="utf-8",
    )
    long_memory = "Long memory entry. " * 5000
    (package / "accepted_changes.md").write_text(long_memory, encoding="utf-8")
    (package / "rejected_changes.md").write_text(long_memory, encoding="utf-8")
    (package / "proposal_revision_requests.md").write_text(
        "\n".join(f"Revision {index}: {long_memory}" for index in range(100)),
        encoding="utf-8",
    )

    observation = build_agent_observation(package, workspace="influenza_medical_v1")

    assert set(observation) == {
        "workspace",
        "generated_at",
        "node_count",
        "edge_count",
        "source_files",
        "metadata",
        "quality",
        "hierarchy_branches",
        "artifact_status",
        "kb_context",
        "rules_memory",
    }
    assert len(json.dumps(observation, ensure_ascii=False)) < 45_000


def test_stage_context_budgets_worst_case_aggregate_sections(tmp_path: Path):
    package = _make_agent_package(tmp_path)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["findings"] = [
        {
            "severity": "medium",
            "category": f"category-{index}",
            "message": "m" * 1000,
            "evidence": [
                f"evidence-{index}-{item} " + ("e" * 1000)
                for item in range(20)
            ],
            "suggested_fix_type": "normalize_relation_direction",
            "requires_approval": True,
        }
        for index in range(100)
    ]
    _write_json(quality_path, quality)
    (package / "proposal_revision_requests.md").write_text(
        "\n".join(
            f"Revision request {index}: " + ("r" * 600)
            for index in range(100)
        ),
        encoding="utf-8",
    )

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="propose",
    )

    assert len(json.dumps(context, ensure_ascii=False)) < 45_000


def test_build_stage_context_selects_missing_branch_candidates(
    tmp_path: Path,
):
    package = _make_agent_package(tmp_path)
    previous = {"missing_branches": [{"key": "symptom", "label": "Symptoms"}]}

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="locate_evidence",
        previous_outputs=previous,
    )

    assert context["workspace"] == "influenza_medical_v1"
    assert context["stage"] == "locate_evidence"
    assert context["previous_outputs"] == previous
    assert [entity["id"] for entity in context["candidate_entities"]] == ["fever"]
    assert [relation["id"] for relation in context["candidate_relations"]] == [
        "edge-flu-fever"
    ]
    assert context["evidence_windows"] == [
        {
            "item_type": "entity",
            "item_id": "fever",
            "evidence_status": "grounded",
        },
        {
            "item_type": "relation",
            "item_id": "edge-flu-fever",
            "evidence_status": "grounded",
        },
    ]


def test_stage_context_includes_schema_and_cleanup_issue_details(tmp_path: Path):
    package = _make_agent_package(tmp_path)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        {
            "edge_id": "edge-dry-cough-flu",
            "suggested_action": "replace_relation",
        }
    ]
    quality["details"]["entity_cleanup_issues"] = []
    _write_json(quality_path, quality)

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="propose",
    )

    assert context["medical_schema_issues"][0]["edge_id"] == "edge-dry-cough-flu"
    assert context["entity_cleanup_issues"] == []


def test_stage_context_compacts_large_schema_issue_details(tmp_path: Path):
    package = _make_agent_package(tmp_path)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        {
            "issue_kind": "legacy_relation_keyword",
            "edge_id": f"edge-{index}",
            "source": f"source-{index}",
            "target": f"target-{index}",
            "keywords": "属于",
            "source_id": f"chunk-{index}",
            "file_path": "guide.md",
            "suggested_action": "replace_relation",
            "candidate_predicates": ["has_manifestation", "is_a"],
            "new_source": f"target-{index}",
            "new_target": f"source-{index}",
            "guidance": "verbose guidance " * 400,
        }
        for index in range(125)
    ]
    _write_json(quality_path, quality)

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="infer_branches",
    )

    assert len(context["medical_schema_issues"]) == 24
    assert context["medical_schema_issues"][0] == {
        "issue_kind": "legacy_relation_keyword",
        "edge_id": "edge-0",
        "source": "source-0",
        "target": "target-0",
        "keywords": "属于",
        "source_id": "chunk-0",
        "file_path": "guide.md",
        "suggested_action": "replace_relation",
        "candidate_predicates": ["has_manifestation", "is_a"],
        "new_source": "target-0",
        "new_target": "source-0",
    }
    assert "guidance" not in context["medical_schema_issues"][0]
    summary = context["medical_schema_issues_summary"]
    assert summary["total"] == 125
    assert summary["visible"] == 24
    assert summary["omitted"] == 101
    assert summary["by_issue_kind"]["legacy_relation_keyword"] == {
        "total": 125,
        "visible": 24,
        "omitted": 101,
    }
    assert len(json.dumps(context, ensure_ascii=False)) < 45_000


def test_stage_context_compacts_large_hierarchy_branch_matches(tmp_path: Path):
    package = _make_agent_package(tmp_path)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["hierarchy_branches"]["present"] = [
        {
            "key": "symptom",
            "label": "Symptoms",
            "aliases": ["sign", "clinical finding"],
            "matched_node_ids": [f"symptom-{index}" for index in range(5000)],
            "notes": "verbose hierarchy branch detail " * 1000,
        }
    ]
    _write_json(quality_path, quality)

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="infer_branches",
    )

    branch = context["hierarchy_branches"]["present"][0]
    assert branch["key"] == "symptom"
    assert branch["label"] == "Symptoms"
    assert branch["aliases"] == ["sign", "clinical finding"]
    assert branch["matched_node_ids"][:2] == ["symptom-0", "symptom-1"]
    assert len(branch["matched_node_ids"]) <= 20
    assert "notes" not in branch
    assert len(json.dumps(context, ensure_ascii=False)) < 45_000


def test_stage_context_schema_issue_retains_medical_subcase(tmp_path: Path):
    package = _make_agent_package(tmp_path)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["medical_schema_issues"] = [
        {
            "issue_kind": "medical_subcase_relation",
            "medical_subcase": "infectious_disease_symptom",
            "edge_id": "edge-flu-fever",
            "guidance": "drop this verbose guidance " * 100,
        }
    ]
    _write_json(quality_path, quality)

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="propose",
    )

    assert context["medical_schema_issues"][0] == {
        "issue_kind": "medical_subcase_relation",
        "edge_id": "edge-flu-fever",
        "medical_subcase": "infectious_disease_symptom",
    }


def test_build_stage_context_marks_blank_provenance_as_missing(
    tmp_path: Path,
):
    package = _make_agent_package(tmp_path)
    snapshot_path = package / "snapshots" / "kg_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["nodes"][1]["source_id"] = f" {GRAPH_FIELD_SEP} "
    snapshot["edges"][0]["source_id"] = "   "
    _write_json(snapshot_path, snapshot)

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="locate_evidence",
        previous_outputs={
            "missing_branches": [{"key": "symptom", "label": "Symptoms"}]
        },
    )

    assert [
        window["evidence_status"] for window in context["evidence_windows"]
    ] == ["missing", "missing"]


def test_build_stage_context_prioritizes_quality_finding_relation_edges(
    tmp_path: Path,
):
    package = _make_agent_package(tmp_path)
    snapshot_path = package / "snapshots" / "kg_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["nodes"].append(
        {
            "id": "cough",
            "label": "Cough",
            "entity_type": "symptom",
            "source_id": "chunk-2",
            "file_path": "guide.md",
        }
    )
    snapshot["edges"].append(
        {
            "id": "edge-cough-flu",
            "source": "cough",
            "target": "flu",
            "keywords": "临床表现",
            "source_id": "chunk-2",
            "file_path": "guide.md",
        }
    )
    _write_json(snapshot_path, snapshot)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["findings"] = [
        {
            "severity": "medium",
            "category": "relation_semantics",
            "message": "Clinical manifestation relation direction should be normalized.",
            "evidence": ["edge:edge-cough-flu"],
            "suggested_fix_type": "normalize_relation_direction",
            "requires_approval": True,
        }
    ]
    _write_json(quality_path, quality)

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="locate_evidence",
        previous_outputs={},
    )

    assert [relation["id"] for relation in context["candidate_relations"]] == [
        "edge-cough-flu"
    ]
    assert {entity["id"] for entity in context["candidate_entities"]} == {
        "cough",
        "flu",
    }
    assert {
        (window["item_type"], window["item_id"])
        for window in context["evidence_windows"]
    } == {
        ("entity", "flu"),
        ("entity", "cough"),
        ("relation", "edge-cough-flu"),
    }


def test_stage_context_caps_quality_finding_edges_before_deriving_nodes(
    tmp_path: Path,
):
    package = _make_agent_package(tmp_path)
    snapshot_path = package / "snapshots" / "kg_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["nodes"] = []
    snapshot["edges"] = []
    for index in range(1000):
        source = f"source-{index}"
        target = f"target-{index}"
        snapshot["nodes"].extend(
            [
                {
                    "id": source,
                    "label": f"Source {index}",
                    "entity_type": "disease",
                    "source_id": f"chunk-source-{index}",
                    "file_path": "source.md",
                    "description": "verbose source node " * 80,
                },
                {
                    "id": target,
                    "label": f"Target {index}",
                    "entity_type": "symptom",
                    "source_id": f"chunk-target-{index}",
                    "file_path": "target.md",
                    "description": "verbose target node " * 80,
                },
            ]
        )
        snapshot["edges"].append(
            {
                "id": f"edge-{index}",
                "source": source,
                "target": target,
                "keywords": "has_manifestation",
                "source_id": f"chunk-edge-{index}",
                "file_path": "edges.md",
                "description": "verbose relation evidence " * 80,
            }
        )
    _write_json(snapshot_path, snapshot)

    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["findings"] = [
        {
            "severity": "medium",
            "category": "relation_semantics",
            "message": "Many relation edges need review. " * 30,
            "evidence": [f"edge:edge-{index}" for index in range(1000)],
            "suggested_fix_type": "normalize_relation_direction",
        }
    ]
    _write_json(quality_path, quality)

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="infer_branches",
        previous_outputs={
            "missing_branches": [
                {"key": "symptom", "label": "Symptoms", "notes": "x" * 5000}
            ],
            "large_llm_output": ["previous output " * 200 for _ in range(100)],
        },
    )

    assert len(context["candidate_relations"]) <= 20
    assert len(context["candidate_entities"]) <= 40
    assert len(context["evidence_windows"]) <= 40
    assert len(json.dumps(context, ensure_ascii=False)) < 45_000
    assert set(context["candidate_entities"][0]) == {
        "id",
        "label",
        "entity_type",
        "source_id",
        "file_path",
    }
    assert set(context["candidate_relations"][0]) == {
        "id",
        "source",
        "target",
        "keywords",
        "source_id",
        "file_path",
    }


def test_stage_context_caps_legacy_memory_text(tmp_path: Path):
    package = _make_agent_package(tmp_path)
    long_memory = "Do not pass raw legacy memory. " * 1000
    (package / "accepted_changes.md").write_text(long_memory, encoding="utf-8")
    (package / "rejected_changes.md").write_text(long_memory, encoding="utf-8")
    (package / "proposal_revision_requests.md").write_text(
        long_memory, encoding="utf-8"
    )

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="infer_branches",
    )

    rules_memory = context["rules_memory"]
    assert "agent_memory_summary" in rules_memory
    assert set(rules_memory) == {
        "accepted_changes",
        "rejected_changes",
        "proposal_revision_requests",
        "agent_memory_summary",
    }
    assert len(rules_memory["accepted_changes"]) < 1500
    assert len(rules_memory["rejected_changes"]) < 1500
    assert len(rules_memory["proposal_revision_requests"]) < 1500
    assert len(json.dumps(context, ensure_ascii=False)) < 45_000


def test_stage_context_cleanup_issue_retains_synonym_identifiers(
    tmp_path: Path,
):
    package = _make_agent_package(tmp_path)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["entity_cleanup_issues"] = [
        {
            "issue_kind": "synonym_cluster",
            "canonical_label": "Influenza",
            "entity_type": "disease",
            "node_ids": [f"flu-{index}" for index in range(50)],
            "nodes": [
                {
                    "id": f"flu-{index}",
                    "label": f"Influenza alias {index}",
                    "entity_type": "disease",
                    "source_id": f"chunk-{index}",
                    "file_path": "guide.md",
                    "description": "verbose alias detail " * 80,
                }
                for index in range(50)
            ],
            "guidance": "drop this verbose cleanup guidance " * 100,
        }
    ]
    _write_json(quality_path, quality)

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="propose",
    )

    issue = context["entity_cleanup_issues"][0]
    assert issue["canonical_label"] == "Influenza"
    assert issue["entity_type"] == "disease"
    assert issue["node_ids"][:2] == ["flu-0", "flu-1"]
    assert len(issue["node_ids"]) <= 20
    assert issue["nodes"][0] == {
        "id": "flu-0",
        "label": "Influenza alias 0",
        "entity_type": "disease",
        "source_id": "chunk-0",
        "file_path": "guide.md",
    }
    assert len(issue["nodes"]) <= 10
    assert "guidance" not in issue


def test_stage_context_cleanup_issue_retains_nested_node_ids(tmp_path: Path):
    package = _make_agent_package(tmp_path)
    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["details"]["entity_cleanup_issues"] = [
        {
            "issue_kind": "duplicate_node",
            "canonical_label": "Influenza",
            "nodes": [
                {
                    "node_id": "node:influenza:alias-a",
                    "label": "Influenza A",
                    "entity_type": "disease",
                    "source_id": "chunk-alias-a",
                    "file_path": "guide.md",
                    "description": "verbose alias detail " * 80,
                }
            ],
        }
    ]
    _write_json(quality_path, quality)

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="propose",
    )

    assert context["entity_cleanup_issues"][0]["nodes"][0] == {
        "node_id": "node:influenza:alias-a",
        "label": "Influenza A",
        "entity_type": "disease",
        "source_id": "chunk-alias-a",
        "file_path": "guide.md",
    }


def test_stage_context_uses_cleanup_connected_edges_for_value_node_context(
    tmp_path: Path,
):
    package = _make_agent_package(tmp_path)
    snapshot_path = package / "snapshots" / "kg_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["nodes"] = [
        {
            "id": "incident-42",
            "label": "Fall incident",
            "entity_type": "incident",
            "source_id": "chunk-incident",
            "file_path": "case.md",
        },
        {
            "id": "carrier-42",
            "label": "Patient A",
            "entity_type": "person",
            "source_id": "chunk-carrier",
            "file_path": "case.md",
        },
        {
            "id": "value-node-42",
            "label": "Severe",
            "entity_type": "qualifier_value",
            "source_id": "chunk-value",
            "file_path": "case.md",
        },
    ]
    snapshot["edges"] = [
        {
            "id": f"filler-edge-{index}",
            "source": f"filler-source-{index}",
            "target": f"filler-target-{index}",
            "keywords": "unrelated",
            "source_id": f"chunk-filler-{index}",
            "file_path": "filler.md",
        }
        for index in range(50)
    ]
    snapshot["edges"].append(
        {
            "id": "edge-carrier-incident",
            "source": "carrier-42",
            "target": "incident-42",
            "keywords": "experienced",
            "source_id": "chunk-carrier-incident",
            "file_path": "case.md",
        }
    )
    _write_json(snapshot_path, snapshot)

    quality_path = package / "snapshots" / "quality_score.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["findings"] = []
    quality["details"]["entity_cleanup_issues"] = [
        {
            "issue_kind": "value_node_to_qualifier",
            "node_id": "value-node-42",
            "connected_edge_ids": ["edge-carrier-incident"],
            "qualifier_key": "severity",
            "qualifier_value": "Severe",
        }
    ]
    _write_json(quality_path, quality)

    context = build_stage_context(
        package,
        workspace="incident_workspace",
        stage="propose",
    )

    assert context["entity_cleanup_issues"][0]["connected_edge_ids"] == [
        "edge-carrier-incident"
    ]
    assert [relation["id"] for relation in context["candidate_relations"]] == [
        "edge-carrier-incident"
    ]
    assert {entity["id"] for entity in context["candidate_entities"]} == {
        "carrier-42",
        "incident-42",
    }


def test_stage_context_caps_previous_output_top_level_keys(tmp_path: Path):
    package = _make_agent_package(tmp_path)
    previous_outputs = {
        f"stage_output_{index:03d}": {
            "items": [f"value {index}-{item}" for item in range(30)],
            "notes": "previous stage detail " * 200,
        }
        for index in range(125)
    }

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="propose",
        previous_outputs=previous_outputs,
    )

    assert len(context["previous_outputs"]) == 20
    assert list(context["previous_outputs"]) == [
        f"stage_output_{index:03d}" for index in range(20)
    ]
    assert len(json.dumps(context, ensure_ascii=False)) < 45_000


def test_write_agent_context_writes_stage_context_json(tmp_path: Path):
    package = tmp_path / "package"
    context = {"stage": "explain", "content": "explain context"}

    written = write_agent_context(package, "explain", context)

    assert written == package / "agent_context" / "explain-context.json"
    raw_context = written.read_text(encoding="utf-8")
    assert json.loads(raw_context) == context
    assert "explain context" in raw_context
    assert raw_context.endswith("\n")


def test_write_agent_context_replaces_windows_junction_without_touching_target(
    tmp_path: Path,
):
    if os.name != "nt":
        pytest.skip("Windows junction regression")

    package = tmp_path / "package"
    package.mkdir()
    external_context = tmp_path / "external_context"
    external_context.mkdir()
    external_file = external_context / "explain-context.json"
    external_file.write_text("outside sentinel", encoding="utf-8")
    context_junction = package / "agent_context"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(context_junction), str(external_context)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"junction creation unavailable: {result.stderr or result.stdout}")

    try:
        written = write_agent_context(package, "explain", {"safe": True})
    finally:
        if context_junction.exists() and _is_windows_junction(context_junction):
            subprocess.run(
                ["cmd", "/c", "rmdir", str(context_junction)],
                check=False,
                capture_output=True,
                text=True,
            )

    assert external_file.read_text(encoding="utf-8") == "outside sentinel"
    assert written == package / "agent_context" / "explain-context.json"
    assert not _is_windows_junction(package / "agent_context")
    assert json.loads(written.read_text(encoding="utf-8")) == {"safe": True}


def _make_agent_package(tmp_path: Path) -> Path:
    package = tmp_path / "package"
    snapshot_dir = package / "snapshots"
    snapshot_dir.mkdir(parents=True)
    _write_json(
        snapshot_dir / "kg_snapshot.json",
        {
            "workspace": "snapshot-workspace",
            "generated_at": "2026-06-18T00:00:00+08:00",
            "source_files": ["guide.md"],
            "metadata": {"profile": "medical"},
            "nodes": [
                {
                    "id": "flu",
                    "label": "Flu",
                    "entity_type": "disease",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                },
                {
                    "id": "fever",
                    "label": "Fever",
                    "entity_type": "symptom",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                },
            ],
            "edges": [
                {
                    "id": "edge-flu-fever",
                    "source": "flu",
                    "target": "fever",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                }
            ],
        },
    )
    _write_json(
        snapshot_dir / "quality_score.json",
        {
            "overall": 88,
            "subscores": {"hierarchy_completeness": 70},
            "metrics": {"hierarchy_missing_branch_count": 1},
            "findings": [
                {
                    "severity": "medium",
                    "category": "hierarchy_completeness",
                    "message": "Medical hierarchy is missing expected branches.",
                    "suggested_fix_type": "add_hierarchy_branch",
                }
            ],
            "critical_blockers": [],
            "details": {
                "hierarchy_branches": {
                    "required": [{"key": "disease", "label": "Diseases"}],
                    "present": [{"key": "disease", "label": "Diseases"}],
                    "missing": [{"key": "symptom", "label": "Symptoms"}],
                }
            },
        },
    )
    (package / "kb_context.md").write_text(
        "# Current KG context\n\nFlu connects to fever.\n", encoding="utf-8"
    )
    (package / "accepted_changes.md").write_text(
        "# Accepted Changes\n\n- Accepted rule.\n", encoding="utf-8"
    )
    (package / "rejected_changes.md").write_text(
        "# Rejected Changes\n\n- Rejected rule.\n", encoding="utf-8"
    )
    return package


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _is_windows_junction(path: Path) -> bool:
    if os.name != "nt" or not path.exists():
        return False
    return bool(os.stat(path, follow_symlinks=False).st_file_attributes & 0x400)
