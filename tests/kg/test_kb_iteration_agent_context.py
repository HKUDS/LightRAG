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
    assert "Current KG context" in observation["kb_context"]
    assert "Accepted rule" in observation["rules_memory"]["accepted_changes"]
    assert "Rejected rule" in observation["rules_memory"]["rejected_changes"]


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
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "evidence_status": "grounded",
        },
        {
            "item_type": "relation",
            "item_id": "edge-flu-fever",
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "evidence_status": "grounded",
        },
    ]


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
