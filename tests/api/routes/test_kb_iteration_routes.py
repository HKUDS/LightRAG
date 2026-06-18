from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient


HEADERS = {"X-API-Key": "test-key"}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _create_artifact_package(tmp_path: Path, workspace: str = "influenza_medical_v1"):
    output_root = tmp_path / "work" / "kb-iteration"
    storage_root = tmp_path / "data" / "rag_storage"
    input_root = tmp_path / "data" / "inputs"
    package = output_root / workspace
    snapshots = package / "snapshots"
    snapshots.mkdir(parents=True)
    (storage_root / workspace).mkdir(parents=True)
    (input_root / workspace).mkdir(parents=True)
    _write_json(
        snapshots / "kg_snapshot.json",
        {
            "workspace": workspace,
            "generated_at": "2026-06-17T00:00:00Z",
            "source_files": ["flu.pdf"],
            "metadata": {"profile": "clinical_guideline_zh"},
            "nodes": [
                {
                    "id": "influenza",
                    "label": "流行性感冒",
                    "entity_type": "Disease",
                    "description": "core disease",
                    "source_id": "chunk-1",
                    "file_path": "flu.pdf",
                    "properties": {},
                },
                {
                    "id": "symptom",
                    "label": "临床表现",
                    "entity_type": "MedicalGroup",
                    "description": "navigation group",
                    "source_id": "",
                    "file_path": "",
                    "properties": {"medical_group": "clinical_manifestation"},
                },
                {
                    "id": "persistent-fever",
                    "label": "高热不退",
                    "entity_type": "Symptom",
                    "description": "specific symptom",
                    "source_id": "chunk-1",
                    "file_path": "flu.pdf",
                    "properties": {"medical_group": "clinical_manifestation"},
                },
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "influenza",
                    "target": "persistent-fever",
                    "keywords": "临床表现",
                    "description": "relation evidence",
                    "source_id": "chunk-1",
                    "file_path": "flu.pdf",
                    "weight": 1.0,
                    "properties": {},
                }
            ],
        },
    )
    _write_json(snapshots / "entity_stats.json", [{"label": "Disease", "count": 1}])
    _write_json(snapshots / "relation_stats.json", [{"label": "临床表现", "count": 1}])
    _write_json(
        snapshots / "source_coverage.json",
        {"source_files": ["flu.pdf"], "coverage": 1.0},
    )
    _write_json(
        snapshots / "hierarchy_paths.json",
        [{"path": ["流行性感冒", "临床表现", "高热不退"]}],
    )
    _write_json(
        snapshots / "diff_summary.json",
        {
            "workspace": workspace,
            "added_nodes": [],
            "removed_nodes": [],
            "changed_entity_types": [],
            "added_edge_pairs": [],
            "removed_edge_pairs": [],
            "changed_relation_keywords": [],
            "dangerous_regression_flags": [],
            "quality_delta": {},
        },
    )
    _write_json(
        snapshots / "quality_score.json",
        {
            "overall": 82,
            "subscores": {"evidence_grounding": 91, "hierarchy_completeness": 76},
            "metrics": {"evidence_coverage": 91, "generic_relation_count": 0},
            "findings": [
                {
                    "severity": "high",
                    "category": "hierarchy",
                    "message": "症状分支缺少中间层",
                    "evidence": ["edge:e1"],
                    "suggested_fix_type": "hierarchy_rule_change",
                    "requires_approval": True,
                }
            ],
            "critical_blockers": [],
        },
    )
    _write_text(package / "kb_context.md", "# KB Context\n")
    _write_text(package / "entity_catalog.md", "# Entity Catalog\n")
    _write_text(package / "relation_catalog.md", "# Relation Catalog\n")
    _write_text(package / "kg_structure.md", "# KG Structure\n")
    _write_text(package / "quality_report.md", "# Quality\n")
    _write_text(
        package / "approval_queue.md",
        "\n".join(
            [
                "# Approval Queue",
                "",
                "proposals:",
                "- id: p1",
                "  type: hierarchy_rule_change",
                "  target: kg_structure.md",
                "  proposed_change: Add symptom layer",
                "  reason: Improve hierarchy",
                "  evidence:",
                "  - edge:e1",
                "  confidence: 0.8",
                "  risk: medium",
                "  requires_approval: true",
                "  expected_metric_change: {}",
                "- id: p2",
                "  type: web_display_change",
                "  target: MedicalHierarchyGraph.tsx",
                "  proposed_change: Improve legend",
                "  reason: Improve readability",
                "  evidence:",
                "  - edge:e1",
                "  confidence: 0.7",
                "  risk: low",
                "  requires_approval: true",
                "  expected_metric_change: {}",
                "- id: p3",
                "  type: quality_report_note",
                "  target: quality_report.md",
                "  proposed_change: Add note",
                "  reason: Needs later review",
                "  evidence:",
                "  - edge:e1",
                "  confidence: 0.6",
                "  risk: low",
                "  requires_approval: true",
                "  expected_metric_change: {}",
                "",
            ]
        ),
    )
    _write_text(
        package / "improvement_backlog.md",
        "# Improvement Backlog\n\nproposals:\n- id: p1\n  requires_approval: true\n- id: p2\n  requires_approval: true\n- id: p3\n  requires_approval: true\n",
    )
    _write_text(package / "quality_rules.md", "# Quality Rules\n")
    _write_text(package / "known_issues.md", "# Known Issues\n")
    _write_text(package / "accepted_changes.md", "# Accepted Changes\n")
    _write_text(package / "rejected_changes.md", "# Rejected Changes\n")
    _write_text(package / "deferred_changes.md", "# Deferred Changes\n")
    _write_text(package / "diff_report.md", "# Diff Report\n")
    _write_text(
        package / "iteration_log.md",
        "\n".join(
            [
                "## Run",
                "",
                f"- workspace: {workspace}",
                "- phase: pending_user_review",
                "- quality_score: snapshots/quality_score.json",
                "",
            ]
        ),
    )
    return SimpleNamespace(
        workspace=workspace,
        output_root=output_root,
        storage_root=storage_root,
        input_root=input_root,
        package=package,
    )


def _client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers.kb_iteration_routes import create_kb_iteration_routes

    fixture = _create_artifact_package(tmp_path)
    args = SimpleNamespace(
        workspace=fixture.workspace,
        working_dir=str(fixture.storage_root),
        input_dir=str(fixture.input_root),
        kb_iteration_output_dir=str(fixture.output_root),
    )
    app = FastAPI()
    app.include_router(
        create_kb_iteration_routes(SimpleNamespace(), args, api_key="test-key")
    )
    return TestClient(app), fixture


def test_summary_reads_latest_artifacts_without_path_input(tmp_path: Path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get(
        "/kb-iteration/influenza_medical_v1/summary",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace"] == "influenza_medical_v1"
    assert payload["latestRunId"] == "latest"
    assert payload["phase"] == "pending_user_review"
    assert payload["quality"]["overall"] == 82
    assert payload["counts"] == {"nodes": 3, "edges": 1, "sources": 1}
    assert payload["pendingApprovalCount"] == 3


def test_workspaces_lists_existing_iteration_packages(tmp_path: Path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get("/kb-iteration/workspaces", headers=HEADERS)

    assert response.status_code == 200
    assert response.json()["workspaces"] == ["influenza_medical_v1"]


def test_artifact_key_is_whitelisted(tmp_path: Path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/%2E%2E%2F.env",
        headers=HEADERS,
    )

    assert response.status_code == 400


def test_invalid_workspace_is_rejected(tmp_path: Path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get(
        "/kb-iteration/..%5Csecret/summary",
        headers=HEADERS,
    )

    assert response.status_code == 400


def test_graph_exposes_directional_relation_labels(tmp_path: Path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get(
        "/kb-iteration/influenza_medical_v1/graph",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["nodes"][0]["role"] == "disease"
    edge = payload["edges"][0]
    assert edge["label"] == "临床表现"
    assert edge["direction"] == "outgoing"
    assert edge["sourceLabel"] == "流行性感冒"
    assert edge["targetLabel"] == "高热不退"
    assert "邻接" not in edge["label"]


def test_artifact_endpoint_reads_whitelisted_markdown(tmp_path: Path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/quality_report",
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {
        "artifactKey": "quality_report",
        "contentType": "text/markdown",
        "content": "# Quality\n",
    }


def test_accept_reject_and_defer_records_are_append_only(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)

    reject = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/reject",
        headers=HEADERS,
        json={"reviewer": "maintainer", "reason": "Needs stronger evidence"},
    )
    accept = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p2/accept",
        headers=HEADERS,
        json={"reviewer": "maintainer", "reason": "Evidence checked"},
    )
    defer = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p3/defer",
        headers=HEADERS,
        json={"reviewer": "maintainer", "reason": "Wait for another run"},
    )

    assert reject.status_code == 200
    assert accept.status_code == 200
    assert defer.status_code == 200
    assert "Needs stronger evidence" in (
        fixture.package / "rejected_changes.md"
    ).read_text(encoding="utf-8")
    assert "Evidence checked" in (
        fixture.package / "accepted_changes.md"
    ).read_text(encoding="utf-8")
    assert "Wait for another run" in (
        fixture.package / "deferred_changes.md"
    ).read_text(encoding="utf-8")


def test_proposal_decision_rejects_unknown_proposal(tmp_path: Path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/not-in-queue/accept",
        headers=HEADERS,
        json={"reviewer": "maintainer", "reason": "No matching proposal"},
    )

    assert response.status_code == 404


def test_proposal_decision_rejects_conflicting_decision(
    tmp_path: Path, monkeypatch
):
    client, _ = _client(tmp_path, monkeypatch)

    first = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/accept",
        headers=HEADERS,
        json={"reviewer": "maintainer", "reason": "Evidence checked"},
    )
    second = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/reject",
        headers=HEADERS,
        json={"reviewer": "maintainer", "reason": "Changed mind"},
    )

    assert first.status_code == 200
    assert second.status_code == 409


def test_run_scoped_artifact_routes_match_latest_routes(tmp_path: Path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    snapshot = client.get(
        "/kb-iteration/influenza_medical_v1/runs/latest/snapshot",
        headers=HEADERS,
    )
    quality = client.get(
        "/kb-iteration/influenza_medical_v1/runs/latest/quality",
        headers=HEADERS,
    )
    entities = client.get(
        "/kb-iteration/influenza_medical_v1/runs/latest/catalog/entities",
        headers=HEADERS,
    )
    relations = client.get(
        "/kb-iteration/influenza_medical_v1/runs/latest/catalog/relations",
        headers=HEADERS,
    )
    diff = client.get(
        "/kb-iteration/influenza_medical_v1/runs/latest/diff",
        headers=HEADERS,
    )

    assert snapshot.status_code == 200
    assert snapshot.json()["workspace"] == "influenza_medical_v1"
    assert quality.status_code == 200
    assert quality.json()["quality"]["overall"] == 82
    assert entities.status_code == 200
    assert len(entities.json()["entities"]) == 3
    assert relations.status_code == 200
    assert len(relations.json()["relations"]) == 1
    assert diff.status_code == 200


def test_run_trigger_rejects_when_workspace_file_lock_is_held(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    with kb_iteration_routes._exclusive_workspace_file_lock(
        fixture.output_root, fixture.workspace
    ):
        response = client.post(
            "/kb-iteration/influenza_medical_v1/runs",
            headers=HEADERS,
            json={"profile": "clinical_guideline_zh"},
        )

    assert response.status_code == 409


def test_workspace_file_lock_failed_acquire_preserves_existing_lock(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    _, fixture = _client(tmp_path, monkeypatch)
    lock_path = fixture.package / ".kb_iteration_run.lock"
    with kb_iteration_routes._exclusive_workspace_file_lock(
        fixture.output_root, fixture.workspace
    ):
        assert lock_path.exists()

        with pytest.raises(HTTPException) as exc_info:
            with kb_iteration_routes._exclusive_workspace_file_lock(
                fixture.output_root, fixture.workspace
            ):
                pass

        assert exc_info.value.status_code == 409
        assert lock_path.exists()

    assert not lock_path.exists()


def test_run_trigger_uses_validated_roots(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    calls = []

    def fake_run_iteration(**kwargs):
        calls.append(kwargs)
        _write_json(
            fixture.package / "snapshots" / "quality_score.json",
            {
                "overall": 88,
                "subscores": {},
                "metrics": {},
                "findings": [],
                "critical_blockers": [],
            },
        )
        return SimpleNamespace(
            output_dir=fixture.package,
            snapshot=SimpleNamespace(nodes=[], edges=[], source_files=[]),
            quality_score=SimpleNamespace(to_dict=lambda: {"overall": 88}),
            artifact_paths={},
        )

    monkeypatch.setattr(kb_iteration_routes, "run_iteration", fake_run_iteration)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/runs",
        headers=HEADERS,
        json={"profile": "clinical_guideline_zh"},
    )

    assert response.status_code == 200
    assert response.json()["phase"] == "pending_user_review"
    assert calls == [
        {
            "workspace": "influenza_medical_v1",
            "storage_root": fixture.storage_root,
            "input_root": fixture.input_root,
            "output_root": fixture.output_root,
            "profile": "clinical_guideline_zh",
        }
    ]


def test_llm_review_artifact_routes_read_whitelisted_outputs(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)
    _write_text(fixture.package / "llm_review_report.md", "# LLM Review\n")
    _write_json(
        fixture.package / "llm_review_trace.json",
        {"stop_reason": "pending_human_review"},
    )
    _write_text(
        fixture.package / "proposals.generated.yaml",
        "# Generated\nproposals: []\n",
    )
    _write_text(fixture.package / "llm_judge_report.md", "# Judge\n")
    _write_json(
        fixture.package / "review_context" / "round-001-context.json",
        {"focus": ["generic_relation"]},
    )
    _write_text(
        fixture.package / "patch_candidates" / "proposal-1.patch",
        "--- a/x\n+++ b/x\n",
    )

    report = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/report",
        headers=HEADERS,
    )
    trace = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/trace",
        headers=HEADERS,
    )
    context = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/context/round-001",
        headers=HEADERS,
    )
    proposals = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/proposals",
        headers=HEADERS,
    )
    judge_report = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/judge-report",
        headers=HEADERS,
    )
    patch = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/patches/proposal-1",
        headers=HEADERS,
    )

    assert report.status_code == 200
    assert report.json()["content"] == "# LLM Review\n"
    assert trace.status_code == 200
    assert trace.json()["payload"]["stop_reason"] == "pending_human_review"
    assert context.status_code == 200
    assert context.json()["payload"]["focus"] == ["generic_relation"]
    assert proposals.status_code == 200
    assert proposals.json()["content"] == "# Generated\nproposals: []\n"
    assert judge_report.status_code == 200
    assert judge_report.json()["content"] == "# Judge\n"
    assert patch.status_code == 200
    assert patch.json()["artifactKey"] == "patch_candidates/proposal-1.patch"
    assert patch.json()["contentType"] == "text/x-diff"
    assert "--- a/x" in patch.json()["content"]


def test_llm_review_context_rejects_invalid_round_id(tmp_path: Path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/context/latest",
        headers=HEADERS,
    )

    assert response.status_code == 400


def test_llm_review_patch_route_rejects_invalid_proposal_id(
    tmp_path: Path, monkeypatch
):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/patches/bad$id",
        headers=HEADERS,
    )

    assert response.status_code == 400


def test_llm_review_patch_route_reads_only_patch_candidates(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)
    _write_text(fixture.package / "proposal-outside.patch", "outside\n")

    outside_only = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/patches/proposal-outside",
        headers=HEADERS,
    )

    assert outside_only.status_code == 404

    _write_text(
        fixture.package / "patch_candidates" / "proposal-outside.patch",
        "inside\n",
    )
    inside = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/patches/proposal-outside",
        headers=HEADERS,
    )

    assert inside.status_code == 200
    assert inside.json()["content"] == "inside\n"


def test_llm_review_patch_candidate_paths_are_confined_to_patch_dir(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    fixture = _create_artifact_package(tmp_path)
    args = SimpleNamespace(
        workspace=fixture.workspace,
        working_dir=str(fixture.storage_root),
        input_dir=str(fixture.input_root),
        kb_iteration_output_dir=str(fixture.output_root),
    )

    valid_path = kb_iteration_routes._safe_patch_candidate_path(
        args, fixture.workspace, "proposal-1"
    )
    assert valid_path == (
        fixture.package / "patch_candidates" / "proposal-1.patch"
    ).resolve()

    with pytest.raises(HTTPException) as exc_info:
        kb_iteration_routes._safe_patch_candidate_path(
            args, fixture.workspace, "../outside"
        )

    assert exc_info.value.status_code == 400


def test_llm_review_run_default_unavailable_client_returns_503(
    tmp_path: Path, monkeypatch
):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/llm-review/runs",
        headers=HEADERS,
        json={"profile": "clinical_guideline_zh"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "LLM review client is not configured"


def test_llm_review_run_missing_workspace_does_not_create_phantom_package(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)
    missing_workspace = "unreviewed_workspace"

    response = client.post(
        f"/kb-iteration/{missing_workspace}/llm-review/runs",
        headers=HEADERS,
        json={"profile": "clinical_guideline_zh"},
    )

    assert not (fixture.output_root / missing_workspace).exists()
    assert response.status_code == 404

    workspaces_response = client.get("/kb-iteration/workspaces", headers=HEADERS)

    assert workspaces_response.status_code == 200
    assert workspaces_response.json()["workspaces"] == ["influenza_medical_v1"]


def test_llm_review_run_rejects_when_deterministic_file_lock_is_held(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    calls = []

    class FakeClient:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return "{}"

    def fake_run_llm_review_loop(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            output_dir=fixture.package,
            stop_reason="should_not_run",
            proposal_ids=[],
            artifact_paths={},
        )

    monkeypatch.setattr(
        kb_iteration_routes,
        "run_llm_review_loop",
        fake_run_llm_review_loop,
    )
    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda rag: FakeClient(),
    )

    with kb_iteration_routes._exclusive_workspace_file_lock(
        fixture.output_root, fixture.workspace
    ):
        response = client.post(
            "/kb-iteration/influenza_medical_v1/llm-review/runs",
            headers=HEADERS,
            json={"profile": "clinical_guideline_zh"},
        )

    assert response.status_code == 409
    assert calls == []


def test_llm_review_run_uses_lock_and_validated_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    calls = []

    class FakeClient:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return json.dumps(
                {
                    "confirmed_issues": [],
                    "hypotheses": [],
                    "missing_evidence": [],
                    "out_of_scope": [],
                    "proposals": [],
                }
            )

    def fake_run_llm_review_loop(**kwargs):
        calls.append(kwargs)
        _write_json(
            fixture.package / "llm_review_trace.json",
            {"stop_reason": "all_proposals_invalid"},
        )
        _write_text(fixture.package / "llm_review_report.md", "# Review\n")
        _write_text(fixture.package / "proposals.generated.yaml", "proposals: []\n")
        return SimpleNamespace(
            output_dir=fixture.package,
            stop_reason="all_proposals_invalid",
            proposal_ids=[],
            artifact_paths={},
        )

    monkeypatch.setattr(
        kb_iteration_routes,
        "run_llm_review_loop",
        fake_run_llm_review_loop,
    )
    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda rag: FakeClient(),
    )

    response = client.post(
        "/kb-iteration/influenza_medical_v1/llm-review/runs",
        headers=HEADERS,
        json={"profile": "clinical_guideline_zh", "max_review_rounds": 1},
    )

    assert response.status_code == 200
    assert response.json()["stopReason"] == "all_proposals_invalid"
    assert calls[0]["workspace"] == "influenza_medical_v1"
    assert calls[0]["package_dir"] == fixture.package
