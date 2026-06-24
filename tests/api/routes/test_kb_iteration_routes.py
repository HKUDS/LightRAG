from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from lightrag.kb_iteration.apply import (
    AcceptedApplyResult,
    ApplyChange,
    ApplyChangeStatus,
)


HEADERS = {"X-API-Key": "test-key"}


def test_kb_iteration_artifacts_include_schema_and_memory_outputs():
    original_argv = sys.argv
    sys.argv = ["lightrag-server"]
    try:
        from lightrag.api.routers.kb_iteration_routes import ARTIFACTS
    finally:
        sys.argv = original_argv

    assert "agent_memory_summary" in ARTIFACTS
    assert "approval_queue" in ARTIFACTS
    assert "accepted_changes_apply_result" in ARTIFACTS
    assert ARTIFACTS["quality_score"][0] == "snapshots/quality_score.json"


class FakeZhClient:
    def __init__(self, response: str = "## 中文显示\n") -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {"system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        return self.response


class FailingZhClient:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("translation failed")


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


def test_display_artifact_markdown_generation_writes_zh_file_and_metadata(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("KB_ITERATION_LLM_MODEL", "unit-test-model")
    client, fixture = _client(tmp_path, monkeypatch)
    from lightrag.api.routers import kb_iteration_routes

    zh_client = FakeZhClient("## 质量报告\n\n保留 source_id。\n")
    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda _rag: zh_client,
    )

    response = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/quality_report/display",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifactKey"] == "quality_report"
    assert payload["contentType"] == "text/markdown"
    assert payload["content"] == "## 质量报告\n\n保留 source_id。\n"
    assert payload["display"]["language"] == "zh"
    assert payload["display"]["sourceFile"] == "quality_report.md"
    assert payload["display"]["zhFile"] == "quality_report.zh.md"
    assert payload["display"]["generated"] is True
    assert payload["display"]["fallbackToSource"] is False
    assert payload["display"]["generatedAt"]
    assert payload["display"]["model"] == "unit-test-model"
    assert payload["display"]["error"] is None
    assert (fixture.package / "quality_report.zh.md").read_text(
        encoding="utf-8"
    ) == "## 质量报告\n\n保留 source_id。\n"
    assert len(zh_client.calls) == 1


def test_display_artifact_json_generation_writes_zh_file_with_labels(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)
    from lightrag.api.routers import kb_iteration_routes

    zh_client = FakeZhClient()
    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda _rag: zh_client,
    )

    response = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/quality_score/display",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifactKey"] == "quality_score"
    assert payload["contentType"] == "application/json"
    assert "content" not in payload
    assert "overall" in payload["payload"]["_zh_labels"]
    assert "metrics" in payload["payload"]["_zh_labels"]
    assert payload["display"]["zhFile"] == "snapshots/quality_score.zh.json"
    assert payload["display"]["generated"] is True
    assert payload["display"]["fallbackToSource"] is False
    assert (fixture.package / "snapshots" / "quality_score.zh.json").exists()
    assert zh_client.calls == []


def test_display_artifact_fallback_on_client_failure_returns_source_content(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)
    from lightrag.api.routers import kb_iteration_routes

    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda _rag: FailingZhClient(),
    )

    response = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/quality_report/display",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["content"] == "# Quality\n"
    assert payload["display"]["fallbackToSource"] is True
    assert payload["display"]["generated"] is False
    assert payload["display"]["error"] == "translation failed"
    assert not (fixture.package / "quality_report.zh.md").exists()


def test_display_artifact_markdown_falls_back_when_llm_env_is_misconfigured(
    tmp_path: Path, monkeypatch
):
    secret = "sk-test-display-secret"
    monkeypatch.setenv("KB_ITERATION_LLM_BINDING", "anthropic")
    monkeypatch.setenv("KB_ITERATION_LLM_BINDING_HOST", "https://llm.example.invalid")
    monkeypatch.setenv("KB_ITERATION_LLM_BINDING_API_KEY", secret)
    monkeypatch.setenv("KB_ITERATION_LLM_MODEL", "claude-test")
    client, fixture = _client(tmp_path, monkeypatch)

    response = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/quality_report/display",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["content"] == "# Quality\n"
    assert payload["display"]["fallbackToSource"] is True
    assert payload["display"]["generated"] is False
    assert payload["display"]["error"]
    assert secret not in payload["display"]["error"]
    assert not (fixture.package / "quality_report.zh.md").exists()


def test_display_artifact_json_ignores_misconfigured_llm_env(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("KB_ITERATION_LLM_BINDING", "anthropic")
    monkeypatch.setenv("KB_ITERATION_LLM_BINDING_HOST", "https://llm.example.invalid")
    monkeypatch.setenv("KB_ITERATION_LLM_BINDING_API_KEY", "sk-json-secret")
    monkeypatch.setenv("KB_ITERATION_LLM_MODEL", "claude-test")
    client, fixture = _client(tmp_path, monkeypatch)

    response = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/quality_score/display",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert "overall" in payload["payload"]["_zh_labels"]
    assert payload["display"]["generated"] is True
    assert payload["display"]["fallbackToSource"] is False
    assert payload["display"]["error"] is None
    assert (fixture.package / "snapshots" / "quality_score.zh.json").exists()


@pytest.mark.parametrize(
    "artifact_key",
    ["unknown", "%2E%2E%2F.env", "quality_report.zh"],
)
def test_display_artifact_rejects_unknown_path_like_and_zh_source_keys(
    tmp_path: Path, monkeypatch, artifact_key: str
):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get(
        f"/kb-iteration/influenza_medical_v1/artifacts/{artifact_key}/display",
        headers=HEADERS,
    )

    assert response.status_code == 400


def test_artifact_manifest_includes_zh_display_metadata_and_exists_updates(
    tmp_path: Path, monkeypatch
):
    client, _ = _client(tmp_path, monkeypatch)
    from lightrag.api.routers import kb_iteration_routes

    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda _rag: FakeZhClient("## 质量报告\n"),
    )

    before = client.get(
        "/kb-iteration/influenza_medical_v1/runs/latest/artifacts",
        headers=HEADERS,
    )
    assert before.status_code == 200
    quality_report = next(
        item for item in before.json()["artifacts"] if item["key"] == "quality_report"
    )
    assert quality_report["sourceFile"] == "quality_report.md"
    assert quality_report["display"] == {
        "language": "zh",
        "zhFile": "quality_report.zh.md",
        "zhExists": False,
    }

    display = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/quality_report/display",
        headers=HEADERS,
    )
    assert display.status_code == 200

    after = client.get(
        "/kb-iteration/influenza_medical_v1/runs/latest/artifacts",
        headers=HEADERS,
    )
    quality_report = next(
        item for item in after.json()["artifacts"] if item["key"] == "quality_report"
    )
    assert quality_report["display"]["zhExists"] is True


def test_display_artifact_regenerate_overwrites_existing_markdown(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)
    from lightrag.api.routers import kb_iteration_routes

    zh_client = FakeZhClient("## New Chinese\n")
    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda _rag: zh_client,
    )
    _write_text(fixture.package / "quality_report.zh.md", "## Old Chinese\n")

    response = client.post(
        "/kb-iteration/influenza_medical_v1/artifacts/quality_report/display/regenerate",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["content"] == "## New Chinese\n"
    assert payload["display"]["generated"] is True
    assert payload["display"]["fallbackToSource"] is False
    assert (fixture.package / "quality_report.zh.md").read_text(
        encoding="utf-8"
    ) == "## New Chinese\n"
    assert len(zh_client.calls) == 1


def test_accept_reject_and_defer_records_are_append_only(tmp_path: Path, monkeypatch):
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
    reject_record = reject.json()["record"]
    assert reject_record["decision_reason_code"] == "insufficient_evidence"
    assert reject_record["rejection_scope"] == "semantic_until_evidence_change"
    assert len(reject_record["semantic_fingerprint"]) == 64
    assert len(reject_record["execution_fingerprint"]) == 64
    assert len(reject_record["evidence_fingerprint"]) == 64
    assert reject_record["action_payload"] == {}
    assert "Needs stronger evidence" in (
        fixture.package / "rejected_changes.md"
    ).read_text(encoding="utf-8")
    assert "Evidence checked" in (fixture.package / "accepted_changes.md").read_text(
        encoding="utf-8"
    )
    assert "Wait for another run" in (
        fixture.package / "deferred_changes.md"
    ).read_text(encoding="utf-8")
    memory_summary = (fixture.package / "agent_memory_summary.md").read_text(
        encoding="utf-8"
    )
    assert "Evidence checked" in memory_summary
    assert "Needs stronger evidence" in memory_summary
    assert "Wait for another run" not in memory_summary


def test_proposal_decision_accepts_empty_review_and_records_defaults(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p2/accept",
        headers=HEADERS,
        json={},
    )

    assert response.status_code == 200
    record = response.json()["record"]
    assert record["reviewer"] == "maintainer"
    assert record["reason"]
    assert record["impact_scope"]
    assert record["verification"]
    assert "p2" in record["reason"]
    assert "web_display_change" in record["impact_scope"]
    assert record["schema_version"]
    assert record["decision_reason_code"] == "accepted"
    assert record["rejection_scope"] == "exact_action"
    assert record["semantic_rejection_class"] == "accepted"
    assert record["action_payload"] == {}
    assert len(record["semantic_fingerprint"]) == 64
    assert len(record["execution_fingerprint"]) == 64
    assert len(record["evidence_fingerprint"]) == 64
    accepted = (fixture.package / "accepted_changes.md").read_text(encoding="utf-8")
    assert "p2" in accepted
    assert "web_display_change" in accepted


def test_proposal_decision_rejects_invalid_rejection_scope(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/reject",
        headers=HEADERS,
        json={
            "reviewer": "maintainer",
            "reason": "Invalid scope should not persist",
            "rejection_scope": "forever",
        },
    )

    assert response.status_code == 422
    assert "forever" not in (
        fixture.package / "rejected_changes.md"
    ).read_text(encoding="utf-8")


def test_proposal_decision_reject_defer_only_scope_is_not_persisted(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/reject",
        headers=HEADERS,
        json={
            "reviewer": "maintainer",
            "reason": "Reject should not use defer-only scope",
            "rejection_scope": "defer_only",
        },
    )

    assert response.status_code == 400
    rejected = (fixture.package / "rejected_changes.md").read_text(encoding="utf-8")
    assert "defer_only" not in rejected
    assert "Reject should not use defer-only scope" not in rejected


def test_proposal_revision_request_accepts_empty_body_and_records_defaults(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/revision-request",
        headers=HEADERS,
        json={},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace"] == "influenza_medical_v1"
    assert payload["proposalId"] == "p1"
    assert payload["artifactKey"] == "proposal_revision_requests"
    record = payload["record"]
    assert record["proposal_id"] == "p1"
    assert record["reviewer"] == "maintainer"
    assert record["proposal_type"] == "hierarchy_rule_change"
    assert record["proposal_target"] == "kg_structure.md"
    assert record["proposal_risk"] == "medium"
    assert "revision" in record["reason"].lower()
    assert record["instruction"]
    instruction = record["instruction"].lower()
    assert "revision" in instruction
    assert "agent" in instruction
    assert "proposal" in instruction
    assert record["requested_at"]

    revision_requests = (
        fixture.package / "proposal_revision_requests.md"
    ).read_text(encoding="utf-8")
    assert "p1" in revision_requests
    assert "hierarchy_rule_change" in revision_requests
    assert "kg_structure.md" in revision_requests
    assert record["instruction"] in revision_requests

    iteration_log = (fixture.package / "iteration_log.md").read_text(encoding="utf-8")
    assert "- phase: proposal_revision_request" in iteration_log
    assert "- event: revision_request_queued" in iteration_log
    assert "- proposal_id: p1" in iteration_log
    memory_summary = (fixture.package / "agent_memory_summary.md").read_text(
        encoding="utf-8"
    )
    assert "p1" in memory_summary
    assert record["instruction"] in memory_summary

    artifact = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/proposal_revision_requests",
        headers=HEADERS,
    )
    assert artifact.status_code == 200
    assert artifact.json()["content"] == revision_requests


def test_proposal_revision_request_defaults_are_neutral_without_rejection(
    tmp_path: Path, monkeypatch
):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/revision-request",
        headers=HEADERS,
        json={},
    )

    assert response.status_code == 200
    record = response.json()["record"]
    reason = record["reason"].lower()
    instruction = record["instruction"].lower()
    assert "rejected proposal" not in reason
    assert "rejected proposal" not in instruction
    assert "proposal revision request" in reason
    assert "revise proposal" in instruction


def test_proposal_revision_request_preserves_reason(tmp_path: Path, monkeypatch):
    client, fixture = _client(tmp_path, monkeypatch)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/revision-request",
        headers=HEADERS,
        json={"reason": "Evidence missing source_id"},
    )

    assert response.status_code == 200
    assert response.json()["record"]["reason"] == "Evidence missing source_id"
    revision_requests = (
        fixture.package / "proposal_revision_requests.md"
    ).read_text(encoding="utf-8")
    assert "Evidence missing source_id" in revision_requests


def test_proposal_revision_request_rejects_unknown_proposal(
    tmp_path: Path, monkeypatch
):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/not-in-queue/revision-request",
        headers=HEADERS,
        json={},
    )

    assert response.status_code == 404


@pytest.mark.parametrize("proposal_id", ["bad%20id", "bad%0Aid", "bad:id", "bad%5Cid"])
def test_proposal_revision_request_rejects_invalid_proposal_id(
    tmp_path: Path, monkeypatch, proposal_id: str
):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post(
        f"/kb-iteration/influenza_medical_v1/proposals/{proposal_id}/revision-request",
        headers=HEADERS,
        json={},
    )

    assert response.status_code == 400


def test_execute_accepted_changes_without_records_writes_empty_apply_result_artifact(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/accepted-changes/execute",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace"] == "influenza_medical_v1"
    assert payload["status"] == "no_accepted_changes"
    assert payload["proposalIds"] == []
    assert payload["appliedCount"] == 0
    assert payload["blockedCount"] == 0
    assert payload["artifactKey"] == "accepted_changes_apply_result"

    apply_result_json = fixture.package / "accepted_changes_apply_result.json"
    apply_result_markdown = fixture.package / "accepted_changes_apply_result.md"
    assert apply_result_json.exists()
    assert apply_result_markdown.exists()
    apply_payload = json.loads(apply_result_json.read_text(encoding="utf-8"))
    assert apply_payload["proposal_ids"] == []
    assert apply_payload["applied_count"] == 0
    assert apply_payload["blocked_count"] == 0
    assert apply_payload["quality_before"]["overall"] == 82
    assert apply_payload["quality_after"]["overall"] == 82
    memory_summary = fixture.package / "agent_memory_summary.md"
    assert memory_summary.exists()
    assert "Agent Memory Summary" in memory_summary.read_text(encoding="utf-8")

    artifact = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/accepted_changes_apply_result",
        headers=HEADERS,
    )
    assert artifact.status_code == 200
    assert "# Apply Result" in artifact.json()["content"]
    assert "No changes recorded." in artifact.json()["content"]

    iteration_log = (fixture.package / "iteration_log.md").read_text(encoding="utf-8")
    assert "- phase: accepted_changes_apply" in iteration_log
    assert "- accepted_proposal_ids: none" in iteration_log


def test_execute_accepted_changes_applies_kg_changes_without_llm_client(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)

    accept = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/accept",
        headers=HEADERS,
        json={},
    )
    assert accept.status_code == 200

    import lightrag.api.routers.kb_iteration_routes as routes

    monkeypatch.setattr(
        routes,
        "_default_llm_review_client",
        lambda _rag: pytest.fail("LLM client must not be required"),
    )
    monkeypatch.setattr(
        routes,
        "load_proposals_by_id",
        lambda package_dir: {
            "p1": {
                "id": "p1",
                "type": "add_hierarchy_branch",
                "target": "kg_structure.md",
            }
        },
        raising=False,
    )

    async def fake_apply_accepted_changes_to_graph(
        *, rag, workspace, records, proposals_by_id
    ):
        assert workspace == "influenza_medical_v1"
        assert [record["proposal_id"] for record in records] == ["p1"]
        assert proposals_by_id["p1"]["type"] == "add_hierarchy_branch"
        return AcceptedApplyResult(
            workspace=workspace,
            applied_at="2026-06-19T00:00:00+00:00",
            source_artifact="accepted_changes.md",
            proposal_ids=["p1"],
            changes=[
                ApplyChange(
                    proposal_id="p1",
                    proposal_type="add_hierarchy_branch",
                    target="kg_structure.md",
                    status=ApplyChangeStatus.APPLIED,
                    action="add_hierarchy_branch",
                    branch_key="clinical_manifestation",
                )
            ],
        )

    monkeypatch.setattr(
        routes,
        "apply_accepted_changes_to_graph",
        fake_apply_accepted_changes_to_graph,
        raising=False,
    )

    run_calls = []

    def fake_run_iteration(**kwargs):
        run_calls.append(kwargs)
        assert kwargs["workspace"] == "influenza_medical_v1"
        assert kwargs["profile"] == "clinical_guideline_zh"
        _write_json(
            fixture.package / "snapshots" / "quality_score.json",
            {"overall": 88, "metrics": {"hierarchy_missing_branch_count": 0}},
        )

    monkeypatch.setattr(routes, "run_iteration", fake_run_iteration)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/accepted-changes/execute",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace"] == "influenza_medical_v1"
    assert payload["status"] == "applied_changes"
    assert payload["proposalIds"] == ["p1"]
    assert payload["appliedCount"] == 1
    assert payload["blockedCount"] == 0
    assert payload["artifactKey"] == "accepted_changes_apply_result"
    assert len(run_calls) == 1

    apply_payload = json.loads(
        (fixture.package / "accepted_changes_apply_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert apply_payload["proposal_ids"] == ["p1"]
    assert apply_payload["applied_count"] == 1
    assert apply_payload["blocked_count"] == 0
    assert apply_payload["quality_before"]["overall"] == 82
    assert apply_payload["quality_after"]["overall"] == 88

    apply_result = (fixture.package / "accepted_changes_apply_result.md").read_text(
        encoding="utf-8"
    )
    assert "# Apply Result" in apply_result
    assert "- Applied: 1" in apply_result
    assert "- p1: applied (clinical_manifestation)" in apply_result

    iteration_log = (fixture.package / "iteration_log.md").read_text(encoding="utf-8")
    assert "- phase: accepted_changes_apply" in iteration_log
    assert "- accepted_proposal_ids: p1" in iteration_log

    artifact = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/accepted_changes_apply_result",
        headers=HEADERS,
    )
    assert artifact.status_code == 200
    assert "# Apply Result" in artifact.json()["content"]


def test_execute_accepted_changes_writes_apply_result_when_rerun_fails(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)

    accept = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/accept",
        headers=HEADERS,
        json={},
    )
    assert accept.status_code == 200

    import lightrag.api.routers.kb_iteration_routes as routes

    monkeypatch.setattr(
        routes,
        "_default_llm_review_client",
        lambda _rag: pytest.fail("LLM client must not be required"),
    )
    monkeypatch.setattr(routes, "load_proposals_by_id", lambda package_dir: {"p1": {}})

    async def fake_apply_accepted_changes_to_graph(
        *, rag, workspace, records, proposals_by_id
    ):
        return AcceptedApplyResult(
            workspace=workspace,
            applied_at="2026-06-19T00:00:00+00:00",
            source_artifact="accepted_changes.md",
            proposal_ids=["p1"],
            changes=[
                ApplyChange(
                    proposal_id="p1",
                    proposal_type="add_hierarchy_branch",
                    target="kg_structure.md",
                    status=ApplyChangeStatus.APPLIED,
                    action="add_hierarchy_branch",
                    branch_key="clinical_manifestation",
                )
            ],
        )

    monkeypatch.setattr(
        routes,
        "apply_accepted_changes_to_graph",
        fake_apply_accepted_changes_to_graph,
    )
    monkeypatch.setattr(
        routes, "run_iteration", lambda **kwargs: (_ for _ in ()).throw(ValueError("bad profile"))
    )

    response = client.post(
        "/kb-iteration/influenza_medical_v1/accepted-changes/execute",
        headers=HEADERS,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "bad profile"
    apply_result_json = fixture.package / "accepted_changes_apply_result.json"
    apply_result_markdown = fixture.package / "accepted_changes_apply_result.md"
    assert apply_result_json.exists()
    assert apply_result_markdown.exists()
    apply_payload = json.loads(apply_result_json.read_text(encoding="utf-8"))
    assert apply_payload["proposal_ids"] == ["p1"]
    assert apply_payload["applied_count"] == 1
    assert apply_payload["quality_before"]["overall"] == 82

    iteration_log = (fixture.package / "iteration_log.md").read_text(encoding="utf-8")
    assert "- phase: accepted_changes_apply" in iteration_log
    assert "- rerun_status: failed" in iteration_log
    assert "bad profile" in iteration_log


def test_execute_accepted_changes_rerun_failure_takes_precedence_over_memory_refresh(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)

    accept = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/accept",
        headers=HEADERS,
        json={},
    )
    assert accept.status_code == 200

    import lightrag.api.routers.kb_iteration_routes as routes

    monkeypatch.setattr(
        routes,
        "_default_llm_review_client",
        lambda _rag: pytest.fail("LLM client must not be required"),
    )
    monkeypatch.setattr(routes, "load_proposals_by_id", lambda package_dir: {"p1": {}})

    async def fake_apply_accepted_changes_to_graph(
        *, rag, workspace, records, proposals_by_id
    ):
        return AcceptedApplyResult(
            workspace=workspace,
            applied_at="2026-06-19T00:00:00+00:00",
            source_artifact="accepted_changes.md",
            proposal_ids=["p1"],
            changes=[
                ApplyChange(
                    proposal_id="p1",
                    proposal_type="add_hierarchy_branch",
                    target="kg_structure.md",
                    status=ApplyChangeStatus.APPLIED,
                    action="add_hierarchy_branch",
                    branch_key="clinical_manifestation",
                )
            ],
        )

    monkeypatch.setattr(
        routes,
        "apply_accepted_changes_to_graph",
        fake_apply_accepted_changes_to_graph,
    )
    monkeypatch.setattr(
        routes,
        "run_iteration",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad profile")),
    )
    monkeypatch.setattr(
        routes,
        "refresh_agent_memory_summary",
        lambda package_dir: (_ for _ in ()).throw(RuntimeError("refresh failed")),
    )

    response = client.post(
        "/kb-iteration/influenza_medical_v1/accepted-changes/execute",
        headers=HEADERS,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "bad profile"
    assert (fixture.package / "accepted_changes_apply_result.json").exists()

    iteration_log = (fixture.package / "iteration_log.md").read_text(encoding="utf-8")
    assert "- phase: accepted_changes_apply" in iteration_log
    assert "- rerun_status: failed" in iteration_log
    assert "bad profile" in iteration_log
    assert "refresh failed" not in response.json()["detail"]


def test_execute_accepted_changes_rejects_when_workspace_run_file_lock_is_held(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    with kb_iteration_routes._exclusive_workspace_file_lock(
        fixture.output_root, fixture.workspace
    ):
        response = client.post(
            "/kb-iteration/influenza_medical_v1/accepted-changes/execute",
            headers=HEADERS,
        )

    assert response.status_code == 409


def test_execute_accepted_changes_skips_iteration_when_nothing_applies(
    tmp_path: Path, monkeypatch
):
    client, _ = _client(tmp_path, monkeypatch)

    accept = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/accept",
        headers=HEADERS,
        json={},
    )
    assert accept.status_code == 200

    import lightrag.api.routers.kb_iteration_routes as routes

    monkeypatch.setattr(
        routes,
        "_default_llm_review_client",
        lambda _rag: pytest.fail("LLM client must not be required"),
    )
    monkeypatch.setattr(
        routes,
        "load_proposals_by_id",
        lambda package_dir: {"p1": {}},
        raising=False,
    )

    async def fake_apply_accepted_changes_to_graph(
        *, rag, workspace, records, proposals_by_id
    ):
        return AcceptedApplyResult(
            workspace=workspace,
            applied_at="2026-06-19T00:00:00+00:00",
            source_artifact="accepted_changes.md",
            proposal_ids=["p1"],
            changes=[
                ApplyChange(
                    proposal_id="p1",
                    proposal_type="add_hierarchy_branch",
                    target="kg_structure.md",
                    status=ApplyChangeStatus.BLOCKED,
                    action="add_hierarchy_branch",
                    reason="Graph storage is not available.",
                )
            ],
        )

    monkeypatch.setattr(
        routes,
        "apply_accepted_changes_to_graph",
        fake_apply_accepted_changes_to_graph,
        raising=False,
    )
    monkeypatch.setattr(
        routes,
        "run_iteration",
        lambda **kwargs: pytest.fail("Iteration should only rerun after applied changes"),
    )

    def fake_write_apply_result_artifacts(result, package_dir):
        target_dir = Path(package_dir)
        json_path = target_dir / "accepted_changes_apply_result.json"
        markdown_path = target_dir / "accepted_changes_apply_result.md"
        _write_json(json_path, result.to_dict())
        _write_text(markdown_path, "# Apply Result\n")
        return json_path, markdown_path

    monkeypatch.setattr(
        routes,
        "write_apply_result_artifacts",
        fake_write_apply_result_artifacts,
        raising=False,
    )

    response = client.post(
        "/kb-iteration/influenza_medical_v1/accepted-changes/execute",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "no_applicable_changes"
    assert payload["proposalIds"] == ["p1"]
    assert payload["appliedCount"] == 0
    assert payload["blockedCount"] == 1
    assert payload["artifactKey"] == "accepted_changes_apply_result"


def test_record_accepted_changes_execution_writes_legacy_agent_artifact(
    tmp_path: Path,
):
    fixture = _create_artifact_package(tmp_path)

    class FakeExecutionClient:
        def __init__(self) -> None:
            self.user_prompt = ""

        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            self.user_prompt = user_prompt
            return json.dumps(
                {
                    "summary": "Accepted change p2 was converted into an execution record.",
                    "executed_changes": [
                        {
                            "proposal_id": "p2",
                            "status": "recorded",
                            "action": "Prepare a bounded WebUI follow-up change.",
                        }
                    ],
                    "blocked_changes": [],
                    "next_steps": ["Run the KB iteration review again."],
                }
            )

    import lightrag.api.routers.kb_iteration_routes as routes

    args = SimpleNamespace(kb_iteration_output_dir=str(fixture.output_root))
    fake_client = FakeExecutionClient()
    result = routes._record_accepted_changes_execution(
        args,
        fixture.workspace,
        [{"proposal_id": "p2", "decision": "accept"}],
        fake_client,
    )

    assert result["workspace"] == "influenza_medical_v1"
    assert result["status"] == "execution_recorded"
    assert result["proposalIds"] == ["p2"]
    assert result["executedCount"] == 1
    assert result["artifactKey"] == "accepted_changes_execution"
    assert "p2" in fake_client.user_prompt

    execution = (fixture.package / "accepted_changes_execution.md").read_text(
        encoding="utf-8"
    )
    assert "# Accepted Changes Execution" in execution
    assert "p2" in execution
    assert "Prepare a bounded WebUI follow-up change." in execution

    iteration_log = (fixture.package / "iteration_log.md").read_text(encoding="utf-8")
    assert "- phase: accepted_changes_execution" in iteration_log
    assert "- accepted_proposal_ids: p2" in iteration_log


def test_proposal_decision_rejects_unknown_proposal(tmp_path: Path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/not-in-queue/accept",
        headers=HEADERS,
        json={"reviewer": "maintainer", "reason": "No matching proposal"},
    )

    assert response.status_code == 404


@pytest.mark.parametrize("proposal_id", ["bad%20id", "bad%0Aid", "bad:id", "bad%5Cid"])
def test_proposal_decision_rejects_invalid_proposal_id(
    tmp_path: Path, monkeypatch, proposal_id: str
):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post(
        f"/kb-iteration/influenza_medical_v1/proposals/{proposal_id}/accept",
        headers=HEADERS,
        json={"reviewer": "maintainer", "reason": "No unsafe identifiers"},
    )

    assert response.status_code == 400


def test_proposal_decision_allows_repeating_same_decision(tmp_path: Path, monkeypatch):
    client, fixture = _client(tmp_path, monkeypatch)

    first = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/accept",
        headers=HEADERS,
        json={"reviewer": "maintainer", "reason": "Evidence checked"},
    )
    second = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/accept",
        headers=HEADERS,
        json={"reviewer": "another-reviewer", "reason": "Clicked again"},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["record"] == first.json()["record"]

    accepted = (fixture.package / "accepted_changes.md").read_text(encoding="utf-8")
    assert accepted.count("\n## p1\n") == 1
    assert "Clicked again" not in accepted


def test_proposal_decision_rejects_conflicting_decision(tmp_path: Path, monkeypatch):
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


@pytest.mark.parametrize(
    ("artifact_key", "filename", "expected_content"),
    [
        ("llm_issue_analysis", "llm_issue_analysis.md", "# Issue Analysis\n"),
        (
            "llm_missing_branch_inference",
            "llm_missing_branch_inference.md",
            "# Missing Branch Inference\n",
        ),
        ("llm_evidence_map", "llm_evidence_map.md", "# Evidence Map\n"),
        ("llm_repair_plan", "llm_repair_plan.md", "# Repair Plan\n"),
    ],
)
def test_llm_agent_artifact_keys_read_whitelisted_markdown(
    tmp_path: Path,
    monkeypatch,
    artifact_key: str,
    filename: str,
    expected_content: str,
):
    client, fixture = _client(tmp_path, monkeypatch)
    _write_text(fixture.package / filename, expected_content)

    response = client.get(
        f"/kb-iteration/influenza_medical_v1/artifacts/{artifact_key}",
        headers=HEADERS,
    )
    run_response = client.get(
        f"/kb-iteration/influenza_medical_v1/runs/latest/artifacts/{artifact_key}",
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {
        "artifactKey": artifact_key,
        "contentType": "text/markdown",
        "content": expected_content,
    }
    assert run_response.status_code == 200
    assert run_response.json()["content"] == expected_content


@pytest.mark.parametrize(
    ("artifact_key", "filename", "content_type", "payload"),
    [
        (
            "proposal_task_packs",
            "proposal_task_packs.json",
            "application/json",
            [{"agent": "schema", "task": "repair_relation"}],
        ),
        (
            "proposal_merge_report",
            "proposal_merge_report.md",
            "text/markdown",
            "# Proposal Merge Report\n",
        ),
        (
            "subagent_output_index",
            "subagent_outputs/index.json",
            "application/json",
            {"outputs": [{"agent": "schema", "file": "schema.json"}]},
        ),
        (
            "issue_ledger",
            "issue_ledger.json",
            "application/json",
            {"issue_routes": [{"issue_ref": "issue-1"}]},
        ),
        (
            "deterministic_proposal_report",
            "deterministic_proposal_report.json",
            "application/json",
            {"families": [{"family": "diagnosis", "raw_issue_count": 3}]},
        ),
        (
            "deterministic_proposal_report_md",
            "deterministic_proposal_report.md",
            "text/markdown",
            "# Deterministic Proposal Report\n",
        ),
        (
            "proposal_funnel_report",
            "proposal_funnel_report.json",
            "application/json",
            {"families": {"diagnosis": {"raw_issue_count": 3}}},
        ),
        (
            "proposal_funnel_report_md",
            "proposal_funnel_report.md",
            "text/markdown",
            "# Proposal Funnel Report\n",
        ),
        (
            "proposal_conflict_groups",
            "proposal_conflict_groups.json",
            "application/json",
            [{"target": "edge-1", "proposal_ids": ["p1", "p2"]}],
        ),
    ],
)
def test_orchestrated_proposal_artifact_keys_read_from_generic_and_run_endpoints(
    tmp_path: Path,
    monkeypatch,
    artifact_key: str,
    filename: str,
    content_type: str,
    payload: object,
):
    client, fixture = _client(tmp_path, monkeypatch)
    path = fixture.package / filename
    if content_type == "application/json":
        _write_json(path, payload)
    else:
        _write_text(path, str(payload))

    response = client.get(
        f"/kb-iteration/influenza_medical_v1/artifacts/{artifact_key}",
        headers=HEADERS,
    )
    run_response = client.get(
        f"/kb-iteration/influenza_medical_v1/runs/latest/artifacts/{artifact_key}",
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert run_response.status_code == 200
    assert response.json()["artifactKey"] == artifact_key
    assert response.json()["contentType"] == content_type
    assert run_response.json()["artifactKey"] == artifact_key
    assert run_response.json()["contentType"] == content_type
    if content_type == "application/json":
        assert response.json()["payload"] == payload
        assert run_response.json()["payload"] == payload
    else:
        assert response.json()["content"] == payload
        assert run_response.json()["content"] == payload


def test_artifact_manifest_includes_orchestrated_proposal_display_metadata(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)
    _write_json(fixture.package / "proposal_task_packs.json", [])
    _write_text(fixture.package / "proposal_merge_report.md", "# Merge\n")
    _write_json(fixture.package / "subagent_outputs" / "index.json", {"outputs": []})
    _write_json(fixture.package / "issue_ledger.json", {"issue_routes": []})
    _write_json(
        fixture.package / "deterministic_proposal_report.json", {"families": []}
    )
    _write_text(
        fixture.package / "deterministic_proposal_report.md", "# Deterministic\n"
    )
    _write_json(fixture.package / "proposal_funnel_report.json", {"families": {}})
    _write_text(fixture.package / "proposal_funnel_report.md", "# Funnel\n")
    _write_json(fixture.package / "proposal_conflict_groups.json", [])

    response = client.get(
        "/kb-iteration/influenza_medical_v1/runs/latest/artifacts",
        headers=HEADERS,
    )

    assert response.status_code == 200
    artifacts = {item["key"]: item for item in response.json()["artifacts"]}
    assert artifacts["proposal_task_packs"] == {
        "key": "proposal_task_packs",
        "contentType": "application/json",
        "exists": True,
        "sourceFile": "proposal_task_packs.json",
        "display": {
            "language": "zh",
            "zhFile": "proposal_task_packs.zh.json",
            "zhExists": False,
        },
    }
    assert artifacts["proposal_merge_report"] == {
        "key": "proposal_merge_report",
        "contentType": "text/markdown",
        "exists": True,
        "sourceFile": "proposal_merge_report.md",
        "display": {
            "language": "zh",
            "zhFile": "proposal_merge_report.zh.md",
            "zhExists": False,
        },
    }
    assert artifacts["subagent_output_index"] == {
        "key": "subagent_output_index",
        "contentType": "application/json",
        "exists": True,
        "sourceFile": "subagent_outputs/index.json",
        "display": {
            "language": "zh",
            "zhFile": "subagent_outputs/index.zh.json",
            "zhExists": False,
        },
    }
    assert artifacts["issue_ledger"] == {
        "key": "issue_ledger",
        "contentType": "application/json",
        "exists": True,
        "sourceFile": "issue_ledger.json",
        "display": {
            "language": "zh",
            "zhFile": "issue_ledger.zh.json",
            "zhExists": False,
        },
    }
    assert artifacts["deterministic_proposal_report"] == {
        "key": "deterministic_proposal_report",
        "contentType": "application/json",
        "exists": True,
        "sourceFile": "deterministic_proposal_report.json",
        "display": {
            "language": "zh",
            "zhFile": "deterministic_proposal_report.zh.json",
            "zhExists": False,
        },
    }
    assert artifacts["deterministic_proposal_report_md"] == {
        "key": "deterministic_proposal_report_md",
        "contentType": "text/markdown",
        "exists": True,
        "sourceFile": "deterministic_proposal_report.md",
        "display": {
            "language": "zh",
            "zhFile": "deterministic_proposal_report.zh.md",
            "zhExists": False,
        },
    }
    assert artifacts["proposal_funnel_report"] == {
        "key": "proposal_funnel_report",
        "contentType": "application/json",
        "exists": True,
        "sourceFile": "proposal_funnel_report.json",
        "display": {
            "language": "zh",
            "zhFile": "proposal_funnel_report.zh.json",
            "zhExists": False,
        },
    }
    assert artifacts["proposal_funnel_report_md"] == {
        "key": "proposal_funnel_report_md",
        "contentType": "text/markdown",
        "exists": True,
        "sourceFile": "proposal_funnel_report.md",
        "display": {
            "language": "zh",
            "zhFile": "proposal_funnel_report.zh.md",
            "zhExists": False,
        },
    }
    assert artifacts["proposal_conflict_groups"] == {
        "key": "proposal_conflict_groups",
        "contentType": "application/json",
        "exists": True,
        "sourceFile": "proposal_conflict_groups.json",
        "display": {
            "language": "zh",
            "zhFile": "proposal_conflict_groups.zh.json",
            "zhExists": False,
        },
    }


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
    assert (
        valid_path
        == (fixture.package / "patch_candidates" / "proposal-1.patch").resolve()
    )

    with pytest.raises(HTTPException) as exc_info:
        kb_iteration_routes._safe_patch_candidate_path(
            args, fixture.workspace, "../outside"
        )

    assert exc_info.value.status_code == 400


def test_llm_review_run_default_unavailable_client_returns_503(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("KB_ITERATION_LLM_BINDING", raising=False)
    monkeypatch.delenv("KB_ITERATION_LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("KB_ITERATION_LLM_BINDING_API_KEY", raising=False)
    monkeypatch.delenv("KB_ITERATION_LLM_MODEL", raising=False)
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/llm-review/runs",
        headers=HEADERS,
        json={"profile": "clinical_guideline_zh"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "LLM review client is not configured"


def test_default_llm_review_client_uses_kb_iteration_openai_compatible_env(
    monkeypatch,
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("KB_ITERATION_LLM_BINDING", "openai")
    monkeypatch.setenv("KB_ITERATION_LLM_BINDING_HOST", "https://api.deepseek.com")
    monkeypatch.setenv("KB_ITERATION_LLM_BINDING_API_KEY", "test-secret")
    monkeypatch.setenv("KB_ITERATION_LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("KB_ITERATION_LLM_TIMEOUT", "45")

    from lightrag.api.routers import kb_iteration_routes

    calls = []

    async def fake_openai_complete_if_cache(
        model: str,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> str:
        calls.append(
            {
                "model": model,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "history_messages": history_messages,
                "base_url": kwargs.get("base_url"),
                "api_key": kwargs.get("api_key"),
                "response_format": kwargs.get("response_format"),
                "timeout": kwargs.get("timeout"),
            }
        )
        return '{"proposals":[]}'

    monkeypatch.setattr(
        "lightrag.llm.openai.openai_complete_if_cache",
        fake_openai_complete_if_cache,
    )

    client = kb_iteration_routes._default_llm_review_client(SimpleNamespace())

    assert not isinstance(client, kb_iteration_routes._UnavailableLLMReviewClient)
    assert client.model == "deepseek-v4-flash"
    assert client.supports_parallel_subagent_calls is True
    assert client.complete(system_prompt="system", user_prompt="user") == (
        '{"proposals":[]}'
    )
    assert calls == [
        {
            "model": "deepseek-v4-flash",
            "prompt": "user",
            "system_prompt": "system",
            "history_messages": [],
            "base_url": "https://api.deepseek.com",
            "api_key": "test-secret",
            "response_format": {"type": "json_object"},
            "timeout": 45,
        }
    ]


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


def test_llm_review_run_defaults_to_agent_pipeline(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    agent_calls = []
    loop_calls = []

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

    def fake_run_llm_agent_pipeline(**kwargs):
        agent_calls.append(kwargs)
        return SimpleNamespace(
            output_dir=fixture.package,
            stop_reason="agent_done",
            proposal_ids=["agent-proposal"],
            artifact_paths={},
        )

    monkeypatch.setattr(
        kb_iteration_routes,
        "run_llm_agent_pipeline",
        fake_run_llm_agent_pipeline,
    )
    monkeypatch.setattr(
        kb_iteration_routes,
        "run_llm_review_loop",
        lambda **kwargs: loop_calls.append(kwargs),
    )
    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda rag: FakeClient(),
    )

    response = client.post(
        "/kb-iteration/influenza_medical_v1/llm-review/runs",
        headers=HEADERS,
        json={
            "profile": "clinical_guideline_zh",
            "max_context_tokens_per_round": 9000,
            "max_stage_retries": 2,
            "max_subagent_tasks": 3,
            "max_parallel_subagents": 2,
            "max_subagent_issues_per_task": 4,
            "max_subagent_proposals_per_task": 2,
            "max_proposals_per_run": 9,
            "deterministic_family_caps": {"diagnosis": 7, "entity_cleanup": 3},
            "strict_subagent_role_contracts": False,
            "prevalidate_action_candidates": False,
            "require_candidate_evidence_allowlist": False,
            "skip_deterministic_subagent_calls": False,
            "allow_llm_judge": False,
            "generate_patch_candidates": True,
            "require_human_for_mutation": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["stopReason"] == "agent_done"
    assert response.json()["proposalIds"] == ["agent-proposal"]
    assert len(agent_calls) == 1
    assert loop_calls == []
    assert agent_calls[0]["workspace"] == "influenza_medical_v1"
    assert agent_calls[0]["package_dir"] == fixture.package
    assert agent_calls[0]["profile"] == "clinical_guideline_zh"
    config = agent_calls[0]["config"]
    assert config.max_context_tokens_per_stage == 9000
    assert config.max_stage_retries == 2
    assert config.max_subagent_tasks == 3
    assert config.max_parallel_subagents == 2
    assert config.max_subagent_issues_per_task == 4
    assert config.max_subagent_proposals_per_task == 2
    assert config.max_proposals_per_run == 9
    assert config.deterministic_family_caps == {"diagnosis": 7, "entity_cleanup": 3}
    assert config.strict_subagent_role_contracts is False
    assert config.prevalidate_action_candidates is False
    assert config.require_candidate_evidence_allowlist is False
    assert config.skip_deterministic_subagent_calls is False
    assert config.allow_llm_judge is False
    assert config.generate_patch_candidates is True
    assert config.require_human_for_mutation is False


@pytest.mark.parametrize(
    "request_body",
    [
        {"max_subagent_tasks": 0},
        {"max_subagent_tasks": 51},
        {"max_parallel_subagents": 0},
        {"max_parallel_subagents": 9},
        {"max_subagent_issues_per_task": 0},
        {"max_subagent_issues_per_task": 21},
        {"max_subagent_proposals_per_task": 0},
        {"max_subagent_proposals_per_task": 11},
        {"max_proposals_per_run": 0},
        {"max_proposals_per_run": 201},
        {"deterministic_family_caps": {"diagnosis": 0}},
        {"deterministic_family_caps": {"diagnosis": 501}},
    ],
)
def test_llm_review_run_rejects_out_of_bounds_subagent_controls(
    tmp_path: Path, monkeypatch, request_body: dict[str, int]
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, _ = _client(tmp_path, monkeypatch)
    agent_calls = []

    class FakeClient:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return "{}"

    def fake_run_llm_agent_pipeline(**kwargs):
        agent_calls.append(kwargs)
        return SimpleNamespace(
            output_dir=Path(),
            stop_reason="agent_done",
            proposal_ids=[],
            artifact_paths={},
        )

    monkeypatch.setattr(
        kb_iteration_routes,
        "run_llm_agent_pipeline",
        fake_run_llm_agent_pipeline,
    )
    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda rag: FakeClient(),
    )

    response = client.post(
        "/kb-iteration/influenza_medical_v1/llm-review/runs",
        headers=HEADERS,
        json=request_body,
    )

    assert response.status_code == 422
    assert agent_calls == []


def test_llm_review_run_defaults_to_one_agent_stage_retry(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    agent_calls = []

    class FakeClient:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return "{}"

    def fake_run_llm_agent_pipeline(**kwargs):
        agent_calls.append(kwargs)
        return SimpleNamespace(
            output_dir=fixture.package,
            stop_reason="agent_done",
            proposal_ids=[],
            artifact_paths={},
        )

    monkeypatch.setattr(
        kb_iteration_routes,
        "run_llm_agent_pipeline",
        fake_run_llm_agent_pipeline,
    )
    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda rag: FakeClient(),
    )

    response = client.post(
        "/kb-iteration/influenza_medical_v1/llm-review/runs",
        headers=HEADERS,
        json={"profile": "clinical_guideline_zh"},
    )

    assert response.status_code == 200
    assert len(agent_calls) == 1
    assert agent_calls[0]["config"].max_stage_retries == 1


def test_llm_review_run_defaults_to_bulk_proposal_limit(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    agent_calls = []

    class FakeClient:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return "{}"

    def fake_run_llm_agent_pipeline(**kwargs):
        agent_calls.append(kwargs)
        return SimpleNamespace(
            output_dir=fixture.package,
            stop_reason="agent_done",
            proposal_ids=[],
            artifact_paths={},
        )

    monkeypatch.setattr(
        kb_iteration_routes,
        "run_llm_agent_pipeline",
        fake_run_llm_agent_pipeline,
    )
    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda rag: FakeClient(),
    )

    response = client.post(
        "/kb-iteration/influenza_medical_v1/llm-review/runs",
        headers=HEADERS,
        json={"profile": "clinical_guideline_zh"},
    )

    assert response.status_code == 200
    assert len(agent_calls) == 1
    assert agent_calls[0]["config"].max_parallel_subagents == 4
    assert agent_calls[0]["config"].max_subagent_issues_per_task == 4
    assert agent_calls[0]["config"].max_subagent_proposals_per_task == 2
    assert agent_calls[0]["config"].max_proposals_per_run == 200
    assert agent_calls[0]["config"].deterministic_family_caps is None
    assert agent_calls[0]["config"].strict_subagent_role_contracts is True
    assert agent_calls[0]["config"].prevalidate_action_candidates is True
    assert agent_calls[0]["config"].require_candidate_evidence_allowlist is True
    assert agent_calls[0]["config"].skip_deterministic_subagent_calls is True


def test_llm_review_run_accepts_agent_stage_retries_up_to_eight(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    agent_calls = []

    class FakeClient:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return "{}"

    def fake_run_llm_agent_pipeline(**kwargs):
        agent_calls.append(kwargs)
        return SimpleNamespace(
            output_dir=fixture.package,
            stop_reason="agent_done",
            proposal_ids=[],
            artifact_paths={},
        )

    monkeypatch.setattr(
        kb_iteration_routes,
        "run_llm_agent_pipeline",
        fake_run_llm_agent_pipeline,
    )
    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda rag: FakeClient(),
    )

    response = client.post(
        "/kb-iteration/influenza_medical_v1/llm-review/runs",
        headers=HEADERS,
        json={"max_stage_retries": 8},
    )

    assert response.status_code == 200
    assert agent_calls[0]["config"].max_stage_retries == 8


def test_llm_review_run_loop_mode_uses_legacy_loop(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    agent_calls = []
    loop_calls = []

    class FakeClient:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return "{}"

    def fake_run_llm_review_loop(**kwargs):
        loop_calls.append(kwargs)
        return SimpleNamespace(
            output_dir=fixture.package,
            stop_reason="all_proposals_invalid",
            proposal_ids=[],
            artifact_paths={},
        )

    monkeypatch.setattr(
        kb_iteration_routes,
        "run_llm_agent_pipeline",
        lambda **kwargs: agent_calls.append(kwargs),
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
        json={
            "mode": "loop",
            "profile": "clinical_guideline_zh",
            "max_review_rounds": 1,
            "max_focus_items_per_round": 2,
            "max_context_tokens_per_round": 9000,
            "allow_llm_judge": False,
            "allow_llm_auto_accept": True,
            "allow_low_risk_auto_reject": False,
            "generate_patch_candidates": True,
            "require_human_for_mutation": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["stopReason"] == "all_proposals_invalid"
    assert agent_calls == []
    assert len(loop_calls) == 1
    assert loop_calls[0]["workspace"] == "influenza_medical_v1"
    assert loop_calls[0]["package_dir"] == fixture.package
    assert loop_calls[0]["profile"] == "clinical_guideline_zh"
    config = loop_calls[0]["config"]
    assert config.max_review_rounds == 1
    assert config.max_focus_items_per_round == 2
    assert config.max_context_tokens_per_round == 9000
    assert config.allow_llm_judge is False
    assert config.allow_llm_auto_accept is True
    assert config.allow_low_risk_auto_reject is False
    assert config.generate_patch_candidates is True
    assert config.require_human_for_mutation is False
