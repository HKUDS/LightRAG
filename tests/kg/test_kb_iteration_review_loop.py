import json
from pathlib import Path

import pytest

from lightrag.kb_iteration.review_loop import (
    LLMReviewLoopConfig,
    run_llm_review_loop,
)


class StaticReviewClient:
    model = "configured-review-model"

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "confirmed_issues": [
                    {
                        "message": "Generic relation labels should be mapped.",
                        "evidence": ["edge:e1"],
                    }
                ],
                "hypotheses": [],
                "missing_evidence": [],
                "out_of_scope": [],
                "proposals": [
                    {
                        "id": "proposal-20260618-001",
                        "type": "relation_keyword_mapping",
                        "target": "lightrag/medical_kg/ontology.py",
                        "proposed_change": (
                            "Map generic relation to controlled relation keywords."
                        ),
                        "reason": "Generic relations reduce readability.",
                        "evidence": ["edge:e1"],
                        "confidence": 0.8,
                        "risk": "medium",
                        "requires_approval": True,
                        "expected_metric_change": {"relation_semantics": 8},
                    }
                ],
            }
        )


class NoProposalReviewClient:
    model = "configured-review-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return json.dumps(
            {
                "confirmed_issues": [],
                "hypotheses": [],
                "missing_evidence": [],
                "out_of_scope": [],
                "proposals": [],
            }
        )


class FailingReviewClient:
    model = "configured-review-model"

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("boom")


class InvalidReviewClient:
    model = "configured-review-model"

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return "not json"


def test_run_llm_review_loop_writes_artifacts_and_stops_for_human_review(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_review_package(package)

    result = run_llm_review_loop(
        workspace="demo",
        package_dir=package,
        client=StaticReviewClient(),
        config=LLMReviewLoopConfig(max_review_rounds=1),
    )

    assert result.stop_reason == "pending_human_review"
    assert (package / "llm_review_trace.json").exists()
    assert (package / "llm_review_report.md").exists()
    assert (package / "proposals.generated.yaml").exists()
    assert "proposal-20260618-001" in (
        package / "approval_queue.md"
    ).read_text(encoding="utf-8")

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    round_trace = trace["rounds"][0]
    assert round_trace["focus"] == ["generic_relation"]
    assert round_trace["proposal_ids"] == ["proposal-20260618-001"]
    assert round_trace["state"] == "queued"
    assert round_trace["context_files"] == ["review_context/round-001-context.json"]
    assert round_trace["model"] == "configured-review-model"
    assert isinstance(round_trace["input_token_estimate"], int)
    assert isinstance(round_trace["output_token_estimate"], int)
    assert round_trace["judge_decision"] == "needs_human"


def test_run_llm_review_loop_writes_trace_when_client_fails(tmp_path: Path):
    package = tmp_path / "package"
    _write_review_package(package)
    _write_stale_llm_artifacts(package)

    result = run_llm_review_loop(
        workspace="demo",
        package_dir=package,
        client=FailingReviewClient(),
        config=LLMReviewLoopConfig(max_review_rounds=1),
    )

    trace_path = package / "llm_review_trace.json"
    assert result.stop_reason == "llm_client_error"
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    round_trace = trace["rounds"][0]
    assert "boom" in round_trace["error"]
    assert round_trace["context_files"] == ["review_context/round-001-context.json"]

    report_text = (package / "llm_review_report.md").read_text(encoding="utf-8")
    proposals_text = (package / "proposals.generated.yaml").read_text(encoding="utf-8")
    assert "stale successful review" not in report_text
    assert "stale-success-proposal" not in proposals_text
    assert "llm_client_error" in report_text
    assert "boom" in report_text
    assert "proposals: []" in proposals_text


def test_run_llm_review_loop_marks_trace_when_llm_output_is_invalid(tmp_path: Path):
    package = tmp_path / "package"
    _write_review_package(package)
    _write_stale_llm_artifacts(package)

    result = run_llm_review_loop(
        workspace="demo",
        package_dir=package,
        client=InvalidReviewClient(),
        config=LLMReviewLoopConfig(max_review_rounds=1),
    )

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    round_trace = trace["rounds"][0]
    assert result.stop_reason == "invalid_llm_output"
    assert round_trace["state"] == "invalid_llm_output"
    assert round_trace["completed_at"]
    assert round_trace["output_token_estimate"] > 0
    assert "valid JSON" in round_trace["error"]

    report_text = (package / "llm_review_report.md").read_text(encoding="utf-8")
    proposals_text = (package / "proposals.generated.yaml").read_text(encoding="utf-8")
    assert "stale successful review" not in report_text
    assert "stale-success-proposal" not in proposals_text
    assert "invalid_llm_output" in report_text
    assert "valid JSON" in report_text
    assert "proposals: []" in proposals_text


def test_run_llm_review_loop_stops_when_no_proposals_are_generated(tmp_path: Path):
    package = tmp_path / "package"
    _write_review_package(package)
    client = NoProposalReviewClient()

    result = run_llm_review_loop(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMReviewLoopConfig(max_review_rounds=2),
    )

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    assert result.stop_reason == "all_proposals_invalid"
    assert client.calls == 2
    assert len(trace["rounds"]) == 2
    assert [round_trace["state"] for round_trace in trace["rounds"]] == [
        "no_valid_proposals",
        "no_valid_proposals",
    ]
    assert [round_trace["proposal_ids"] for round_trace in trace["rounds"]] == [[], []]
    assert [round_trace["judge_decision"] for round_trace in trace["rounds"]] == ["", ""]


def test_run_llm_review_loop_stops_before_client_when_context_exceeds_budget(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_review_package(package)
    client = NoProposalReviewClient()

    result = run_llm_review_loop(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMReviewLoopConfig(
            max_review_rounds=2,
            max_context_tokens_per_round=1,
        ),
    )

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    round_trace = trace["rounds"][0]
    assert result.stop_reason == "context_too_large"
    assert client.calls == 0
    assert len(trace["rounds"]) == 1
    assert round_trace["state"] == "context_too_large"
    assert round_trace["input_token_estimate"] > 1
    assert "max_context_tokens_per_round" in round_trace["error"]


def test_run_llm_review_loop_selects_multiple_focus_items_in_priority_order(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_review_package(
        package,
        findings=[
            {
                "severity": "medium",
                "category": "evidence_grounding",
                "message": "Source evidence is missing.",
                "evidence": ["edge:e1"],
                "suggested_fix_type": "restore_evidence",
                "requires_approval": True,
            },
            {
                "severity": "high",
                "category": "hierarchy_completeness",
                "message": "Hierarchy is missing expected branches.",
                "evidence": ["node:flu"],
                "suggested_fix_type": "add_hierarchy_branch",
                "requires_approval": True,
            },
            {
                "severity": "high",
                "category": "relation_semantics",
                "message": "Generic relation keywords should be reviewed.",
                "evidence": ["edge:e1"],
                "suggested_fix_type": "replace_relation_keyword",
                "requires_approval": True,
            },
        ],
    )

    run_llm_review_loop(
        workspace="demo",
        package_dir=package,
        client=StaticReviewClient(),
        config=LLMReviewLoopConfig(max_review_rounds=1, max_focus_items_per_round=2),
    )

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    assert trace["rounds"][0]["focus"] == [
        "generic_relation",
        "hierarchy_missing_branch",
    ]


def test_run_llm_review_loop_evidence_focus_uses_evidence_linked_context(
    tmp_path: Path,
):
    package = tmp_path / "package"
    _write_review_package(
        package,
        findings=[
            {
                "severity": "high",
                "category": "evidence_grounding",
                "message": "Source evidence is missing from one snapshot item.",
                "evidence": ["edge:e2 missing source_id"],
                "suggested_fix_type": "restore_evidence",
                "requires_approval": True,
            }
        ],
        edges=[
            {
                "id": "e1",
                "source": "flu",
                "target": "fever",
                "keywords": "clinical_manifestation",
                "source_id": "chunk-1",
                "file_path": "guide.md",
            },
            {
                "id": "e2",
                "source": "flu",
                "target": "cough",
                "keywords": "clinical_manifestation",
            },
        ],
    )

    run_llm_review_loop(
        workspace="demo",
        package_dir=package,
        client=StaticReviewClient(),
        config=LLMReviewLoopConfig(max_review_rounds=1),
    )

    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    context_path = package / trace["rounds"][0]["context_files"][0]
    context = json.loads(context_path.read_text(encoding="utf-8"))
    assert trace["rounds"][0]["focus"] == ["missing_evidence"]
    assert [finding["category"] for finding in context["quality_findings"]] == [
        "evidence_grounding"
    ]
    assert [relation["id"] for relation in context["relations"]] == ["e2"]


def test_run_llm_review_loop_rejects_missing_deterministic_artifacts(
    tmp_path: Path,
):
    package = tmp_path / "package"
    client = NoProposalReviewClient()

    with pytest.raises(ValueError, match="kg_snapshot.json"):
        run_llm_review_loop(
            workspace="demo",
            package_dir=package,
            client=client,
            config=LLMReviewLoopConfig(max_review_rounds=1),
        )

    assert client.calls == 0
    assert not (package / "llm_review_trace.json").exists()
    assert not (package / "review_context").exists()


@pytest.mark.parametrize(
    "config",
    [
        LLMReviewLoopConfig(max_review_rounds=0),
        LLMReviewLoopConfig(max_focus_items_per_round=0),
        LLMReviewLoopConfig(max_context_tokens_per_round=0),
    ],
)
def test_run_llm_review_loop_rejects_invalid_config_before_writing_artifacts(
    tmp_path: Path, config: LLMReviewLoopConfig
):
    package = tmp_path / "package"

    with pytest.raises(ValueError):
        run_llm_review_loop(
            workspace="demo",
            package_dir=package,
            client=StaticReviewClient(),
            config=config,
        )

    assert not (package / "llm_review_trace.json").exists()
    assert not (package / "review_context").exists()


def _write_review_package(
    package: Path,
    findings: list[dict] | None = None,
    edges: list[dict] | None = None,
) -> None:
    snapshot_dir = package / "snapshots"
    snapshot_dir.mkdir(parents=True)
    _write_json(
        snapshot_dir / "kg_snapshot.json",
        {
            "workspace": "demo",
            "generated_at": "2026-06-18T00:00:00Z",
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
            "edges": edges
            or [
                {
                    "id": "e1",
                    "source": "flu",
                    "target": "fever",
                    "keywords": "相关",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                }
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
            "findings": findings
            or [
                {
                    "severity": "high",
                    "category": "relation_semantics",
                    "message": "Generic relation keywords should be reviewed.",
                    "evidence": ["edge:e1"],
                    "suggested_fix_type": "replace_relation_keyword",
                    "requires_approval": True,
                }
            ],
            "critical_blockers": [],
        },
    )
    (package / "accepted_changes.md").write_text(
        "# Accepted Changes\n\n- Keep specific symptom edges.\n",
        encoding="utf-8",
    )
    (package / "rejected_changes.md").write_text(
        "# Rejected Changes\n\n- Do not merge flu and fever.\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_stale_llm_artifacts(package: Path) -> None:
    (package / "llm_review_report.md").write_text(
        "# stale successful review\n\n- stale-success-proposal\n",
        encoding="utf-8",
    )
    (package / "proposals.generated.yaml").write_text(
        "# Generated Proposals\n\n"
        "proposals:\n"
        "- id: stale-success-proposal\n"
        "  requires_approval: true\n",
        encoding="utf-8",
    )
