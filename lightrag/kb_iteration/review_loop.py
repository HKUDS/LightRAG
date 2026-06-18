from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lightrag.utils import validate_workspace

from .llm_review import LLMReviewClient, parse_llm_review_output, write_llm_review_artifacts
from .proposals import validate_proposal, write_approval_queue, write_improvement_backlog
from .review_context import build_review_context, write_review_context


@dataclass(frozen=True)
class LLMReviewLoopConfig:
    max_review_rounds: int = 4
    max_focus_items_per_round: int = 3
    max_context_tokens_per_round: int = 12000
    allow_llm_judge: bool = True
    allow_llm_auto_accept: bool = False
    allow_low_risk_auto_reject: bool = True
    generate_patch_candidates: bool = False
    require_human_for_mutation: bool = True


@dataclass(frozen=True)
class LLMReviewRunResult:
    output_dir: Path
    stop_reason: str
    proposal_ids: list[str] = field(default_factory=list)
    artifact_paths: dict[str, Path] = field(default_factory=dict)


_FAILURE_STOP_REASONS = {
    "context_too_large",
    "invalid_llm_output",
    "llm_client_error",
}


def run_llm_review_loop(
    *,
    workspace: str,
    package_dir: str | Path,
    client: LLMReviewClient,
    config: LLMReviewLoopConfig | None = None,
    profile: str | None = None,
) -> LLMReviewRunResult:
    validated_workspace = validate_workspace(workspace)
    loop_config = config or LLMReviewLoopConfig()
    _validate_config(loop_config)
    output_dir = Path(package_dir)
    _validate_required_artifacts(output_dir, validated_workspace)

    trace: dict[str, Any] = {
        "workspace": validated_workspace,
        "started_at": _utc_timestamp(),
        "completed_at": "",
        "stop_reason": "",
        "profile": profile,
        "config": asdict(loop_config),
        "rounds": [],
    }
    artifact_paths: dict[str, Path] = {}

    for round_index in range(1, loop_config.max_review_rounds + 1):
        round_id = f"round-{round_index:03d}"
        focus = _select_focus(output_dir)[: loop_config.max_focus_items_per_round]
        context = build_review_context(output_dir, round_id=round_id, focus=focus)
        context_path = write_review_context(output_dir, round_id, context)
        artifact_paths[f"{round_id}_context"] = context_path
        user_prompt = json.dumps(context, ensure_ascii=False, sort_keys=True)
        input_token_estimate = _estimate_tokens(user_prompt)

        round_trace: dict[str, Any] = {
            "round_id": round_id,
            "started_at": _utc_timestamp(),
            "focus": focus,
            "state": "running",
            "context_files": [_relative_artifact_path(output_dir, context_path)],
            "model": _client_model(client),
            "input_token_estimate": input_token_estimate,
            "output_token_estimate": 0,
            "judge_decision": "",
            "proposal_ids": [],
            "error": "",
        }

        if input_token_estimate > loop_config.max_context_tokens_per_round:
            round_trace["state"] = "context_too_large"
            round_trace["error"] = (
                "input_token_estimate exceeds max_context_tokens_per_round"
            )
            round_trace["completed_at"] = _utc_timestamp()
            trace["rounds"].append(round_trace)
            return _finish_run(
                output_dir=output_dir,
                stop_reason="context_too_large",
                proposal_ids=[],
                artifact_paths=artifact_paths,
                trace=trace,
            )

        try:
            raw_output = client.complete(
                system_prompt=_system_prompt(loop_config, profile),
                user_prompt=user_prompt,
            )
        except Exception as exc:
            round_trace["state"] = "client_error"
            round_trace["error"] = str(exc)
            round_trace["completed_at"] = _utc_timestamp()
            trace["rounds"].append(round_trace)
            return _finish_run(
                output_dir=output_dir,
                stop_reason="llm_client_error",
                proposal_ids=[],
                artifact_paths=artifact_paths,
                trace=trace,
            )

        try:
            review_output = parse_llm_review_output(raw_output)
            for proposal in review_output.proposals:
                validate_proposal(proposal)
        except ValueError as exc:
            round_trace["state"] = "invalid_llm_output"
            round_trace["error"] = str(exc)
            round_trace["output_token_estimate"] = _estimate_tokens(raw_output)
            round_trace["completed_at"] = _utc_timestamp()
            trace["rounds"].append(round_trace)
            return _finish_run(
                output_dir=output_dir,
                stop_reason="invalid_llm_output",
                proposal_ids=[],
                artifact_paths=artifact_paths,
                trace=trace,
            )

        artifact_paths.update(write_llm_review_artifacts(review_output, output_dir))
        approval_queue_path = write_approval_queue(review_output.proposals, output_dir)
        backlog_path = write_improvement_backlog(review_output.proposals, output_dir)
        artifact_paths["approval_queue"] = approval_queue_path
        artifact_paths["improvement_backlog"] = backlog_path

        proposal_ids = [proposal.id for proposal in review_output.proposals]
        requires_human_review = any(
            proposal.requires_approval for proposal in review_output.proposals
        )
        round_trace["proposal_ids"] = proposal_ids
        round_trace["output_token_estimate"] = _estimate_tokens(raw_output)

        if not proposal_ids:
            round_trace["state"] = "no_valid_proposals"
            round_trace["judge_decision"] = ""
        elif requires_human_review:
            round_trace["state"] = "queued"
            round_trace["judge_decision"] = "needs_human"
        else:
            round_trace["state"] = "reviewed"
            round_trace["judge_decision"] = "not_required"

        round_trace["completed_at"] = _utc_timestamp()
        trace["rounds"].append(round_trace)

        if not proposal_ids:
            continue

        if requires_human_review:
            return _finish_run(
                output_dir=output_dir,
                stop_reason="pending_human_review",
                proposal_ids=proposal_ids,
                artifact_paths=artifact_paths,
                trace=trace,
            )

        return _finish_run(
            output_dir=output_dir,
            stop_reason="review_complete",
            proposal_ids=proposal_ids,
            artifact_paths=artifact_paths,
            trace=trace,
        )

    return _finish_run(
        output_dir=output_dir,
        stop_reason="all_proposals_invalid",
        proposal_ids=[],
        artifact_paths=artifact_paths,
        trace=trace,
    )


def _select_focus(output_dir: str | Path) -> list[str]:
    quality_path = Path(output_dir) / "snapshots" / "quality_score.json"
    quality = _read_json(quality_path, default={})
    findings = quality.get("findings", [])
    normalized_findings = [
        _finding_text(finding).casefold() for finding in findings
    ]

    focus = []
    if any("generic" in text or "relation" in text for text in normalized_findings):
        focus.append("generic_relation")
    if any("hierarchy" in text for text in normalized_findings):
        focus.append("hierarchy_missing_branch")
    if any(
        "evidence" in text or "source" in text for text in normalized_findings
    ):
        focus.append("missing_evidence")
    return focus or ["generic_relation"]


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_config(config: LLMReviewLoopConfig) -> None:
    if config.max_review_rounds <= 0:
        raise ValueError("max_review_rounds must be greater than 0")
    if config.max_focus_items_per_round <= 0:
        raise ValueError("max_focus_items_per_round must be greater than 0")
    if config.max_context_tokens_per_round <= 0:
        raise ValueError("max_context_tokens_per_round must be greater than 0")


def _validate_required_artifacts(output_dir: Path, workspace: str) -> None:
    snapshot_path = output_dir / "snapshots" / "kg_snapshot.json"
    quality_path = output_dir / "snapshots" / "quality_score.json"
    for path in (snapshot_path, quality_path):
        if not path.exists():
            raise ValueError(
                "KB iteration package is missing required artifact: "
                f"{path.relative_to(output_dir).as_posix()}"
            )

    snapshot = _read_json(snapshot_path, default={})
    quality = _read_json(quality_path, default={})
    if not isinstance(snapshot, dict):
        raise ValueError("kg_snapshot.json must contain a JSON object")
    if not isinstance(quality, dict):
        raise ValueError("quality_score.json must contain a JSON object")

    snapshot_workspace = snapshot.get("workspace")
    if snapshot_workspace and str(snapshot_workspace) != workspace:
        raise ValueError(
            "kg_snapshot.json workspace does not match requested workspace"
        )


def _client_model(client: LLMReviewClient) -> str:
    model = getattr(client, "model", "")
    if isinstance(model, str) and model.strip():
        return model
    return "unknown"


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _relative_artifact_path(output_dir: Path, artifact_path: Path) -> str:
    return artifact_path.relative_to(output_dir).as_posix()


def _finish_run(
    *,
    output_dir: Path,
    stop_reason: str,
    proposal_ids: list[str],
    artifact_paths: dict[str, Path],
    trace: dict[str, Any],
) -> LLMReviewRunResult:
    trace["completed_at"] = _utc_timestamp()
    trace["stop_reason"] = stop_reason
    if stop_reason in _FAILURE_STOP_REASONS:
        artifact_paths.update(_write_failure_artifacts(output_dir, stop_reason, trace))
    trace_path = output_dir / "llm_review_trace.json"
    with trace_path.open("w", encoding="utf-8") as file:
        json.dump(trace, file, ensure_ascii=False, indent=2)
        file.write("\n")
    artifact_paths["llm_review_trace"] = trace_path
    return LLMReviewRunResult(
        output_dir=output_dir,
        stop_reason=stop_reason,
        proposal_ids=proposal_ids,
        artifact_paths=artifact_paths,
    )


def _write_failure_artifacts(
    output_dir: Path, stop_reason: str, trace: dict[str, Any]
) -> dict[str, Path]:
    report_path = output_dir / "llm_review_report.md"
    proposals_path = output_dir / "proposals.generated.yaml"
    latest_round = _latest_round_trace(trace)
    error = str(latest_round.get("error", "")).strip()
    state = str(latest_round.get("state", "")).strip()
    round_id = str(latest_round.get("round_id", "")).strip()

    report_lines = [
        "# LLM Review Report",
        "",
        "## Summary",
        "",
        f"- Stop reason: {stop_reason}",
        "- Generated proposals: 0",
        "",
        "## Failure",
        "",
    ]
    if round_id:
        report_lines.append(f"- Round: {round_id}")
    if state:
        report_lines.append(f"- State: {state}")
    if error:
        report_lines.append(f"- Error: {_single_line(error)}")
    if not any(line.startswith("- ") for line in report_lines[-3:]):
        report_lines.append("- Error: unknown")
    report_lines.extend(
        [
            "",
            "## Generated Proposals",
            "",
            "- none",
            "",
        ]
    )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    proposals_path.write_text(
        "# Generated Proposals\n\nproposals: []\n",
        encoding="utf-8",
    )
    return {
        "llm_review_report": report_path,
        "proposals_generated": proposals_path,
    }


def _latest_round_trace(trace: dict[str, Any]) -> dict[str, Any]:
    rounds = trace.get("rounds", [])
    if isinstance(rounds, list) and rounds and isinstance(rounds[-1], dict):
        return rounds[-1]
    return {}


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _system_prompt(config: LLMReviewLoopConfig, profile: str | None) -> str:
    return "\n".join(
        [
            "You are reviewing a LightRAG KB iteration package.",
            "Return only JSON accepted by parse_llm_review_output.",
            f"profile: {profile or 'default'}",
            f"allow_llm_judge: {config.allow_llm_judge}",
            f"allow_llm_auto_accept: {config.allow_llm_auto_accept}",
            f"allow_low_risk_auto_reject: {config.allow_low_risk_auto_reject}",
            f"generate_patch_candidates: {config.generate_patch_candidates}",
            f"require_human_for_mutation: {config.require_human_for_mutation}",
            f"max_context_tokens_per_round: {config.max_context_tokens_per_round}",
        ]
    )


def _finding_text(finding: Any) -> str:
    if not isinstance(finding, dict):
        return ""
    parts = []
    for value in finding.values():
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, dict):
            parts.extend(str(item) for item in value.values())
        else:
            parts.append(str(value))
    return " ".join(parts)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
