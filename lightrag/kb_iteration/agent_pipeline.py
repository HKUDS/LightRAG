from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import validate_workspace

from .agent_context import (
    build_agent_observation,
    build_stage_context,
    remove_agent_context_link,
    safe_package_child_path,
    write_agent_context,
)
from .agent_outputs import parse_agent_stage_output, write_agent_stage_artifacts
from .llm_review import LLMReviewClient, LLMReviewOutput, write_llm_review_artifacts
from .models import ImprovementProposal
from .proposals import validate_proposal, write_approval_queue, write_improvement_backlog


@dataclass(frozen=True)
class LLMAgentPipelineConfig:
    max_context_tokens_per_stage: int = 12000
    max_stage_retries: int = 5
    allow_llm_judge: bool = True
    generate_patch_candidates: bool = False
    require_human_for_mutation: bool = True


@dataclass(frozen=True)
class LLMAgentPipelineResult:
    output_dir: Path
    stop_reason: str
    proposal_ids: list[str] = field(default_factory=list)
    artifact_paths: dict[str, Path] = field(default_factory=dict)


_STAGES = (
    "explain",
    "infer_branches",
    "locate_evidence",
    "propose",
    "rank_repairs",
    "judge",
)
_PROMPT_FILES = {
    "explain": "explain_zh.md",
    "infer_branches": "infer_branches_zh.md",
    "locate_evidence": "locate_evidence_zh.md",
    "propose": "propose_zh.md",
    "rank_repairs": "rank_repairs_zh.md",
    "judge": "judge_zh.md",
}

_EVIDENCE_REFERENCE_KEYS = {
    "source_id",
    "file_path",
    "item_id",
    "entity_id",
    "relation_id",
    "metric",
}
_FAILURE_STOP_REASONS = {
    "context_too_large",
    "invalid_config",
    "invalid_kb_package",
    "invalid_llm_output",
    "llm_client_error",
}
_GENERATED_AGENT_ARTIFACTS = (
    "llm_issue_analysis.json",
    "llm_issue_analysis.md",
    "llm_missing_branch_inference.json",
    "llm_missing_branch_inference.md",
    "llm_evidence_map.json",
    "llm_evidence_map.md",
    "llm_repair_plan.json",
    "llm_repair_plan.md",
    "llm_judge_report.json",
    "llm_judge_report.md",
    "llm_review_report.md",
    "proposals.generated.yaml",
    "approval_queue.md",
    "improvement_backlog.md",
    "llm_review_trace.json",
)
_GENERATED_AGENT_CONTEXT_FILES = tuple(
    f"agent_context/{stage}-context.json" for stage in _STAGES
)
_JUDGE_DECISIONS = {
    "recommend_accept",
    "recommend_reject",
    "needs_human",
    "needs_more_evidence",
}
_JUDGE_HUMAN_DECISIONS = {"needs_human", "needs_more_evidence"}


@dataclass
class _EvidenceReferenceTokens:
    all: set[str] = field(default_factory=set)
    by_key: dict[str, set[str]] = field(
        default_factory=lambda: {key: set() for key in _EVIDENCE_REFERENCE_KEYS}
    )


def run_llm_agent_pipeline(
    *,
    workspace: str,
    package_dir: str | Path,
    client: LLMReviewClient,
    config: LLMAgentPipelineConfig | None = None,
    profile: str | None = None,
) -> LLMAgentPipelineResult:
    validated_workspace = validate_workspace(workspace)
    pipeline_config = config or LLMAgentPipelineConfig()
    output_dir = Path(package_dir)
    trace: dict[str, Any] = {
        "workspace": validated_workspace,
        "profile": profile,
        "mode": "agent_pipeline",
        "started_at": _utc_timestamp(),
        "completed_at": "",
        "stop_reason": "",
        "config": asdict(pipeline_config),
        "stages": [],
        "proposal_ids": [],
    }
    artifact_paths: dict[str, Path] = {}
    _cleanup_generated_agent_artifacts(output_dir)

    try:
        _validate_config(pipeline_config)
    except ValueError as exc:
        return _finish_preflight_failure(
            output_dir=output_dir,
            stop_reason="invalid_config",
            error=str(exc),
            artifact_paths=artifact_paths,
            trace=trace,
        )

    try:
        _validate_required_artifacts(output_dir, validated_workspace)
    except ValueError as exc:
        return _finish_preflight_failure(
            output_dir=output_dir,
            stop_reason="invalid_kb_package",
            error=str(exc),
            artifact_paths=artifact_paths,
            trace=trace,
        )

    previous_outputs: dict[str, Any] = {}
    proposals = []

    for stage in _selected_stages(pipeline_config):
        context = (
            build_agent_observation(output_dir, workspace=validated_workspace)
            if stage == "explain"
            else build_stage_context(
                output_dir,
                workspace=validated_workspace,
                stage=stage,
                previous_outputs=previous_outputs,
            )
        )
        context_path = write_agent_context(output_dir, stage, context)
        artifact_paths[f"{stage}_context"] = context_path
        user_prompt = json.dumps(context, ensure_ascii=False, sort_keys=True)
        base_user_prompt = user_prompt
        input_token_estimate = _estimate_tokens(user_prompt)
        stage_trace: dict[str, Any] = {
            "stage": stage,
            "started_at": _utc_timestamp(),
            "completed_at": "",
            "state": "running",
            "attempts": 0,
            "context_files": [_relative_artifact_path(output_dir, context_path)],
            "model": _client_model(client),
            "input_token_estimate": input_token_estimate,
            "output_token_estimate": 0,
            "attempt_logs": [],
            "proposal_ids": [],
            "artifact_keys": [],
            "error": "",
        }

        if input_token_estimate > pipeline_config.max_context_tokens_per_stage:
            stage_trace["state"] = "context_too_large"
            stage_trace["error"] = (
                "input_token_estimate exceeds max_context_tokens_per_stage"
            )
            stage_trace["completed_at"] = _utc_timestamp()
            if stage == "judge" and proposals:
                return _finish_judge_unavailable(
                    output_dir=output_dir,
                    error=stage_trace["error"],
                    proposals=proposals,
                    previous_outputs=previous_outputs,
                    artifact_paths=artifact_paths,
                    trace=trace,
                    stage_trace=stage_trace,
                )
            trace["stages"].append(stage_trace)
            return _finish_run(
                output_dir=output_dir,
                stop_reason="context_too_large",
                proposal_ids=[],
                artifact_paths=artifact_paths,
                trace=trace,
            )

        raw_output = ""
        parsed = None
        max_attempts = pipeline_config.max_stage_retries + 1
        for attempt in range(1, max_attempts + 1):
            stage_trace["attempts"] = attempt
            try:
                raw_output = client.complete(
                    system_prompt=_stage_prompt(stage, profile),
                    user_prompt=user_prompt,
                )
            except Exception as exc:
                stage_trace["error"] = str(exc)
                if attempt < max_attempts:
                    continue
                if stage == "judge" and proposals:
                    return _finish_judge_unavailable(
                        output_dir=output_dir,
                        error=str(exc),
                        proposals=proposals,
                        previous_outputs=previous_outputs,
                        artifact_paths=artifact_paths,
                        trace=trace,
                        stage_trace=stage_trace,
                    )
                stage_trace["state"] = "client_error"
                stage_trace["completed_at"] = _utc_timestamp()
                trace["stages"].append(stage_trace)
                return _finish_run(
                    output_dir=output_dir,
                    stop_reason="llm_client_error",
                    proposal_ids=[],
                    artifact_paths=artifact_paths,
                    trace=trace,
                )

            try:
                parsed = parse_agent_stage_output(stage, raw_output)
                for proposal in parsed.proposals:
                    validate_proposal(proposal)
                if stage == "propose":
                    _validate_grounded_proposal_evidence(
                        parsed.proposals,
                        output_dir=output_dir,
                        previous_outputs=previous_outputs,
                    )
                if stage == "judge" and proposals:
                    _normalized_judge_results(parsed.payload, proposals)
            except ValueError as exc:
                error = str(exc)
                stage_trace["error"] = error
                stage_trace["output_token_estimate"] = _estimate_tokens(raw_output)
                _append_attempt_log(
                    stage_trace,
                    attempt=attempt,
                    state="invalid_llm_output",
                    error=error,
                    raw_output=raw_output,
                )
                if attempt < max_attempts:
                    user_prompt = _retry_user_prompt(
                        base_user_prompt,
                        stage=stage,
                        error=error,
                        previous_errors=_attempt_error_lines(stage_trace),
                    )
                    continue
                if stage == "judge" and proposals:
                    return _finish_judge_unavailable(
                        output_dir=output_dir,
                        error=error,
                        proposals=proposals,
                        previous_outputs=previous_outputs,
                        artifact_paths=artifact_paths,
                        trace=trace,
                        stage_trace=stage_trace,
                    )
                stage_trace["state"] = "invalid_llm_output"
                stage_trace["completed_at"] = _utc_timestamp()
                trace["stages"].append(stage_trace)
                return _finish_run(
                    output_dir=output_dir,
                    stop_reason="invalid_llm_output",
                    proposal_ids=[],
                    artifact_paths=artifact_paths,
                    trace=trace,
                )
            break

        if parsed is None:
            stage_trace["state"] = "invalid_llm_output"
            stage_trace["error"] = "agent stage output was not parsed"
            stage_trace["completed_at"] = _utc_timestamp()
            trace["stages"].append(stage_trace)
            return _finish_run(
                output_dir=output_dir,
                stop_reason="invalid_llm_output",
                proposal_ids=[],
                artifact_paths=artifact_paths,
                trace=trace,
            )

        stage_trace["output_token_estimate"] = _estimate_tokens(raw_output)
        stage_artifact_paths: dict[str, Path] = {}
        if stage == "propose":
            proposals = parsed.proposals
            proposal_ids = [proposal.id for proposal in proposals]
            stage_trace["proposal_ids"] = proposal_ids
        else:
            stage_artifact_paths = write_agent_stage_artifacts(output_dir, parsed)
            artifact_paths.update(stage_artifact_paths)

        previous_outputs.update(parsed.payload)
        if stage == "judge" and proposals:
            proposals = _attach_judge_results(proposals, parsed.payload)
        if stage == "propose" and not proposals:
            stage_artifact_paths = _write_proposal_artifacts(
                output_dir, previous_outputs, []
            )
            artifact_paths.update(stage_artifact_paths)
            stage_trace["artifact_keys"] = list(stage_artifact_paths)
            stage_trace["state"] = "completed"
            stage_trace["completed_at"] = _utc_timestamp()
            trace["stages"].append(stage_trace)
            return _finish_run(
                output_dir=output_dir,
                stop_reason="needs_more_evidence",
                proposal_ids=[],
                artifact_paths=artifact_paths,
                trace=trace,
            )

        stage_trace["artifact_keys"] = list(stage_artifact_paths)
        stage_trace["state"] = "completed"
        stage_trace["error"] = ""
        stage_trace["completed_at"] = _utc_timestamp()
        trace["stages"].append(stage_trace)

    proposal_ids = [proposal.id for proposal in proposals]
    trace["proposal_ids"] = proposal_ids
    if not proposals:
        artifact_paths.update(_write_proposal_artifacts(output_dir, previous_outputs, []))
        return _finish_run(
            output_dir=output_dir,
            stop_reason="needs_more_evidence",
            proposal_ids=[],
            artifact_paths=artifact_paths,
            trace=trace,
        )

    if not pipeline_config.allow_llm_judge:
        reason = "judge stage disabled by configuration"
        return _finish_judge_unavailable(
            output_dir=output_dir,
            error=reason,
            proposals=proposals,
            previous_outputs=previous_outputs,
            artifact_paths=artifact_paths,
            trace=trace,
            stage_trace=_synthetic_judge_stage_trace(reason),
        )

    proposals = _human_gate_proposals(
        _attach_judge_results(proposals, previous_outputs)
    )
    proposal_artifact_paths = _write_proposal_artifacts(
        output_dir, previous_outputs, proposals
    )
    artifact_paths.update(proposal_artifact_paths)
    _add_stage_artifact_keys(trace, "propose", list(proposal_artifact_paths))

    return _finish_run(
        output_dir=output_dir,
        stop_reason="pending_human_review",
        proposal_ids=proposal_ids,
        artifact_paths=artifact_paths,
        trace=trace,
    )


def _selected_stages(config: LLMAgentPipelineConfig) -> tuple[str, ...]:
    if config.allow_llm_judge:
        return _STAGES
    return tuple(stage for stage in _STAGES if stage != "judge")


def _append_attempt_log(
    stage_trace: dict[str, Any],
    *,
    attempt: int,
    state: str,
    error: str,
    raw_output: str = "",
) -> None:
    attempt_logs = stage_trace.setdefault("attempt_logs", [])
    if not isinstance(attempt_logs, list):
        attempt_logs = []
        stage_trace["attempt_logs"] = attempt_logs
    attempt_logs.append(
        {
            "attempt": attempt,
            "state": state,
            "error": error,
            "output_token_estimate": _estimate_tokens(raw_output) if raw_output else 0,
        }
    )


def _attempt_error_lines(stage_trace: dict[str, Any]) -> list[str]:
    lines = []
    attempt_logs = stage_trace.get("attempt_logs")
    if not isinstance(attempt_logs, list):
        return lines
    for item in attempt_logs:
        if not isinstance(item, dict):
            continue
        attempt = item.get("attempt")
        error = item.get("error")
        if isinstance(attempt, int) and isinstance(error, str) and error:
            lines.append(f"Attempt {attempt}: {error}")
    return lines


def _retry_user_prompt(
    base_user_prompt: str,
    *,
    stage: str,
    error: str,
    previous_errors: list[str] | None = None,
) -> str:
    guidance = [
        f"Previous output was rejected: {error}.",
        "Return corrected JSON only.",
    ]
    if previous_errors:
        guidance.extend(["Previous rejected attempts:", *previous_errors])
    guidance.extend(
        [
            "Preserve deterministic evidence requirements.",
            "Do not invent source_id, file_path, entity_id, relation_id, item_id, or metric references.",
            "Do not claim LLM inference as medical evidence.",
        ]
    )
    if stage == "propose":
        guidance.append(
            'For "expected_metric_change", use finite JSON numbers only; '
            "use {} if no numeric estimate is available."
        )
        guidance.append(
            "Every non-context proposal must keep requires_approval=true when it can mutate KG, rules, prompts, workspace, or WebUI behavior."
        )
        guidance.append("Do not change evidence/human-approval requirements.")
    return "\n\n".join([base_user_prompt, *guidance])


def _synthetic_judge_stage_trace(reason: str) -> dict[str, Any]:
    return {
        "stage": "judge",
        "started_at": _utc_timestamp(),
        "completed_at": "",
        "state": "running",
        "attempts": 0,
        "attempt_logs": [],
        "context_files": [],
        "model": "not_called",
        "input_token_estimate": 0,
        "output_token_estimate": 0,
        "proposal_ids": [],
        "artifact_keys": [],
        "error": reason,
    }


def _cleanup_generated_agent_artifacts(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    remove_agent_context_link(output_dir)
    for relative_path in (*_GENERATED_AGENT_ARTIFACTS, *_GENERATED_AGENT_CONTEXT_FILES):
        artifact_path = safe_package_child_path(output_dir, relative_path)
        if artifact_path.is_file():
            artifact_path.unlink()


def _attach_judge_results(
    proposals: list[ImprovementProposal], payload: dict[str, Any]
) -> list[ImprovementProposal]:
    judge_by_id = _normalized_judge_results(payload, proposals)
    if not judge_by_id:
        return proposals
    return [
        _proposal_with_judge_result(proposal, judge_by_id.get(proposal.id))
        for proposal in proposals
    ]


def _proposal_with_judge_result(
    proposal: ImprovementProposal, judge_result: dict[str, Any] | None
) -> ImprovementProposal:
    if judge_result is None:
        return proposal
    decision = str(judge_result.get("decision", "")).strip().casefold()
    return replace(
        proposal,
        requires_approval=proposal.requires_approval
        or decision in _JUDGE_HUMAN_DECISIONS,
        judge={**proposal.judge, **judge_result},
    )


def _human_gate_proposals(
    proposals: list[ImprovementProposal],
) -> list[ImprovementProposal]:
    return [
        proposal
        if proposal.requires_approval
        else replace(proposal, requires_approval=True)
        for proposal in proposals
    ]


def _normalized_judge_results(
    payload: dict[str, Any], proposals: list[ImprovementProposal]
) -> dict[str, dict[str, Any]]:
    judge_results = payload.get("judge_results")
    if judge_results is None:
        if "decision" in payload and len(proposals) == 1:
            judge_results = [{**payload, "proposal_id": proposals[0].id}]
        else:
            raise ValueError("judge_results is required for judge output")
    if not isinstance(judge_results, list):
        raise ValueError("judge_results must be a list")

    proposal_ids = {proposal.id for proposal in proposals}
    normalized: dict[str, dict[str, Any]] = {}
    for result in judge_results:
        if not isinstance(result, dict):
            raise ValueError("judge_results entries must be objects")
        proposal_id = str(result.get("proposal_id", "")).strip()
        if proposal_id not in proposal_ids:
            raise ValueError("judge_results proposal_id must match a proposal")
        if proposal_id in normalized:
            raise ValueError("judge_results contains duplicate proposal_id")
        decision = str(result.get("decision", "")).strip().casefold()
        if not decision:
            raise ValueError("judge_results decision is required")
        if decision not in _JUDGE_DECISIONS:
            raise ValueError("judge_results decision must match the judge schema")
        reason = str(result.get("reason", "")).strip()
        if not reason:
            raise ValueError("judge_results reason is required")
        normalized[proposal_id] = {
            **result,
            "proposal_id": proposal_id,
            "decision": decision,
            "reason": reason,
        }
    missing_proposal_ids = sorted(proposal_ids - set(normalized))
    if missing_proposal_ids:
        raise ValueError(
            "judge_results missing proposal_id entries: "
            + ", ".join(missing_proposal_ids)
        )
    return normalized


def _write_proposal_artifacts(
    output_dir: Path,
    previous_outputs: dict[str, Any],
    proposals: list[ImprovementProposal],
) -> dict[str, Path]:
    review_output = LLMReviewOutput(
        confirmed_issues=_list_of_dicts(previous_outputs.get("issue_explanations")),
        hypotheses=[],
        missing_evidence=_list_of_dicts(previous_outputs.get("missing_evidence")),
        out_of_scope=[],
        proposals=proposals,
    )
    artifact_paths = write_llm_review_artifacts(review_output, output_dir)
    artifact_paths["approval_queue"] = write_approval_queue(proposals, output_dir)
    artifact_paths["improvement_backlog"] = write_improvement_backlog(
        proposals, output_dir
    )
    return artifact_paths


def _validate_grounded_proposal_evidence(
    proposals: list[ImprovementProposal],
    *,
    output_dir: Path,
    previous_outputs: dict[str, Any],
) -> None:
    reference_tokens = _grounded_reference_tokens(output_dir, previous_outputs)
    for proposal in proposals:
        if proposal.type == "review_context_request":
            continue
        for evidence in proposal.evidence:
            if not evidence.strip():
                continue
            if _evidence_references_known_artifact(evidence, reference_tokens):
                continue
            raise ValueError(
                f"proposal {proposal.id} evidence is not grounded in deterministic "
                "artifacts"
            )


def _grounded_reference_tokens(
    output_dir: Path, previous_outputs: dict[str, Any]
) -> _EvidenceReferenceTokens:
    tokens = _EvidenceReferenceTokens()
    snapshot = _read_json(output_dir / "snapshots" / "kg_snapshot.json")
    quality = _read_json(output_dir / "snapshots" / "quality_score.json")

    if isinstance(snapshot, dict):
        _add_typed_reference_values(tokens, "file_path", snapshot.get("source_files"))
        for node in _list_of_dicts(snapshot.get("nodes")):
            _add_typed_reference_token(tokens, "entity_id", node.get("id"))
            _add_typed_reference_token(tokens, "item_id", node.get("id"))
            _add_typed_reference_token(tokens, "source_id", node.get("source_id"))
            _add_typed_reference_token(tokens, "file_path", node.get("file_path"))
        for edge in _list_of_dicts(snapshot.get("edges")):
            _add_typed_reference_token(tokens, "relation_id", edge.get("id"))
            _add_typed_reference_token(tokens, "item_id", edge.get("id"))
            _add_typed_reference_token(tokens, "entity_id", edge.get("source"))
            _add_typed_reference_token(tokens, "entity_id", edge.get("target"))
            _add_typed_reference_token(tokens, "item_id", edge.get("source"))
            _add_typed_reference_token(tokens, "item_id", edge.get("target"))
            _add_typed_reference_token(tokens, "source_id", edge.get("source_id"))
            _add_typed_reference_token(tokens, "file_path", edge.get("file_path"))

    if isinstance(quality, dict):
        subscores = quality.get("subscores")
        for key in _dict_keys(subscores):
            _add_typed_reference_token(tokens, "metric", key)
            _add_quality_reference_token(tokens, key, _dict_value(subscores, key))
        metrics = quality.get("metrics")
        for key in _dict_keys(metrics):
            _add_typed_reference_token(tokens, "metric", key)
            _add_quality_reference_token(tokens, key, _dict_value(metrics, key))
        for finding in _list_of_dicts(quality.get("findings")):
            _add_reference_values(tokens, finding.get("evidence"))

    return tokens


def _evidence_references_known_artifact(
    evidence: Any, reference_tokens: _EvidenceReferenceTokens
) -> bool:
    if not isinstance(evidence, str) or not evidence.strip():
        return False
    evidence_text = _normalize_reference_part(evidence)
    if evidence_text in reference_tokens.all:
        return True
    if ":" not in evidence:
        return False
    return _structured_evidence_references_known_artifacts(
        evidence, reference_tokens
    )


def _structured_evidence_references_known_artifacts(
    evidence: str, reference_tokens: _EvidenceReferenceTokens
) -> bool:
    segments = [segment.strip() for segment in evidence.split(";")]
    if not segments or any(not segment for segment in segments):
        return False

    has_value = False
    for segment in segments:
        key, separator, value = segment.partition(":")
        normalized_key = key.strip().casefold()
        if not separator or normalized_key not in _EVIDENCE_REFERENCE_KEYS:
            return False

        value_parts = [part.strip() for part in value.split(",")]
        non_empty_values = [part for part in value_parts if part]
        if not non_empty_values:
            return False
        allowed_tokens = reference_tokens.by_key[normalized_key]
        for value_part in non_empty_values:
            if _normalize_reference_part(value_part) not in allowed_tokens:
                return False
        has_value = True

    return has_value


def _add_reference_values(tokens: _EvidenceReferenceTokens, value: Any) -> None:
    if isinstance(value, dict):
        for nested_value in value.values():
            _add_reference_values(tokens, nested_value)
        return
    if isinstance(value, list):
        for item in value:
            _add_reference_values(tokens, item)
        return
    _add_reference_token(tokens, value)


def _add_reference_token(tokens: _EvidenceReferenceTokens, value: Any) -> None:
    if value is None:
        return
    normalized = str(value).replace(GRAPH_FIELD_SEP, "\n").replace("|", "\n")
    for part in normalized.splitlines():
        token = _normalize_reference_part(part)
        if len(token) >= 3:
            tokens.all.add(token)


def _add_typed_reference_values(
    tokens: _EvidenceReferenceTokens, key: str, value: Any
) -> None:
    if isinstance(value, dict):
        for nested_value in value.values():
            _add_typed_reference_values(tokens, key, nested_value)
        return
    if isinstance(value, list):
        for item in value:
            _add_typed_reference_values(tokens, key, item)
        return
    _add_typed_reference_token(tokens, key, value)


def _add_typed_reference_token(
    tokens: _EvidenceReferenceTokens, key: str, value: Any
) -> None:
    _add_reference_token(tokens, value)
    if value is None:
        return
    normalized = str(value).replace(GRAPH_FIELD_SEP, "\n").replace("|", "\n")
    for part in normalized.splitlines():
        token = _normalize_reference_part(part)
        if len(token) >= 3:
            tokens.by_key[key].add(token)


def _add_quality_reference_token(
    tokens: _EvidenceReferenceTokens, key: str, value: Any
) -> None:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return
    _add_typed_reference_token(tokens, "metric", f"quality:{key}={value}")


def _normalize_reference_part(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _dict_value(value: Any, key: str) -> Any:
    if not isinstance(value, dict):
        return None
    return value.get(key)


def _dict_keys(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    return [str(key) for key in value if str(key).strip()]


def _validate_config(config: LLMAgentPipelineConfig) -> None:
    if config.max_context_tokens_per_stage <= 0:
        raise ValueError("max_context_tokens_per_stage must be greater than 0")
    if config.max_stage_retries < 0:
        raise ValueError("max_stage_retries must be greater than or equal to 0")


def _validate_required_artifacts(output_dir: Path, workspace: str) -> None:
    snapshot_path = output_dir / "snapshots" / "kg_snapshot.json"
    quality_path = output_dir / "snapshots" / "quality_score.json"
    for path in (snapshot_path, quality_path):
        if not path.exists():
            raise ValueError(
                "KB iteration package is missing required artifact: "
                f"{path.relative_to(output_dir).as_posix()}"
            )

    snapshot = _read_json(snapshot_path)
    quality = _read_json(quality_path)
    if not isinstance(snapshot, dict):
        raise ValueError("kg_snapshot.json must contain a JSON object")
    if not isinstance(quality, dict):
        raise ValueError("quality_score.json must contain a JSON object")

    snapshot_workspace = snapshot.get("workspace")
    if snapshot_workspace and str(snapshot_workspace) != workspace:
        raise ValueError(
            "kg_snapshot.json workspace does not match requested workspace"
        )


def _finish_run(
    *,
    output_dir: Path,
    stop_reason: str,
    proposal_ids: list[str],
    artifact_paths: dict[str, Path],
    trace: dict[str, Any],
) -> LLMAgentPipelineResult:
    trace["completed_at"] = _utc_timestamp()
    trace["stop_reason"] = stop_reason
    trace["proposal_ids"] = proposal_ids
    if stop_reason in _FAILURE_STOP_REASONS:
        artifact_paths.update(_write_failure_artifacts(output_dir, stop_reason, trace))

    trace_path = output_dir / "llm_review_trace.json"
    with trace_path.open("w", encoding="utf-8") as file:
        json.dump(trace, file, ensure_ascii=False, indent=2)
        file.write("\n")
    artifact_paths["llm_review_trace"] = trace_path
    return LLMAgentPipelineResult(
        output_dir=output_dir,
        stop_reason=stop_reason,
        proposal_ids=proposal_ids,
        artifact_paths=artifact_paths,
    )


def _finish_judge_unavailable(
    *,
    output_dir: Path,
    error: str,
    proposals: list[ImprovementProposal],
    previous_outputs: dict[str, Any],
    artifact_paths: dict[str, Path],
    trace: dict[str, Any],
    stage_trace: dict[str, Any],
) -> LLMAgentPipelineResult:
    proposal_ids = [proposal.id for proposal in proposals]
    reason = _single_line(error) or "judge stage failed"
    proposals_with_judge = [
        replace(
            proposal,
            judge={
                **proposal.judge,
                "decision": "judge_unavailable",
                "reason": reason,
            },
        )
        for proposal in proposals
    ]
    proposals_with_judge = _human_gate_proposals(proposals_with_judge)
    judge_artifacts = _write_judge_unavailable_artifact(output_dir, reason)
    artifact_paths.update(judge_artifacts)
    stage_trace["state"] = "judge_unavailable"
    stage_trace["artifact_keys"] = list(judge_artifacts)
    stage_trace["completed_at"] = _utc_timestamp()
    trace["stages"].append(stage_trace)

    proposal_artifact_paths = _write_proposal_artifacts(
        output_dir, previous_outputs, proposals_with_judge
    )
    artifact_paths.update(proposal_artifact_paths)
    _add_stage_artifact_keys(trace, "propose", list(proposal_artifact_paths))
    return _finish_run(
        output_dir=output_dir,
        stop_reason="pending_human_review",
        proposal_ids=proposal_ids,
        artifact_paths=artifact_paths,
        trace=trace,
    )


def _write_judge_unavailable_artifact(output_dir: Path, reason: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "llm_judge_report.md"
    path.write_text(
        "\n".join(
            [
                "# Judge Report",
                "",
                "## Summary",
                "",
                "- Decision: judge_unavailable",
                f"- Reason: {reason}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"llm_judge_report": path}


def _finish_preflight_failure(
    *,
    output_dir: Path,
    stop_reason: str,
    error: str,
    artifact_paths: dict[str, Path],
    trace: dict[str, Any],
) -> LLMAgentPipelineResult:
    timestamp = _utc_timestamp()
    trace["stages"].append(
        {
            "stage": "preflight",
            "started_at": timestamp,
            "completed_at": timestamp,
            "state": stop_reason,
            "attempts": 0,
            "context_files": [],
            "model": "not_called",
            "input_token_estimate": 0,
            "output_token_estimate": 0,
            "proposal_ids": [],
            "artifact_keys": [],
            "error": error,
        }
    )
    return _finish_run(
        output_dir=output_dir,
        stop_reason=stop_reason,
        proposal_ids=[],
        artifact_paths=artifact_paths,
        trace=trace,
    )


def _write_failure_artifacts(
    output_dir: Path, stop_reason: str, trace: dict[str, Any]
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "llm_review_report.md"
    proposals_path = output_dir / "proposals.generated.yaml"
    latest_stage = _latest_stage_trace(trace)
    error = str(latest_stage.get("error", "")).strip()
    state = str(latest_stage.get("state", "")).strip()
    stage = str(latest_stage.get("stage", "")).strip()

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
    if stage:
        report_lines.append(f"- Stage: {stage}")
    if state:
        report_lines.append(f"- State: {state}")
    if error:
        report_lines.append(f"- Error: {_single_line(error)}")
    if not any(line.startswith("- ") for line in report_lines[-3:]):
        report_lines.append("- Error: unknown")
    attempt_logs = latest_stage.get("attempt_logs")
    if isinstance(attempt_logs, list) and attempt_logs:
        report_lines.extend(["", "## Rejected Attempts", ""])
        for item in attempt_logs:
            if not isinstance(item, dict):
                continue
            attempt = item.get("attempt")
            attempt_error = item.get("error")
            if isinstance(attempt, int) and isinstance(attempt_error, str):
                clean_error = _single_line(attempt_error)
                if clean_error:
                    report_lines.append(f"- Attempt {attempt}: {clean_error}")
    report_lines.extend(["", "## Generated Proposals", "", "- none", ""])

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    proposals_path.write_text(
        "# Generated Proposals\n\nproposals: []\n",
        encoding="utf-8",
    )
    approval_queue_path = write_approval_queue([], output_dir)
    backlog_path = write_improvement_backlog([], output_dir)
    return {
        "llm_review_report": report_path,
        "proposals_generated": proposals_path,
        "approval_queue": approval_queue_path,
        "improvement_backlog": backlog_path,
    }


def _stage_prompt(stage: str, profile: str | None) -> str:
    prompt_path = Path(__file__).parent / "prompts" / _PROMPT_FILES[stage]
    prompt = prompt_path.read_text(encoding="utf-8")
    return "\n".join(
        [
            prompt.strip(),
            "",
            f"profile: {profile or 'default'}",
            "Return only JSON.",
        ]
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


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_stage_trace(trace: dict[str, Any]) -> dict[str, Any]:
    stages = trace.get("stages", [])
    if isinstance(stages, list) and stages and isinstance(stages[-1], dict):
        return stages[-1]
    return {}


def _add_stage_artifact_keys(
    trace: dict[str, Any], stage_name: str, artifact_keys: list[str]
) -> None:
    stages = trace.get("stages", [])
    if not isinstance(stages, list):
        return
    for stage in stages:
        if not isinstance(stage, dict) or stage.get("stage") != stage_name:
            continue
        current_keys = stage.get("artifact_keys", [])
        if not isinstance(current_keys, list):
            current_keys = []
        stage["artifact_keys"] = [*current_keys, *artifact_keys]
        return


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
