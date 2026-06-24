from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import validate_workspace

from .agent_context import (
    _is_link_or_junction,
    _remove_link,
    build_agent_observation,
    build_stage_context,
    remove_agent_context_link,
    safe_package_child_path,
    write_agent_context,
)
from .agent_outputs import (
    AgentStageOutput,
    parse_agent_stage_output,
    write_agent_stage_artifacts,
)
from .llm_review import LLMReviewClient, LLMReviewOutput, write_llm_review_artifacts
from .medical_schema import render_medical_relation_schema_prompt
from .models import ImprovementProposal
from .issue_ledger import (
    DeterministicScanResult,
    normalize_deterministic_family_caps,
    scan_deterministic_candidates,
)
from .proposal_orchestrator import (
    build_llm_residual_task_packs,
    merge_subagent_proposals,
)
from .proposal_funnel import (
    build_proposal_funnel_report,
    write_proposal_funnel_artifacts,
)
from .proposals import validate_proposal, write_approval_queue, write_improvement_backlog
from .subagent_contracts import role_contract, validate_proposal_role_contract
from .zh_artifacts import artifact_zh_relative_path


MAX_SUBAGENT_TASKS = 50
MAX_PARALLEL_SUBAGENTS = 8
MAX_PROPOSALS_PER_RUN = 200
MAX_STAGE_RETRIES = 8
MAX_SUBAGENT_COMPACT_LIST_ITEMS = 8
MAX_SUBAGENT_COMPACT_DICT_ITEMS = 12
MAX_SUBAGENT_COMPACT_STRING_CHARS = 360
MAX_SUBAGENT_RULES_MEMORY_CHARS = 1200
MAX_SUBAGENT_ISSUES_PER_TASK = 4
MAX_SUBAGENT_PROPOSALS_PER_TASK = 2
MAX_AUXILIARY_CONTEXT_LIST_ITEMS = 8
MAX_AUXILIARY_CONTEXT_DICT_ITEMS = 12
MAX_AUXILIARY_CONTEXT_STRING_CHARS = 240
MAX_AUXILIARY_PROPOSAL_TEXT_CHARS = 180
MAX_AUXILIARY_PROPOSAL_EVIDENCE_ITEMS = 2
MAX_AUXILIARY_ACTION_VALUE_CHARS = 120
MAX_AUXILIARY_RULES_MEMORY_CHARS = 600
MAX_AUXILIARY_PROPOSALS = 32


@dataclass(frozen=True)
class LLMAgentPipelineConfig:
    max_context_tokens_per_stage: int = 12000
    max_stage_retries: int = 1
    allow_llm_judge: bool = True
    generate_patch_candidates: bool = False
    require_human_for_mutation: bool = True
    max_subagent_tasks: int = 8
    max_parallel_subagents: int = 4
    max_subagent_issues_per_task: int = MAX_SUBAGENT_ISSUES_PER_TASK
    max_subagent_proposals_per_task: int = MAX_SUBAGENT_PROPOSALS_PER_TASK
    max_proposals_per_run: int = MAX_PROPOSALS_PER_RUN
    strict_subagent_role_contracts: bool = True
    prevalidate_action_candidates: bool = True
    require_candidate_evidence_allowlist: bool = True
    skip_deterministic_subagent_calls: bool = False
    deterministic_family_caps: dict[str, int] | None = None


@dataclass(frozen=True)
class LLMAgentPipelineResult:
    output_dir: Path
    stop_reason: str
    proposal_ids: list[str] = field(default_factory=list)
    artifact_paths: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionCandidateProposalBuildResult:
    proposals: list[ImprovementProposal] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)


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
_SUBAGENT_PROMPT_FILES = {
    "schema_repair": "schema_repair_zh.md",
    "treatment": "treatment_zh.md",
    "treatment_split": "treatment_split_zh.md",
    "prevention": "prevention_zh.md",
    "risk_safety": "risk_safety_zh.md",
    "diagnosis": "diagnosis_zh.md",
    "clinical_modeling": "clinical_modeling_zh.md",
    "evidence_grounding": "evidence_grounding_zh.md",
    "general": "general_zh.md",
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
    "proposal_task_packs.json",
    "proposal_funnel_report.json",
    "proposal_funnel_report.md",
    "proposal_conflict_groups.json",
    "proposal_merge_report.md",
    "issue_ledger.json",
    "proposal_issue_ledger.json",
    "deterministic_proposal_report.json",
    "deterministic_proposal_report.md",
    "subagent_outputs/index.json",
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
_EXECUTABLE_MEDICAL_PROPOSAL_TYPES = {
    "medical_relation_schema_migration",
    "value_node_to_qualifier",
    "candidate_kg_expansion",
    "entity_alias_merge",
    "medical_fact_role_split",
}
_ACTION_PAYLOAD_TYPED_REFERENCE_KEYS = {
    "edge_id": "relation_id",
    "expected_source": "entity_id",
    "expected_target": "entity_id",
    "new_source": "entity_id",
    "new_target": "entity_id",
}
_CANDIDATE_KG_EXPANSION_PAYLOAD_REFERENCE_KEYS = {
    "source_id": "source_id",
    "file_path": "file_path",
}
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b("
    r"x-api-key|api[_-]?key|"
    r"token|access[_-]?token|refresh[_-]?token|[a-z0-9]+[_-]token|"
    r"secret|client[_-]?secret|[a-z0-9]+[_-]secret|password"
    r")(\s*[:=]\s*)([\"']?)([^\s,;\"']+)([\"']?)"
)
_BEARER_TOKEN_RE = re.compile(
    r"(?i)\b(Authorization\s*:\s*Bearer\s+|Bearer\s+)([A-Za-z0-9._~+/=-]+)"
)
_OPENAI_KEY_RE = re.compile(r"\bsk-[^\s,;\"']+\b")
_PROPOSAL_DECISION_MEMORY_FILES = (
    "accepted_changes.md",
    "rejected_changes.md",
    "deferred_changes.md",
    "proposal_revision_requests.md",
)
_PROPOSAL_ID_HEADING_RE = re.compile(
    r"(?m)^##\s+([A-Za-z0-9_.-]+)\s*$"
)
_PROPOSAL_ID_JSON_FIELD_RE = re.compile(
    r'"proposal_id"\s*:\s*"([A-Za-z0-9_.-]+)"'
)


@dataclass
class _EvidenceReferenceTokens:
    all: set[str] = field(default_factory=set)
    by_key: dict[str, set[str]] = field(
        default_factory=lambda: {key: set() for key in _EVIDENCE_REFERENCE_KEYS}
    )
    text_corpus: str = ""
    text_corpus_by_reference: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class _ProposeStageResult:
    parsed: AgentStageOutput | None = None
    raw_output: str = ""
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    stop_reason: str = ""


@dataclass
class _SubagentTaskResult:
    task_id: str
    role: str
    proposals: list[ImprovementProposal] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    index_entry: dict[str, Any] | None = None
    attempt_logs: list[dict[str, Any]] = field(default_factory=list)
    attempts: int = 0
    input_token_estimate: int = 0
    state: str = "completed"
    error: str = ""
    stop_reason: str = ""


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
        if stage in {"rank_repairs", "judge"} and proposals:
            context = _compact_auxiliary_stage_context(
                context,
                stage=stage,
                proposals=proposals,
                previous_outputs=previous_outputs,
                max_context_tokens=pipeline_config.max_context_tokens_per_stage,
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
        if stage == "propose":
            stage_trace["context_artifact_token_estimate"] = input_token_estimate
            stage_trace["input_token_estimate"] = 0

        if (
            stage != "propose"
            and input_token_estimate > pipeline_config.max_context_tokens_per_stage
        ):
            stage_trace["state"] = "context_too_large"
            stage_trace["error"] = (
                "input_token_estimate exceeds max_context_tokens_per_stage"
            )
            stage_trace["completed_at"] = _utc_timestamp()
            if stage in {"rank_repairs", "judge"} and proposals:
                return _finish_judge_unavailable(
                    output_dir=output_dir,
                    error=_auxiliary_stage_unavailable_reason(
                        stage, stage_trace["error"]
                    ),
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
        stage_artifact_paths: dict[str, Path] = {}
        if stage == "propose":
            propose_result = _run_orchestrated_propose_stage(
                output_dir=output_dir,
                client=client,
                config=pipeline_config,
                profile=profile,
                context=context,
                previous_outputs=previous_outputs,
                stage_trace=stage_trace,
            )
            raw_output = propose_result.raw_output
            parsed = propose_result.parsed
            stage_artifact_paths = propose_result.artifact_paths
            artifact_paths.update(stage_artifact_paths)
            if propose_result.stop_reason:
                stage_trace["completed_at"] = _utc_timestamp()
                trace["stages"].append(stage_trace)
                return _finish_run(
                    output_dir=output_dir,
                    stop_reason=propose_result.stop_reason,
                    proposal_ids=[],
                    artifact_paths=artifact_paths,
                    trace=trace,
                )
        else:
            max_attempts = pipeline_config.max_stage_retries + 1
            for attempt in range(1, max_attempts + 1):
                stage_trace["attempts"] = attempt
                try:
                    raw_output = client.complete(
                        system_prompt=_stage_prompt(stage, profile),
                        user_prompt=user_prompt,
                    )
                except Exception as exc:
                    error = _sanitize_error_text(exc)
                    stage_trace["error"] = error
                    _append_attempt_log(
                        stage_trace,
                        attempt=attempt,
                        state="client_error",
                        error=error,
                    )
                    if attempt < max_attempts:
                        user_prompt = _client_error_retry_user_prompt(
                            base_user_prompt,
                            error=error,
                            previous_errors=_attempt_error_lines(stage_trace),
                        )
                        continue
                    if stage in {"rank_repairs", "judge"} and proposals:
                        return _finish_judge_unavailable(
                            output_dir=output_dir,
                            error=_auxiliary_stage_unavailable_reason(stage, error),
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
                    if stage == "judge" and proposals:
                        _normalized_judge_results(
                            parsed.payload,
                            _auxiliary_context_proposals(context, proposals),
                        )
                except ValueError as exc:
                    error = _sanitize_error_text(exc)
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
                    if stage in {"rank_repairs", "judge"} and proposals:
                        return _finish_judge_unavailable(
                            output_dir=output_dir,
                            error=_auxiliary_stage_unavailable_reason(stage, error),
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
        if stage == "propose":
            proposals = parsed.proposals
            proposal_ids = [proposal.id for proposal in proposals]
            stage_trace["proposal_ids"] = proposal_ids
        else:
            stage_artifact_paths = write_agent_stage_artifacts(output_dir, parsed)
            artifact_paths.update(stage_artifact_paths)

        previous_outputs.update(parsed.payload)
        if stage == "judge" and proposals:
            proposals = _attach_judge_results(
                proposals,
                parsed.payload,
                expected_proposals=_auxiliary_context_proposals(context, proposals),
            )
        if stage == "propose" and not proposals:
            proposal_artifact_paths = _write_proposal_artifacts(
                output_dir, previous_outputs, []
            )
            stage_artifact_paths.update(proposal_artifact_paths)
            artifact_paths.update(proposal_artifact_paths)
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

    proposals = _human_gate_proposals(proposals)
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


def _run_orchestrated_propose_stage(
    *,
    output_dir: Path,
    client: LLMReviewClient,
    config: LLMAgentPipelineConfig,
    profile: str | None,
    context: dict[str, Any],
    previous_outputs: dict[str, Any],
    stage_trace: dict[str, Any],
) -> _ProposeStageResult:
    artifact_paths: dict[str, Path] = {}
    deterministic_scan = scan_deterministic_candidates(
        output_dir,
        prevalidate_action_candidates=config.prevalidate_action_candidates,
        deterministic_family_caps=config.deterministic_family_caps,
    )
    raw_residual_task_pack_dicts = _proposal_task_pack_dicts(
        build_llm_residual_task_packs(
            output_dir,
            deterministic_scan,
            max_packs=MAX_SUBAGENT_TASKS,
            require_candidate_evidence_allowlist=(
                config.require_candidate_evidence_allowlist
            ),
        )
    )
    raw_task_pack_dicts = raw_residual_task_pack_dicts
    task_pack_dicts = _fit_proposal_task_packs_to_subagent_budget(
        raw_task_pack_dicts,
        previous_outputs=previous_outputs,
        context=context,
        profile=profile,
        max_context_tokens=config.max_context_tokens_per_stage,
        max_task_packs=config.max_subagent_tasks,
        max_subagent_issues_per_task=config.max_subagent_issues_per_task,
        max_subagent_proposals_per_task=config.max_subagent_proposals_per_task,
        require_candidate_evidence_allowlist=(
            config.require_candidate_evidence_allowlist
        ),
    )
    task_packs_path = _write_json_artifact(
        output_dir,
        "proposal_task_packs.json",
        task_pack_dicts,
    )
    artifact_paths["proposal_task_packs"] = task_packs_path
    stage_trace["subagent_task_count"] = len(task_pack_dicts)
    stage_trace["raw_task_count"] = len(raw_task_pack_dicts)
    stage_trace["raw_issue_count"] = len(deterministic_scan.issues)
    stage_trace["residual_issue_count"] = _task_pack_detected_issue_count(
        raw_residual_task_pack_dicts
    )
    stage_trace["fitted_task_count"] = len(task_pack_dicts)
    stage_trace["fitted_issue_count"] = _task_pack_issue_count(task_pack_dicts)
    stage_trace["subagent_issue_family_counts"] = _task_pack_issue_family_counts(
        task_pack_dicts
    )
    stage_trace["subagent_omitted_issue_family_counts"] = (
        _task_pack_omitted_family_counts(task_pack_dicts)
    )
    stage_trace["action_candidate_count"] = len(deterministic_scan.candidates)
    stage_trace["fitted_action_candidate_count"] = _task_pack_action_candidate_count(
        task_pack_dicts
    )
    if len(task_pack_dicts) != len(raw_task_pack_dicts):
        stage_trace["subagent_task_pack_fit"] = {
            "raw_task_count": len(raw_task_pack_dicts),
            "fitted_task_count": len(task_pack_dicts),
            "raw_issue_count": _task_pack_detected_issue_count(raw_task_pack_dicts),
            "fitted_issue_count": _task_pack_issue_count(task_pack_dicts),
        }

    max_attempts = config.max_stage_retries + 1
    task_results = _run_subagent_tasks(
        output_dir=output_dir,
        client=client,
        config=config,
        profile=profile,
        context=context,
        previous_outputs=previous_outputs,
        task_pack_dicts=task_pack_dicts,
        max_attempts=max_attempts,
        stage_trace=stage_trace,
    )

    proposal_batches: list[list[ImprovementProposal]] = []
    raw_outputs: list[str] = []
    subagent_index: list[dict[str, Any]] = []
    failed_results: list[_SubagentTaskResult] = []
    attempt_logs = stage_trace.setdefault("attempt_logs", [])
    if not isinstance(attempt_logs, list):
        attempt_logs = []
        stage_trace["attempt_logs"] = attempt_logs

    for result in task_results:
        artifact_paths.update(result.artifact_paths)
        raw_outputs.extend(result.raw_outputs)
        stage_trace["attempts"] = int(stage_trace.get("attempts") or 0) + (
            result.attempts
        )
        if result.input_token_estimate:
            stage_trace["input_token_estimate"] = max(
                int(stage_trace.get("input_token_estimate") or 0),
                result.input_token_estimate,
            )
        attempt_logs.extend(result.attempt_logs)
        if result.index_entry is not None:
            subagent_index.append(result.index_entry)
        if result.stop_reason:
            failed_results.append(result)
            continue
        proposal_batches.append(result.proposals)
    stage_trace["invalid_subagent_output_count"] = sum(
        1 for result in task_results if result.stop_reason == "invalid_llm_output"
    )
    stage_trace["llm_proposal_count"] = sum(
        len(batch) for batch in proposal_batches
    )

    if subagent_index:
        index_path = _write_json_artifact(
            output_dir,
            "subagent_outputs/index.json",
            subagent_index,
        )
        artifact_paths["subagent_output_index"] = index_path

    action_candidate_build = action_candidate_proposals_from_scan(
        deterministic_scan,
        output_dir=output_dir,
        previous_outputs=previous_outputs,
    )
    action_candidate_proposals = action_candidate_build.proposals
    action_candidate_proposal_ids = {
        proposal.id for proposal in action_candidate_proposals
    }
    merged_batches = (
        [action_candidate_proposals, *proposal_batches]
        if action_candidate_proposals
        else proposal_batches
    )
    merged = merge_subagent_proposals(
        merged_batches,
        max_proposals=config.max_proposals_per_run,
    )
    stage_trace["action_candidate_generated_proposal_count"] = len(
        action_candidate_proposals
    )
    stage_trace["candidate_to_proposal_rejected_count"] = len(
        action_candidate_build.rejected
    )
    stage_trace["candidate_to_proposal_rejections"] = action_candidate_build.rejected
    filtered_proposals, stale_drops = _drop_stale_relation_replacement_proposals(
        merged.proposals,
        output_dir=output_dir,
    )
    selected_proposals, proposal_id_disambiguations = (
        _disambiguate_proposal_ids_used_by_decision_memory(
            filtered_proposals,
            output_dir=output_dir,
        )
    )
    if proposal_id_disambiguations:
        _sync_issue_route_proposal_ids(
            deterministic_scan,
            proposal_id_disambiguations,
        )
        action_candidate_proposal_ids = _remapped_proposal_id_set(
            action_candidate_proposal_ids,
            proposal_id_disambiguations,
        )
    stage_trace["selected_proposal_count"] = len(selected_proposals)
    stage_trace["action_candidate_proposal_count"] = sum(
        1
        for proposal in selected_proposals
        if proposal.id in action_candidate_proposal_ids
    )
    stage_trace["llm_selected_proposal_count"] = (
        len(selected_proposals) - stage_trace["action_candidate_proposal_count"]
    )
    if proposal_id_disambiguations:
        stage_trace["proposal_id_disambiguation_count"] = len(
            proposal_id_disambiguations
        )
        stage_trace["proposal_id_disambiguations"] = proposal_id_disambiguations
    merge_payload = {
        "proposals": [proposal.to_dict() for proposal in selected_proposals],
        "conflicts": merged.conflicts,
        "dropped": list(merged.dropped),
    }
    merge_payload["dropped"].extend(stale_drops)
    failed_subagent_drops: list[dict[str, Any]] = []
    for result in failed_results:
        drop = {
            "proposal_id": result.task_id,
            "reason": result.state,
        }
        metadata = _proposal_validation_error_metadata(result.error)
        if metadata:
            drop["error_code"] = metadata["error_code"]
        failed_subagent_drops.append(drop)
    merge_payload["dropped"].extend(failed_subagent_drops)
    dropped_items = _list_of_dicts(merge_payload.get("dropped"))
    stage_trace["dropped_proposal_count"] = len(dropped_items)
    stage_trace["dropped_proposal_reasons"] = _dropped_reason_counts(dropped_items)
    funnel_report = build_proposal_funnel_report(
        deterministic_scan,
        selected_proposal_ids=[proposal.id for proposal in selected_proposals],
        dropped=dropped_items,
        conflict_groups=merged.conflicts,
    )
    funnel_summary = funnel_report["summary"]
    stage_trace["issue_route_state_counts"] = funnel_summary["route_state_counts"]
    stage_trace["issue_accounting_balanced"] = funnel_summary["accounting_balanced"]
    stage_trace["issue_unrouted_count"] = funnel_summary["unrouted_issue_count"]
    artifact_paths.update(
        write_proposal_funnel_artifacts(
            output_dir,
            deterministic_scan,
            selected_proposal_ids=[proposal.id for proposal in selected_proposals],
            dropped=dropped_items,
            conflict_groups=merged.conflicts,
            task_packs=task_pack_dicts,
            merge_payload=merge_payload,
            subagent_index=subagent_index,
        )
    )

    if failed_results:
        stage_trace["dropped_subagent_count"] = len(failed_results)
        dropped_subagents: list[dict[str, Any]] = []
        for result in failed_results:
            dropped_subagent = {
                "task_id": result.task_id,
                "role": result.role,
                "state": result.state,
                "error": result.error,
            }
            metadata = _proposal_validation_error_metadata(result.error)
            if metadata:
                dropped_subagent["error_code"] = metadata["error_code"]
            dropped_subagents.append(dropped_subagent)
        stage_trace["dropped_subagents"] = dropped_subagents

    if failed_results and not selected_proposals:
        first_failure = failed_results[0]
        merge_report_path = _write_proposal_merge_report(
            output_dir,
            merge_payload,
        )
        artifact_paths["proposal_merge_report"] = merge_report_path
        stage_trace["state"] = first_failure.state
        stage_trace["error"] = first_failure.error
        metadata = _proposal_validation_error_metadata(first_failure.error)
        if metadata:
            stage_trace["error_code"] = metadata["error_code"]
        stage_trace["output_token_estimate"] = _estimate_tokens(
            "\n\n".join(raw_outputs)
        )
        return _ProposeStageResult(
            raw_output="\n\n".join(raw_outputs),
            artifact_paths=artifact_paths,
            stop_reason=first_failure.stop_reason,
        )

    for proposal in selected_proposals:
        validate_proposal(proposal)
    merge_report_path = _write_proposal_merge_report(output_dir, merge_payload)
    artifact_paths["proposal_merge_report"] = merge_report_path
    payload = {
        "proposals": [proposal.to_dict() for proposal in selected_proposals],
        "proposal_task_packs": _proposal_task_pack_summaries(task_pack_dicts),
        "proposal_merge": _proposal_merge_summary(merge_payload),
    }
    return _ProposeStageResult(
        parsed=AgentStageOutput(
            stage="propose",
            payload=payload,
            proposals=selected_proposals,
        ),
        raw_output="\n\n".join(raw_outputs),
        artifact_paths=artifact_paths,
    )


def _run_subagent_tasks(
    *,
    output_dir: Path,
    client: LLMReviewClient,
    config: LLMAgentPipelineConfig,
    profile: str | None,
    context: dict[str, Any],
    previous_outputs: dict[str, Any],
    task_pack_dicts: list[dict[str, Any]],
    max_attempts: int,
    stage_trace: dict[str, Any],
) -> list[_SubagentTaskResult]:
    immediate_results: list[_SubagentTaskResult] = []
    prepared_tasks: list[tuple[dict[str, Any], str, int]] = []
    for task_pack in task_pack_dicts:
        task_pack = _task_pack_for_subagent_llm(task_pack)
        immediate_result = _pre_llm_subagent_result(
            output_dir=output_dir,
            task_pack=task_pack,
            config=config,
        )
        if immediate_result is not None:
            immediate_results.append(immediate_result)
            continue
        user_prompt = _subagent_user_prompt(
            task_pack=task_pack,
            previous_outputs=previous_outputs,
            profile=profile,
            context=context,
            max_proposals_per_task=config.max_subagent_proposals_per_task,
        )
        input_token_estimate = _estimate_tokens(user_prompt)
        if input_token_estimate > config.max_context_tokens_per_stage:
            return [
                _subagent_context_too_large_result(
                    output_dir=output_dir,
                    task_pack=task_pack,
                    input_token_estimate=input_token_estimate,
                    max_context_tokens=config.max_context_tokens_per_stage,
                )
            ]
        prepared_tasks.append((task_pack, user_prompt, input_token_estimate))

    if prepared_tasks or immediate_results:
        safe_package_child_path(output_dir, "subagent_outputs").mkdir(
            parents=True,
            exist_ok=True,
        )

    def run_task(prepared_task: tuple[dict[str, Any], str, int]) -> _SubagentTaskResult:
        task_pack, user_prompt, input_token_estimate = prepared_task
        return _run_subagent_task(
            output_dir=output_dir,
            client=client,
            profile=profile,
            previous_outputs=previous_outputs,
            task_pack=task_pack,
            user_prompt=user_prompt,
            input_token_estimate=input_token_estimate,
            max_attempts=max_attempts,
            config=config,
        )

    fallback_reason = _subagent_parallel_fallback_reason(
        client=client,
        config=config,
        task_count=len(prepared_tasks),
    )
    if fallback_reason:
        stage_trace["subagent_parallel_fallback"] = fallback_reason

    if _should_run_subagents_parallel(
        client=client,
        config=config,
        task_count=len(prepared_tasks),
    ):
        max_workers = min(config.max_parallel_subagents, len(prepared_tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(run_task, prepared_tasks))
    else:
        results = [run_task(prepared_task) for prepared_task in prepared_tasks]
    return sorted([*immediate_results, *results], key=lambda result: result.task_id)


def _task_pack_for_subagent_llm(task_pack: dict[str, Any]) -> dict[str, Any]:
    if str(task_pack.get("execution_mode") or "") != "hybrid":
        return task_pack
    residual_refs = {
        str(ref).strip()
        for ref in task_pack.get("residual_issue_refs", [])
        if str(ref).strip()
    }
    if not residual_refs:
        return task_pack
    llm_task_pack = dict(task_pack)
    llm_task_pack["issues"] = [
        issue
        for issue in _list_of_dicts(task_pack.get("issues"))
        if str(issue.get("issue_ref") or "").strip() in residual_refs
    ]
    llm_task_pack["action_candidates"] = []
    llm_task_pack["llm_residual_only"] = True
    return llm_task_pack


def _pre_llm_subagent_result(
    *,
    output_dir: Path,
    task_pack: dict[str, Any],
    config: LLMAgentPipelineConfig,
) -> _SubagentTaskResult | None:
    execution_mode = str(task_pack.get("execution_mode") or "")
    if execution_mode == "blocked" and config.skip_deterministic_subagent_calls:
        reason = str(task_pack.get("block_reason") or "subagent task is blocked")
        state = _blocked_subagent_state(reason)
        return _non_llm_subagent_result(
            output_dir=output_dir,
            task_pack=task_pack,
            state=state,
            error=reason,
        )
    if (
        execution_mode == "deterministic_only"
        and config.skip_deterministic_subagent_calls
    ):
        return _non_llm_subagent_result(
            output_dir=output_dir,
            task_pack=task_pack,
            state="deterministic_only",
            error="",
        )
    return None


def _blocked_subagent_state(reason: str) -> str:
    normalized = reason.casefold()
    if "rejected" in normalized or "unsafe" in normalized:
        return "blocked_unsafe_candidates"
    if "ground" in normalized or "evidence" in normalized:
        return "blocked_no_grounding"
    return "blocked_no_executable_candidate"


def _non_llm_subagent_result(
    *,
    output_dir: Path,
    task_pack: dict[str, Any],
    state: str,
    error: str,
) -> _SubagentTaskResult:
    task_id = str(task_pack["task_id"])
    role = str(task_pack.get("role", ""))
    attempts: list[dict[str, Any]] = []
    output_path = _write_subagent_output_artifact(
        output_dir,
        task_id=task_id,
        role=role,
        raw_output="",
        parsed=None,
        error=error,
        attempts=attempts,
    )
    return _subagent_task_result(
        output_dir=output_dir,
        task_id=task_id,
        role=role,
        raw_outputs=[],
        artifact_paths={f"subagent_output_{task_id}": output_path},
        output_path=output_path,
        attempts=attempts,
        input_token_estimate=0,
        state=state,
        error=error,
        stop_reason="",
    )


def _should_run_subagents_parallel(
    *,
    client: LLMReviewClient,
    config: LLMAgentPipelineConfig,
    task_count: int,
) -> bool:
    return (
        config.max_parallel_subagents > 1
        and task_count > 1
        and bool(getattr(client, "supports_parallel_subagent_calls", False))
    )


def _subagent_parallel_fallback_reason(
    *,
    client: LLMReviewClient,
    config: LLMAgentPipelineConfig,
    task_count: int,
) -> str:
    if config.max_parallel_subagents <= 1 or task_count <= 1:
        return ""
    if getattr(client, "supports_parallel_subagent_calls", False):
        return ""
    return "client_not_thread_safe"


def _run_subagent_task(
    *,
    output_dir: Path,
    client: LLMReviewClient,
    profile: str | None,
    previous_outputs: dict[str, Any],
    task_pack: dict[str, Any],
    user_prompt: str,
    input_token_estimate: int,
    max_attempts: int,
    config: LLMAgentPipelineConfig,
) -> _SubagentTaskResult:
    task_id = str(task_pack["task_id"])
    role = str(task_pack.get("role", ""))
    base_user_prompt = user_prompt
    attempts: list[dict[str, Any]] = []
    raw_outputs: list[str] = []
    artifact_paths: dict[str, Path] = {}

    for attempt in range(1, max_attempts + 1):
        try:
            raw_output = client.complete(
                system_prompt=_subagent_propose_system_prompt(
                    profile,
                    role=role,
                    max_proposals_per_task=config.max_subagent_proposals_per_task,
                ),
                user_prompt=user_prompt,
            )
        except Exception as exc:
            error = _sanitize_error_text(exc)
            attempts.append(
                _subagent_attempt_record(
                    task_id=task_id,
                    role=role,
                    attempt=attempt,
                    state="client_error",
                    error=error,
                    raw_output="",
                )
            )
            output_path = _write_subagent_output_artifact(
                output_dir,
                task_id=task_id,
                role=role,
                raw_output="",
                parsed=None,
                error=error,
                attempts=attempts,
            )
            artifact_paths[f"subagent_output_{task_id}"] = output_path
            if attempt < max_attempts:
                user_prompt = _client_error_retry_user_prompt(
                    base_user_prompt,
                    error=error,
                    previous_errors=_attempt_record_error_lines(attempts),
                )
                continue
            return _subagent_task_result(
                output_dir=output_dir,
                task_id=task_id,
                role=role,
                raw_outputs=raw_outputs,
                artifact_paths=artifact_paths,
                output_path=output_path,
                attempts=attempts,
                input_token_estimate=input_token_estimate,
                state="client_error",
                error=error,
                stop_reason="llm_client_error",
            )

        raw_outputs.append(raw_output)
        try:
            parsed = _parse_and_validate_propose_output(
                raw_output,
                output_dir=output_dir,
                previous_outputs=previous_outputs,
                task_pack=task_pack,
                strict_role_contracts=config.strict_subagent_role_contracts,
                require_candidate_evidence_allowlist=(
                    config.require_candidate_evidence_allowlist
                ),
            )
            attempts.append(
                _subagent_attempt_record(
                    task_id=task_id,
                    role=role,
                    attempt=attempt,
                    state="completed",
                    error="",
                    raw_output=raw_output,
                )
            )
        except ValueError as exc:
            error = _sanitize_error_text(exc)
            attempts.append(
                _subagent_attempt_record(
                    task_id=task_id,
                    role=role,
                    attempt=attempt,
                    state="invalid_llm_output",
                    error=error,
                    raw_output=raw_output,
                )
            )
            output_path = _write_subagent_output_artifact(
                output_dir,
                task_id=task_id,
                role=role,
                raw_output=raw_output,
                parsed=None,
                error=error,
                attempts=attempts,
            )
            artifact_paths[f"subagent_output_{task_id}"] = output_path
            if attempt < max_attempts:
                user_prompt = _retry_user_prompt(
                    base_user_prompt,
                    stage="propose",
                    error=error,
                    previous_errors=_attempt_record_error_lines(attempts),
                    task_pack=task_pack,
                )
                continue
            return _subagent_task_result(
                output_dir=output_dir,
                task_id=task_id,
                role=role,
                raw_outputs=raw_outputs,
                artifact_paths=artifact_paths,
                output_path=output_path,
                attempts=attempts,
                input_token_estimate=input_token_estimate,
                state="invalid_llm_output",
                error=error,
                stop_reason="invalid_llm_output",
            )

        output_path = _write_subagent_output_artifact(
            output_dir,
            task_id=task_id,
            role=role,
            raw_output=raw_output,
            parsed=parsed,
            error="",
            attempts=attempts,
        )
        artifact_paths[f"subagent_output_{task_id}"] = output_path
        return _subagent_task_result(
            output_dir=output_dir,
            task_id=task_id,
            role=role,
            proposals=parsed.proposals,
            raw_outputs=raw_outputs,
            artifact_paths=artifact_paths,
            output_path=output_path,
            attempts=attempts,
            input_token_estimate=input_token_estimate,
            state="completed",
            error="",
            stop_reason="",
        )

    return _SubagentTaskResult(
        task_id=task_id,
        role=role,
        attempts=max_attempts,
        input_token_estimate=input_token_estimate,
        state="invalid_llm_output",
        error="agent stage output was not parsed",
        stop_reason="invalid_llm_output",
    )


def _subagent_task_result(
    *,
    output_dir: Path,
    task_id: str,
    role: str,
    raw_outputs: list[str],
    artifact_paths: dict[str, Path],
    output_path: Path,
    attempts: list[dict[str, Any]],
    input_token_estimate: int,
    state: str,
    error: str,
    stop_reason: str,
    proposals: list[ImprovementProposal] | None = None,
) -> _SubagentTaskResult:
    sanitized_error = _sanitize_error_text(error)
    metadata = _proposal_validation_error_metadata(sanitized_error)
    index_entry: dict[str, Any] = {
        "task_id": task_id,
        "role": role,
        "proposal_ids": [proposal.id for proposal in proposals or []],
        "path": _relative_artifact_path(output_dir, output_path),
        "state": state,
    }
    if metadata:
        index_entry["error_code"] = metadata["error_code"]
    return _SubagentTaskResult(
        task_id=task_id,
        role=role,
        proposals=proposals or [],
        raw_outputs=raw_outputs,
        artifact_paths=artifact_paths,
        index_entry=index_entry,
        attempt_logs=[
            attempt for attempt in attempts if attempt.get("state") != "completed"
        ],
        attempts=len(attempts),
        input_token_estimate=input_token_estimate,
        state=state,
        error=sanitized_error,
        stop_reason=stop_reason,
    )


def _subagent_context_too_large_result(
    *,
    output_dir: Path,
    task_pack: dict[str, Any],
    input_token_estimate: int,
    max_context_tokens: int,
) -> _SubagentTaskResult:
    task_id = str(task_pack["task_id"])
    role = str(task_pack.get("role", ""))
    error = (
        "subagent prompt input_token_estimate exceeds "
        f"max_context_tokens_per_stage ({input_token_estimate} > "
        f"{max_context_tokens})"
    )
    output_path = _write_subagent_output_artifact(
        output_dir,
        task_id=task_id,
        role=role,
        raw_output="",
        parsed=None,
        error=error,
        attempts=[],
    )
    return _SubagentTaskResult(
        task_id=task_id,
        role=role,
        artifact_paths={f"subagent_output_{task_id}": output_path},
        index_entry={
            "task_id": task_id,
            "role": role,
            "proposal_ids": [],
            "path": _relative_artifact_path(output_dir, output_path),
            "state": "context_too_large",
        },
        attempts=0,
        input_token_estimate=input_token_estimate,
        state="context_too_large",
        error=error,
        stop_reason="context_too_large",
    )


def _attempt_record_error_lines(attempts: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in attempts:
        attempt = item.get("attempt")
        error = item.get("error")
        if isinstance(attempt, int) and isinstance(error, str) and error:
            lines.append(f"Attempt {attempt}: {_sanitize_error_text(error)}")
    return lines


def _parse_and_validate_propose_output(
    raw_output: str,
    *,
    output_dir: Path,
    previous_outputs: dict[str, Any],
    task_pack: dict[str, Any] | None = None,
    strict_role_contracts: bool = True,
    require_candidate_evidence_allowlist: bool = True,
) -> AgentStageOutput:
    allowed_evidence_spans = None
    if task_pack is not None:
        task_pack_allowed_spans = _list_of_dicts(task_pack.get("allowed_evidence_spans"))
        if task_pack_allowed_spans:
            allowed_evidence_spans = task_pack_allowed_spans
    parsed = parse_agent_stage_output(
        "propose",
        raw_output,
        allowed_evidence_spans=allowed_evidence_spans,
    )
    _validate_executable_medical_action_payloads(parsed.proposals)
    for proposal in parsed.proposals:
        validate_proposal(proposal)
    if task_pack is not None and strict_role_contracts:
        _validate_proposals_against_role_contract(parsed.proposals, task_pack=task_pack)
    if task_pack is not None:
        _validate_action_payload_matches_deterministic_candidates(
            parsed.proposals,
            task_pack=task_pack,
        )
        if require_candidate_evidence_allowlist:
            _validate_candidate_expansion_evidence_allowlist(
                parsed.proposals,
                task_pack=task_pack,
            )
    _validate_grounded_proposal_evidence(
        parsed.proposals,
        output_dir=output_dir,
        previous_outputs=previous_outputs,
    )
    return parsed


def _validate_proposals_against_role_contract(
    proposals: list[ImprovementProposal],
    *,
    task_pack: dict[str, Any],
) -> None:
    raw_contract = task_pack.get("role_contract")
    if isinstance(raw_contract, dict) and raw_contract.get("role"):
        contract = role_contract(str(raw_contract["role"]))
    else:
        contract = role_contract(str(task_pack.get("role") or "schema_repair"))
    for proposal in proposals:
        validate_proposal_role_contract(proposal, contract)


def _proposal_task_pack_dicts(task_packs: list[Any]) -> list[dict[str, Any]]:
    role_counts: dict[str, int] = {}
    pack_dicts: list[dict[str, Any]] = []
    for task_pack in task_packs:
        pack_dict = task_pack.to_dict()
        role = str(pack_dict.get("role") or "task")
        role_counts[role] = role_counts.get(role, 0) + 1
        pack_dicts.append(
            {
                "task_id": f"{_task_id_slug(role)}-{role_counts[role]:03d}",
                **pack_dict,
            }
        )
    return pack_dicts


def _fit_proposal_task_packs_to_subagent_budget(
    task_pack_dicts: list[dict[str, Any]],
    *,
    previous_outputs: dict[str, Any],
    context: dict[str, Any],
    profile: str | None,
    max_context_tokens: int,
    max_task_packs: int,
    max_subagent_issues_per_task: int,
    max_subagent_proposals_per_task: int,
    require_candidate_evidence_allowlist: bool,
) -> list[dict[str, Any]]:
    raw_family_totals = _task_pack_raw_issue_family_totals(task_pack_dicts)
    split_packs: list[dict[str, Any]] = []
    for task_pack in task_pack_dicts:
        split_packs.extend(
            _split_task_pack_to_subagent_budget(
                task_pack,
                previous_outputs=previous_outputs,
                context=context,
                profile=profile,
                max_context_tokens=max_context_tokens,
                max_subagent_issues_per_task=max_subagent_issues_per_task,
                max_subagent_proposals_per_task=max_subagent_proposals_per_task,
                require_candidate_evidence_allowlist=(
                    require_candidate_evidence_allowlist
                ),
            )
        )
    selected = _select_task_packs_by_family_round_robin(
        split_packs,
        max_task_packs=max_task_packs,
    )
    selected = _refresh_task_pack_omitted_issue_counts(
        selected,
        raw_family_totals=raw_family_totals,
    )
    return _renumber_task_pack_dicts(selected)


def _select_task_packs_by_family_round_robin(
    task_pack_dicts: list[dict[str, Any]],
    *,
    max_task_packs: int,
) -> list[dict[str, Any]]:
    if max_task_packs <= 0:
        return []

    families: list[str] = []
    packs_by_family: dict[str, list[dict[str, Any]]] = {}
    for task_pack in task_pack_dicts:
        family = _task_pack_issue_family(task_pack)
        if family not in packs_by_family:
            families.append(family)
            packs_by_family[family] = []
        packs_by_family[family].append(task_pack)

    selected: list[dict[str, Any]] = []
    family_positions = {family: 0 for family in families}
    while len(selected) < max_task_packs:
        added = False
        for family in families:
            family_packs = packs_by_family[family]
            start = family_positions[family]
            if start >= len(family_packs):
                continue
            selected.append(family_packs[start])
            family_positions[family] = start + 1
            added = True
            if len(selected) >= max_task_packs:
                break
        if not added:
            break
    return selected


def _task_pack_raw_issue_family_totals(
    task_pack_dicts: list[dict[str, Any]],
) -> dict[str, int]:
    totals: dict[str, int] = {}
    represented: dict[str, int] = {}
    for task_pack in task_pack_dicts:
        family = _task_pack_issue_family(task_pack)
        issue_count = len(_list_of_dicts(task_pack.get("issues")))
        represented[family] = represented.get(family, 0) + issue_count
        totals[family] = max(
            totals.get(family, 0),
            represented[family] + _task_pack_omitted_issue_count(task_pack),
        )
    return totals


def _refresh_task_pack_omitted_issue_counts(
    task_pack_dicts: list[dict[str, Any]],
    *,
    raw_family_totals: dict[str, int],
) -> list[dict[str, Any]]:
    represented: dict[str, int] = {}
    refreshed: list[dict[str, Any]] = []
    for task_pack in task_pack_dicts:
        family = _task_pack_issue_family(task_pack)
        issue_count = len(_list_of_dicts(task_pack.get("issues")))
        represented[family] = represented.get(family, 0) + issue_count
        raw_total = max(raw_family_totals.get(family, 0), represented[family])
        refreshed.append(
            {
                **task_pack,
                "omitted_issue_count": max(0, raw_total - represented[family]),
            }
        )
    return refreshed


def _task_pack_issue_family(task_pack: dict[str, Any]) -> str:
    return str(task_pack.get("issue_family") or "unspecified")


def _task_pack_omitted_issue_count(task_pack: dict[str, Any]) -> int:
    value = task_pack.get("omitted_issue_count")
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return 0


def _split_task_pack_to_subagent_budget(
    task_pack: dict[str, Any],
    *,
    previous_outputs: dict[str, Any],
    context: dict[str, Any],
    profile: str | None,
    max_context_tokens: int,
    max_subagent_issues_per_task: int,
    max_subagent_proposals_per_task: int,
    require_candidate_evidence_allowlist: bool,
) -> list[dict[str, Any]]:
    pending = [dict(task_pack)]
    fitted: list[dict[str, Any]] = []
    while pending:
        candidate = pending.pop(0)
        issues = _list_of_dicts(candidate.get("issues"))
        if len(issues) > max_subagent_issues_per_task:
            right = _task_pack_with_issue_subset(
                candidate,
                issues[max_subagent_issues_per_task:],
                require_candidate_evidence_allowlist=(
                    require_candidate_evidence_allowlist
                ),
            )
            left = _task_pack_with_issue_subset(
                candidate,
                issues[:max_subagent_issues_per_task],
                require_candidate_evidence_allowlist=(
                    require_candidate_evidence_allowlist
                ),
            )
            pending.insert(0, right)
            pending.insert(0, left)
            continue

        if _task_pack_fits_subagent_budget(
            candidate,
            previous_outputs=previous_outputs,
            context=context,
            profile=profile,
            max_context_tokens=max_context_tokens,
            max_subagent_proposals_per_task=max_subagent_proposals_per_task,
        ):
            fitted.append(candidate)
            continue

        if len(issues) > 1:
            midpoint = max(1, len(issues) // 2)
            right = _task_pack_with_issue_subset(
                candidate,
                issues[midpoint:],
                require_candidate_evidence_allowlist=(
                    require_candidate_evidence_allowlist
                ),
            )
            left = _task_pack_with_issue_subset(
                candidate,
                issues[:midpoint],
                require_candidate_evidence_allowlist=(
                    require_candidate_evidence_allowlist
                ),
            )
            pending.insert(0, right)
            pending.insert(0, left)
            continue

        action_candidates = _list_of_dicts(candidate.get("action_candidates"))
        if len(action_candidates) > 1:
            midpoint = max(1, len(action_candidates) // 2)
            right = dict(candidate)
            right["action_candidates"] = action_candidates[midpoint:]
            _refresh_task_pack_execution_mode(
                right,
                require_candidate_evidence_allowlist=(
                    require_candidate_evidence_allowlist
                ),
            )
            left = dict(candidate)
            left["action_candidates"] = action_candidates[:midpoint]
            _refresh_task_pack_execution_mode(
                left,
                require_candidate_evidence_allowlist=(
                    require_candidate_evidence_allowlist
                ),
            )
            pending.insert(0, right)
            pending.insert(0, left)
            continue

        fitted.append(candidate)
    return fitted


def _task_pack_fits_subagent_budget(
    task_pack: dict[str, Any],
    *,
    previous_outputs: dict[str, Any],
    context: dict[str, Any],
    profile: str | None,
    max_context_tokens: int,
    max_subagent_proposals_per_task: int,
) -> bool:
    user_prompt = _subagent_user_prompt(
        task_pack=task_pack,
        previous_outputs=previous_outputs,
        profile=profile,
        context=context,
        max_proposals_per_task=max_subagent_proposals_per_task,
    )
    return _estimate_tokens(user_prompt) <= max_context_tokens


def _task_pack_with_issue_subset(
    task_pack: dict[str, Any],
    issues: list[dict[str, Any]],
    *,
    require_candidate_evidence_allowlist: bool,
) -> dict[str, Any]:
    subset = dict(task_pack)
    original_issue_count = len(_list_of_dicts(task_pack.get("issues")))
    subset["issues"] = issues
    subset["action_candidates"] = _action_candidates_for_issue_subset(
        _list_of_dicts(task_pack.get("action_candidates")),
        issues,
    )
    subset["rejected_action_candidates"] = _action_candidates_for_issue_subset(
        _list_of_dicts(task_pack.get("rejected_action_candidates")),
        issues,
    )
    covered_issue_refs = _candidate_covered_issue_refs(
        _list_of_dicts(subset.get("action_candidates"))
    )
    subset["covered_issue_refs"] = covered_issue_refs
    subset["residual_issue_refs"] = _residual_issue_refs(
        issues,
        covered_issue_refs=covered_issue_refs,
    )
    subset["allowed_evidence_spans"] = _allowed_evidence_spans_for_issue_subset(
        _list_of_dicts(task_pack.get("allowed_evidence_spans")),
        issues,
    )
    _refresh_task_pack_execution_mode(
        subset,
        require_candidate_evidence_allowlist=require_candidate_evidence_allowlist,
    )
    if original_issue_count > len(issues):
        subset["split"] = {
            "parent_task_id": task_pack.get("task_id", ""),
            "parent_issue_count": original_issue_count,
            "issue_count": len(issues),
        }
    return subset


def _refresh_task_pack_execution_mode(
    task_pack: dict[str, Any],
    *,
    require_candidate_evidence_allowlist: bool,
) -> None:
    action_candidates = _list_of_dicts(task_pack.get("action_candidates"))
    rejected_candidates = _list_of_dicts(task_pack.get("rejected_action_candidates"))
    allowed_evidence_spans = _list_of_dicts(task_pack.get("allowed_evidence_spans"))
    contract = role_contract(str(task_pack.get("role") or "schema_repair"))
    residual_issue_refs = [
        str(ref).strip()
        for ref in task_pack.get("residual_issue_refs", [])
        if str(ref).strip()
    ]
    if action_candidates and not residual_issue_refs:
        task_pack["execution_mode"] = "deterministic_only"
        task_pack["block_reason"] = ""
        return
    if action_candidates and residual_issue_refs:
        task_pack["execution_mode"] = "hybrid"
        task_pack["block_reason"] = ""
        return
    if contract.payload_mode == "grounded_expansion":
        if require_candidate_evidence_allowlist and not allowed_evidence_spans:
            task_pack["execution_mode"] = "blocked"
            task_pack["block_reason"] = "no grounded evidence allowlist"
        else:
            task_pack["execution_mode"] = "grounded_expansion"
            task_pack["block_reason"] = ""
        return
    if contract.require_action_candidate:
        task_pack["execution_mode"] = "blocked"
        task_pack["block_reason"] = (
            "all deterministic action candidates were rejected"
            if rejected_candidates
            else "no executable action candidate"
        )
        return
    task_pack["execution_mode"] = "llm_assisted"
    task_pack["block_reason"] = ""


def _action_candidates_for_issue_subset(
    action_candidates: list[dict[str, Any]], issues: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not action_candidates:
        return []
    issue_refs = _issue_action_reference_values(issues)
    if not issue_refs:
        return []
    return [
        candidate
        for candidate in action_candidates
        if _action_candidate_matches_issue_refs(candidate, issue_refs)
    ]


def _candidate_covered_issue_refs(candidates: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(candidate.get("issue_ref") or "").strip()
            for candidate in candidates
            if str(candidate.get("issue_ref") or "").strip()
        }
    )


def _residual_issue_refs(
    issues: list[dict[str, Any]],
    *,
    covered_issue_refs: list[str],
) -> list[str]:
    covered = set(covered_issue_refs)
    return [
        issue_ref
        for issue in issues
        if (issue_ref := str(issue.get("issue_ref") or "").strip())
        and issue_ref not in covered
    ]


def _allowed_evidence_spans_for_issue_subset(
    allowed_evidence_spans: list[dict[str, Any]], issues: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not allowed_evidence_spans:
        return []
    issue_evidence_tuples = _issue_evidence_span_tuples(issues)
    if issue_evidence_tuples:
        return [
            span
            for span in allowed_evidence_spans
            if _evidence_span_tuple(span) in issue_evidence_tuples
        ]

    issue_refs = _issue_reference_values(issues)
    if not issue_refs:
        return []
    spans: list[dict[str, Any]] = []
    for span in allowed_evidence_spans:
        source_id = str(span.get("source_id") or "").strip()
        file_path = str(span.get("file_path") or "").strip()
        if source_id in issue_refs or file_path in issue_refs:
            spans.append(span)
    return spans


def _issue_evidence_span_tuples(
    issues: list[dict[str, Any]],
) -> set[tuple[str, str, str]]:
    tuples: set[tuple[str, str, str]] = set()
    for issue in issues:
        for span in _list_of_dicts(issue.get("evidence_spans")):
            evidence_tuple = _evidence_span_tuple(span)
            if all(evidence_tuple):
                tuples.add(evidence_tuple)
    return tuples


def _evidence_span_tuple(span: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(span.get("source_id") or "").strip(),
        str(span.get("file_path") or "").strip(),
        str(span.get("evidence_quote") or "").strip(),
    )


def _issue_reference_values(issues: list[dict[str, Any]]) -> set[str]:
    refs: set[str] = set()
    for issue in issues:
        for key in (
            "edge_id",
            "relation_id",
            "node_id",
            "value_node_id",
            "source_id",
            "file_path",
            "source",
            "target",
            "new_source",
            "new_target",
        ):
            value = issue.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    refs.add(text)
                    if text.startswith("edge:"):
                        refs.add(text.removeprefix("edge:"))
        for span in _list_of_dicts(issue.get("evidence_spans")):
            for key in ("source_id", "file_path", "evidence_quote"):
                value = span.get(key)
                if value is not None:
                    text = str(value).strip()
                    if text:
                        refs.add(text)
    return refs


def _issue_action_reference_values(issues: list[dict[str, Any]]) -> set[str]:
    refs: set[str] = set()
    for issue in issues:
        for key in ("issue_ref", "edge_id", "relation_id", "node_id", "value_node_id"):
            value = issue.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    refs.add(text)
                    if text.startswith("edge:"):
                        refs.add(text.removeprefix("edge:"))
    return refs


def _action_candidate_matches_issue_refs(
    candidate: dict[str, Any], refs: set[str]
) -> bool:
    issue_ref = str(candidate.get("issue_ref") or "").strip()
    if issue_ref and issue_ref in refs:
        return True
    target = str(candidate.get("target", "")).strip()
    if target in refs or target.removeprefix("edge:") in refs:
        return True
    payload = candidate.get("action_payload")
    if isinstance(payload, dict):
        for key in (
            "edge_id",
            "expected_source",
            "expected_target",
            "new_source",
            "new_target",
        ):
            value = payload.get(key)
            if value is not None and str(value).strip() in refs:
                return True
    return False


def _renumber_task_pack_dicts(
    task_pack_dicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    role_counts: dict[str, int] = {}
    renumbered: list[dict[str, Any]] = []
    for task_pack in task_pack_dicts:
        role = str(task_pack.get("role") or "task")
        role_counts[role] = role_counts.get(role, 0) + 1
        renumbered.append(
            {
                **task_pack,
                "task_id": f"{_task_id_slug(role)}-{role_counts[role]:03d}",
            }
        )
    return renumbered


def _task_pack_issue_count(task_pack_dicts: list[dict[str, Any]]) -> int:
    return sum(
        len(_list_of_dicts(task_pack.get("issues")))
        for task_pack in task_pack_dicts
    )


def _task_pack_detected_issue_count(task_pack_dicts: list[dict[str, Any]]) -> int:
    return sum(_task_pack_raw_issue_family_totals(task_pack_dicts).values())


def _task_pack_issue_family_counts(
    task_packs: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pack in task_packs:
        family = str(pack.get("issue_family") or "unspecified")
        counts[family] = counts.get(family, 0) + len(
            _list_of_dicts(pack.get("issues"))
        )
    return counts


def _task_pack_action_candidate_count(
    task_packs: list[dict[str, Any]],
) -> int:
    return sum(
        len(_list_of_dicts(task_pack.get("action_candidates")))
        for task_pack in task_packs
    )


def _task_pack_omitted_family_counts(
    task_packs: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pack in task_packs:
        family = _task_pack_issue_family(pack)
        counts[family] = _task_pack_omitted_issue_count(pack)
    return {family: omitted for family, omitted in counts.items() if omitted}


def _proposal_task_pack_summaries(
    task_pack_dicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task_pack.get("task_id", ""),
            "role": task_pack.get("role", ""),
            "issue_count": len(_list_of_dicts(task_pack.get("issues"))),
            "action_candidate_count": len(
                _list_of_dicts(task_pack.get("action_candidates"))
            ),
        }
        for task_pack in task_pack_dicts
    ]


def _deterministic_scan_action_task_packs(
    scan: DeterministicScanResult,
) -> list[dict[str, Any]]:
    if not scan.candidates:
        return []
    issue_refs = {
        str(candidate.get("issue_ref") or "").strip()
        for candidate in scan.candidates
        if str(candidate.get("issue_ref") or "").strip()
    }
    issues = [
        issue
        for issue in scan.issues
        if str(issue.get("issue_ref") or "").strip() in issue_refs
    ]
    contract = role_contract("schema_repair")
    return [
        {
            "task_id": "deterministic-scan-candidates",
            "role": "schema_repair",
            "issues": issues,
            "action_candidates": scan.candidates,
            "rejected_action_candidates": [],
            "covered_issue_refs": sorted(issue_refs),
            "residual_issue_refs": [],
            "role_contract": contract.to_dict(),
            "execution_mode": "deterministic_only",
            "issue_family": "deterministic_scan",
        }
    ]


def _action_candidate_proposals_from_task_packs(
    task_pack_dicts: list[dict[str, Any]],
    *,
    output_dir: Path,
    previous_outputs: dict[str, Any],
) -> ActionCandidateProposalBuildResult:
    proposals: list[ImprovementProposal] = []
    rejected: list[dict[str, Any]] = []
    for task_pack in task_pack_dicts:
        issues = _list_of_dicts(task_pack.get("issues"))
        for candidate in _list_of_dicts(task_pack.get("action_candidates")):
            try:
                proposal = _action_candidate_proposal(candidate, issues=issues)
                if proposal is None:
                    raise ValueError("candidate could not be converted")
                _validate_grounded_proposal_evidence(
                    [proposal],
                    output_dir=output_dir,
                    previous_outputs=previous_outputs,
                )
            except ValueError as exc:
                _route_state, error_code = _candidate_to_proposal_failure_route(
                    str(exc)
                )
                rejected.append(
                    _candidate_conversion_rejection(
                        candidate,
                        issue_ref=str(candidate.get("issue_ref") or ""),
                        issues=issues,
                        error_code=error_code,
                        error=str(exc),
                        fallback_issue_family=task_pack.get("issue_family"),
                    )
                )
                continue
            proposals.append(proposal)
    return ActionCandidateProposalBuildResult(
        proposals=proposals,
        rejected=rejected,
    )


def action_candidate_proposals_from_scan(
    scan: DeterministicScanResult,
    *,
    output_dir: Path,
    previous_outputs: dict[str, Any],
) -> ActionCandidateProposalBuildResult:
    proposals: list[ImprovementProposal] = []
    rejected: list[dict[str, Any]] = []
    routes_by_ref = {
        str(route.issue_ref or "").strip(): route
        for route in scan.issue_routes
        if str(route.issue_ref or "").strip()
    }
    issues = _list_of_dicts(scan.issues)

    for candidate in _list_of_dicts(scan.candidates):
        issue_ref = str(candidate.get("issue_ref") or "").strip()
        route = routes_by_ref.get(issue_ref)
        try:
            proposal = _action_candidate_proposal(candidate, issues=issues)
            if proposal is None:
                raise ValueError(_malformed_action_candidate_error(candidate))
            _validate_grounded_proposal_evidence(
                [proposal],
                output_dir=output_dir,
                previous_outputs=previous_outputs,
            )
        except ValueError as exc:
            route_state, error_code = _candidate_to_proposal_failure_route(
                str(exc)
            )
            rejection = {
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "issue_ref": issue_ref,
                "stage": "proposal_conversion",
                "error_code": error_code,
                "error": str(exc),
            }
            issue_family = _action_candidate_issue_family(
                candidate,
                issues=issues,
                fallback_issue_family=route.family if route is not None else "",
            )
            if issue_family:
                rejection["issue_family"] = issue_family
            rejected.append(rejection)
            _append_scan_rejection_once(scan, rejection)
            if route is not None:
                route.set_state(
                    route_state,
                    reason_code=error_code,
                    reason=str(exc),
                    generation_disposition=_candidate_to_proposal_failure_disposition(
                        error_code
                    ),
                )
                route.proposal_ids.clear()
            continue

        proposals.append(proposal)
        if route is not None:
            route.set_state("deterministic_covered")
            if proposal.id not in route.proposal_ids:
                route.proposal_ids.append(proposal.id)

    return ActionCandidateProposalBuildResult(
        proposals=proposals,
        rejected=rejected,
    )


def _candidate_to_proposal_failure_route(error: str) -> tuple[str, str]:
    normalized = error.casefold()
    if (
        "candidate could not be converted" in normalized
        or "missing action_payload" in normalized
        or "required candidate" in normalized
    ):
        return "blocked_safety", "CONVERSION_FAILED"
    if "evidence" in normalized or "grounded" in normalized:
        return "blocked_evidence", "CANDIDATE_EVIDENCE_INVALID"
    if "apply" in normalized or "capability" in normalized or "unsupported" in normalized:
        return "blocked_apply", "CANDIDATE_APPLY_UNSUPPORTED"
    if (
        "validation" in normalized
        or "safety" in normalized
        or "no-op" in normalized
        or "noop" in normalized
        or "domain-range" in normalized
        or "unsafe" in normalized
    ):
        return "blocked_safety", "CANDIDATE_VALIDATION_FAILED"
    return "blocked_safety", "CANDIDATE_TO_PROPOSAL_FAILED"


def _candidate_to_proposal_failure_disposition(error_code: str) -> str:
    if error_code in {
        "CONVERSION_FAILED",
        "CANDIDATE_MALFORMED",
        "CANDIDATE_TO_PROPOSAL_FAILED",
    }:
        return "conversion_failed"
    if error_code == "CANDIDATE_EVIDENCE_INVALID":
        return "blocked_evidence"
    if error_code == "CANDIDATE_APPLY_UNSUPPORTED":
        return "blocked_apply"
    return "blocked_safety"


def _malformed_action_candidate_error(candidate: dict[str, Any]) -> str:
    if not isinstance(candidate.get("action_payload"), dict):
        return "candidate could not be converted: missing action_payload"
    return "candidate could not be converted: missing required candidate fields"


def _candidate_conversion_rejection(
    candidate: dict[str, Any],
    *,
    issue_ref: str,
    issues: list[dict[str, Any]],
    error_code: str,
    error: str,
    fallback_issue_family: object = "",
) -> dict[str, Any]:
    rejection = {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "issue_ref": issue_ref,
        "stage": "proposal_conversion",
        "error_code": error_code,
        "error": error,
    }
    issue_family = _action_candidate_issue_family(
        candidate,
        issues=issues,
        fallback_issue_family=fallback_issue_family,
    )
    if issue_family:
        rejection["issue_family"] = issue_family
    return rejection


def _action_candidate_issue_family(
    candidate: dict[str, Any],
    *,
    issues: list[dict[str, Any]],
    fallback_issue_family: object = "",
) -> str:
    for value in (
        candidate.get("issue_family"),
        _issue_for_action_candidate(candidate, issues).get("issue_family"),
        fallback_issue_family,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _append_scan_rejection_once(
    scan: DeterministicScanResult,
    rejection: dict[str, Any],
) -> None:
    rejection_key = _scan_rejection_key(rejection)
    if any(_scan_rejection_key(existing) == rejection_key for existing in scan.rejections):
        return
    scan.rejections.append(rejection)


def _scan_rejection_key(rejection: dict[str, Any]) -> tuple[str, str, str, str, str]:
    issue_refs = rejection.get("issue_refs")
    if isinstance(issue_refs, list):
        issue_ref_key = "|".join(sorted(str(ref).strip() for ref in issue_refs))
    else:
        issue_ref_key = str(rejection.get("issue_ref") or "").strip()
    return (
        str(rejection.get("candidate_id") or "").strip(),
        issue_ref_key,
        str(rejection.get("stage") or "").strip(),
        str(rejection.get("error_code") or "").strip(),
        str(rejection.get("error") or "").strip(),
    )


def _action_candidate_proposal(
    candidate: dict[str, Any],
    *,
    issues: list[dict[str, Any]],
) -> ImprovementProposal | None:
    proposal_type = str(candidate.get("proposal_type") or "").strip()
    target = str(candidate.get("target") or "").strip()
    action_payload = candidate.get("action_payload")
    if not proposal_type or not target or not isinstance(action_payload, dict):
        return None

    issue = _issue_for_action_candidate(candidate, issues)
    issue_kind = str(candidate.get("issue_kind") or issue.get("issue_kind") or "")
    edge_id = str(action_payload.get("edge_id") or "").strip()
    proposal = ImprovementProposal(
        id=_action_candidate_proposal_id(candidate),
        type=proposal_type,
        target=target,
        proposed_change=_action_candidate_proposed_change(
            action_payload=action_payload,
            issue_kind=issue_kind,
            target=target,
        ),
        reason=_action_candidate_reason(issue=issue, issue_kind=issue_kind),
        evidence=_action_candidate_evidence(issue=issue, edge_id=edge_id),
        confidence=0.9,
        risk=_action_candidate_risk(issue),
        requires_approval=True,
        expected_metric_change={
            "medical_schema_issue_count": -1,
            "relation_semantic_issue_count": -1,
        },
        action_payload=dict(action_payload),
    )
    validate_proposal(proposal)
    return proposal


def _issue_for_action_candidate(
    candidate: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    issue_ref = str(candidate.get("issue_ref") or "").strip()
    if issue_ref:
        for issue in issues:
            if str(issue.get("issue_ref") or "").strip() == issue_ref:
                return issue
    action_payload = candidate.get("action_payload")
    edge_id = ""
    if isinstance(action_payload, dict):
        edge_id = str(action_payload.get("edge_id") or "").strip()
    issue_kind = str(candidate.get("issue_kind") or "").strip()
    for issue in issues:
        if edge_id and str(issue.get("edge_id") or "").strip() != edge_id:
            continue
        if issue_kind and str(issue.get("issue_kind") or "").strip() != issue_kind:
            continue
        return issue
    for issue in issues:
        if edge_id and str(issue.get("edge_id") or "").strip() == edge_id:
            return issue
    return {}


def _action_candidate_proposal_id(candidate: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        _stable_json_key(
            {
                "proposal_type": candidate.get("proposal_type"),
                "target": candidate.get("target"),
                "action_payload": candidate.get("action_payload"),
            }
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"prop-action-candidate-{digest}"


def _disambiguate_proposal_ids_used_by_decision_memory(
    proposals: list[ImprovementProposal],
    *,
    output_dir: Path,
) -> tuple[list[ImprovementProposal], list[dict[str, str]]]:
    reserved_ids = _proposal_ids_from_decision_memory(output_dir)
    if not proposals:
        return [], []

    seen_ids: set[str] = set()
    disambiguated: list[ImprovementProposal] = []
    changes: list[dict[str, str]] = []
    for proposal in proposals:
        proposal_id = proposal.id
        if proposal_id in reserved_ids or proposal_id in seen_ids:
            new_id = _disambiguated_proposal_id(
                proposal,
                reserved_ids=reserved_ids | seen_ids,
            )
            changes.append(
                {
                    "old_id": proposal_id,
                    "new_id": new_id,
                    "reason": (
                        "proposal_id already exists in accepted/rejected/deferred "
                        "decision memory"
                    ),
                }
            )
            proposal = _proposal_with_id(proposal, new_id)
            proposal_id = new_id
        seen_ids.add(proposal_id)
        disambiguated.append(proposal)
    return disambiguated, changes


def _sync_issue_route_proposal_ids(
    scan: DeterministicScanResult,
    proposal_id_disambiguations: list[dict[str, str]],
) -> None:
    id_map = {
        str(change.get("old_id") or "").strip(): str(change.get("new_id") or "").strip()
        for change in proposal_id_disambiguations
        if str(change.get("old_id") or "").strip()
        and str(change.get("new_id") or "").strip()
    }
    if not id_map:
        return
    for route in scan.issue_routes:
        route.proposal_ids = _deduped_proposal_ids(
            id_map.get(proposal_id, proposal_id)
            for proposal_id in route.proposal_ids
        )


def _remapped_proposal_id_set(
    proposal_ids: set[str],
    proposal_id_disambiguations: list[dict[str, str]],
) -> set[str]:
    id_map = {
        str(change.get("old_id") or "").strip(): str(change.get("new_id") or "").strip()
        for change in proposal_id_disambiguations
        if str(change.get("old_id") or "").strip()
        and str(change.get("new_id") or "").strip()
    }
    if not id_map:
        return proposal_ids
    return {id_map.get(proposal_id, proposal_id) for proposal_id in proposal_ids}


def _deduped_proposal_ids(proposal_ids: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for proposal_id in proposal_ids:
        proposal_id = str(proposal_id or "").strip()
        if not proposal_id or proposal_id in seen:
            continue
        seen.add(proposal_id)
        deduped.append(proposal_id)
    return deduped


def _drop_stale_relation_replacement_proposals(
    proposals: list[ImprovementProposal],
    *,
    output_dir: Path,
) -> tuple[list[ImprovementProposal], list[dict[str, str]]]:
    edges = _snapshot_edges_for_stale_proposal_guard(output_dir)
    if not edges:
        return proposals, []

    kept: list[ImprovementProposal] = []
    dropped: list[dict[str, str]] = []
    for proposal in proposals:
        reason = _stale_relation_replacement_reason(proposal, edges)
        if reason:
            dropped.append(
                {
                    "proposal_id": proposal.id,
                    "reason": reason,
                }
            )
            continue
        kept.append(proposal)
    return kept, dropped


def _snapshot_edges_for_stale_proposal_guard(output_dir: Path) -> list[dict[str, Any]]:
    snapshot_path = output_dir / "snapshots" / "kg_snapshot.json"
    if not snapshot_path.exists() or not snapshot_path.is_file():
        return []
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    edges = snapshot.get("edges") if isinstance(snapshot, dict) else None
    return [edge for edge in edges if isinstance(edge, dict)] if isinstance(edges, list) else []


def _stale_relation_replacement_reason(
    proposal: ImprovementProposal,
    edges: list[dict[str, Any]],
) -> str:
    if proposal.type != "medical_relation_schema_migration":
        return ""
    payload = proposal.action_payload
    if payload.get("action") != "replace_relation":
        return ""

    expected_source = str(payload.get("expected_source") or "").strip()
    expected_target = str(payload.get("expected_target") or "").strip()
    edge_id = str(payload.get("edge_id") or "").strip()
    current_keywords = str(payload.get("current_keywords") or "").strip()
    new_source = str(payload.get("new_source") or "").strip()
    new_target = str(payload.get("new_target") or "").strip()
    new_keywords = str(payload.get("new_keywords") or "").strip()
    if not (
        expected_source
        and expected_target
        and current_keywords
        and new_source
        and new_target
        and new_keywords
    ):
        return ""

    current_edge = _snapshot_edge_for_relation_replacement(
        edges,
        source=expected_source,
        target=expected_target,
        edge_id=edge_id,
    )
    if current_edge is not None:
        actual_keywords = str(current_edge.get("keywords") or "").strip()
        if actual_keywords and actual_keywords != current_keywords:
            return "stale_current_keywords"
        return ""

    replacement_edge = _snapshot_edge_for_relation_replacement(
        edges,
        source=new_source,
        target=new_target,
        edge_id="",
    )
    replacement_keywords = (
        str(replacement_edge.get("keywords") or "").strip()
        if replacement_edge is not None
        else ""
    )
    if replacement_keywords == new_keywords:
        return "relation_replacement_already_present"
    return ""


def _snapshot_edge_for_relation_replacement(
    edges: list[dict[str, Any]],
    *,
    source: str,
    target: str,
    edge_id: str,
) -> dict[str, Any] | None:
    for edge in edges:
        if (
            str(edge.get("source") or "").strip() == source
            and str(edge.get("target") or "").strip() == target
        ):
            return edge
    if edge_id:
        for edge in edges:
            if str(edge.get("id") or "").strip() == edge_id:
                return edge
    return None


def _proposal_ids_from_decision_memory(output_dir: Path) -> set[str]:
    proposal_ids: set[str] = set()
    for filename in _PROPOSAL_DECISION_MEMORY_FILES:
        path = output_dir / filename
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        proposal_ids.update(_PROPOSAL_ID_HEADING_RE.findall(text))
        proposal_ids.update(_PROPOSAL_ID_JSON_FIELD_RE.findall(text))
    return proposal_ids


def _disambiguated_proposal_id(
    proposal: ImprovementProposal,
    *,
    reserved_ids: set[str],
) -> str:
    digest_source = {
        "id": proposal.id,
        "type": proposal.type,
        "target": proposal.target,
        "proposed_change": proposal.proposed_change,
        "reason": proposal.reason,
        "evidence": proposal.evidence,
        "action_payload": proposal.action_payload,
    }
    digest = hashlib.sha256(
        _stable_json_key(digest_source).encode("utf-8")
    ).hexdigest()
    for length in (12, 16, 24, 32):
        candidate = f"{proposal.id}-{digest[:length]}"
        if candidate not in reserved_ids:
            return candidate
    counter = 2
    while True:
        candidate = f"{proposal.id}-{digest[:32]}-{counter}"
        if candidate not in reserved_ids:
            return candidate
        counter += 1


def _proposal_with_id(
    proposal: ImprovementProposal,
    proposal_id: str,
) -> ImprovementProposal:
    judge = dict(proposal.judge)
    if "proposal_id" in judge:
        judge["proposal_id"] = proposal_id
    return replace(proposal, id=proposal_id, judge=judge)


def _action_candidate_proposed_change(
    *,
    action_payload: dict[str, Any],
    issue_kind: str,
    target: str,
) -> str:
    action = str(action_payload.get("action") or "review").strip()
    edge_id = str(action_payload.get("edge_id") or target).strip()
    new_source = str(action_payload.get("new_source") or "").strip()
    new_target = str(action_payload.get("new_target") or "").strip()
    new_keywords = str(action_payload.get("new_keywords") or "").strip()
    if action == "replace_relation" and new_source and new_target and new_keywords:
        return (
            "Queue deterministic medical relation repair "
            f"for {edge_id}: {new_source} -> {new_keywords} -> {new_target}."
        )
    if action == "split_relation":
        new_edges = action_payload.get("new_edges")
        if isinstance(new_edges, list) and new_edges:
            predicates = [
                str(edge.get("predicate"))
                for edge in new_edges
                if isinstance(edge, dict) and edge.get("predicate")
            ]
            return (
                "Queue deterministic medical relation split "
                f"for {edge_id}: {', '.join(str(item) for item in predicates)}."
            )
    if issue_kind:
        return f"Queue deterministic medical KG repair for {issue_kind} on {edge_id}."
    return f"Queue deterministic medical KG repair for {edge_id}."


def _action_candidate_reason(
    *,
    issue: dict[str, Any],
    issue_kind: str,
) -> str:
    guidance = str(issue.get("guidance") or "").strip()
    if guidance:
        return (
            "The deterministic medical schema detector produced this bounded "
            f"action candidate for {issue_kind or 'a schema issue'}: {guidance}"
        )
    return (
        "The deterministic medical schema detector produced this bounded action "
        f"candidate for {issue_kind or 'a schema issue'}."
    )


def _action_candidate_evidence(issue: dict[str, Any], *, edge_id: str) -> list[str]:
    parts: list[str] = []
    source_id = _first_reference_value(issue.get("source_id"))
    file_path = _first_reference_value(issue.get("file_path"))
    if source_id:
        parts.append(f"source_id: {source_id}")
    if file_path:
        parts.append(f"file_path: {file_path}")
    if edge_id:
        parts.append(f"relation_id: {edge_id}")
    return ["; ".join(parts)] if parts else []


def _first_reference_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    for separator in (GRAPH_FIELD_SEP, "|", "\n", "\r"):
        text = text.replace(separator, "\n")
    for part in text.splitlines():
        stripped = part.strip()
        if stripped:
            return stripped
    return ""


def _action_candidate_risk(issue: dict[str, Any]) -> str:
    risk = str(issue.get("risk") or "").strip().casefold()
    return risk if risk in {"low", "medium", "high"} else "medium"


def _proposal_merge_summary(merge_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_proposal_ids": [
            str(proposal.get("id", ""))
            for proposal in _list_of_dicts(merge_payload.get("proposals"))
        ],
        "conflicts": merge_payload.get("conflicts", []),
        "dropped": merge_payload.get("dropped", []),
    }


def _dropped_reason_counts(dropped: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in dropped:
        reason = str(item.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _task_id_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-_")
    return slug or "task"


def _subagent_user_prompt(
    *,
    task_pack: dict[str, Any],
    previous_outputs: dict[str, Any],
    profile: str | None,
    context: dict[str, Any],
    max_proposals_per_task: int,
) -> str:
    payload = {
        "task_id": task_pack["task_id"],
        "profile": profile or "default",
        "task_pack": _subagent_prompt_task_pack(task_pack),
        "previous_outputs": _compact_subagent_previous_outputs(
            previous_outputs,
            task_pack=task_pack,
        ),
        "stage_context": _compact_subagent_stage_context(
            context,
            task_pack=task_pack,
        ),
        "safety_instructions": [
            "Return only propose-stage JSON with a proposals array.",
            f"Return at most {max_proposals_per_task} proposals for this task.",
            "Use deterministic source_id, file_path, entity_id, relation_id, item_id, or metric references only.",
            "Do not treat earlier LLM outputs as medical evidence.",
            "Executable medical KG proposal types require grounded action_payload values.",
            "Keep requires_approval=true for any mutation proposal.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _subagent_prompt_task_pack(task_pack: dict[str, Any]) -> dict[str, Any]:
    prompt_task_pack = dict(task_pack)
    contract = prompt_task_pack.get("role_contract")
    if isinstance(contract, dict):
        prompt_contract = dict(contract)
        prompt_contract.pop("retry_contract", None)
        prompt_contract.pop("retry_error_codes", None)
        prompt_contract.pop("evidence_tuple_fields", None)
        prompt_task_pack["role_contract"] = prompt_contract
    return prompt_task_pack


def _compact_subagent_previous_outputs(
    previous_outputs: dict[str, Any], *, task_pack: dict[str, Any]
) -> dict[str, Any]:
    refs = _task_pack_reference_values(task_pack)
    compact: dict[str, Any] = {"summary": _compact_output_summary(previous_outputs)}
    for key, value in previous_outputs.items():
        if isinstance(value, list):
            items = _relevant_or_first_items(value, refs, MAX_SUBAGENT_COMPACT_LIST_ITEMS)
            compact[str(key)] = _compact_subagent_value(items)
        elif key in {"issue_explanations", "missing_branches", "evidence_map"}:
            compact[str(key)] = _compact_subagent_value(value)
    return compact


def _compact_subagent_stage_context(
    context: dict[str, Any], *, task_pack: dict[str, Any]
) -> dict[str, Any]:
    refs = _task_pack_reference_values(task_pack)
    compact: dict[str, Any] = {}
    for key in (
        "workspace",
        "stage",
        "hierarchy_branches",
        "medical_schema_issues_summary",
        "entity_cleanup_issues_summary",
        "proposal_revision_requests",
    ):
        if key in context:
            compact[key] = _compact_subagent_value(context[key])

    for key in ("quality_findings", "candidate_entities", "candidate_relations", "evidence_windows"):
        value = context.get(key)
        if isinstance(value, list):
            compact[key] = _compact_subagent_value(
                _relevant_or_first_items(value, refs, MAX_SUBAGENT_COMPACT_LIST_ITEMS)
            )

    rules_memory = context.get("rules_memory")
    if isinstance(rules_memory, dict):
        compact["rules_memory"] = {
            str(key): _clip_subagent_text(str(value), MAX_SUBAGENT_RULES_MEMORY_CHARS)
            for key, value in list(rules_memory.items())[:MAX_SUBAGENT_COMPACT_DICT_ITEMS]
        }
    return compact


def _compact_output_summary(previous_outputs: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in previous_outputs.items():
        if isinstance(value, list):
            summary[str(key)] = {"item_count": len(value)}
        elif isinstance(value, dict):
            summary[str(key)] = {"key_count": len(value)}
        elif isinstance(value, str):
            summary[str(key)] = {"chars": len(value)}
    return summary


def _compact_auxiliary_stage_context(
    context: dict[str, Any],
    *,
    stage: str,
    proposals: list[ImprovementProposal],
    previous_outputs: dict[str, Any],
    max_context_tokens: int | None = None,
) -> dict[str, Any]:
    included_proposals = _select_auxiliary_proposals(proposals)
    compact = _build_auxiliary_stage_context(
        context,
        stage=stage,
        proposals=proposals,
        previous_outputs=previous_outputs,
        included_proposals=included_proposals,
    )
    if max_context_tokens is None:
        return compact

    while len(included_proposals) > 1:
        prompt = json.dumps(compact, ensure_ascii=False, sort_keys=True)
        if _estimate_tokens(prompt) <= max_context_tokens:
            return compact
        included_proposals = included_proposals[:-1]
        compact = _build_auxiliary_stage_context(
            context,
            stage=stage,
            proposals=proposals,
            previous_outputs=previous_outputs,
            included_proposals=included_proposals,
            dynamically_reduced=True,
            max_context_tokens=max_context_tokens,
        )
    return compact


def _build_auxiliary_stage_context(
    context: dict[str, Any],
    *,
    stage: str,
    proposals: list[ImprovementProposal],
    previous_outputs: dict[str, Any],
    included_proposals: list[ImprovementProposal],
    dynamically_reduced: bool = False,
    max_context_tokens: int | None = None,
) -> dict[str, Any]:
    compaction: dict[str, Any] = {
        "mode": "rank_judge_auxiliary",
        "proposal_count": len(proposals),
        "included_proposal_count": len(included_proposals),
        "proposal_omitted_count": max(
            0, len(proposals) - len(included_proposals)
        ),
        "included_proposal_ids": [proposal.id for proposal in included_proposals],
        "source": "full proposals retained in generated proposal artifacts",
    }
    if dynamically_reduced:
        compaction["dynamic_fit"] = True
    if max_context_tokens is not None:
        compaction["max_context_tokens"] = max_context_tokens

    compact: dict[str, Any] = {
        "workspace": context.get("workspace"),
        "stage": stage,
        "context_compaction": compaction,
        "previous_outputs": _compact_auxiliary_previous_outputs(
            previous_outputs,
            stage=stage,
            proposals=included_proposals,
        ),
    }
    for key in (
        "medical_schema_issues_summary",
        "entity_cleanup_issues_summary",
        "hierarchy_branches",
        "proposal_revision_requests",
    ):
        if key in context:
            compact[key] = _compact_auxiliary_value(context[key])

    for key in ("quality_findings", "medical_schema_issues", "evidence_windows"):
        value = context.get(key)
        if isinstance(value, list):
            compact[key] = _compact_auxiliary_value(
                value[:MAX_AUXILIARY_CONTEXT_LIST_ITEMS]
            )

    rules_memory = context.get("rules_memory")
    if isinstance(rules_memory, dict):
        compact["rules_memory"] = {
            str(key): _clip_auxiliary_text(
                str(value), MAX_AUXILIARY_RULES_MEMORY_CHARS
            )
            for key, value in list(rules_memory.items())[
                :MAX_AUXILIARY_CONTEXT_DICT_ITEMS
            ]
        }
    return compact


def _select_auxiliary_proposals(
    proposals: list[ImprovementProposal],
) -> list[ImprovementProposal]:
    return proposals[:MAX_AUXILIARY_PROPOSALS]


def _auxiliary_context_proposals(
    context: dict[str, Any],
    proposals: list[ImprovementProposal],
) -> list[ImprovementProposal]:
    compaction = context.get("context_compaction")
    if not isinstance(compaction, dict):
        return proposals
    included_ids = compaction.get("included_proposal_ids")
    if not isinstance(included_ids, list):
        return proposals
    ordered_ids = [str(item) for item in included_ids if isinstance(item, str)]
    if not ordered_ids:
        return proposals
    proposals_by_id = {proposal.id: proposal for proposal in proposals}
    return [
        proposals_by_id[proposal_id]
        for proposal_id in ordered_ids
        if proposal_id in proposals_by_id
    ]


def _compact_auxiliary_previous_outputs(
    previous_outputs: dict[str, Any],
    *,
    stage: str,
    proposals: list[ImprovementProposal],
) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "summary": _compact_output_summary(previous_outputs),
        "proposals": [
            _compact_auxiliary_proposal(proposal) for proposal in proposals
        ],
    }
    for key in ("issue_explanations", "evidence_map", "missing_branches"):
        value = previous_outputs.get(key)
        if isinstance(value, list):
            compact[key] = _compact_auxiliary_value(
                value[:MAX_AUXILIARY_CONTEXT_LIST_ITEMS]
            )
    proposal_merge = previous_outputs.get("proposal_merge")
    if isinstance(proposal_merge, dict):
        compact["proposal_merge"] = _compact_auxiliary_value(proposal_merge)
    if stage == "judge":
        repair_plan = previous_outputs.get("repair_plan")
        if isinstance(repair_plan, list):
            proposal_ids = {proposal.id for proposal in proposals}
            selected_repair_plan = [
                item
                for item in repair_plan
                if not isinstance(item, dict)
                or str(item.get("proposal_id", "")).strip() in proposal_ids
            ]
            compact["repair_plan"] = _compact_auxiliary_value(
                selected_repair_plan[
                    : max(len(proposals), MAX_AUXILIARY_CONTEXT_LIST_ITEMS)
                ]
            )
    return compact


def _compact_auxiliary_proposal(
    proposal: ImprovementProposal,
) -> dict[str, Any]:
    payload = proposal.to_dict()
    compact: dict[str, Any] = {
        "id": proposal.id,
        "type": proposal.type,
        "target": _clip_auxiliary_text(
            proposal.target, MAX_AUXILIARY_PROPOSAL_TEXT_CHARS
        ),
        "proposed_change": _clip_auxiliary_text(
            proposal.proposed_change, MAX_AUXILIARY_PROPOSAL_TEXT_CHARS
        ),
        "reason": _clip_auxiliary_text(
            proposal.reason, MAX_AUXILIARY_PROPOSAL_TEXT_CHARS
        ),
        "confidence": proposal.confidence,
        "risk": proposal.risk,
        "requires_approval": proposal.requires_approval,
        "expected_metric_change": _compact_auxiliary_value(
            proposal.expected_metric_change
        ),
    }
    evidence = [
        _clip_auxiliary_text(str(item), MAX_AUXILIARY_PROPOSAL_TEXT_CHARS)
        for item in proposal.evidence[:MAX_AUXILIARY_PROPOSAL_EVIDENCE_ITEMS]
    ]
    if evidence:
        compact["evidence"] = evidence
    if len(proposal.evidence) > len(evidence):
        compact["evidence_omitted_count"] = len(proposal.evidence) - len(evidence)
    action_payload = payload.get("action_payload")
    if isinstance(action_payload, dict) and action_payload:
        compact["action_payload"] = _compact_auxiliary_action_payload(action_payload)
    if proposal.judge:
        compact["judge"] = _compact_auxiliary_value(proposal.judge)
    return compact


def _compact_auxiliary_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _compact_auxiliary_value(
            value,
            string_chars=MAX_AUXILIARY_ACTION_VALUE_CHARS,
        )
        for key, value in list(payload.items())[:MAX_AUXILIARY_CONTEXT_DICT_ITEMS]
    }


def _compact_auxiliary_value(
    value: Any,
    *,
    string_chars: int = MAX_AUXILIARY_CONTEXT_STRING_CHARS,
) -> Any:
    if isinstance(value, str):
        return _clip_auxiliary_text(value, string_chars)
    if isinstance(value, int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [
            _compact_auxiliary_value(item, string_chars=string_chars)
            for item in value[:MAX_AUXILIARY_CONTEXT_LIST_ITEMS]
        ]
    if isinstance(value, dict):
        return {
            str(key): _compact_auxiliary_value(item, string_chars=string_chars)
            for key, item in list(value.items())[:MAX_AUXILIARY_CONTEXT_DICT_ITEMS]
        }
    return _clip_auxiliary_text(str(value), string_chars)


def _clip_auxiliary_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3].rstrip() + "..."


def _task_pack_reference_values(task_pack: dict[str, Any]) -> set[str]:
    refs = _issue_reference_values(_list_of_dicts(task_pack.get("issues")))
    for candidate in _list_of_dicts(task_pack.get("action_candidates")):
        target = str(candidate.get("target", "")).strip()
        if target:
            refs.add(target)
            if target.startswith("edge:"):
                refs.add(target.removeprefix("edge:"))
        payload = candidate.get("action_payload")
        if isinstance(payload, dict):
            for value in payload.values():
                if value is not None and not isinstance(value, dict | list):
                    text = str(value).strip()
                    if text:
                        refs.add(text)
    return refs


def _relevant_or_first_items(
    items: list[Any], refs: set[str], limit: int
) -> list[Any]:
    dict_items = [item for item in items if isinstance(item, dict)]
    if refs:
        relevant = [
            item
            for item in dict_items
            if _item_text_references_task(item, refs)
        ]
        if relevant:
            return relevant[:limit]
    return dict_items[:limit]


def _item_text_references_task(item: dict[str, Any], refs: set[str]) -> bool:
    text = json.dumps(item, ensure_ascii=False, sort_keys=True).casefold()
    return any(ref.casefold() in text for ref in refs if ref)


def _compact_subagent_value(value: Any) -> Any:
    if isinstance(value, str):
        return _clip_subagent_text(value, MAX_SUBAGENT_COMPACT_STRING_CHARS)
    if isinstance(value, int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [
            _compact_subagent_value(item)
            for item in value[:MAX_SUBAGENT_COMPACT_LIST_ITEMS]
        ]
    if isinstance(value, dict):
        return {
            str(key): _compact_subagent_value(item)
            for key, item in list(value.items())[:MAX_SUBAGENT_COMPACT_DICT_ITEMS]
        }
    return _clip_subagent_text(str(value), MAX_SUBAGENT_COMPACT_STRING_CHARS)


def _clip_subagent_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3].rstrip() + "..."


def _subagent_propose_system_prompt(
    profile: str | None,
    *,
    role: str | None = None,
    max_proposals_per_task: int = MAX_SUBAGENT_PROPOSALS_PER_TASK,
) -> str:
    parts = _subagent_role_prompt_parts(role)
    parts.extend(
        [
        "Proposal Orchestrator Subagent",
        "",
        "You generate candidate ImprovementProposal JSON for exactly one task_pack.",
        "Use the task_pack role, issues, action_candidates, and snapshot_context as your bounded scope.",
        f"Return at most {max_proposals_per_task} highest-confidence proposals.",
        "Do not use the monolithic propose-stage prompt or broaden the task beyond the supplied pack.",
        "Validate your own output against the propose-stage schema before returning it.",
        "Return only JSON.",
        "",
        "ImprovementProposal required fields:",
        "- id: ASCII [A-Za-z0-9_.-]+.",
        "- type: canonical lowercase snake_case.",
        "- target, proposed_change, reason: non-empty strings.",
        "- evidence: list of strings only. Do not output object/dict evidence; invalid structured evidence is rejected with EVIDENCE_MUST_BE_STRING.",
        "- If source_id, file_path, or evidence_quote is needed, put it in action_payload or candidate_edges, copied from one exact allowed_evidence_spans record.",
        "- confidence: number from 0 to 1.",
        "- risk: one of low, medium, high.",
        "- requires_approval: true for every KG mutation proposal.",
        "- expected_metric_change: object with numeric values, or {}.",
        "",
        "medical_relation_schema_migration action_payload:",
        "- Use type medical_relation_schema_migration for executable relation replacements or tightly scoped relation retirements.",
        "- action_payload must contain action=replace_relation, edge_id, expected_source, expected_target, current_keywords, new_source, new_target, new_keywords, qualifiers.",
        "- For action=retire_relation, action_payload must contain edge_id, expected_source, expected_target, current_keywords, retirement_reason.",
        "- Use retire_relation only when the current edge is an invalid or misleading medical fact that cannot be safely converted into a grounded canonical replacement, such as category/section nodes modeled as symptoms or broad diagnostic category nodes modeled as concrete diagnostic criteria.",
        "- Do not use retire_relation for edges that can be safely migrated with replace_relation.",
        "- new_keywords must be a canonical medical relation id.",
        "- Copy action_payload from task_pack.action_candidates when available.",
        "- current_keywords must copy the exact full current keyword string from the existing edge, including all comma-separated or field-separated labels; if the edge has multiple meanings, split into executable single-edge proposals or omit the proposal instead of silently dropping one.",
        "- Do not emit no-op replace_relation proposals: if expected_source/new_source, expected_target/new_target, and current_keywords/new_keywords are unchanged and qualifiers are empty, skip it.",
        "- Return one relation replacement per proposal; split multi-edge changes into separate proposals.",
        "- For has_indication, new_target must be a disease or clinical condition such as 流行性感冒 or 流感病毒感染, never a bare pathogen such as 流感病毒.",
        "- For orders_test, new_source must be a disease, clinical condition, care process, or recommendation context, never a bare pathogen such as 流感病毒.",
        "- For targets_disease, new_target must be a disease or clinical condition such as 流行性感冒 or 季节性流感, never a bare pathogen such as 流感病毒.",
        "- For reduces_risk_of, if the current edge target is a population/patient group such as 急性心衰患者, do not replace it with a disease target such as 急性心力衰竭 unless qualifiers preserve the population and the new target is the actual outcome, e.g. 死亡/再入院.",
        "- For has_manifestation, new_target must be a patient symptom, sign, or clinical finding; never use category/section targets such as 流感临床表现, 临床表现, or 症状表现.",
        "- For causative_agent on influenza diseases, do not use bacterial secondary-infection pathogens such as 肺炎链球菌 or 金黄色葡萄球菌 as the cause of 甲型流感 or 流行性感冒.",
        "- For causative_agent on typed influenza diseases such as 甲型流感 or 乙型流感, do not target generic 流感病毒; use or propose a typed pathogen such as 甲型流感病毒 or 乙型流感病毒, with is_a -> 流感病毒.",
        "- If task_pack.action_candidates contains a typed influenza pathogen candidate_kg_expansion, copy that action_payload instead of generating 甲型流感/乙型流感 -> causative_agent -> 流感病毒.",
        "- For supports_or_refutes, diagnostic evidence, tests, or findings should point to the disease diagnosis; do not emit disease -> supports_or_refutes -> generic test patterns such as 流行性感冒 -> supports_or_refutes -> 病原学检查.",
        "- For supports_or_refutes on influenza, do not use nonspecific labs or complication assessment such as 血常规, 血生化, 动脉血气分析, 丙氨酸氨基转移酶, 天门冬氨酸氨基转移酶, or MRI as direct support/refutation for 流行性感冒 diagnosis; attach them to clinical findings, severity/complication assessment, or a more specific complication when grounded.",
        "- For has_diagnostic_criterion on influenza, reserve it for pathogen/etiology tests or true disease-defining criteria; do not model nonspecific labs or complication assessment such as 血常规, 血生化, 动脉血气分析, 丙氨酸氨基转移酶, 天门冬氨酸氨基转移酶, or MRI as direct 流行性感冒 diagnostic criteria.",
        "- In severe/critical influenza contexts, signs such as 鼻翼扇动, 三凹征, and 呼吸急促 are severity/diagnostic criteria; prefer 流感重型 -> has_diagnostic_criterion -> specific sign over has_manifestation.",
        "- Do not retire nonspecific test edges such as 血常规, 血生化, 动脉血气分析, CT, or MRI only because they are not direct influenza criteria; convert them to orders_test, monitor_with, or a grounded complication/severity-assessment relation.",
        "- Do not convert CT or MRI to generic 流行性感冒 -> orders_test -> CT/MRI; attach CT to a concrete endpoint such as 流感肺炎/原发性病毒性肺炎 or attach MRI/CT to 急性坏死性脑病/神经系统并发症 when grounded.",
        "- Do not emit 流行性感冒 -> orders_test -> bare lab analytes such as AST/ALT/LDH, 肌酐, 肌酸激酶, or 肌红蛋白; order the parent panel such as 血生化, then model analytes with observes, has_result, clinical findings, or complication/severity assessment.",
        "- If task_pack.action_candidates contains orders_test for a nonspecific test edge, copy that action_payload instead of generating retire_relation or review-only output.",
        "- For grounded zanamivir split tasks such as 扎那米韦->哮喘 or 扎那米韦->儿童, do not emit review-only output; produce an executable split with purpose, route, age, dose, duration, and precaution/not-recommended qualifiers when the source text provides them.",
        "- For TCM syndrome indication edges, do not emit review-only proposals when the source relation is clear; create a candidate_kg_expansion or executable migration that models the syndrome as ClinicalCondition / TCM syndrome and preserves has_indication.",
        "- For has_complication on influenza, do not model chronic underlying conditions such as 慢性阻塞性肺疾病(COPD) as direct flu complications; use risk-factor/high-risk-population or acute-exacerbation semantics only when the evidence supports that endpoint.",
        "- For has_complication on influenza, 死亡/住院/早产/低出生体重/重症/危重症/心肌梗死/缺血性中风 are outcomes, utilization, severity states, or risk events; do not use has_complication for them. Use outcome/risk/severity semantics or request executable schema refinement.",
        "- For is_a taxonomy, direction is child/subtype -> parent/supertype; emit 乙型流感病毒 -> is_a -> 流感病毒, never 流感病毒 -> is_a -> 乙型流感病毒.",
        "- For recommended_for targeting broad populations such as 儿童, require purpose=treatment|prevention plus qualifiers like condition/context/age/route/risk/reason; do not emit empty-qualifier or missing-purpose broad relations such as 扎那米韦 -> recommended_for -> 儿童.",
        "- For has_manifestation, do not use electrolyte or laboratory abnormalities such as 低钾血症 as ordinary symptom targets; use complication/clinical finding semantics or omit the proposal when no executable grounded repair is available.",
        "- Do not emit review_context_request in subagent output; this stage must return executable proposals with action_payload, grounded candidate_kg_expansion, or no proposal.",
        "- You may emit medical_fact_role_split only with action_payload.action=split_relation, non-empty new_edges, canonical predicates, grounded endpoints, and qualifiers needed to keep each split edge clinically scoped. Do not emit draft_split_relation.",
        "",
        "candidate_kg_expansion action_payload:",
        "- Use type candidate_kg_expansion only when adding a missing node or edge is necessary and directly grounded in deterministic source text.",
        "- action_payload must contain top-level candidate_nodes, candidate_edges, source_id, file_path, evidence_quote, why_not_existing.",
        "- candidate_nodes must be a JSON list; candidate_edges must be a JSON list. Use [] when there are no nodes or no edges in that proposal.",
        "- Each candidate_edges[] item must include source, target, source_type, target_type, keywords, source_id, and file_path.",
        "- retire_edges is optional and may be used only when the new candidate nodes/edges clearly replace a wrong existing edge; each item must include source, target, keywords, and reason.",
        "- source_id and file_path are top-level strings copied from deterministic artifacts and repeated on each candidate edge; do not place them inside nested evidence.",
        "- source_id, file_path, and evidence_quote must come from the same allowed_evidence_spans record; do not cross-combine fields and do not join references with <SEP>.",
        "- evidence_quote must be a top-level source-text quote copied from the deterministic artifact and long enough to ground the new candidate.",
        "- why_not_existing must explain why existing KG nodes/edges do not already cover the proposed fact.",
        "",
        "Evidence must be strings only; structured evidence objects are invalid.",
        "",
        f"profile: {profile or 'default'}",
        ]
    )
    if role:
        contract = role_contract(role)
        parts.extend(
            [
                "",
                f"Subagent role contract: {contract.role}",
                "- allowed_proposal_types: "
                + ", ".join(contract.allowed_proposal_types),
                "- allowed_predicates: " + ", ".join(contract.allowed_predicates),
                "- forbidden_predicates: "
                + ", ".join(contract.forbidden_predicates),
                "- payload_mode: " + contract.payload_mode,
                "- require_action_candidate: "
                + ("true" if contract.require_action_candidate else "false"),
                *[f"- {instruction}" for instruction in contract.instructions],
            ]
        )
    schema_prompt = render_medical_relation_schema_prompt(profile)
    if schema_prompt:
        parts.extend(["", schema_prompt.strip()])
    return "\n".join(parts)


def _subagent_role_prompt_parts(role: str | None) -> list[str]:
    if not role:
        return []
    prompt_root = Path(__file__).parent / "prompts" / "subagents"
    role_file = _SUBAGENT_PROMPT_FILES.get(role)
    if role_file is None:
        raise ValueError(f"unsupported subagent role: {role}")
    parts: list[str] = []
    for filename in ("base_zh.md", role_file):
        path = prompt_root / filename
        text = path.read_text(encoding="utf-8").strip()
        if text:
            parts.extend([text, ""])
    return parts


def _write_json_artifact(output_dir: Path, relative_path: str, payload: Any) -> Path:
    path = safe_package_child_path(output_dir, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _write_subagent_output_artifact(
    output_dir: Path,
    *,
    task_id: str,
    role: str,
    raw_output: str,
    parsed: AgentStageOutput | None,
    error: str,
    attempts: list[dict[str, Any]],
) -> Path:
    sanitized_error = _sanitize_error_text(error)
    payload: dict[str, Any] = {
        "task_id": task_id,
        "role": role,
        "raw_output": raw_output,
        "parsed_output": parsed.payload if parsed is not None else None,
        "proposal_ids": (
            [proposal.id for proposal in parsed.proposals]
            if parsed is not None
            else []
        ),
        "error": sanitized_error,
        "attempts": attempts,
    }
    metadata = _proposal_validation_error_metadata(sanitized_error)
    if metadata:
        payload["error_code"] = metadata["error_code"]
    return _write_json_artifact(
        output_dir,
        f"subagent_outputs/{task_id}.json",
        payload,
    )


def _subagent_attempt_record(
    *,
    task_id: str,
    role: str,
    attempt: int,
    state: str,
    error: str,
    raw_output: str,
) -> dict[str, Any]:
    sanitized_error = _sanitize_error_text(error)
    record = {
        "task_id": task_id,
        "role": role,
        "attempt": attempt,
        "state": state,
        "error": sanitized_error,
        "output_token_estimate": _estimate_tokens(raw_output) if raw_output else 0,
    }
    metadata = _proposal_validation_error_metadata(sanitized_error)
    if metadata:
        record["error_code"] = metadata["error_code"]
    return record


def _write_proposal_merge_report(
    output_dir: Path, merge_payload: dict[str, Any]
) -> Path:
    lines = [
        "# Proposal Merge Report",
        "",
        "## Selected Proposals",
        "",
    ]
    proposals = _list_of_dicts(merge_payload.get("proposals"))
    if proposals:
        lines.extend(f"- {proposal.get('id', '')}" for proposal in proposals)
    else:
        lines.append("- none")
    lines.extend(["", "## Conflicts", ""])
    conflicts = _list_of_dicts(merge_payload.get("conflicts"))
    if conflicts:
        for conflict in conflicts:
            proposal_ids = ", ".join(
                str(proposal_id)
                for proposal_id in conflict.get("proposal_ids", [])
            )
            lines.append(
                f"- {conflict.get('target', '')}: {proposal_ids} "
                f"({conflict.get('reason', '')})"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Dropped", ""])
    dropped = _list_of_dicts(merge_payload.get("dropped"))
    if dropped:
        for item in dropped:
            error_code = str(item.get("error_code") or "").strip()
            error_code_text = f" [{error_code}]" if error_code else ""
            lines.append(
                f"- {item.get('proposal_id', '')}: {item.get('reason', '')}"
                f"{error_code_text}"
            )
    else:
        lines.append("- none")
    lines.append("")
    path = safe_package_child_path(output_dir, "proposal_merge_report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _selected_stages(config: LLMAgentPipelineConfig) -> tuple[str, ...]:
    if config.allow_llm_judge:
        return _STAGES
    return tuple(stage for stage in _STAGES if stage != "judge")


def _sanitize_error_text(error: object) -> str:
    text = str(error)
    text = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2)}"
            f"{match.group(3)}[REDACTED]{match.group(5)}"
        ),
        text,
    )
    text = _BEARER_TOKEN_RE.sub(lambda match: f"{match.group(1)}[REDACTED]", text)
    return _OPENAI_KEY_RE.sub("[REDACTED]", text)


def _proposal_validation_error_metadata(error: object) -> dict[str, Any]:
    normalized = str(error).casefold()
    if not normalized:
        return {}

    if "evidence_must_be_string" in normalized:
        return {
            "error_code": "EVIDENCE_MUST_BE_STRING",
            "missing_fields": [],
            "repair_instruction": (
                "Return evidence as a list of strings only; do not output object "
                "or dict evidence."
            ),
        }
    if "candidate_edge_types_required" in normalized:
        return {
            "error_code": "CANDIDATE_EDGE_TYPES_REQUIRED",
            "missing_fields": ["source_type", "target_type"],
            "candidate_edge_required_fields": [
                "source",
                "target",
                "source_type",
                "target_type",
                "keywords",
                "source_id",
                "file_path",
            ],
            "repair_instruction": (
                "Every candidate_edges item must include source_type and "
                "target_type."
            ),
        }
    if "candidate_edge_required_fields" in normalized:
        return {
            "error_code": "CANDIDATE_EDGE_REQUIRED_FIELDS",
            "missing_fields": [
                "source",
                "target",
                "source_type",
                "target_type",
                "keywords",
                "source_id",
                "file_path",
            ],
        }
    if "candidate_nodes[" in normalized and "requires id and entity_type" in normalized:
        return {
            "error_code": "CANDIDATE_NODE_REQUIRED_FIELDS",
            "missing_fields": ["id", "entity_type"],
            "repair_instruction": (
                "Every candidate_nodes item must include id and entity_type."
            ),
        }
    if "evidence_tuple_must_match_allowed_span" in normalized:
        return {
            "error_code": "EVIDENCE_TUPLE_MUST_MATCH_ALLOWED_SPAN",
            "missing_fields": ["source_id", "file_path", "evidence_quote"],
            "repair_instruction": (
                "Copy source_id, file_path, and evidence_quote from the same "
                "allowed_evidence_spans record."
            ),
        }
    if "action_payload" in normalized and "grounded" in normalized:
        return {
            "error_code": "ACTION_PAYLOAD_NOT_GROUNDED",
            "missing_fields": ["action_payload"],
            "repair_instruction": (
                "Copy action_payload from the provided action_candidates; do not "
                "invent KG identifiers or endpoints."
            ),
        }
    if "evidence" in normalized and "grounded" in normalized:
        return {
            "error_code": "EVIDENCE_NOT_GROUNDED",
            "missing_fields": ["source_id", "file_path", "evidence_quote"],
            "repair_instruction": (
                "Evidence must quote deterministic artifacts or allowed evidence "
                "spans exactly."
            ),
        }
    if "expected_metric_change" in normalized and (
        "must be a dict" in normalized or "values must be numbers" in normalized
    ):
        return {
            "error_code": "EXPECTED_METRIC_CHANGE_INVALID",
            "missing_fields": ["expected_metric_change"],
            "repair_instruction": (
                "Use finite JSON numbers for expected_metric_change values, or {} "
                "when no numeric estimate is available."
            ),
        }
    if "replace_relation" in normalized and (
        "no-op" in normalized or "noop" in normalized
    ):
        return {
            "error_code": "NO_OP_REPLACE_RELATION",
            "missing_fields": [],
            "repair_instruction": (
                "Do not emit no-op replace_relation proposals; skip the proposal "
                "or choose a candidate that changes the relation."
            ),
        }
    if (
        "quality_report_note" in normalized
        and "target quality_report.md" in normalized
    ):
        return {
            "error_code": "QUALITY_REPORT_NOTE_TARGET_INVALID",
            "missing_fields": ["target"],
            "repair_instruction": (
                "Report-only quality_report_note proposals must target "
                "quality_report.md."
            ),
        }
    if (
        "orders_test" in normalized
        and "source must be a disease" in normalized
        and "bare pathogen" in normalized
    ):
        return {
            "error_code": "RELATION_SCHEMA_VIOLATION",
            "missing_fields": ["new_source"],
            "repair_instruction": (
                "For orders_test, new_source must be a disease, clinical "
                "condition, care process, or recommendation context."
            ),
        }
    if "candidate_edge_schema_violation" in normalized:
        return {
            "error_code": "CANDIDATE_EDGE_SCHEMA_VIOLATION",
            "missing_fields": ["source_type", "target_type", "keywords"],
            "repair_instruction": (
                "Candidate edges must satisfy the medical schema domain and "
                "range for the chosen predicate."
            ),
        }
    if "orders_test" in normalized and "bare lab marker" in normalized:
        return {
            "error_code": "RELATION_SCHEMA_VIOLATION",
            "missing_fields": ["new_target"],
            "repair_instruction": (
                "For orders_test, order the parent test panel or model the lab "
                "marker as an observed finding."
            ),
        }
    if (
        "supports_or_refutes" in normalized
        and "nonspecific labs" in normalized
    ):
        return {
            "error_code": "RELATION_SCHEMA_VIOLATION",
            "missing_fields": ["new_target", "new_keywords"],
            "repair_instruction": (
                "Do not attach nonspecific labs or complication-imaging findings "
                "as direct support/refutation for influenza diagnosis."
            ),
        }
    if "domain-range" in normalized or "canonical relation id" in normalized:
        return {
            "error_code": "RELATION_SCHEMA_VIOLATION",
            "missing_fields": ["new_keywords"],
        }

    return {}


def _sanitize_trace_errors(trace: dict[str, Any]) -> None:
    stages = trace.get("stages")
    if not isinstance(stages, list):
        return
    for stage_trace in stages:
        if not isinstance(stage_trace, dict):
            continue
        error = stage_trace.get("error")
        if isinstance(error, str):
            stage_trace["error"] = _sanitize_error_text(error)
            metadata = _proposal_validation_error_metadata(error)
            if metadata:
                stage_trace.setdefault("error_code", metadata["error_code"])
        attempt_logs = stage_trace.get("attempt_logs")
        if not isinstance(attempt_logs, list):
            continue
        for item in attempt_logs:
            if not isinstance(item, dict):
                continue
            attempt_error = item.get("error")
            if isinstance(attempt_error, str):
                item["error"] = _sanitize_error_text(attempt_error)
                metadata = _proposal_validation_error_metadata(attempt_error)
                if metadata:
                    item.setdefault("error_code", metadata["error_code"])


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
    sanitized_error = _sanitize_error_text(error)
    record = {
        "attempt": attempt,
        "state": state,
        "error": sanitized_error,
        "output_token_estimate": _estimate_tokens(raw_output) if raw_output else 0,
    }
    metadata = _proposal_validation_error_metadata(sanitized_error)
    if metadata:
        record["error_code"] = metadata["error_code"]
    attempt_logs.append(record)


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
            lines.append(f"Attempt {attempt}: {_sanitize_error_text(error)}")
    return lines


def _retry_user_prompt(
    base_user_prompt: str,
    *,
    stage: str,
    error: str,
    previous_errors: list[str] | None = None,
    task_pack: dict[str, Any] | None = None,
) -> str:
    error = _sanitize_error_text(error)
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
        guidance.append(_proposal_validation_retry_hint(error))
        constraints = _proposal_retry_constraints(error, task_pack=task_pack)
        if constraints:
            guidance.append(
                "必须遵守以下机器可读修复约束："
                + json.dumps(
                    constraints,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        guidance.append(
            'For "expected_metric_change", use finite JSON numbers only; '
            "use {} if no numeric estimate is available."
        )
        guidance.append(
            "Every non-context proposal must keep requires_approval=true when it can mutate KG, rules, prompts, workspace, or WebUI behavior."
        )
        guidance.append("Do not change evidence/human-approval requirements.")
    return "\n\n".join([base_user_prompt, *guidance])


def _proposal_retry_constraints(
    error: str,
    *,
    task_pack: dict[str, Any] | None,
) -> dict[str, Any]:
    if not task_pack:
        return {}
    normalized = error.casefold()
    contract = task_pack.get("role_contract")
    if not isinstance(contract, dict):
        contract = {}
    constraints: dict[str, Any] = {
        "role": task_pack.get("role", ""),
        "allowed_proposal_types": contract.get("allowed_proposal_types", []),
        "allowed_predicates": contract.get("allowed_predicates", []),
        "forbidden_predicates": contract.get("forbidden_predicates", []),
    }
    metadata = _proposal_validation_error_metadata(error)
    if metadata:
        constraints.update(metadata)
    if "evidence_must_be_string" in normalized:
        constraints.update(
            {
                "error_code": "EVIDENCE_MUST_BE_STRING",
                "missing_fields": [],
                "repair_instruction": (
                    "evidence 必须是字符串列表；不要输出 object/dict evidence。"
                ),
            }
        )
    if "candidate_edge_types_required" in normalized:
        constraints.update(
            {
                "error_code": "CANDIDATE_EDGE_TYPES_REQUIRED",
                "missing_fields": ["source_type", "target_type"],
                "candidate_edge_required_fields": [
                    "source",
                    "target",
                    "source_type",
                    "target_type",
                    "keywords",
                    "source_id",
                    "file_path",
                ],
                "repair_instruction": (
                    "candidate_edges 每条边都必须补齐 source_type 和 target_type。"
                ),
            }
        )
    if "candidate_edge_required_fields" in normalized:
        constraints.update(
            {
                "error_code": "CANDIDATE_EDGE_REQUIRED_FIELDS",
                "missing_fields": [
                    "source",
                    "target",
                    "source_type",
                    "target_type",
                    "keywords",
                    "source_id",
                    "file_path",
                ],
            }
        )
    if "evidence_tuple_must_match_allowed_span" in normalized:
        constraints.update(
            {
                "error_code": "EVIDENCE_TUPLE_MUST_MATCH_ALLOWED_SPAN",
                "missing_fields": ["source_id", "file_path", "evidence_quote"],
                "repair_instruction": (
                    "source_id、file_path、evidence_quote 必须从同一条 "
                    "allowed_evidence_spans 逐字复制；不得使用 <SEP>。"
                ),
            }
        )

    action_candidates = _list_of_dicts(task_pack.get("action_candidates"))
    if action_candidates:
        constraints["allowed_candidate_ids"] = [
            str(item.get("candidate_id") or "")
            for item in action_candidates
            if str(item.get("candidate_id") or "").strip()
        ]

    if "has_complication" in normalized:
        constraints["repair_instruction"] = (
            "不要再次使用 has_complication；本任务只能使用 "
            "role_contract.allowed_predicates；无法安全映射时返回空 proposals。"
        )

    if "medical_fact_role_split" in normalized or "split_relation" in normalized:
        constraints["must_copy_fields"] = [
            "edge_id",
            "expected_source",
            "expected_target",
            "current_keywords",
            "retire_original",
            "new_edges",
        ]
        constraints["repair_instruction"] = (
            "split payload 必须完整复制 action_candidate，不得自行修改 new_edges。"
        )

    if (
        "source_id" in normalized
        or "file_path" in normalized
        or "evidence_quote" in normalized
        or "grounded" in normalized
    ):
        constraints["allowed_evidence_spans"] = _list_of_dicts(
            task_pack.get("allowed_evidence_spans")
        )
        constraints["repair_instruction"] = (
            "source_id、file_path、evidence_quote 必须从同一条 "
            "allowed_evidence_spans 记录逐字复制；不得使用 <SEP>。"
        )

    if "action_candidate" in normalized:
        constraints["repair_instruction"] = (
            "不得自己拼 action_payload；只能选择列出的 candidate_id。"
        )

    return constraints


def _proposal_validation_retry_hint(error: str) -> str:
    normalized = error.casefold()
    if "candidate_kg_expansion" in normalized:
        return (
            "For candidate_kg_expansion action_payload, provide top-level "
            "candidate_nodes (list), candidate_edges (list), source_id (string), "
            "file_path (string), evidence_quote (string copied from deterministic "
            "source text), and why_not_existing (string). Do not hide the only "
            "source_id/file_path inside candidate_nodes, candidate_edges, or nested "
            "evidence. Optional retire_edges may be used only when the new "
            "candidate nodes/edges clearly replace a wrong existing edge."
        )
    if "target must be a non-empty string" in normalized:
        return (
            "Set proposal.target to an existing deterministic target such as "
            "edge:<edge_id> or entity:<entity_id>."
        )
    if "expected_source" in normalized:
        return (
            "For replace_relation or retire_relation action_payload, copy "
            "expected_source exactly from the current edge source."
        )
    if "expected_metric_change must be a dict" in normalized:
        return (
            'Use expected_metric_change as an object, for example '
            '{"relation_semantic_issue_count": -1}.'
        )
    if "unknown structured field" in normalized:
        return (
            "Evidence structured fields may only use source_id, file_path, "
            "item_id, entity_id, relation_id, or metric."
        )
    if "action_payload" in normalized and (
        "ungrounded" in normalized
        or "not grounded" in normalized
        or "grounded in deterministic artifacts" in normalized
    ):
        return (
            "Copy action_payload from the provided action_candidates when "
            "available; do not invent edge_id/source/target values."
        )
    return "Return JSON only and satisfy the ImprovementProposal schema exactly."


def _client_error_retry_user_prompt(
    base_user_prompt: str,
    *,
    error: str,
    previous_errors: list[str] | None = None,
) -> str:
    guidance = [
        f"Previous LLM client call failed: {_sanitize_error_text(error)}.",
        "Retry the same stage and return the stage JSON only.",
    ]
    if previous_errors:
        guidance.extend(["Previous client errors:", *previous_errors])
    return "\n\n".join([base_user_prompt, *guidance])


def _auxiliary_stage_unavailable_reason(stage: str, error: str) -> str:
    reason = _single_line(_sanitize_error_text(error))
    if stage == "judge":
        return reason
    return f"{stage} stage unavailable: {reason}"


def _synthetic_judge_stage_trace(reason: str) -> dict[str, Any]:
    reason = _sanitize_error_text(reason)
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
    subagent_dir = Path(output_dir) / "subagent_outputs"
    if _is_link_or_junction(subagent_dir):
        _remove_link(subagent_dir)
    for relative_path in (*_GENERATED_AGENT_ARTIFACTS, *_GENERATED_AGENT_CONTEXT_FILES):
        for cleanup_relative_path in _generated_artifact_cleanup_paths(relative_path):
            artifact_path = safe_package_child_path(output_dir, cleanup_relative_path)
            if artifact_path.is_file():
                artifact_path.unlink()
    subagent_dir = safe_package_child_path(output_dir, "subagent_outputs")
    if not subagent_dir.is_dir():
        return
    for artifact_path in subagent_dir.iterdir():
        if artifact_path.is_file():
            artifact_path.unlink()
    try:
        subagent_dir.rmdir()
    except OSError:
        pass


def _generated_artifact_cleanup_paths(relative_path: str) -> tuple[str, ...]:
    path = Path(relative_path)
    if not path.suffix:
        return (relative_path,)
    zh_relative_path = artifact_zh_relative_path(path).as_posix()
    return (relative_path, zh_relative_path)


def _attach_judge_results(
    proposals: list[ImprovementProposal],
    payload: dict[str, Any],
    *,
    expected_proposals: list[ImprovementProposal] | None = None,
) -> list[ImprovementProposal]:
    judge_by_id = _normalized_judge_results(payload, expected_proposals or proposals)
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
        _validate_action_payload_grounding(proposal, reference_tokens)
        for evidence in proposal.evidence:
            if not evidence.strip():
                continue
            if _evidence_references_known_artifact(evidence, reference_tokens):
                continue
            raise ValueError(
                f"proposal {proposal.id} evidence is not grounded in deterministic "
                "artifacts"
            )


def _validate_executable_medical_action_payloads(
    proposals: list[ImprovementProposal],
) -> None:
    for proposal in proposals:
        if proposal.type == "review_context_request":
            raise ValueError(
                f"proposal {proposal.id} review_context_request is not executable "
                "in orchestrated propose output; return an executable proposal "
                "with action_payload or omit the proposal when the evidence is "
                "insufficient"
            )
        if (
            proposal.type in _EXECUTABLE_MEDICAL_PROPOSAL_TYPES
            and not proposal.action_payload
        ):
            raise ValueError(
                f"proposal {proposal.id} action_payload is required for "
                "executable medical proposal type"
            )


def _validate_action_payload_matches_deterministic_candidates(
    proposals: list[ImprovementProposal],
    *,
    task_pack: dict[str, Any],
) -> None:
    action_candidates = _list_of_dicts(task_pack.get("action_candidates"))
    if not action_candidates:
        return
    for proposal in proposals:
        if (
            proposal.type not in _EXECUTABLE_MEDICAL_PROPOSAL_TYPES
            or not proposal.action_payload
            or proposal.type == "candidate_kg_expansion"
        ):
            continue
        matching_candidate_payloads = [
            candidate.get("action_payload")
            for candidate in action_candidates
            if _candidate_matches_proposal_action_target(candidate, proposal)
            and isinstance(candidate.get("action_payload"), dict)
        ]
        if not matching_candidate_payloads:
            continue
        proposal_payload_key = _stable_json_key(proposal.action_payload)
        if any(
            proposal_payload_key == _stable_json_key(candidate_payload)
            for candidate_payload in matching_candidate_payloads
        ):
            continue
        raise ValueError(
            f"proposal {proposal.id} action_payload does not match provided "
            "deterministic action_candidates"
        )


def _validate_candidate_expansion_evidence_allowlist(
    proposals: list[ImprovementProposal],
    *,
    task_pack: dict[str, Any],
) -> None:
    candidate_expansions = [
        proposal for proposal in proposals if proposal.type == "candidate_kg_expansion"
    ]
    if not candidate_expansions:
        return
    allowed_spans = _list_of_dicts(task_pack.get("allowed_evidence_spans"))
    allowed_tuples = {
        (
            str(span.get("source_id") or "").strip(),
            str(span.get("file_path") or "").strip(),
            str(span.get("evidence_quote") or "").strip(),
        )
        for span in allowed_spans
        if str(span.get("source_id") or "").strip()
        and str(span.get("file_path") or "").strip()
        and str(span.get("evidence_quote") or "").strip()
    }
    if not allowed_tuples:
        raise ValueError(
            "candidate_kg_expansion requires non-empty "
            "task_pack.allowed_evidence_spans"
        )
    for proposal in candidate_expansions:
        payload_tuple = (
            str(proposal.action_payload.get("source_id") or "").strip(),
            str(proposal.action_payload.get("file_path") or "").strip(),
            str(proposal.action_payload.get("evidence_quote") or "").strip(),
        )
        if payload_tuple in allowed_tuples:
            continue
        raise ValueError(
            f"proposal {proposal.id} candidate_kg_expansion evidence tuple "
            "is not in task_pack.allowed_evidence_spans"
        )


def _candidate_matches_proposal_action_target(
    candidate: dict[str, Any], proposal: ImprovementProposal
) -> bool:
    if str(candidate.get("proposal_type") or "").strip() != proposal.type:
        return False
    candidate_payload = candidate.get("action_payload")
    if not isinstance(candidate_payload, dict):
        return False
    proposal_edge_id = _normalize_reference_part(proposal.action_payload.get("edge_id"))
    candidate_edge_id = _normalize_reference_part(candidate_payload.get("edge_id"))
    if proposal_edge_id and candidate_edge_id and proposal_edge_id == candidate_edge_id:
        return True
    candidate_target = _normalize_action_target(candidate.get("target"))
    proposal_target = _normalize_action_target(proposal.target)
    return bool(candidate_target and proposal_target and candidate_target == proposal_target)


def _normalize_action_target(value: Any) -> str:
    normalized = _normalize_reference_part(value)
    for prefix in ("kg:relation:", "relation:", "edge:"):
        if normalized.startswith(prefix):
            return normalized.removeprefix(prefix)
    return normalized


def _stable_json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _validate_action_payload_grounding(
    proposal: ImprovementProposal,
    reference_tokens: _EvidenceReferenceTokens,
) -> None:
    payload = proposal.action_payload
    if not payload:
        return
    for key, token_key in _ACTION_PAYLOAD_TYPED_REFERENCE_KEYS.items():
        _validate_action_payload_reference(proposal, reference_tokens, key, token_key)
    if proposal.type == "candidate_kg_expansion":
        _validate_candidate_kg_expansion_payload_grounding(
            proposal,
            reference_tokens,
        )


def _validate_candidate_kg_expansion_payload_grounding(
    proposal: ImprovementProposal,
    reference_tokens: _EvidenceReferenceTokens,
) -> None:
    for key, token_key in _CANDIDATE_KG_EXPANSION_PAYLOAD_REFERENCE_KEYS.items():
        _validate_action_payload_reference(proposal, reference_tokens, key, token_key)

    evidence_quote = proposal.action_payload.get("evidence_quote")
    normalized_quote = _normalize_reference_part(evidence_quote)
    if (
        normalized_quote
        and _candidate_quote_references_declared_source(
            proposal.action_payload,
            normalized_quote,
            reference_tokens,
        )
    ):
        return
    raise ValueError(
        f"proposal {proposal.id} action_payload evidence_quote is not grounded "
        "in deterministic artifacts"
    )


def _validate_action_payload_reference(
    proposal: ImprovementProposal,
    reference_tokens: _EvidenceReferenceTokens,
    key: str,
    token_key: str,
) -> None:
    value = proposal.action_payload.get(key)
    if value is None or value == "":
        return
    normalized = _normalize_reference_part(value)
    if normalized in reference_tokens.by_key[token_key]:
        return
    raise ValueError(
        f"proposal {proposal.id} action_payload {key} is not grounded "
        "in deterministic artifacts"
    )


def _grounded_reference_tokens(
    output_dir: Path, previous_outputs: dict[str, Any]
) -> _EvidenceReferenceTokens:
    tokens = _EvidenceReferenceTokens()
    snapshot = _read_json(output_dir / "snapshots" / "kg_snapshot.json")
    quality = _read_json(output_dir / "snapshots" / "quality_score.json")
    text_corpus_parts: list[str] = []

    if isinstance(snapshot, dict):
        text_corpus_parts.append(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
        _add_typed_reference_values(tokens, "file_path", snapshot.get("source_files"))
        for node in _list_of_dicts(snapshot.get("nodes")):
            node_text = json.dumps(node, ensure_ascii=False, sort_keys=True)
            _add_scoped_reference_text(
                tokens,
                node_text,
                node.get("source_id"),
                node.get("file_path"),
            )
            _add_typed_reference_token(tokens, "entity_id", node.get("id"))
            _add_typed_reference_token(tokens, "item_id", node.get("id"))
            _add_typed_reference_token(tokens, "source_id", node.get("source_id"))
            _add_typed_reference_token(tokens, "file_path", node.get("file_path"))
        for edge in _list_of_dicts(snapshot.get("edges")):
            edge_text = json.dumps(edge, ensure_ascii=False, sort_keys=True)
            _add_scoped_reference_text(
                tokens,
                edge_text,
                edge.get("source_id"),
                edge.get("file_path"),
            )
            _add_typed_reference_token(tokens, "relation_id", edge.get("id"))
            _add_typed_reference_token(tokens, "item_id", edge.get("id"))
            _add_typed_reference_token(tokens, "entity_id", edge.get("source"))
            _add_typed_reference_token(tokens, "entity_id", edge.get("target"))
            _add_typed_reference_token(tokens, "item_id", edge.get("source"))
            _add_typed_reference_token(tokens, "item_id", edge.get("target"))
            _add_typed_reference_token(tokens, "source_id", edge.get("source_id"))
            _add_typed_reference_token(tokens, "file_path", edge.get("file_path"))

    if isinstance(quality, dict):
        text_corpus_parts.append(json.dumps(quality, ensure_ascii=False, sort_keys=True))
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
        _add_quality_issue_reference_tokens(tokens, quality.get("details"))

    for relative_path in (
        "kb_context.md",
        "entity_catalog.md",
        "relation_catalog.md",
        "kg_structure.md",
        "quality_report.md",
    ):
        text = _read_optional_text_artifact(output_dir / relative_path)
        if text:
            text_corpus_parts.append(text)
            _add_text_artifact_scopes(tokens, text)
    tokens.text_corpus = _normalize_reference_part("\n".join(text_corpus_parts))
    return tokens


def _candidate_quote_references_declared_source(
    action_payload: dict[str, Any],
    normalized_quote: str,
    reference_tokens: _EvidenceReferenceTokens,
) -> bool:
    if normalized_quote not in reference_tokens.text_corpus:
        return False

    scoped_corpora: list[str] = []
    for key in ("source_id", "file_path"):
        normalized_ref = _normalize_reference_part(action_payload.get(key))
        if not normalized_ref:
            continue
        scoped_corpora.extend(reference_tokens.text_corpus_by_reference.get(normalized_ref, []))
    if scoped_corpora:
        return any(normalized_quote in corpus for corpus in scoped_corpora)
    return True


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

        value_parts = _structured_evidence_value_parts(value)
        non_empty_values = [part for part in value_parts if part]
        if not non_empty_values:
            return False
        allowed_tokens = reference_tokens.by_key[normalized_key]
        for value_part in non_empty_values:
            if _normalize_reference_part(value_part) not in allowed_tokens:
                return False
        has_value = True

    return has_value


def _structured_evidence_value_parts(value: str) -> list[str]:
    normalized = (
        value.replace(GRAPH_FIELD_SEP, ",")
        .replace("|", ",")
        .replace("\n", ",")
        .replace("\r", ",")
    )
    return [part.strip() for part in normalized.split(",")]


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
        if token:
            tokens.by_key[key].add(token)


def _add_scoped_reference_text(
    tokens: _EvidenceReferenceTokens, text: str, *references: Any
) -> None:
    normalized_text = _normalize_reference_part(text)
    if not normalized_text:
        return
    for reference in references:
        normalized_reference = _normalize_reference_part(reference)
        if not normalized_reference:
            continue
        tokens.text_corpus_by_reference.setdefault(normalized_reference, []).append(
            normalized_text
        )


def _add_text_artifact_scopes(tokens: _EvidenceReferenceTokens, text: str) -> None:
    normalized_text = _normalize_reference_part(text)
    if not normalized_text:
        return
    for key in ("source_id", "file_path"):
        for reference in tokens.by_key[key]:
            if reference in normalized_text:
                tokens.text_corpus_by_reference.setdefault(reference, []).append(
                    normalized_text
                )


def _add_quality_reference_token(
    tokens: _EvidenceReferenceTokens, key: str, value: Any
) -> None:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return
    _add_typed_reference_token(tokens, "metric", f"quality:{key}={value}")


def _add_quality_issue_reference_tokens(
    tokens: _EvidenceReferenceTokens, details: Any
) -> None:
    if not isinstance(details, dict):
        return
    for issue_key in ("medical_schema_issues", "entity_cleanup_issues"):
        for issue in _list_of_dicts(details.get(issue_key)):
            _add_issue_id_reference_tokens(tokens, issue)


def _add_issue_id_reference_tokens(
    tokens: _EvidenceReferenceTokens, value: Any
) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if str(key).endswith("_id"):
                _add_reference_values(tokens, nested_value)
            else:
                _add_issue_id_reference_tokens(tokens, nested_value)
        return
    if isinstance(value, list):
        for item in value:
            _add_issue_id_reference_tokens(tokens, item)


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
    if config.max_stage_retries > MAX_STAGE_RETRIES:
        raise ValueError(f"max_stage_retries must be <= {MAX_STAGE_RETRIES}")
    if config.max_subagent_tasks <= 0:
        raise ValueError("max_subagent_tasks must be greater than 0")
    if config.max_subagent_tasks > MAX_SUBAGENT_TASKS:
        raise ValueError(f"max_subagent_tasks must be <= {MAX_SUBAGENT_TASKS}")
    if config.max_parallel_subagents <= 0:
        raise ValueError("max_parallel_subagents must be greater than 0")
    if config.max_parallel_subagents > MAX_PARALLEL_SUBAGENTS:
        raise ValueError(
            f"max_parallel_subagents must be <= {MAX_PARALLEL_SUBAGENTS}"
        )
    if config.max_proposals_per_run <= 0:
        raise ValueError("max_proposals_per_run must be greater than 0")
    if config.max_proposals_per_run > MAX_PROPOSALS_PER_RUN:
        raise ValueError(f"max_proposals_per_run must be <= {MAX_PROPOSALS_PER_RUN}")
    if config.deterministic_family_caps is not None:
        normalize_deterministic_family_caps(config.deterministic_family_caps)


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
    _sanitize_trace_errors(trace)
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
    reason = _single_line(_sanitize_error_text(error)) or "judge stage failed"
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
    unavailable_stage = str(stage_trace.get("stage") or "judge")
    stage_trace["state"] = (
        "judge_unavailable"
        if unavailable_stage == "judge"
        else f"{unavailable_stage}_unavailable"
    )
    stage_trace["error"] = reason
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
    error = _sanitize_error_text(error)
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
            "attempt_logs": [],
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
    error = _sanitize_error_text(latest_stage.get("error", "")).strip()
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
                clean_error = _single_line(_sanitize_error_text(attempt_error))
                if clean_error:
                    report_lines.append(
                        f"- {_failure_attempt_label(item, attempt)}: {clean_error}"
                    )
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
    parts = [
        prompt.strip(),
        "",
        f"profile: {profile or 'default'}",
    ]
    schema_prompt = render_medical_relation_schema_prompt(profile)
    if schema_prompt:
        parts.extend(["", schema_prompt.strip()])
    parts.append("Return only JSON.")
    return "\n".join(parts)


def _failure_attempt_label(item: dict[str, Any], attempt: int) -> str:
    task_id = str(item.get("task_id", "")).strip()
    role = str(item.get("role", "")).strip()
    if task_id and role:
        return f"{task_id} ({role}) attempt {attempt}"
    if task_id:
        return f"{task_id} attempt {attempt}"
    return f"Attempt {attempt}"


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


def _read_optional_text_artifact(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


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
