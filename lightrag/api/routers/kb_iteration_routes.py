from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterator, Literal, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from lightrag.kb_iteration.agent_pipeline import (
    LLMAgentPipelineConfig,
    run_llm_agent_pipeline,
)
from lightrag.kb_iteration.apply import (
    APPLY_RESULT_MARKDOWN,
    AcceptedApplyResult,
    apply_accepted_changes_to_graph,
    load_proposals_by_id,
    write_apply_result_artifacts,
)
from lightrag.kb_iteration.zh_artifacts import (
    artifact_zh_relative_path,
    ensure_zh_artifact,
)
from lightrag.kb_iteration.proposals import validate_proposal_id
from lightrag.kb_iteration.review_loop import (
    LLMReviewLoopConfig,
    run_llm_review_loop,
)
from lightrag.kb_iteration.runner import PENDING_REVIEW_PHASE, run_iteration
from lightrag.utils import always_get_an_event_loop, validate_workspace

from ..utils_api import get_combined_auth_dependency


RUN_ID_LATEST = "latest"
UNMARKED_RELATION_LABEL = "未标注关系"
GENERIC_RELATION_LABELS = {"", "邻接", "相关", "关联", "neighbor", "adjacent"}

ARTIFACTS: dict[str, tuple[str, str]] = {
    "kg_snapshot": ("snapshots/kg_snapshot.json", "application/json"),
    "entity_stats": ("snapshots/entity_stats.json", "application/json"),
    "relation_stats": ("snapshots/relation_stats.json", "application/json"),
    "hierarchy_paths": ("snapshots/hierarchy_paths.json", "application/json"),
    "source_coverage": ("snapshots/source_coverage.json", "application/json"),
    "quality_score": ("snapshots/quality_score.json", "application/json"),
    "diff_summary": ("snapshots/diff_summary.json", "application/json"),
    "kb_context": ("kb_context.md", "text/markdown"),
    "entity_catalog": ("entity_catalog.md", "text/markdown"),
    "relation_catalog": ("relation_catalog.md", "text/markdown"),
    "kg_structure": ("kg_structure.md", "text/markdown"),
    "quality_report": ("quality_report.md", "text/markdown"),
    "iteration_log": ("iteration_log.md", "text/markdown"),
    "approval_queue": ("approval_queue.md", "text/markdown"),
    "proposal_revision_requests": (
        "proposal_revision_requests.md",
        "text/markdown",
    ),
    "improvement_backlog": ("improvement_backlog.md", "text/markdown"),
    "accepted_changes": ("accepted_changes.md", "text/markdown"),
    "accepted_changes_execution": (
        "accepted_changes_execution.md",
        "text/markdown",
    ),
    "rejected_changes": ("rejected_changes.md", "text/markdown"),
    "deferred_changes": ("deferred_changes.md", "text/markdown"),
    "accepted_changes_apply_result": (APPLY_RESULT_MARKDOWN, "text/markdown"),
    "diff_report": ("diff_report.md", "text/markdown"),
    "quality_rules": ("quality_rules.md", "text/markdown"),
    "known_issues": ("known_issues.md", "text/markdown"),
    "llm_review_trace": ("llm_review_trace.json", "application/json"),
    "llm_review_report": ("llm_review_report.md", "text/markdown"),
    "llm_judge_report": ("llm_judge_report.md", "text/markdown"),
    "llm_issue_analysis": ("llm_issue_analysis.md", "text/markdown"),
    "llm_missing_branch_inference": (
        "llm_missing_branch_inference.md",
        "text/markdown",
    ),
    "llm_evidence_map": ("llm_evidence_map.md", "text/markdown"),
    "llm_repair_plan": ("llm_repair_plan.md", "text/markdown"),
    "proposals_generated": ("proposals.generated.yaml", "text/markdown"),
}

DECISION_FILES = {
    "accept": "accepted_changes.md",
    "reject": "rejected_changes.md",
    "defer": "deferred_changes.md",
}
_RUN_LOCKS: dict[str, Lock] = {}
_RUN_LOCKS_GUARD = Lock()


class ProposalDecisionRequest(BaseModel):
    reviewer: str = Field(default="maintainer", min_length=1, max_length=120)
    reason: str = Field(default="", max_length=4000)
    impact_scope: str = Field(default="", max_length=1000)
    verification: str = Field(default="", max_length=1000)


class ProposalRevisionRequest(BaseModel):
    reviewer: str = Field(default="maintainer", min_length=1, max_length=120)
    reason: str = Field(default="", max_length=4000)
    instruction: str = Field(default="", max_length=4000)


class RunIterationRequest(BaseModel):
    profile: Optional[str] = Field(default=None, max_length=200)


class RunLLMReviewRequest(BaseModel):
    profile: Optional[str] = Field(default=None, max_length=200)
    mode: Literal["agent_pipeline", "loop"] = "agent_pipeline"
    max_review_rounds: int = Field(default=4, ge=1, le=10)
    max_focus_items_per_round: int = Field(default=3, ge=1, le=10)
    max_context_tokens_per_round: int = Field(default=12000, ge=1000, le=100000)
    max_stage_retries: int = Field(default=5, ge=0, le=8)
    allow_llm_judge: bool = True
    allow_llm_auto_accept: bool = False
    allow_low_risk_auto_reject: bool = True
    generate_patch_candidates: bool = False
    require_human_for_mutation: bool = True


class _UnavailableLLMReviewClient:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("LLM review client is not configured")


class _DisplayArtifactFallbackClient:
    def __init__(self, error: str) -> None:
        self.error = error

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt, user_prompt
        raise RuntimeError(self.error)


class _OpenAICompatibleLLMReviewClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        timeout: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        from lightrag.llm.openai import openai_complete_if_cache

        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            openai_complete_if_cache(
                self.model,
                user_prompt,
                system_prompt=system_prompt,
                history_messages=[],
                base_url=self.base_url,
                api_key=self.api_key,
                response_format={"type": "json_object"},
                timeout=self.timeout,
            )
        )


def _default_llm_review_client(rag):
    _ = rag
    binding = os.getenv("KB_ITERATION_LLM_BINDING", "").strip().lower()
    model = os.getenv("KB_ITERATION_LLM_MODEL", "").strip()
    base_url = os.getenv("KB_ITERATION_LLM_BINDING_HOST", "").strip()
    api_key = os.getenv("KB_ITERATION_LLM_BINDING_API_KEY", "").strip()
    if not any([binding, model, base_url, api_key]):
        return _UnavailableLLMReviewClient()
    if binding != "openai":
        raise RuntimeError(
            "KB iteration LLM review only supports KB_ITERATION_LLM_BINDING=openai"
        )
    if not model or not base_url or not api_key:
        return _UnavailableLLMReviewClient()
    return _OpenAICompatibleLLMReviewClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=_optional_int_env("KB_ITERATION_LLM_TIMEOUT"),
    )


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def create_kb_iteration_routes(rag, args, api_key: Optional[str] = None):
    router = APIRouter(prefix="/kb-iteration", tags=["kb-iteration"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/workspaces", dependencies=[Depends(combined_auth)])
    async def list_workspaces() -> dict[str, Any]:
        output_root = _output_root(args)
        if not output_root.exists():
            return {"workspaces": []}

        workspaces = []
        for child in sorted(output_root.iterdir(), key=lambda path: path.name):
            if not child.is_dir():
                continue
            try:
                workspaces.append(_validate_workspace_or_400(child.name))
            except HTTPException:
                continue
        return {"workspaces": workspaces}

    @router.get("/{workspace}/summary", dependencies=[Depends(combined_auth)])
    async def get_summary(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _build_summary(args, workspace)

    @router.get("/{workspace}/runs", dependencies=[Depends(combined_auth)])
    async def list_runs(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        package_dir = _workspace_dir(args, workspace)
        runs = []
        if package_dir.exists():
            runs.append(_build_run_summary(args, workspace, RUN_ID_LATEST))
        return {"workspace": workspace, "runs": runs}

    @router.post("/{workspace}/runs", dependencies=[Depends(combined_auth)])
    async def create_run(
        workspace: str, request: RunIterationRequest | None = None
    ) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        lock = _run_lock_for_workspace(workspace)
        if not lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409, detail="KB iteration run is already in progress"
            )
        try:
            with _exclusive_workspace_file_lock(_output_root(args), workspace):
                await asyncio.to_thread(
                    run_iteration,
                    workspace=workspace,
                    storage_root=_storage_root(args),
                    input_root=_input_root(args),
                    output_root=_output_root(args),
                    profile=request.profile if request else None,
                )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            lock.release()

        return _build_summary(args, workspace)

    @router.post("/{workspace}/llm-review/runs", dependencies=[Depends(combined_auth)])
    async def create_llm_review_run(
        workspace: str, request: RunLLMReviewRequest | None = None
    ) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        request = request or RunLLMReviewRequest()
        package_dir = _workspace_dir(args, workspace)
        if not package_dir.is_dir():
            raise HTTPException(
                status_code=404, detail="KB iteration workspace package not found"
            )
        run_lock = _run_lock_for_workspace(workspace)
        llm_review_lock = _run_lock_for_workspace(f"{workspace}:llm-review")
        acquired_locks: list[Lock] = []
        try:
            if not run_lock.acquire(blocking=False):
                raise HTTPException(
                    status_code=409, detail="KB iteration run is already in progress"
                )
            acquired_locks.append(run_lock)
            if not llm_review_lock.acquire(blocking=False):
                raise HTTPException(
                    status_code=409,
                    detail="KB LLM review run is already in progress",
                )
            acquired_locks.append(llm_review_lock)
            with _exclusive_workspace_file_lock(_output_root(args), workspace):
                with _exclusive_workspace_file_lock(
                    _output_root(args), workspace, ".kb_iteration_llm_review.lock"
                ):
                    client = _default_llm_review_client(rag)
                    if isinstance(client, _UnavailableLLMReviewClient):
                        raise RuntimeError("LLM review client is not configured")
                    if request.mode == "agent_pipeline":
                        result = await asyncio.to_thread(
                            run_llm_agent_pipeline,
                            workspace=workspace,
                            package_dir=package_dir,
                            client=client,
                            config=LLMAgentPipelineConfig(
                                max_context_tokens_per_stage=(
                                    request.max_context_tokens_per_round
                                ),
                                max_stage_retries=request.max_stage_retries,
                                allow_llm_judge=request.allow_llm_judge,
                                generate_patch_candidates=(
                                    request.generate_patch_candidates
                                ),
                                require_human_for_mutation=(
                                    request.require_human_for_mutation
                                ),
                            ),
                            profile=request.profile,
                        )
                    else:
                        result = await asyncio.to_thread(
                            run_llm_review_loop,
                            workspace=workspace,
                            package_dir=package_dir,
                            client=client,
                            config=LLMReviewLoopConfig(
                                max_review_rounds=request.max_review_rounds,
                                max_focus_items_per_round=(
                                    request.max_focus_items_per_round
                                ),
                                max_context_tokens_per_round=(
                                    request.max_context_tokens_per_round
                                ),
                                allow_llm_judge=request.allow_llm_judge,
                                allow_llm_auto_accept=request.allow_llm_auto_accept,
                                allow_low_risk_auto_reject=(
                                    request.allow_low_risk_auto_reject
                                ),
                                generate_patch_candidates=(
                                    request.generate_patch_candidates
                                ),
                                require_human_for_mutation=(
                                    request.require_human_for_mutation
                                ),
                            ),
                            profile=request.profile,
                        )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            for acquired_lock in reversed(acquired_locks):
                acquired_lock.release()

        return {
            "workspace": workspace,
            "stopReason": result.stop_reason,
            "proposalIds": result.proposal_ids,
        }

    @router.get("/{workspace}/llm-review/trace", dependencies=[Depends(combined_auth)])
    async def get_llm_review_trace(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _read_artifact(args, workspace, "llm_review_trace")

    @router.get("/{workspace}/llm-review/report", dependencies=[Depends(combined_auth)])
    async def get_llm_review_report(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _read_artifact(args, workspace, "llm_review_report")

    @router.get(
        "/{workspace}/llm-review/proposals", dependencies=[Depends(combined_auth)]
    )
    async def get_llm_review_proposals(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _read_artifact(args, workspace, "proposals_generated")

    @router.get(
        "/{workspace}/llm-review/judge-report",
        dependencies=[Depends(combined_auth)],
    )
    async def get_llm_review_judge_report(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _read_artifact(args, workspace, "llm_judge_report")

    @router.get(
        "/{workspace}/llm-review/context/{round_id}",
        dependencies=[Depends(combined_auth)],
    )
    async def get_llm_review_context(workspace: str, round_id: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        if not re.fullmatch(r"round-\d{3}", round_id):
            raise HTTPException(status_code=400, detail="Invalid LLM review round id")
        path = _safe_workspace_path(
            args, workspace, Path("review_context") / f"{round_id}-context.json"
        )
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="LLM review context not found")
        return {
            "workspace": workspace,
            "roundId": round_id,
            "contentType": "application/json",
            "payload": _load_json(path),
        }

    @router.get(
        "/{workspace}/llm-review/patches/{proposal_id}",
        dependencies=[Depends(combined_auth)],
    )
    async def get_llm_review_patch(workspace: str, proposal_id: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        proposal_id = _validate_proposal_id_or_400(proposal_id)
        path = _safe_patch_candidate_path(args, workspace, proposal_id)
        if not path.exists() or not path.is_file():
            raise HTTPException(
                status_code=404, detail="LLM review patch candidate not found"
            )
        return {
            "artifactKey": f"patch_candidates/{proposal_id}.patch",
            "workspace": workspace,
            "proposalId": proposal_id,
            "contentType": "text/x-diff",
            "content": path.read_text(encoding="utf-8"),
        }

    @router.get("/{workspace}/runs/{run_id}", dependencies=[Depends(combined_auth)])
    async def get_run(workspace: str, run_id: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        _require_latest_run(run_id)
        return _build_run_summary(args, workspace, run_id)

    @router.get(
        "/{workspace}/runs/{run_id}/snapshot",
        dependencies=[Depends(combined_auth)],
    )
    async def get_run_snapshot(workspace: str, run_id: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        _require_latest_run(run_id)
        return _read_json_artifact(args, workspace, "kg_snapshot")

    @router.get(
        "/{workspace}/runs/{run_id}/quality",
        dependencies=[Depends(combined_auth)],
    )
    async def get_run_quality(workspace: str, run_id: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        _require_latest_run(run_id)
        return _quality_response(args, workspace)

    @router.get(
        "/{workspace}/runs/{run_id}/catalog/entities",
        dependencies=[Depends(combined_auth)],
    )
    async def get_run_entity_catalog(workspace: str, run_id: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        _require_latest_run(run_id)
        return _entity_catalog_response(args, workspace)

    @router.get(
        "/{workspace}/runs/{run_id}/catalog/relations",
        dependencies=[Depends(combined_auth)],
    )
    async def get_run_relation_catalog(workspace: str, run_id: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        _require_latest_run(run_id)
        return _relation_catalog_response(args, workspace)

    @router.get(
        "/{workspace}/runs/{run_id}/diff",
        dependencies=[Depends(combined_auth)],
    )
    async def get_run_diff(workspace: str, run_id: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        _require_latest_run(run_id)
        return _diff_response(args, workspace)

    @router.get(
        "/{workspace}/runs/{run_id}/artifacts",
        dependencies=[Depends(combined_auth)],
    )
    async def list_run_artifacts(workspace: str, run_id: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        _require_latest_run(run_id)
        return {
            "workspace": workspace,
            "runId": RUN_ID_LATEST,
            "artifacts": _artifact_manifest(args, workspace),
        }

    @router.get(
        "/{workspace}/runs/{run_id}/artifacts/{artifact_key:path}",
        dependencies=[Depends(combined_auth)],
    )
    async def get_run_artifact(
        workspace: str, run_id: str, artifact_key: str
    ) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        _require_latest_run(run_id)
        return _read_artifact(args, workspace, artifact_key)

    @router.get(
        "/{workspace}/artifacts/{artifact_key:path}/display",
        dependencies=[Depends(combined_auth)],
    )
    async def get_display_artifact(
        workspace: str, artifact_key: str
    ) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return await asyncio.to_thread(
            _display_artifact_response,
            args,
            rag,
            workspace,
            artifact_key,
            force=False,
        )

    @router.post(
        "/{workspace}/artifacts/{artifact_key:path}/display/regenerate",
        dependencies=[Depends(combined_auth)],
    )
    async def regenerate_display_artifact(
        workspace: str, artifact_key: str
    ) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return await asyncio.to_thread(
            _display_artifact_response,
            args,
            rag,
            workspace,
            artifact_key,
            force=True,
        )

    @router.get(
        "/{workspace}/artifacts/{artifact_key:path}",
        dependencies=[Depends(combined_auth)],
    )
    async def get_artifact(workspace: str, artifact_key: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _read_artifact(args, workspace, artifact_key)

    @router.get("/{workspace}/snapshot", dependencies=[Depends(combined_auth)])
    async def get_snapshot(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _read_json_artifact(args, workspace, "kg_snapshot")

    @router.get("/{workspace}/quality", dependencies=[Depends(combined_auth)])
    async def get_quality(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _quality_response(args, workspace)

    @router.get("/{workspace}/catalog/entities", dependencies=[Depends(combined_auth)])
    async def get_entity_catalog(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _entity_catalog_response(args, workspace)

    @router.get("/{workspace}/catalog/relations", dependencies=[Depends(combined_auth)])
    async def get_relation_catalog(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _relation_catalog_response(args, workspace)

    @router.get("/{workspace}/graph", dependencies=[Depends(combined_auth)])
    async def get_graph(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _build_graph_response(args, workspace)

    @router.get("/{workspace}/diff", dependencies=[Depends(combined_auth)])
    async def get_diff(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _diff_response(args, workspace)

    @router.get("/{workspace}/rules", dependencies=[Depends(combined_auth)])
    async def get_rules(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return {
            "workspace": workspace,
            "qualityRules": _optional_text_artifact(
                args, workspace, "quality_rules", ""
            ),
            "knownIssues": _optional_text_artifact(args, workspace, "known_issues", ""),
            "acceptedChanges": _optional_text_artifact(
                args, workspace, "accepted_changes", ""
            ),
            "rejectedChanges": _optional_text_artifact(
                args, workspace, "rejected_changes", ""
            ),
        }

    @router.post(
        "/{workspace}/accepted-changes/execute",
        dependencies=[Depends(combined_auth)],
    )
    async def execute_accepted_changes(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        package_dir = _workspace_dir(args, workspace)
        if not package_dir.is_dir():
            raise HTTPException(
                status_code=404, detail="KB iteration workspace package not found"
            )

        run_lock = _run_lock_for_workspace(workspace)
        execution_lock = _run_lock_for_workspace(f"{workspace}:accepted-execution")
        acquired_locks: list[Lock] = []
        try:
            if not run_lock.acquire(blocking=False):
                raise HTTPException(
                    status_code=409, detail="KB iteration run is already in progress"
                )
            acquired_locks.append(run_lock)
            if not execution_lock.acquire(blocking=False):
                raise HTTPException(
                    status_code=409,
                    detail="KB accepted changes execution is already in progress",
                )
            acquired_locks.append(execution_lock)
            with _exclusive_workspace_file_lock(_output_root(args), workspace):
                records = _accepted_decision_records(args, workspace)
                quality_before = _optional_json_artifact(
                    args, workspace, "quality_score", {}
                )
                if not records:
                    result = _accepted_apply_result_with_quality(
                        AcceptedApplyResult(
                            workspace=workspace,
                            applied_at=datetime.now(timezone.utc).isoformat(),
                            source_artifact="accepted_changes.md",
                            proposal_ids=[],
                        ),
                        quality_before,
                        quality_before,
                    )
                    write_apply_result_artifacts(result, package_dir)
                    _append_accepted_apply_log(args, workspace, result)
                    return {
                        "workspace": workspace,
                        "status": "no_accepted_changes",
                        "proposalIds": [],
                        "appliedCount": 0,
                        "blockedCount": 0,
                        "artifactKey": "accepted_changes_apply_result",
                    }

                proposals_by_id = load_proposals_by_id(package_dir)
                result = await apply_accepted_changes_to_graph(
                    rag=rag,
                    workspace=workspace,
                    records=records,
                    proposals_by_id=proposals_by_id,
                )
                rerun_error: FileNotFoundError | ValueError | None = None
                if result.applied_count > 0:
                    try:
                        await asyncio.to_thread(
                            run_iteration,
                            workspace=workspace,
                            storage_root=_storage_root(args),
                            input_root=_input_root(args),
                            output_root=_output_root(args),
                            profile=_current_snapshot_profile(args, workspace),
                        )
                    except (FileNotFoundError, ValueError) as exc:
                        rerun_error = exc
                quality_after = _optional_json_artifact(
                    args, workspace, "quality_score", {}
                )
                result = _accepted_apply_result_with_quality(
                    result, quality_before, quality_after
                )
                write_apply_result_artifacts(result, package_dir)
                _append_accepted_apply_log(
                    args, workspace, result, rerun_error=rerun_error
                )
                if rerun_error is not None:
                    _raise_accepted_apply_rerun_error(rerun_error)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            for acquired_lock in reversed(acquired_locks):
                acquired_lock.release()

        return {
            "workspace": workspace,
            "status": (
                "applied_changes"
                if result.applied_count > 0
                else "no_applicable_changes"
            ),
            "proposalIds": result.proposal_ids,
            "appliedCount": result.applied_count,
            "blockedCount": result.blocked_count,
            "artifactKey": "accepted_changes_apply_result",
        }

    @router.get(
        "/{workspace}/evidence/{source_id:path}", dependencies=[Depends(combined_auth)]
    )
    async def get_evidence(workspace: str, source_id: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        snapshot = _read_json_artifact(args, workspace, "kg_snapshot")
        nodes = [
            node
            for node in snapshot.get("nodes", [])
            if _contains_provenance(node.get("source_id", ""), source_id)
        ]
        edges = [
            edge
            for edge in snapshot.get("edges", [])
            if _contains_provenance(edge.get("source_id", ""), source_id)
        ]
        return {
            "workspace": workspace,
            "sourceId": source_id,
            "nodes": nodes,
            "edges": edges,
        }

    @router.post(
        "/{workspace}/proposals/{proposal_id}/revision-request",
        dependencies=[Depends(combined_auth)],
    )
    async def record_proposal_revision_request(
        workspace: str,
        proposal_id: str,
        request: ProposalRevisionRequest | None = None,
    ) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        proposal_id = _validate_proposal_id_or_400(proposal_id)
        with _exclusive_workspace_file_lock(
            _output_root(args), workspace, ".kb_iteration_decisions.lock"
        ):
            record = _append_proposal_revision_request(
                args,
                workspace,
                proposal_id,
                request or ProposalRevisionRequest(),
            )
        return {
            "workspace": workspace,
            "proposalId": proposal_id,
            "artifactKey": "proposal_revision_requests",
            "record": record,
        }

    @router.post(
        "/{workspace}/proposals/{proposal_id}/{decision}",
        dependencies=[Depends(combined_auth)],
    )
    async def record_proposal_decision(
        workspace: str,
        proposal_id: str,
        decision: Literal["accept", "reject", "defer"],
        request: ProposalDecisionRequest,
    ) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        proposal_id = _validate_proposal_id_or_400(proposal_id)
        with _exclusive_workspace_file_lock(
            _output_root(args), workspace, ".kb_iteration_decisions.lock"
        ):
            record = _append_decision(args, workspace, proposal_id, decision, request)
        return {
            "workspace": workspace,
            "proposalId": proposal_id,
            "decision": decision,
            "record": record,
        }

    return router


def _validate_workspace_or_400(workspace: str) -> str:
    try:
        return validate_workspace(workspace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_proposal_id_or_400(proposal_id: str) -> str:
    try:
        validate_proposal_id(proposal_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return proposal_id


def _output_root(args) -> Path:
    configured = getattr(args, "kb_iteration_output_dir", None)
    return Path(configured) if configured else Path("work") / "kb-iteration"


def _storage_root(args) -> Path:
    return Path(getattr(args, "working_dir", "data/rag_storage"))


def _input_root(args) -> Path:
    return Path(getattr(args, "input_dir", "data/inputs"))


def _workspace_dir(args, workspace: str) -> Path:
    workspace = _validate_workspace_or_400(workspace)
    return _output_root(args) / workspace


def _safe_workspace_path(args, workspace: str, relative_path: Path) -> Path:
    base_dir = _workspace_dir(args, workspace).resolve()
    path = (base_dir / relative_path).resolve()
    try:
        path.relative_to(base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    return path


def _safe_patch_candidate_path(args, workspace: str, proposal_id: str) -> Path:
    patch_dir = (_workspace_dir(args, workspace) / "patch_candidates").resolve()
    path = (patch_dir / f"{proposal_id}.patch").resolve()
    try:
        path.relative_to(patch_dir)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Unsafe patch candidate path"
        ) from exc
    return path


def _safe_artifact_path(args, workspace: str, artifact_key: str) -> tuple[Path, str]:
    if artifact_key not in ARTIFACTS:
        raise HTTPException(status_code=400, detail="Unknown KB iteration artifact")

    relative_path, content_type = ARTIFACTS[artifact_key]
    path = _safe_workspace_path(args, workspace, Path(relative_path))
    return path, content_type


def _require_artifact_file(args, workspace: str, artifact_key: str) -> tuple[Path, str]:
    path, content_type = _safe_artifact_path(args, workspace, artifact_key)
    if not path.exists() or not path.is_file():
        raise HTTPException(
            status_code=404, detail=f"KB iteration artifact not found: {artifact_key}"
        )
    return path, content_type


def _read_artifact(args, workspace: str, artifact_key: str) -> dict[str, Any]:
    path, content_type = _require_artifact_file(args, workspace, artifact_key)
    if content_type == "application/json":
        return {
            "artifactKey": artifact_key,
            "contentType": content_type,
            "payload": _load_json(path),
        }

    return {
        "artifactKey": artifact_key,
        "contentType": content_type,
        "content": path.read_text(encoding="utf-8"),
    }


def _display_artifact_response(
    args,
    rag,
    workspace: str,
    artifact_key: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    source_path, content_type = _require_artifact_file(args, workspace, artifact_key)
    result = ensure_zh_artifact(
        source_path,
        artifact_key=artifact_key,
        content_type=content_type,
        client=_display_artifact_client(rag, content_type),
        model=_zh_model_name(),
        force=force,
        artifact_root=_workspace_dir(args, workspace),
    )
    response = {
        "artifactKey": artifact_key,
        "contentType": content_type,
        "display": {
            "language": "zh",
            "sourceFile": result.source_relative_path.as_posix(),
            "zhFile": result.zh_relative_path.as_posix(),
            "generated": result.generated,
            "fallbackToSource": result.fallback_to_source,
            "generatedAt": result.generated_at,
            "model": result.model,
            "error": _redact_display_error(result.error),
        },
    }
    if content_type == "application/json":
        response["payload"] = result.payload
    else:
        response["content"] = result.content
    return response


def _display_artifact_client(rag, content_type: str):
    if content_type == "application/json":
        return _UnavailableLLMReviewClient()
    try:
        return _default_llm_review_client(rag)
    except Exception as exc:
        return _DisplayArtifactFallbackClient(_redact_display_error(str(exc)) or "")


def _zh_model_name() -> str | None:
    model = os.getenv("KB_ITERATION_LLM_MODEL", "").strip()
    return model or None


def _redact_display_error(error: str | None) -> str | None:
    if not error:
        return None
    redacted = error
    for env_name in ("KB_ITERATION_LLM_BINDING_API_KEY", "OPENAI_API_KEY"):
        secret = os.getenv(env_name, "").strip()
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


def _read_json_artifact(args, workspace: str, artifact_key: str) -> dict[str, Any]:
    path, _ = _require_artifact_file(args, workspace, artifact_key)
    return _load_json(path)


def _optional_json_artifact(
    args, workspace: str, artifact_key: str, default: Any
) -> Any:
    path, _ = _safe_artifact_path(args, workspace, artifact_key)
    if not path.exists() or not path.is_file():
        return default
    return _load_json(path)


def _current_snapshot_profile(args, workspace: str) -> str | None:
    snapshot = _optional_json_artifact(args, workspace, "kg_snapshot", {})
    if not isinstance(snapshot, dict):
        return None
    metadata = snapshot.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    profile = str(metadata.get("profile") or "").strip()
    return profile or None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"Invalid JSON artifact: {path.name}"
        ) from exc


def _read_text_artifact(args, workspace: str, artifact_key: str) -> str:
    path, _ = _require_artifact_file(args, workspace, artifact_key)
    return path.read_text(encoding="utf-8")


def _optional_text_artifact(
    args, workspace: str, artifact_key: str, default: str
) -> str:
    path, _ = _safe_artifact_path(args, workspace, artifact_key)
    if not path.exists() or not path.is_file():
        return default
    return path.read_text(encoding="utf-8")


def _quality_response(args, workspace: str) -> dict[str, Any]:
    return {
        "workspace": workspace,
        "runId": RUN_ID_LATEST,
        "quality": _read_json_artifact(args, workspace, "quality_score"),
        "report": _read_text_artifact(args, workspace, "quality_report"),
    }


def _entity_catalog_response(args, workspace: str) -> dict[str, Any]:
    snapshot = _read_json_artifact(args, workspace, "kg_snapshot")
    return {
        "workspace": workspace,
        "runId": RUN_ID_LATEST,
        "catalog": _read_text_artifact(args, workspace, "entity_catalog"),
        "stats": _optional_json_artifact(args, workspace, "entity_stats", []),
        "entities": snapshot.get("nodes", []),
    }


def _relation_catalog_response(args, workspace: str) -> dict[str, Any]:
    snapshot = _read_json_artifact(args, workspace, "kg_snapshot")
    return {
        "workspace": workspace,
        "runId": RUN_ID_LATEST,
        "catalog": _read_text_artifact(args, workspace, "relation_catalog"),
        "stats": _optional_json_artifact(args, workspace, "relation_stats", []),
        "relations": snapshot.get("edges", []),
    }


def _diff_response(args, workspace: str) -> dict[str, Any]:
    return {
        "workspace": workspace,
        "runId": RUN_ID_LATEST,
        "summary": _optional_json_artifact(args, workspace, "diff_summary", {}),
        "report": _optional_text_artifact(args, workspace, "diff_report", ""),
    }


def _artifact_manifest(args, workspace: str) -> list[dict[str, Any]]:
    manifest = []
    for key, (relative_path, content_type) in ARTIFACTS.items():
        path, _ = _safe_artifact_path(args, workspace, key)
        zh_relative_path = artifact_zh_relative_path(Path(relative_path))
        zh_path = _safe_workspace_path(args, workspace, zh_relative_path)
        manifest.append(
            {
                "key": key,
                "contentType": content_type,
                "exists": path.exists() and path.is_file(),
                "sourceFile": relative_path,
                "display": {
                    "language": "zh",
                    "zhFile": zh_relative_path.as_posix(),
                    "zhExists": zh_path.exists() and zh_path.is_file(),
                },
            }
        )
    return manifest


def _build_summary(args, workspace: str) -> dict[str, Any]:
    snapshot = _read_json_artifact(args, workspace, "kg_snapshot")
    quality = _optional_json_artifact(args, workspace, "quality_score", {})
    approval_queue = _optional_text_artifact(args, workspace, "approval_queue", "")
    iteration_log = _optional_text_artifact(args, workspace, "iteration_log", "")
    source_files = snapshot.get("source_files", [])

    return {
        "workspace": workspace,
        "latestRunId": RUN_ID_LATEST,
        "generatedAt": snapshot.get("generated_at", ""),
        "profile": snapshot.get("metadata", {}).get("profile", ""),
        "phase": _latest_phase(iteration_log),
        "counts": {
            "nodes": len(snapshot.get("nodes", [])),
            "edges": len(snapshot.get("edges", [])),
            "sources": len(source_files),
        },
        "quality": quality,
        "pendingApprovalCount": _count_pending_approvals(approval_queue),
        "highRiskFindingCount": _count_high_risk_findings(quality),
        "artifacts": _artifact_manifest(args, workspace),
    }


def _build_run_summary(args, workspace: str, run_id: str) -> dict[str, Any]:
    _require_latest_run(run_id)
    summary = _build_summary(args, workspace)
    return {
        "workspace": workspace,
        "runId": RUN_ID_LATEST,
        "phase": summary["phase"],
        "generatedAt": summary["generatedAt"],
        "quality": summary["quality"],
        "counts": summary["counts"],
        "artifacts": summary["artifacts"],
    }


def _require_latest_run(run_id: str) -> None:
    if run_id != RUN_ID_LATEST:
        raise HTTPException(
            status_code=404,
            detail="Only the latest KB iteration run is available in this version",
        )


def _latest_phase(iteration_log: str) -> str:
    matches = re.findall(r"^\s*-\s*phase:\s*(\S+)\s*$", iteration_log, re.MULTILINE)
    return matches[-1] if matches else PENDING_REVIEW_PHASE


def _count_pending_approvals(approval_queue: str) -> int:
    proposals = _parse_proposals(approval_queue)
    if proposals:
        return sum(1 for proposal in proposals if proposal.get("requires_approval"))
    return len(re.findall(r"requires_approval:\s*true", approval_queue, re.IGNORECASE))


def _parse_proposals(markdown: str) -> list[dict[str, Any]]:
    start = markdown.find("proposals:")
    if start < 0:
        return []
    try:
        payload = yaml.safe_load(markdown[start:]) or {}
    except yaml.YAMLError:
        return []
    proposals = payload.get("proposals", []) if isinstance(payload, dict) else []
    return [proposal for proposal in proposals if isinstance(proposal, dict)]


def _proposal_by_id(args, workspace: str, proposal_id: str) -> dict[str, Any]:
    proposals = _parse_proposals(
        _optional_text_artifact(args, workspace, "approval_queue", "")
    )
    for proposal in proposals:
        if str(proposal.get("id", "")).strip() == proposal_id:
            return proposal
    raise HTTPException(status_code=404, detail="Proposal is not in the approval queue")


def _existing_decision(
    args, workspace: str, proposal_id: str
) -> tuple[str, dict[str, Any] | None] | None:
    for decision, file_name in DECISION_FILES.items():
        path = _workspace_dir(args, workspace) / file_name
        if not path.exists() or not path.is_file():
            continue
        decision_pattern = rf"^##\s+{re.escape(proposal_id)}\s*$"
        decision_text = path.read_text(encoding="utf-8")
        match = re.search(decision_pattern, decision_text, re.MULTILINE)
        if match:
            return decision, _parse_decision_record(decision_text, match.end())
    return None


def _parse_decision_record(
    decision_text: str, section_start: int
) -> dict[str, Any] | None:
    next_section = re.search(r"^##\s+.+$", decision_text[section_start:], re.MULTILINE)
    section_end = (
        section_start + next_section.start() if next_section else len(decision_text)
    )
    section = decision_text[section_start:section_end]
    json_block = re.search(r"```json\s*(.*?)\s*```", section, re.DOTALL)
    if not json_block:
        return None
    try:
        record = json.loads(json_block.group(1))
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def _accepted_decision_records(args, workspace: str) -> list[dict[str, Any]]:
    decision_text = _optional_text_artifact(args, workspace, "accepted_changes", "")
    records: list[dict[str, Any]] = []
    for match in re.finditer(
        r"^##\s+([A-Za-z0-9_.-]+)\s*$", decision_text, re.MULTILINE
    ):
        proposal_id = _validate_proposal_id_or_400(match.group(1))
        record = _parse_decision_record(decision_text, match.end())
        if not record:
            record = {
                "proposal_id": proposal_id,
                "decision": "accept",
            }
        record["proposal_id"] = str(record.get("proposal_id") or proposal_id)
        records.append(record)
    return records


def _accepted_apply_result_with_quality(
    result: AcceptedApplyResult,
    quality_before: Any,
    quality_after: Any,
) -> AcceptedApplyResult:
    return AcceptedApplyResult(
        workspace=result.workspace,
        applied_at=result.applied_at,
        source_artifact=result.source_artifact,
        proposal_ids=result.proposal_ids,
        changes=result.changes,
        quality_before=quality_before if isinstance(quality_before, dict) else {},
        quality_after=quality_after if isinstance(quality_after, dict) else {},
    )


def _raise_accepted_apply_rerun_error(exc: FileNotFoundError | ValueError) -> None:
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail=str(exc)) from exc


def _record_accepted_changes_execution(
    args,
    workspace: str,
    records: list[dict[str, Any]],
    client,
) -> dict[str, Any]:
    package_dir = _workspace_dir(args, workspace)
    proposal_ids = [str(record["proposal_id"]) for record in records]
    prompt_payload = _accepted_execution_prompt_payload(args, workspace, records)
    raw_response = client.complete(
        system_prompt=_accepted_execution_system_prompt(),
        user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
    )
    execution_payload = _parse_accepted_execution_response(raw_response)
    execution_payload["workspace"] = workspace
    execution_payload["proposal_ids"] = proposal_ids
    execution_payload["executed_at"] = datetime.now(timezone.utc).isoformat()
    execution_payload["source_artifact"] = "accepted_changes.md"

    json_path = package_dir / "accepted_changes_execution.json"
    json_path.write_text(
        json.dumps(execution_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    markdown_path = package_dir / "accepted_changes_execution.md"
    markdown_path.write_text(
        _accepted_execution_markdown(execution_payload),
        encoding="utf-8",
    )
    _append_accepted_execution_log(args, workspace, proposal_ids, execution_payload)

    return {
        "workspace": workspace,
        "status": "execution_recorded",
        "proposalIds": proposal_ids,
        "executedCount": _count_execution_items(
            execution_payload.get("executed_changes")
        ),
        "artifactKey": "accepted_changes_execution",
    }


def _accepted_execution_prompt_payload(
    args, workspace: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "workspace": workspace,
        "accepted_decision_records": records,
        "accepted_changes": _optional_text_artifact(
            args, workspace, "accepted_changes", ""
        ),
        "improvement_backlog": _optional_text_artifact(
            args, workspace, "improvement_backlog", ""
        ),
        "quality_report": _optional_text_artifact(
            args, workspace, "quality_report", ""
        ),
        "kb_context": _optional_text_artifact(args, workspace, "kb_context", ""),
        "llm_repair_plan": _optional_text_artifact(
            args, workspace, "llm_repair_plan", ""
        ),
        "patch_candidates": _patch_candidate_names(args, workspace),
    }


def _accepted_execution_system_prompt() -> str:
    return "\n".join(
        [
            "You are the KB iteration accepted-change execution agent.",
            "Return only JSON.",
            "Use only accepted_decision_records and supplied artifacts.",
            "Do not claim direct KG mutation unless the supplied artifacts prove it.",
            "Constrain every action to the proposal scope, target, evidence, and risk.",
            "Use this schema: summary, executed_changes, blocked_changes, next_steps.",
            "Each executed_changes item must include proposal_id, status, and action.",
        ]
    )


def _patch_candidate_names(args, workspace: str) -> list[str]:
    patch_dir = _workspace_dir(args, workspace) / "patch_candidates"
    if not patch_dir.is_dir():
        return []
    return sorted(path.name for path in patch_dir.glob("*.patch") if path.is_file())


def _parse_accepted_execution_response(raw_response: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Accepted changes execution returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Accepted changes execution must return a JSON object")
    return {
        "summary": str(payload.get("summary") or "").strip(),
        "executed_changes": _list_of_records(payload.get("executed_changes")),
        "blocked_changes": _list_of_records(payload.get("blocked_changes")),
        "next_steps": _list_of_strings(payload.get("next_steps")),
    }


def _list_of_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _count_execution_items(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _accepted_execution_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Accepted Changes Execution",
        "",
        f"- Workspace: {_markdown_inline(payload.get('workspace'))}",
        f"- Executed at: {_markdown_inline(payload.get('executed_at'))}",
        f"- Source: {_markdown_inline(payload.get('source_artifact'))}",
        f"- Proposal IDs: {_markdown_inline(', '.join(payload.get('proposal_ids', [])))}",
        "",
        "## Summary",
        "",
        _markdown_inline(payload.get("summary") or "No summary returned."),
        "",
        "## Executed Changes",
        "",
    ]
    lines.extend(_render_execution_items(payload.get("executed_changes")))
    lines.extend(["", "## Blocked Changes", ""])
    lines.extend(_render_execution_items(payload.get("blocked_changes")))
    lines.extend(["", "## Next Steps", ""])
    next_steps = _list_of_strings(payload.get("next_steps"))
    lines.extend(f"- {_markdown_inline(step)}" for step in next_steps)
    if not next_steps:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def _render_execution_items(value: Any) -> list[str]:
    items = _list_of_records(value)
    if not items:
        return ["- none"]
    rendered = []
    for item in items:
        proposal_id = _markdown_inline(item.get("proposal_id") or "unknown")
        status = _markdown_inline(item.get("status") or "unknown")
        action = _markdown_inline(
            item.get("action") or json.dumps(item, ensure_ascii=False)
        )
        rendered.append(f"- {proposal_id} [{status}]: {action}")
    return rendered


def _markdown_inline(value: Any) -> str:
    return str(value or "").replace("\n", " ").replace("|", "\\|")


def _append_accepted_execution_log(
    args, workspace: str, proposal_ids: list[str], payload: dict[str, Any]
) -> None:
    log_path = _workspace_dir(args, workspace) / "iteration_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = [
        "",
        "## Accepted Changes Execution",
        "",
        "- phase: accepted_changes_execution",
        f"- accepted_proposal_ids: {', '.join(proposal_ids)}",
        "- artifact: accepted_changes_execution.md",
    ]
    summary = str(payload.get("summary") or "").strip()
    if summary:
        entry.append(f"- summary: {_markdown_inline(summary)}")
    entry.append("")
    with log_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(entry))


def _append_accepted_apply_log(
    args,
    workspace: str,
    result: AcceptedApplyResult,
    *,
    rerun_error: FileNotFoundError | ValueError | None = None,
) -> None:
    log_path = _workspace_dir(args, workspace) / "iteration_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_ids = ", ".join(result.proposal_ids) or "none"
    entry = [
        "",
        "## Accepted Changes Apply",
        "",
        "- phase: accepted_changes_apply",
        f"- accepted_proposal_ids: {proposal_ids}",
        f"- applied_count: {result.applied_count}",
        f"- blocked_count: {result.blocked_count}",
        f"- artifact: {APPLY_RESULT_MARKDOWN}",
    ]
    if rerun_error is not None:
        entry.extend(
            [
                "- rerun_status: failed",
                (
                    "- rerun_error: "
                    f"{rerun_error.__class__.__name__}: {_markdown_inline(rerun_error)}"
                ),
            ]
        )
    elif result.applied_count > 0:
        entry.append("- rerun_status: completed")
    else:
        entry.append("- rerun_status: not_run")
    entry.append("")
    with log_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(entry))


def _count_high_risk_findings(quality: dict[str, Any]) -> int:
    findings = quality.get("findings", []) if isinstance(quality, dict) else []
    return sum(
        1
        for finding in findings
        if isinstance(finding, dict) and finding.get("severity") in {"critical", "high"}
    )


def _build_graph_response(args, workspace: str) -> dict[str, Any]:
    snapshot = _read_json_artifact(args, workspace, "kg_snapshot")
    labels = {
        node.get("id", ""): node.get("label") or node.get("id", "")
        for node in snapshot.get("nodes", [])
    }
    nodes = [
        {
            **node,
            "role": _node_role(node),
            "evidenceStatus": _evidence_status(node),
            "qualityFlags": _node_quality_flags(node),
        }
        for node in snapshot.get("nodes", [])
    ]
    edges = []
    for edge in snapshot.get("edges", []):
        label = _relation_label(edge.get("keywords", ""))
        edges.append(
            {
                **edge,
                "label": label,
                "direction": "outgoing",
                "sourceLabel": labels.get(
                    edge.get("source", ""), edge.get("source", "")
                ),
                "targetLabel": labels.get(
                    edge.get("target", ""), edge.get("target", "")
                ),
                "evidenceStatus": _evidence_status(edge),
                "qualityFlags": _edge_quality_flags(edge),
            }
        )
    return {
        "workspace": workspace,
        "runId": RUN_ID_LATEST,
        "generatedAt": snapshot.get("generated_at", ""),
        "nodes": nodes,
        "edges": edges,
        "metadata": snapshot.get("metadata", {}),
    }


def _node_role(node: dict[str, Any]) -> str:
    entity_type = str(node.get("entity_type", "")).strip().lower()
    properties = node.get("properties", {})
    if entity_type == "disease":
        return "disease"
    if entity_type == "medicalgroup":
        return "category"
    if isinstance(properties, dict) and properties.get("medical_group"):
        return "leaf"
    return "entity"


def _relation_label(keywords: str) -> str:
    keyword = str(keywords or "").strip()
    if keyword.lower() in GENERIC_RELATION_LABELS:
        return UNMARKED_RELATION_LABEL
    return keyword


def _evidence_status(item: dict[str, Any]) -> str:
    source_id = str(item.get("source_id", "")).strip()
    file_path = str(item.get("file_path", "")).strip()
    return "grounded" if source_id and file_path else "missing"


def _node_quality_flags(node: dict[str, Any]) -> list[str]:
    return ["missing_evidence"] if _evidence_status(node) == "missing" else []


def _edge_quality_flags(edge: dict[str, Any]) -> list[str]:
    flags = []
    if _evidence_status(edge) == "missing":
        flags.append("missing_evidence")
    if _relation_label(edge.get("keywords", "")) == UNMARKED_RELATION_LABEL:
        flags.append("generic_relation")
    return flags


def _contains_provenance(raw_value: str, source_id: str) -> bool:
    return source_id in {
        part.strip() for part in str(raw_value or "").replace("|", ",").split(",")
    }


def _append_decision(
    args,
    workspace: str,
    proposal_id: str,
    decision: Literal["accept", "reject", "defer"],
    request: ProposalDecisionRequest,
) -> dict[str, Any]:
    package_dir = _workspace_dir(args, workspace)
    package_dir.mkdir(parents=True, exist_ok=True)
    proposal = _proposal_by_id(args, workspace, proposal_id)
    existing_decision = _existing_decision(args, workspace, proposal_id)
    if existing_decision:
        recorded_decision, recorded_record = existing_decision
        if recorded_decision == decision:
            return recorded_record or _build_proposal_decision_record(
                proposal_id,
                decision,
                proposal,
                request,
            )
        raise HTTPException(
            status_code=409,
            detail=(f"Proposal already has a recorded decision: {recorded_decision}"),
        )

    path = package_dir / DECISION_FILES[decision]
    record = _build_proposal_decision_record(proposal_id, decision, proposal, request)
    header = "" if path.exists() else f"# {decision.title()}ed Changes\n"
    entry = (
        f"{header}\n## {proposal_id}\n\n"
        "```json\n"
        f"{json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)}\n"
        "```\n"
    )
    with path.open("a", encoding="utf-8") as file:
        file.write(entry)
    return record


def _append_proposal_revision_request(
    args,
    workspace: str,
    proposal_id: str,
    request: ProposalRevisionRequest,
) -> dict[str, Any]:
    package_dir = _workspace_dir(args, workspace)
    package_dir.mkdir(parents=True, exist_ok=True)
    proposal = _proposal_by_id(args, workspace, proposal_id)
    record = _build_proposal_revision_request_record(
        proposal_id, proposal, request, _existing_decision(args, workspace, proposal_id)
    )

    path = package_dir / "proposal_revision_requests.md"
    header = "" if path.exists() else "# Proposal Revision Requests\n"
    entry = (
        f"{header}\n## {proposal_id}\n\n"
        "```json\n"
        f"{json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)}\n"
        "```\n"
    )
    with path.open("a", encoding="utf-8") as file:
        file.write(entry)
    _append_proposal_revision_request_log(args, workspace, record)
    return record


def _build_proposal_revision_request_record(
    proposal_id: str,
    proposal: dict[str, Any],
    request: ProposalRevisionRequest,
    existing_decision: tuple[str, dict[str, Any] | None] | None,
) -> dict[str, Any]:
    requested_at = datetime.now(timezone.utc).isoformat()
    reason = request.reason.strip() or request.instruction.strip()
    if not reason:
        reason = _default_revision_request_reason(
            proposal_id, proposal, existing_decision
        )
    return {
        "proposal_id": proposal_id,
        "requested_at": requested_at,
        "reviewer": request.reviewer,
        "reason": reason,
        "instruction": request.instruction.strip(),
        "proposal_type": proposal.get("type", ""),
        "proposal_target": proposal.get("target", ""),
        "proposal_risk": proposal.get("risk", ""),
        "proposal_confidence": proposal.get("confidence", ""),
        "proposal_reason": proposal.get("reason", ""),
        "proposal_evidence": proposal.get("evidence", []),
    }


def _default_revision_request_reason(
    proposal_id: str,
    proposal: dict[str, Any],
    existing_decision: tuple[str, dict[str, Any] | None] | None,
) -> str:
    if existing_decision:
        decision, record = existing_decision
        if decision == "reject" and record:
            rejected_reason = str(record.get("reason", "")).strip()
            if rejected_reason:
                return (
                    f"Revise rejected proposal {proposal_id}. "
                    f"Maintainer rejection reason: {rejected_reason}"
                )

    proposal_reason = str(proposal.get("reason", "")).strip() or "not provided"
    return (
        f"Queue a revision for rejected proposal {proposal_id} in the next agent iteration. "
        f"Use the parent proposal reason as context: {proposal_reason}."
    )


def _append_proposal_revision_request_log(
    args, workspace: str, record: dict[str, Any]
) -> None:
    log_path = _workspace_dir(args, workspace) / "iteration_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = [
        "",
        "## Proposal Revision Request",
        "",
        "- phase: proposal_revision_request",
        "- event: revision_request_queued",
        f"- proposal_id: {_markdown_inline(record.get('proposal_id'))}",
        f"- reviewer: {_markdown_inline(record.get('reviewer'))}",
        "- artifact: proposal_revision_requests.md",
    ]
    reason = str(record.get("reason") or "").strip()
    if reason:
        entry.append(f"- reason: {_markdown_inline(reason)}")
    entry.append("")
    with log_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(entry))


def _build_proposal_decision_record(
    proposal_id: str,
    decision: Literal["accept", "reject", "defer"],
    proposal: dict[str, Any],
    request: ProposalDecisionRequest,
) -> dict[str, Any]:
    audit_review = _proposal_decision_audit_defaults(proposal_id, proposal, request)
    return {
        "proposal_id": proposal_id,
        "proposal_type": proposal.get("type", ""),
        "proposal_target": proposal.get("target", ""),
        "decision": decision,
        "reviewer": request.reviewer,
        "reason": audit_review["reason"],
        "impact_scope": audit_review["impact_scope"],
        "verification": audit_review["verification"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def _proposal_decision_audit_defaults(
    proposal_id: str, proposal: dict[str, Any], request: ProposalDecisionRequest
) -> dict[str, str]:
    proposal_type = str(proposal.get("type", "")).strip() or "unknown"
    proposal_target = str(proposal.get("target", "")).strip() or "unknown"
    proposal_reason = str(proposal.get("reason", "")).strip()
    proposal_risk = str(proposal.get("risk", "")).strip() or "unknown"
    evidence = proposal.get("evidence")
    if isinstance(evidence, list) and evidence:
        evidence_summary = "; ".join(
            str(item).strip() for item in evidence if str(item).strip()
        )
    else:
        evidence_summary = "no explicit evidence"

    reason = request.reason.strip() or (
        f"Maintainer selected a decision for proposal {proposal_id}. "
        f"Proposal reason: {proposal_reason or 'not provided'}."
    )
    impact_scope = request.impact_scope.strip() or (
        f"Scope is constrained to proposal type {proposal_type}, "
        f"target {proposal_target}, and risk {proposal_risk}."
    )
    verification = request.verification.strip() or (
        "Use the proposal evidence and LLM judge constraints for review. "
        f"Evidence: {evidence_summary}."
    )
    return {
        "reason": reason,
        "impact_scope": impact_scope,
        "verification": verification,
    }


def _run_lock_for_workspace(workspace: str) -> Lock:
    with _RUN_LOCKS_GUARD:
        lock = _RUN_LOCKS.get(workspace)
        if lock is None:
            lock = Lock()
            _RUN_LOCKS[workspace] = lock
        return lock


@contextmanager
def _exclusive_workspace_file_lock(
    output_root: Path, workspace: str, lock_name: str = ".kb_iteration_run.lock"
) -> Iterator[None]:
    workspace = _validate_workspace_or_400(workspace)
    package_dir = Path(output_root) / workspace
    package_dir.mkdir(parents=True, exist_ok=True)
    lock_path = package_dir / lock_name
    fd: int | None = None
    created_lock = False
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        created_lock = True
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            fd = None
            file.write(
                json.dumps(
                    {
                        "workspace": workspace,
                        "pid": os.getpid(),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        yield
    except FileExistsError as exc:
        raise HTTPException(
            status_code=409, detail="KB iteration workspace is locked"
        ) from exc
    finally:
        if fd is not None:
            os.close(fd)
        if created_lock:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
