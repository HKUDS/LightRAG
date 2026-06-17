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

from lightrag.kb_iteration.runner import PENDING_REVIEW_PHASE, run_iteration
from lightrag.utils import validate_workspace

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
    "improvement_backlog": ("improvement_backlog.md", "text/markdown"),
    "accepted_changes": ("accepted_changes.md", "text/markdown"),
    "rejected_changes": ("rejected_changes.md", "text/markdown"),
    "deferred_changes": ("deferred_changes.md", "text/markdown"),
    "diff_report": ("diff_report.md", "text/markdown"),
    "quality_rules": ("quality_rules.md", "text/markdown"),
    "known_issues": ("known_issues.md", "text/markdown"),
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
    reason: str = Field(min_length=1, max_length=4000)
    impact_scope: str = Field(default="", max_length=1000)
    verification: str = Field(default="", max_length=1000)


class RunIterationRequest(BaseModel):
    profile: Optional[str] = Field(default=None, max_length=200)


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

    @router.get(
        "/{workspace}/catalog/entities", dependencies=[Depends(combined_auth)]
    )
    async def get_entity_catalog(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _entity_catalog_response(args, workspace)

    @router.get(
        "/{workspace}/catalog/relations", dependencies=[Depends(combined_auth)]
    )
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


def _safe_artifact_path(args, workspace: str, artifact_key: str) -> tuple[Path, str]:
    if artifact_key not in ARTIFACTS:
        raise HTTPException(status_code=400, detail="Unknown KB iteration artifact")

    relative_path, content_type = ARTIFACTS[artifact_key]
    base_dir = _workspace_dir(args, workspace).resolve()
    path = (base_dir / relative_path).resolve()
    try:
        path.relative_to(base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
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
    for key, (_relative_path, content_type) in ARTIFACTS.items():
        path, _ = _safe_artifact_path(args, workspace, key)
        manifest.append(
            {
                "key": key,
                "contentType": content_type,
                "exists": path.exists() and path.is_file(),
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


def _existing_decision(args, workspace: str, proposal_id: str) -> str | None:
    for decision, file_name in DECISION_FILES.items():
        path = _workspace_dir(args, workspace) / file_name
        if not path.exists() or not path.is_file():
            continue
        decision_pattern = rf"^##\s+{re.escape(proposal_id)}\s*$"
        decision_text = path.read_text(encoding="utf-8")
        if re.search(decision_pattern, decision_text, re.MULTILINE):
            return decision
    return None


def _count_high_risk_findings(quality: dict[str, Any]) -> int:
    findings = quality.get("findings", []) if isinstance(quality, dict) else []
    return sum(
        1
        for finding in findings
        if isinstance(finding, dict)
        and finding.get("severity") in {"critical", "high"}
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
                "sourceLabel": labels.get(edge.get("source", ""), edge.get("source", "")),
                "targetLabel": labels.get(edge.get("target", ""), edge.get("target", "")),
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
        raise HTTPException(
            status_code=409,
            detail=(
                "Proposal already has a recorded decision: "
                f"{existing_decision}"
            ),
        )

    path = package_dir / DECISION_FILES[decision]
    record = {
        "proposal_id": proposal_id,
        "proposal_type": proposal.get("type", ""),
        "proposal_target": proposal.get("target", ""),
        "decision": decision,
        "reviewer": request.reviewer,
        "reason": request.reason,
        "impact_scope": request.impact_scope,
        "verification": request.verification,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
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
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
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
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
