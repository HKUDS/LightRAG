# KG Iteration Agent Workbench Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the KG iteration maintenance UI into a Chinese, process-first Agent workbench with `.zh` display artifacts, focused approval flow, table-based snapshots, deterministic apply visibility, and validation summaries.

**Architecture:** Keep original artifacts as the only source for system logic, and add `.zh` artifacts as a read-only WebUI display layer. The backend owns `.zh` generation and safe artifact metadata; the frontend owns the 5-step workflow, next-action resolution, artifact display switching, proposal rows, snapshot tables, execution summary, and validation result view.

**Tech Stack:** Python FastAPI routes and pytest for backend; React 19, TypeScript, Bun test, Tailwind, lucide-react, and existing LightRAG WebUI component patterns for frontend.

---

## Scope Check

The design spans backend artifact display support and frontend UI restructuring, but they are tightly coupled: the WebUI cannot show stable Chinese display content without backend `.zh` artifact generation, and the backend display layer is only useful through the redesigned workbench. Keep this as one plan with independently testable tasks.

## File Structure

### Backend files

- Create `lightrag/kb_iteration/zh_artifacts.py`
  - Defines `.zh` artifact path mapping, display metadata, label dictionaries, machine-token preservation rules, JSON label injection, Markdown translation prompts, and `ensure_zh_artifact`.
- Modify `lightrag/api/routers/kb_iteration_routes.py`
  - Adds display artifact endpoints, manifest fields for zh status, regeneration endpoint, and proposal revision endpoint.
- Modify `lightrag/kb_iteration/agent_context.py`
  - Includes proposal revision requests in Agent context so rejected proposals can be regenerated with a parent link.
- Modify `lightrag/kb_iteration/markdown.py`
  - Adds default `proposal_revision_requests.md` content.
- Create or modify backend tests:
  - `tests/kg/test_kb_iteration_zh_artifacts.py`
  - `tests/api/routes/test_kb_iteration_routes.py`
  - `tests/kg/test_kb_iteration_review_context.py`
  - `tests/kg/test_kb_iteration_agent_context.py`

### Frontend files

- Modify `lightrag_webui/src/api/lightrag.ts`
  - Adds display artifact types and endpoints.
- Modify `lightrag_webui/src/stores/kgMaintenance.ts`
  - Replaces old section set with 5 workflow sections.
- Modify `lightrag_webui/src/features/KGMaintenanceConsole.tsx`
  - Loads display artifacts, computes next action, wires workflow actions, removes permanent inspector column.
- Modify `lightrag_webui/src/components/kg-maintenance/kgIterationLoadUtils.ts`
  - Loads `.zh` display artifacts with original fallback metadata.
- Create `lightrag_webui/src/components/kg-maintenance/kgMaintenanceArtifacts.ts`
  - Central artifact catalog grouped by the 5 workflow steps.
- Create `lightrag_webui/src/components/kg-maintenance/kgMaintenanceNextAction.ts`
  - Pure next-action resolver.
- Create `lightrag_webui/src/components/kg-maintenance/kgMaintenanceDisplay.ts`
  - Pure display helpers for Chinese labels, artifact extraction, quality deltas, evidence issue rows, and proposal grouping.
- Modify `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.tsx`
  - Top bar: workspace, refresh, all artifacts. Left nav: 5 steps only. Main area only.
- Create `lightrag_webui/src/components/kg-maintenance/AgentStepHeader.tsx`
  - Renders current state, step badges, and one recommended action.
- Create `lightrag_webui/src/components/kg-maintenance/ArtifactFileSection.tsx`
  - Per-step collapsed associated file section with Chinese/original toggle and regenerate control.
- Create `lightrag_webui/src/components/kg-maintenance/ArtifactDrawer.tsx`
  - All artifacts drawer grouped by workflow step.
- Create `lightrag_webui/src/components/kg-maintenance/SnapshotTables.tsx`
  - Desktop table workspace for nodes, relations, and evidence issues.
- Modify `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx`
  - Stage-by-stage collapsible LLM review panels with associated files.
- Modify `lightrag_webui/src/components/kg-maintenance/QualityAndApprovalPanels.tsx`
  - Simplifies approval rows and removes required reason/impact/verification textareas.
- Create `lightrag_webui/src/components/kg-maintenance/ExecutionAndValidationPanels.tsx`
  - Execution button summary and validation delta view.
- Update frontend tests:
  - `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx`
  - `lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx`
  - `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.test.tsx`
  - `lightrag_webui/src/components/kg-maintenance/QualityAndApprovalPanels.test.tsx`
  - New tests for `kgMaintenanceArtifacts`, `kgMaintenanceNextAction`, `kgMaintenanceDisplay`, `ArtifactFileSection`, `ArtifactDrawer`, and `SnapshotTables`.

---

### Task 1: Backend `.zh` Artifact Generator

**Files:**
- Create: `lightrag/kb_iteration/zh_artifacts.py`
- Test: `tests/kg/test_kb_iteration_zh_artifacts.py`

- [ ] **Step 1: Write failing tests for zh path mapping and JSON label injection**

Add `tests/kg/test_kb_iteration_zh_artifacts.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from lightrag.kb_iteration.zh_artifacts import (
    artifact_zh_relative_path,
    build_zh_json_payload,
    machine_token_should_be_preserved,
)


def test_artifact_zh_relative_path_preserves_extension_and_directory():
    assert artifact_zh_relative_path(Path("quality_report.md")) == Path(
        "quality_report.zh.md"
    )
    assert artifact_zh_relative_path(Path("snapshots/quality_score.json")) == Path(
        "snapshots/quality_score.zh.json"
    )
    assert artifact_zh_relative_path(Path("proposals.generated.yaml")) == Path(
        "proposals.generated.zh.yaml"
    )


def test_build_zh_json_payload_keeps_keys_and_adds_labels():
    payload = {
        "overall": 97,
        "metrics": {"hierarchy_missing_branch_count": 0},
        "findings": [{"message": "No blockers", "source_id": "chunk-1"}],
    }

    result = build_zh_json_payload(payload)

    assert result["overall"] == 97
    assert result["metrics"]["hierarchy_missing_branch_count"] == 0
    assert result["findings"][0]["source_id"] == "chunk-1"
    assert result["_zh_labels"]["overall"] == "总分"
    assert result["_zh_labels"]["metrics"] == "指标"
    assert result["metrics"]["_zh_labels"]["hierarchy_missing_branch_count"] == "缺失层级分支数"


def test_machine_token_preservation_rules():
    assert machine_token_should_be_preserved("prop-normalize-relation-keywords")
    assert machine_token_should_be_preserved("doc-b29c711f27db9ad51c2851d9db562957-chunk-006")
    assert machine_token_should_be_preserved("snapshots/quality_score.json")
    assert machine_token_should_be_preserved("deepseek-v4-pro")
    assert not machine_token_should_be_preserved("Improve hierarchy readability")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
.\scripts\test.sh tests\kg\test_kb_iteration_zh_artifacts.py
```

Expected: fails with `ModuleNotFoundError: No module named 'lightrag.kb_iteration.zh_artifacts'`.

- [ ] **Step 3: Create `zh_artifacts.py` with deterministic JSON support**

Create `lightrag/kb_iteration/zh_artifacts.py`:

```python
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class ZhArtifactClient(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass(frozen=True)
class ZhArtifactResult:
    artifact_key: str
    source_relative_path: str
    zh_relative_path: str
    content_type: str
    generated: bool
    fallback_to_source: bool
    generated_at: str
    model: str
    error: str = ""
    payload: Any | None = None
    content: str = ""


FIELD_LABELS: dict[str, str] = {
    "overall": "总分",
    "subscores": "分项得分",
    "metrics": "指标",
    "findings": "发现的问题",
    "critical_blockers": "关键阻塞问题",
    "hierarchy_missing_branch_count": "缺失层级分支数",
    "hierarchy_present_branch_count": "已存在层级分支数",
    "hierarchy_required_branch_count": "必需层级分支数",
    "entity_hygiene": "实体卫生",
    "relation_semantics": "关系语义",
    "hierarchy_completeness": "层级完整性",
    "evidence_grounding": "证据支撑",
    "web_readability": "Web 可读性",
    "iteration_readiness": "迭代就绪度",
    "message": "说明",
    "severity": "严重级别",
    "category": "类别",
    "evidence": "证据",
    "suggested_fix_type": "建议修复类型",
    "requires_approval": "需要审批",
    "nodes": "节点",
    "edges": "关系",
    "source_files": "来源文件",
    "workspace": "工作区",
    "generated_at": "生成时间",
}

MACHINE_TOKEN_PATTERN = re.compile(
    r"^(?:prop-|doc-|chunk-|[a-z]+_[a-z0-9_]*$|[A-Za-z0-9_.-]+\.(?:md|json|yaml|yml|pdf)|deepseek-|gpt-|qwen-|glm-)"
)


def artifact_zh_relative_path(relative_path: Path) -> Path:
    suffix = relative_path.suffix
    if not suffix:
        return relative_path.with_name(f"{relative_path.name}.zh")
    return relative_path.with_name(f"{relative_path.stem}.zh{suffix}")


def machine_token_should_be_preserved(value: str) -> bool:
    stripped = value.strip()
    return bool(MACHINE_TOKEN_PATTERN.match(stripped))


def build_zh_json_payload(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        labels: dict[str, str] = {}
        for key, item in value.items():
            result[key] = build_zh_json_payload(item)
            if key in FIELD_LABELS:
                labels[key] = FIELD_LABELS[key]
        if labels:
            result["_zh_labels"] = labels
        return result
    if isinstance(value, list):
        return [build_zh_json_payload(item) for item in value]
    return value


def dump_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```powershell
.\scripts\test.sh tests\kg\test_kb_iteration_zh_artifacts.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add lightrag/kb_iteration/zh_artifacts.py tests/kg/test_kb_iteration_zh_artifacts.py
git commit -m "feat: add kb iteration zh artifact helpers"
```

---

### Task 2: Backend Display Artifact Generation and Safe Fallback

**Files:**
- Modify: `lightrag/kb_iteration/zh_artifacts.py`
- Test: `tests/kg/test_kb_iteration_zh_artifacts.py`

- [ ] **Step 1: Add failing tests for Markdown translation, JSON file writing, and fallback**

Append to `tests/kg/test_kb_iteration_zh_artifacts.py`:

```python
from lightrag.kb_iteration.zh_artifacts import ensure_zh_artifact


class FakeZhClient:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.text


class FailingZhClient:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("translator unavailable")


def test_ensure_zh_artifact_writes_markdown_translation(tmp_path: Path):
    source = tmp_path / "quality_report.md"
    source.write_text("# Quality\n\n- Generated proposals: 2\n", encoding="utf-8")
    client = FakeZhClient("# 质量报告\n\n- 生成的提案：2\n")

    result = ensure_zh_artifact(
        artifact_key="quality_report",
        source_path=source,
        content_type="text/markdown",
        client=client,
        model="deepseek-v4-pro",
        force=True,
    )

    zh_path = tmp_path / "quality_report.zh.md"
    assert zh_path.read_text(encoding="utf-8").startswith("# 质量报告")
    assert result.generated is True
    assert result.fallback_to_source is False
    assert result.model == "deepseek-v4-pro"
    assert "source_id" in client.calls[0]["system_prompt"]


def test_ensure_zh_artifact_writes_json_with_labels_without_llm(tmp_path: Path):
    source = tmp_path / "snapshots" / "quality_score.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"overall": 97}, ensure_ascii=False), encoding="utf-8")

    result = ensure_zh_artifact(
        artifact_key="quality_score",
        source_path=source,
        content_type="application/json",
        client=FakeZhClient("{}"),
        model="deepseek-v4-pro",
        force=True,
    )

    zh_payload = json.loads((tmp_path / "snapshots" / "quality_score.zh.json").read_text(encoding="utf-8"))
    assert zh_payload["overall"] == 97
    assert zh_payload["_zh_labels"]["overall"] == "总分"
    assert result.generated is True
    assert result.payload["_zh_labels"]["overall"] == "总分"


def test_ensure_zh_artifact_falls_back_to_source_when_translation_fails(tmp_path: Path):
    source = tmp_path / "approval_queue.md"
    source.write_text("# Approval Queue\n", encoding="utf-8")

    result = ensure_zh_artifact(
        artifact_key="approval_queue",
        source_path=source,
        content_type="text/markdown",
        client=FailingZhClient(),
        model="deepseek-v4-pro",
        force=True,
    )

    assert result.generated is False
    assert result.fallback_to_source is True
    assert "translator unavailable" in result.error
    assert result.content == "# Approval Queue\n"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
.\scripts\test.sh tests\kg\test_kb_iteration_zh_artifacts.py
```

Expected: fails because `ensure_zh_artifact` does not exist.

- [ ] **Step 3: Implement `ensure_zh_artifact`**

Add to `lightrag/kb_iteration/zh_artifacts.py`:

```python
def ensure_zh_artifact(
    *,
    artifact_key: str,
    source_path: Path,
    content_type: str,
    client: ZhArtifactClient,
    model: str,
    force: bool = False,
) -> ZhArtifactResult:
    zh_path = source_path.parent / artifact_zh_relative_path(Path(source_path.name)).name
    if source_path.parent.name == "snapshots":
        zh_path = source_path.with_name(artifact_zh_relative_path(Path(source_path.name)).name)

    generated_at = now_iso()
    if zh_path.exists() and zh_path.is_file() and not force:
        return _read_zh_result(
            artifact_key=artifact_key,
            source_path=source_path,
            zh_path=zh_path,
            content_type=content_type,
            generated_at=generated_at,
            model=model,
        )

    try:
        if content_type == "application/json":
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            zh_payload = build_zh_json_payload(payload)
            dump_json_file(zh_path, zh_payload)
            return ZhArtifactResult(
                artifact_key=artifact_key,
                source_relative_path=source_path.name,
                zh_relative_path=zh_path.name,
                content_type=content_type,
                generated=True,
                fallback_to_source=False,
                generated_at=generated_at,
                model=model,
                payload=zh_payload,
            )

        source_text = source_path.read_text(encoding="utf-8")
        translated = client.complete(
            system_prompt=_translation_system_prompt(),
            user_prompt=_translation_user_prompt(artifact_key, source_path.name, source_text),
        )
        zh_path.parent.mkdir(parents=True, exist_ok=True)
        zh_path.write_text(_normalize_markdown_translation(translated), encoding="utf-8")
        return ZhArtifactResult(
            artifact_key=artifact_key,
            source_relative_path=source_path.name,
            zh_relative_path=zh_path.name,
            content_type=content_type,
            generated=True,
            fallback_to_source=False,
            generated_at=generated_at,
            model=model,
            content=zh_path.read_text(encoding="utf-8"),
        )
    except Exception as exc:
        return ZhArtifactResult(
            artifact_key=artifact_key,
            source_relative_path=source_path.name,
            zh_relative_path=zh_path.name,
            content_type=content_type,
            generated=False,
            fallback_to_source=True,
            generated_at=generated_at,
            model=model,
            error=str(exc),
            content=source_path.read_text(encoding="utf-8") if source_path.exists() else "",
        )


def _read_zh_result(
    *,
    artifact_key: str,
    source_path: Path,
    zh_path: Path,
    content_type: str,
    generated_at: str,
    model: str,
) -> ZhArtifactResult:
    if content_type == "application/json":
        return ZhArtifactResult(
            artifact_key=artifact_key,
            source_relative_path=source_path.name,
            zh_relative_path=zh_path.name,
            content_type=content_type,
            generated=False,
            fallback_to_source=False,
            generated_at=generated_at,
            model=model,
            payload=json.loads(zh_path.read_text(encoding="utf-8")),
        )
    return ZhArtifactResult(
        artifact_key=artifact_key,
        source_relative_path=source_path.name,
        zh_relative_path=zh_path.name,
        content_type=content_type,
        generated=False,
        fallback_to_source=False,
        generated_at=generated_at,
        model=model,
        content=zh_path.read_text(encoding="utf-8"),
    )


def _translation_system_prompt() -> str:
    return (
        "Translate this LightRAG KG iteration artifact into concise Chinese for WebUI display. "
        "Preserve proposal_id, source_id, doc_id, file_path, workspace, run_id, JSON/YAML keys, "
        "node IDs, relation IDs, evidence IDs, model names, paths, code blocks, and API field names. "
        "Do not add medical claims. Keep Markdown structure."
    )


def _translation_user_prompt(artifact_key: str, source_name: str, source_text: str) -> str:
    return (
        f"artifact_key: {artifact_key}\n"
        f"source_file: {source_name}\n\n"
        "Translate the natural-language text to Chinese while preserving machine identifiers.\n\n"
        f"{source_text}"
    )


def _normalize_markdown_translation(text: str) -> str:
    stripped = text.strip()
    return stripped + "\n" if stripped else ""
```

- [ ] **Step 4: Run tests and fix path handling if required**

Run:

```powershell
.\scripts\test.sh tests\kg\test_kb_iteration_zh_artifacts.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add lightrag/kb_iteration/zh_artifacts.py tests/kg/test_kb_iteration_zh_artifacts.py
git commit -m "feat: generate kb iteration zh display artifacts"
```

---

### Task 3: Backend API for Display Artifacts and Manifests

**Files:**
- Modify: `lightrag/api/routers/kb_iteration_routes.py`
- Modify: `lightrag/kb_iteration/zh_artifacts.py`
- Test: `tests/api/routes/test_kb_iteration_routes.py`

- [ ] **Step 1: Add failing route tests for display artifact generation**

Append to `tests/api/routes/test_kb_iteration_routes.py`:

```python
def test_display_artifact_generates_zh_markdown_and_reports_metadata(
    tmp_path: Path, monkeypatch
):
    client, fixture = _client(tmp_path, monkeypatch)

    import lightrag.api.routers.kb_iteration_routes as routes

    class FakeClient:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            assert "source_id" in system_prompt
            assert "quality_report.md" in user_prompt
            return "# 质量报告\n\n- 中文展示\n"

    monkeypatch.setattr(routes, "_default_llm_review_client", lambda _rag: FakeClient())
    monkeypatch.setenv("KB_ITERATION_LLM_MODEL", "deepseek-v4-pro")

    response = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/quality_report/display",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifactKey"] == "quality_report"
    assert payload["display"]["language"] == "zh"
    assert payload["display"]["sourceFile"] == "quality_report.md"
    assert payload["display"]["zhFile"] == "quality_report.zh.md"
    assert payload["display"]["model"] == "deepseek-v4-pro"
    assert payload["content"] == "# 质量报告\n\n- 中文展示\n"
    assert (fixture.package / "quality_report.zh.md").exists()


def test_display_artifact_json_keeps_payload_and_adds_zh_labels(tmp_path: Path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/quality_score/display",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["contentType"] == "application/json"
    assert payload["payload"]["overall"] == 82
    assert payload["payload"]["_zh_labels"]["overall"] == "总分"


def test_display_artifact_can_force_regeneration(tmp_path: Path, monkeypatch):
    client, fixture = _client(tmp_path, monkeypatch)
    (fixture.package / "quality_report.zh.md").write_text("# 旧翻译\n", encoding="utf-8")

    import lightrag.api.routers.kb_iteration_routes as routes

    class FakeClient:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return "# 新翻译\n"

    monkeypatch.setattr(routes, "_default_llm_review_client", lambda _rag: FakeClient())

    response = client.post(
        "/kb-iteration/influenza_medical_v1/artifacts/quality_report/display/regenerate",
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["content"] == "# 新翻译\n"
```

- [ ] **Step 2: Run route tests and verify failure**

Run:

```powershell
.\scripts\test.sh tests\api\routes\test_kb_iteration_routes.py -k display_artifact
```

Expected: fails with 404 for missing display endpoints.

- [ ] **Step 3: Add display endpoints**

In `lightrag/api/routers/kb_iteration_routes.py`, import:

```python
from lightrag.kb_iteration.zh_artifacts import ensure_zh_artifact
```

Add the display endpoints before the generic `/{workspace}/artifacts/{artifact_key:path}` route. The generic path route must remain after display routes so it does not capture `quality_report/display` as an artifact key.

```python
    @router.get(
        "/{workspace}/artifacts/{artifact_key:path}/display",
        dependencies=[Depends(combined_auth)],
    )
    async def get_display_artifact(workspace: str, artifact_key: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _read_display_artifact(args, rag, workspace, artifact_key, force=False)

    @router.post(
        "/{workspace}/artifacts/{artifact_key:path}/display/regenerate",
        dependencies=[Depends(combined_auth)],
    )
    async def regenerate_display_artifact(
        workspace: str, artifact_key: str
    ) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _read_display_artifact(args, rag, workspace, artifact_key, force=True)
```

Add helper below `_read_artifact`:

```python
def _read_display_artifact(
    args, rag, workspace: str, artifact_key: str, *, force: bool
) -> dict[str, Any]:
    source_path, content_type = _require_artifact_file(args, workspace, artifact_key)
    client = _default_llm_review_client(rag)
    model = os.getenv("KB_ITERATION_LLM_MODEL", "").strip() or "unknown"
    result = ensure_zh_artifact(
        artifact_key=artifact_key,
        source_path=source_path,
        content_type=content_type,
        client=client,
        model=model,
        force=force,
    )
    base = {
        "artifactKey": artifact_key,
        "contentType": content_type,
        "display": {
            "language": "zh",
            "sourceFile": source_path.relative_to(_workspace_dir(args, workspace)).as_posix(),
            "zhFile": (source_path.parent / result.zh_relative_path)
            .relative_to(_workspace_dir(args, workspace))
            .as_posix(),
            "generated": result.generated,
            "fallbackToSource": result.fallback_to_source,
            "generatedAt": result.generated_at,
            "model": result.model,
            "error": result.error,
        },
    }
    if content_type == "application/json":
        return {**base, "payload": result.payload}
    return {**base, "content": result.content}
```

- [ ] **Step 4: Extend artifact manifest with display status**

Change `_artifact_manifest` so each item includes `display`:

```python
def _artifact_manifest(args, workspace: str) -> list[dict[str, Any]]:
    manifest = []
    base_dir = _workspace_dir(args, workspace)
    for key, (_relative_path, content_type) in ARTIFACTS.items():
        path, _ = _safe_artifact_path(args, workspace, key)
        zh_path = path.parent / f"{path.stem}.zh{path.suffix}"
        manifest.append(
            {
                "key": key,
                "contentType": content_type,
                "exists": path.exists() and path.is_file(),
                "display": {
                    "language": "zh",
                    "sourceFile": path.relative_to(base_dir).as_posix(),
                    "zhFile": zh_path.relative_to(base_dir).as_posix(),
                    "exists": zh_path.exists() and zh_path.is_file(),
                },
            }
        )
    return manifest
```

- [ ] **Step 5: Run route tests**

Run:

```powershell
.\scripts\test.sh tests\api\routes\test_kb_iteration_routes.py -k "display_artifact or summary_reads"
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```powershell
git add lightrag/api/routers/kb_iteration_routes.py lightrag/kb_iteration/zh_artifacts.py tests/api/routes/test_kb_iteration_routes.py
git commit -m "feat: expose kb iteration zh display artifacts"
```

---

### Task 4: Proposal Revision Request API

**Files:**
- Modify: `lightrag/api/routers/kb_iteration_routes.py`
- Modify: `lightrag/kb_iteration/markdown.py`
- Modify: `lightrag/kb_iteration/agent_context.py`
- Test: `tests/api/routes/test_kb_iteration_routes.py`
- Test: `tests/kg/test_kb_iteration_review_context.py`

- [ ] **Step 1: Add failing API test for revise request**

Append to `tests/api/routes/test_kb_iteration_routes.py`:

```python
def test_revise_rejected_proposal_records_revision_request(tmp_path: Path, monkeypatch):
    client, fixture = _client(tmp_path, monkeypatch)

    reject = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/reject",
        headers=HEADERS,
        json={},
    )
    assert reject.status_code == 200

    response = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/revise",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace"] == "influenza_medical_v1"
    assert payload["proposalId"] == "p1"
    assert payload["artifactKey"] == "proposal_revision_requests"
    revision_text = (fixture.package / "proposal_revision_requests.md").read_text(
        encoding="utf-8"
    )
    assert "parent_proposal_id: p1" in revision_text
    assert "status: requested" in revision_text
```

- [ ] **Step 2: Add failing context test**

Append to `tests/kg/test_kb_iteration_review_context.py`:

```python
def test_review_context_includes_proposal_revision_requests(tmp_path: Path):
    package = _make_package(tmp_path)
    (package / "proposal_revision_requests.md").write_text(
        "# Proposal Revision Requests\n\n- parent_proposal_id: p1\n  status: requested\n",
        encoding="utf-8",
    )

    context = build_review_context(package)

    assert "proposal_revision_requests" in context["rules_memory"]
    assert "parent_proposal_id: p1" in context["rules_memory"]["proposal_revision_requests"]
```

- [ ] **Step 3: Run focused tests and verify failure**

Run:

```powershell
.\scripts\test.sh tests\api\routes\test_kb_iteration_routes.py -k revise_rejected_proposal
.\scripts\test.sh tests\kg\test_kb_iteration_review_context.py -k proposal_revision_requests
```

Expected: API route missing and context missing revision requests.

- [ ] **Step 4: Add default file template**

In `lightrag/kb_iteration/markdown.py`, extend the default artifact dictionary:

```python
"proposal_revision_requests.md": "# Proposal Revision Requests\n\n- Record rejected proposal revision requests here.\n",
```

- [ ] **Step 5: Add context loading**

In `lightrag/kb_iteration/agent_context.py`, add `proposal_revision_requests.md` to the read set and expose it under `rules_memory["proposal_revision_requests"]`.

Implementation shape:

```python
revision_requests = _read_text(package_path / "proposal_revision_requests.md")
context["rules_memory"]["proposal_revision_requests"] = revision_requests
```

- [ ] **Step 6: Add revise endpoint**

In `lightrag/api/routers/kb_iteration_routes.py`, add to `ARTIFACTS`:

```python
"proposal_revision_requests": ("proposal_revision_requests.md", "text/markdown"),
```

Add endpoint near proposal decision endpoints:

```python
    @router.post(
        "/{workspace}/proposals/{proposal_id}/revise",
        dependencies=[Depends(combined_auth)],
    )
    async def request_proposal_revision(workspace: str, proposal_id: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        proposal_id = validate_proposal_id(proposal_id)
        _proposal_by_id(args, workspace, proposal_id)
        package_dir = _workspace_dir(args, workspace)
        path = package_dir / "proposal_revision_requests.md"
        if not path.exists():
            path.write_text(
                "# Proposal Revision Requests\n\n- Record rejected proposal revision requests here.\n",
                encoding="utf-8",
            )
        entry = [
            "",
            f"## {proposal_id}",
            "",
            f"- parent_proposal_id: {proposal_id}",
            "- status: requested",
            f"- requested_at: {datetime.now(timezone.utc).isoformat()}",
            "- instruction: Let the Agent revise this rejected proposal using rejected_changes.md and current quality artifacts.",
            "",
        ]
        with path.open("a", encoding="utf-8") as file:
            file.write("\n".join(entry))
        return {
            "workspace": workspace,
            "proposalId": proposal_id,
            "artifactKey": "proposal_revision_requests",
            "status": "revision_requested",
        }
```

- [ ] **Step 7: Run tests**

Run:

```powershell
.\scripts\test.sh tests\api\routes\test_kb_iteration_routes.py -k "revise_rejected_proposal or artifact_key_is_whitelisted"
.\scripts\test.sh tests\kg\test_kb_iteration_review_context.py -k proposal_revision_requests
```

Expected: tests pass.

- [ ] **Step 8: Commit**

```powershell
git add lightrag/api/routers/kb_iteration_routes.py lightrag/kb_iteration/markdown.py lightrag/kb_iteration/agent_context.py tests/api/routes/test_kb_iteration_routes.py tests/kg/test_kb_iteration_review_context.py
git commit -m "feat: record rejected proposal revision requests"
```

---

### Task 5: Frontend API Types, Artifact Catalog, and Loaders

**Files:**
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Modify: `lightrag_webui/src/components/kg-maintenance/kgIterationLoadUtils.ts`
- Create: `lightrag_webui/src/components/kg-maintenance/kgMaintenanceArtifacts.ts`
- Test: `lightrag_webui/src/components/kg-maintenance/kgMaintenanceArtifacts.test.ts`
- Test: `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx`

- [ ] **Step 1: Write failing artifact catalog test**

Create `lightrag_webui/src/components/kg-maintenance/kgMaintenanceArtifacts.test.ts`:

```ts
import { describe, expect, test } from 'bun:test'
import {
  KG_MAINTENANCE_ARTIFACTS,
  WORKFLOW_STEPS,
  findArtifactDefinition
} from './kgMaintenanceArtifacts'

describe('kg maintenance artifact catalog', () => {
  test('uses exactly five workflow steps', () => {
    expect(WORKFLOW_STEPS.map((step) => step.id)).toEqual([
      'check',
      'llm-review',
      'approval',
      'execute',
      'validate'
    ])
  })

  test('keeps original and zh file names for core artifacts', () => {
    const quality = findArtifactDefinition('quality_score')
    expect(quality?.title).toBe('质量分数')
    expect(quality?.sourceFile).toBe('snapshots/quality_score.json')
    expect(quality?.zhFile).toBe('snapshots/quality_score.zh.json')

    const approval = findArtifactDefinition('approval_queue')
    expect(approval?.title).toBe('待审批提案')
    expect(approval?.sourceFile).toBe('approval_queue.md')
    expect(approval?.zhFile).toBe('approval_queue.zh.md')
  })

  test('groups every visible artifact into one workflow step', () => {
    const keys = KG_MAINTENANCE_ARTIFACTS.map((item) => item.key)
    expect(keys).toContain('llm_issue_analysis')
    expect(keys).toContain('accepted_changes_apply_result')
    expect(keys).toContain('proposal_revision_requests')
  })
})
```

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance/kgMaintenanceArtifacts.test.ts
```

Expected: fails because the module does not exist.

- [ ] **Step 3: Add API types and endpoints**

In `lightrag_webui/src/api/lightrag.ts`, extend artifact manifest item:

```ts
export type KBIterationDisplayMetadata = {
  language: 'zh'
  sourceFile: string
  zhFile: string
  exists?: boolean
  generated?: boolean
  fallbackToSource?: boolean
  generatedAt?: string
  model?: string
  error?: string
}

export type KBIterationArtifactManifestItem = {
  key: string
  contentType: 'application/json' | 'text/markdown' | string
  exists: boolean
  display?: KBIterationDisplayMetadata
}
```

Add display response type and API calls:

```ts
export type KBIterationDisplayArtifactResponse =
  | {
      artifactKey: string
      contentType: 'application/json' | string
      display: KBIterationDisplayMetadata
      payload: any
    }
  | {
      artifactKey: string
      contentType: 'text/markdown' | string
      display: KBIterationDisplayMetadata
      content: string
    }

export const getKBIterationDisplayArtifact = async (
  workspace: string,
  artifactKey: string
): Promise<KBIterationDisplayArtifactResponse> => {
  return kbIterationGet(
    `/kb-iteration/${encodePathSegment(workspace)}/artifacts/${encodePathSegment(artifactKey)}/display`
  )
}

export const regenerateKBIterationDisplayArtifact = async (
  workspace: string,
  artifactKey: string
): Promise<KBIterationDisplayArtifactResponse> => {
  return kbIterationPost(
    `/kb-iteration/${encodePathSegment(workspace)}/artifacts/${encodePathSegment(artifactKey)}/display/regenerate`,
    {}
  )
}

export const requestKBIterationProposalRevision = async (
  workspace: string,
  proposalId: string
): Promise<{ workspace: string; proposalId: string; artifactKey: string; status: string }> => {
  return kbIterationPost(
    `/kb-iteration/${encodePathSegment(workspace)}/proposals/${encodePathSegment(proposalId)}/revise`,
    {}
  )
}
```

- [ ] **Step 4: Create artifact catalog**

Create `lightrag_webui/src/components/kg-maintenance/kgMaintenanceArtifacts.ts`:

```ts
import type { KGMaintenanceSection } from '@/stores/kgMaintenance'

export type KGMaintenanceWorkflowStep = {
  id: KGMaintenanceSection
  title: string
  description: string
}

export type KGMaintenanceArtifactDefinition = {
  key: string
  title: string
  sourceFile: string
  zhFile: string
  step: KGMaintenanceSection
  contentType: 'markdown' | 'json' | 'yaml'
}

export const WORKFLOW_STEPS: KGMaintenanceWorkflowStep[] = [
  { id: 'check', title: '检查知识库', description: '生成快照、质量分和结构检查结果。' },
  { id: 'llm-review', title: 'LLM 审阅', description: '解释问题、定位证据、生成提案并排序。' },
  { id: 'approval', title: 'Proposal 审批', description: '接受或拒绝 Agent 生成的候选变更。' },
  { id: 'execute', title: '执行变更', description: '用确定性 Apply Engine 执行已接受变更。' },
  { id: 'validate', title: '验证结果', description: '对比执行前后质量变化，决定是否进入下一轮。' }
]

export const KG_MAINTENANCE_ARTIFACTS: KGMaintenanceArtifactDefinition[] = [
  { key: 'kb_context', title: '当前 KB 摘要', sourceFile: 'kb_context.md', zhFile: 'kb_context.zh.md', step: 'check', contentType: 'markdown' },
  { key: 'quality_report', title: '质量报告', sourceFile: 'quality_report.md', zhFile: 'quality_report.zh.md', step: 'check', contentType: 'markdown' },
  { key: 'kg_snapshot', title: '图谱快照', sourceFile: 'snapshots/kg_snapshot.json', zhFile: 'snapshots/kg_snapshot.zh.json', step: 'check', contentType: 'json' },
  { key: 'quality_score', title: '质量分数', sourceFile: 'snapshots/quality_score.json', zhFile: 'snapshots/quality_score.zh.json', step: 'check', contentType: 'json' },
  { key: 'entity_catalog', title: '实体目录', sourceFile: 'entity_catalog.md', zhFile: 'entity_catalog.zh.md', step: 'check', contentType: 'markdown' },
  { key: 'relation_catalog', title: '关系目录', sourceFile: 'relation_catalog.md', zhFile: 'relation_catalog.zh.md', step: 'check', contentType: 'markdown' },
  { key: 'llm_review_trace', title: 'LLM 审阅轨迹', sourceFile: 'llm_review_trace.json', zhFile: 'llm_review_trace.zh.json', step: 'llm-review', contentType: 'json' },
  { key: 'llm_issue_analysis', title: '问题解释', sourceFile: 'llm_issue_analysis.md', zhFile: 'llm_issue_analysis.zh.md', step: 'llm-review', contentType: 'markdown' },
  { key: 'llm_missing_branch_inference', title: '缺失分支推断', sourceFile: 'llm_missing_branch_inference.md', zhFile: 'llm_missing_branch_inference.zh.md', step: 'llm-review', contentType: 'markdown' },
  { key: 'llm_evidence_map', title: '证据定位', sourceFile: 'llm_evidence_map.md', zhFile: 'llm_evidence_map.zh.md', step: 'llm-review', contentType: 'markdown' },
  { key: 'llm_repair_plan', title: '修复方案排序', sourceFile: 'llm_repair_plan.md', zhFile: 'llm_repair_plan.zh.md', step: 'llm-review', contentType: 'markdown' },
  { key: 'llm_judge_report', title: 'Judge 评判', sourceFile: 'llm_judge_report.md', zhFile: 'llm_judge_report.zh.md', step: 'llm-review', contentType: 'markdown' },
  { key: 'proposals_generated', title: '生成的候选提案', sourceFile: 'proposals.generated.yaml', zhFile: 'proposals.generated.zh.yaml', step: 'llm-review', contentType: 'yaml' },
  { key: 'approval_queue', title: '待审批提案', sourceFile: 'approval_queue.md', zhFile: 'approval_queue.zh.md', step: 'approval', contentType: 'markdown' },
  { key: 'accepted_changes', title: '已接受变更', sourceFile: 'accepted_changes.md', zhFile: 'accepted_changes.zh.md', step: 'approval', contentType: 'markdown' },
  { key: 'rejected_changes', title: '已拒绝变更', sourceFile: 'rejected_changes.md', zhFile: 'rejected_changes.zh.md', step: 'approval', contentType: 'markdown' },
  { key: 'proposal_revision_requests', title: '提案修改请求', sourceFile: 'proposal_revision_requests.md', zhFile: 'proposal_revision_requests.zh.md', step: 'approval', contentType: 'markdown' },
  { key: 'accepted_changes_apply_result', title: '真实应用结果', sourceFile: 'accepted_changes_apply_result.md', zhFile: 'accepted_changes_apply_result.zh.md', step: 'execute', contentType: 'markdown' },
  { key: 'accepted_changes_execution', title: '执行报告', sourceFile: 'accepted_changes_execution.md', zhFile: 'accepted_changes_execution.zh.md', step: 'execute', contentType: 'markdown' },
  { key: 'iteration_log', title: '迭代日志', sourceFile: 'iteration_log.md', zhFile: 'iteration_log.zh.md', step: 'validate', contentType: 'markdown' },
  { key: 'improvement_backlog', title: '改进 Backlog', sourceFile: 'improvement_backlog.md', zhFile: 'improvement_backlog.zh.md', step: 'validate', contentType: 'markdown' }
]

export function findArtifactDefinition(key: string) {
  return KG_MAINTENANCE_ARTIFACTS.find((artifact) => artifact.key === key) || null
}

export function artifactsForStep(step: KGMaintenanceSection) {
  return KG_MAINTENANCE_ARTIFACTS.filter((artifact) => artifact.step === step)
}
```

- [ ] **Step 5: Run tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance/kgMaintenanceArtifacts.test.ts
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```powershell
git add lightrag_webui/src/api/lightrag.ts lightrag_webui/src/components/kg-maintenance/kgMaintenanceArtifacts.ts lightrag_webui/src/components/kg-maintenance/kgMaintenanceArtifacts.test.ts
git commit -m "feat: add kg maintenance display artifact catalog"
```

---

### Task 6: Frontend Workflow Store, Shell, and Top-Level Layout

**Files:**
- Modify: `lightrag_webui/src/stores/kgMaintenance.ts`
- Modify: `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.tsx`
- Modify: `lightrag_webui/src/features/KGMaintenanceConsole.tsx`
- Test: `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx`
- Test: `lightrag_webui/src/stores/kgMaintenance.test.ts`

- [ ] **Step 1: Update failing shell tests**

In `KGMaintenanceShell.test.tsx`, replace old section expectations with:

```ts
test('shell renders five workflow steps and no permanent inspector', () => {
  const markup = renderToStaticMarkup(
    <KGMaintenanceShell
      activeSection="check"
      onSectionChange={() => undefined}
      workspaces={['influenza_medical_v1']}
      selectedWorkspace="influenza_medical_v1"
      onWorkspaceChange={() => undefined}
      onRefresh={() => undefined}
      loading={false}
      running={false}
      error={null}
      onOpenArtifacts={() => undefined}
    >
      <div>Console body</div>
    </KGMaintenanceShell>
  )

  expect(markup).toContain('检查知识库')
  expect(markup).toContain('LLM 审阅')
  expect(markup).toContain('Proposal 审批')
  expect(markup).toContain('执行变更')
  expect(markup).toContain('验证结果')
  expect(markup).toContain('全部产物')
  expect(markup).not.toContain('当前 KB 摘要')
  expect(markup).not.toContain('辅助材料')
  expect(markup).not.toContain('Inspector')
})
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance/KGMaintenanceShell.test.tsx src/stores/kgMaintenance.test.ts
```

Expected: fails because old sections still exist and props mismatch.

- [ ] **Step 3: Replace section type**

In `lightrag_webui/src/stores/kgMaintenance.ts`:

```ts
export type KGMaintenanceSection = 'check' | 'llm-review' | 'approval' | 'execute' | 'validate'
```

Set default:

```ts
activeSection: 'check',
```

- [ ] **Step 4: Refactor shell props and layout**

In `KGMaintenanceShell.tsx`, remove `inspector` and `onRunReview`. Use workflow steps from `kgMaintenanceArtifacts.ts`.

Implementation shape:

```tsx
import { WORKFLOW_STEPS } from './kgMaintenanceArtifacts'
import { BoxIcon, ClipboardCheckIcon, FileSearchIcon, ListChecksIcon, RefreshCwIcon, ShieldCheckIcon } from 'lucide-react'

const stepIcons = {
  check: FileSearchIcon,
  'llm-review': ShieldCheckIcon,
  approval: ClipboardCheckIcon,
  execute: ListChecksIcon,
  validate: BoxIcon
} as const
```

Top bar controls:

```tsx
<Button variant="outline" size="sm" onClick={onRefresh} disabled={loading}>
  <RefreshCwIcon className={cn('size-4', loading && 'animate-spin')} />
  刷新
</Button>
<Button variant="outline" size="sm" onClick={onOpenArtifacts}>
  <FileSearchIcon className="size-4" />
  全部产物
</Button>
```

Grid layout:

```tsx
<div className="grid min-h-0 flex-1 grid-cols-[220px_minmax(0,1fr)] overflow-hidden">
  <aside className="border-border/70 bg-muted/20 min-h-0 overflow-auto border-r p-3">
    ...
  </aside>
  <main className="min-h-0 overflow-auto p-4">{children}</main>
</div>
```

- [ ] **Step 5: Update console shell usage**

In `KGMaintenanceConsole.tsx`, remove `inspector` prop and add `artifactDrawerOpen` state:

```ts
const [artifactDrawerOpen, setArtifactDrawerOpen] = useState(false)
```

Pass:

```tsx
onOpenArtifacts={() => setArtifactDrawerOpen(true)}
```

Render drawer in a later task with a temporary placeholder:

```tsx
{artifactDrawerOpen && (
  <div role="dialog" aria-label="全部产物">
    <button type="button" onClick={() => setArtifactDrawerOpen(false)}>关闭</button>
  </div>
)}
```

- [ ] **Step 6: Run shell tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance/KGMaintenanceShell.test.tsx src/stores/kgMaintenance.test.ts
```

Expected: tests pass.

- [ ] **Step 7: Commit**

```powershell
git add lightrag_webui/src/stores/kgMaintenance.ts lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.tsx lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx lightrag_webui/src/features/KGMaintenanceConsole.tsx lightrag_webui/src/stores/kgMaintenance.test.ts
git commit -m "feat: simplify kg maintenance workflow shell"
```

---

### Task 7: Next Action Resolver and Step Header

**Files:**
- Create: `lightrag_webui/src/components/kg-maintenance/kgMaintenanceNextAction.ts`
- Create: `lightrag_webui/src/components/kg-maintenance/kgMaintenanceNextAction.test.ts`
- Create: `lightrag_webui/src/components/kg-maintenance/AgentStepHeader.tsx`
- Test: `lightrag_webui/src/components/kg-maintenance/AgentStepHeader.test.tsx`

- [ ] **Step 1: Write failing next-action tests**

Create `kgMaintenanceNextAction.test.ts`:

```ts
import { describe, expect, test } from 'bun:test'
import { resolveKGMaintenanceNextAction } from './kgMaintenanceNextAction'

describe('resolveKGMaintenanceNextAction', () => {
  test('recommends checking when there is no summary', () => {
    expect(resolveKGMaintenanceNextAction({ summary: null }).id).toBe('run-check')
  })

  test('recommends llm review when quality has findings and no llm trace', () => {
    const action = resolveKGMaintenanceNextAction({
      summary: { quality: { findings: [{ severity: 'high' }] }, pendingApprovalCount: 0 },
      llmTrace: null
    })
    expect(action.id).toBe('run-llm-review')
  })

  test('recommends approval when pending approvals exist', () => {
    const action = resolveKGMaintenanceNextAction({
      summary: { quality: { findings: [] }, pendingApprovalCount: 2 },
      llmTrace: { stop_reason: 'pending_human_review' }
    })
    expect(action.id).toBe('open-approval')
  })

  test('recommends execute when accepted changes are not reflected in apply result', () => {
    const action = resolveKGMaintenanceNextAction({
      summary: { quality: { findings: [] }, pendingApprovalCount: 0 },
      acceptedChanges: '## prop-a',
      acceptedApplyResult: ''
    })
    expect(action.id).toBe('execute-accepted')
  })

  test('recommends next iteration when quality is clean and nothing is pending', () => {
    const action = resolveKGMaintenanceNextAction({
      summary: { quality: { findings: [], metrics: { hierarchy_missing_branch_count: 0 } }, pendingApprovalCount: 0 },
      acceptedChanges: '',
      acceptedApplyResult: '# Apply Result\n- Applied: 0\n'
    })
    expect(action.id).toBe('start-next-iteration')
  })
})
```

- [ ] **Step 2: Implement resolver**

Create `kgMaintenanceNextAction.ts`:

```ts
import type { KGMaintenanceSection } from '@/stores/kgMaintenance'

export type KGMaintenanceNextActionId =
  | 'run-check'
  | 'run-llm-review'
  | 'open-approval'
  | 'execute-accepted'
  | 'validate-result'
  | 'start-next-iteration'

export type KGMaintenanceNextAction = {
  id: KGMaintenanceNextActionId
  label: string
  section: KGMaintenanceSection
  reason: string
}

type ResolveArgs = {
  summary: any
  llmTrace?: Record<string, any> | null
  acceptedChanges?: string
  acceptedApplyResult?: string
  justExecuted?: boolean
}

export function resolveKGMaintenanceNextAction(args: ResolveArgs): KGMaintenanceNextAction {
  const summary = args.summary
  if (!summary) {
    return { id: 'run-check', label: '检查知识库', section: 'check', reason: '当前还没有最新检查结果。' }
  }
  const findings = Array.isArray(summary.quality?.findings) ? summary.quality.findings : []
  if (findings.length > 0 && !args.llmTrace) {
    return { id: 'run-llm-review', label: '运行 LLM 审阅', section: 'llm-review', reason: '质量检查发现问题，需要 Agent 解释并生成提案。' }
  }
  if ((summary.pendingApprovalCount ?? 0) > 0) {
    return { id: 'open-approval', label: '审批 Proposal', section: 'approval', reason: '存在待审批提案。' }
  }
  if (hasAcceptedChanges(args.acceptedChanges) && !hasApplyResultForAcceptedChanges(args.acceptedApplyResult)) {
    return { id: 'execute-accepted', label: '执行已接受变更', section: 'execute', reason: '存在已接受但尚未执行的变更。' }
  }
  if (args.justExecuted) {
    return { id: 'validate-result', label: '验证结果', section: 'validate', reason: '刚执行过变更，需要确认质量变化。' }
  }
  return { id: 'start-next-iteration', label: '开始下一轮迭代', section: 'llm-review', reason: '当前没有待处理提案，可以进入下一轮审阅。' }
}

function hasAcceptedChanges(text?: string) {
  return /^##\s+/m.test(text || '')
}

function hasApplyResultForAcceptedChanges(text?: string) {
  return Boolean((text || '').trim())
}
```

- [ ] **Step 3: Add header component test**

Create `AgentStepHeader.test.tsx`:

```tsx
import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { AgentStepHeader } from './AgentStepHeader'

describe('AgentStepHeader', () => {
  test('renders current status and one action', () => {
    const markup = renderToStaticMarkup(
      <AgentStepHeader
        title="Proposal 审批"
        description="接受或拒绝 Agent 生成的候选变更。"
        action={{ id: 'open-approval', label: '审批 Proposal', section: 'approval', reason: '存在待审批提案。' }}
        badges={[{ label: '待审', value: '1' }]}
        onAction={() => undefined}
      />
    )

    expect(markup).toContain('Proposal 审批')
    expect(markup).toContain('存在待审批提案')
    expect(markup).toContain('审批 Proposal')
    expect(markup).toContain('待审')
  })
})
```

- [ ] **Step 4: Implement header**

Create `AgentStepHeader.tsx`:

```tsx
import Button from '@/components/ui/Button'
import type { KGMaintenanceNextAction } from './kgMaintenanceNextAction'
import { ArrowRightIcon } from 'lucide-react'

export function AgentStepHeader({
  title,
  description,
  action,
  badges,
  onAction
}: {
  title: string
  description: string
  action: KGMaintenanceNextAction
  badges: Array<{ label: string; value: string }>
  onAction: (action: KGMaintenanceNextAction) => void
}) {
  return (
    <section className="border-border/70 bg-background rounded-lg border p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold">{title}</h2>
          <p className="text-muted-foreground mt-1 text-sm">{description}</p>
          <p className="mt-3 text-sm">{action.reason}</p>
        </div>
        <Button type="button" onClick={() => onAction(action)}>
          {action.label}
          <ArrowRightIcon className="size-4" />
        </Button>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {badges.map((badge) => (
          <span key={badge.label} className="bg-muted rounded-md px-2 py-1 text-xs">
            {badge.label}: <span className="font-medium">{badge.value}</span>
          </span>
        ))}
      </div>
    </section>
  )
}
```

- [ ] **Step 5: Run tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance/kgMaintenanceNextAction.test.ts src/components/kg-maintenance/AgentStepHeader.test.tsx
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```powershell
git add lightrag_webui/src/components/kg-maintenance/kgMaintenanceNextAction.ts lightrag_webui/src/components/kg-maintenance/kgMaintenanceNextAction.test.ts lightrag_webui/src/components/kg-maintenance/AgentStepHeader.tsx lightrag_webui/src/components/kg-maintenance/AgentStepHeader.test.tsx
git commit -m "feat: add kg maintenance next action header"
```

---

### Task 8: Artifact File Section and All Artifacts Drawer

**Files:**
- Create: `lightrag_webui/src/components/kg-maintenance/ArtifactFileSection.tsx`
- Create: `lightrag_webui/src/components/kg-maintenance/ArtifactFileSection.test.tsx`
- Create: `lightrag_webui/src/components/kg-maintenance/ArtifactDrawer.tsx`
- Create: `lightrag_webui/src/components/kg-maintenance/ArtifactDrawer.test.tsx`
- Modify: `lightrag_webui/src/features/KGMaintenanceConsole.tsx`

- [ ] **Step 1: Write failing tests**

Create `ArtifactFileSection.test.tsx`:

```tsx
import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { ArtifactFileSection } from './ArtifactFileSection'

describe('ArtifactFileSection', () => {
  test('renders Chinese and original file names with regeneration action', () => {
    const markup = renderToStaticMarkup(
      <ArtifactFileSection
        title="关联文件"
        artifacts={[
          {
            key: 'quality_report',
            title: '质量报告',
            sourceFile: 'quality_report.md',
            zhFile: 'quality_report.zh.md',
            contentType: 'markdown',
            displayStatus: '已生成',
            generatedAt: '2026-06-20T00:00:00Z',
            model: 'deepseek-v4-pro',
            content: '# 质量报告',
            originalContent: '# Quality'
          }
        ]}
        onRegenerate={() => undefined}
      />
    )

    expect(markup).toContain('质量报告')
    expect(markup).toContain('quality_report.md')
    expect(markup).toContain('quality_report.zh.md')
    expect(markup).toContain('重新生成中文展示')
  })
})
```

Create `ArtifactDrawer.test.tsx`:

```tsx
import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { ArtifactDrawer } from './ArtifactDrawer'

describe('ArtifactDrawer', () => {
  test('groups artifacts by workflow step', () => {
    const markup = renderToStaticMarkup(
      <ArtifactDrawer
        open
        artifacts={[
          { key: 'quality_score', title: '质量分数', sourceFile: 'snapshots/quality_score.json', zhFile: 'snapshots/quality_score.zh.json', step: 'check', status: '已生成' },
          { key: 'approval_queue', title: '待审批提案', sourceFile: 'approval_queue.md', zhFile: 'approval_queue.zh.md', step: 'approval', status: '缺失' }
        ]}
        onClose={() => undefined}
        onOpenArtifact={() => undefined}
      />
    )

    expect(markup).toContain('全部产物')
    expect(markup).toContain('检查知识库')
    expect(markup).toContain('Proposal 审批')
    expect(markup).toContain('quality_score.zh.json')
  })
})
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance/ArtifactFileSection.test.tsx src/components/kg-maintenance/ArtifactDrawer.test.tsx
```

Expected: modules missing.

- [ ] **Step 3: Implement `ArtifactFileSection`**

Create component with a native `<details>` section and per-file Chinese/original toggle:

```tsx
import Button from '@/components/ui/Button'
import { RefreshCwIcon } from 'lucide-react'
import { useState } from 'react'

type DisplayArtifactItem = {
  key: string
  title: string
  sourceFile: string
  zhFile: string
  contentType: string
  displayStatus: string
  generatedAt?: string
  model?: string
  content: string
  originalContent: string
}

export function ArtifactFileSection({
  title,
  artifacts,
  onRegenerate
}: {
  title: string
  artifacts: DisplayArtifactItem[]
  onRegenerate: (artifactKey: string) => void
}) {
  const [viewMode, setViewMode] = useState<Record<string, 'zh' | 'original'>>({})
  return (
    <details className="border-border/70 rounded-lg border">
      <summary className="cursor-pointer px-3 py-2 text-sm font-medium">{title}</summary>
      <div className="space-y-3 border-t p-3">
        {artifacts.map((artifact) => {
          const mode = viewMode[artifact.key] || 'zh'
          return (
            <article key={artifact.key} className="border-border/70 rounded-md border p-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="text-sm font-medium">{artifact.title}</h3>
                  <p className="text-muted-foreground mt-1 text-xs">
                    原始文件：{artifact.sourceFile} / 中文展示：{artifact.zhFile}
                  </p>
                  <p className="text-muted-foreground mt-1 text-xs">
                    {artifact.displayStatus}
                    {artifact.generatedAt ? ` / ${artifact.generatedAt}` : ''}
                    {artifact.model ? ` / ${artifact.model}` : ''}
                  </p>
                </div>
                <Button variant="outline" size="sm" onClick={() => onRegenerate(artifact.key)}>
                  <RefreshCwIcon className="size-4" />
                  重新生成中文展示
                </Button>
              </div>
              <div className="mt-3 flex gap-2">
                <button type="button" className="text-xs underline" onClick={() => setViewMode({ ...viewMode, [artifact.key]: 'zh' })}>中文展示</button>
                <button type="button" className="text-xs underline" onClick={() => setViewMode({ ...viewMode, [artifact.key]: 'original' })}>原始文件</button>
              </div>
              <pre className="bg-muted mt-2 max-h-80 overflow-auto rounded-md p-3 text-xs whitespace-pre-wrap">
                {mode === 'zh' ? artifact.content : artifact.originalContent}
              </pre>
            </article>
          )
        })}
      </div>
    </details>
  )
}
```

- [ ] **Step 4: Implement `ArtifactDrawer`**

Create drawer with fixed positioning and grouped entries:

```tsx
import Button from '@/components/ui/Button'
import { WORKFLOW_STEPS } from './kgMaintenanceArtifacts'

type DrawerArtifact = {
  key: string
  title: string
  sourceFile: string
  zhFile: string
  step: string
  status: string
}

export function ArtifactDrawer({
  open,
  artifacts,
  onClose,
  onOpenArtifact
}: {
  open: boolean
  artifacts: DrawerArtifact[]
  onClose: () => void
  onOpenArtifact: (artifactKey: string) => void
}) {
  if (!open) return null
  return (
    <div role="dialog" aria-label="全部产物" className="fixed inset-0 z-40 bg-black/20">
      <aside className="bg-background ml-auto h-full w-[520px] overflow-auto border-l p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold">全部产物</h2>
          <Button variant="outline" size="sm" onClick={onClose}>关闭</Button>
        </div>
        <div className="mt-4 space-y-5">
          {WORKFLOW_STEPS.map((step) => (
            <section key={step.id}>
              <h3 className="text-sm font-medium">{step.title}</h3>
              <div className="mt-2 space-y-2">
                {artifacts.filter((artifact) => artifact.step === step.id).map((artifact) => (
                  <button key={artifact.key} type="button" className="border-border/70 w-full rounded-md border p-2 text-left text-sm" onClick={() => onOpenArtifact(artifact.key)}>
                    <span className="font-medium">{artifact.title}</span>
                    <span className="text-muted-foreground block text-xs">{artifact.sourceFile} / {artifact.zhFile}</span>
                    <span className="text-muted-foreground block text-xs">{artifact.status}</span>
                  </button>
                ))}
              </div>
            </section>
          ))}
        </div>
      </aside>
    </div>
  )
}
```

- [ ] **Step 5: Run tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance/ArtifactFileSection.test.tsx src/components/kg-maintenance/ArtifactDrawer.test.tsx
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```powershell
git add lightrag_webui/src/components/kg-maintenance/ArtifactFileSection.tsx lightrag_webui/src/components/kg-maintenance/ArtifactFileSection.test.tsx lightrag_webui/src/components/kg-maintenance/ArtifactDrawer.tsx lightrag_webui/src/components/kg-maintenance/ArtifactDrawer.test.tsx lightrag_webui/src/features/KGMaintenanceConsole.tsx
git commit -m "feat: add kg maintenance artifact display surfaces"
```

---

### Task 9: Snapshot Tables and Evidence Issue View

**Files:**
- Create: `lightrag_webui/src/components/kg-maintenance/SnapshotTables.tsx`
- Create: `lightrag_webui/src/components/kg-maintenance/SnapshotTables.test.tsx`
- Create or modify: `lightrag_webui/src/components/kg-maintenance/kgMaintenanceDisplay.ts`
- Test: `lightrag_webui/src/components/kg-maintenance/kgMaintenanceDisplay.test.ts`
- Modify: `lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.tsx`

- [ ] **Step 1: Write pure helper tests for evidence issue extraction**

Create `kgMaintenanceDisplay.test.ts`:

```ts
import { describe, expect, test } from 'bun:test'
import { buildEvidenceIssueRows } from './kgMaintenanceDisplay'

describe('kg maintenance display helpers', () => {
  test('builds evidence issue rows from nodes and edges', () => {
    const rows = buildEvidenceIssueRows({
      nodes: [{ id: 'n1', label: '节点一', source_id: '', file_path: 'doc.pdf' }],
      edges: [{ id: 'e1', source: 'n1', target: 'n2', source_id: 'chunk-1', file_path: '' }]
    })

    expect(rows).toEqual([
      { id: 'node:n1:source_id', itemType: '节点', itemId: 'n1', issue: '缺少证据来源 ID' },
      { id: 'edge:e1:file_path', itemType: '关系', itemId: 'e1', issue: '缺少来源文件' }
    ])
  })
})
```

- [ ] **Step 2: Write SnapshotTables render test**

Create `SnapshotTables.test.tsx`:

```tsx
import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { SnapshotTables } from './SnapshotTables'

describe('SnapshotTables', () => {
  test('renders node relation and evidence tabs without graph markup', () => {
    const markup = renderToStaticMarkup(
      <SnapshotTables
        snapshot={{
          nodes: [{ id: 'n1', label: '流感', entity_type: 'Disease', source_id: '', file_path: '' }],
          edges: [{ id: 'e1', source: 'n1', target: 'n2', keywords: '相关', source_id: 'chunk-1', file_path: '' }]
        }}
      />
    )

    expect(markup).toContain('节点')
    expect(markup).toContain('关系')
    expect(markup).toContain('证据问题')
    expect(markup).toContain('搜索')
    expect(markup).not.toContain('<svg')
  })
})
```

- [ ] **Step 3: Implement helpers**

Create `kgMaintenanceDisplay.ts`:

```ts
type SnapshotItem = Record<string, any>

export type EvidenceIssueRow = {
  id: string
  itemType: '节点' | '关系'
  itemId: string
  issue: string
}

export function buildEvidenceIssueRows(snapshot: { nodes?: SnapshotItem[]; edges?: SnapshotItem[] }): EvidenceIssueRow[] {
  const rows: EvidenceIssueRow[] = []
  for (const node of snapshot.nodes || []) {
    if (!node.source_id) rows.push({ id: `node:${node.id}:source_id`, itemType: '节点', itemId: String(node.id), issue: '缺少证据来源 ID' })
    if (!node.file_path) rows.push({ id: `node:${node.id}:file_path`, itemType: '节点', itemId: String(node.id), issue: '缺少来源文件' })
  }
  for (const edge of snapshot.edges || []) {
    const id = String(edge.id || `${edge.source}->${edge.target}`)
    if (!edge.source_id) rows.push({ id: `edge:${id}:source_id`, itemType: '关系', itemId: id, issue: '缺少证据来源 ID' })
    if (!edge.file_path) rows.push({ id: `edge:${id}:file_path`, itemType: '关系', itemId: id, issue: '缺少来源文件' })
  }
  return rows
}
```

- [ ] **Step 4: Implement `SnapshotTables`**

Create `SnapshotTables.tsx` with fixed-height scroll containers:

```tsx
import { useMemo, useState } from 'react'
import { buildEvidenceIssueRows } from './kgMaintenanceDisplay'

export function SnapshotTables({ snapshot }: { snapshot: Record<string, any> | null }) {
  const [tab, setTab] = useState<'nodes' | 'edges' | 'evidence'>('nodes')
  const [query, setQuery] = useState('')
  const nodes = Array.isArray(snapshot?.nodes) ? snapshot.nodes : []
  const edges = Array.isArray(snapshot?.edges) ? snapshot.edges : []
  const evidenceRows = useMemo(() => buildEvidenceIssueRows({ nodes, edges }), [nodes, edges])
  const normalizedQuery = query.trim().toLowerCase()
  const visibleNodes = nodes.filter((node) => JSON.stringify(node).toLowerCase().includes(normalizedQuery))
  const visibleEdges = edges.filter((edge) => JSON.stringify(edge).toLowerCase().includes(normalizedQuery))
  const visibleEvidence = evidenceRows.filter((row) => JSON.stringify(row).toLowerCase().includes(normalizedQuery))

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex gap-2">
          <button type="button" onClick={() => setTab('nodes')} className="rounded-md border px-3 py-1 text-sm">节点</button>
          <button type="button" onClick={() => setTab('edges')} className="rounded-md border px-3 py-1 text-sm">关系</button>
          <button type="button" onClick={() => setTab('evidence')} className="rounded-md border px-3 py-1 text-sm">证据问题</button>
        </div>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索" className="border-input h-9 rounded-md border px-3 text-sm" />
      </div>
      {tab === 'nodes' && <SimpleTable rows={visibleNodes} columns={['id', 'label', 'entity_type', 'source_id', 'file_path']} />}
      {tab === 'edges' && <SimpleTable rows={visibleEdges} columns={['id', 'source', 'target', 'keywords', 'source_id', 'file_path']} />}
      {tab === 'evidence' && <SimpleTable rows={visibleEvidence} columns={['itemType', 'itemId', 'issue']} />}
    </section>
  )
}

function SimpleTable({ rows, columns }: { rows: Record<string, any>[]; columns: string[] }) {
  return (
    <div className="border-border/70 h-[520px] overflow-auto rounded-md border">
      <table className="w-full text-left text-sm">
        <thead className="bg-muted sticky top-0">
          <tr>{columns.map((column) => <th key={column} className="px-3 py-2 font-medium">{column}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.id || index} className="border-t">
              {columns.map((column) => <td key={column} className="px-3 py-2 align-top">{String(row[column] ?? '')}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 5: Replace snapshot panel content**

In `IterationWorkbenchPanels.tsx`, make `SnapshotReviewPanel` render summary metrics and `<SnapshotTables snapshot={snapshot} />` instead of raw JSON-first content.

- [ ] **Step 6: Run tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance/kgMaintenanceDisplay.test.ts src/components/kg-maintenance/SnapshotTables.test.tsx src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx
```

Expected: tests pass.

- [ ] **Step 7: Commit**

```powershell
git add lightrag_webui/src/components/kg-maintenance/kgMaintenanceDisplay.ts lightrag_webui/src/components/kg-maintenance/kgMaintenanceDisplay.test.ts lightrag_webui/src/components/kg-maintenance/SnapshotTables.tsx lightrag_webui/src/components/kg-maintenance/SnapshotTables.test.tsx lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.tsx lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx
git commit -m "feat: show kg snapshot as searchable tables"
```

---

### Task 10: Proposal Approval Row UI and Revision Button

**Files:**
- Modify: `lightrag_webui/src/components/kg-maintenance/QualityAndApprovalPanels.tsx`
- Modify: `lightrag_webui/src/components/kg-maintenance/kgMaintenanceData.ts`
- Modify: `lightrag_webui/src/components/kg-maintenance/kgIterationLoadUtils.ts`
- Modify: `lightrag_webui/src/features/KGMaintenanceConsole.tsx`
- Test: `lightrag_webui/src/components/kg-maintenance/QualityAndApprovalPanels.test.tsx`
- Test: `lightrag_webui/src/components/kg-maintenance/kgMaintenanceData.test.ts`

- [ ] **Step 1: Write failing data tests for grouping versions**

Append to `kgMaintenanceData.test.ts`:

```ts
import { groupProposalVersions } from './kgMaintenanceData'

test('groups proposal versions and exposes latest version', () => {
  const proposals = [
    { id: 'prop-a', type: 'rule_change', target: 'x', proposedChange: 'v1', reason: '', confidence: '', risk: 'medium', requiresApproval: true, evidence: [], expectedMetricChange: '' },
    { id: 'prop-a-v2', type: 'rule_change', target: 'x', proposedChange: 'v2', reason: '', confidence: '', risk: 'low', requiresApproval: true, evidence: [], expectedMetricChange: '', parentId: 'prop-a' }
  ]

  const grouped = groupProposalVersions(proposals)

  expect(grouped).toHaveLength(1)
  expect(grouped[0].baseId).toBe('prop-a')
  expect(grouped[0].latest.id).toBe('prop-a-v2')
  expect(grouped[0].versions.map((item) => item.id)).toEqual(['prop-a', 'prop-a-v2'])
})
```

- [ ] **Step 2: Write failing approval panel test**

Create or update `QualityAndApprovalPanels.test.tsx`:

```tsx
import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { ApprovalPanel } from './QualityAndApprovalPanels'

describe('ApprovalPanel', () => {
  test('renders compact rows with accept reject expand and no required textareas', () => {
    const markup = renderToStaticMarkup(
      <ApprovalPanel
        approvalQueue={`# Approval Queue

proposals:
- id: prop-a
  type: rule_change
  target: relation_keyword_extraction
  proposed_change: Normalize relation keywords
  reason: Improve readability
  evidence:
  - source_id: chunk-1
  confidence: 0.8
  risk: medium
  requires_approval: true
  expected_metric_change: {}`}
        improvementBacklog=""
        acceptedChanges=""
        rejectedChanges=""
        onDecision={() => undefined}
        onRequestRevision={() => undefined}
      />
    )

    expect(markup).toContain('prop-a')
    expect(markup).toContain('接受')
    expect(markup).toContain('拒绝')
    expect(markup).toContain('展开')
    expect(markup).not.toContain('审批理由')
    expect(markup).not.toContain('影响范围')
    expect(markup).not.toContain('验证 / 回滚说明')
  })

  test('shows agent revision action for rejected proposal', () => {
    const markup = renderToStaticMarkup(
      <ApprovalPanel
        approvalQueue={`proposals:\n- id: prop-a\n  type: rule_change\n  target: x\n  proposed_change: y\n  reason: z\n  evidence: []\n  confidence: 0.8\n  risk: medium\n  requires_approval: true\n  expected_metric_change: {}`}
        improvementBacklog=""
        acceptedChanges=""
        rejectedChanges="## prop-a\n"
        onDecision={() => undefined}
        onRequestRevision={() => undefined}
      />
    )

    expect(markup).toContain('已拒绝')
    expect(markup).toContain('让 Agent 修改')
  })
})
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance/kgMaintenanceData.test.ts src/components/kg-maintenance/QualityAndApprovalPanels.test.tsx
```

Expected: missing `groupProposalVersions`, old textareas still present.

- [ ] **Step 4: Update proposal data helpers**

In `kgMaintenanceData.ts`, extend `ProposalSummary`:

```ts
parentId?: string
```

Parse optional `parent_proposal_id`:

```ts
if (key === 'parent_proposal_id') current.parentId = value
```

Add grouping:

```ts
export type ProposalVersionGroup = {
  baseId: string
  latest: ProposalSummary
  versions: ProposalSummary[]
}

export function groupProposalVersions(proposals: ProposalSummary[]): ProposalVersionGroup[] {
  const groups = new Map<string, ProposalSummary[]>()
  for (const proposal of proposals) {
    const baseId = proposal.parentId || proposal.id.replace(/-v\d+$/i, '')
    groups.set(baseId, [...(groups.get(baseId) || []), proposal])
  }
  return Array.from(groups.entries()).map(([baseId, versions]) => ({
    baseId,
    versions,
    latest: versions[versions.length - 1]
  }))
}
```

- [ ] **Step 5: Simplify approval panel**

Change `ApprovalPanelProps`:

```ts
onRequestRevision?: (proposal: ProposalSummary) => void | Promise<void>
```

Remove textarea state and `defer` button from default row. Render compact row:

```tsx
<div className="grid grid-cols-[120px_minmax(0,1fr)_100px_180px_48px] items-center gap-2">
  <StatusBadge decision={recordedDecision} />
  <div className="truncate text-sm font-medium">{proposal.proposedChange || proposal.id}</div>
  <span className="text-sm">{proposal.risk || '未知'}</span>
  <div className="flex gap-2">
    <Button size="sm" disabled={Boolean(recordedDecision)} onClick={() => void onDecision?.(proposal, 'accept', { reason: '', impactScope: '', verification: '', confirmation: '' })}>接受</Button>
    <Button variant="outline" size="sm" disabled={Boolean(recordedDecision)} onClick={() => void onDecision?.(proposal, 'reject', { reason: '', impactScope: '', verification: '', confirmation: '' })}>拒绝</Button>
  </div>
  <Button variant="ghost" size="sm" onClick={toggle}>展开</Button>
</div>
```

When rejected:

```tsx
{recordedDecision === 'reject' && (
  <Button variant="outline" size="sm" onClick={() => void onRequestRevision?.(proposal)}>
    让 Agent 修改
  </Button>
)}
```

- [ ] **Step 6: Wire revision API**

In `kgIterationLoadUtils.ts`, add `requestKBIterationProposalRevision` import and helper:

```ts
export async function requestProposalRevisionForWorkspace({
  requestWorkspace,
  getCurrentWorkspace,
  proposal,
  reloadWorkspaceData,
  requestRevision = requestKBIterationProposalRevision,
  onError
}: {
  requestWorkspace: string
  getCurrentWorkspace: () => string | null
  proposal: ProposalSummary
  reloadWorkspaceData: () => Promise<void>
  requestRevision?: typeof requestKBIterationProposalRevision
  onError?: (error: unknown) => void
}) {
  await runWorkspaceAction({
    requestWorkspace,
    getCurrentWorkspace,
    action: () => requestRevision(requestWorkspace, proposal.id),
    onSuccess: async () => reloadWorkspaceData(),
    onError: (error) => onError?.(error)
  })
}
```

Wire `KGMaintenanceConsole.tsx` to pass `onRequestRevision`.

- [ ] **Step 7: Run tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance/kgMaintenanceData.test.ts src/components/kg-maintenance/QualityAndApprovalPanels.test.tsx src/components/kg-maintenance/KGMaintenanceShell.test.tsx
```

Expected: tests pass.

- [ ] **Step 8: Commit**

```powershell
git add lightrag_webui/src/components/kg-maintenance/QualityAndApprovalPanels.tsx lightrag_webui/src/components/kg-maintenance/QualityAndApprovalPanels.test.tsx lightrag_webui/src/components/kg-maintenance/kgMaintenanceData.ts lightrag_webui/src/components/kg-maintenance/kgMaintenanceData.test.ts lightrag_webui/src/components/kg-maintenance/kgIterationLoadUtils.ts lightrag_webui/src/features/KGMaintenanceConsole.tsx
git commit -m "feat: simplify kg proposal approval workflow"
```

---

### Task 11: Execution and Validation Panels

**Files:**
- Create: `lightrag_webui/src/components/kg-maintenance/ExecutionAndValidationPanels.tsx`
- Create: `lightrag_webui/src/components/kg-maintenance/ExecutionAndValidationPanels.test.tsx`
- Modify: `lightrag_webui/src/features/KGMaintenanceConsole.tsx`
- Modify: `lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.tsx`

- [ ] **Step 1: Write failing panel tests**

Create `ExecutionAndValidationPanels.test.tsx`:

```tsx
import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { ExecutionPanel, ValidationPanel } from './ExecutionAndValidationPanels'

describe('ExecutionAndValidationPanels', () => {
  test('execution panel renders one primary action and apply summary', () => {
    const markup = renderToStaticMarkup(
      <ExecutionPanel
        acceptedChanges="## prop-a"
        applyResult="# Apply Result\n\n- Applied: 2\n- Blocked: 0\n"
        executing={false}
        onExecute={() => undefined}
      />
    )

    expect(markup).toContain('执行已接受变更')
    expect(markup).toContain('Applied: 2')
    expect(markup).not.toContain('improvement_backlog.md')
  })

  test('validation panel shows before after quality deltas and already achieved message', () => {
    const markup = renderToStaticMarkup(
      <ValidationPanel
        qualityBefore={{ overall: 88, metrics: { hierarchy_missing_branch_count: 4 } }}
        qualityAfter={{ overall: 97, metrics: { hierarchy_missing_branch_count: 0 } }}
        applyResult="# Apply Result\n\n- Applied: 0\n- hierarchy_missing_branch_count: 0 -> 0\n"
      />
    )

    expect(markup).toContain('88 → 97')
    expect(markup).toContain('4 → 0')
    expect(markup).toContain('没有新增写入，但当前质量已达标')
  })
})
```

- [ ] **Step 2: Implement panels**

Create `ExecutionAndValidationPanels.tsx`:

```tsx
import Button from '@/components/ui/Button'
import { PlayCircleIcon } from 'lucide-react'

export function ExecutionPanel({
  acceptedChanges,
  applyResult,
  executing,
  onExecute
}: {
  acceptedChanges: string
  applyResult: string
  executing: boolean
  onExecute: () => void
}) {
  const hasAccepted = /^##\s+/m.test(acceptedChanges)
  return (
    <section className="space-y-4">
      <div className="border-border/70 rounded-lg border p-4">
        <h2 className="text-base font-semibold">执行变更</h2>
        <p className="text-muted-foreground mt-1 text-sm">
          只执行已接受 Proposal，真实写入由确定性 Apply Engine 完成。
        </p>
        <Button className="mt-4" disabled={!hasAccepted || executing} onClick={onExecute}>
          <PlayCircleIcon className="size-4" />
          {executing ? '执行中' : '执行已接受变更'}
        </Button>
      </div>
      <pre className="bg-muted max-h-96 overflow-auto rounded-md p-3 text-sm whitespace-pre-wrap">
        {applyResult || '暂无执行结果。'}
      </pre>
    </section>
  )
}

export function ValidationPanel({
  qualityBefore,
  qualityAfter,
  applyResult
}: {
  qualityBefore: any
  qualityAfter: any
  applyResult: string
}) {
  const beforeOverall = qualityBefore?.overall ?? '—'
  const afterOverall = qualityAfter?.overall ?? qualityBefore?.overall ?? '—'
  const beforeMissing = qualityBefore?.metrics?.hierarchy_missing_branch_count ?? '—'
  const afterMissing = qualityAfter?.metrics?.hierarchy_missing_branch_count ?? qualityBefore?.metrics?.hierarchy_missing_branch_count ?? '—'
  const noApplied = /Applied:\s*0/i.test(applyResult)
  const isClean = Number(afterMissing) === 0
  return (
    <section className="space-y-4">
      <div className="grid gap-3 md:grid-cols-2">
        <MetricDelta label="总分" value={`${beforeOverall} → ${afterOverall}`} />
        <MetricDelta label="缺失分支" value={`${beforeMissing} → ${afterMissing}`} />
      </div>
      {noApplied && isClean ? (
        <div className="border-emerald-200 bg-emerald-50 rounded-lg border p-3 text-sm text-emerald-900">
          没有新增写入，但当前质量已达标。
        </div>
      ) : null}
      <pre className="bg-muted max-h-96 overflow-auto rounded-md p-3 text-sm whitespace-pre-wrap">
        {applyResult || '暂无验证结果。'}
      </pre>
    </section>
  )
}

function MetricDelta({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-border/70 rounded-lg border p-3">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="mt-1 text-xl font-semibold">{value}</div>
    </div>
  )
}
```

- [ ] **Step 3: Wire execute and validate sections**

In `MainPanel`, map:

```tsx
if (activeSection === 'execute') {
  return (
    <ExecutionPanel
      acceptedChanges={rules?.acceptedChanges || ''}
      applyResult={acceptedApplyResult}
      executing={acceptedExecuting}
      onExecute={onExecuteAcceptedChanges}
    />
  )
}
if (activeSection === 'validate') {
  return (
    <ValidationPanel
      qualityBefore={extractQualityBefore(acceptedApplyResult)}
      qualityAfter={qualityScore}
      applyResult={acceptedApplyResult}
    />
  )
}
```

Add `extractQualityBefore` in `kgMaintenanceDisplay.ts`:

```ts
export function extractQualityBefore(applyResult: string) {
  const missingMatch = applyResult.match(/hierarchy_missing_branch_count:\s*(\d+)\s*->\s*(\d+)/i)
  if (missingMatch) {
    return {
      metrics: {
        hierarchy_missing_branch_count: Number(missingMatch[1])
      }
    }
  }
  return {}
}
```

Use the current `qualityScore` as `qualityAfter`. `extractQualityBefore` extracts known before-values from `accepted_changes_apply_result.md`; unknown values render as `—`.

- [ ] **Step 4: Run tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance/ExecutionAndValidationPanels.test.tsx src/components/kg-maintenance/KGMaintenanceShell.test.tsx
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```powershell
git add lightrag_webui/src/components/kg-maintenance/ExecutionAndValidationPanels.tsx lightrag_webui/src/components/kg-maintenance/ExecutionAndValidationPanels.test.tsx lightrag_webui/src/features/KGMaintenanceConsole.tsx lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.tsx
git commit -m "feat: add kg execution and validation panels"
```

---

### Task 12: Wire Display Artifacts Into Workspace Bundle

**Files:**
- Modify: `lightrag_webui/src/components/kg-maintenance/kgIterationLoadUtils.ts`
- Modify: `lightrag_webui/src/features/KGMaintenanceConsole.tsx`
- Modify: `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx`
- Modify: `lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.tsx`
- Test: `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx`

- [ ] **Step 1: Add failing loader test for display artifact preference**

In `KGMaintenanceShell.test.tsx`, add:

```ts
test('workspace bundle loads zh display artifacts when available', async () => {
  const loaders = {
    getSummary: async () => summary,
    getQuality: async () => ({ workspace: 'medical-kb', runId: 'latest', quality: {}, report: '# Quality' }),
    getRules: async () => ({ workspace: 'medical-kb', qualityRules: '', knownIssues: '', acceptedChanges: '', rejectedChanges: '' }),
    getArtifact: async (_workspace: string, key: string) => ({ artifactKey: key, contentType: 'text/markdown', content: `original:${key}` }),
    getDisplayArtifact: async (_workspace: string, key: string) => ({ artifactKey: key, contentType: 'text/markdown', display: { language: 'zh', sourceFile: `${key}.md`, zhFile: `${key}.zh.md` }, content: `zh:${key}` }),
    getTrace: async () => ({ artifactKey: 'llm_review_trace', contentType: 'application/json', payload: {} }),
    getReport: async () => ({ artifactKey: 'llm_review_report', contentType: 'text/markdown', content: 'original report' }),
    getProposals: async () => ({ artifactKey: 'proposals_generated', contentType: 'text/markdown', content: 'original proposals' }),
    getJudgeReport: async () => ({ artifactKey: 'llm_judge_report', contentType: 'text/markdown', content: 'original judge' })
  }

  const bundle = await loadKGMaintenanceWorkspaceBundle('medical-kb', loaders as any)

  expect(bundle.displayArtifacts.kb_context.content).toBe('zh:kb_context')
  expect(bundle.kbContextArtifact).toBe('zh:kb_context')
})
```

- [ ] **Step 2: Modify loader type and helper**

Add to `kgIterationLoadUtils.ts`:

```ts
import { getKBIterationDisplayArtifact, regenerateKBIterationDisplayArtifact } from '@/api/lightrag'
```

Extend loader interface:

```ts
getDisplayArtifact: typeof getKBIterationDisplayArtifact
```

Add:

```ts
const optionalDisplayArtifact = async (
  loader: () => Promise<Awaited<ReturnType<typeof getKBIterationDisplayArtifact>>>,
  originalLoader: () => Promise<Awaited<ReturnType<typeof getKBIterationArtifact>>>
) => {
  try {
    return await loader()
  } catch {
    return await optionalMissingResponse(originalLoader, null as any)
  }
}
```

Return both `displayArtifacts` and legacy text fields for incremental compatibility.

- [ ] **Step 3: Use display artifacts in panels**

In `KGMaintenanceConsole.tsx`, add a state:

```ts
const [displayArtifacts, setDisplayArtifacts] = useState<Record<string, any>>({})
```

Assign after load:

```ts
setDisplayArtifacts(displayArtifactsPayload)
```

Pass display artifacts to `LLMReviewPanel`, `ArtifactFileSection`, and `ArtifactDrawer`.

- [ ] **Step 4: Run tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance/KGMaintenanceShell.test.tsx src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```powershell
git add lightrag_webui/src/components/kg-maintenance/kgIterationLoadUtils.ts lightrag_webui/src/features/KGMaintenanceConsole.tsx lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.tsx lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx
git commit -m "feat: load kg maintenance zh display artifacts"
```

---

### Task 13: Integrate Five Workflow Pages

**Files:**
- Modify: `lightrag_webui/src/features/KGMaintenanceConsole.tsx`
- Modify: `lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.tsx`
- Modify: `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx`
- Modify: `lightrag_webui/src/components/kg-maintenance/QualityAndApprovalPanels.tsx`
- Test: `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx`

- [ ] **Step 1: Add integration render test**

In `KGMaintenanceShell.test.tsx`, add:

```ts
test('main panel maps each workflow section to the correct Chinese work surface', () => {
  expect(renderMainPanel('check')).toContain('检查知识库')
  expect(renderMainPanel('llm-review')).toContain('LLM 审阅')
  expect(renderMainPanel('approval')).toContain('Proposal 审批')
  expect(renderMainPanel('execute')).toContain('执行变更')
  expect(renderMainPanel('validate')).toContain('验证结果')
})
```

- [ ] **Step 2: Update `MainPanel` section mapping**

Use this mapping:

```tsx
if (activeSection === 'check') return <CheckStep ... />
if (activeSection === 'llm-review') return <LLMReviewStep ... />
if (activeSection === 'approval') return <ApprovalStep ... />
if (activeSection === 'execute') return <ExecuteStep ... />
if (activeSection === 'validate') return <ValidateStep ... />
```

Each step starts with `AgentStepHeader`, then its focused content, then `ArtifactFileSection`.

- [ ] **Step 3: Remove old sections from store and tests**

Delete references to:

- `overview`
- `stage`
- `kb-summary`
- `quality`
- `snapshot`
- `decisions`
- `backlog`
- `memory`

Keep old panels only if they are reused inside the new 5 steps.

- [ ] **Step 4: Run integration frontend tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance/KGMaintenanceShell.test.tsx src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/QualityAndApprovalPanels.test.tsx
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```powershell
git add lightrag_webui/src/features/KGMaintenanceConsole.tsx lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.tsx lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx lightrag_webui/src/components/kg-maintenance/QualityAndApprovalPanels.tsx lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx
git commit -m "feat: organize kg maintenance as five step workflow"
```

---

### Task 14: Backend and Frontend Regression Test Sweep

**Files:**
- No source files unless tests expose a regression.

- [ ] **Step 1: Run backend focused tests**

Run:

```powershell
.\scripts\test.sh tests\kg\test_kb_iteration_zh_artifacts.py
.\scripts\test.sh tests\api\routes\test_kb_iteration_routes.py
.\scripts\test.sh tests\kg\test_kb_iteration_review_context.py
.\scripts\test.sh tests\kg\test_kb_iteration_agent_context.py
```

Expected: all pass.

- [ ] **Step 2: Run frontend focused tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance
```

Expected: all KG maintenance component tests pass.

- [ ] **Step 3: Run frontend build**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun run build
```

Expected: Vite build completes and writes production assets.

- [ ] **Step 4: Fix failures using systematic-debugging**

If any command fails, invoke `superpowers:systematic-debugging`, identify the first failing assertion or compiler error, make the smallest fix, and rerun the same command.

- [ ] **Step 5: Commit fixes if any**

```powershell
git add lightrag lightrag_webui tests
git commit -m "fix: stabilize kg maintenance workflow tests"
```

Skip this commit when no files changed.

---

### Task 15: Browser Verification

**Files:**
- No source files unless browser verification exposes UI defects.

- [ ] **Step 1: Start or reuse the WebUI dev server**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun run dev -- --host 127.0.0.1
```

Expected: Vite prints a local URL such as `http://127.0.0.1:5173/`.

- [ ] **Step 2: Open KG maintenance page**

Use the in-app browser or Playwright to open the running WebUI. If the app is served by the API server, use the existing local server URL instead.

Expected visual checks:

- Left nav contains exactly five steps.
- Top bar contains workspace, refresh, and all artifacts.
- No permanent right inspector column.
- Step header shows one recommended action.
- Proposal list rows show status, title/ID, risk, accept, reject, expand.
- Snapshot uses table tabs for nodes, relations, and evidence issues.
- Execution page has one main execute button.
- Validation page shows before/after quality numbers.

- [ ] **Step 3: Capture desktop screenshot**

Use Playwright or browser screenshot at a desktop viewport:

```ts
await page.setViewportSize({ width: 1440, height: 1000 })
await page.screenshot({ path: 'tmp/kg-maintenance-redesign-desktop.png', fullPage: true })
```

Expected: no overlapping text, no clipped buttons, no mixed English UI labels except preserved machine identifiers and source file names.

- [ ] **Step 4: Commit browser-discovered fixes if any**

```powershell
git add lightrag_webui/src
git commit -m "fix: polish kg maintenance workflow UI"
```

Skip this commit when no source files changed.

---

### Task 16: Final Verification and Handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-06-20-kg-iteration-agent-workbench-redesign-implementation.md` only if execution status checkboxes are updated during implementation.

- [ ] **Step 1: Run final targeted backend tests**

Run:

```powershell
.\scripts\test.sh tests\api\routes\test_kb_iteration_routes.py
.\scripts\test.sh tests\kg\test_kb_iteration_zh_artifacts.py
```

Expected: all pass.

- [ ] **Step 2: Run final frontend tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src/components/kg-maintenance
bun run build
```

Expected: all tests pass and build completes.

- [ ] **Step 3: Check git status**

Run:

```powershell
git status --short
```

Expected: only intentionally uncommitted user files remain, such as pre-existing `env.example` if still dirty.

- [ ] **Step 4: Summarize implementation**

Report:

- commits created
- tests run
- build result
- browser verification result
- any remaining risk, especially LLM translation cost or `.zh` generation fallback
