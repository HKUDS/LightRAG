# KG Accepted Proposal Real Apply Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把“Agent 执行已接受变更”从 LLM 文字执行记录升级为确定性的真实 KG apply 引擎，让已接受的 `add_hierarchy_branch` proposal 真正写入图谱、刷新快照与质量分数，并保持可重复、可审计、可回滚。

**Architecture:** 新增 `lightrag/kb_iteration/apply.py` 作为确定性 apply 层：读取 `accepted_changes.md` 的已接受记录，回连 `approval_queue.md` / `improvement_backlog.md` 中的 proposal 定义，只支持白名单 proposal 类型，先实现 `add_hierarchy_branch`。API 路由复用现有执行按钮入口，但不再依赖 LLM 才能执行；执行后提交 graph storage、重新运行 `run_iteration()` 生成 `kg_snapshot.json` / `quality_score.json` / `quality_report.md`，再写 `accepted_changes_apply_result.md/json`。

**Tech Stack:** Python 3.10+, FastAPI route, LightRAG graph storage `upsert_nodes_batch` / `upsert_edges_batch`, NetworkX GraphML snapshot runner, PyYAML, pytest, React 19 + TypeScript + Bun tests.

---

## Current Diagnosis

- Current branch: `codex/medical-kg-profile`.
- Current latest implementation commit: `f3475599 feat: execute accepted kg iteration changes`.
- Dirty worktree note: `env.example` is modified from unrelated prior work. Do not stage, edit, or revert it.
- Current button `POST /kb-iteration/{workspace}/accepted-changes/execute` writes `accepted_changes_execution.md/json` from an LLM response.
- The LLM report can say all four hierarchy branches were executed, but deterministic `snapshots/quality_score.json` still reports:
  - `hierarchy_required_branch_count: 9`
  - `hierarchy_present_branch_count: 5`
  - `hierarchy_missing_branch_count: 4`
- Missing branch keys remain:
  - `transmission_epidemiology`
  - `diagnosis_testing`
  - `high_risk_population`
  - `guideline_evidence`
- The next implementation must make deterministic quality the source of truth.

## Files And Responsibilities

- Modify `lightrag/kb_iteration/proposals.py`
  - Add `add_hierarchy_branch` to controlled mutation proposal types.

- Create `lightrag/kb_iteration/apply.py`
  - Parse proposal YAML from `approval_queue.md` and `improvement_backlog.md`.
  - Match accepted decision records to proposals.
  - Extract and validate ontology branch keys from proposal scope.
  - Apply supported changes to `rag.chunk_entity_relation_graph`.
  - Persist graph changes via `index_done_callback()`.
  - Render `accepted_changes_apply_result.md/json`.

- Modify `lightrag/api/routers/kb_iteration_routes.py`
  - Add artifact allowlist entries for `accepted_changes_apply_result`.
  - Route `/accepted-changes/execute` through deterministic apply first.
  - Rerun `run_iteration()` after graph apply.
  - Return status derived from real apply result, not LLM prose.

- Modify `lightrag_webui/src/api/lightrag.ts`
  - Add apply result status types and artifact metadata.

- Modify `lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.tsx`
  - Show deterministic apply result as “真实应用结果”.
  - Keep old `accepted_changes_execution.md` only as historical execution/report material if present.

- Modify `lightrag_webui/src/components/kg-maintenance/kgIterationLoadUtils.ts`
  - Load `accepted_changes_apply_result` alongside existing decision artifacts.

- Modify `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx`
  - Assert the new artifact appears in the decision/execution workflow.

- Modify `lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx`
  - Assert Chinese UI copy and disabled/enabled button behavior remain correct.

- Create `tests/kg/test_kb_iteration_apply.py`
  - Unit coverage for parsing, branch extraction, idempotency, unsupported cases, and quality improvement.

- Modify `tests/api/routes/test_kb_iteration_routes.py`
  - API coverage for deterministic execution, no accepted changes, no LLM requirement, and rerun quality.

---

## Task 1: Formalize `add_hierarchy_branch` As A Controlled Mutation

**Status:** Complete. Code-quality review requested and fixed an additional protective test so the whitelist entry is actually covered.

**Files:**
- Modify: `lightrag/kb_iteration/proposals.py`
- Test: `tests/kg/test_kb_iteration_apply.py`

- [x] **Step 1: Write the failing proposal-type test**

Create `tests/kg/test_kb_iteration_apply.py` with this initial test:

```python
from __future__ import annotations

from lightrag.kb_iteration.models import ImprovementProposal
from lightrag.kb_iteration.proposals import validate_proposal


def test_add_hierarchy_branch_is_a_known_mutation_type() -> None:
    proposal = ImprovementProposal(
        id="prop-add-branch-diagnosis",
        type="add_hierarchy_branch",
        target="hierarchy",
        proposed_change="Create branch diagnosis_testing.",
        reason="Missing required branch diagnosis_testing.",
        evidence=["item_id: influenza; source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=True,
        expected_metric_change={"hierarchy_missing_branch_count": -1},
    )

    validate_proposal(proposal)
```

- [x] **Step 2: Run the focused test and confirm current behavior**

Run:

```bash
uv run pytest tests/kg/test_kb_iteration_apply.py::test_add_hierarchy_branch_is_a_known_mutation_type -q
```

Expected before implementation:

```text
1 passed
```

If it already passes because unknown approved mutation types are currently allowed when `requires_approval=True`, keep the test and proceed. The next step still makes the type explicit for safety and clarity.

- [x] **Step 3: Add the proposal type to the mutation whitelist**

Modify `lightrag/kb_iteration/proposals.py`:

```python
MUTATION_PROPOSAL_TYPES = {
    "add_hierarchy_branch",
    "prompt_edit",
    "ontology_rule_change",
    "hierarchy_rule_change",
    "relation_rule_change",
    "workspace_rebuild",
    "kg_fact_correction",
    "web_display_change",
    "source_evidence_repair",
    "synonym_merge_rule",
    "relation_keyword_mapping",
}
```

- [x] **Step 4: Re-run the focused test**

Run:

```bash
uv run pytest tests/kg/test_kb_iteration_apply.py::test_add_hierarchy_branch_is_a_known_mutation_type -q
```

Expected:

```text
1 passed
```

- [x] **Step 5: Commit**

```bash
git add lightrag/kb_iteration/proposals.py tests/kg/test_kb_iteration_apply.py
git commit -m "feat: recognize hierarchy branch proposals"
```

---

## Task 2: Add Deterministic Apply Models And Proposal Loading

**Status:** Complete. Spec review required one correction to separate real apply artifacts from legacy execution artifacts; code-quality review required stricter proposal parsing and branch-key boundary tests.

**Files:**
- Create: `lightrag/kb_iteration/apply.py`
- Modify: `tests/kg/test_kb_iteration_apply.py`

- [x] **Step 1: Add failing tests for loading proposal definitions**

Append these tests to `tests/kg/test_kb_iteration_apply.py`:

```python
from pathlib import Path

from lightrag.kb_iteration.apply import (
    ApplyChangeStatus,
    branch_key_from_proposal,
    load_proposals_by_id,
)


def test_load_proposals_by_id_reads_approval_and_backlog(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "approval_queue.md").write_text(
        "\n".join(
            [
                "# Approval Queue",
                "",
                "proposals:",
                "- id: prop-add-branch-diagnosis",
                "  type: add_hierarchy_branch",
                "  target: hierarchy",
                "  proposed_change: Create branch diagnosis_testing.",
                "  reason: Missing diagnosis branch.",
                "  evidence:",
                "  - 'item_id: influenza; source_id: chunk-1'",
                "  confidence: 0.9",
                "  risk: low",
                "  requires_approval: true",
                "  expected_metric_change: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (package / "improvement_backlog.md").write_text(
        "\n".join(
            [
                "# Improvement Backlog",
                "",
                "proposals:",
                "- id: prop-add-branch-guideline",
                "  type: add_hierarchy_branch",
                "  target: hierarchy",
                "  proposed_change: Create branch guideline_evidence.",
                "  reason: Missing guideline branch.",
                "  evidence:",
                "  - 'item_id: guideline; source_id: chunk-2'",
                "  confidence: 0.8",
                "  risk: low",
                "  requires_approval: true",
                "  expected_metric_change: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(package)

    assert set(proposals) == {
        "prop-add-branch-diagnosis",
        "prop-add-branch-guideline",
    }
    assert proposals["prop-add-branch-diagnosis"]["type"] == "add_hierarchy_branch"


def test_branch_key_from_proposal_extracts_known_ontology_key() -> None:
    proposal = {
        "id": "prop-add-branch-diagnosis",
        "type": "add_hierarchy_branch",
        "target": "hierarchy",
        "proposed_change": "Create a new first-level branch with key 'diagnosis_testing'.",
        "reason": "The graph is missing diagnosis_testing.",
        "evidence": ["item_id: influenza; source_id: chunk-1"],
    }

    assert branch_key_from_proposal(proposal) == "diagnosis_testing"


def test_branch_key_from_proposal_rejects_unknown_key() -> None:
    proposal = {
        "id": "prop-bad",
        "type": "add_hierarchy_branch",
        "target": "hierarchy",
        "proposed_change": "Create a new first-level branch with key 'unsupported_branch'.",
        "reason": "Not in ontology.",
        "evidence": ["item_id: influenza; source_id: chunk-1"],
    }

    assert branch_key_from_proposal(proposal) == ""
    assert ApplyChangeStatus.BLOCKED.value == "blocked"
```

- [x] **Step 2: Run tests and confirm import failure**

Run:

```bash
uv run pytest tests/kg/test_kb_iteration_apply.py -q
```

Expected before implementation:

```text
ModuleNotFoundError: No module named 'lightrag.kb_iteration.apply'
```

- [x] **Step 3: Create `lightrag/kb_iteration/apply.py` with models and parsing**

Add this file:

```python
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.medical_kg.hierarchy import GROUP_NODE_TYPE
from lightrag.medical_kg.ontology import (
    CATEGORY_BY_KEY,
    TOP_LEVEL_MEDICAL_CATEGORIES,
    MedicalCategory,
)


APPLY_SOURCE = "kb_iteration_apply"
APPLY_RESULT_JSON = "accepted_changes_apply_result.json"
APPLY_RESULT_MARKDOWN = "accepted_changes_apply_result.md"


class ApplyChangeStatus(str, Enum):
    APPLIED = "applied"
    ALREADY_PRESENT = "already_present"
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ApplyChange:
    proposal_id: str
    proposal_type: str
    target: str
    status: ApplyChangeStatus
    action: str
    branch_key: str = ""
    branch_label: str = ""
    evidence: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class AcceptedApplyResult:
    workspace: str
    applied_at: str
    source_artifact: str
    proposal_ids: list[str]
    changes: list[ApplyChange]
    quality_before: dict[str, Any] = field(default_factory=dict)
    quality_after: dict[str, Any] = field(default_factory=dict)

    @property
    def applied_count(self) -> int:
        return sum(1 for change in self.changes if change.status == ApplyChangeStatus.APPLIED)

    @property
    def blocked_count(self) -> int:
        return sum(1 for change in self.changes if change.status == ApplyChangeStatus.BLOCKED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "applied_at": self.applied_at,
            "source_artifact": self.source_artifact,
            "proposal_ids": self.proposal_ids,
            "applied_count": self.applied_count,
            "blocked_count": self.blocked_count,
            "changes": [change.to_dict() for change in self.changes],
            "quality_before": self.quality_before,
            "quality_after": self.quality_after,
        }


def load_proposals_by_id(package_dir: str | Path) -> dict[str, dict[str, Any]]:
    package = Path(package_dir)
    proposals: dict[str, dict[str, Any]] = {}
    for filename in ("approval_queue.md", "improvement_backlog.md"):
        for proposal in _parse_proposals_markdown(package / filename):
            proposal_id = str(proposal.get("id", "")).strip()
            if proposal_id:
                proposals.setdefault(proposal_id, proposal)
    return proposals


def branch_key_from_proposal(proposal: dict[str, Any]) -> str:
    haystack = "\n".join(
        str(proposal.get(field_name, ""))
        for field_name in ("target", "proposed_change", "reason")
    )
    for category in TOP_LEVEL_MEDICAL_CATEGORIES:
        if _contains_identifier(haystack, category.key):
            return category.key
    return ""


def category_for_branch_key(branch_key: str) -> MedicalCategory | None:
    category = CATEGORY_BY_KEY.get(branch_key)
    if category is None:
        return None
    if category not in TOP_LEVEL_MEDICAL_CATEGORIES:
        return None
    return category


def write_apply_result_artifacts(result: AcceptedApplyResult, package_dir: str | Path) -> dict[str, Path]:
    package = Path(package_dir)
    package.mkdir(parents=True, exist_ok=True)
    json_path = package / APPLY_RESULT_JSON
    markdown_path = package / APPLY_RESULT_MARKDOWN
    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_apply_result_markdown(result), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def render_apply_result_markdown(result: AcceptedApplyResult) -> str:
    lines = [
        "# Accepted Changes Apply Result",
        "",
        f"- Workspace: {_inline(result.workspace)}",
        f"- Applied at: {_inline(result.applied_at)}",
        f"- Source: {_inline(result.source_artifact)}",
        f"- Proposal IDs: {_inline(', '.join(result.proposal_ids))}",
        f"- Applied count: {result.applied_count}",
        f"- Blocked count: {result.blocked_count}",
        "",
        "## Quality Delta",
        "",
    ]
    before_missing = _metric(result.quality_before, "hierarchy_missing_branch_count")
    after_missing = _metric(result.quality_after, "hierarchy_missing_branch_count")
    if before_missing is not None and after_missing is not None:
        lines.append(f"- hierarchy_missing_branch_count: {before_missing} -> {after_missing}")
    else:
        lines.append("- hierarchy_missing_branch_count: not available")
    lines.extend(["", "## Changes", ""])
    if not result.changes:
        lines.append("- none")
    for change in result.changes:
        label = change.branch_label or change.branch_key or change.target
        lines.append(
            "- "
            f"{_inline(change.proposal_id)} [{change.status.value}]: "
            f"{_inline(change.action)}"
        )
        if label:
            lines.append(f"  - branch: {_inline(label)}")
        if change.reason:
            lines.append(f"  - reason: {_inline(change.reason)}")
    return "\n".join(lines).rstrip() + "\n"


def _parse_proposals_markdown(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    markdown = path.read_text(encoding="utf-8")
    start = markdown.find("proposals:")
    if start < 0:
        return []
    try:
        payload = yaml.safe_load(markdown[start:]) or {}
    except yaml.YAMLError:
        return []
    proposals = payload.get("proposals", []) if isinstance(payload, dict) else []
    return [proposal for proposal in proposals if isinstance(proposal, dict)]


def _contains_identifier(text: str, identifier: str) -> bool:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])"
    )
    return bool(pattern.search(text))


def _inline(value: Any) -> str:
    return str(value or "").replace("\n", " ").replace("|", "\\|")


def _metric(score: dict[str, Any], name: str) -> Any:
    metrics = score.get("metrics", {}) if isinstance(score, dict) else {}
    return metrics.get(name) if isinstance(metrics, dict) else None
```

- [x] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/kg/test_kb_iteration_apply.py -q
```

Expected:

```text
4 passed
```

- [x] **Step 5: Commit**

```bash
git add lightrag/kb_iteration/apply.py tests/kg/test_kb_iteration_apply.py
git commit -m "feat: load accepted kg apply proposals"
```

---

## Task 3: Implement Real `add_hierarchy_branch` Graph Application

**Status:** Complete. Implemented deterministic graph apply with graph-key locks, scalar GraphML-safe node payloads, no evidence edges, idempotent `already_present` behavior, a runtime bugfix so explicit proposal branch keys win over unrelated ontology keywords in `reason`, and a final safety fix so accepted records with missing proposal definitions are blocked instead of inferred from accepted-record fields.

**Files:**
- Modify: `lightrag/kb_iteration/apply.py`
- Modify: `tests/kg/test_kb_iteration_apply.py`

- [x] **Step 1: Add fake graph and failing apply tests**

Append to `tests/kg/test_kb_iteration_apply.py`:

```python
import pytest

from lightrag.kb_iteration.apply import (
    AcceptedApplyResult,
    apply_accepted_changes_to_graph,
)
from lightrag.kb_iteration.models import KGSnapshot, SnapshotEdge, SnapshotNode
from lightrag.kb_iteration.quality import evaluate_snapshot_quality


class FakeGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, object]] = {}
        self.edges: dict[tuple[str, str], dict[str, object]] = {}
        self.committed = False

    async def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        return {node_id for node_id in node_ids if node_id in self.nodes}

    async def has_edge(self, source: str, target: str) -> bool:
        return tuple(sorted((source, target))) in self.edges

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict[str, object]]]) -> None:
        for node_id, node_data in nodes:
            self.nodes[node_id] = dict(node_data)

    async def upsert_edges_batch(self, edges: list[tuple[str, str, dict[str, object]]]) -> None:
        for source, target, edge_data in edges:
            self.edges[tuple(sorted((source, target)))] = dict(edge_data)

    async def index_done_callback(self) -> None:
        self.committed = True


class FakeRAG:
    def __init__(self, graph: FakeGraph) -> None:
        self.chunk_entity_relation_graph = graph


def _accepted_record(proposal_id: str) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "decision": "accept",
        "proposal_type": "add_hierarchy_branch",
        "proposal_target": "hierarchy",
    }


def _branch_proposal(proposal_id: str, branch_key: str, evidence: list[str] | None = None) -> dict[str, object]:
    return {
        "id": proposal_id,
        "type": "add_hierarchy_branch",
        "target": "hierarchy",
        "proposed_change": f"Create a new first-level branch with key '{branch_key}'.",
        "reason": f"The graph is missing {branch_key}.",
        "evidence": evidence or ["item_id: influenza; source_id: chunk-1"],
        "risk": "low",
        "requires_approval": True,
    }


@pytest.mark.asyncio
async def test_apply_add_hierarchy_branch_upserts_medical_group_node() -> None:
    graph = FakeGraph()
    graph.nodes["influenza"] = {
        "entity_id": "influenza",
        "entity_type": "Disease",
        "description": "disease",
        "source_id": "chunk-1",
        "file_path": "flu.pdf",
    }
    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[_accepted_record("prop-add-branch-diagnosis")],
        proposals_by_id={
            "prop-add-branch-diagnosis": _branch_proposal(
                "prop-add-branch-diagnosis",
                "diagnosis_testing",
            )
        },
    )

    assert isinstance(result, AcceptedApplyResult)
    assert result.applied_count == 1
    assert result.blocked_count == 0
    assert graph.committed is True
    assert any(
        data.get("medical_group") == "diagnosis_testing"
        and data.get("entity_type") == "MedicalGroup"
        for data in graph.nodes.values()
    )
    assert any("influenza" in edge_key for edge_key in graph.edges)


@pytest.mark.asyncio
async def test_apply_add_hierarchy_branch_is_idempotent() -> None:
    graph = FakeGraph()
    first = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[_accepted_record("prop-add-branch-guideline")],
        proposals_by_id={
            "prop-add-branch-guideline": _branch_proposal(
                "prop-add-branch-guideline",
                "guideline_evidence",
            )
        },
    )
    second = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[_accepted_record("prop-add-branch-guideline")],
        proposals_by_id={
            "prop-add-branch-guideline": _branch_proposal(
                "prop-add-branch-guideline",
                "guideline_evidence",
            )
        },
    )

    assert first.applied_count == 1
    assert second.applied_count == 0
    assert second.changes[0].status.value == "already_present"


@pytest.mark.asyncio
async def test_apply_blocks_unsupported_proposal_type() -> None:
    graph = FakeGraph()
    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "p2", "decision": "accept", "proposal_type": "web_display_change"}],
        proposals_by_id={
            "p2": {
                "id": "p2",
                "type": "web_display_change",
                "target": "MedicalHierarchyGraph.tsx",
                "proposed_change": "Improve legend.",
                "reason": "UI only.",
                "evidence": ["edge:e1"],
            }
        },
    )

    assert result.applied_count == 0
    assert result.blocked_count == 0
    assert result.changes[0].status.value == "unsupported"
    assert graph.nodes == {}


def test_quality_missing_count_goes_down_when_branch_node_exists() -> None:
    before = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-19T00:00:00Z",
        source_files=["flu.pdf"],
        metadata={"profile": "clinical_guideline_zh"},
        nodes=[],
        edges=[],
    )
    after = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-19T00:00:00Z",
        source_files=["flu.pdf"],
        metadata={"profile": "clinical_guideline_zh"},
        nodes=[
            SnapshotNode(
                id="诊断/检查",
                label="诊断/检查",
                entity_type="MedicalGroup",
                source_id="accepted_changes.md",
                file_path="accepted_changes.md",
                properties={"medical_group": "diagnosis_testing"},
            )
        ],
        edges=[
            SnapshotEdge(
                id="influenza->diagnosis",
                source="influenza",
                target="诊断/检查",
                keywords="属于",
                source_id="accepted_changes.md",
                file_path="accepted_changes.md",
            )
        ],
    )

    assert evaluate_snapshot_quality(after).metrics["hierarchy_missing_branch_count"] < (
        evaluate_snapshot_quality(before).metrics["hierarchy_missing_branch_count"]
    )
```

- [x] **Step 2: Run tests and confirm missing function failure**

Run:

```bash
uv run pytest tests/kg/test_kb_iteration_apply.py -q
```

Expected before implementation:

```text
ImportError: cannot import name 'apply_accepted_changes_to_graph'
```

- [x] **Step 3: Implement graph apply logic**

Extend `lightrag/kb_iteration/apply.py`:

```python
async def apply_accepted_changes_to_graph(
    *,
    rag: Any,
    workspace: str,
    records: list[dict[str, Any]],
    proposals_by_id: dict[str, dict[str, Any]],
) -> AcceptedApplyResult:
    graph = getattr(rag, "chunk_entity_relation_graph", None)
    if graph is None:
        changes = [
            ApplyChange(
                proposal_id=str(record.get("proposal_id", "")),
                proposal_type=str(record.get("proposal_type", "")),
                target=str(record.get("proposal_target", "")),
                status=ApplyChangeStatus.BLOCKED,
                action="Graph storage is not available.",
                reason="LightRAG graph storage is required for deterministic KG apply.",
            )
            for record in records
        ]
        return _result(workspace, records, changes)

    changes: list[ApplyChange] = []
    node_upserts: list[tuple[str, dict[str, object]]] = []
    edge_upserts: list[tuple[str, str, dict[str, object]]] = []

    for record in records:
        proposal_id = str(record.get("proposal_id", "")).strip()
        proposal = proposals_by_id.get(proposal_id, {})
        proposal_type = str(proposal.get("type") or record.get("proposal_type") or "")
        target = str(proposal.get("target") or record.get("proposal_target") or "")
        evidence = [str(item) for item in proposal.get("evidence", []) if str(item).strip()]

        if proposal_type != "add_hierarchy_branch":
            changes.append(
                ApplyChange(
                    proposal_id=proposal_id,
                    proposal_type=proposal_type,
                    target=target,
                    status=ApplyChangeStatus.UNSUPPORTED,
                    action="Skipped unsupported proposal type.",
                    evidence=evidence,
                    reason="Only add_hierarchy_branch is executable in this apply engine version.",
                )
            )
            continue

        branch_key = branch_key_from_proposal(proposal)
        category = category_for_branch_key(branch_key)
        if category is None:
            changes.append(
                ApplyChange(
                    proposal_id=proposal_id,
                    proposal_type=proposal_type,
                    target=target,
                    status=ApplyChangeStatus.BLOCKED,
                    action="Blocked invalid or missing hierarchy branch key.",
                    branch_key=branch_key,
                    evidence=evidence,
                    reason="The proposal must reference a TOP_LEVEL_MEDICAL_CATEGORIES key.",
                )
            )
            continue

        node_id = category.label
        if await graph.has_node(node_id):
            changes.append(
                ApplyChange(
                    proposal_id=proposal_id,
                    proposal_type=proposal_type,
                    target=target,
                    status=ApplyChangeStatus.ALREADY_PRESENT,
                    action="Hierarchy branch node already exists.",
                    branch_key=category.key,
                    branch_label=category.label,
                    evidence=evidence,
                )
            )
            continue

        source_ids = _source_ids_from_evidence(evidence)
        item_ids = _entity_item_ids_from_evidence(evidence)
        source_id = GRAPH_FIELD_SEP.join(source_ids) if source_ids else APPLY_SOURCE
        node_upserts.append((node_id, _branch_node_data(category, proposal_id, source_id)))
        for item_id in item_ids:
            if await graph.has_node(item_id):
                if hasattr(graph, "has_edge") and await graph.has_edge(item_id, node_id):
                    continue
                edge_upserts.append(
                    (
                        item_id,
                        node_id,
                        _branch_edge_data(item_id, category, proposal_id, source_id),
                    )
                )
        changes.append(
            ApplyChange(
                proposal_id=proposal_id,
                proposal_type=proposal_type,
                target=target,
                status=ApplyChangeStatus.APPLIED,
                action="Upserted MedicalGroup hierarchy branch node.",
                branch_key=category.key,
                branch_label=category.label,
                evidence=evidence,
            )
        )

    if node_upserts:
        await graph.upsert_nodes_batch(node_upserts)
    if edge_upserts:
        await graph.upsert_edges_batch(edge_upserts)
    if node_upserts or edge_upserts:
        await graph.index_done_callback()

    return _result(workspace, records, changes)


def _result(
    workspace: str,
    records: list[dict[str, Any]],
    changes: list[ApplyChange],
) -> AcceptedApplyResult:
    return AcceptedApplyResult(
        workspace=workspace,
        applied_at=datetime.now(UTC).isoformat(),
        source_artifact="accepted_changes.md",
        proposal_ids=[str(record.get("proposal_id", "")) for record in records],
        changes=changes,
    )


def _branch_node_data(
    category: MedicalCategory,
    proposal_id: str,
    source_id: str,
) -> dict[str, object]:
    return {
        "entity_id": category.label,
        "label": category.label,
        "entity_type": GROUP_NODE_TYPE,
        "description": f"KB iteration accepted proposal created top-level medical hierarchy branch: {category.label} ({category.key}).",
        "source_id": source_id,
        "file_path": "accepted_changes.md",
        "medical_group": category.key,
        "aliases": GRAPH_FIELD_SEP.join(category.aliases),
        "generated_by": APPLY_SOURCE,
        "accepted_proposal_ids": proposal_id,
    }


def _branch_edge_data(
    item_id: str,
    category: MedicalCategory,
    proposal_id: str,
    source_id: str,
) -> dict[str, object]:
    return {
        "weight": 0.0,
        "keywords": "属于",
        "description": f"{item_id} belongs to accepted hierarchy branch {category.label}.",
        "source_id": source_id,
        "file_path": "accepted_changes.md",
        "generated_by": APPLY_SOURCE,
        "accepted_proposal_ids": proposal_id,
    }


def _source_ids_from_evidence(evidence: list[str]) -> list[str]:
    source_ids = []
    for item in evidence:
        match = re.search(r"source_id:\s*([^;]+)", item)
        if match:
            source_ids.append(match.group(1).strip())
    return sorted({source_id for source_id in source_ids if source_id})


def _entity_item_ids_from_evidence(evidence: list[str]) -> list[str]:
    item_ids = []
    for item in evidence:
        match = re.search(r"item_id:\s*([^;]+)", item)
        if not match:
            continue
        item_id = match.group(1).strip()
        if item_id and "->" not in item_id:
            item_ids.append(item_id)
    return sorted(set(item_ids))
```

- [x] **Step 4: Run the apply tests**

Run:

```bash
uv run pytest tests/kg/test_kb_iteration_apply.py -q
```

Expected:

```text
8 passed
```

- [x] **Step 5: Commit**

```bash
git add lightrag/kb_iteration/apply.py tests/kg/test_kb_iteration_apply.py
git commit -m "feat: apply accepted hierarchy branch changes"
```

---

## Task 4: Wire The API Route To Real Apply And Refresh Quality

**Status:** Complete. The production execute endpoint now uses deterministic apply, writes `accepted_changes_apply_result.md/json`, refreshes snapshots/quality when changes apply, shares the normal run file lock, preserves audit artifacts when rerun fails, and keeps the legacy LLM execution helper as historical material only.

**Files:**
- Modify: `lightrag/api/routers/kb_iteration_routes.py`
- Modify: `tests/api/routes/test_kb_iteration_routes.py`

- [x] **Step 1: Add failing API tests**

Add tests near the existing accepted-changes execution tests in `tests/api/routes/test_kb_iteration_routes.py`:

```python
def test_execute_accepted_changes_does_not_require_llm_client(tmp_path: Path, monkeypatch):
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
        lambda _rag: pytest.fail("LLM client must not be required for deterministic apply"),
    )

    class FakeResult:
        applied_count = 1
        blocked_count = 0

        def __init__(self):
            self.proposal_ids = ["p1"]
            self.quality_before = {}
            self.quality_after = {}

        def to_dict(self):
            return {
                "workspace": "influenza_medical_v1",
                "proposal_ids": ["p1"],
                "applied_count": 1,
                "blocked_count": 0,
                "changes": [],
            }

    async def fake_apply(**kwargs):
        return FakeResult()

    def fake_write(result, package_dir):
        (Path(package_dir) / "accepted_changes_apply_result.md").write_text(
            "# Accepted Changes Apply Result\n",
            encoding="utf-8",
        )
        (Path(package_dir) / "accepted_changes_apply_result.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        return {}

    monkeypatch.setattr(routes, "apply_accepted_changes_to_graph", fake_apply)
    monkeypatch.setattr(routes, "load_proposals_by_id", lambda package: {"p1": {"id": "p1"}})
    monkeypatch.setattr(routes, "write_apply_result_artifacts", fake_write)
    monkeypatch.setattr(routes, "run_iteration", lambda **kwargs: None)

    response = client.post(
        "/kb-iteration/influenza_medical_v1/accepted-changes/execute",
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "applied_changes"
    assert payload["artifactKey"] == "accepted_changes_apply_result"
    assert payload["appliedCount"] == 1
    assert (fixture.package / "accepted_changes_apply_result.md").exists()
```

- [x] **Step 2: Run the focused API test and confirm failure**

Run:

```bash
uv run pytest tests/api/routes/test_kb_iteration_routes.py::test_execute_accepted_changes_does_not_require_llm_client -q
```

Expected before implementation:

```text
Failed: LLM client must not be required for deterministic apply
```

- [x] **Step 3: Add imports and artifact allowlist**

Modify `lightrag/api/routers/kb_iteration_routes.py` imports:

```python
from lightrag.kb_iteration.apply import (
    APPLY_RESULT_JSON,
    APPLY_RESULT_MARKDOWN,
    apply_accepted_changes_to_graph,
    load_proposals_by_id,
    write_apply_result_artifacts,
)
```

Add to `ARTIFACTS`:

```python
    "accepted_changes_apply_result": (
        APPLY_RESULT_MARKDOWN,
        "text/markdown",
    ),
```

- [x] **Step 4: Replace LLM execution inside the route**

In `execute_accepted_changes`, replace the block that creates `_default_llm_review_client()` and calls `_record_accepted_changes_execution()` with:

```python
                proposals_by_id = load_proposals_by_id(package_dir)
                quality_before = _optional_json_artifact(
                    args, workspace, "quality_score", {}
                )
                result = await apply_accepted_changes_to_graph(
                    rag=rag,
                    workspace=workspace,
                    records=records,
                    proposals_by_id=proposals_by_id,
                )
                if result.applied_count:
                    profile = _current_snapshot_profile(args, workspace)
                    await asyncio.to_thread(
                        run_iteration,
                        workspace=workspace,
                        storage_root=args.working_dir,
                        input_root=args.input_dir,
                        output_root=_output_root(args),
                        profile=profile,
                    )
                quality_after = _optional_json_artifact(
                    args, workspace, "quality_score", {}
                )
                result = AcceptedApplyResult(
                    workspace=result.workspace,
                    applied_at=result.applied_at,
                    source_artifact=result.source_artifact,
                    proposal_ids=result.proposal_ids,
                    changes=result.changes,
                    quality_before=quality_before,
                    quality_after=quality_after,
                )
                write_apply_result_artifacts(result, package_dir)
                _append_accepted_apply_log(args, workspace, result)
                response_status = (
                    "applied_changes"
                    if result.applied_count
                    else "no_applicable_changes"
                )
                return {
                    "workspace": workspace,
                    "status": response_status,
                    "proposalIds": result.proposal_ids,
                    "appliedCount": result.applied_count,
                    "blockedCount": result.blocked_count,
                    "artifactKey": "accepted_changes_apply_result",
                }
```

Also import `AcceptedApplyResult` from the new module.

- [x] **Step 5: Add route helpers**

Add below `_append_accepted_execution_log`:

```python
def _current_snapshot_profile(args, workspace: str) -> str | None:
    snapshot = _optional_json_artifact(args, workspace, "kg_snapshot", {})
    if not isinstance(snapshot, dict):
        return None
    metadata = snapshot.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    profile = metadata.get("profile")
    return str(profile) if profile else None


def _append_accepted_apply_log(args, workspace: str, result: AcceptedApplyResult) -> None:
    log_path = _workspace_dir(args, workspace) / "iteration_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = [
        "",
        "## Accepted Changes Apply",
        "",
        "- phase: accepted_changes_apply",
        f"- accepted_proposal_ids: {', '.join(result.proposal_ids)}",
        "- artifact: accepted_changes_apply_result.md",
        f"- applied_count: {result.applied_count}",
        f"- blocked_count: {result.blocked_count}",
        "",
    ]
    with log_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(entry))
```

- [x] **Step 6: Run API tests**

Run:

```bash
uv run pytest tests/api/routes/test_kb_iteration_routes.py -q
```

Expected:

```text
all tests in test_kb_iteration_routes.py passed
```

- [x] **Step 7: Commit**

```bash
git add lightrag/api/routers/kb_iteration_routes.py tests/api/routes/test_kb_iteration_routes.py
git commit -m "feat: execute accepted kg changes deterministically"
```

---

## Task 5: Update Web UI To Show Real Apply Result

**Status:** Complete. WebUI now loads `accepted_changes_apply_result`, shows `accepted_changes_apply_result.md` as `真实应用结果`, keeps `accepted_changes_execution.md` as `执行报告`, and reloads workspace artifacts after execution through the existing refresh path.

**Files:**
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Modify: `lightrag_webui/src/components/kg-maintenance/kgIterationLoadUtils.ts`
- Modify: `lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.tsx`
- Modify: `lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx`
- Modify: `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx`

- [x] **Step 1: Add failing UI tests**

In `IterationWorkbenchPanels.test.tsx`, add an assertion in the decision execution panel test:

```tsx
expect(markup).toContain('真实应用结果')
expect(markup).toContain('accepted_changes_apply_result.md')
```

In `KGMaintenanceShell.test.tsx`, extend the artifact keys fixture to include:

```ts
'accepted_changes_apply_result',
```

Then assert:

```tsx
expect(markup).toContain('accepted_changes_apply_result.md')
```

- [x] **Step 2: Run focused UI tests and confirm failure**

Run:

```bash
cd lightrag_webui
npx --yes bun test src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx src/components/kg-maintenance/KGMaintenanceShell.test.tsx
```

Expected before implementation:

```text
tests fail because accepted_changes_apply_result.md is not rendered
```

- [x] **Step 3: Extend API and loader types**

In `lightrag_webui/src/api/lightrag.ts`, update accepted execution status type:

```ts
export interface KBIterationAcceptedChangesExecutionResponse {
  workspace: string
  status:
    | 'applied_changes'
    | 'no_applicable_changes'
    | 'no_accepted_changes'
    | 'execution_recorded'
    | string
  proposalIds: string[]
  executedCount?: number
  appliedCount?: number
  blockedCount?: number
  artifactKey: string
}
```

In `kgIterationLoadUtils.ts`, add a loader call for:

```ts
loaders.getArtifact(requestWorkspace, 'accepted_changes_apply_result')
```

Expose it as `acceptedApplyResult`.

- [x] **Step 4: Render real apply result**

In `IterationWorkbenchPanels.tsx`, extend `DecisionExecutionPanel` props:

```ts
acceptedApplyResult: string
```

Render this panel before historical execution report:

```tsx
<MarkdownArtifactPanel
  icon={<PlayCircleIcon className="size-4 text-emerald-600" />}
  title="真实应用结果"
  fileName="accepted_changes_apply_result.md"
  content={acceptedApplyResult}
  emptyText="暂无真实应用结果。点击执行后，Agent 会写入图谱并刷新质量报告。"
/>
```

Keep the existing `accepted_changes_execution.md` panel but retitle it as:

```tsx
title="执行报告"
```

- [x] **Step 5: Update container wiring**

In `KGMaintenanceConsole.tsx` or the component that passes loaded artifacts into `DecisionExecutionPanel`, pass:

```tsx
acceptedApplyResult={artifacts.acceptedApplyResult}
```

After `executeKBIterationAcceptedChanges()` succeeds, reload summary and artifact content so the user sees “已应用 / 已跳过 / 已阻塞” without a manual refresh.

- [x] **Step 6: Run focused UI tests**

Run:

```bash
cd lightrag_webui
npx --yes bun test src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx src/components/kg-maintenance/KGMaintenanceShell.test.tsx src/components/kg-maintenance/kgIterationLoadUtils.test.ts
```

Expected:

```text
all focused UI tests pass
```

- [x] **Step 7: Commit**

```bash
git add lightrag_webui/src/api/lightrag.ts lightrag_webui/src/components/kg-maintenance/kgIterationLoadUtils.ts lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.tsx lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx
git commit -m "feat: show kg apply results in maintenance UI"
```

---

## Task 6: Verification, Real Workspace Run, And Safety Review

**Status:** Complete. Focused backend/frontend verification passed. Real workspace `influenza_medical_v1` applied accepted hierarchy branches, corrected a discovered branch-key parsing bug, removed the isolated erroneous `treatment` node generated by the pre-fix run, reached deterministic quality `required=9 / present=9 / missing=0`, passed idempotency, and was browser-verified in the WebUI.

**Files:**
- Modify if needed: `docs/superpowers/plans/2026-06-19-kg-accepted-proposal-real-apply-engine-implementation.md`
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

- [x] **Step 1: Run backend focused tests**

Run:

```bash
uv run pytest tests/kg/test_kb_iteration_apply.py tests/kg/test_kb_iteration_quality.py tests/api/routes/test_kb_iteration_routes.py -q
```

Expected:

```text
all focused backend tests pass
```

- [x] **Step 2: Run backend lint**

Run:

```bash
uv run ruff check lightrag/kb_iteration/apply.py lightrag/kb_iteration/proposals.py lightrag/api/routers/kb_iteration_routes.py tests/kg/test_kb_iteration_apply.py tests/api/routes/test_kb_iteration_routes.py
```

Expected:

```text
All checks passed!
```

- [x] **Step 3: Run frontend tests, lint, and build**

Run:

```bash
cd lightrag_webui
npx --yes bun test src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx src/components/kg-maintenance/KGMaintenanceShell.test.tsx src/components/kg-maintenance/kgIterationLoadUtils.test.ts
npx --yes bun run lint
npx --yes bun run build
```

Expected:

```text
focused tests pass
lint passes
build passes
```

- [x] **Step 4: Restart server and run real workspace apply**

Run the maintained desktop restart script or the existing API server workflow for workspace `influenza_medical_v1`, then call:

```bash
curl -X POST "http://127.0.0.1:9621/kb-iteration/influenza_medical_v1/accepted-changes/execute" -H "X-API-Key: <configured key>"
```

Expected response shape:

```json
{
  "workspace": "influenza_medical_v1",
  "status": "applied_changes",
  "proposalIds": [
    "prop-add-branch-diagnosis",
    "prop-add-branch-guideline",
    "prop-add-branch-transmission",
    "prop-add-branch-high-risk"
  ],
  "appliedCount": 4,
  "blockedCount": 0,
  "artifactKey": "accepted_changes_apply_result"
}
```

Expected deterministic quality after apply:

```json
{
  "hierarchy_required_branch_count": 9,
  "hierarchy_present_branch_count": 9,
  "hierarchy_missing_branch_count": 0
}
```

- [x] **Step 5: Run the same apply call again to verify idempotency**

Run the same POST again.

Expected:

```json
{
  "status": "no_applicable_changes",
  "appliedCount": 0,
  "blockedCount": 0
}
```

`accepted_changes_apply_result.md` should show the four changes as `already_present`, and the graph should not gain duplicate nodes or duplicate hierarchy edges.

- [x] **Step 6: Browser verification**

Open the WebUI at the active server URL and verify:

- Decision/execution area shows `真实应用结果`.
- The result file is `accepted_changes_apply_result.md`.
- The quality panel shows hierarchy missing count `0`.
- The old LLM execution report no longer serves as proof of mutation.
- No graph canvas is added to the KG Maintenance snapshot view.

- [x] **Step 7: Update planning memory**

Update:

- `task_plan.md`
- `findings.md`
- `progress.md`

Record:

- real apply engine implemented
- verification commands and results
- real workspace before/after quality metrics
- `env.example` stayed untouched

- [x] **Step 8: Final review and commit docs**

Run a focused review pass. If clean:

```bash
git add docs/superpowers/plans/2026-06-19-kg-accepted-proposal-real-apply-engine-implementation.md task_plan.md findings.md progress.md
git commit -m "docs: plan accepted kg proposal apply engine"
```

---

## Acceptance Criteria

- The execution button applies accepted `add_hierarchy_branch` proposals without requiring an LLM client.
- Only accepted records from `accepted_changes.md` are considered.
- The engine only executes proposal IDs that can be found in `approval_queue.md` or `improvement_backlog.md`.
- Unsupported proposal types are skipped and recorded as `unsupported`, not silently executed.
- Invalid branch keys are recorded as `blocked`, not guessed.
- The graph receives `MedicalGroup` nodes with `medical_group` equal to the ontology key.
- Evidence entity links are added only when the evidence item exists as a graph node.
- Running apply twice does not duplicate branch nodes or edges.
- `run_iteration()` refreshes snapshot and quality artifacts after real graph changes.
- `accepted_changes_apply_result.md/json` records deterministic results and quality before/after.
- WebUI shows the deterministic apply result in Chinese.
- `snapshots/quality_score.json` is the proof of success; LLM execution prose is not.

## Execution Choice

Plan complete and saved to `docs/superpowers/plans/2026-06-19-kg-accepted-proposal-real-apply-engine-implementation.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fastest and safest for this multi-file change.
2. **Inline Execution** - execute tasks in this session using executing-plans, with checkpoints after backend and UI phases.
