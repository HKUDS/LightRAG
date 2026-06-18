# KG Iteration Multistage LLM Agent Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade KB iteration LLM review from a single review loop into a staged Agent pipeline that explains issues, infers missing branches, locates evidence, generates proposals, ranks repairs, and judges proposals while preserving human approval.

**Architecture:** Keep deterministic KB iteration artifacts as the source of truth. Add explicit hierarchy branch details, then add a staged backend pipeline that writes auditable JSON/Markdown artifacts and remains compatible with existing `llm-review` API responses. Extend the WebUI LLM review area to display the new stage artifacts in Chinese without applying any KG mutation automatically.

**Tech Stack:** Python dataclasses/JSON/YAML, FastAPI route tests, existing LightRAG KB iteration modules, React 19 + TypeScript + Bun tests, Tailwind UI components.

---

## Design Source

Read before implementation:

- `D:\LightRAG\docs\superpowers\specs\2026-06-19-kg-iteration-multistage-agent-pipeline-design.md`
- `D:\LightRAG\docs\superpowers\specs\2026-06-18-kg-maintenance-llm-review-loop-design.md`
- `D:\LightRAG\docs\superpowers\specs\2026-06-18-kg-maintenance-iteration-agent-workbench-design.md`
- `D:\LightRAG\AGENTS.md`

Important constraints:

- Do not expose or log the user's API key.
- Do not auto-apply KG mutations, patches, rule changes, prompt changes, workspace rebuilds, or WebUI behavior changes.
- Mutation proposals must remain `requires_approval=true`.
- LLM output is not medical evidence. Evidence must reference existing `source_id`, `file_path`, entity ids, relation ids, or quality metrics.
- UI text for the new Agent workflow must be Chinese while preserving terms such as LLM, KG, proposal, source_id, file_path, JSON, Markdown, workspace, and profile.
- Preserve unrelated dirty changes. At the time this plan was written, these files had pre-existing modifications: `env.example`, `lightrag/api/routers/kb_iteration_routes.py`, `tests/api/routes/test_kb_iteration_routes.py`.

## File Structure

Create:

- `D:\LightRAG\lightrag\kb_iteration\agent_context.py`
  - Builds observe context and stage-specific payloads from deterministic artifacts.
- `D:\LightRAG\lightrag\kb_iteration\agent_outputs.py`
  - Parses stage JSON, validates proposal stage output, and writes JSON/Markdown artifacts.
- `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`
  - Orchestrates Explain, Infer, Evidence, Propose, Rank, and Judge stages.
- `D:\LightRAG\lightrag\kb_iteration\prompts\explain_zh.md`
- `D:\LightRAG\lightrag\kb_iteration\prompts\infer_branches_zh.md`
- `D:\LightRAG\lightrag\kb_iteration\prompts\locate_evidence_zh.md`
- `D:\LightRAG\lightrag\kb_iteration\prompts\propose_zh.md`
- `D:\LightRAG\lightrag\kb_iteration\prompts\rank_repairs_zh.md`
- `D:\LightRAG\tests\kg\test_kb_iteration_agent_context.py`
- `D:\LightRAG\tests\kg\test_kb_iteration_agent_outputs.py`
- `D:\LightRAG\tests\kg\test_kb_iteration_agent_pipeline.py`

Modify:

- `D:\LightRAG\lightrag\kb_iteration\models.py`
  - Add optional structured branch details to `QualityScore`.
- `D:\LightRAG\lightrag\kb_iteration\quality.py`
  - Compute and report `hierarchy_branches`.
- `D:\LightRAG\lightrag\kb_iteration\review_context.py`
  - Include `hierarchy_branches` in LLM context and avoid falling back to unrelated edges for hierarchy-only findings.
- `D:\LightRAG\lightrag\kb_iteration\review_loop.py`
  - Add mode-aware wrapper or delegate from API to the new pipeline while keeping old loop available.
- `D:\LightRAG\lightrag\api\routers\kb_iteration_routes.py`
  - Add new artifact keys, request mode, and pipeline dispatch.
- `D:\LightRAG\tests\kg\test_kb_iteration_quality.py`
- `D:\LightRAG\tests\kg\test_kb_iteration_review_context.py`
- `D:\LightRAG\tests\kg\test_kb_iteration_review_loop.py`
- `D:\LightRAG\tests\api\routes\test_kb_iteration_routes.py`
- `D:\LightRAG\lightrag_webui\src\api\lightrag.ts`
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\kgIterationLoadUtils.ts`
- `D:\LightRAG\lightrag_webui\src\features\KGMaintenanceConsole.tsx`
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx`
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.test.tsx`

## Task 1: Deterministic Hierarchy Branch Details

**Files:**

- Modify: `D:\LightRAG\lightrag\kb_iteration\models.py`
- Modify: `D:\LightRAG\lightrag\kb_iteration\quality.py`
- Modify: `D:\LightRAG\tests\kg\test_kb_iteration_quality.py`

- [ ] **Step 1: Add failing tests for hierarchy branch details**

Append these tests to `D:\LightRAG\tests\kg\test_kb_iteration_quality.py`:

```python
def test_quality_score_exposes_hierarchy_branch_details_for_medical_profile():
    categories = TOP_LEVEL_MEDICAL_CATEGORIES
    present = categories[:2]
    missing = categories[2:]
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-19T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode(
                present[0].key,
                "Category by key",
                "MedicalCategory",
                source_id="medical_kg_profile",
                file_path="medical_kg_profile",
            ),
            SnapshotNode(
                "alias-node",
                present[1].aliases[0],
                "MedicalCategory",
                source_id="medical_kg_profile",
                file_path="medical_kg_profile",
            ),
        ],
        edges=[],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    branches = score.details["hierarchy_branches"]
    assert [branch["key"] for branch in branches["required"]] == [
        category.key for category in categories
    ]
    assert [branch["key"] for branch in branches["present"]] == [
        category.key for category in present
    ]
    assert [branch["key"] for branch in branches["missing"]] == [
        category.key for category in missing
    ]
    assert branches["present"][0]["matched_node_ids"] == [present[0].key]
    assert branches["present"][1]["matched_node_ids"] == ["alias-node"]


def test_quality_artifacts_include_hierarchy_branch_details(tmp_path: Path):
    category = TOP_LEVEL_MEDICAL_CATEGORIES[0]
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-19T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode(
                category.label,
                category.label,
                "MedicalCategory",
                source_id="medical_kg_profile",
                file_path="medical_kg_profile",
            )
        ],
        edges=[],
        metadata={"profile": "clinical_guideline_zh"},
    )
    score = evaluate_snapshot_quality(snapshot)

    written = write_quality_artifacts(score, tmp_path)

    payload = json.loads(written["quality_score"].read_text(encoding="utf-8"))
    assert "details" in payload
    assert "hierarchy_branches" in payload["details"]
    report = written["quality_report"].read_text(encoding="utf-8")
    assert "## Hierarchy Branches" in report
    assert category.key in report
```

- [ ] **Step 2: Run the failing quality tests**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_quality.py -q
```

Expected:

- FAIL because `QualityScore` has no `details` field and `quality_score.json` does not include `hierarchy_branches`.

- [ ] **Step 3: Extend `QualityScore` with details**

In `D:\LightRAG\lightrag\kb_iteration\models.py`, change `QualityScore` to:

```python
@dataclass(frozen=True)
class QualityScore:
    overall: int
    subscores: dict[str, int]
    metrics: dict[str, Any]
    findings: list[QualityFinding] = field(default_factory=list)
    critical_blockers: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "subscores": self.subscores,
            "metrics": self.metrics,
            "findings": [finding.to_dict() for finding in self.findings],
            "critical_blockers": self.critical_blockers,
            "details": self.details,
        }
```

- [ ] **Step 4: Implement hierarchy branch detail computation**

In `D:\LightRAG\lightrag\kb_iteration\quality.py`, replace `_hierarchy_metrics()` with a branch-detail helper and metrics wrapper:

```python
def _hierarchy_details(snapshot: KGSnapshot) -> dict[str, list[dict[str, Any]]]:
    required = []
    present = []
    missing = []
    if not _needs_medical_hierarchy(snapshot):
        return {"required": required, "present": present, "missing": missing}

    node_matches_by_key = _matched_hierarchy_nodes_by_key(snapshot)
    for category in TOP_LEVEL_MEDICAL_CATEGORIES:
        branch = {
            "key": category.key,
            "label": category.label,
            "aliases": list(category.aliases),
        }
        required.append(branch)
        matched_node_ids = sorted(node_matches_by_key.get(category.key, set()))
        if matched_node_ids:
            present.append({**branch, "matched_node_ids": matched_node_ids})
        else:
            missing.append(branch)
    return {"required": required, "present": present, "missing": missing}


def _hierarchy_metrics_from_details(
    details: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    return {
        "required": len(details["required"]),
        "present": len(details["present"]),
        "missing": len(details["missing"]),
    }


def _matched_hierarchy_nodes_by_key(snapshot: KGSnapshot) -> dict[str, set[str]]:
    required_identifiers = _medical_category_identifiers()
    matched: dict[str, set[str]] = {key: set() for key in required_identifiers}
    for node in snapshot.nodes:
        node_identifiers = _node_hierarchy_identifiers(node)
        for key, identifiers in required_identifiers.items():
            if node_identifiers & identifiers:
                matched[key].add(node.id)
    return matched
```

Then in `evaluate_snapshot_quality()` replace:

```python
hierarchy = _hierarchy_metrics(snapshot)
```

with:

```python
hierarchy_details = _hierarchy_details(snapshot)
hierarchy = _hierarchy_metrics_from_details(hierarchy_details)
```

and return:

```python
return QualityScore(
    overall=overall,
    subscores=subscores,
    metrics=metrics,
    findings=findings,
    critical_blockers=critical_blockers,
    details={"hierarchy_branches": hierarchy_details},
)
```

- [ ] **Step 5: Update quality artifact tests for the new field**

In existing `test_write_quality_artifacts_creates_score_json_and_report`, update the key assertion:

```python
assert set(payload) == {
    "overall",
    "subscores",
    "metrics",
    "findings",
    "critical_blockers",
    "details",
}
assert payload["details"] == score.details
```

- [ ] **Step 6: Render hierarchy details in `quality_report.md`**

In `D:\LightRAG\lightrag\kb_iteration\quality.py`, add this block near the end of `_quality_report()` before `## Critical Blockers`:

```python
    branches = score.details.get("hierarchy_branches", {})
    if isinstance(branches, dict):
        lines.extend(["", "## Hierarchy Branches"])
        for section in ("required", "present", "missing"):
            values = branches.get(section, [])
            lines.append(f"- {section}: {len(values) if isinstance(values, list) else 0}")
            if isinstance(values, list):
                for branch in values:
                    if not isinstance(branch, dict):
                        continue
                    key = branch.get("key", "")
                    label = branch.get("label", "")
                    matched = branch.get("matched_node_ids", [])
                    if matched:
                        lines.append(f"  - {key} | {label} | matched: {', '.join(matched)}")
                    else:
                        lines.append(f"  - {key} | {label}")
```

- [ ] **Step 7: Run quality tests and commit**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_quality.py -q
```

Expected:

- PASS.

Commit:

```powershell
cd D:\LightRAG
git add lightrag/kb_iteration/models.py lightrag/kb_iteration/quality.py tests/kg/test_kb_iteration_quality.py
git commit -m "feat: expose kg hierarchy branch details"
```

## Task 2: Agent Context Builder

**Files:**

- Create: `D:\LightRAG\lightrag\kb_iteration\agent_context.py`
- Create: `D:\LightRAG\tests\kg\test_kb_iteration_agent_context.py`
- Modify: `D:\LightRAG\lightrag\kb_iteration\review_context.py`
- Modify: `D:\LightRAG\tests\kg\test_kb_iteration_review_context.py`

- [ ] **Step 1: Write failing agent context tests**

Create `D:\LightRAG\tests\kg\test_kb_iteration_agent_context.py`:

```python
import json
from pathlib import Path

from lightrag.kb_iteration.agent_context import (
    build_agent_observation,
    build_stage_context,
    write_agent_context,
)


def test_build_agent_observation_includes_quality_and_branch_details(tmp_path: Path):
    package = _write_agent_package(tmp_path)

    observation = build_agent_observation(package, workspace="influenza_medical_v1")

    assert observation["workspace"] == "influenza_medical_v1"
    assert observation["quality"]["overall"] == 88
    assert observation["hierarchy_branches"]["missing"][0]["key"] == "symptom"
    assert observation["artifact_status"]["kb_context.md"] is True
    assert "previously rejected" in observation["rules_memory"]["rejected_changes"]


def test_build_stage_context_for_evidence_limits_selected_items(tmp_path: Path):
    package = _write_agent_package(tmp_path)
    previous = {
        "issue_explanations": [
            {
                "id": "issue-1",
                "category": "hierarchy_completeness",
                "evidence_refs": ["quality:hierarchy_missing_branch_count=1"],
            }
        ],
        "missing_branches": [{"key": "symptom", "label": "症状"}],
    }

    context = build_stage_context(
        package,
        workspace="influenza_medical_v1",
        stage="locate_evidence",
        previous_outputs=previous,
    )

    assert context["stage"] == "locate_evidence"
    assert context["previous_outputs"] == previous
    assert [node["id"] for node in context["candidate_entities"]] == ["fever"]
    assert [edge["id"] for edge in context["candidate_relations"]] == ["edge-flu-fever"]
    assert context["evidence_windows"][0]["source_id"] == "chunk-1"


def test_write_agent_context_writes_json(tmp_path: Path):
    path = write_agent_context(
        tmp_path,
        "explain",
        {"stage": "explain", "focus": ["hierarchy_completeness"]},
    )

    assert path == tmp_path / "agent_context" / "explain-context.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stage"] == "explain"
    assert path.read_text(encoding="utf-8").endswith("\n")


def _write_agent_package(tmp_path: Path) -> Path:
    package = tmp_path / "package"
    snapshot_dir = package / "snapshots"
    snapshot_dir.mkdir(parents=True)
    _write_json(
        snapshot_dir / "kg_snapshot.json",
        {
            "workspace": "influenza_medical_v1",
            "generated_at": "2026-06-19T00:00:00Z",
            "source_files": ["guide.md"],
            "nodes": [
                {
                    "id": "flu",
                    "label": "流感",
                    "entity_type": "Disease",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                },
                {
                    "id": "fever",
                    "label": "发热",
                    "entity_type": "Symptom",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                },
            ],
            "edges": [
                {
                    "id": "edge-flu-fever",
                    "source": "flu",
                    "target": "fever",
                    "keywords": "临床表现",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                }
            ],
            "metadata": {"profile": "clinical_guideline_zh"},
        },
    )
    _write_json(
        snapshot_dir / "quality_score.json",
        {
            "overall": 88,
            "metrics": {"hierarchy_missing_branch_count": 1},
            "findings": [
                {
                    "severity": "medium",
                    "category": "hierarchy_completeness",
                    "message": "Medical hierarchy is missing expected branches.",
                    "evidence": ["branch:symptom"],
                    "suggested_fix_type": "add_hierarchy_branch",
                    "requires_approval": True,
                }
            ],
            "details": {
                "hierarchy_branches": {
                    "required": [{"key": "symptom", "label": "症状", "aliases": ["临床表现"]}],
                    "present": [],
                    "missing": [{"key": "symptom", "label": "症状", "aliases": ["临床表现"]}],
                }
            },
        },
    )
    (package / "kb_context.md").write_text("# KB Context\n", encoding="utf-8")
    (package / "accepted_changes.md").write_text("# Accepted\n", encoding="utf-8")
    (package / "rejected_changes.md").write_text(
        "# Rejected\n\n- previously rejected\n", encoding="utf-8"
    )
    return package


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
```

- [ ] **Step 2: Run the failing agent context tests**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_context.py -q
```

Expected:

- FAIL because `lightrag.kb_iteration.agent_context` does not exist.

- [ ] **Step 3: Implement `agent_context.py`**

Create `D:\LightRAG\lightrag\kb_iteration\agent_context.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


AGENT_CONTEXT_DIR = "agent_context"


def build_agent_observation(package_dir: str | Path, *, workspace: str) -> dict[str, Any]:
    package_path = Path(package_dir)
    quality = _read_json(package_path / "snapshots" / "quality_score.json", default={})
    snapshot = _read_json(package_path / "snapshots" / "kg_snapshot.json", default={})
    return {
        "workspace": workspace,
        "snapshot": {
            "workspace": snapshot.get("workspace", ""),
            "generated_at": snapshot.get("generated_at", ""),
            "node_count": len(_dict_items(snapshot.get("nodes"))),
            "edge_count": len(_dict_items(snapshot.get("edges"))),
            "source_files": snapshot.get("source_files", []),
            "metadata": snapshot.get("metadata", {}),
        },
        "quality": {
            "overall": quality.get("overall"),
            "subscores": quality.get("subscores", {}),
            "metrics": quality.get("metrics", {}),
            "findings": _dict_items(quality.get("findings")),
            "critical_blockers": quality.get("critical_blockers", []),
        },
        "hierarchy_branches": _hierarchy_branches(quality),
        "artifact_status": {
            "kb_context.md": (package_path / "kb_context.md").exists(),
            "quality_report.md": (package_path / "quality_report.md").exists(),
            "snapshots/kg_snapshot.json": (package_path / "snapshots" / "kg_snapshot.json").exists(),
            "snapshots/quality_score.json": (package_path / "snapshots" / "quality_score.json").exists(),
            "approval_queue.md": (package_path / "approval_queue.md").exists(),
            "improvement_backlog.md": (package_path / "improvement_backlog.md").exists(),
        },
        "kb_context": _read_text(package_path / "kb_context.md"),
        "rules_memory": {
            "accepted_changes": _read_text(package_path / "accepted_changes.md"),
            "rejected_changes": _read_text(package_path / "rejected_changes.md"),
        },
    }


def build_stage_context(
    package_dir: str | Path,
    *,
    workspace: str,
    stage: str,
    previous_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    package_path = Path(package_dir)
    snapshot = _read_json(package_path / "snapshots" / "kg_snapshot.json", default={})
    quality = _read_json(package_path / "snapshots" / "quality_score.json", default={})
    nodes = _dict_items(snapshot.get("nodes"))
    edges = _dict_items(snapshot.get("edges"))
    selected_nodes, selected_edges = _select_stage_items(nodes, edges, previous_outputs or {})
    return {
        "workspace": workspace,
        "stage": stage,
        "quality_findings": _dict_items(quality.get("findings")),
        "hierarchy_branches": _hierarchy_branches(quality),
        "previous_outputs": previous_outputs or {},
        "candidate_entities": selected_nodes,
        "candidate_relations": selected_edges,
        "evidence_windows": _evidence_windows(selected_nodes, selected_edges),
        "rules_memory": {
            "accepted_changes": _read_text(package_path / "accepted_changes.md"),
            "rejected_changes": _read_text(package_path / "rejected_changes.md"),
        },
    }


def write_agent_context(package_dir: str | Path, stage: str, context: dict[str, Any]) -> Path:
    output_path = Path(package_dir) / AGENT_CONTEXT_DIR / f"{stage}-context.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(context, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path


def _hierarchy_branches(quality: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    details = quality.get("details", {})
    if not isinstance(details, dict):
        return {"required": [], "present": [], "missing": []}
    branches = details.get("hierarchy_branches", {})
    if not isinstance(branches, dict):
        return {"required": [], "present": [], "missing": []}
    return {
        "required": _dict_items(branches.get("required")),
        "present": _dict_items(branches.get("present")),
        "missing": _dict_items(branches.get("missing")),
    }


def _select_stage_items(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    previous_outputs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    missing_keys = {
        str(branch.get("key", "")).casefold()
        for branch in previous_outputs.get("missing_branches", [])
        if isinstance(branch, dict)
    }
    missing_labels = {
        str(branch.get("label", "")).casefold()
        for branch in previous_outputs.get("missing_branches", [])
        if isinstance(branch, dict)
    }
    selected_nodes = [
        node
        for node in nodes
        if str(node.get("entity_type", "")).casefold() in missing_keys
        or str(node.get("label", "")).casefold() in missing_labels
    ]
    selected_ids = {str(node.get("id", "")) for node in selected_nodes}
    selected_edges = [
        edge
        for edge in edges
        if str(edge.get("source", "")) in selected_ids or str(edge.get("target", "")) in selected_ids
    ]
    if selected_nodes or selected_edges:
        return selected_nodes[:20], selected_edges[:20]
    return nodes[:20], edges[:20]


def _evidence_windows(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for item_type, items in (("entity", nodes), ("relation", edges)):
        for item in items:
            source_id = str(item.get("source_id", "")).strip()
            file_path = str(item.get("file_path", "")).strip()
            windows.append(
                {
                    "item_type": item_type,
                    "item_id": item.get("id", ""),
                    "source_id": source_id,
                    "file_path": file_path,
                    "evidence_status": "grounded" if source_id and file_path else "missing",
                }
            )
    return windows


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
```

- [ ] **Step 4: Update legacy review context for hierarchy-only focus**

Append this test to `D:\LightRAG\tests\kg\test_kb_iteration_review_context.py`:

```python
def test_build_review_context_includes_hierarchy_branches_without_unrelated_edges(
    tmp_path: Path,
):
    package = tmp_path / "package"
    snapshot_dir = package / "snapshots"
    snapshot_dir.mkdir(parents=True)
    _write_json(
        snapshot_dir / "kg_snapshot.json",
        {
            "nodes": [{"id": "flu", "label": "Flu", "entity_type": "Disease"}],
            "edges": [
                {"id": "edge-unrelated", "source": "flu", "target": "flu", "keywords": "self"}
            ],
        },
    )
    _write_json(
        snapshot_dir / "quality_score.json",
        {
            "findings": [
                {
                    "category": "hierarchy_completeness",
                    "message": "Medical hierarchy is missing expected branches.",
                    "evidence": ["branch:symptom"],
                    "suggested_fix_type": "add_hierarchy_branch",
                }
            ],
            "details": {
                "hierarchy_branches": {
                    "required": [{"key": "symptom", "label": "症状"}],
                    "present": [],
                    "missing": [{"key": "symptom", "label": "症状"}],
                }
            },
        },
    )

    context = build_review_context(
        package, round_id="round-001", focus=["hierarchy_missing_branch"]
    )

    assert context["hierarchy_branches"]["missing"] == [{"key": "symptom", "label": "症状"}]
    assert context["relations"] == []
```

Then modify `D:\LightRAG\lightrag\kb_iteration\review_context.py`:

```python
    return {
        "round_id": round_id,
        "focus": focus,
        "quality_findings": findings,
        "hierarchy_branches": _hierarchy_branches(quality),
        "entities": selected_nodes,
        "relations": selected_edges,
        "evidence_windows": _evidence_windows(selected_nodes, selected_edges),
        "rules_memory": {
            "accepted_changes": _read_text(package_path / "accepted_changes.md"),
            "rejected_changes": _read_text(package_path / "rejected_changes.md"),
        },
    }
```

Add helper:

```python
def _hierarchy_branches(quality: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    details = quality.get("details", {})
    if not isinstance(details, dict):
        return {"required": [], "present": [], "missing": []}
    branches = details.get("hierarchy_branches", {})
    if not isinstance(branches, dict):
        return {"required": [], "present": [], "missing": []}
    return {
        "required": _dict_items(branches.get("required")),
        "present": _dict_items(branches.get("present")),
        "missing": _dict_items(branches.get("missing")),
    }
```

In `_select_edges()`, before `return edges[:10]`, add:

```python
    if "hierarchy_missing_branch" in normalized_focus:
        return []
```

- [ ] **Step 5: Run context tests and commit**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_context.py tests/kg/test_kb_iteration_review_context.py -q
```

Expected:

- PASS.

Commit:

```powershell
cd D:\LightRAG
git add lightrag/kb_iteration/agent_context.py lightrag/kb_iteration/review_context.py tests/kg/test_kb_iteration_agent_context.py tests/kg/test_kb_iteration_review_context.py
git commit -m "feat: build staged kb agent contexts"
```

## Task 3: Agent Output Parsing And Artifacts

**Files:**

- Create: `D:\LightRAG\lightrag\kb_iteration\agent_outputs.py`
- Create: `D:\LightRAG\tests\kg\test_kb_iteration_agent_outputs.py`

- [ ] **Step 1: Write failing output parser tests**

Create `D:\LightRAG\tests\kg\test_kb_iteration_agent_outputs.py`:

```python
import json
from pathlib import Path

import pytest

from lightrag.kb_iteration.agent_outputs import (
    AgentStageOutput,
    parse_agent_stage_output,
    stage_output_to_markdown,
    write_agent_stage_artifacts,
)


def test_parse_agent_stage_output_requires_json_object():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_agent_stage_output("explain", "not json")

    with pytest.raises(ValueError, match="JSON object"):
        parse_agent_stage_output("explain", "[]")


def test_parse_propose_stage_validates_proposals():
    output = parse_agent_stage_output(
        "propose",
        json.dumps(
            {
                "proposals": [
                    {
                        "id": "proposal-hierarchy-symptom-001",
                        "type": "hierarchy_rule_change",
                        "target": "lightrag/medical_kg/ontology.py",
                        "proposed_change": "Add symptom as an expected hierarchy branch.",
                        "reason": "Symptom branch is missing from hierarchy details.",
                        "evidence": ["branch:symptom", "entity:fever"],
                        "confidence": 0.78,
                        "risk": "medium",
                        "requires_approval": True,
                        "expected_metric_change": {"hierarchy_completeness": 10},
                    }
                ]
            }
        ),
    )

    assert output.stage == "propose"
    assert output.proposals[0].id == "proposal-hierarchy-symptom-001"


def test_parse_propose_stage_rejects_ungrounded_mutation_proposal():
    with pytest.raises(ValueError, match="evidence"):
        parse_agent_stage_output(
            "propose",
            json.dumps(
                {
                    "proposals": [
                        {
                            "id": "proposal-empty-evidence",
                            "type": "hierarchy_rule_change",
                            "target": "lightrag/medical_kg/ontology.py",
                            "proposed_change": "Add a branch.",
                            "reason": "Missing branch.",
                            "evidence": [],
                            "confidence": 0.6,
                            "risk": "medium",
                            "requires_approval": True,
                            "expected_metric_change": {"hierarchy_completeness": 10},
                        }
                    ]
                }
            ),
        )


def test_stage_output_to_markdown_renders_chinese_sections():
    output = AgentStageOutput(
        stage="explain",
        payload={
            "issue_explanations": [
                {
                    "id": "issue-1",
                    "category": "hierarchy_completeness",
                    "explanation": "层级分支缺失。",
                    "impact": "影响导航。",
                    "evidence_refs": ["quality:hierarchy_missing_branch_count=1"],
                }
            ]
        },
    )

    markdown = stage_output_to_markdown(output)

    assert "# 问题解释" in markdown
    assert "层级分支缺失。" in markdown
    assert "quality:hierarchy_missing_branch_count=1" in markdown


def test_write_agent_stage_artifacts_writes_json_and_markdown(tmp_path: Path):
    output = AgentStageOutput(
        stage="rank_repairs",
        payload={
            "repair_plan": [
                {
                    "rank": 1,
                    "proposal_id": "proposal-1",
                    "priority": "high",
                    "reason": "收益高。",
                    "risk": "medium",
                    "human_checks": ["确认证据"],
                }
            ]
        },
    )

    paths = write_agent_stage_artifacts(tmp_path, output)

    assert paths["llm_repair_plan_json"] == tmp_path / "llm_repair_plan.json"
    assert paths["llm_repair_plan"] == tmp_path / "llm_repair_plan.md"
    assert json.loads(paths["llm_repair_plan_json"].read_text(encoding="utf-8"))[
        "repair_plan"
    ][0]["proposal_id"] == "proposal-1"
    assert "修复方案排序" in paths["llm_repair_plan"].read_text(encoding="utf-8")
```

- [ ] **Step 2: Run failing output tests**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_outputs.py -q
```

Expected:

- FAIL because `agent_outputs.py` does not exist.

- [ ] **Step 3: Implement `agent_outputs.py`**

Create `D:\LightRAG\lightrag\kb_iteration\agent_outputs.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm_review import parse_llm_review_output
from .models import ImprovementProposal


@dataclass(frozen=True)
class AgentStageOutput:
    stage: str
    payload: dict[str, Any]
    proposals: list[ImprovementProposal] = field(default_factory=list)


STAGE_ARTIFACTS = {
    "explain": ("llm_issue_analysis.json", "llm_issue_analysis.md", "llm_issue_analysis"),
    "infer_branches": (
        "llm_missing_branch_inference.json",
        "llm_missing_branch_inference.md",
        "llm_missing_branch_inference",
    ),
    "locate_evidence": ("llm_evidence_map.json", "llm_evidence_map.md", "llm_evidence_map"),
    "rank_repairs": ("llm_repair_plan.json", "llm_repair_plan.md", "llm_repair_plan"),
    "judge": ("llm_judge_report.json", "llm_judge_report.md", "llm_judge_report"),
}


def parse_agent_stage_output(stage: str, raw_text: str) -> AgentStageOutput:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{stage} output must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{stage} output must be a JSON object")
    if stage == "propose":
        review_output = parse_llm_review_output(
            json.dumps({"proposals": payload.get("proposals", [])}, ensure_ascii=False)
        )
        for proposal in review_output.proposals:
            if proposal.type != "review_context_request" and not proposal.evidence:
                raise ValueError("mutation proposal evidence must not be empty")
        return AgentStageOutput(stage=stage, payload=payload, proposals=review_output.proposals)
    return AgentStageOutput(stage=stage, payload=payload)


def write_agent_stage_artifacts(
    output_dir: str | Path, output: AgentStageOutput
) -> dict[str, Path]:
    if output.stage not in STAGE_ARTIFACTS:
        return {}
    json_name, markdown_name, key = STAGE_ARTIFACTS[output.stage]
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / json_name
    markdown_path = target_dir / markdown_name
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(output.payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    markdown_path.write_text(stage_output_to_markdown(output), encoding="utf-8")
    return {f"{key}_json": json_path, key: markdown_path}


def stage_output_to_markdown(output: AgentStageOutput) -> str:
    if output.stage == "explain":
        return _render_issue_analysis(output.payload)
    if output.stage == "infer_branches":
        return _render_missing_branch_inference(output.payload)
    if output.stage == "locate_evidence":
        return _render_evidence_map(output.payload)
    if output.stage == "rank_repairs":
        return _render_repair_plan(output.payload)
    if output.stage == "judge":
        return _render_judge_report(output.payload)
    return "# LLM Agent Stage\n\n" + json.dumps(output.payload, ensure_ascii=False, indent=2) + "\n"


def _render_issue_analysis(payload: dict[str, Any]) -> str:
    lines = ["# 问题解释", ""]
    issues = _dict_items(payload.get("issue_explanations"))
    if not issues:
        return "# 问题解释\n\n- 暂无问题解释。\n"
    for issue in issues:
        lines.append(f"## {issue.get('id', 'issue')}")
        lines.append(f"- 类别: {issue.get('category', '')}")
        lines.append(f"- 解释: {issue.get('explanation', '')}")
        lines.append(f"- 影响: {issue.get('impact', '')}")
        lines.extend(_render_list("证据", issue.get("evidence_refs")))
        lines.append("")
    return "\n".join(lines)


def _render_missing_branch_inference(payload: dict[str, Any]) -> str:
    lines = ["# 缺失分支推断", ""]
    for key in ("required", "present", "missing"):
        branches = _dict_items(payload.get(key))
        lines.append(f"## {key}")
        if not branches:
            lines.append("- none")
        for branch in branches:
            lines.append(f"- {branch.get('key', '')}: {branch.get('label', '')}")
        lines.append("")
    return "\n".join(lines)


def _render_evidence_map(payload: dict[str, Any]) -> str:
    lines = ["# 证据定位", ""]
    for item in _dict_items(payload.get("evidence_map")):
        lines.append(f"## {item.get('issue_id', 'issue')}")
        lines.append(f"- 目标: {item.get('target', '')}")
        lines.append(f"- 置信度: {item.get('confidence', '')}")
        lines.extend(_render_list("缺失证据", item.get("missing_evidence")))
        for supporting in _dict_items(item.get("supporting_items")):
            lines.append(
                "- "
                f"{supporting.get('item_type', '')}:{supporting.get('item_id', '')} "
                f"source_id={supporting.get('source_id', '')} "
                f"file_path={supporting.get('file_path', '')} "
                f"status={supporting.get('evidence_status', '')}"
            )
        lines.append("")
    if len(lines) == 2:
        lines.append("- 暂无证据定位。")
    return "\n".join(lines)


def _render_repair_plan(payload: dict[str, Any]) -> str:
    lines = ["# 修复方案排序", ""]
    for item in _dict_items(payload.get("repair_plan")):
        lines.append(f"## {item.get('rank', '')}. {item.get('proposal_id', '')}")
        lines.append(f"- 优先级: {item.get('priority', '')}")
        lines.append(f"- 风险: {item.get('risk', '')}")
        lines.append(f"- 原因: {item.get('reason', '')}")
        lines.extend(_render_list("人工检查", item.get("human_checks")))
        lines.append("")
    if len(lines) == 2:
        lines.append("- 暂无修复方案。")
    return "\n".join(lines)


def _render_judge_report(payload: dict[str, Any]) -> str:
    lines = ["# Judge 评判", ""]
    results = _dict_items(payload.get("judge_results"))
    if not results and payload:
        results = [payload]
    for result in results:
        lines.append(f"## {result.get('proposal_id', 'proposal')}")
        lines.append(f"- 决策: {result.get('decision', '')}")
        lines.append(f"- 原因: {result.get('reason', '')}")
        lines.append(f"- 风险覆盖: {result.get('risk_override', '')}")
        lines.extend(_render_list("人工检查", result.get("required_human_checks")))
        lines.append("")
    if len(lines) == 2:
        lines.append("- 暂无 Judge 评判。")
    return "\n".join(lines)


def _render_list(title: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        return [f"- {title}: none"]
    return [f"- {title}: " + ", ".join(str(item) for item in value)]


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
```

- [ ] **Step 4: Run output tests and commit**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_outputs.py -q
```

Expected:

- PASS.

Commit:

```powershell
cd D:\LightRAG
git add lightrag/kb_iteration/agent_outputs.py tests/kg/test_kb_iteration_agent_outputs.py
git commit -m "feat: parse kg agent stage outputs"
```

## Task 4: Multistage Agent Pipeline

**Files:**

- Create: `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`
- Create: `D:\LightRAG\tests\kg\test_kb_iteration_agent_pipeline.py`
- Create: prompt files under `D:\LightRAG\lightrag\kb_iteration\prompts\`
- Modify: `D:\LightRAG\lightrag\kb_iteration\review_loop.py`
- Modify: `D:\LightRAG\tests\kg\test_kb_iteration_review_loop.py`

- [ ] **Step 1: Write failing pipeline tests**

Create `D:\LightRAG\tests\kg\test_kb_iteration_agent_pipeline.py`:

```python
import json
from pathlib import Path

from lightrag.kb_iteration.agent_pipeline import (
    LLMAgentPipelineConfig,
    run_llm_agent_pipeline,
)


class SequencedAgentClient:
    model = "agent-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []
        self.outputs = [
            {
                "issue_explanations": [
                    {
                        "id": "issue-1",
                        "category": "hierarchy_completeness",
                        "explanation": "缺少症状分支。",
                        "impact": "影响导航。",
                        "evidence_refs": ["branch:symptom"],
                    }
                ]
            },
            {
                "required": [{"key": "symptom", "label": "症状"}],
                "present": [],
                "missing": [{"key": "symptom", "label": "症状"}],
                "missing_branches": [{"key": "symptom", "label": "症状"}],
            },
            {
                "evidence_map": [
                    {
                        "issue_id": "issue-1",
                        "target": "symptom",
                        "supporting_items": [
                            {
                                "item_type": "entity",
                                "item_id": "fever",
                                "source_id": "chunk-1",
                                "file_path": "guide.md",
                                "evidence_status": "grounded",
                            }
                        ],
                        "missing_evidence": [],
                        "confidence": 0.8,
                    }
                ]
            },
            {
                "proposals": [
                    {
                        "id": "proposal-hierarchy-symptom-001",
                        "type": "hierarchy_rule_change",
                        "target": "lightrag/medical_kg/ontology.py",
                        "proposed_change": "Add symptom as an expected branch.",
                        "reason": "The quality report shows the symptom branch is missing.",
                        "evidence": ["branch:symptom", "entity:fever"],
                        "confidence": 0.8,
                        "risk": "medium",
                        "requires_approval": True,
                        "expected_metric_change": {"hierarchy_completeness": 10},
                    }
                ]
            },
            {
                "repair_plan": [
                    {
                        "rank": 1,
                        "proposal_id": "proposal-hierarchy-symptom-001",
                        "priority": "high",
                        "reason": "Evidence is grounded and metric impact is direct.",
                        "risk": "medium",
                        "human_checks": ["确认症状分支命名"],
                    }
                ]
            },
            {
                "judge_results": [
                    {
                        "proposal_id": "proposal-hierarchy-symptom-001",
                        "decision": "needs_human",
                        "reason": "Hierarchy rule changes require maintainer approval.",
                        "risk_override": "medium",
                        "required_human_checks": ["确认源文档支持"],
                    }
                ]
            },
        ]

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return json.dumps(self.outputs.pop(0), ensure_ascii=False)


def test_run_llm_agent_pipeline_writes_stage_artifacts_and_queues_proposal(
    tmp_path: Path,
):
    package = _write_pipeline_package(tmp_path)
    client = SequencedAgentClient()

    result = run_llm_agent_pipeline(
        workspace="influenza_medical_v1",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(),
        profile="clinical_guideline_zh",
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-hierarchy-symptom-001"]
    assert len(client.calls) == 6
    assert (package / "llm_issue_analysis.md").exists()
    assert (package / "llm_missing_branch_inference.md").exists()
    assert (package / "llm_evidence_map.md").exists()
    assert (package / "llm_repair_plan.md").exists()
    assert (package / "llm_judge_report.md").exists()
    assert "proposal-hierarchy-symptom-001" in (
        package / "approval_queue.md"
    ).read_text(encoding="utf-8")
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    assert trace["mode"] == "agent_pipeline"
    assert [stage["stage"] for stage in trace["stages"]] == [
        "explain",
        "infer_branches",
        "locate_evidence",
        "propose",
        "rank_repairs",
        "judge",
    ]


class NoEvidenceAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.outputs[2] = {
            "evidence_map": [
                {
                    "issue_id": "issue-1",
                    "target": "symptom",
                    "supporting_items": [],
                    "missing_evidence": ["source_id/file_path unavailable"],
                    "confidence": 0.2,
                }
            ]
        }
        self.outputs[3] = {"proposals": []}


def test_run_llm_agent_pipeline_preserves_analysis_when_no_proposals(
    tmp_path: Path,
):
    package = _write_pipeline_package(tmp_path)

    result = run_llm_agent_pipeline(
        workspace="influenza_medical_v1",
        package_dir=package,
        client=NoEvidenceAgentClient(),
        config=LLMAgentPipelineConfig(),
        profile="clinical_guideline_zh",
    )

    assert result.stop_reason == "needs_more_evidence"
    assert result.proposal_ids == []
    assert (package / "llm_issue_analysis.md").exists()
    assert (package / "llm_evidence_map.md").exists()
    assert "proposals: []" in (package / "proposals.generated.yaml").read_text(
        encoding="utf-8"
    )


def _write_pipeline_package(tmp_path: Path) -> Path:
    package = tmp_path / "package"
    snapshot_dir = package / "snapshots"
    snapshot_dir.mkdir(parents=True)
    _write_json(
        snapshot_dir / "kg_snapshot.json",
        {
            "workspace": "influenza_medical_v1",
            "generated_at": "2026-06-19T00:00:00Z",
            "source_files": ["guide.md"],
            "nodes": [
                {
                    "id": "fever",
                    "label": "发热",
                    "entity_type": "Symptom",
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                }
            ],
            "edges": [],
            "metadata": {"profile": "clinical_guideline_zh"},
        },
    )
    _write_json(
        snapshot_dir / "quality_score.json",
        {
            "overall": 88,
            "metrics": {"hierarchy_missing_branch_count": 1},
            "findings": [
                {
                    "severity": "medium",
                    "category": "hierarchy_completeness",
                    "message": "Medical hierarchy is missing expected branches.",
                    "evidence": ["branch:symptom"],
                    "suggested_fix_type": "add_hierarchy_branch",
                    "requires_approval": True,
                }
            ],
            "details": {
                "hierarchy_branches": {
                    "required": [{"key": "symptom", "label": "症状"}],
                    "present": [],
                    "missing": [{"key": "symptom", "label": "症状"}],
                }
            },
        },
    )
    (package / "kb_context.md").write_text("# KB Context\n", encoding="utf-8")
    (package / "accepted_changes.md").write_text("# Accepted\n", encoding="utf-8")
    (package / "rejected_changes.md").write_text("# Rejected\n", encoding="utf-8")
    return package


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
```

- [ ] **Step 2: Run failing pipeline tests**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_pipeline.py -q
```

Expected:

- FAIL because `agent_pipeline.py` does not exist.

- [ ] **Step 3: Add prompt files**

Create these prompt files with exact content:

`D:\LightRAG\lightrag\kb_iteration\prompts\explain_zh.md`:

```markdown
你是知识库迭代 Agent 的问题解释阶段。
只输出 JSON object。
不要输出隐藏推理链。
根据 quality findings、metrics、kb_context 和历史 accepted/rejected 记忆，用中文解释问题、影响、证据引用和是否需要更多证据。
输出字段：
{
  "issue_explanations": [
    {
      "id": "",
      "category": "",
      "severity": "",
      "explanation": "",
      "impact": "",
      "evidence_refs": [],
      "needs_more_evidence": false
    }
  ]
}
```

`D:\LightRAG\lightrag\kb_iteration\prompts\infer_branches_zh.md`:

```markdown
你是知识库迭代 Agent 的缺失分支推断阶段。
只输出 JSON object。
必须基于输入里的 hierarchy_branches.required、present、missing。
不要发明 ontology 中不存在的 required branch。
输出字段：
{
  "required": [],
  "present": [],
  "missing": [],
  "missing_branches": [
    {
      "key": "",
      "label": "",
      "reason": "",
      "risk": "low|medium|high",
      "needs_more_evidence": false
    }
  ]
}
```

`D:\LightRAG\lightrag\kb_iteration\prompts\locate_evidence_zh.md`:

```markdown
你是知识库迭代 Agent 的证据定位阶段。
只输出 JSON object。
只能整理输入中已有 entity、relation、source_id、file_path 和 quality evidence。
source_id 或 file_path 缺失时标记 evidence_status="missing"。
不能把 LLM 判断当作医学证据。
输出字段：
{
  "evidence_map": [
    {
      "issue_id": "",
      "target": "",
      "supporting_items": [],
      "missing_evidence": [],
      "confidence": 0.0
    }
  ]
}
```

`D:\LightRAG\lightrag\kb_iteration\prompts\propose_zh.md`:

```markdown
你是知识库迭代 Agent 的 proposal 生成阶段。
只输出 JSON object。
proposal 必须满足 ImprovementProposal schema。
mutation proposal 必须 requires_approval=true。
证据为空时不要生成 mutation proposal，可以生成 review_context_request。
允许类型：hierarchy_rule_change, ontology_rule_change, relation_rule_change, source_evidence_repair, quality_report_note, review_context_request。
输出字段：
{
  "proposals": []
}
```

`D:\LightRAG\lightrag\kb_iteration\prompts\rank_repairs_zh.md`:

```markdown
你是知识库迭代 Agent 的修复方案排序阶段。
只输出 JSON object。
按质量收益、医学风险、证据充分度、人工成本、是否影响 rebuild、是否重复 rejected history 排序。
输出字段：
{
  "repair_plan": [
    {
      "rank": 1,
      "proposal_id": "",
      "priority": "low|medium|high",
      "reason": "",
      "risk": "low|medium|high",
      "human_checks": []
    }
  ]
}
```

- [ ] **Step 4: Implement `agent_pipeline.py`**

Create `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lightrag.utils import validate_workspace

from .agent_context import build_agent_observation, build_stage_context, write_agent_context
from .agent_outputs import parse_agent_stage_output, write_agent_stage_artifacts
from .llm_review import LLMReviewClient, LLMReviewOutput, write_llm_review_artifacts
from .proposals import write_approval_queue, write_improvement_backlog


@dataclass(frozen=True)
class LLMAgentPipelineConfig:
    max_context_tokens_per_stage: int = 12000
    max_stage_retries: int = 1
    allow_llm_judge: bool = True
    generate_patch_candidates: bool = False
    require_human_for_mutation: bool = True


@dataclass(frozen=True)
class LLMAgentPipelineResult:
    output_dir: Path
    stop_reason: str
    proposal_ids: list[str] = field(default_factory=list)
    artifact_paths: dict[str, Path] = field(default_factory=dict)


STAGES = [
    "explain",
    "infer_branches",
    "locate_evidence",
    "propose",
    "rank_repairs",
    "judge",
]


def run_llm_agent_pipeline(
    *,
    workspace: str,
    package_dir: str | Path,
    client: LLMReviewClient,
    config: LLMAgentPipelineConfig | None = None,
    profile: str | None = None,
) -> LLMAgentPipelineResult:
    validated_workspace = validate_workspace(workspace)
    output_dir = Path(package_dir)
    pipeline_config = config or LLMAgentPipelineConfig()
    _validate_required_artifacts(output_dir)
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
    previous_outputs: dict[str, Any] = {}
    proposals = []

    for stage in STAGES:
        if stage == "judge" and not pipeline_config.allow_llm_judge:
            continue
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
        user_prompt = json.dumps(context, ensure_ascii=False, sort_keys=True)
        input_token_estimate = _estimate_tokens(user_prompt)
        stage_trace = {
            "stage": stage,
            "state": "running",
            "model": _client_model(client),
            "context_files": [_relative_artifact_path(output_dir, context_path)],
            "input_token_estimate": input_token_estimate,
            "output_token_estimate": 0,
            "artifact_keys": [],
            "proposal_ids": [],
            "error": "",
        }
        if input_token_estimate > pipeline_config.max_context_tokens_per_stage:
            stage_trace["state"] = "context_too_large"
            stage_trace["error"] = "input_token_estimate exceeds max_context_tokens_per_stage"
            trace["stages"].append(stage_trace)
            return _finish_pipeline(output_dir, "context_too_large", [], artifact_paths, trace)
        try:
            raw_output = client.complete(
                system_prompt=_stage_prompt(stage, profile),
                user_prompt=user_prompt,
            )
            parsed = parse_agent_stage_output(stage, raw_output)
        except Exception as exc:
            stage_trace["state"] = "failed"
            stage_trace["error"] = str(exc)
            trace["stages"].append(stage_trace)
            return _finish_pipeline(output_dir, "invalid_llm_output", [], artifact_paths, trace)

        stage_trace["state"] = "completed"
        stage_trace["output_token_estimate"] = _estimate_tokens(raw_output)
        written = write_agent_stage_artifacts(output_dir, parsed)
        artifact_paths.update(written)
        stage_trace["artifact_keys"] = list(written)
        previous_outputs.update(parsed.payload)
        if stage == "propose":
            proposals = parsed.proposals
            stage_trace["proposal_ids"] = [proposal.id for proposal in proposals]
            review_output = LLMReviewOutput(
                confirmed_issues=previous_outputs.get("issue_explanations", []),
                hypotheses=[],
                missing_evidence=previous_outputs.get("missing_evidence", []),
                out_of_scope=[],
                proposals=proposals,
            )
            artifact_paths.update(write_llm_review_artifacts(review_output, output_dir))
            artifact_paths["approval_queue"] = write_approval_queue(proposals, output_dir)
            artifact_paths["improvement_backlog"] = write_improvement_backlog(proposals, output_dir)
        trace["stages"].append(stage_trace)

    proposal_ids = [proposal.id for proposal in proposals]
    trace["proposal_ids"] = proposal_ids
    if proposal_ids:
        return _finish_pipeline(output_dir, "pending_human_review", proposal_ids, artifact_paths, trace)
    _write_empty_proposal_artifacts(output_dir, artifact_paths)
    return _finish_pipeline(output_dir, "needs_more_evidence", [], artifact_paths, trace)


def _write_empty_proposal_artifacts(output_dir: Path, artifact_paths: dict[str, Path]) -> None:
    proposal_path = output_dir / "proposals.generated.yaml"
    approval_path = output_dir / "approval_queue.md"
    backlog_path = output_dir / "improvement_backlog.md"
    report_path = output_dir / "llm_review_report.md"
    proposal_path.write_text("# Generated Proposals\n\nproposals: []\n", encoding="utf-8")
    approval_path.write_text("# Approval Queue\n\nproposals: []\n", encoding="utf-8")
    backlog_path.write_text("# Improvement Backlog\n\n- needs_more_evidence\n", encoding="utf-8")
    report_path.write_text("# LLM Review Report\n\n- No valid proposal generated.\n", encoding="utf-8")
    artifact_paths.update(
        {
            "proposals_generated": proposal_path,
            "approval_queue": approval_path,
            "improvement_backlog": backlog_path,
            "llm_review_report": report_path,
        }
    )


def _finish_pipeline(
    output_dir: Path,
    stop_reason: str,
    proposal_ids: list[str],
    artifact_paths: dict[str, Path],
    trace: dict[str, Any],
) -> LLMAgentPipelineResult:
    trace["completed_at"] = _utc_timestamp()
    trace["stop_reason"] = stop_reason
    trace["proposal_ids"] = proposal_ids
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


def _stage_prompt(stage: str, profile: str | None) -> str:
    prompt_path = Path(__file__).parent / "prompts" / f"{stage}_zh.md"
    if stage == "judge":
        prompt_path = Path(__file__).parent / "prompts" / "judge_zh.md"
    prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else stage
    return f"profile: {profile or 'default'}\n{prompt}"


def _validate_required_artifacts(output_dir: Path) -> None:
    for relative in ("snapshots/kg_snapshot.json", "snapshots/quality_score.json"):
        path = output_dir / relative
        if not path.exists():
            raise ValueError(f"KB iteration package is missing required artifact: {relative}")


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _client_model(client: LLMReviewClient) -> str:
    model = getattr(client, "model", "")
    return model if isinstance(model, str) and model.strip() else "unknown"


def _relative_artifact_path(output_dir: Path, artifact_path: Path) -> str:
    return artifact_path.relative_to(output_dir).as_posix()
```

- [ ] **Step 5: Run pipeline tests**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_pipeline.py -q
```

Expected:

- PASS.

- [ ] **Step 6: Keep old loop tests passing**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_review_loop.py -q
```

Expected:

- PASS. Do not change old loop behavior in this task unless a test reveals a direct compatibility break.

- [ ] **Step 7: Commit pipeline**

Commit:

```powershell
cd D:\LightRAG
git add lightrag/kb_iteration/agent_pipeline.py lightrag/kb_iteration/prompts/explain_zh.md lightrag/kb_iteration/prompts/infer_branches_zh.md lightrag/kb_iteration/prompts/locate_evidence_zh.md lightrag/kb_iteration/prompts/propose_zh.md lightrag/kb_iteration/prompts/rank_repairs_zh.md tests/kg/test_kb_iteration_agent_pipeline.py
git commit -m "feat: run multistage kg llm agent pipeline"
```

## Task 5: API Integration And Artifact Keys

**Files:**

- Modify: `D:\LightRAG\lightrag\api\routers\kb_iteration_routes.py`
- Modify: `D:\LightRAG\tests\api\routes\test_kb_iteration_routes.py`
- Modify: `D:\LightRAG\lightrag_webui\src\api\lightrag.ts`

- [ ] **Step 1: Add failing API artifact and dispatch tests**

Append to `D:\LightRAG\tests\api\routes\test_kb_iteration_routes.py`:

```python
def test_llm_review_agent_artifact_routes(tmp_path: Path, monkeypatch):
    client, fixture = _client(tmp_path, monkeypatch)
    _write_text(fixture.package / "llm_issue_analysis.md", "# 问题解释\n")
    _write_text(fixture.package / "llm_missing_branch_inference.md", "# 缺失分支推断\n")
    _write_text(fixture.package / "llm_evidence_map.md", "# 证据定位\n")
    _write_text(fixture.package / "llm_repair_plan.md", "# 修复方案排序\n")

    for key, marker in [
        ("llm_issue_analysis", "问题解释"),
        ("llm_missing_branch_inference", "缺失分支推断"),
        ("llm_evidence_map", "证据定位"),
        ("llm_repair_plan", "修复方案排序"),
    ]:
        response = client.get(
            f"/kb-iteration/influenza_medical_v1/runs/latest/artifacts/{key}",
            headers=HEADERS,
        )
        assert response.status_code == 200
        assert marker in response.json()["content"]


def test_llm_review_run_defaults_to_agent_pipeline(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    calls = []

    class FakeClient:
        model = "fake"

        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return "{}"

    def fake_run_llm_agent_pipeline(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            output_dir=fixture.package,
            stop_reason="pending_human_review",
            proposal_ids=["proposal-1"],
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
    assert response.json()["stopReason"] == "pending_human_review"
    assert calls[0]["workspace"] == "influenza_medical_v1"
    assert calls[0]["profile"] == "clinical_guideline_zh"


def test_llm_review_run_can_use_legacy_loop_mode(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    calls = []

    class FakeClient:
        model = "fake"

        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return "{}"

    def fake_run_llm_review_loop(**kwargs):
        calls.append(kwargs)
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
        json={"mode": "loop", "profile": "clinical_guideline_zh"},
    )

    assert response.status_code == 200
    assert response.json()["stopReason"] == "all_proposals_invalid"
    assert calls[0]["workspace"] == "influenza_medical_v1"
```

- [ ] **Step 2: Run failing API tests**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/api/routes/test_kb_iteration_routes.py -q
```

Expected:

- FAIL because new artifacts and `agent_pipeline` mode are not wired.

- [ ] **Step 3: Wire backend API**

In `D:\LightRAG\lightrag\api\routers\kb_iteration_routes.py`, add imports:

```python
from lightrag.kb_iteration.agent_pipeline import (
    LLMAgentPipelineConfig,
    run_llm_agent_pipeline,
)
```

Extend `ARTIFACTS`:

```python
    "llm_issue_analysis": ("llm_issue_analysis.md", "text/markdown"),
    "llm_missing_branch_inference": (
        "llm_missing_branch_inference.md",
        "text/markdown",
    ),
    "llm_evidence_map": ("llm_evidence_map.md", "text/markdown"),
    "llm_repair_plan": ("llm_repair_plan.md", "text/markdown"),
```

Change request mode:

```python
class RunLLMReviewRequest(BaseModel):
    profile: Optional[str] = Field(default=None, max_length=200)
    mode: Literal["agent_pipeline", "loop"] = "agent_pipeline"
    max_review_rounds: int = Field(default=4, ge=1, le=10)
    max_focus_items_per_round: int = Field(default=3, ge=1, le=10)
    max_context_tokens_per_round: int = Field(default=12000, ge=1000, le=100000)
    max_stage_retries: int = Field(default=1, ge=0, le=3)
    allow_llm_judge: bool = True
    allow_llm_auto_accept: bool = False
    allow_low_risk_auto_reject: bool = True
    generate_patch_candidates: bool = False
    require_human_for_mutation: bool = True
```

In `create_llm_review_run`, replace the single `run_llm_review_loop` call with:

```python
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
```

- [ ] **Step 4: Update frontend API type**

In `D:\LightRAG\lightrag_webui\src\api\lightrag.ts`, change:

```ts
export type KBIterationLLMReviewRunRequest = {
  profile?: string | null
  mode?: 'agent_pipeline' | 'loop'
  max_review_rounds?: number
  max_focus_items_per_round?: number
  max_context_tokens_per_round?: number
  max_stage_retries?: number
  allow_llm_judge?: boolean
  allow_llm_auto_accept?: boolean
  allow_low_risk_auto_reject?: boolean
  generate_patch_candidates?: boolean
  require_human_for_mutation?: boolean
}
```

- [ ] **Step 5: Run API tests and commit**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/api/routes/test_kb_iteration_routes.py -q
```

Expected:

- PASS.

Commit:

```powershell
cd D:\LightRAG
git add lightrag/api/routers/kb_iteration_routes.py tests/api/routes/test_kb_iteration_routes.py lightrag_webui/src/api/lightrag.ts
git commit -m "feat: expose multistage kg llm agent api"
```

## Task 6: Frontend Data Loading For Agent Artifacts

**Files:**

- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\kgIterationLoadUtils.ts`
- Modify: `D:\LightRAG\lightrag_webui\src\features\KGMaintenanceConsole.tsx`
- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.test.tsx`

- [ ] **Step 1: Add failing load bundle test**

If `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.test.tsx` already has a load utility section, add this there. If not, create `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\kgIterationLoadUtils.test.ts`:

```ts
import { describe, expect, test } from 'bun:test'
import { loadKGMaintenanceWorkspaceBundle } from './kgIterationLoadUtils'

describe('KG iteration LLM agent artifact loading', () => {
  test('loads multistage LLM agent artifacts as optional markdown', async () => {
    const requestedKeys: string[] = []
    const bundle = await loadKGMaintenanceWorkspaceBundle('influenza_medical_v1', {
      getSummary: async () => ({
        workspace: 'influenza_medical_v1',
        latestRunId: 'latest',
        phase: 'pending_user_review',
        counts: { nodes: 1, edges: 1, sources: 1 },
        quality: {},
        pendingApprovalCount: 0,
        highRiskFindingCount: 0,
        artifacts: []
      }),
      getQuality: async () => ({
        workspace: 'influenza_medical_v1',
        runId: 'latest',
        quality: {},
        report: ''
      }),
      getRules: async () => ({
        workspace: 'influenza_medical_v1',
        qualityRules: '',
        knownIssues: '',
        acceptedChanges: '',
        rejectedChanges: ''
      }),
      getArtifact: async (_workspace, key) => {
        requestedKeys.push(key)
        return { artifactKey: key, contentType: 'text/markdown', content: `# ${key}` }
      },
      getTrace: async () => ({ artifactKey: 'llm_review_trace', contentType: 'application/json', payload: {} }),
      getReport: async () => ({ artifactKey: 'llm_review_report', contentType: 'text/markdown', content: '# Review' }),
      getProposals: async () => ({ artifactKey: 'proposals_generated', contentType: 'text/markdown', content: 'proposals: []' }),
      getJudgeReport: async () => ({ artifactKey: 'llm_judge_report', contentType: 'text/markdown', content: '# Judge' })
    })

    expect(requestedKeys).toContain('llm_issue_analysis')
    expect(requestedKeys).toContain('llm_missing_branch_inference')
    expect(requestedKeys).toContain('llm_evidence_map')
    expect(requestedKeys).toContain('llm_repair_plan')
    expect(bundle.llmIssueAnalysisArtifact).toBe('# llm_issue_analysis')
    expect(bundle.llmRepairPlanArtifact).toBe('# llm_repair_plan')
  })
})
```

- [ ] **Step 2: Run failing frontend load test**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/kgIterationLoadUtils.test.ts
```

Expected:

- FAIL because the bundle does not include new LLM agent artifacts.

- [ ] **Step 3: Extend `kgIterationLoadUtils.ts` bundle**

In `KGMaintenanceWorkspaceBundle`, add:

```ts
  llmIssueAnalysisArtifact: string
  llmMissingBranchInferenceArtifact: string
  llmEvidenceMapArtifact: string
  llmRepairPlanArtifact: string
```

In the `Promise.all` destructure, add:

```ts
    llmIssueAnalysisArtifact,
    llmMissingBranchInferenceArtifact,
    llmEvidenceMapArtifact,
    llmRepairPlanArtifact,
```

In the requests array, add:

```ts
    optionalArtifactContent(() => loaders.getArtifact(requestWorkspace, 'llm_issue_analysis')),
    optionalArtifactContent(() =>
      loaders.getArtifact(requestWorkspace, 'llm_missing_branch_inference')
    ),
    optionalArtifactContent(() => loaders.getArtifact(requestWorkspace, 'llm_evidence_map')),
    optionalArtifactContent(() => loaders.getArtifact(requestWorkspace, 'llm_repair_plan'))
```

Return these fields from the bundle.

- [ ] **Step 4: Extend `KGMaintenanceConsole.tsx` state and props**

Add state:

```ts
  const [llmIssueAnalysis, setLlmIssueAnalysis] = useState('')
  const [llmMissingBranchInference, setLlmMissingBranchInference] = useState('')
  const [llmEvidenceMap, setLlmEvidenceMap] = useState('')
  const [llmRepairPlan, setLlmRepairPlan] = useState('')
```

Read from the loaded bundle:

```ts
        llmIssueAnalysisArtifact,
        llmMissingBranchInferenceArtifact,
        llmEvidenceMapArtifact,
        llmRepairPlanArtifact
```

Set state:

```ts
      setLlmIssueAnalysis(normalizeOptionalMarkdown(llmIssueAnalysisArtifact))
      setLlmMissingBranchInference(normalizeOptionalMarkdown(llmMissingBranchInferenceArtifact))
      setLlmEvidenceMap(normalizeOptionalMarkdown(llmEvidenceMapArtifact))
      setLlmRepairPlan(normalizeOptionalMarkdown(llmRepairPlanArtifact))
```

Pass into `MainPanel`:

```tsx
        llmIssueAnalysis={llmIssueAnalysis}
        llmMissingBranchInference={llmMissingBranchInference}
        llmEvidenceMap={llmEvidenceMap}
        llmRepairPlan={llmRepairPlan}
```

Extend `MainPanelProps` with the same four `string` fields.

- [ ] **Step 5: Run frontend load test and commit**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/kgIterationLoadUtils.test.ts
```

Expected:

- PASS.

Commit:

```powershell
cd D:\LightRAG
git add lightrag_webui/src/components/kg-maintenance/kgIterationLoadUtils.ts lightrag_webui/src/components/kg-maintenance/kgIterationLoadUtils.test.ts lightrag_webui/src/features/KGMaintenanceConsole.tsx
git commit -m "feat: load kg llm agent artifacts"
```

## Task 7: Frontend Multistage Agent View

**Files:**

- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx`
- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.test.tsx`
- Modify: `D:\LightRAG\lightrag_webui\src\features\KGMaintenanceConsole.tsx`

- [ ] **Step 1: Replace LLM panel tests with multistage assertions**

In `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.test.tsx`, add a test:

```tsx
test('renders multistage LLM agent materials', () => {
  const markup = renderToStaticMarkup(
    <LLMReviewPanel
      trace={{
        mode: 'agent_pipeline',
        stop_reason: 'pending_human_review',
        stages: [
          { stage: 'explain', state: 'completed', artifact_keys: ['llm_issue_analysis'] },
          { stage: 'infer_branches', state: 'completed', artifact_keys: ['llm_missing_branch_inference'] },
          { stage: 'locate_evidence', state: 'completed', artifact_keys: ['llm_evidence_map'] },
          { stage: 'propose', state: 'completed', proposal_ids: ['p1'] },
          { stage: 'rank_repairs', state: 'completed', artifact_keys: ['llm_repair_plan'] },
          { stage: 'judge', state: 'completed', artifact_keys: ['llm_judge_report'] }
        ]
      }}
      report="# LLM Review Report"
      proposals="id: p1"
      issueAnalysis="# 问题解释\n\n- 层级缺失"
      missingBranchInference="# 缺失分支推断\n\n- symptom"
      evidenceMap="# 证据定位\n\nsource_id=chunk-1"
      repairPlan="# 修复方案排序\n\nproposal p1"
      running={false}
      onRun={() => undefined}
    />
  )

  expect(markup).toContain('多阶段 LLM Agent')
  expect(markup).toContain('Explain')
  expect(markup).toContain('Infer')
  expect(markup).toContain('Evidence')
  expect(markup).toContain('Propose')
  expect(markup).toContain('Rank')
  expect(markup).toContain('Judge')
  expect(markup).toContain('# 问题解释')
  expect(markup).toContain('# 缺失分支推断')
  expect(markup).toContain('# 证据定位')
  expect(markup).toContain('# 修复方案排序')
  expect(markup).toContain('LLM Agent 只生成分析')
})
```

- [ ] **Step 2: Run failing LLM panel tests**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx
```

Expected:

- FAIL because `LLMReviewPanel` does not accept the four new artifact props and does not render stages.

- [ ] **Step 3: Extend `LLMReviewPanelProps`**

In `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx`, add:

```ts
export type TraceStage = {
  stage?: string
  state?: string
  artifact_keys?: string[]
  proposal_ids?: string[]
}
```

Extend props:

```ts
  issueAnalysis: string
  missingBranchInference: string
  evidenceMap: string
  repairPlan: string
```

In the component:

```ts
  const stages = Array.isArray(trace?.stages) ? (trace.stages as TraceStage[]) : []
```

- [ ] **Step 4: Render multistage view**

Add this block before the old rounds block:

```tsx
      <div className="border-border/70 bg-muted/20 rounded-lg border p-3 text-sm">
        LLM Agent 只生成分析、proposal、证据定位和修复排序。所有会改变 KG、规则、prompt、workspace 或 WebUI 行为的 proposal 都必须人工审批。
      </div>

      <div className="space-y-2">
        <h3 className="text-sm font-medium">多阶段 LLM Agent</h3>
        {stages.length ? (
          <div className="grid gap-2 md:grid-cols-3">
            {stages.map((stage, index) => (
              <article key={stage.stage || index} className="border-border/70 rounded-lg border p-3">
                <div className="text-sm font-medium">{stageLabel(stage.stage || '')}</div>
                <div className="text-muted-foreground mt-1 text-xs">{stage.state || 'unknown'}</div>
                <RoundField
                  label="proposal ID"
                  values={stage.proposal_ids}
                  emptyText="暂无 proposal ID。"
                />
              </article>
            ))}
          </div>
        ) : (
          <EmptyBlock>暂无多阶段 Agent trace。</EmptyBlock>
        )}
      </div>

      <ArtifactBlock title="问题解释" content={issueAnalysis} emptyText="暂无问题解释。" />
      <ArtifactBlock
        title="缺失分支推断"
        content={missingBranchInference}
        emptyText="暂无缺失分支推断。"
      />
      <ArtifactBlock title="证据定位" content={evidenceMap} emptyText="暂无证据定位。" />
      <ArtifactBlock title="修复方案排序" content={repairPlan} emptyText="暂无修复方案排序。" />
```

Add helper:

```ts
function stageLabel(stage: string) {
  const labels: Record<string, string> = {
    explain: 'Explain / 问题解释',
    infer_branches: 'Infer / 缺失分支推断',
    locate_evidence: 'Evidence / 证据定位',
    propose: 'Propose / proposal 生成',
    rank_repairs: 'Rank / 修复排序',
    judge: 'Judge / 评判'
  }
  return labels[stage] || stage || '未知阶段'
}
```

- [ ] **Step 5: Pass props from `KGMaintenanceConsole.tsx`**

In the `LLMReviewPanel` call, pass:

```tsx
          issueAnalysis={llmIssueAnalysis}
          missingBranchInference={llmMissingBranchInference}
          evidenceMap={llmEvidenceMap}
          repairPlan={llmRepairPlan}
```

In `handleRunLLMReview`, include mode:

```ts
          mode: 'agent_pipeline',
          max_stage_retries: 1,
```

- [ ] **Step 6: Run frontend tests and commit**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/kgIterationLoadUtils.test.ts
```

Expected:

- PASS.

Commit:

```powershell
cd D:\LightRAG
git add lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.test.tsx lightrag_webui/src/features/KGMaintenanceConsole.tsx
git commit -m "feat: show multistage kg llm agent review"
```

## Task 8: End-To-End Verification And Polish

**Files:**

- Modify only if verification finds concrete issues:
  - `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`
  - `D:\LightRAG\lightrag\kb_iteration\agent_outputs.py`
  - `D:\LightRAG\lightrag\api\routers\kb_iteration_routes.py`
  - `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx`

- [ ] **Step 1: Run focused backend tests**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_quality.py tests/kg/test_kb_iteration_review_context.py tests/kg/test_kb_iteration_agent_context.py tests/kg/test_kb_iteration_agent_outputs.py tests/kg/test_kb_iteration_agent_pipeline.py tests/kg/test_kb_iteration_review_loop.py tests/api/routes/test_kb_iteration_routes.py -q
```

Expected:

- PASS.

- [ ] **Step 2: Run focused frontend tests**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/kgIterationLoadUtils.test.ts src/components/kg-maintenance/KGMaintenanceShell.test.tsx
```

Expected:

- PASS.

- [ ] **Step 3: Run frontend lint and build**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun run lint
npx --yes bun run build
```

Expected:

- PASS.

- [ ] **Step 4: Run one real Agent review with the configured model**

Use the running server if available. If the API server is not running, start it using the user's existing LightRAG scripts rather than adding a new startup method.

From a PowerShell prompt with the API server running:

```powershell
$body = @{
  profile = "clinical_guideline_zh"
  mode = "agent_pipeline"
  allow_llm_judge = $true
  require_human_for_mutation = $true
  generate_patch_candidates = $false
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:9621/kb-iteration/influenza_medical_v1/llm-review/runs" -ContentType "application/json" -Body $body
```

Expected:

- Response includes `stopReason`.
- `D:\LightRAG\work\kb-iteration\influenza_medical_v1\llm_review_trace.json` has `"mode": "agent_pipeline"`.
- The following files exist after the run:
  - `llm_issue_analysis.md`
  - `llm_missing_branch_inference.md`
  - `llm_evidence_map.md`
  - `llm_repair_plan.md`
  - `llm_judge_report.md`
  - `proposals.generated.yaml`

- [ ] **Step 5: Browser review**

Open `http://127.0.0.1:5176/#/` or the current WebUI URL and verify:

- KG Maintenance `LLM 审阅材料` shows `多阶段 LLM Agent`.
- The page shows `问题解释`, `缺失分支推断`, `证据定位`, `修复方案排序`, `Judge`.
- It clearly states LLM does not automatically modify KG.
- If proposals are generated, they still appear in `Proposal 审批`.
- If proposals are not generated, the user can still inspect why via issue analysis and evidence map.

- [ ] **Step 6: Commit verification polish if needed**

If verification required edits:

```powershell
cd D:\LightRAG
git add lightrag/kb_iteration lightrag/api/routers/kb_iteration_routes.py tests/kg tests/api/routes/test_kb_iteration_routes.py lightrag_webui/src/api/lightrag.ts lightrag_webui/src/components/kg-maintenance lightrag_webui/src/features/KGMaintenanceConsole.tsx
git commit -m "fix: polish multistage kg llm agent pipeline"
```

If no files changed, do not create an empty commit.

## Completion Checklist

- [ ] `quality_score.json` includes concrete hierarchy branch details.
- [ ] `review_context` no longer gives hierarchy-only LLM reviews unrelated edge fallbacks.
- [ ] `agent_context.py`, `agent_outputs.py`, and `agent_pipeline.py` exist and have focused tests.
- [ ] LLM Agent stages write Markdown/JSON artifacts.
- [ ] Existing `llm-review` API defaults to `agent_pipeline` and can still run legacy `loop`.
- [ ] New artifact keys are readable through the artifact route.
- [ ] WebUI loads and displays issue analysis, missing branch inference, evidence map, repair plan, proposals, and judge report.
- [ ] Human approval remains required for mutation proposals.
- [ ] Backend focused tests pass.
- [ ] Frontend focused tests, lint, and build pass.
- [ ] Real DeepSeek-backed run produces stage trace or clear failure artifacts without exposing the API key.
