# KB Iteration Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first deterministic, file-based KB iteration agent for LightRAG workspaces.

**Architecture:** Add a small `lightrag.kb_iteration` package that reads workspace graph data into stable snapshots, writes layered Markdown memory, computes deterministic quality scores, prepares approval-gated proposals, and compares snapshot versions. The first implementation must work without an LLM call; LLM review remains an extension point that consumes the generated snapshot and Markdown files.

**Tech Stack:** Python 3.10+, dataclasses, pathlib, json, networkx GraphML parsing, pytest.

---

## Scope

This plan implements the deterministic core promised by `docs/superpowers/specs/2026-06-17-kb-iteration-agent-design.md` and `docs/superpowers/specs/2026-06-17-kb-iteration-agent-design-zh.md`.

It creates artifacts under:

```text
work/kb-iteration/<workspace>/
```

It does not mutate extraction prompts, ontology rules, hierarchy rules, KG storage, or workspace data. Any proposal that would change behavior or data is written to `approval_queue.md`.

## File Structure

- Create `lightrag/kb_iteration/__init__.py`
  - Public exports for snapshot, memory, quality, proposals, diff, and runner helpers.
- Create `lightrag/kb_iteration/models.py`
  - Dataclasses for `SnapshotNode`, `SnapshotEdge`, `KGSnapshot`, `QualityFinding`, `QualityScore`, `ImprovementProposal`, and `SnapshotDiff`.
- Create `lightrag/kb_iteration/snapshot.py`
  - GraphML and in-memory graph conversion.
  - Workspace source-file discovery.
  - Entity/relation/type statistics.
  - JSON snapshot writer.
- Create `lightrag/kb_iteration/markdown.py`
  - Deterministic Markdown writers for `kb_context.md`, `entity_catalog.md`, `relation_catalog.md`, `kg_structure.md`, rule memory files, and `iteration_log.md`.
- Create `lightrag/kb_iteration/quality.py`
  - Deterministic quality checks and `quality_score.json`.
  - Medical KG metrics for value-like nodes, generic relation labels, missing evidence, hierarchy coverage, and disease hub overload.
- Create `lightrag/kb_iteration/proposals.py`
  - Structured proposal schema validation.
  - Markdown rendering for `improvement_backlog.md` and `approval_queue.md`.
- Create `lightrag/kb_iteration/diff.py`
  - Snapshot-to-snapshot comparison and `diff_report.md`.
- Create `lightrag/kb_iteration/runner.py`
  - High-level `run_iteration(...)` orchestration that performs observe, deterministic think, propose, evaluate, and remember.
- Create `tests/kg/test_kb_iteration_snapshot.py`
- Create `tests/kg/test_kb_iteration_markdown.py`
- Create `tests/kg/test_kb_iteration_quality.py`
- Create `tests/kg/test_kb_iteration_proposals.py`
- Create `tests/kg/test_kb_iteration_diff.py`
- Create or update docs only in the final task after code behavior is verified.

## Fixture Conventions

Use small in-test fixtures rather than the live `data/rag_storage` workspace. Tests must not depend on the user's local PDFs or running API server.

Shared fixture graph concept:

```text
流行性感冒 -> 发热                         keywords=临床表现
发热 -> 全身症状                           keywords=症状归类
流行性感冒 -> 75 mg                        keywords=邻接
流行性感冒 -> 流感                         keywords=相关
流感病毒 -> 流行性感冒                     keywords=病原导致
流行性感冒 -> 奥司他韦                     keywords=推荐治疗
```

The fixture intentionally includes:

- a value-like node: `75 mg`
- a generic relation: `邻接`
- a synonym duplicate candidate: `流感` and `流行性感冒`
- source metadata on some records
- missing source metadata on at least one edge

## Task 1: Snapshot Models And GraphML Snapshotter

**Files:**
- Create: `lightrag/kb_iteration/__init__.py`
- Create: `lightrag/kb_iteration/models.py`
- Create: `lightrag/kb_iteration/snapshot.py`
- Test: `tests/kg/test_kb_iteration_snapshot.py`

- [ ] **Step 1: Write the failing snapshot tests**

Add tests that build a `networkx.MultiDiGraph`, write it to GraphML, load it through the new snapshotter, and assert stable node/edge fields plus stats.

```python
from pathlib import Path

import networkx as nx

from lightrag.kb_iteration.snapshot import (
    build_snapshot_from_graphml,
    write_snapshot_artifacts,
)


def test_build_snapshot_from_graphml_preserves_medical_fields(tmp_path: Path):
    graph = nx.MultiDiGraph()
    graph.add_node(
        "流行性感冒",
        entity_type="Disease",
        description="急性呼吸道传染病",
        source_id="chunk-1",
        file_path="guideline.md",
    )
    graph.add_node(
        "发热",
        entity_type="Symptom",
        description="常见临床表现",
        source_id="chunk-1",
        file_path="guideline.md",
    )
    graph.add_edge(
        "流行性感冒",
        "发热",
        keywords="临床表现",
        description="流行性感冒可表现为发热",
        source_id="chunk-1",
        file_path="guideline.md",
        weight=1.0,
    )
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)

    snapshot = build_snapshot_from_graphml(
        graph_path,
        workspace="influenza_medical_v1",
        source_files=["guideline.md"],
        profile="clinical_guideline_zh",
    )

    assert snapshot.workspace == "influenza_medical_v1"
    assert snapshot.metadata["profile"] == "clinical_guideline_zh"
    assert snapshot.metadata["graph_node_count"] == 2
    assert snapshot.metadata["graph_edge_count"] == 1
    assert snapshot.nodes[0].id == "流行性感冒"
    assert snapshot.nodes[0].entity_type == "Disease"
    assert snapshot.edges[0].source == "流行性感冒"
    assert snapshot.edges[0].target == "发热"
    assert snapshot.edges[0].keywords == "临床表现"


def test_write_snapshot_artifacts_creates_machine_readable_stats(tmp_path: Path):
    graph = nx.MultiDiGraph()
    graph.add_node("流行性感冒", entity_type="Disease", source_id="chunk-1", file_path="a.md")
    graph.add_node("发热", entity_type="Symptom", source_id="chunk-1", file_path="a.md")
    graph.add_edge("流行性感冒", "发热", keywords="临床表现", source_id="chunk-1", file_path="a.md")
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)
    snapshot = build_snapshot_from_graphml(graph_path, workspace="demo")

    output_dir = tmp_path / "work" / "kb-iteration" / "demo"
    written = write_snapshot_artifacts(snapshot, output_dir)

    assert (output_dir / "snapshots" / "kg_snapshot.json").exists()
    assert (output_dir / "snapshots" / "entity_stats.json").exists()
    assert (output_dir / "snapshots" / "relation_stats.json").exists()
    assert written["kg_snapshot"].name == "kg_snapshot.json"
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_snapshot.py -q
```

Expected: import failure because `lightrag.kb_iteration.snapshot` does not exist.

- [ ] **Step 3: Implement snapshot models**

Create dataclasses with stable serialization.

```python
@dataclass(frozen=True)
class SnapshotNode:
    id: str
    label: str
    entity_type: str
    description: str = ""
    source_id: str = ""
    file_path: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SnapshotEdge:
    id: str
    source: str
    target: str
    keywords: str
    description: str = ""
    source_id: str = ""
    file_path: str = ""
    weight: float | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KGSnapshot:
    workspace: str
    generated_at: str
    source_files: list[str]
    nodes: list[SnapshotNode]
    edges: list[SnapshotEdge]
    metadata: dict[str, Any]
```

- [ ] **Step 4: Implement GraphML loading and artifact writing**

Implementation requirements:

- Use `networkx.read_graphml`.
- Preserve direction by using `source` and `target`.
- Normalize missing attributes to empty strings.
- Convert `weight` to `float` when possible.
- Sort stats deterministically by count descending then label ascending.
- Write JSON with `ensure_ascii=False` and `indent=2`.

- [ ] **Step 5: Run GREEN verification**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_snapshot.py -q
```

Expected: 2 passed.

- [ ] **Step 6: Commit Task 1**

Stage only Task 1 files:

```powershell
git add lightrag/kb_iteration/__init__.py lightrag/kb_iteration/models.py lightrag/kb_iteration/snapshot.py tests/kg/test_kb_iteration_snapshot.py
git commit -m "feat: add kb iteration snapshotter"
```

## Task 2: Markdown Memory Writer

**Files:**
- Modify: `lightrag/kb_iteration/models.py`
- Create: `lightrag/kb_iteration/markdown.py`
- Test: `tests/kg/test_kb_iteration_markdown.py`

- [ ] **Step 1: Write failing Markdown tests**

```python
from pathlib import Path

from lightrag.kb_iteration.markdown import write_markdown_memory
from lightrag.kb_iteration.models import KGSnapshot, SnapshotEdge, SnapshotNode


def _snapshot() -> KGSnapshot:
    return KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=["guideline.md"],
        nodes=[
            SnapshotNode("流行性感冒", "流行性感冒", "Disease", "疾病", "chunk-1", "guideline.md"),
            SnapshotNode("发热", "发热", "Symptom", "症状", "chunk-1", "guideline.md"),
        ],
        edges=[
            SnapshotEdge("e1", "流行性感冒", "发热", "临床表现", "表现为发热", "chunk-1", "guideline.md"),
        ],
        metadata={"profile": "clinical_guideline_zh", "graph_node_count": 2, "graph_edge_count": 1},
    )


def test_write_markdown_memory_creates_llm_entrypoint_and_catalogs(tmp_path: Path):
    paths = write_markdown_memory(_snapshot(), tmp_path)

    assert (tmp_path / "kb_context.md").exists()
    assert (tmp_path / "entity_catalog.md").exists()
    assert (tmp_path / "relation_catalog.md").exists()
    assert (tmp_path / "kg_structure.md").exists()
    assert "influenza_medical_v1" in (tmp_path / "kb_context.md").read_text(encoding="utf-8")
    assert "Disease" in (tmp_path / "entity_catalog.md").read_text(encoding="utf-8")
    assert "临床表现" in (tmp_path / "relation_catalog.md").read_text(encoding="utf-8")
    assert paths["kb_context"].name == "kb_context.md"
```

- [ ] **Step 2: Run RED**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_markdown.py -q
```

Expected: import failure because `markdown.py` does not exist.

- [ ] **Step 3: Implement Markdown writers**

Implementation requirements:

- `kb_context.md` includes workspace, profile, source files, graph scale, entity type distribution, relation keyword distribution, latest score summary when available, and links to detail files.
- `entity_catalog.md` groups by entity type and lists representative nodes.
- `relation_catalog.md` groups by keyword and preserves `source -> target`.
- `kg_structure.md` lists hierarchy-like edges where keywords are `属于`, `症状归类`, or node type is `MedicalGroup`.
- Initialize missing rule-memory files if absent: `quality_rules.md`, `known_issues.md`, `accepted_changes.md`, `rejected_changes.md`, `approval_queue.md`, `improvement_backlog.md`, `iteration_log.md`, `diff_report.md`.
- Keep output deterministic.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_markdown.py tests\kg\test_kb_iteration_snapshot.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add lightrag/kb_iteration/models.py lightrag/kb_iteration/markdown.py tests/kg/test_kb_iteration_markdown.py
git commit -m "feat: write kb iteration markdown memory"
```

## Task 3: Deterministic Quality Scoring

**Files:**
- Modify: `lightrag/kb_iteration/models.py`
- Create: `lightrag/kb_iteration/quality.py`
- Test: `tests/kg/test_kb_iteration_quality.py`

- [ ] **Step 1: Write failing quality tests**

```python
from lightrag.kb_iteration.models import KGSnapshot, SnapshotEdge, SnapshotNode
from lightrag.kb_iteration.quality import evaluate_snapshot_quality


def test_quality_flags_value_nodes_generic_relations_and_missing_evidence():
    snapshot = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=[
            SnapshotNode("流行性感冒", "流行性感冒", "Disease"),
            SnapshotNode("75 mg", "75 mg", "Dosage"),
            SnapshotNode("发热", "发热", "Symptom", source_id="chunk-1", file_path="guide.md"),
        ],
        edges=[
            SnapshotEdge("e1", "流行性感冒", "75 mg", "邻接"),
            SnapshotEdge("e2", "流行性感冒", "发热", "临床表现", source_id="chunk-1", file_path="guide.md"),
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )

    score = evaluate_snapshot_quality(snapshot)

    assert score.metrics["value_like_node_count"] == 1
    assert score.metrics["generic_relation_count"] == 1
    assert score.metrics["missing_edge_source_count"] == 1
    assert score.subscores["entity_hygiene"] < 100
    assert score.subscores["relation_semantics"] < 100
    assert score.overall < 100
    assert any(f.severity == "high" for f in score.findings)
```

- [ ] **Step 2: Run RED**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_quality.py -q
```

Expected: import failure because `quality.py` does not exist.

- [ ] **Step 3: Implement deterministic checks**

Implementation requirements:

- Reuse `lightrag.medical_kg.ontology.is_value_like_entity`.
- Treat `相关`, `邻接`, empty keywords, and `related` as generic relations.
- Count missing `source_id` and `file_path` separately for nodes and edges.
- Compute hierarchy coverage for flu top-level categories when the workspace or profile indicates the medical profile.
- Compute disease hub overload as max disease degree divided by total edge count.
- Produce sub-scores: `entity_hygiene`, `relation_semantics`, `hierarchy_completeness`, `evidence_grounding`, `web_readability`, `iteration_readiness`.
- Overall score is the weighted average from the spec.
- Findings include severity, category, message, evidence, suggested fix type, and `requires_approval`.

- [ ] **Step 4: Add score JSON writer**

Expose:

```python
def write_quality_artifacts(score: QualityScore, output_dir: Path) -> dict[str, Path]:
    ...
```

It writes:

- `snapshots/quality_score.json`
- `quality_report.md`

- [ ] **Step 5: Run GREEN**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_quality.py tests\kg\test_kb_iteration_snapshot.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add lightrag/kb_iteration/models.py lightrag/kb_iteration/quality.py tests/kg/test_kb_iteration_quality.py
git commit -m "feat: score kb iteration quality"
```

## Task 4: Structured Proposals And Approval Queue

**Files:**
- Modify: `lightrag/kb_iteration/models.py`
- Create: `lightrag/kb_iteration/proposals.py`
- Test: `tests/kg/test_kb_iteration_proposals.py`

- [ ] **Step 1: Write failing proposal tests**

```python
from pathlib import Path

import pytest

from lightrag.kb_iteration.models import ImprovementProposal
from lightrag.kb_iteration.proposals import validate_proposal, write_approval_queue


def test_approval_queue_requires_mutation_items_to_be_gated(tmp_path: Path):
    proposal = ImprovementProposal(
        id="proposal-20260617-001",
        type="hierarchy_rule_change",
        target="lightrag/medical_kg/hierarchy.py",
        proposed_change="Add a controlled symptom branch for fever.",
        reason="Direct disease-to-leaf overload was detected.",
        evidence=["流行性感冒 -> 发热"],
        confidence=0.8,
        risk="medium",
        requires_approval=True,
        expected_metric_change={"hierarchy_completeness": 5},
    )

    validate_proposal(proposal)
    path = write_approval_queue([proposal], tmp_path)

    text = path.read_text(encoding="utf-8")
    assert "proposal-20260617-001" in text
    assert "requires_approval: true" in text
    assert "hierarchy_rule_change" in text


def test_validate_proposal_rejects_ungated_mutation():
    proposal = ImprovementProposal(
        id="proposal-20260617-002",
        type="prompt_edit",
        target="prompts/entity_type/医学实体类型提示词.yml",
        proposed_change="Change extraction prompt.",
        reason="Prompt change affects extraction.",
        evidence=["quality finding"],
        confidence=0.9,
        risk="high",
        requires_approval=False,
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="requires approval"):
        validate_proposal(proposal)
```

- [ ] **Step 2: Run RED**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_proposals.py -q
```

Expected: import failure because `proposals.py` does not exist.

- [ ] **Step 3: Implement proposal schema and queue writer**

Implementation requirements:

- Required proposal fields: `id`, `type`, `target`, `proposed_change`, `reason`, `evidence`, `confidence`, `risk`, `requires_approval`, `expected_metric_change`.
- Mutation proposal types require approval: `prompt_edit`, `ontology_rule_change`, `hierarchy_rule_change`, `relation_rule_change`, `workspace_rebuild`, `kg_fact_correction`, `web_display_change`.
- `approval_queue.md` renders YAML-like blocks so a human can review.
- `improvement_backlog.md` includes both approval-required and no-approval items with severity and expected metric effects.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_proposals.py tests\kg\test_kb_iteration_quality.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```powershell
git add lightrag/kb_iteration/models.py lightrag/kb_iteration/proposals.py tests/kg/test_kb_iteration_proposals.py
git commit -m "feat: add kb iteration approval proposals"
```

## Task 5: Snapshot Diff And Regression Report

**Files:**
- Modify: `lightrag/kb_iteration/models.py`
- Create: `lightrag/kb_iteration/diff.py`
- Test: `tests/kg/test_kb_iteration_diff.py`

- [ ] **Step 1: Write failing diff tests**

```python
from pathlib import Path

from lightrag.kb_iteration.diff import compare_snapshots, write_diff_report
from lightrag.kb_iteration.models import KGSnapshot, SnapshotEdge, SnapshotNode


def _snapshot(nodes, edges):
    return KGSnapshot(
        workspace="demo",
        generated_at="2026-06-17T00:00:00+08:00",
        source_files=[],
        nodes=nodes,
        edges=edges,
        metadata={},
    )


def test_compare_snapshots_reports_added_removed_and_changed_relations(tmp_path: Path):
    before = _snapshot(
        [SnapshotNode("流行性感冒", "流行性感冒", "Disease")],
        [SnapshotEdge("e1", "流行性感冒", "发热", "邻接")],
    )
    after = _snapshot(
        [
            SnapshotNode("流行性感冒", "流行性感冒", "Disease"),
            SnapshotNode("发热", "发热", "Symptom"),
        ],
        [SnapshotEdge("e1", "流行性感冒", "发热", "临床表现")],
    )

    diff = compare_snapshots(before, after)
    path = write_diff_report(diff, tmp_path)

    assert diff.added_nodes == ["发热"]
    assert ("流行性感冒", "发热") in diff.changed_relation_keywords
    assert "Changed relation keywords" in path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run RED**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_diff.py -q
```

Expected: import failure because `diff.py` does not exist.

- [ ] **Step 3: Implement deterministic diff**

Implementation requirements:

- Compare added and removed node ids.
- Compare changed entity types for common nodes.
- Compare added and removed edge pairs.
- Compare changed relation keywords for common edge pairs.
- Include dangerous regression flags:
  - core disease node removed
  - generic relation count increased when quality scores are supplied
  - evidence coverage decreased when quality scores are supplied
- Write both `diff_report.md` and `snapshots/diff_summary.json`.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_diff.py tests\kg\test_kb_iteration_snapshot.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

```powershell
git add lightrag/kb_iteration/models.py lightrag/kb_iteration/diff.py tests/kg/test_kb_iteration_diff.py
git commit -m "feat: diff kb iteration snapshots"
```

## Task 6: End-To-End Runner

**Files:**
- Create: `lightrag/kb_iteration/runner.py`
- Test: `tests/kg/test_kb_iteration_runner.py`

- [ ] **Step 1: Write failing runner test**

```python
from pathlib import Path

import networkx as nx

from lightrag.kb_iteration.runner import run_iteration


def test_run_iteration_writes_core_artifacts(tmp_path: Path):
    storage_dir = tmp_path / "rag_storage" / "demo"
    input_dir = tmp_path / "inputs" / "demo"
    storage_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)
    (input_dir / "guideline.md").write_text("source", encoding="utf-8")

    graph = nx.MultiDiGraph()
    graph.add_node("流行性感冒", entity_type="Disease", source_id="chunk-1", file_path="guideline.md")
    graph.add_node("发热", entity_type="Symptom", source_id="chunk-1", file_path="guideline.md")
    graph.add_edge("流行性感冒", "发热", keywords="临床表现", source_id="chunk-1", file_path="guideline.md")
    nx.write_graphml(graph, storage_dir / "graph_chunk_entity_relation.graphml")

    result = run_iteration(
        workspace="demo",
        storage_root=tmp_path / "rag_storage",
        input_root=tmp_path / "inputs",
        output_root=tmp_path / "work" / "kb-iteration",
        profile="clinical_guideline_zh",
    )

    assert result.output_dir == tmp_path / "work" / "kb-iteration" / "demo"
    assert (result.output_dir / "snapshots" / "kg_snapshot.json").exists()
    assert (result.output_dir / "kb_context.md").exists()
    assert (result.output_dir / "quality_report.md").exists()
    assert (result.output_dir / "iteration_log.md").exists()
```

- [ ] **Step 2: Run RED**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_runner.py -q
```

Expected: import failure because `runner.py` does not exist.

- [ ] **Step 3: Implement runner**

Implementation requirements:

- Resolve GraphML path as `<storage_root>/<workspace>/graph_chunk_entity_relation.graphml`.
- Discover source files from `<input_root>/<workspace>`.
- Write snapshot artifacts.
- Write Markdown memory.
- Evaluate quality.
- Write quality artifacts.
- Append one loop-state record to `iteration_log.md` with phase `pending_user_review`.
- Stop without applying any mutation.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_runner.py tests\kg\test_kb_iteration_quality.py tests\kg\test_kb_iteration_markdown.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 6**

```powershell
git add lightrag/kb_iteration/runner.py tests/kg/test_kb_iteration_runner.py
git commit -m "feat: run kb iteration reports"
```

## Task 7: Documentation And Verification

**Files:**
- Modify: `docs/superpowers/specs/2026-06-17-kb-iteration-agent-design.md`
- Modify: `docs/superpowers/specs/2026-06-17-kb-iteration-agent-design-zh.md`
- Create: `docs/KBIterationAgent.md`
- Create: `docs/KBIterationAgent-zh.md`

- [ ] **Step 1: Write docs**

Document:

- purpose and safety boundary
- artifact layout
- first deterministic run workflow
- source-grounding limits
- approval queue behavior
- example command using `run_iteration(...)`

- [ ] **Step 2: Run focused tests**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_snapshot.py tests\kg\test_kb_iteration_markdown.py tests\kg\test_kb_iteration_quality.py tests\kg\test_kb_iteration_proposals.py tests\kg\test_kb_iteration_diff.py tests\kg\test_kb_iteration_runner.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run broader medical KG tests**

Run:

```powershell
uv run pytest tests\kg\test_medical_kg_quality_snapshot.py tests\extraction\test_medical_kg_hierarchy.py tests\api\routes\test_medical_graph_projection.py -q
```

Expected: all tests pass or known unrelated failures are documented in `progress.md`.

- [ ] **Step 4: Check identifiers and formatting**

Run:

```powershell
git diff --check
rg -n "T[B]D|TO[D]O|implement[ ]later|fill[ ]in" docs/KBIterationAgent.md docs/KBIterationAgent-zh.md lightrag/kb_iteration tests/kg/test_kb_iteration_*.py
```

Expected:

- `git diff --check` has no whitespace errors.
- `rg` exits with no matches.

- [ ] **Step 5: Commit Task 7**

```powershell
git add docs/KBIterationAgent.md docs/KBIterationAgent-zh.md docs/superpowers/specs/2026-06-17-kb-iteration-agent-design.md docs/superpowers/specs/2026-06-17-kb-iteration-agent-design-zh.md
git commit -m "docs: describe kb iteration agent workflow"
```

## Final Review

After all tasks are complete:

- Run the full focused KB iteration test set.
- Request final code review with base SHA before Task 1 and head SHA after Task 7.
- Update `task_plan.md`, `findings.md`, and `progress.md`.
- Summarize generated artifacts and remaining approval-gated behavior for the user.
