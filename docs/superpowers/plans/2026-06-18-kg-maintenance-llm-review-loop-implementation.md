# KG Maintenance LLM Review Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an approval-gated LLM review loop to KG Maintenance so maintainers can generate focused LLM analysis, structured proposals, Judge recommendations, and optional patch candidates without automatically mutating KG facts, prompts, rules, WebUI code, or workspaces.

**Architecture:** Keep the existing deterministic `lightrag/kb_iteration` runner as the source of snapshot, Markdown memory, and quality artifacts. Add a separate LLM review loop that reads those artifacts, builds focused context packages, calls an injectable LLM client, validates structured outputs through deterministic code, writes audit artifacts, and exposes read/run APIs to the WebUI. WebUI changes only display and control the loop; all mutation remains approval-gated.

**Tech Stack:** Python dataclasses, FastAPI, PyYAML, pytest, React 19, TypeScript, Zustand, Tailwind, Bun test runner.

---

## Scope

This plan implements the first useful version plus extension points:

- Backend LLM review models and artifact writers.
- Focused review context builder.
- Injectable LLM review client with mock-friendly interfaces.
- Review loop runner with deterministic validation and trace output.
- Optional Judge result parsing and report output.
- Patch candidate file safety helpers, without applying patches.
- API routes for running and reading LLM review artifacts.
- WebUI panels for LLM review, Judge, and patch candidates.

This plan does not implement automatic patch application, workspace rebuild execution, or direct KG mutation.

## File Structure

### Backend

- Create `lightrag/kb_iteration/llm_review.py`
  - Owns LLM review dataclasses, JSON/YAML parsing, report rendering, and proposal conversion.
- Create `lightrag/kb_iteration/review_context.py`
  - Builds focused context packages from existing snapshot, quality, catalog, and rule artifacts.
- Create `lightrag/kb_iteration/review_loop.py`
  - Orchestrates observe/select/retrieve/analyze/propose/validate/judge/queue.
- Create `lightrag/kb_iteration/patches.py`
  - Validates patch candidate targets and writes patch candidate files.
- Create prompt files:
  - `lightrag/kb_iteration/prompts/planner_zh.md`
  - `lightrag/kb_iteration/prompts/reviewer_zh.md`
  - `lightrag/kb_iteration/prompts/judge_zh.md`
  - `lightrag/kb_iteration/prompts/patch_generator_zh.md`
- Modify `lightrag/kb_iteration/models.py`
  - Extend `ImprovementProposal` with optional `patch_candidate` and `judge`.
- Modify `lightrag/kb_iteration/proposals.py`
  - Allow new proposal types and render optional extension fields.
- Modify `lightrag/kb_iteration/__init__.py`
  - Export the public LLM review runner and key dataclasses.
- Modify `lightrag/api/routers/kb_iteration_routes.py`
  - Add LLM review artifact whitelist, run endpoint, read endpoints, and workspace locks.

### Backend Tests

- Create `tests/kg/test_kb_iteration_llm_review.py`
- Create `tests/kg/test_kb_iteration_review_context.py`
- Create `tests/kg/test_kb_iteration_review_loop.py`
- Create `tests/kg/test_kb_iteration_patches.py`
- Modify `tests/kg/test_kb_iteration_proposals.py`
- Modify `tests/api/routes/test_kb_iteration_routes.py`

### WebUI

- Modify `lightrag_webui/src/api/lightrag.ts`
  - Add LLM review response types and API wrappers.
- Modify `lightrag_webui/src/stores/kgMaintenance.ts`
  - Add `llm-review`, `patches`, and `judge` sections.
- Modify `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.tsx`
  - Add navigation items.
- Create `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx`
  - Render LLM review report, trace, proposals, patch candidates, and Judge output.
- Modify `lightrag_webui/src/features/KGMaintenanceConsole.tsx`
  - Load LLM review artifacts and route new sections.
- Modify `lightrag_webui/src/components/kg-maintenance/kgMaintenanceData.ts`
  - Parse proposal extension fields.

### WebUI Tests

- Create `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.test.tsx`
- Modify `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx`
- Modify `lightrag_webui/src/api/lightrag.test.ts`

---

## Task 1: Extend Proposal Model Safely

**Files:**
- Modify: `lightrag/kb_iteration/models.py`
- Modify: `lightrag/kb_iteration/proposals.py`
- Modify: `tests/kg/test_kb_iteration_proposals.py`

- [ ] **Step 1: Write failing tests for proposal extension fields and new types**

Append to `tests/kg/test_kb_iteration_proposals.py`:

```python
def test_improvement_proposal_renders_patch_candidate_and_judge(tmp_path: Path):
    proposal = ImprovementProposal(
        id="proposal-20260618-001",
        type="relation_keyword_mapping",
        target="lightrag/medical_kg/ontology.py",
        proposed_change="Map generic relation keywords to controlled relation labels.",
        reason="Generic relation labels reduce KG readability.",
        evidence=["edge:e1"],
        confidence=0.82,
        risk="medium",
        requires_approval=True,
        expected_metric_change={"relation_semantics": 8},
        patch_candidate="patch_candidates/proposal-20260618-001.patch",
        judge={
            "decision": "needs_human",
            "reason": "Rule change requires maintainer review.",
        },
    )

    text = write_approval_queue([proposal], tmp_path).read_text(encoding="utf-8")
    payload = _load_yaml_body(text)
    rendered = payload["proposals"][0]

    assert rendered["type"] == "relation_keyword_mapping"
    assert rendered["patch_candidate"] == "patch_candidates/proposal-20260618-001.patch"
    assert rendered["judge"]["decision"] == "needs_human"


@pytest.mark.parametrize(
    "proposal_type",
    [
        "source_evidence_repair",
        "synonym_merge_rule",
        "relation_keyword_mapping",
        "review_context_request",
        "llm_judge_rejection",
    ],
)
def test_new_llm_review_proposal_types_require_approval_by_default(proposal_type: str):
    proposal = ImprovementProposal(
        id=f"proposal-20260618-{proposal_type}",
        type=proposal_type,
        target="review-target",
        proposed_change="Record or prepare a review action.",
        reason="LLM review generated a structured proposal.",
        evidence=["review-context:round-001"],
        confidence=0.7,
        risk="medium",
        requires_approval=False,
        expected_metric_change={},
    )

    with pytest.raises(ValueError, match="requires approval"):
        validate_proposal(proposal)
```

- [ ] **Step 2: Run proposal tests to verify failure**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_proposals.py -q
```

Expected: fail with `TypeError: ImprovementProposal.__init__() got an unexpected keyword argument 'patch_candidate'`.

- [ ] **Step 3: Extend `ImprovementProposal`**

Modify `lightrag/kb_iteration/models.py`:

```python
@dataclass(frozen=True)
class ImprovementProposal:
    id: str
    type: str
    target: str
    proposed_change: str
    reason: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    risk: str = "medium"
    requires_approval: bool = True
    expected_metric_change: dict[str, int | float] = field(default_factory=dict)
    patch_candidate: str = ""
    judge: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

- [ ] **Step 4: Extend proposal type sets and rendering**

Modify `lightrag/kb_iteration/proposals.py`:

```python
MUTATION_PROPOSAL_TYPES = {
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

NO_APPROVAL_PROPOSAL_TYPES = {
    "quality_report_note",
}

REVIEW_ONLY_PROPOSAL_TYPES = {
    "review_context_request",
    "llm_judge_rejection",
}
```

Update the unknown-type check in `validate_proposal`:

```python
    if (
        proposal.type not in MUTATION_PROPOSAL_TYPES
        and proposal.type not in NO_APPROVAL_PROPOSAL_TYPES
        and proposal.type not in REVIEW_ONLY_PROPOSAL_TYPES
        and proposal.requires_approval is not True
    ):
        raise ValueError(f"unknown proposal type {proposal.type} requires approval")

    if proposal.type in REVIEW_ONLY_PROPOSAL_TYPES and proposal.requires_approval is not True:
        raise ValueError(f"proposal type {proposal.type} requires approval")
```

Update `_proposal_to_render_dict`:

```python
def _proposal_to_render_dict(proposal: ImprovementProposal) -> dict[str, object]:
    rendered: dict[str, object] = {
        "id": proposal.id,
        "type": proposal.type,
        "target": proposal.target,
        "proposed_change": proposal.proposed_change,
        "reason": proposal.reason,
        "evidence": proposal.evidence,
        "confidence": proposal.confidence,
        "risk": proposal.risk,
        "requires_approval": proposal.requires_approval,
        "expected_metric_change": {
            key: proposal.expected_metric_change[key]
            for key in sorted(proposal.expected_metric_change)
        },
    }
    if proposal.patch_candidate:
        rendered["patch_candidate"] = proposal.patch_candidate
    if proposal.judge:
        rendered["judge"] = proposal.judge
    return rendered
```

- [ ] **Step 5: Run proposal tests**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_proposals.py -q
```

Expected: all proposal tests pass.

- [ ] **Step 6: Commit**

```powershell
git add lightrag\kb_iteration\models.py lightrag\kb_iteration\proposals.py tests\kg\test_kb_iteration_proposals.py
git commit -m "feat: extend kb iteration proposals for llm review"
```

---

## Task 2: Add Patch Candidate Safety Helpers

**Files:**
- Create: `lightrag/kb_iteration/patches.py`
- Create: `tests/kg/test_kb_iteration_patches.py`

- [ ] **Step 1: Write failing patch safety tests**

Create `tests/kg/test_kb_iteration_patches.py`:

```python
from pathlib import Path

import pytest

from lightrag.kb_iteration.patches import (
    PatchCandidate,
    validate_patch_candidate,
    write_patch_candidate,
)


def test_write_patch_candidate_stays_inside_patch_directory(tmp_path: Path):
    candidate = PatchCandidate(
        proposal_id="proposal-20260618-001",
        target_path="lightrag/medical_kg/hierarchy.py",
        diff_text=(
            "--- a/lightrag/medical_kg/hierarchy.py\n"
            "+++ b/lightrag/medical_kg/hierarchy.py\n"
            "@@\n"
            "+# candidate change\n"
        ),
    )

    path = write_patch_candidate(candidate, tmp_path)

    assert path == tmp_path / "patch_candidates" / "proposal-20260618-001.patch"
    assert "candidate change" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "target_path",
    [
        ".env",
        "data/rag_storage/demo/graph_chunk_entity_relation.graphml",
        "work/kb-iteration/demo/approval_queue.md",
        "uv.lock",
        "../outside.py",
        "C:/Users/secret/file.py",
    ],
)
def test_patch_candidate_rejects_disallowed_targets(target_path: str):
    candidate = PatchCandidate(
        proposal_id="proposal-20260618-unsafe",
        target_path=target_path,
        diff_text="--- a/file\n+++ b/file\n",
    )

    with pytest.raises(ValueError):
        validate_patch_candidate(candidate)


def test_patch_candidate_allows_whitelisted_targets():
    candidate = PatchCandidate(
        proposal_id="proposal-20260618-safe",
        target_path="lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx",
        diff_text="--- a/file\n+++ b/file\n",
    )

    validate_patch_candidate(candidate)
```

- [ ] **Step 2: Run test to verify import failure**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_patches.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'lightrag.kb_iteration.patches'`.

- [ ] **Step 3: Implement patch helper**

Create `lightrag/kb_iteration/patches.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath, Path


ALLOWED_PATCH_TARGET_PREFIXES = (
    "prompts/",
    "lightrag/medical_kg/",
    "docs/",
    "lightrag_webui/src/components/kg-maintenance/",
)
DISALLOWED_PATCH_TARGETS = {
    ".env",
    "uv.lock",
}
PROPOSAL_ID_PATTERN = re.compile(r"^[a-z0-9_.-]+$")


@dataclass(frozen=True)
class PatchCandidate:
    proposal_id: str
    target_path: str
    diff_text: str


def validate_patch_candidate(candidate: PatchCandidate) -> None:
    if not PROPOSAL_ID_PATTERN.fullmatch(candidate.proposal_id):
        raise ValueError("proposal_id is not safe for a patch filename")
    if not candidate.diff_text.strip():
        raise ValueError("patch diff text must be non-empty")

    normalized = _normalize_target(candidate.target_path)
    if normalized in DISALLOWED_PATCH_TARGETS:
        raise ValueError(f"patch target is disallowed: {normalized}")
    if not any(normalized.startswith(prefix) for prefix in ALLOWED_PATCH_TARGET_PREFIXES):
        raise ValueError(f"patch target is outside the allowed prefixes: {normalized}")


def write_patch_candidate(candidate: PatchCandidate, output_dir: str | Path) -> Path:
    validate_patch_candidate(candidate)
    patch_dir = Path(output_dir) / "patch_candidates"
    patch_dir.mkdir(parents=True, exist_ok=True)
    path = patch_dir / f"{candidate.proposal_id}.patch"
    path.write_text(candidate.diff_text, encoding="utf-8")
    return path


def _normalize_target(target_path: str) -> str:
    raw = str(target_path).replace("\\", "/").strip()
    if not raw or ":" in raw:
        raise ValueError("patch target must be a relative repository path")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("patch target must not be absolute or traverse directories")
    return path.as_posix()
```

- [ ] **Step 4: Run patch tests**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_patches.py -q
```

Expected: all patch tests pass.

- [ ] **Step 5: Commit**

```powershell
git add lightrag\kb_iteration\patches.py tests\kg\test_kb_iteration_patches.py
git commit -m "feat: add safe kb patch candidates"
```

---

## Task 3: Add LLM Review Parsing And Artifact Rendering

**Files:**
- Create: `lightrag/kb_iteration/llm_review.py`
- Create: `tests/kg/test_kb_iteration_llm_review.py`

- [ ] **Step 1: Write failing parser and renderer tests**

Create `tests/kg/test_kb_iteration_llm_review.py`:

```python
from pathlib import Path

import pytest

from lightrag.kb_iteration.llm_review import (
    LLMReviewOutput,
    parse_llm_review_output,
    write_llm_review_artifacts,
)


def test_parse_llm_review_output_returns_proposals():
    output = parse_llm_review_output(
        """
        {
          "confirmed_issues": [{"message": "Generic relation is unclear."}],
          "hypotheses": [],
          "missing_evidence": [],
          "out_of_scope": [],
          "proposals": [
            {
              "id": "proposal-20260618-001",
              "type": "relation_keyword_mapping",
              "target": "lightrag/medical_kg/ontology.py",
              "proposed_change": "Map generic relation labels to controlled keywords.",
              "reason": "Generic relation labels reduce readability.",
              "evidence": ["edge:e1"],
              "confidence": 0.82,
              "risk": "medium",
              "requires_approval": true,
              "expected_metric_change": {"relation_semantics": 8}
            }
          ]
        }
        """
    )

    assert len(output.proposals) == 1
    assert output.proposals[0].type == "relation_keyword_mapping"
    assert output.confirmed_issues[0]["message"] == "Generic relation is unclear."


def test_parse_llm_review_output_rejects_invalid_json():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_llm_review_output("not json")


def test_write_llm_review_artifacts_writes_report_and_generated_yaml(tmp_path: Path):
    output = LLMReviewOutput(
        confirmed_issues=[{"message": "Issue"}],
        hypotheses=[],
        missing_evidence=[{"message": "Need chunk text"}],
        out_of_scope=[],
        proposals=[],
    )

    paths = write_llm_review_artifacts(output, tmp_path)

    assert paths["llm_review_report"] == tmp_path / "llm_review_report.md"
    assert paths["proposals_generated"] == tmp_path / "proposals.generated.yaml"
    assert "Issue" in paths["llm_review_report"].read_text(encoding="utf-8")
    assert "proposals:" in paths["proposals_generated"].read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify import failure**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_llm_review.py -q
```

Expected: fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `llm_review.py`**

Create `lightrag/kb_iteration/llm_review.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from .models import ImprovementProposal


class LLMReviewClient(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class LLMReviewOutput:
    confirmed_issues: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence: list[dict[str, Any]] = field(default_factory=list)
    out_of_scope: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[ImprovementProposal] = field(default_factory=list)


@dataclass(frozen=True)
class LLMJudgeResult:
    decision: str
    reason: str
    risk_override: str = ""
    required_human_checks: list[str] = field(default_factory=list)
    patch_consistency: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "risk_override": self.risk_override,
            "required_human_checks": self.required_human_checks,
            "patch_consistency": self.patch_consistency,
        }


def parse_llm_review_output(raw_text: str) -> LLMReviewOutput:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM review output must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM review output must be a JSON object")

    proposals = [
        ImprovementProposal(**proposal)
        for proposal in payload.get("proposals", [])
        if isinstance(proposal, dict)
    ]
    return LLMReviewOutput(
        confirmed_issues=_list_of_dicts(payload.get("confirmed_issues", [])),
        hypotheses=_list_of_dicts(payload.get("hypotheses", [])),
        missing_evidence=_list_of_dicts(payload.get("missing_evidence", [])),
        out_of_scope=_list_of_dicts(payload.get("out_of_scope", [])),
        proposals=proposals,
    )


def write_llm_review_artifacts(
    output: LLMReviewOutput, output_dir: str | Path
) -> dict[str, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / "llm_review_report.md"
    proposals_path = target_dir / "proposals.generated.yaml"

    report_path.write_text(_render_report(output), encoding="utf-8")
    proposals_path.write_text(
        "# Generated Proposals\n\n"
        + yaml.safe_dump(
            {"proposals": [proposal.to_dict() for proposal in output.proposals]},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "llm_review_report": report_path,
        "proposals_generated": proposals_path,
    }


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _render_report(output: LLMReviewOutput) -> str:
    lines = [
        "# LLM Review Report",
        "",
        "## Summary",
        "",
        f"- Confirmed issues: {len(output.confirmed_issues)}",
        f"- Hypotheses: {len(output.hypotheses)}",
        f"- Missing evidence: {len(output.missing_evidence)}",
        f"- Out of scope: {len(output.out_of_scope)}",
        f"- Generated proposals: {len(output.proposals)}",
    ]
    for title, values in (
        ("Confirmed Issues", output.confirmed_issues),
        ("Hypotheses", output.hypotheses),
        ("Missing Evidence", output.missing_evidence),
        ("Out Of Scope", output.out_of_scope),
    ):
        lines.extend(["", f"## {title}", ""])
        if values:
            lines.extend(f"- {_short_message(item)}" for item in values)
        else:
            lines.append("- none")
    lines.extend(["", "## Generated Proposals", ""])
    if output.proposals:
        lines.extend(f"- {proposal.id}: {proposal.type}" for proposal in output.proposals)
    else:
        lines.append("- none")
    lines.extend(["", "## Patch Candidates", "", "- none"])
    lines.extend(["", "## Human Review Required", ""])
    gated = [proposal for proposal in output.proposals if proposal.requires_approval]
    if gated:
        lines.extend(f"- {proposal.id}" for proposal in gated)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _short_message(item: dict[str, Any]) -> str:
    for key in ("message", "reason", "summary", "id"):
        value = str(item.get(key, "")).strip()
        if value:
            return value
    return json.dumps(item, ensure_ascii=False, sort_keys=True)
```

- [ ] **Step 4: Run parser tests**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_llm_review.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add lightrag\kb_iteration\llm_review.py tests\kg\test_kb_iteration_llm_review.py
git commit -m "feat: parse kb llm review outputs"
```

---

## Task 4: Build Focused Review Context Packages

**Files:**
- Create: `lightrag/kb_iteration/review_context.py`
- Create: `tests/kg/test_kb_iteration_review_context.py`

- [ ] **Step 1: Write failing context builder tests**

Create `tests/kg/test_kb_iteration_review_context.py`:

```python
from pathlib import Path

from lightrag.kb_iteration.review_context import build_review_context, write_review_context


def _write_json(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_review_context_selects_generic_relations(tmp_path: Path):
    package = tmp_path / "work" / "kb-iteration" / "demo"
    _write_json(
        package / "snapshots" / "kg_snapshot.json",
        """
        {
          "workspace": "demo",
          "nodes": [
            {"id": "flu", "label": "Influenza", "entity_type": "Disease", "source_id": "s1", "file_path": "flu.md"},
            {"id": "fever", "label": "Fever", "entity_type": "Symptom", "source_id": "s1", "file_path": "flu.md"}
          ],
          "edges": [
            {"id": "e1", "source": "flu", "target": "fever", "keywords": "相关", "source_id": "s1", "file_path": "flu.md"},
            {"id": "e2", "source": "flu", "target": "fever", "keywords": "clinical_manifestation", "source_id": "s1", "file_path": "flu.md"}
          ]
        }
        """,
    )
    _write_json(
        package / "snapshots" / "quality_score.json",
        """
        {
          "findings": [
            {
              "severity": "high",
              "category": "relation_semantics",
              "message": "Generic relation keywords should be replaced.",
              "evidence": ["e1"],
              "suggested_fix_type": "replace_relation_keyword",
              "requires_approval": true
            }
          ]
        }
        """,
    )
    (package / "accepted_changes.md").write_text("# Accepted\n", encoding="utf-8")
    (package / "rejected_changes.md").write_text("# Rejected\n- Do not merge flu and fever.\n", encoding="utf-8")

    context = build_review_context(package, round_id="round-001", focus=["generic_relation"])

    assert context["focus"] == ["generic_relation"]
    assert [edge["id"] for edge in context["relations"]] == ["e1"]
    assert {entity["id"] for entity in context["entities"]} == {"flu", "fever"}
    assert "Do not merge flu and fever" in context["rules_memory"]["rejected_changes"]


def test_write_review_context_creates_round_file(tmp_path: Path):
    package = tmp_path / "package"
    context = {"focus": ["generic_relation"], "relations": []}

    path = write_review_context(package, "round-001", context)

    assert path == package / "review_context" / "round-001-context.json"
    assert "generic_relation" in path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify import failure**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_review_context.py -q
```

Expected: fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement context builder**

Create `lightrag/kb_iteration/review_context.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


GENERIC_RELATION_KEYWORDS = {"", "相关", "邻接", "related", "neighbor", "adjacent"}


def build_review_context(
    package_dir: str | Path, *, round_id: str, focus: list[str]
) -> dict[str, Any]:
    package_path = Path(package_dir)
    snapshot = _read_json(package_path / "snapshots" / "kg_snapshot.json", {})
    quality = _read_json(package_path / "snapshots" / "quality_score.json", {})
    nodes = snapshot.get("nodes", []) if isinstance(snapshot, dict) else []
    edges = snapshot.get("edges", []) if isinstance(snapshot, dict) else []
    node_by_id = {str(node.get("id", "")): node for node in nodes if isinstance(node, dict)}

    selected_edges = _select_edges(edges, quality, focus)
    selected_node_ids = {
        str(edge.get("source", ""))
        for edge in selected_edges
        if isinstance(edge, dict)
    } | {
        str(edge.get("target", ""))
        for edge in selected_edges
        if isinstance(edge, dict)
    }
    selected_nodes = [
        node_by_id[node_id]
        for node_id in sorted(selected_node_ids)
        if node_id in node_by_id
    ]

    return {
        "round_id": round_id,
        "focus": focus,
        "quality_findings": _quality_findings_for_focus(quality, focus),
        "entities": selected_nodes,
        "relations": selected_edges,
        "evidence_windows": [],
        "rules_memory": {
            "accepted_changes": _read_text(package_path / "accepted_changes.md"),
            "rejected_changes": _read_text(package_path / "rejected_changes.md"),
        },
    }


def write_review_context(
    package_dir: str | Path, round_id: str, context: dict[str, Any]
) -> Path:
    target_dir = Path(package_dir) / "review_context"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{round_id}-context.json"
    path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _select_edges(
    edges: list[Any], quality: dict[str, Any], focus: list[str]
) -> list[dict[str, Any]]:
    normalized_focus = set(focus)
    if "generic_relation" in normalized_focus:
        return [
            edge
            for edge in edges
            if isinstance(edge, dict)
            and str(edge.get("keywords", "")).strip().casefold() in GENERIC_RELATION_KEYWORDS
        ]
    evidence_ids = {
        evidence.removeprefix("edge:")
        for finding in _quality_findings_for_focus(quality, focus)
        for evidence in finding.get("evidence", [])
        if isinstance(evidence, str)
    }
    if evidence_ids:
        return [
            edge
            for edge in edges
            if isinstance(edge, dict) and str(edge.get("id", "")) in evidence_ids
        ]
    return [edge for edge in edges[:10] if isinstance(edge, dict)]


def _quality_findings_for_focus(
    quality: dict[str, Any], focus: list[str]
) -> list[dict[str, Any]]:
    findings = quality.get("findings", []) if isinstance(quality, dict) else []
    if not focus:
        return [finding for finding in findings if isinstance(finding, dict)]
    focus_text = " ".join(focus).casefold()
    return [
        finding
        for finding in findings
        if isinstance(finding, dict)
        and (
            str(finding.get("category", "")).casefold() in focus_text
            or str(finding.get("suggested_fix_type", "")).casefold() in focus_text
            or str(finding.get("message", "")).casefold() in focus_text
        )
    ]


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")
```

- [ ] **Step 4: Run context tests**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_review_context.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add lightrag\kb_iteration\review_context.py tests\kg\test_kb_iteration_review_context.py
git commit -m "feat: build focused kb review context"
```

---

## Task 5: Implement The Mock-Friendly Review Loop Runner

**Files:**
- Create: `lightrag/kb_iteration/review_loop.py`
- Modify: `lightrag/kb_iteration/__init__.py`
- Create: `tests/kg/test_kb_iteration_review_loop.py`

- [ ] **Step 1: Write failing review loop tests**

Create `tests/kg/test_kb_iteration_review_loop.py`:

```python
import json
from pathlib import Path

from lightrag.kb_iteration.review_loop import LLMReviewLoopConfig, run_llm_review_loop


class StaticReviewClient:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "confirmed_issues": [{"message": "Generic relation needs review."}],
                "hypotheses": [],
                "missing_evidence": [],
                "out_of_scope": [],
                "proposals": [
                    {
                        "id": "proposal-20260618-001",
                        "type": "relation_keyword_mapping",
                        "target": "lightrag/medical_kg/ontology.py",
                        "proposed_change": "Map generic relation to controlled relation keywords.",
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


def _write_package(package: Path) -> None:
    snapshots = package / "snapshots"
    snapshots.mkdir(parents=True)
    (snapshots / "kg_snapshot.json").write_text(
        json.dumps(
            {
                "workspace": "demo",
                "nodes": [
                    {"id": "flu", "label": "Influenza", "entity_type": "Disease"},
                    {"id": "fever", "label": "Fever", "entity_type": "Symptom"},
                ],
                "edges": [
                    {"id": "e1", "source": "flu", "target": "fever", "keywords": "相关"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (snapshots / "quality_score.json").write_text(
        json.dumps(
            {
                "findings": [
                    {
                        "severity": "high",
                        "category": "relation_semantics",
                        "message": "Generic relation keywords should be replaced.",
                        "evidence": ["edge:e1"],
                        "suggested_fix_type": "replace_relation_keyword",
                        "requires_approval": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    for name in ("accepted_changes.md", "rejected_changes.md"):
        (package / name).write_text(f"# {name}\n", encoding="utf-8")


def test_run_llm_review_loop_writes_trace_report_and_queues(tmp_path: Path):
    package = tmp_path / "work" / "kb-iteration" / "demo"
    _write_package(package)

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
    assert "proposal-20260618-001" in (package / "approval_queue.md").read_text(
        encoding="utf-8"
    )
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    assert trace["rounds"][0]["focus"] == ["generic_relation"]
    assert trace["rounds"][0]["proposal_ids"] == ["proposal-20260618-001"]
```

- [ ] **Step 2: Run tests to verify import failure**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_review_loop.py -q
```

Expected: fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `review_loop.py`**

Create `lightrag/kb_iteration/review_loop.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lightrag.utils import validate_workspace

from .llm_review import LLMReviewClient, parse_llm_review_output, write_llm_review_artifacts
from .proposals import write_approval_queue, write_improvement_backlog, validate_proposal
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


def run_llm_review_loop(
    *,
    workspace: str,
    package_dir: str | Path,
    client: LLMReviewClient,
    config: LLMReviewLoopConfig | None = None,
    profile: str | None = None,
) -> LLMReviewRunResult:
    workspace = validate_workspace(workspace)
    cfg = config or LLMReviewLoopConfig()
    output_dir = Path(package_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_proposal_ids: list[str] = []
    artifact_paths: dict[str, Path] = {}
    trace: dict[str, Any] = {
        "workspace": workspace,
        "profile": profile or "",
        "started_at": _utc_timestamp(),
        "rounds": [],
        "stop_reason": "",
    }

    stop_reason = "max_review_rounds"
    for index in range(cfg.max_review_rounds):
        round_id = f"round-{index + 1:03d}"
        focus = _select_focus(output_dir)[: cfg.max_focus_items_per_round]
        context = build_review_context(output_dir, round_id=round_id, focus=focus)
        context_path = write_review_context(output_dir, round_id, context)

        raw_output = client.complete(
            system_prompt="You are a KG maintenance reviewer. Return valid JSON only.",
            user_prompt=json.dumps(context, ensure_ascii=False, sort_keys=True),
        )
        parsed = parse_llm_review_output(raw_output)
        for proposal in parsed.proposals:
            validate_proposal(proposal)

        artifact_paths.update(write_llm_review_artifacts(parsed, output_dir))
        artifact_paths["approval_queue"] = write_approval_queue(parsed.proposals, output_dir)
        artifact_paths["improvement_backlog"] = write_improvement_backlog(
            parsed.proposals, output_dir
        )

        proposal_ids = [proposal.id for proposal in parsed.proposals]
        all_proposal_ids.extend(proposal_ids)
        trace["rounds"].append(
            {
                "round_id": round_id,
                "focus": focus,
                "state": "queued" if proposal_ids else "no_valid_proposals",
                "context_files": [context_path.relative_to(output_dir).as_posix()],
                "model": "configured-review-model",
                "input_token_estimate": len(json.dumps(context, ensure_ascii=False)) // 4,
                "output_token_estimate": len(raw_output) // 4,
                "proposal_ids": proposal_ids,
                "judge_decision": "needs_human" if proposal_ids else "",
            }
        )

        if proposal_ids:
            stop_reason = "pending_human_review"
            break
        stop_reason = "all_proposals_invalid"

    trace["stop_reason"] = stop_reason
    trace_path = output_dir / "llm_review_trace.json"
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_paths["llm_review_trace"] = trace_path

    return LLMReviewRunResult(
        output_dir=output_dir,
        stop_reason=stop_reason,
        proposal_ids=all_proposal_ids,
        artifact_paths=artifact_paths,
    )


def _select_focus(output_dir: Path) -> list[str]:
    quality_path = output_dir / "snapshots" / "quality_score.json"
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        findings = quality.get("findings", []) if isinstance(quality, dict) else []
        for finding in findings:
            text = " ".join(
                str(finding.get(key, ""))
                for key in ("category", "message", "suggested_fix_type")
            ).casefold()
            if "generic" in text or "relation" in text:
                return ["generic_relation"]
            if "hierarchy" in text:
                return ["hierarchy_missing_branch"]
            if "evidence" in text or "source" in text:
                return ["missing_evidence"]
    return ["generic_relation"]


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
```

- [ ] **Step 4: Export public runner**

Modify `lightrag/kb_iteration/__init__.py`:

```python
from .review_loop import LLMReviewLoopConfig, LLMReviewRunResult, run_llm_review_loop
```

Keep existing exports intact.

- [ ] **Step 5: Run review loop tests**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_review_loop.py tests\kg\test_kb_iteration_llm_review.py tests\kg\test_kb_iteration_review_context.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add lightrag\kb_iteration\review_loop.py lightrag\kb_iteration\__init__.py tests\kg\test_kb_iteration_review_loop.py
git commit -m "feat: run kb llm review loop"
```

---

## Task 6: Add Versioned Prompt Files

**Files:**
- Create: `lightrag/kb_iteration/prompts/planner_zh.md`
- Create: `lightrag/kb_iteration/prompts/reviewer_zh.md`
- Create: `lightrag/kb_iteration/prompts/judge_zh.md`
- Create: `lightrag/kb_iteration/prompts/patch_generator_zh.md`
- Create: `tests/kg/test_kb_iteration_prompts.py`

- [ ] **Step 1: Write prompt presence tests**

Create `tests/kg/test_kb_iteration_prompts.py`:

```python
from pathlib import Path


def test_llm_review_prompts_exist_and_keep_safety_rules():
    prompt_dir = Path("lightrag/kb_iteration/prompts")
    expected = [
        "planner_zh.md",
        "reviewer_zh.md",
        "judge_zh.md",
        "patch_generator_zh.md",
    ]

    for filename in expected:
        text = (prompt_dir / filename).read_text(encoding="utf-8")
        assert "LLM" in text
        assert "source_id" in text
        assert "requires_approval" in text
```

- [ ] **Step 2: Run prompt test to verify failure**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_prompts.py -q
```

Expected: fail because prompt files do not exist.

- [ ] **Step 3: Add prompt files**

Create `lightrag/kb_iteration/prompts/planner_zh.md`:

```markdown
# Planner Prompt

你是 LightRAG KG 维护 Planner。你只选择本轮审阅焦点，不生成 proposal。

必须遵守：
- LLM 输出不是医学证据。
- 医学事实必须依赖 source_id、file_path 和 chunk。
- 涉及 prompt、rule、KG、workspace、WebUI 的变更必须 requires_approval=true。

输出 JSON：
{
  "focus_items": [
    {
      "category": "generic_relation",
      "reason": "why this focus matters",
      "priority": "high",
      "needed_context": ["relations", "source_target_entities", "evidence_windows"]
    }
  ],
  "stop_if": []
}
```

Create `lightrag/kb_iteration/prompts/reviewer_zh.md`:

```markdown
# Reviewer Prompt

你是医学 KG 维护审阅器。你只能根据提供的 review_context 分析问题。

必须遵守：
- 不得新增原文没有支持的医学事实。
- 不得把 LLM 判断当作医学证据。
- 只能使用 review_context 中的 source_id、file_path、chunk、entity、relation 作为证据。
- 证据不足时输出 missing_evidence。
- 涉及 prompt/rule/KG/workspace/WebUI 行为的 proposal 必须 requires_approval=true。

输出 JSON：
{
  "confirmed_issues": [],
  "hypotheses": [],
  "missing_evidence": [],
  "out_of_scope": [],
  "proposals": []
}
```

Create `lightrag/kb_iteration/prompts/judge_zh.md`:

```markdown
# Judge Prompt

你是 KG proposal Judge。你不重新生成方案，只评判 proposal 和候选 patch 是否可信。

必须检查：
- proposal 是否有 source_id、file_path 或 chunk 证据。
- patch 是否匹配 proposal。
- patch 是否触碰允许文件。
- 是否引入原文未支持的医学事实。
- mutation proposal 是否 requires_approval=true。
- rejected_changes 是否已经拒绝过类似建议。

输出 JSON：
{
  "decision": "recommend_accept | recommend_reject | needs_human | needs_more_evidence",
  "reason": "",
  "risk_override": "low | medium | high",
  "required_human_checks": [],
  "patch_consistency": {
    "matches_proposal": true,
    "touches_allowed_files": true,
    "introduces_unsupported_medical_claim": false
  }
}
```

Create `lightrag/kb_iteration/prompts/patch_generator_zh.md`:

```markdown
# Patch Generator Prompt

你为已通过 validate_proposal 的 proposal 生成候选 patch。候选 patch 不能自动应用。

必须遵守：
- 不修改 data/rag_storage、work/kb-iteration、.env、uv.lock。
- 不新增原文未支持的医学事实。
- 所有 mutation 仍然 requires_approval=true。
- patch 必须只服务当前 proposal。

输出 unified diff 文本。
```

- [ ] **Step 4: Run prompt tests**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_prompts.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add lightrag\kb_iteration\prompts tests\kg\test_kb_iteration_prompts.py
git commit -m "docs: add kb llm review prompts"
```

---

## Task 7: Add API Routes For LLM Review Artifacts

**Files:**
- Modify: `lightrag/api/routers/kb_iteration_routes.py`
- Modify: `tests/api/routes/test_kb_iteration_routes.py`

- [ ] **Step 1: Add failing API tests**

Append to `tests/api/routes/test_kb_iteration_routes.py`:

```python
def test_llm_review_artifact_routes_read_whitelisted_outputs(tmp_path: Path, monkeypatch):
    client, fixture = _client(tmp_path, monkeypatch)
    _write_text(fixture.package / "llm_review_report.md", "# LLM Review\n")
    _write_json(fixture.package / "llm_review_trace.json", {"stop_reason": "pending_human_review"})
    _write_text(fixture.package / "proposals.generated.yaml", "# Generated\nproposals: []\n")
    _write_text(fixture.package / "llm_judge_report.md", "# Judge\n")
    _write_json(fixture.package / "review_context" / "round-001-context.json", {"focus": ["generic_relation"]})
    _write_text(fixture.package / "patch_candidates" / "proposal-1.patch", "--- a/x\n+++ b/x\n")

    report = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/report",
        headers=HEADERS,
    )
    trace = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/trace",
        headers=HEADERS,
    )
    context = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/context/round-001",
        headers=HEADERS,
    )
    patch = client.get(
        "/kb-iteration/influenza_medical_v1/llm-review/patches/proposal-1",
        headers=HEADERS,
    )

    assert report.status_code == 200
    assert report.json()["content"] == "# LLM Review\n"
    assert trace.status_code == 200
    assert trace.json()["payload"]["stop_reason"] == "pending_human_review"
    assert context.status_code == 200
    assert context.json()["payload"]["focus"] == ["generic_relation"]
    assert patch.status_code == 200
    assert "--- a/x" in patch.json()["content"]


def test_llm_review_run_uses_lock_and_validated_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    calls = []

    class FakeClient:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return json.dumps(
                {
                    "confirmed_issues": [],
                    "hypotheses": [],
                    "missing_evidence": [],
                    "out_of_scope": [],
                    "proposals": [],
                }
            )

    def fake_run_llm_review_loop(**kwargs):
        calls.append(kwargs)
        _write_json(fixture.package / "llm_review_trace.json", {"stop_reason": "all_proposals_invalid"})
        _write_text(fixture.package / "llm_review_report.md", "# Review\n")
        _write_text(fixture.package / "proposals.generated.yaml", "proposals: []\n")
        return SimpleNamespace(
            output_dir=fixture.package,
            stop_reason="all_proposals_invalid",
            proposal_ids=[],
            artifact_paths={},
        )

    monkeypatch.setattr(kb_iteration_routes, "run_llm_review_loop", fake_run_llm_review_loop)
    monkeypatch.setattr(kb_iteration_routes, "_default_llm_review_client", lambda rag: FakeClient())

    response = client.post(
        "/kb-iteration/influenza_medical_v1/llm-review/runs",
        headers=HEADERS,
        json={"profile": "clinical_guideline_zh", "max_review_rounds": 1},
    )

    assert response.status_code == 200
    assert response.json()["stopReason"] == "all_proposals_invalid"
    assert calls[0]["workspace"] == "influenza_medical_v1"
    assert calls[0]["package_dir"] == fixture.package
```

- [ ] **Step 2: Run API tests to verify 404/import failure**

Run:

```powershell
uv run pytest tests\api\routes\test_kb_iteration_routes.py -q
```

Expected: new LLM review route tests fail with 404 or missing import.

- [ ] **Step 3: Add imports, artifact keys, request model, and fallback client**

Modify `lightrag/api/routers/kb_iteration_routes.py` imports:

```python
from lightrag.kb_iteration.review_loop import (
    LLMReviewLoopConfig,
    run_llm_review_loop,
)
```

Extend `ARTIFACTS`:

```python
    "llm_review_trace": ("llm_review_trace.json", "application/json"),
    "llm_review_report": ("llm_review_report.md", "text/markdown"),
    "llm_judge_report": ("llm_judge_report.md", "text/markdown"),
    "proposals_generated": ("proposals.generated.yaml", "text/markdown"),
```

Add request model near `RunIterationRequest`:

```python
class RunLLMReviewRequest(BaseModel):
    profile: Optional[str] = Field(default=None, max_length=200)
    mode: str = Field(default="loop", max_length=40)
    max_review_rounds: int = Field(default=4, ge=1, le=10)
    max_focus_items_per_round: int = Field(default=3, ge=1, le=10)
    max_context_tokens_per_round: int = Field(default=12000, ge=1000, le=100000)
    allow_llm_judge: bool = True
    allow_llm_auto_accept: bool = False
    allow_low_risk_auto_reject: bool = True
    generate_patch_candidates: bool = False
    require_human_for_mutation: bool = True
```

Add fallback client near helpers:

```python
class _UnavailableLLMReviewClient:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("LLM review client is not configured")


def _default_llm_review_client(rag):
    return _UnavailableLLMReviewClient()
```

- [ ] **Step 4: Add routes inside `create_kb_iteration_routes`**

Add after existing run routes:

```python
    @router.post("/{workspace}/llm-review/runs", dependencies=[Depends(combined_auth)])
    async def create_llm_review_run(
        workspace: str, request: RunLLMReviewRequest | None = None
    ) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        req = request or RunLLMReviewRequest()
        lock = _run_lock_for_workspace(f"{workspace}:llm-review")
        if not lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409, detail="KB LLM review run is already in progress"
            )
        try:
            with _exclusive_workspace_file_lock(
                _output_root(args), workspace, ".kb_iteration_llm_review.lock"
            ):
                result = await asyncio.to_thread(
                    run_llm_review_loop,
                    workspace=workspace,
                    package_dir=_workspace_dir(args, workspace),
                    client=_default_llm_review_client(rag),
                    config=LLMReviewLoopConfig(
                        max_review_rounds=req.max_review_rounds,
                        max_focus_items_per_round=req.max_focus_items_per_round,
                        max_context_tokens_per_round=req.max_context_tokens_per_round,
                        allow_llm_judge=req.allow_llm_judge,
                        allow_llm_auto_accept=req.allow_llm_auto_accept,
                        allow_low_risk_auto_reject=req.allow_low_risk_auto_reject,
                        generate_patch_candidates=req.generate_patch_candidates,
                        require_human_for_mutation=req.require_human_for_mutation,
                    ),
                    profile=req.profile,
                )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            lock.release()

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

    @router.get("/{workspace}/llm-review/proposals", dependencies=[Depends(combined_auth)])
    async def get_llm_review_proposals(workspace: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        return _read_artifact(args, workspace, "proposals_generated")

    @router.get("/{workspace}/llm-review/judge-report", dependencies=[Depends(combined_auth)])
    async def get_llm_judge_report(workspace: str) -> dict[str, Any]:
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
        path = _workspace_dir(args, workspace) / "review_context" / f"{round_id}-context.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="LLM review context not found")
        return {
            "artifactKey": f"review_context/{round_id}",
            "contentType": "application/json",
            "payload": _load_json(path),
        }

    @router.get(
        "/{workspace}/llm-review/patches/{proposal_id}",
        dependencies=[Depends(combined_auth)],
    )
    async def get_llm_review_patch(workspace: str, proposal_id: str) -> dict[str, Any]:
        workspace = _validate_workspace_or_400(workspace)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", proposal_id):
            raise HTTPException(status_code=400, detail="Invalid proposal id")
        base_dir = (_workspace_dir(args, workspace) / "patch_candidates").resolve()
        path = (base_dir / f"{proposal_id}.patch").resolve()
        try:
            path.relative_to(base_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Unsafe patch path") from exc
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="Patch candidate not found")
        return {
            "artifactKey": f"patch_candidates/{proposal_id}.patch",
            "contentType": "text/x-diff",
            "content": path.read_text(encoding="utf-8"),
        }
```

- [ ] **Step 5: Run API tests**

Run:

```powershell
uv run pytest tests\api\routes\test_kb_iteration_routes.py -q
```

Expected: all route tests pass.

- [ ] **Step 6: Commit**

```powershell
git add lightrag\api\routers\kb_iteration_routes.py tests\api\routes\test_kb_iteration_routes.py
git commit -m "feat: expose kb llm review artifacts"
```

---

## Task 8: Add WebUI API Wrappers

**Files:**
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Modify: `lightrag_webui/src/api/lightrag.test.ts`

- [ ] **Step 1: Add failing API wrapper tests**

Append to `lightrag_webui/src/api/lightrag.test.ts`:

```ts
import {
  __resetKBIterationRequestsForTests,
  __setKBIterationGetForTests,
  __setKBIterationPostForTests,
  getKBIterationLLMReviewReport,
  getKBIterationLLMReviewTrace,
  getKBIterationLLMReviewPatch,
  runKBIterationLLMReview
} from './lightrag'

test('kb llm review wrappers call expected paths', async () => {
  const getCalls: string[] = []
  const postCalls: Array<{ path: string; body: any }> = []
  __setKBIterationGetForTests(async (path) => {
    getCalls.push(path)
    return { artifactKey: path, contentType: 'text/markdown', content: 'ok' }
  })
  __setKBIterationPostForTests(async (path, body) => {
    postCalls.push({ path, body })
    return { workspace: 'demo', stopReason: 'pending_human_review', proposalIds: [] }
  })

  await runKBIterationLLMReview('demo', { max_review_rounds: 1 })
  await getKBIterationLLMReviewTrace('demo')
  await getKBIterationLLMReviewReport('demo')
  await getKBIterationLLMReviewPatch('demo', 'proposal-1')

  expect(postCalls[0]).toEqual({
    path: '/kb-iteration/demo/llm-review/runs',
    body: { max_review_rounds: 1 }
  })
  expect(getCalls).toEqual([
    '/kb-iteration/demo/llm-review/trace',
    '/kb-iteration/demo/llm-review/report',
    '/kb-iteration/demo/llm-review/patches/proposal-1'
  ])
  __resetKBIterationRequestsForTests()
})
```

- [ ] **Step 2: Run API test to verify missing exports**

Run:

```powershell
cd lightrag_webui
npx --yes bun test src/api/lightrag.test.ts
```

Expected: fail because new wrapper exports do not exist.

- [ ] **Step 3: Add types and wrappers**

In `lightrag_webui/src/api/lightrag.ts`, add near existing KB iteration types:

```ts
export type KBIterationLLMReviewRunRequest = {
  profile?: string | null
  mode?: 'loop' | string
  max_review_rounds?: number
  max_focus_items_per_round?: number
  max_context_tokens_per_round?: number
  allow_llm_judge?: boolean
  allow_llm_auto_accept?: boolean
  allow_low_risk_auto_reject?: boolean
  generate_patch_candidates?: boolean
  require_human_for_mutation?: boolean
}

export type KBIterationLLMReviewRunResponse = {
  workspace: string
  stopReason: string
  proposalIds: string[]
}
```

Add wrappers near existing KB iteration functions:

```ts
export const runKBIterationLLMReview = async (
  workspace: string,
  request: KBIterationLLMReviewRunRequest = {}
): Promise<KBIterationLLMReviewRunResponse> => {
  return kbIterationPost(
    `/kb-iteration/${encodePathSegment(workspace)}/llm-review/runs`,
    request
  )
}

export const getKBIterationLLMReviewTrace = async (
  workspace: string
): Promise<KBIterationArtifactResponse> => {
  return kbIterationGet(`/kb-iteration/${encodePathSegment(workspace)}/llm-review/trace`)
}

export const getKBIterationLLMReviewReport = async (
  workspace: string
): Promise<KBIterationArtifactResponse> => {
  return kbIterationGet(`/kb-iteration/${encodePathSegment(workspace)}/llm-review/report`)
}

export const getKBIterationLLMReviewProposals = async (
  workspace: string
): Promise<KBIterationArtifactResponse> => {
  return kbIterationGet(`/kb-iteration/${encodePathSegment(workspace)}/llm-review/proposals`)
}

export const getKBIterationLLMJudgeReport = async (
  workspace: string
): Promise<KBIterationArtifactResponse> => {
  return kbIterationGet(`/kb-iteration/${encodePathSegment(workspace)}/llm-review/judge-report`)
}

export const getKBIterationLLMReviewPatch = async (
  workspace: string,
  proposalId: string
): Promise<KBIterationArtifactResponse> => {
  return kbIterationGet(
    `/kb-iteration/${encodePathSegment(workspace)}/llm-review/patches/${encodePathSegment(proposalId)}`
  )
}
```

- [ ] **Step 4: Run API test**

Run:

```powershell
cd lightrag_webui
npx --yes bun test src/api/lightrag.test.ts
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add lightrag_webui\src\api\lightrag.ts lightrag_webui\src\api\lightrag.test.ts
git commit -m "feat: add kg llm review web api"
```

---

## Task 9: Add WebUI State And Navigation Sections

**Files:**
- Modify: `lightrag_webui/src/stores/kgMaintenance.ts`
- Modify: `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.tsx`
- Modify: `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx`

- [ ] **Step 1: Add failing shell test for new navigation labels**

Append to `KGMaintenanceShell.test.tsx`:

```tsx
test('renders llm review navigation sections', () => {
  render(
    <KGMaintenanceShell
      activeSection="llm-review"
      onSectionChange={() => {}}
      workspaces={['demo']}
      selectedWorkspace="demo"
      onWorkspaceChange={() => {}}
      onRefresh={() => {}}
      onRunReview={() => {}}
      loading={false}
      running={false}
      error={null}
      inspector={<div />}
    >
      <div />
    </KGMaintenanceShell>
  )

  expect(screen.getByText('LLM 审阅')).toBeInTheDocument()
  expect(screen.getByText('候选 Patch')).toBeInTheDocument()
  expect(screen.getByText('Judge 评判')).toBeInTheDocument()
})
```

- [ ] **Step 2: Run test to verify type/label failure**

Run:

```powershell
cd lightrag_webui
npx --yes bun test src/components/kg-maintenance/KGMaintenanceShell.test.tsx
```

Expected: fail because `llm-review` is not in `KGMaintenanceSection`.

- [ ] **Step 3: Extend section union**

Modify `lightrag_webui/src/stores/kgMaintenance.ts`:

```ts
export type KGMaintenanceSection =
  | 'overview'
  | 'graph'
  | 'entities'
  | 'relations'
  | 'evidence'
  | 'quality'
  | 'llm-review'
  | 'patches'
  | 'judge'
  | 'approval'
  | 'runs'
  | 'diff'
  | 'rules'
```

- [ ] **Step 4: Add navigation items**

Modify the `sections` array in `KGMaintenanceShell.tsx`:

```ts
  { id: 'llm-review', label: 'LLM 审阅', group: '审阅', icon: FileSearchIcon },
  { id: 'patches', label: '候选 Patch', group: '审阅', icon: GitCompareIcon },
  { id: 'judge', label: 'Judge 评判', group: '审阅', icon: ShieldCheckIcon },
```

Place them between `quality` and `approval`.

- [ ] **Step 5: Run shell test**

Run:

```powershell
cd lightrag_webui
npx --yes bun test src/components/kg-maintenance/KGMaintenanceShell.test.tsx
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add lightrag_webui\src\stores\kgMaintenance.ts lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.tsx lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.test.tsx
git commit -m "feat: add kg llm review navigation"
```

---

## Task 10: Add LLM Review WebUI Panels

**Files:**
- Create: `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx`
- Create: `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.test.tsx`

- [ ] **Step 1: Write failing panel tests**

Create `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import {
  LLMJudgePanel,
  LLMReviewPanel,
  PatchCandidatesPanel
} from './LLMReviewPanels'

test('llm review panel renders trace and report', () => {
  render(
    <LLMReviewPanel
      trace={{
        stop_reason: 'pending_human_review',
        rounds: [{ round_id: 'round-001', focus: ['generic_relation'], proposal_ids: ['p1'] }]
      }}
      report="# LLM Review Report"
      proposals="proposals: []"
      running={false}
      onRun={() => {}}
    />
  )

  expect(screen.getByText('LLM 审阅')).toBeInTheDocument()
  expect(screen.getByText('pending_human_review')).toBeInTheDocument()
  expect(screen.getByText('generic_relation')).toBeInTheDocument()
})

test('patch candidates panel renders selected patch', () => {
  render(
    <PatchCandidatesPanel
      proposals="proposals:\n- id: p1\n  patch_candidate: patch_candidates/p1.patch\n"
      patchText="--- a/file\n+++ b/file\n"
      onLoadPatch={() => {}}
    />
  )

  expect(screen.getByText('候选 Patch')).toBeInTheDocument()
  expect(screen.getByText(/--- a\/file/)).toBeInTheDocument()
})

test('judge panel renders judge report', () => {
  render(<LLMJudgePanel report="# Judge\nneeds_human" />)

  expect(screen.getByText('Judge 评判')).toBeInTheDocument()
  expect(screen.getByText(/needs_human/)).toBeInTheDocument()
})
```

- [ ] **Step 2: Run tests to verify missing component**

Run:

```powershell
cd lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx
```

Expected: fail because `LLMReviewPanels.tsx` does not exist.

- [ ] **Step 3: Implement panels**

Create `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx`:

```tsx
import Button from '@/components/ui/Button'
import { BrainCircuitIcon, FileDiffIcon, ShieldCheckIcon } from 'lucide-react'

type TraceRound = {
  round_id?: string
  focus?: string[]
  proposal_ids?: string[]
  state?: string
}

type LLMReviewPanelProps = {
  trace: Record<string, any> | null
  report: string
  proposals: string
  running: boolean
  onRun: () => void
}

export function LLMReviewPanel({
  trace,
  report,
  proposals,
  running,
  onRun
}: LLMReviewPanelProps) {
  const rounds = Array.isArray(trace?.rounds) ? (trace?.rounds as TraceRound[]) : []
  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold">LLM 审阅</h2>
          <p className="text-muted-foreground mt-1 text-sm">
            LLM 只生成分析、proposal 和候选 patch；医学事实仍以 source_id、file_path 和 chunk 为准。
          </p>
        </div>
        <Button size="sm" onClick={onRun} disabled={running}>
          <BrainCircuitIcon className="size-4" />
          {running ? '运行中' : '运行 LLM 审阅'}
        </Button>
      </div>
      <div className="border-border/70 rounded-lg border p-3 text-sm">
        <div className="text-muted-foreground text-xs">停止原因</div>
        <div className="mt-1 font-medium">{String(trace?.stop_reason || '尚未运行')}</div>
      </div>
      <div className="space-y-2">
        {rounds.map((round) => (
          <article key={round.round_id} className="border-border/70 rounded-lg border p-3">
            <div className="text-sm font-medium">{round.round_id}</div>
            <div className="text-muted-foreground mt-1 text-xs">
              {(round.focus || []).join(', ') || '无 focus'}
            </div>
            <div className="text-muted-foreground mt-1 text-xs">
              proposals: {(round.proposal_ids || []).join(', ') || '无'}
            </div>
          </article>
        ))}
      </div>
      <MarkdownBlock title="LLM 审阅报告" content={report} />
      <MarkdownBlock title="Generated proposals" content={proposals} />
    </section>
  )
}

export function PatchCandidatesPanel({
  proposals,
  patchText,
  onLoadPatch
}: {
  proposals: string
  patchText: string
  onLoadPatch: (proposalId: string) => void
}) {
  const proposalIds = Array.from(proposals.matchAll(/id:\s*([A-Za-z0-9_.-]+)/g)).map(
    (match) => match[1]
  )
  return (
    <section className="space-y-4">
      <PanelTitle icon={<FileDiffIcon className="size-4" />} title="候选 Patch" />
      <div className="flex flex-wrap gap-2">
        {proposalIds.map((proposalId) => (
          <Button key={proposalId} variant="outline" size="sm" onClick={() => onLoadPatch(proposalId)}>
            {proposalId}
          </Button>
        ))}
      </div>
      <MarkdownBlock title="Patch diff" content={patchText || '暂无候选 patch'} />
    </section>
  )
}

export function LLMJudgePanel({ report }: { report: string }) {
  return (
    <section className="space-y-4">
      <PanelTitle icon={<ShieldCheckIcon className="size-4" />} title="Judge 评判" />
      <MarkdownBlock title="Judge report" content={report || '暂无 Judge 评判'} />
    </section>
  )
}

function PanelTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="flex items-center gap-2">
      {icon}
      <h2 className="text-sm font-semibold">{title}</h2>
    </div>
  )
}

function MarkdownBlock({ title, content }: { title: string; content: string }) {
  return (
    <details className="border-border/70 rounded-lg border p-3" open>
      <summary className="cursor-pointer text-sm font-medium">{title}</summary>
      <pre className="text-muted-foreground mt-3 max-h-96 overflow-auto whitespace-pre-wrap text-xs">
        {content || '无内容'}
      </pre>
    </details>
  )
}
```

- [ ] **Step 4: Run panel tests**

Run:

```powershell
cd lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.test.tsx
git commit -m "feat: render kg llm review panels"
```

---

## Task 11: Wire LLM Review Panels Into The Console

**Files:**
- Modify: `lightrag_webui/src/features/KGMaintenanceConsole.tsx`
- Modify: `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx`

- [ ] **Step 1: Add console integration test**

If `KGMaintenanceShell.test.tsx` already covers navigation only, create or extend `KGMaintenanceShell.test.tsx` with a light smoke test that the section type compiles. Add:

```tsx
test('llm review section type is accepted by shell', () => {
  render(
    <KGMaintenanceShell
      activeSection="llm-review"
      onSectionChange={() => {}}
      workspaces={['demo']}
      selectedWorkspace="demo"
      onWorkspaceChange={() => {}}
      onRefresh={() => {}}
      onRunReview={() => {}}
      loading={false}
      running={false}
      error={null}
      inspector={<div />}
    >
      <div>LLM body</div>
    </KGMaintenanceShell>
  )

  expect(screen.getByText('LLM body')).toBeInTheDocument()
})
```

- [ ] **Step 2: Modify console imports and state**

In `KGMaintenanceConsole.tsx`, import wrappers and panels:

```ts
  getKBIterationLLMJudgeReport,
  getKBIterationLLMReviewPatch,
  getKBIterationLLMReviewProposals,
  getKBIterationLLMReviewReport,
  getKBIterationLLMReviewTrace,
  runKBIterationLLMReview,
```

```ts
import {
  LLMJudgePanel,
  LLMReviewPanel,
  PatchCandidatesPanel
} from '@/components/kg-maintenance/LLMReviewPanels'
```

Add state:

```ts
  const [llmTrace, setLlmTrace] = useState<Record<string, any> | null>(null)
  const [llmReport, setLlmReport] = useState('')
  const [llmProposals, setLlmProposals] = useState('')
  const [llmJudgeReport, setLlmJudgeReport] = useState('')
  const [patchText, setPatchText] = useState('')
  const [llmRunning, setLlmRunning] = useState(false)
```

- [ ] **Step 3: Load LLM review artifacts with graceful 404 handling**

Add helper:

```ts
const optionalArtifactContent = async (
  loader: () => Promise<Awaited<ReturnType<typeof getKBIterationArtifact>>>
) => {
  try {
    const artifact = await loader()
    if ('content' in artifact) return artifact.content
    if ('payload' in artifact) return artifact.payload
    return ''
  } catch {
    return ''
  }
}
```

Inside `loadWorkspaceData`, after existing artifact loads, add:

```ts
      const [traceArtifact, reportArtifact, proposalsArtifact, judgeArtifact] =
        await Promise.all([
          optionalArtifactContent(() => getKBIterationLLMReviewTrace(selectedWorkspace)),
          optionalArtifactContent(() => getKBIterationLLMReviewReport(selectedWorkspace)),
          optionalArtifactContent(() => getKBIterationLLMReviewProposals(selectedWorkspace)),
          optionalArtifactContent(() => getKBIterationLLMJudgeReport(selectedWorkspace))
        ])
      setLlmTrace(typeof traceArtifact === 'object' ? traceArtifact : null)
      setLlmReport(typeof reportArtifact === 'string' ? reportArtifact : '')
      setLlmProposals(typeof proposalsArtifact === 'string' ? proposalsArtifact : '')
      setLlmJudgeReport(typeof judgeArtifact === 'string' ? judgeArtifact : '')
```

- [ ] **Step 4: Add run and patch handlers**

Add:

```ts
  const handleRunLLMReview = useCallback(async () => {
    if (!selectedWorkspace) return
    setLlmRunning(true)
    setError(null)
    try {
      await runKBIterationLLMReview(selectedWorkspace, {
        profile: activeProfile,
        max_review_rounds: 4,
        max_focus_items_per_round: 3,
        allow_llm_judge: true,
        allow_llm_auto_accept: false,
        allow_low_risk_auto_reject: true,
        generate_patch_candidates: false,
        require_human_for_mutation: true
      })
      await loadWorkspaceData()
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setLlmRunning(false)
    }
  }, [activeProfile, loadWorkspaceData, selectedWorkspace])

  const handleLoadPatch = useCallback(
    async (proposalId: string) => {
      if (!selectedWorkspace) return
      setError(null)
      try {
        const artifact = await getKBIterationLLMReviewPatch(selectedWorkspace, proposalId)
        setPatchText('content' in artifact ? artifact.content : '')
      } catch (err) {
        setError(errorMessage(err))
      }
    },
    [selectedWorkspace]
  )
```

- [ ] **Step 5: Extend `MainPanelProps` and rendering**

Add props:

```ts
  llmTrace: Record<string, any> | null
  llmReport: string
  llmProposals: string
  llmJudgeReport: string
  patchText: string
  llmRunning: boolean
  onRunLLMReview: () => void
  onLoadPatch: (proposalId: string) => void
```

Pass props from `KGMaintenanceConsole` into `MainPanel`.

Add branches in `MainPanel`:

```tsx
  if (activeSection === 'llm-review') {
    return (
      <LLMReviewPanel
        trace={llmTrace}
        report={llmReport}
        proposals={llmProposals}
        running={llmRunning}
        onRun={onRunLLMReview}
      />
    )
  }
  if (activeSection === 'patches') {
    return (
      <PatchCandidatesPanel
        proposals={llmProposals}
        patchText={patchText}
        onLoadPatch={onLoadPatch}
      />
    )
  }
  if (activeSection === 'judge') {
    return <LLMJudgePanel report={llmJudgeReport} />
  }
```

- [ ] **Step 6: Run focused WebUI tests**

Run:

```powershell
cd lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/KGMaintenanceShell.test.tsx
```

Expected: pass.

- [ ] **Step 7: Commit**

```powershell
git add lightrag_webui\src\features\KGMaintenanceConsole.tsx lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.test.tsx
git commit -m "feat: wire kg llm review console"
```

---

## Task 12: Verification And Documentation Update

**Files:**
- Modify: `docs/KBIterationAgent.md`
- Modify: `docs/KBIterationAgent-zh.md`
- Optionally modify: `docs/superpowers/specs/2026-06-18-kg-maintenance-llm-review-loop-design.md` only if implementation details changed materially.

- [ ] **Step 1: Update English docs**

In `docs/KBIterationAgent.md`, add a section after "First Deterministic Run":

```markdown
## LLM Review Loop

After deterministic artifacts exist, maintainers can run the optional LLM review loop. The loop reads the snapshot, quality score, Markdown memory, rule memory, accepted changes, and rejected changes. It builds focused context packages under `review_context/`, writes `llm_review_trace.json`, writes `llm_review_report.md`, writes `proposals.generated.yaml`, and updates the approval queue through the existing proposal validator.

The LLM review loop does not apply patches, mutate KG facts, edit prompts or rules, or rebuild a workspace. LLM output is analysis and proposal material only. Mutation proposals still require approval.
```

- [ ] **Step 2: Update Chinese docs**

In `docs/KBIterationAgent-zh.md`, add the equivalent Chinese section. Use UTF-8 aware editor or PowerShell configured for UTF-8; do not trust mojibake terminal display.

Suggested text:

```markdown
## LLM 审阅循环

确定性审阅产物生成后，维护者可以运行可选的 LLM 审阅循环。该循环读取快照、质量评分、Markdown 记忆、规则记忆、已接受变更和已拒绝变更；随后在 `review_context/` 下生成聚焦上下文，写入 `llm_review_trace.json`、`llm_review_report.md`、`proposals.generated.yaml`，并通过现有 proposal 校验器更新审批队列。

LLM 审阅循环不会应用 patch，不会修改 KG 事实，不会编辑提示词或规则，也不会重建 workspace。LLM 输出只作为分析和建议材料；所有 mutation proposal 仍然必须经过审批。
```

- [ ] **Step 3: Run backend focused tests**

Run:

```powershell
uv run pytest tests\kg\test_kb_iteration_llm_review.py tests\kg\test_kb_iteration_review_context.py tests\kg\test_kb_iteration_review_loop.py tests\kg\test_kb_iteration_patches.py tests\kg\test_kb_iteration_proposals.py tests\api\routes\test_kb_iteration_routes.py -q
```

Expected: all pass.

- [ ] **Step 4: Run frontend focused tests**

Run:

```powershell
cd lightrag_webui
npx --yes bun test src/api/lightrag.test.ts src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/KGMaintenanceShell.test.tsx src/components/kg-maintenance/KGMaintenanceShell.test.tsx src/components/kg-maintenance/kgMaintenanceGraph.test.ts src/stores/kgMaintenance.test.ts
```

Expected: all pass.

- [ ] **Step 5: Run lint/build**

Run:

```powershell
cd lightrag_webui
npx --yes bun run lint
npx --yes bun run build
```

Expected: lint and build pass.

- [ ] **Step 6: Commit docs and final integration**

```powershell
git add docs\KBIterationAgent.md docs\KBIterationAgent-zh.md
git commit -m "docs: describe kb llm review loop"
```

- [ ] **Step 7: Final status check**

Run:

```powershell
git status --short
```

Expected: only unrelated pre-existing user changes remain, or the worktree is clean if those were handled separately.

---

## Full Verification Before Completion

Run backend focused suite:

```powershell
uv run pytest tests\kg\test_kb_iteration_snapshot.py tests\kg\test_kb_iteration_markdown.py tests\kg\test_kb_iteration_quality.py tests\kg\test_kb_iteration_proposals.py tests\kg\test_kb_iteration_diff.py tests\kg\test_kb_iteration_runner.py tests\kg\test_kb_iteration_llm_review.py tests\kg\test_kb_iteration_review_context.py tests\kg\test_kb_iteration_review_loop.py tests\kg\test_kb_iteration_patches.py tests\api\routes\test_kb_iteration_routes.py -q
```

Run frontend focused suite:

```powershell
cd lightrag_webui
npx --yes bun test src/api/lightrag.test.ts src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/KGMaintenanceShell.test.tsx src/components/kg-maintenance/MedicalHierarchyGraph.test.tsx src/components/kg-maintenance/kgMaintenanceGraph.test.ts src/components/kg-maintenance/kgMaintenanceData.test.ts src/stores/kgMaintenance.test.ts
```

Run frontend quality gates:

```powershell
cd lightrag_webui
npx --yes bun run lint
npx --yes bun run build
```

Do not claim full `npx --yes bun test` is green unless it has been rerun and the known unrelated graph/settings baseline failures have been resolved.

## Self-Review Notes

- Spec coverage: backend loop, context, parsing, proposals, patch candidates, Judge hooks, API, WebUI, errors, tests, and docs are covered.
- Deliberate MVP limit: real provider wiring is injectable through `LLMReviewClient`; the fallback API client returns 503 until a configured client is added. This keeps the first implementation testable and safe.
- No automatic mutation path is planned.
- No apply-patch API or rebuild API is included.
- High-risk mutation proposals remain approval-gated.
