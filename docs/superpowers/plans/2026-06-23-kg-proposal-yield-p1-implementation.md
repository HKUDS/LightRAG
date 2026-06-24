# KG Proposal Yield P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise KG iteration Agent proposal yield without duplicate generation, silent drops, or unsafe queued proposals.

**Architecture:** Use one raw issue collection pass to build a canonical issue ledger. The deterministic layer scans that ledger, records a single route state for every issue, arbitrates candidate duplicates/conflicts, and only then converts eligible candidates to Proposals. The LLM layer is built from `llm_residual` issue routes only, so deterministic-covered issues cannot be proposed twice.

**Tech Stack:** Python 3.11+, pytest, ruff, LightRAG KG iteration modules, FastAPI route models, React TypeScript WebUI, Bun tests when available.

---

## Review Adjustments Accepted

This revision intentionally replaces the previous Task 2 design. Do not implement the old plan that called `build_proposal_task_packs()` twice with different `max_packs`.

Accepted corrections:

- Use a single raw issue collection pass, not two task-pack builds.
- Add a formal Raw Issue Contract before expanding generators.
- Introduce an issue ledger keyed by stable `issue_ref`.
- Route every issue into exactly one main state: `deterministic_covered`, `llm_residual`, `blocked_safety`, `blocked_apply`, `blocked_evidence`, `deferred_budget`, `duplicate`, or `stale`.
- Build LLM task packs only from `llm_residual`.
- Compute funnel metrics from the ledger, not from task-pack rows.
- Use an Apply handler registry as the single source of apply capability truth.
- Add candidate arbitration before Proposal conversion.
- Use one family budget configuration, not two overlapping caps.
- Treat `selected proposals >= 50` as an observation target only when at least 50 eligible safe issues exist.

## Baseline

Latest real run on `influenza_medical_v1`:

- raw issues: `767`
- fitted issues: `47`
- selected proposals: `11`
- Judge recommend accept: `8`
- needs human: `3`
- invalid subagent outputs: `4`
- after P0 closure: approval queue is empty, 8 accepted proposals applied, 0 blocked

Current bottleneck:

- raw issue to executable candidate coverage is too low;
- deterministic generation is still conceptually tied to task packs;
- LLM subagents can still see issues already covered by deterministic candidates;
- current reporting cannot prove full raw issue accounting.

## Quality Gates

Hard gates for P1:

- raw issue accounting rate = `100%`;
- schema-invalid queued proposals = `0`;
- apply-unsupported queued proposals = `0`;
- stale or previously applied duplicate proposals = `0`;
- silent candidate drops = `0`;
- candidate-to-proposal unexplained failures = `0`;
- LLM residual packs contain only issues routed as `llm_residual`.

Observation targets:

- if eligible safe issues >= `50`, selected proposals should be >= `50`;
- candidate-to-proposal pass rate is computed only over deduped, schema-valid, apply-supported, in-budget candidates.

Excluded from that pass-rate denominator:

- duplicate candidates;
- same-edge conflict candidates;
- apply-unsupported candidates;
- family-budget deferred candidates;
- safety-blocked candidates.

## File Map

Create:

- `lightrag/kb_iteration/issue_ledger.py`
- `lightrag/kb_iteration/proposal_funnel.py`
- `tests/kg/test_kb_iteration_issue_ledger.py`
- `tests/kg/test_kb_iteration_proposal_funnel.py`

Modify:

- `lightrag/kb_iteration/quality.py`
- `lightrag/kb_iteration/proposal_orchestrator.py`
- `lightrag/kb_iteration/agent_pipeline.py`
- `lightrag/kb_iteration/apply.py`
- `lightrag/kb_iteration/proposals.py`
- `lightrag/kb_iteration/deterministic_proposals/base.py`
- `lightrag/kb_iteration/deterministic_proposals/clinical_modeling.py`
- `lightrag/kb_iteration/deterministic_proposals/diagnosis.py`
- `lightrag/kb_iteration/deterministic_proposals/treatment.py`
- `lightrag/kb_iteration/deterministic_proposals/risk_safety.py`
- `lightrag/kb_iteration/deterministic_proposals/prevention.py`
- `lightrag/kb_iteration/deterministic_proposals/entity_cleanup.py`
- `lightrag/api/routers/kb_iteration_routes.py`
- `lightrag_webui/src/api/lightrag.ts`
- `lightrag_webui/src/components/kg-maintenance/kgMaintenanceLLMReviewRequest.ts`
- `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx`
- `docs/KBIterationAgent.md`
- `docs/KBIterationAgent-zh.md`

Test:

- `tests/kg/test_kb_iteration_quality.py`
- `tests/kg/test_kb_iteration_issue_ledger.py`
- `tests/kg/test_kb_iteration_proposal_orchestrator.py`
- `tests/kg/test_kb_iteration_agent_pipeline.py`
- `tests/kg/test_kb_iteration_apply.py`
- `tests/kg/test_kb_iteration_proposal_funnel.py`
- `tests/api/routes/test_kb_iteration_routes.py`
- WebUI touched tests under `lightrag_webui/src/components/kg-maintenance/`

---

### Task 1: Raw Issue Contract

**Files:**
- Modify: `lightrag/kb_iteration/quality.py`
- Create: `lightrag/kb_iteration/issue_ledger.py`
- Modify: `tests/kg/test_kb_iteration_quality.py`
- Create: `tests/kg/test_kb_iteration_issue_ledger.py`

- [ ] **Step 1: Write quality contract regression tests**

Add tests proving structured issues emitted by `evaluate_snapshot_quality()` include the fields used by deterministic generators.

```python
def test_quality_schema_issue_includes_raw_issue_contract_fields() -> None:
    snapshot = _snapshot_with_reversed_diagnostic_test_edge()

    score = evaluate_snapshot_quality(snapshot)

    issue = score.details["medical_schema_issues"][0]
    for key in (
        "issue_kind",
        "edge_id",
        "source",
        "source_type",
        "target",
        "target_type",
        "keywords",
        "qualifiers",
        "candidate_predicates",
        "repair_options",
        "source_id",
        "file_path",
        "evidence_quote",
    ):
        assert key in issue
```

Add a cleanup issue test:

```python
def test_quality_entity_cleanup_issue_includes_contract_fields() -> None:
    snapshot = _snapshot_with_value_node_cleanup_issue()

    score = evaluate_snapshot_quality(snapshot)

    issue = score.details["entity_cleanup_issues"][0]
    for key in (
        "issue_kind",
        "node_id",
        "candidate_predicates",
        "repair_options",
        "source_id",
        "file_path",
        "evidence_quote",
    ):
        assert key in issue
```

- [ ] **Step 2: Add raw issue normalization types**

Create `issue_ledger.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

IssueRouteState = Literal[
    "unrouted",
    "deterministic_covered",
    "llm_residual",
    "blocked_safety",
    "blocked_apply",
    "blocked_evidence",
    "deferred_budget",
    "duplicate",
    "stale",
]


@dataclass
class IssueRoute:
    issue_ref: str
    family: str
    issue_kind: str
    route_state: IssueRouteState = "unrouted"
    candidate_ids: list[str] = field(default_factory=list)
    proposal_ids: list[str] = field(default_factory=list)
    reason_code: str = ""
    reason: str = ""


@dataclass
class DeterministicScanResult:
    issues: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    rejections: list[dict[str, Any]]
    issue_routes: list[IssueRoute]
```

- [ ] **Step 3: Implement `normalize_raw_issues()`**

In `issue_ledger.py`, implement:

```python
def normalize_raw_issues(quality: dict[str, Any]) -> list[dict[str, Any]]:
    details = quality.get("details") if isinstance(quality, dict) else {}
    structured: list[dict[str, Any]] = []
    if isinstance(details, dict):
        for issue_source in ("medical_schema_issues", "entity_cleanup_issues"):
            for index, issue in enumerate(_dict_items(details.get(issue_source))):
                structured.append(
                    _normalize_issue(issue, issue_source=issue_source, issue_order=index)
                )
    return _dedupe_issues(structured) if structured else _fallback_quality_findings(quality)
```

The normalized issue must always contain:

```python
{
    "issue_ref": str,
    "issue_source": str,
    "issue_family": str,
    "issue_kind": str,
    "issue_order": int,
    "edge_id": str,
    "source": str,
    "source_type": str,
    "target": str,
    "target_type": str,
    "keywords": str,
    "qualifiers": dict,
    "candidate_predicates": list[str],
    "repair_options": list[dict],
    "suggested_qualifiers": dict,
    "source_id": str,
    "file_path": str,
    "evidence_quote": str,
    "auto_fixable": bool,
    "blocked_reason": str,
}
```

- [ ] **Step 4: Run tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_quality.py tests/kg/test_kb_iteration_issue_ledger.py -q
```

Expected: pass after implementation.

---

### Task 2: RED Tests For Ledger Routing And No Duplicate Downstream Work

**Files:**
- Create: `tests/kg/test_kb_iteration_issue_ledger.py`
- Modify: `tests/kg/test_kb_iteration_agent_pipeline.py`
- Modify: `tests/kg/test_kb_iteration_proposal_orchestrator.py`

- [ ] **Step 1: Add accounting invariant test**

```python
def test_issue_ledger_routes_every_raw_issue_once(tmp_path: Path) -> None:
    package = _make_package_with_mixed_quality_issues(tmp_path)

    result = scan_deterministic_candidates(package)
    counts = route_state_counts(result.issue_routes)

    assert len(result.issues) == (
        counts["deterministic_covered"]
        + counts["llm_residual"]
        + counts["blocked_safety"]
        + counts["blocked_apply"]
        + counts["blocked_evidence"]
        + counts["deferred_budget"]
        + counts["duplicate"]
        + counts["stale"]
    )
```

- [ ] **Step 2: Add LLM residual-only task-pack test**

```python
def test_llm_task_packs_are_built_only_from_residual_issue_routes(tmp_path: Path) -> None:
    package = _make_package_with_one_safe_deterministic_and_one_residual_issue(tmp_path)

    scan = scan_deterministic_candidates(package)
    packs = build_llm_residual_task_packs(package, scan)

    residual_refs = {
        route.issue_ref
        for route in scan.issue_routes
        if route.route_state == "llm_residual"
    }
    packed_refs = {
        issue["issue_ref"]
        for pack in packs
        for issue in pack.issues
    }
    assert packed_refs == residual_refs
```

- [ ] **Step 3: Add full-scan stale/decision-memory tests**

Add these tests:

```python
def test_full_scan_does_not_requeue_already_applied_action(tmp_path: Path) -> None: ...
def test_full_scan_does_not_repeat_rejected_action_fingerprint(tmp_path: Path) -> None: ...
def test_full_scan_drops_stale_expected_keywords(tmp_path: Path) -> None: ...
def test_same_edge_conflicting_candidates_enter_conflict_ledger(tmp_path: Path) -> None: ...
```

Each test must assert a route state, not just proposal absence:

- already applied: `stale` or `duplicate`;
- rejected action fingerprint: `duplicate`;
- stale expected keywords: `stale`;
- same-edge conflict: `llm_residual` with `reason_code="SAME_EDGE_CONFLICT"`.

- [ ] **Step 4: Run RED tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_issue_ledger.py tests/kg/test_kb_iteration_agent_pipeline.py::test_llm_task_packs_are_built_only_from_residual_issue_routes -q
```

Expected: fail before implementation.

---

### Task 3: Single Collection Deterministic Scan

**Files:**
- Modify: `lightrag/kb_iteration/issue_ledger.py`
- Modify: `lightrag/kb_iteration/proposal_orchestrator.py`
- Modify: `lightrag/kb_iteration/agent_pipeline.py`
- Modify: `tests/kg/test_kb_iteration_issue_ledger.py`
- Modify: `tests/kg/test_kb_iteration_proposal_orchestrator.py`

- [ ] **Step 1: Move raw issue collection to `issue_ledger.py`**

Keep compatibility wrappers in `proposal_orchestrator.py`, but make the ledger the owner of raw issue collection.

```python
def collect_raw_issues(package_dir: str | Path) -> list[dict[str, Any]]:
    package = Path(package_dir)
    quality = _read_json(package / "snapshots" / "quality_score.json")
    return normalize_raw_issues(quality)
```

- [ ] **Step 2: Implement deterministic scan**

```python
def scan_deterministic_candidates(
    package_dir: str | Path,
    *,
    prevalidate_action_candidates: bool = True,
) -> DeterministicScanResult:
    issues = collect_raw_issues(package_dir)
    snapshot = _read_json(Path(package_dir) / "snapshots" / "kg_snapshot.json")
    candidates, rejections = _generate_candidates_for_issues(issues, snapshot)
    candidates, rejections, routes = arbitrate_candidates(
        issues=issues,
        candidates=candidates,
        rejections=rejections,
        package_dir=package_dir,
        prevalidate_action_candidates=prevalidate_action_candidates,
    )
    return DeterministicScanResult(
        issues=issues,
        candidates=candidates,
        rejections=rejections,
        issue_routes=routes,
    )
```

Do not call `build_proposal_task_packs()` from this scan.

- [ ] **Step 3: Keep `ProposalTaskPack` LLM-facing**

Refactor `build_proposal_task_packs()` into:

```python
def build_llm_residual_task_packs(
    package_dir: str | Path,
    scan: DeterministicScanResult,
    *,
    max_issues_per_pack: int = 50,
    max_packs: int = 20,
    require_candidate_evidence_allowlist: bool = True,
) -> list[ProposalTaskPack]:
    residual_refs = {
        route.issue_ref
        for route in scan.issue_routes
        if route.route_state == "llm_residual"
    }
    residual_issues = [
        issue for issue in scan.issues if issue["issue_ref"] in residual_refs
    ]
    return _quality_or_structured_task_packs_from_issues(...)
```

Use the existing pack splitting/fitting code only after this residual filtering.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_issue_ledger.py tests/kg/test_kb_iteration_proposal_orchestrator.py -q
```

Expected: pass.

---

### Task 4: Candidate Arbitration And Apply Capability Registry

**Files:**
- Modify: `lightrag/kb_iteration/issue_ledger.py`
- Modify: `lightrag/kb_iteration/apply.py`
- Modify: `lightrag/kb_iteration/agent_pipeline.py`
- Modify: `lightrag/kb_iteration/proposal_orchestrator.py`
- Modify: `tests/kg/test_kb_iteration_apply.py`
- Modify: `tests/kg/test_kb_iteration_issue_ledger.py`
- Modify: `tests/kg/test_kb_iteration_agent_pipeline.py`

- [ ] **Step 1: Add Apply handler registry**

In `apply.py`, define an internal dispatch table and reuse it for capability checks.

```python
APPLY_HANDLERS: dict[tuple[str, str], str] = {
    ("medical_relation_schema_migration", "replace_relation"): "_apply_medical_relation_schema_migration",
    ("medical_relation_schema_migration", "retire_relation"): "_apply_medical_relation_schema_migration",
    ("medical_fact_role_split", "split_relation"): "_apply_medical_fact_role_split",
    ("value_node_to_qualifier", "value_node_to_qualifier"): "_apply_value_node_to_qualifier",
    ("candidate_kg_expansion", "candidate_kg_expansion"): "_apply_candidate_kg_expansion",
}
```

Use function names or callable wrappers consistently with the current async apply flow. The key requirement is one table, not a second hardcoded list.

- [ ] **Step 2: Implement capability helper from the registry**

```python
@dataclass(frozen=True)
class ProposalApplyCapability:
    supported: bool
    action: str
    reason: str = ""


def proposal_apply_capability(proposal: ImprovementProposal) -> ProposalApplyCapability:
    action = str(proposal.action_payload.get("action") or proposal.type).strip()
    key = (proposal.type, action)
    if key in APPLY_HANDLERS:
        return ProposalApplyCapability(True, action)
    return ProposalApplyCapability(
        False,
        action,
        f"Apply Engine does not support {proposal.type}:{action}.",
    )
```

Static capability is checked after candidate generation and before Judge. Dynamic edge/node preconditions still belong in real Apply.

- [ ] **Step 3: Add arbitration pipeline**

The scan must run:

```text
Candidate generation
  -> exact fingerprint dedupe
  -> same-edge conflict arbitration
  -> schema validation
  -> apply capability
  -> family budget routing
  -> Proposal conversion
```

Rules:

- identical payload: merge `issue_refs` and evidence;
- same edge with different payload: route affected issues to `llm_residual` with `reason_code="SAME_EDGE_CONFLICT"`;
- multiple legal medical predicates with no deterministic winner: `llm_residual`;
- explicit safety block: `blocked_safety`, not LLM guessing;
- apply unsupported: `blocked_apply`;
- family cap reached: `deferred_budget`, not generator rejection.

- [ ] **Step 4: Preserve stale and decision-memory gates**

Full scan candidates must pass the existing stale relation and decision-memory logic. Add ledger-level helpers instead of only filtering after merge:

```python
def candidate_is_stale(candidate: dict[str, Any], snapshot_edges: list[dict[str, Any]]) -> str: ...
def candidate_fingerprint(candidate: dict[str, Any]) -> str: ...
def decision_memory_fingerprints(package_dir: str | Path) -> set[str]: ...
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_apply.py tests/kg/test_kb_iteration_issue_ledger.py tests/kg/test_kb_iteration_agent_pipeline.py -q
```

Expected: pass.

---

### Task 5: Candidate-To-Proposal From Ledger Only

**Files:**
- Modify: `lightrag/kb_iteration/agent_pipeline.py`
- Modify: `lightrag/kb_iteration/issue_ledger.py`
- Modify: `tests/kg/test_kb_iteration_agent_pipeline.py`
- Modify: `tests/kg/test_kb_iteration_issue_ledger.py`

- [ ] **Step 1: Change action-candidate conversion input**

Replace task-pack based conversion with scan-result based conversion.

```python
def action_candidate_proposals_from_scan(
    scan: DeterministicScanResult,
    *,
    output_dir: Path,
    previous_outputs: dict[str, Any],
) -> ActionCandidateProposalBuildResult:
    ...
```

Every `ValueError` must update the corresponding issue route and appear in `scan.rejections`.

- [ ] **Step 2: Add proposal IDs back to issue routes**

When a candidate becomes a Proposal:

```python
route.route_state = "deterministic_covered"
route.proposal_ids.append(proposal.id)
```

If conversion fails:

```python
route.route_state = "blocked_safety" | "blocked_apply" | "blocked_evidence"
route.reason_code = parsed_error_code
```

- [ ] **Step 3: Run tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_agent_pipeline.py tests/kg/test_kb_iteration_issue_ledger.py -q
```

Expected: pass.

---

### Task 6: Funnel Report From Issue Ledger

**Files:**
- Create: `lightrag/kb_iteration/proposal_funnel.py`
- Modify: `lightrag/kb_iteration/agent_pipeline.py`
- Create: `tests/kg/test_kb_iteration_proposal_funnel.py`

- [ ] **Step 1: Build report from issue routes**

Create:

```python
def build_proposal_funnel_report(
    scan: DeterministicScanResult,
    *,
    selected_proposal_ids: list[str],
    dropped: list[dict[str, Any]],
) -> dict[str, Any]:
    ...
```

Per-family metrics must distinguish:

- `raw_issue_count`;
- `issue_with_candidate_count`;
- `action_candidate_count`;
- `deterministic_covered_count`;
- `llm_residual_count`;
- `blocked_safety_count`;
- `blocked_apply_count`;
- `blocked_evidence_count`;
- `deferred_budget_count`;
- `duplicate_count`;
- `stale_count`;
- `selected_proposal_count`;
- `reason_code_counts`.

- [ ] **Step 2: Enforce accounting invariant**

The report builder should include:

```python
report["summary"]["accounting_balanced"] = raw_issue_count == routed_issue_count
report["summary"]["unrouted_issue_count"] = ...
```

Tests must assert `accounting_balanced is True`.

- [ ] **Step 3: Write artifacts**

Write:

- `issue_ledger.json`;
- `deterministic_proposal_report.json`;
- `deterministic_proposal_report.md`.

- [ ] **Step 4: Run tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_proposal_funnel.py tests/kg/test_kb_iteration_agent_pipeline.py -q
```

Expected: pass.

---

### Task 7: Expand High-Yield Generators Against Real Quality Output

**Files:**
- Modify: `lightrag/kb_iteration/deterministic_proposals/base.py`
- Modify: `lightrag/kb_iteration/deterministic_proposals/clinical_modeling.py`
- Modify: `lightrag/kb_iteration/deterministic_proposals/diagnosis.py`
- Modify: `lightrag/kb_iteration/deterministic_proposals/treatment.py`
- Modify: `lightrag/kb_iteration/deterministic_proposals/risk_safety.py`
- Modify: `lightrag/kb_iteration/deterministic_proposals/prevention.py`
- Modify: `lightrag/kb_iteration/deterministic_proposals/entity_cleanup.py`
- Modify: `tests/kg/test_kb_iteration_quality.py`
- Modify: `tests/kg/test_kb_iteration_issue_ledger.py`

- [ ] **Step 1: Add end-to-end generator tests**

Use this real path, not hand-written issue dictionaries only:

```text
KGSnapshot
-> evaluate_snapshot_quality()
-> normalize_raw_issues()
-> scan_deterministic_candidates()
-> validate_proposal()
```

- [ ] **Step 2: Add shared qualifier merge helper**

```python
def merged_qualifier_repair(issue: Mapping[str, Any]) -> dict[str, Any]:
    qualifiers = issue_qualifiers(issue)
    for key in ("suggested_qualifiers", "qualifier_repairs", "missing_qualifiers"):
        value = issue.get(key)
        if isinstance(value, Mapping):
            qualifiers.update(value)
    return qualifiers
```

- [ ] **Step 3: Expand diagnosis**

Cover:

- finding/sign to disease: `has_diagnostic_criterion`;
- disease to test: `orders_test` with `indication`;
- test/result pattern to disease: `supports_or_refutes` only when `polarity` exists.

- [ ] **Step 4: Expand treatment**

Cover:

- drug/procedure to disease: `has_indication`;
- guideline/recommendation to intervention: `recommends`;
- intervention to population: `recommended_for` only with `purpose` and scope;
- safety predicates require one unambiguous candidate and required reason qualifiers.

- [ ] **Step 5: Expand risk/safety**

Cover:

- population to disease: `high_risk_for`;
- risk factor to disease/outcome: `risk_factor_for`;
- explicit risk increase: `increases_risk_of`;
- chronic underlying disease must not become `has_complication`;
- severity/outcome nodes are rejected with precise reason codes.

- [ ] **Step 6: Expand prevention**

Cover:

- vaccine/public-health measure to disease: `targets_disease`;
- prevention to outcome: `reduces_risk_of` only with population/context when scoped;
- prevention `recommended_for` always sets `purpose=prevention`.

- [ ] **Step 7: Keep entity merge blocked but visible**

For `synonym_duplicate`, `alias_duplicate`, and `near_duplicate_entity`, route to `blocked_apply` with:

```python
reason_code = "ENTITY_MERGE_APPLY_NOT_SUPPORTED"
```

- [ ] **Step 8: Run tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_quality.py tests/kg/test_kb_iteration_issue_ledger.py tests/kg/test_kb_iteration_proposal_orchestrator.py -q
```

Expected: pass.

---

### Task 8: Family Budgets And Stable Ordering

**Files:**
- Modify: `lightrag/kb_iteration/issue_ledger.py`
- Modify: `lightrag/kb_iteration/agent_pipeline.py`
- Modify: `lightrag/api/routers/kb_iteration_routes.py`
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Modify: `lightrag_webui/src/components/kg-maintenance/kgMaintenanceLLMReviewRequest.ts`
- Modify: `tests/kg/test_kb_iteration_issue_ledger.py`
- Modify: `tests/api/routes/test_kb_iteration_routes.py`

- [ ] **Step 1: Use one budget config**

Add:

```python
DEFAULT_DETERMINISTIC_FAMILY_CAPS = {
    "diagnosis": 40,
    "treatment": 40,
    "risk_safety": 35,
    "prevention": 30,
    "clinical_modeling": 30,
    "entity_cleanup": 25,
    "legacy_schema": 20,
}
```

Do not add `max_deterministic_candidates_per_family`. The only family budget input is `deterministic_family_caps`.

- [ ] **Step 2: Apply cap after dedupe and validation**

`FAMILY_CAP_REACHED` routes to `deferred_budget`, not generator rejection.

- [ ] **Step 3: Stable ordering**

Sort candidate proposals by:

1. family priority;
2. mutation risk ascending: `low -> medium -> high`;
3. `issue_order`;
4. `candidate_id`.

- [ ] **Step 4: API/Web request shape**

Add optional request field:

```python
deterministic_family_caps: dict[str, int] | None
```

Validate each cap is `1..500`.

- [ ] **Step 5: Run tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_issue_ledger.py tests/api/routes/test_kb_iteration_routes.py -q
```

Expected: pass.

---

### Task 9: API/Web Funnel Display

**Files:**
- Modify: `lightrag/api/routers/kb_iteration_routes.py`
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Modify: `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx`
- Modify: `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.test.tsx`
- Modify: `docs/KBIterationAgent.md`
- Modify: `docs/KBIterationAgent-zh.md`

- [ ] **Step 1: Expose artifacts**

Add artifact keys:

- `issue_ledger`: `issue_ledger.json`;
- `deterministic_proposal_report`: `deterministic_proposal_report.json`;
- `deterministic_proposal_report_md`: `deterministic_proposal_report.md`.

- [ ] **Step 2: Render Chinese family funnel table**

Columns:

- family label in Chinese;
- raw issues;
- issue with candidate;
- action candidates;
- deterministic covered;
- LLM residual;
- blocked;
- deferred;
- selected proposals;
- top reason code.

- [ ] **Step 3: Update docs**

Explain:

- raw issue = detected defect;
- issue route = exactly one main fate per issue;
- deterministic covered = safe candidate became queueable Proposal;
- LLM residual = only the leftover reasoning set;
- blocked/deferred states are safety gates, not transport failures.

- [ ] **Step 4: Run Web tests**

Run:

```powershell
cd lightrag_webui
bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx
```

If Bun is unavailable on PATH, record that and run the available ESLint command for touched files.

---

### Task 10: Validation Run Sequence

**Files:**
- Runtime artifacts under `work/kb-iteration/influenza_medical_v1/`
- Update `task_plan.md`, `findings.md`, `progress.md`, and `docs/KBIterationAgent-zh.md`

- [ ] **Step 1: Backend regression**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_quality.py tests/kg/test_kb_iteration_issue_ledger.py tests/kg/test_kb_iteration_proposal_orchestrator.py tests/kg/test_kb_iteration_agent_pipeline.py tests/kg/test_kb_iteration_apply.py tests/kg/test_kb_iteration_proposal_funnel.py tests/api/routes/test_kb_iteration_routes.py -q
```

Expected: pass.

- [ ] **Step 2: Ruff**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check lightrag/kb_iteration tests/kg/test_kb_iteration_quality.py tests/kg/test_kb_iteration_issue_ledger.py tests/kg/test_kb_iteration_proposal_orchestrator.py tests/kg/test_kb_iteration_agent_pipeline.py tests/kg/test_kb_iteration_apply.py tests/kg/test_kb_iteration_proposal_funnel.py tests/api/routes/test_kb_iteration_routes.py
```

Expected: pass.

- [ ] **Step 3: Deterministic-only validation**

Run one pipeline pass with deterministic scan and LLM subagent calls disabled. Do not apply proposals.

Validate:

- `issue_ledger.json` exists;
- `deterministic_proposal_report.json` exists;
- raw issue accounting is balanced;
- no schema-invalid queued proposals;
- no apply-unsupported queued proposals;
- no stale or previously applied duplicates.

- [ ] **Step 4: Flash + 200 validation**

Run one real Agent pass with current Flash model and `max_proposals_per_run=200`. Do not apply proposals.

Validate:

- LLM task packs contain only `llm_residual`;
- no `review_context_request` proposals;
- no accepted/previously applied stale duplicates;
- no Apply Engine unsupported proposals;
- if eligible safe issues >= 50, selected proposals >= 50.

- [ ] **Step 5: Document results**

Record:

- raw issues;
- route-state counts;
- issue-with-candidate count;
- action-candidate count;
- selected proposal count;
- action-candidate selected proposals;
- LLM selected proposals;
- top blocked/deferred reason codes;
- stale severe-sign recurrence status.

Update:

- `task_plan.md`;
- `findings.md`;
- `progress.md`;
- `docs/KBIterationAgent-zh.md`.

---

## Implementation Order

1. Raw Issue Contract and issue ledger.
2. RED tests for accounting, residual-only LLM packs, stale/decision-memory gates, and same-edge conflicts.
3. Single full issue collection and deterministic scan.
4. Candidate dedupe, same-edge arbitration, schema validation, and Apply capability registry.
5. Candidate-to-Proposal conversion from scan result only.
6. Funnel JSON/Markdown from issue ledger.
7. High-yield family generator expansion against real `quality.py` output.
8. Family budgets and stable ordering.
9. API/Web display.
10. Deterministic-only validation.
11. Flash + 200 validation.
12. Approval queue inspection only; do not auto-Apply.

## Subagent Execution Notes

This plan is not fully parallel. The pipeline/ledger work owns the shared interfaces and should land first.

Recommended Subagent-Driven split:

1. Ledger owner: Tasks 1-6, because these touch shared route states and pipeline semantics.
2. Generator owner: Task 7, after ledger interfaces are stable.
3. Apply registry owner: Task 4 can proceed after ledger route states are defined, but must merge before Task 5.
4. Web/API owner: Tasks 8-9 after report schema is stable.
5. Validation owner: Task 10 after all preceding work passes focused tests.

## Not In P1

- Do not call `build_proposal_task_packs()` twice to simulate a full scan.
- Do not compute funnel metrics from task-pack rows.
- Do not route deterministic-covered issues into LLM residual packs.
- Do not maintain a second hardcoded apply capability list.
- Do not let entity merge proposals reach approval until deterministic merge apply support exists.
- Do not apply generated proposals automatically during validation.
- Do not optimize for proposal count at the expense of schema validity and apply support.
