# KG Maintenance Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `KG Maintenance` WebUI console from `docs/superpowers/specs/2026-06-17-kb-maintenance-console-design-zh.md`, backed by safe KB iteration APIs.

**Architecture:** Add a new backend router under `/kb-iteration` that exposes only whitelisted KB iteration artifacts and append-only review decisions. Add a new WebUI tab with a three-column product-console layout: workflow navigation, main artifact/graph workspace, and right-side evidence/review inspector. Keep the first implementation source-grounded and conservative: read artifacts, run deterministic reports, and record approvals without applying KG or prompt/rule mutations.

**Tech Stack:** FastAPI, Pydantic, existing LightRAG auth dependencies, `lightrag.kb_iteration`, React 19, TypeScript, Zustand, Tailwind v4, Bun test runner, pytest.

---

## File Structure

### Backend

- Create `lightrag/api/routers/kb_iteration_routes.py`
  - Owns `/kb-iteration` routes, artifact whitelist, workspace validation, safe path resolution, run trigger, summary readers, graph/evidence projections, and append-only approval decision records.
- Modify `lightrag/api/lightrag_server.py`
  - Import and register `create_kb_iteration_routes(rag, args, api_key)`.
- Create `tests/api/routes/test_kb_iteration_routes.py`
  - Tests safe artifact reads, workspace validation, run trigger, summary shape, graph labels, evidence lookup, and approval recording.

### Frontend

- Modify `lightrag_webui/src/api/lightrag.ts`
  - Add KB maintenance response types and API wrappers.
- Modify `lightrag_webui/src/stores/settings.ts`
  - Add `kg-maintenance` to the persisted top-level tab union and migration.
- Create `lightrag_webui/src/stores/kgMaintenance.ts`
  - Store active console section, selected node/edge, selected workspace, and latest run id.
- Modify `lightrag_webui/src/features/SiteHeader.tsx`
  - Add the `KG Maintenance` tab.
- Modify `lightrag_webui/src/App.tsx`
  - Mount `KGMaintenanceConsole` for the new tab.
- Create `lightrag_webui/src/features/KGMaintenanceConsole.tsx`
  - Feature entry point and data loading boundary.
- Create `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.tsx`
  - Three-column console shell and workflow navigation.
- Create `lightrag_webui/src/components/kg-maintenance/KGMaintenanceOverview.tsx`
  - Workspace summary, quality cards, quick actions, and status states.
- Create `lightrag_webui/src/components/kg-maintenance/MedicalHierarchyGraph.tsx`
  - SVG/HTML hierarchy graph panel with node roles, differentiated sizes, relation labels, and legend.
- Create `lightrag_webui/src/components/kg-maintenance/EvidenceInspector.tsx`
  - Right-side details for selected nodes/edges, evidence metadata, and relation direction.
- Create `lightrag_webui/src/components/kg-maintenance/CatalogPanels.tsx`
  - Entity and relation catalog panels.
- Create `lightrag_webui/src/components/kg-maintenance/QualityAndApprovalPanels.tsx`
  - Quality report, approval queue, diff, run log, and rule-memory panels.
- Create `lightrag_webui/src/components/kg-maintenance/kgMaintenanceGraph.ts`
  - Pure graph transformation helpers for tests.
- Create `lightrag_webui/src/components/kg-maintenance/kgMaintenanceGraph.test.ts`
  - Tests role-based node sizing, relation labels, missing keyword fallback, and quality flags.
- Create `lightrag_webui/src/features/KGMaintenanceConsole.test.tsx`
  - Focused rendering and state tests if the existing test setup supports React DOM. If the project has no React component test harness, keep behavior tests in pure helper tests and API wrapper tests.
- Modify `lightrag_webui/src/locales/en.json` and `lightrag_webui/src/locales/zh.json`
  - Add navigation and console labels.

---

## Task 1: Backend Route Tests And Safe Artifact Contract

**Files:**
- Create: `tests/api/routes/test_kb_iteration_routes.py`
- Create: `lightrag/api/routers/kb_iteration_routes.py`
- Modify: `lightrag/api/lightrag_server.py`

- [ ] **Step 1: Write failing route tests**

Create `tests/api/routes/test_kb_iteration_routes.py` with tests that build a temporary artifact workspace and include only the new router:

```python
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient

def _write_json(path, payload):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

def _client(tmp_path):
    from lightrag.api.routers.kb_iteration_routes import create_kb_iteration_routes
    output_root = tmp_path / "work" / "kb-iteration"
    storage_root = tmp_path / "data" / "rag_storage"
    input_root = tmp_path / "data" / "inputs"
    workspace = "influenza_medical_v1"
    package = output_root / workspace
    snapshots = package / "snapshots"
    snapshots.mkdir(parents=True)
    (storage_root / workspace).mkdir(parents=True)
    (input_root / workspace).mkdir(parents=True)
    _write_json(
        snapshots / "kg_snapshot.json",
        {
            "workspace": workspace,
            "generated_at": "2026-06-17T00:00:00Z",
            "source_files": ["flu.pdf"],
            "metadata": {"profile": "clinical_guideline_zh"},
            "nodes": [
                {
                    "id": "流行性感冒",
                    "label": "流行性感冒",
                    "entity_type": "Disease",
                    "description": "core",
                    "source_id": "chunk-1",
                    "file_path": "flu.pdf",
                    "properties": {},
                },
                {
                    "id": "高热不退",
                    "label": "高热不退",
                    "entity_type": "Symptom",
                    "description": "symptom",
                    "source_id": "chunk-1",
                    "file_path": "flu.pdf",
                    "properties": {"medical_group": "clinical_manifestation"},
                },
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "流行性感冒",
                    "target": "高热不退",
                    "keywords": "临床表现",
                    "description": "relation",
                    "source_id": "chunk-1",
                    "file_path": "flu.pdf",
                    "weight": 1.0,
                    "properties": {},
                }
            ],
        },
    )
    _write_json(snapshots / "entity_stats.json", [{"label": "Disease", "count": 1}])
    _write_json(snapshots / "relation_stats.json", [{"label": "临床表现", "count": 1}])
    _write_json(snapshots / "source_coverage.json", {"source_files": ["flu.pdf"]})
    _write_json(
        snapshots / "quality_score.json",
        {
            "overall": 82,
            "subscores": {"evidence_grounding": 91},
            "metrics": {"evidence_coverage": 91, "generic_relation_count": 0},
            "findings": [
                {
                    "severity": "high",
                    "category": "hierarchy",
                    "message": "症状分支缺少中间层",
                    "evidence": ["edge:e1"],
                    "suggested_fix_type": "hierarchy_rule_change",
                    "requires_approval": True,
                }
            ],
            "critical_blockers": [],
        },
    )
    (package / "kb_context.md").write_text("# KB Context\n", encoding="utf-8")
    (package / "quality_report.md").write_text("# Quality\n", encoding="utf-8")
    (package / "approval_queue.md").write_text(
        "# Approval Queue\n\nproposals:\n- id: p1\n  type: hierarchy_rule_change\n  target: kg_structure.md\n  proposed_change: Add symptom layer\n  reason: Improve hierarchy\n  evidence:\n  - edge:e1\n  confidence: 0.8\n  risk: medium\n  requires_approval: true\n  expected_metric_change: {}\n",
        encoding="utf-8",
    )
    (package / "iteration_log.md").write_text(
        "## Run\n\n- workspace: influenza_medical_v1\n- phase: pending_user_review\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        workspace=workspace,
        working_dir=str(storage_root),
        input_dir=str(input_root),
        kb_iteration_output_dir=str(output_root),
    )
    app = FastAPI()
    app.include_router(create_kb_iteration_routes(SimpleNamespace(), args, api_key="test-key"))
    return TestClient(app), package
```

Add tests:

```python
def test_summary_reads_latest_artifacts_without_path_input(tmp_path):
    client, _ = _client(tmp_path)
    response = client.get(
        "/kb-iteration/influenza_medical_v1/summary",
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace"] == "influenza_medical_v1"
    assert payload["latestRunId"] == "latest"
    assert payload["quality"]["overall"] == 82
    assert payload["phase"] == "pending_user_review"
    assert payload["counts"]["nodes"] == 2
    assert payload["pendingApprovalCount"] == 1

def test_artifact_key_is_whitelisted(tmp_path):
    client, _ = _client(tmp_path)
    response = client.get(
        "/kb-iteration/influenza_medical_v1/artifacts/../../.env",
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code in {400, 404}

def test_invalid_workspace_is_rejected(tmp_path):
    client, _ = _client(tmp_path)
    response = client.get(
        "/kb-iteration/../secret/summary",
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code in {400, 404}

def test_graph_exposes_directional_relation_labels(tmp_path):
    client, _ = _client(tmp_path)
    response = client.get(
        "/kb-iteration/influenza_medical_v1/graph",
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    edge = response.json()["edges"][0]
    assert edge["label"] == "临床表现"
    assert edge["direction"] == "outgoing"
    assert "邻接" not in edge["label"]

def test_accept_reject_records_are_append_only(tmp_path):
    client, package = _client(tmp_path)
    response = client.post(
        "/kb-iteration/influenza_medical_v1/proposals/p1/reject",
        headers={"X-API-Key": "test-key"},
        json={"reviewer": "maintainer", "reason": "Needs stronger evidence"},
    )
    assert response.status_code == 200
    text = (package / "rejected_changes.md").read_text(encoding="utf-8")
    assert "p1" in text
    assert "Needs stronger evidence" in text
```

- [ ] **Step 2: Run route tests and verify they fail**

Run:

```powershell
uv run pytest tests\api\routes\test_kb_iteration_routes.py -q
```

Expected: import or 404 failures because `kb_iteration_routes.py` does not exist yet.

- [ ] **Step 3: Implement the route**

Create `lightrag/api/routers/kb_iteration_routes.py` with:

- `ARTIFACTS` whitelist mapping keys to workspace-relative files.
- `_validate_workspace_or_400(workspace)`.
- `_output_root(args)`.
- `_workspace_dir(args, workspace)`.
- JSON/Markdown readers that never accept raw paths.
- Summary, quality, catalog, graph, rules, artifact, run, and decision endpoints.

Use `validate_workspace` before path joins and `get_combined_auth_dependency(api_key)` for auth. `run_id` is `latest` in this first implementation.

- [ ] **Step 4: Register the router**

Modify `lightrag/api/lightrag_server.py`:

```python
from lightrag.api.routers.kb_iteration_routes import create_kb_iteration_routes
```

and inside `create_app()` beside the existing routers:

```python
app.include_router(create_kb_iteration_routes(rag, args, api_key))
```

- [ ] **Step 5: Run backend verification**

Run:

```powershell
uv run pytest tests\api\routes\test_kb_iteration_routes.py -q
uv run pytest tests\kg\test_kb_iteration_runner.py tests\kg\test_kb_iteration_quality.py -q
uv run ruff check lightrag\api\routers\kb_iteration_routes.py tests\api\routes\test_kb_iteration_routes.py
```

Expected: all pass.

- [ ] **Step 6: Commit backend slice**

```powershell
git add lightrag\api\routers\kb_iteration_routes.py lightrag\api\lightrag_server.py tests\api\routes\test_kb_iteration_routes.py
git commit -m "feat: expose kb iteration review api"
```

---

## Task 2: Frontend API Wrappers, Store, And Tab Shell

**Files:**
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Modify: `lightrag_webui/src/stores/settings.ts`
- Create: `lightrag_webui/src/stores/kgMaintenance.ts`
- Modify: `lightrag_webui/src/features/SiteHeader.tsx`
- Modify: `lightrag_webui/src/App.tsx`
- Create: `lightrag_webui/src/features/KGMaintenanceConsole.tsx`
- Create: `lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.tsx`
- Modify: `lightrag_webui/src/locales/en.json`
- Modify: `lightrag_webui/src/locales/zh.json`

- [ ] **Step 1: Write frontend tests first**

Create or update focused tests:

```typescript
// lightrag_webui/src/stores/kgMaintenance.test.ts
import { describe, expect, test } from 'bun:test'
import { useKGMaintenanceStore } from '@/stores/kgMaintenance'

describe('KG maintenance store', () => {
  test('tracks active section and selected graph item', () => {
    useKGMaintenanceStore.getState().setActiveSection('graph')
    useKGMaintenanceStore.getState().setSelectedItem({ kind: 'node', id: '高热不退' })

    expect(useKGMaintenanceStore.getState().activeSection).toBe('graph')
    expect(useKGMaintenanceStore.getState().selectedItem).toEqual({
      kind: 'node',
      id: '高热不退'
    })
  })
})
```

If adding API wrapper tests to `lightrag_webui/src/api/lightrag.test.ts`, preserve existing dirty edits and add only assertions for URL construction.

- [ ] **Step 2: Run frontend tests and verify failure**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src\stores\kgMaintenance.test.ts
```

Expected: FAIL because the store does not exist.

- [ ] **Step 3: Add API wrapper types**

In `lightrag_webui/src/api/lightrag.ts`, add types for:

- `KBIWorkspaceSummary`
- `KBISummaryResponse`
- `KBIQualityResponse`
- `KBIEntityCatalogResponse`
- `KBIRelationCatalogResponse`
- `KBIGraphResponse`
- `KBIRulesResponse`
- `KBIProposalDecisionRequest`

Add wrappers:

```typescript
export const getKBIterationWorkspaces = async () => {
  const response = await axiosInstance.get('/kb-iteration/workspaces')
  return response.data
}

export const getKBIterationSummary = async (workspace: string) => {
  const response = await axiosInstance.get(`/kb-iteration/${encodeURIComponent(workspace)}/summary`)
  return response.data as KBISummaryResponse
}
```

Repeat for quality, graph, catalogs, rules, run trigger, and proposal decisions.

- [ ] **Step 4: Add the store**

Create `lightrag_webui/src/stores/kgMaintenance.ts` with:

```typescript
import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'

export type KGMaintenanceSection =
  | 'overview'
  | 'graph'
  | 'entities'
  | 'relations'
  | 'evidence'
  | 'quality'
  | 'approval'
  | 'runs'
  | 'diff'
  | 'rules'

export type KGMaintenanceSelectedItem =
  | { kind: 'node'; id: string }
  | { kind: 'edge'; id: string }
  | null

interface KGMaintenanceState {
  activeSection: KGMaintenanceSection
  selectedItem: KGMaintenanceSelectedItem
  selectedWorkspace: string | null
  latestRunId: string
  setActiveSection: (section: KGMaintenanceSection) => void
  setSelectedItem: (item: KGMaintenanceSelectedItem) => void
  setSelectedWorkspace: (workspace: string | null) => void
  setLatestRunId: (runId: string) => void
}

const useKGMaintenanceStoreBase = create<KGMaintenanceState>()((set) => ({
  activeSection: 'overview',
  selectedItem: null,
  selectedWorkspace: null,
  latestRunId: 'latest',
  setActiveSection: (activeSection) => set({ activeSection }),
  setSelectedItem: (selectedItem) => set({ selectedItem }),
  setSelectedWorkspace: (selectedWorkspace) => set({ selectedWorkspace }),
  setLatestRunId: (latestRunId) => set({ latestRunId })
}))

export const useKGMaintenanceStore = createSelectors(useKGMaintenanceStoreBase)
```

- [ ] **Step 5: Add the tab**

Update `settings.ts` tab union to include `'kg-maintenance'`, bump persisted version by one, and normalize unknown tabs to `documents` during migration.

Update `SiteHeader.tsx`:

```tsx
<NavigationTab value="kg-maintenance" currentTab={currentTab}>
  {t('header.kgMaintenance', 'KG Maintenance')}
</NavigationTab>
```

Update `App.tsx` to import and mount `KGMaintenanceConsole`.

- [ ] **Step 6: Add the shell**

Implement a first renderable shell with the product layout:

- left workflow nav
- center content area
- right inspector
- skeleton/loading/error/empty states

Guard data loading so it only runs when `currentTab === 'kg-maintenance'`.

- [ ] **Step 7: Add i18n labels**

Add `header.kgMaintenance` and `kgMaintenance.*` keys to English and Chinese locale files. Preserve existing mojibake/encoding state; do not rewrite the whole JSON.

- [ ] **Step 8: Verify frontend slice**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src\stores\kgMaintenance.test.ts
bun test src\stores\settings.ts src\stores\graph.test.ts
bun run lint
```

Expected: focused tests and lint pass. If lint is broad and fails on unrelated dirty files, record exact unrelated failures and run scoped tests.

- [ ] **Step 9: Commit frontend shell slice**

```powershell
git add lightrag_webui\src\api\lightrag.ts lightrag_webui\src\stores\settings.ts lightrag_webui\src\stores\kgMaintenance.ts lightrag_webui\src\stores\kgMaintenance.test.ts lightrag_webui\src\features\SiteHeader.tsx lightrag_webui\src\App.tsx lightrag_webui\src\features\KGMaintenanceConsole.tsx lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.tsx lightrag_webui\src\locales\en.json lightrag_webui\src\locales\zh.json
git commit -m "feat: add kg maintenance console shell"
```

---

## Task 3: Read-Only Console Panels

**Files:**
- Create: `lightrag_webui/src/components/kg-maintenance/KGMaintenanceOverview.tsx`
- Create: `lightrag_webui/src/components/kg-maintenance/CatalogPanels.tsx`
- Create: `lightrag_webui/src/components/kg-maintenance/QualityAndApprovalPanels.tsx`
- Modify: `lightrag_webui/src/features/KGMaintenanceConsole.tsx`

- [ ] **Step 1: Write pure render-data tests**

Create helper functions in the panel files or a small `kgMaintenanceData.ts` module:

```typescript
export const countApprovalRequired = (proposals: { requires_approval: boolean }[]) =>
  proposals.filter((proposal) => proposal.requires_approval).length

export const highRiskFindingCount = (findings: { severity: string }[]) =>
  findings.filter((finding) => finding.severity === 'critical' || finding.severity === 'high').length
```

Test them first with Bun.

- [ ] **Step 2: Implement overview**

Use summary fields:

- workspace
- phase
- latestRunId
- quality overall
- node/edge/source counts
- pendingApprovalCount
- highRiskFindingCount
- artifact manifest

Include empty state copy: "Run KB iteration review first" / "先运行一次 KB iteration 审阅".

- [ ] **Step 3: Implement quality panel**

Render `overall`, `subscores`, `metrics`, `critical_blockers`, and findings grouped by category. Each finding must show severity, evidence, suggested fix type, and approval requirement.

- [ ] **Step 4: Implement entity and relation catalogs**

Render table-like dense lists. Relation rows must use `source - relation -> target` and must never use `邻接` as the main fallback. Missing keywords should render `未标注关系`.

- [ ] **Step 5: Implement approval, diff, run, and rules panels**

Render the current artifacts read-only first:

- approval queue proposals
- improvement backlog proposals
- diff summary/report
- iteration log
- quality rules
- known issues
- accepted/rejected changes

- [ ] **Step 6: Verify**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src\components\kg-maintenance
bun run lint
```

- [ ] **Step 7: Commit**

```powershell
git add lightrag_webui\src\components\kg-maintenance lightrag_webui\src\features\KGMaintenanceConsole.tsx
git commit -m "feat: render kb iteration review panels"
```

---

## Task 4: Medical Hierarchy Graph And Evidence Inspector

**Files:**
- Create: `lightrag_webui/src/components/kg-maintenance/kgMaintenanceGraph.ts`
- Create: `lightrag_webui/src/components/kg-maintenance/kgMaintenanceGraph.test.ts`
- Create: `lightrag_webui/src/components/kg-maintenance/MedicalHierarchyGraph.tsx`
- Create: `lightrag_webui/src/components/kg-maintenance/EvidenceInspector.tsx`
- Modify: `lightrag_webui/src/features/KGMaintenanceConsole.tsx`

- [ ] **Step 1: Write failing graph helper tests**

Add tests:

```typescript
import { describe, expect, test } from 'bun:test'
import { buildKGMaintenanceGraphView, formatRelationLabel } from './kgMaintenanceGraph'

describe('KG maintenance graph helpers', () => {
  test('sizes nodes by role and hierarchy depth instead of making all nodes equal', () => {
    const graph = buildKGMaintenanceGraphView({
      nodes: [
        { id: 'flu', label: '流行性感冒', entity_type: 'Disease', properties: {} },
        { id: 'symptom', label: '临床表现', entity_type: 'MedicalGroup', properties: {} },
        { id: 'fever', label: '高热不退', entity_type: 'Symptom', properties: { medical_group: 'clinical_manifestation' } }
      ],
      edges: []
    } as any)
    const sizes = new Map(graph.nodes.map((node) => [node.id, node.size]))
    expect(sizes.get('flu')).toBeGreaterThan(sizes.get('symptom')!)
    expect(sizes.get('symptom')).toBeGreaterThan(sizes.get('fever')!)
  })

  test('formats missing relation keyword as unmarked relation instead of adjacency', () => {
    expect(formatRelationLabel('')).toBe('未标注关系')
    expect(formatRelationLabel('邻接')).toBe('未标注关系')
    expect(formatRelationLabel('临床表现')).toBe('临床表现')
  })
})
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src\components\kg-maintenance\kgMaintenanceGraph.test.ts
```

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Implement graph helpers**

Implement:

- role detection: disease, category, subgroup, leaf.
- size scale: disease 36, category 28, subgroup 22, leaf 14, with bounded degree/evidence increments.
- relation label fallback.
- quality flags from missing source, missing file_path, generic relation.
- deterministic radial positions for disease/category/subgroup/leaf nodes.

- [ ] **Step 4: Implement graph component**

Use SVG or positioned HTML. Requirements:

- segmented controls for `医学层级`, `原始抽取`, `证据`, `质量`.
- distinct node sizes.
- right-bottom legend explaining color, size, evidence, risk.
- click node/edge sets the selected item in `useKGMaintenanceStore`.
- no canvas dependency required in first version.

- [ ] **Step 5: Implement EvidenceInspector**

Show selected item:

- id/label/type.
- description.
- source_id/file_path.
- relation direction for edges.
- evidence status.
- quality flags.
- a short reminder that LLM output is not evidence.

- [ ] **Step 6: Verify graph slice**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src\components\kg-maintenance\kgMaintenanceGraph.test.ts
bun run lint
```

- [ ] **Step 7: Commit**

```powershell
git add lightrag_webui\src\components\kg-maintenance\kgMaintenanceGraph.ts lightrag_webui\src\components\kg-maintenance\kgMaintenanceGraph.test.ts lightrag_webui\src\components\kg-maintenance\MedicalHierarchyGraph.tsx lightrag_webui\src\components\kg-maintenance\EvidenceInspector.tsx lightrag_webui\src\features\KGMaintenanceConsole.tsx
git commit -m "feat: add medical hierarchy maintenance graph"
```

---

## Task 5: Proposal Decisions And Run Trigger

**Files:**
- Modify: `lightrag/api/routers/kb_iteration_routes.py`
- Modify: `tests/api/routes/test_kb_iteration_routes.py`
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Modify: `lightrag_webui/src/components/kg-maintenance/QualityAndApprovalPanels.tsx`
- Modify: `lightrag_webui/src/features/KGMaintenanceConsole.tsx`

- [ ] **Step 1: Add failing tests for run trigger and decisions**

Backend:

- `POST /kb-iteration/{workspace}/runs` calls `run_iteration()` with validated roots and returns `phase: pending_user_review`.
- `accept/reject/defer` endpoints append review records and do not mutate KG.

Frontend:

- Button handlers call the API wrappers.
- High-risk proposal actions require visible confirmation text before API call.

- [ ] **Step 2: Verify RED**

Run:

```powershell
uv run pytest tests\api\routes\test_kb_iteration_routes.py -q
Set-Location D:\LightRAG\lightrag_webui
bun test src\components\kg-maintenance
```

- [ ] **Step 3: Implement**

Backend:

- Add `ProposalDecisionRequest`.
- Append records to accepted/rejected/deferred memory files.
- Add `RunIterationRequest` with optional profile.
- Call `run_iteration()` only after validation.

Frontend:

- Add action buttons with loading/error states.
- Refresh summary after run trigger or decision.
- Render the high-risk confirmation copy from the spec.

- [ ] **Step 4: Verify**

Run:

```powershell
uv run pytest tests\api\routes\test_kb_iteration_routes.py tests\kg\test_kb_iteration_runner.py -q
Set-Location D:\LightRAG\lightrag_webui
bun test src\components\kg-maintenance
bun run lint
```

- [ ] **Step 5: Commit**

```powershell
git add lightrag\api\routers\kb_iteration_routes.py tests\api\routes\test_kb_iteration_routes.py lightrag_webui\src\api\lightrag.ts lightrag_webui\src\components\kg-maintenance lightrag_webui\src\features\KGMaintenanceConsole.tsx
git commit -m "feat: review kb iteration proposals safely"
```

---

## Task 6: Browser Verification, Polish, And Final Audit

**Files:**
- Modify only files needed to fix verified issues.

- [ ] **Step 1: Run backend focused verification**

```powershell
uv run pytest tests\api\routes\test_kb_iteration_routes.py tests\kg\test_kb_iteration_snapshot.py tests\kg\test_kb_iteration_markdown.py tests\kg\test_kb_iteration_quality.py tests\kg\test_kb_iteration_proposals.py tests\kg\test_kb_iteration_diff.py tests\kg\test_kb_iteration_runner.py -q
```

- [ ] **Step 2: Run frontend verification**

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test
bun run lint
bun run build
```

- [ ] **Step 3: Start server and browser-check**

Use the existing LightRAG startup flow or a Vite dev server when API mocking is enough. Verify:

- `KG Maintenance` tab appears.
- Layout is three-column on desktop.
- Small viewport does not overflow text.
- Overview shows workspace summary.
- Medical graph has differentiated node sizes.
- Relation labels show semantic labels or `未标注关系`, not primary `邻接`.
- Evidence inspector is visible beside graph.
- Quality report and approval queue render.
- Empty/error/loading states render.

- [ ] **Step 4: Completion audit**

Create a checklist from `docs/superpowers/specs/2026-06-17-kb-maintenance-console-design-zh.md` and mark each requirement with evidence:

- file evidence
- route test evidence
- frontend test evidence
- browser evidence

Do not mark the goal complete until every explicit acceptance criterion has evidence.

- [ ] **Step 5: Final commit**

```powershell
git status --short
git diff --check
git add <only KG maintenance console files>
git commit -m "feat: build kg maintenance console"
```

---

## Self-Review Notes

- The plan implements the spec in phases and does not redefine the console as a smaller graph-only feature.
- The first backend API uses `latest` because current `run_iteration()` writes a latest workspace package rather than versioned run directories.
- Artifact reads are server-side whitelisted; no arbitrary file path is accepted.
- Proposal decisions are append-only and do not mutate KG facts, prompts, rules, or workspace data.
- The medical hierarchy graph is implemented inside the new maintenance console; existing generic `Knowledge Graph` behavior remains separate.
- Current dirty worktree changes are unrelated medical browse removal changes. Every task must stage only the files listed for that task unless a reviewed dependency requires another file.
