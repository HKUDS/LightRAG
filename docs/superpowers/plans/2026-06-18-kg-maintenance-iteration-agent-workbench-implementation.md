# KG Maintenance Iteration Agent Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the KG Maintenance WebUI into a Chinese "知识库迭代 Agent" workbench that exposes the required review artifacts without rendering a graph canvas.

**Architecture:** Keep the existing React/Bun/Tailwind WebUI and KB iteration API. Replace the KG Maintenance main navigation and primary panels with workflow-oriented artifact panels while preserving approval-gated proposal decisions and LLM review as auxiliary material. Backend artifact keys already exist, so this is frontend-only unless tests reveal a route contract mismatch.

**Tech Stack:** React 19, TypeScript, Zustand, Bun test runner, Vite, Tailwind CSS, lucide-react, existing `@/api/lightrag` KB iteration client.

---

## Design Source

Read before implementing:

- `D:\LightRAG\docs\superpowers\specs\2026-06-18-kg-maintenance-iteration-agent-workbench-design.md`
- `D:\LightRAG\PRODUCT.md`
- `D:\LightRAG\AGENTS.md`

Important constraints:

- UI text should be Chinese, while keeping terms like KG, LLM, workspace, profile, JSON, Markdown, Diff, proposal.
- Do not render a graph canvas in the KG Maintenance main workflow.
- Do not change backend mutation behavior.
- Do not apply patches automatically.
- Proposal approval remains explicit and human-gated.

## File Structure

Modify:

- `D:\LightRAG\lightrag_webui\src\stores\kgMaintenance.ts`
  - Own the workflow section ids for the KG Maintenance workbench.
- `D:\LightRAG\lightrag_webui\src\stores\kgMaintenance.test.ts`
  - Cover the new default section and one state transition.
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.tsx`
  - Render the Chinese top bar, workflow navigation, status copy, and three-column workbench layout.
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.test.tsx`
  - Cover Chinese navigation, no graph entry, MainPanel routing, and artifact visibility.
- `D:\LightRAG\lightrag_webui\src\features\KGMaintenanceConsole.tsx`
  - Load `kb_context`, `kg_snapshot`, and `quality_score` artifacts and route the new workflow sections.
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\QualityAndApprovalPanels.tsx`
  - Chinese labels for quality, approval, backlog, run log, and decision memory.
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx`
  - Clean mojibake and mark LLM materials as auxiliary.
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.test.tsx`
  - Update assertions to real Chinese text.

Create:

- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\IterationWorkbenchPanels.tsx`
  - Focused artifact panels:
    - `IterationOverviewPanel`
    - `IterationStagePanel`
    - `KBSummaryPanel`
    - `SnapshotReviewPanel`
    - `BacklogPanel`
    - `DecisionMemoryPanel`
    - `IterationReviewAside`
    - shared `MarkdownArtifactPanel`
    - shared `JsonArtifactPanel`
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\IterationWorkbenchPanels.test.tsx`
  - Direct tests for the new panels.

Do not delete in this implementation:

- `MedicalHierarchyGraph.tsx`
- `CatalogPanels.tsx`
- `EvidenceInspector.tsx`

They can remain for future or legacy use, but they must not be rendered from the new KG Maintenance main navigation.

---

### Task 1: Store And Shell Navigation

**Files:**

- Modify: `D:\LightRAG\lightrag_webui\src\stores\kgMaintenance.ts`
- Modify: `D:\LightRAG\lightrag_webui\src\stores\kgMaintenance.test.ts`
- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.tsx`
- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.test.tsx`

- [ ] **Step 1: Write the failing store test**

Replace the test body in `D:\LightRAG\lightrag_webui\src\stores\kgMaintenance.test.ts` with this behavior:

```ts
import { describe, expect, test } from 'bun:test'
import { useKGMaintenanceStore } from './kgMaintenance'

describe('KG maintenance store', () => {
  test('tracks workflow section, workspace, and latest run id', () => {
    useKGMaintenanceStore.getState().setActiveSection('snapshot')
    useKGMaintenanceStore.getState().setSelectedWorkspace('influenza_medical_v1')
    useKGMaintenanceStore.getState().setLatestRunId('latest')

    expect(useKGMaintenanceStore.getState().activeSection).toBe('snapshot')
    expect(useKGMaintenanceStore.getState().selectedWorkspace).toBe('influenza_medical_v1')
    expect(useKGMaintenanceStore.getState().latestRunId).toBe('latest')
  })
})
```

- [ ] **Step 2: Write the failing shell navigation test**

In `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.test.tsx`, replace the old `"renders LLM review navigation sections"` test with:

```tsx
test('renders Chinese iteration agent workflow without a graph entry', () => {
  const markup = renderToStaticMarkup(
    <KGMaintenanceShell
      activeSection="overview"
      onSectionChange={() => undefined}
      workspaces={['influenza_medical_v1']}
      selectedWorkspace="influenza_medical_v1"
      onWorkspaceChange={() => undefined}
      onRefresh={() => undefined}
      onRunReview={() => undefined}
      loading={false}
      running={false}
      error={null}
      inspector={<div>审阅侧栏</div>}
    >
      <div>Console body</div>
    </KGMaintenanceShell>
  )

  expect(markup).toContain('知识库迭代 Agent')
  expect(markup).toContain('当前阶段')
  expect(markup).toContain('当前 KB 摘要')
  expect(markup).toContain('质量检查')
  expect(markup).toContain('快照审阅')
  expect(markup).toContain('Proposal 审批')
  expect(markup).toContain('改进 backlog')
  expect(markup).toContain('决策记忆')
  expect(markup).toContain('LLM 审阅材料')
  expect(markup).not.toContain('Medical Graph')
  expect(markup).not.toContain('图谱画布')
})
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/stores/kgMaintenance.test.ts src/components/kg-maintenance/KGMaintenanceShell.test.tsx
```

Expected:

- `kgMaintenance.test.ts` fails because `snapshot` is not assignable or not supported yet.
- `KGMaintenanceShell.test.tsx` fails because the shell still renders old English/garbled labels.

- [ ] **Step 4: Implement the new workflow section type**

In `D:\LightRAG\lightrag_webui\src\stores\kgMaintenance.ts`, replace the section union and remove graph selected item state if it is no longer referenced by the compiler after later tasks.

Use this section union now:

```ts
export type KGMaintenanceSection =
  | 'overview'
  | 'stage'
  | 'kb-summary'
  | 'quality'
  | 'snapshot'
  | 'approval'
  | 'backlog'
  | 'memory'
  | 'llm-review'
```

Keep the state shape minimal:

```ts
interface KGMaintenanceState {
  activeSection: KGMaintenanceSection
  selectedWorkspace: string | null
  latestRunId: string
  setActiveSection: (section: KGMaintenanceSection) => void
  setSelectedWorkspace: (workspace: string | null) => void
  setLatestRunId: (runId: string) => void
}

const useKGMaintenanceStoreBase = create<KGMaintenanceState>()((set) => ({
  activeSection: 'overview',
  selectedWorkspace: null,
  latestRunId: 'latest',
  setActiveSection: (activeSection) => set({ activeSection }),
  setSelectedWorkspace: (selectedWorkspace) => set({ selectedWorkspace }),
  setLatestRunId: (latestRunId) => set({ latestRunId })
}))
```

- [ ] **Step 5: Implement the Chinese workflow shell**

In `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.tsx`, replace the `sections` array with:

```ts
const sections: SectionItem[] = [
  { id: 'overview', label: '审阅包概览', group: '知识库迭代', icon: LayoutDashboardIcon },
  { id: 'stage', label: '当前阶段', group: '知识库迭代', icon: HistoryIcon },
  { id: 'kb-summary', label: '当前 KB 摘要', group: '知识库迭代', icon: BookOpenIcon },
  { id: 'quality', label: '质量检查', group: '质量与快照', icon: ShieldCheckIcon },
  { id: 'snapshot', label: '快照审阅', group: '质量与快照', icon: FileSearchIcon },
  { id: 'approval', label: 'Proposal 审批', group: '人工审阅', icon: ClipboardCheckIcon },
  { id: 'backlog', label: '改进 backlog', group: '人工审阅', icon: ListTreeIcon },
  { id: 'memory', label: '决策记忆', group: '人工审阅', icon: BookOpenIcon },
  { id: 'llm-review', label: 'LLM 审阅材料', group: '辅助材料', icon: FileSearchIcon }
]
```

Update visible copy:

```tsx
<h1 className="text-base font-semibold">知识库迭代 Agent</h1>
<p className="text-muted-foreground truncate text-xs">
  {selectedWorkspace || '未选择 workspace'}
</p>
```

Use these button labels:

```tsx
刷新
运行审阅包
运行中
```

Use these status messages:

```tsx
running
  ? '正在生成 KB 审阅包。产物会进入人工审阅，不会自动修改 KG。'
  : '正在加载审阅包产物...'
```

- [ ] **Step 6: Run the tests and verify GREEN for Task 1**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/stores/kgMaintenance.test.ts src/components/kg-maintenance/KGMaintenanceShell.test.tsx
```

Expected:

- Store test passes.
- Shell navigation test passes.
- MainPanel tests may still fail because section routing is not updated yet; leave those failures for Task 3 if they refer to missing props or old routes.

- [ ] **Step 7: Commit Task 1**

Run:

```powershell
cd D:\LightRAG
git add lightrag_webui/src/stores/kgMaintenance.ts lightrag_webui/src/stores/kgMaintenance.test.ts lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.tsx lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx
git commit -m "feat: add kg iteration workflow shell"
```

---

### Task 2: Artifact Workbench Panels

**Files:**

- Create: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\IterationWorkbenchPanels.tsx`
- Create: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\IterationWorkbenchPanels.test.tsx`

- [ ] **Step 1: Write failing panel tests**

Create `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\IterationWorkbenchPanels.test.tsx`:

```tsx
import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import {
  IterationOverviewPanel,
  JsonArtifactPanel,
  DecisionMemoryPanel
} from './IterationWorkbenchPanels'

const artifacts = [
  { key: 'kb_context', contentType: 'text/markdown', exists: true },
  { key: 'quality_report', contentType: 'text/markdown', exists: true },
  { key: 'kg_snapshot', contentType: 'application/json', exists: true },
  { key: 'quality_score', contentType: 'application/json', exists: true },
  { key: 'approval_queue', contentType: 'text/markdown', exists: true },
  { key: 'improvement_backlog', contentType: 'text/markdown', exists: true },
  { key: 'accepted_changes', contentType: 'text/markdown', exists: true },
  { key: 'rejected_changes', contentType: 'text/markdown', exists: true },
  { key: 'iteration_log', contentType: 'text/markdown', exists: true }
]

describe('Iteration workbench panels', () => {
  test('renders all required review package artifacts', () => {
    const markup = renderToStaticMarkup(
      <IterationOverviewPanel
        summary={{
          workspace: 'influenza_medical_v1',
          latestRunId: 'latest',
          profile: 'clinical_guideline_zh',
          phase: 'pending_user_review',
          counts: { nodes: 428, edges: 1240, sources: 12 },
          quality: { overall: 82 },
          pendingApprovalCount: 17,
          highRiskFindingCount: 2,
          artifacts
        }}
        loading={false}
        onOpenSection={() => undefined}
      />
    )

    expect(markup).toContain('当前 KB 摘要')
    expect(markup).toContain('kb_context.md')
    expect(markup).toContain('质量报告')
    expect(markup).toContain('quality_report.md')
    expect(markup).toContain('图谱快照')
    expect(markup).toContain('snapshots/kg_snapshot.json')
    expect(markup).toContain('质量分数')
    expect(markup).toContain('snapshots/quality_score.json')
    expect(markup).toContain('待审批 proposal')
    expect(markup).toContain('approval_queue.md')
    expect(markup).toContain('改进 backlog')
    expect(markup).toContain('improvement_backlog.md')
    expect(markup).toContain('已接受变更记忆')
    expect(markup).toContain('accepted_changes.md')
    expect(markup).toContain('已拒绝变更记忆')
    expect(markup).toContain('rejected_changes.md')
    expect(markup).toContain('当前阶段')
    expect(markup).toContain('iteration_log.md')
  })

  test('renders JSON artifact summary and raw JSON without a graph canvas', () => {
    const markup = renderToStaticMarkup(
      <JsonArtifactPanel
        title="图谱快照"
        fileName="snapshots/kg_snapshot.json"
        payload={{
          workspace: 'influenza_medical_v1',
          snapshot_id: 'snapshot-1',
          nodes: [{ id: 'flu' }, { id: 'fever' }],
          edges: [{ source: 'flu', target: 'fever' }]
        }}
        summaryRows={[
          ['节点数', '2'],
          ['关系数', '1']
        ]}
        emptyText="暂无图谱快照。运行审阅包后生成。"
      />
    )

    expect(markup).toContain('图谱快照')
    expect(markup).toContain('snapshots/kg_snapshot.json')
    expect(markup).toContain('节点数')
    expect(markup).toContain('关系数')
    expect(markup).toContain('&quot;snapshot_id&quot;')
    expect(markup).not.toContain('<svg')
    expect(markup).not.toContain('Medical knowledge graph hierarchy')
  })

  test('renders accepted and rejected decision memory together', () => {
    const markup = renderToStaticMarkup(
      <DecisionMemoryPanel
        acceptedChanges="# Accepted\n\n- ok"
        rejectedChanges="# Rejected\n\n- no"
      />
    )

    expect(markup).toContain('已接受变更记忆')
    expect(markup).toContain('accepted_changes.md')
    expect(markup).toContain('已拒绝变更记忆')
    expect(markup).toContain('rejected_changes.md')
    expect(markup).toContain('# Accepted')
    expect(markup).toContain('# Rejected')
  })
})
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx
```

Expected:

- FAIL because `IterationWorkbenchPanels.tsx` does not exist.

- [ ] **Step 3: Implement `IterationWorkbenchPanels.tsx`**

Create `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\IterationWorkbenchPanels.tsx` with these exports and behavior:

```tsx
import Button from '@/components/ui/Button'
import type { KBIterationSummaryResponse } from '@/api/lightrag'
import type { KGMaintenanceSection } from '@/stores/kgMaintenance'
import {
  BookOpenIcon,
  ClipboardCheckIcon,
  FileJsonIcon,
  HistoryIcon,
  ShieldCheckIcon
} from 'lucide-react'
import type { ReactNode } from 'react'

type ArtifactCard = {
  key: string
  label: string
  fileName: string
  section: KGMaintenanceSection
  description: string
}

const REQUIRED_ARTIFACTS: ArtifactCard[] = [
  {
    key: 'kb_context',
    label: '当前 KB 摘要',
    fileName: 'kb_context.md',
    section: 'kb-summary',
    description: '当前知识库范围、主要实体类别、关系密度和维护上下文。'
  },
  {
    key: 'quality_report',
    label: '质量报告',
    fileName: 'quality_report.md',
    section: 'quality',
    description: '结构质量、证据接地、重复实体和待复核问题。'
  },
  {
    key: 'kg_snapshot',
    label: '图谱快照',
    fileName: 'snapshots/kg_snapshot.json',
    section: 'snapshot',
    description: '节点、关系、类型分布和 snapshot 元数据。'
  },
  {
    key: 'quality_score',
    label: '质量分数',
    fileName: 'snapshots/quality_score.json',
    section: 'quality',
    description: 'overall、subscores、metrics、findings 等结构化质量结果。'
  },
  {
    key: 'approval_queue',
    label: '待审批 proposal',
    fileName: 'approval_queue.md',
    section: 'approval',
    description: '需要人工接受、拒绝或延后的 proposal。'
  },
  {
    key: 'improvement_backlog',
    label: '改进 backlog',
    fileName: 'improvement_backlog.md',
    section: 'backlog',
    description: '暂不处理但需要继续跟踪的维护事项。'
  },
  {
    key: 'accepted_changes',
    label: '已接受变更记忆',
    fileName: 'accepted_changes.md',
    section: 'memory',
    description: '已接受 proposal 的原因、影响范围和验证记录。'
  },
  {
    key: 'rejected_changes',
    label: '已拒绝变更记忆',
    fileName: 'rejected_changes.md',
    section: 'memory',
    description: '已拒绝建议和拒绝理由，避免重复建议。'
  },
  {
    key: 'iteration_log',
    label: '当前阶段',
    fileName: 'iteration_log.md',
    section: 'stage',
    description: '本轮运行状态、产物路径、下一步和错误记录。'
  }
]

export function IterationOverviewPanel({
  summary,
  loading,
  onOpenSection
}: {
  summary: KBIterationSummaryResponse | null
  loading: boolean
  onOpenSection: (section: KGMaintenanceSection) => void
}) {
  if (loading && !summary) {
    return (
      <div className="space-y-3">
        <div className="bg-muted h-20 animate-pulse rounded-lg" />
        <div className="grid gap-3 md:grid-cols-3">
          <div className="bg-muted h-24 animate-pulse rounded-lg" />
          <div className="bg-muted h-24 animate-pulse rounded-lg" />
          <div className="bg-muted h-24 animate-pulse rounded-lg" />
        </div>
      </div>
    )
  }

  if (!summary) {
    return (
      <EmptyState
        title="还没有审阅包"
        message="请选择 workspace，然后运行审阅包生成 KB iteration artifacts。"
      />
    )
  }

  const existing = new Set(summary.artifacts.filter((item) => item.exists).map((item) => item.key))

  return (
    <section className="space-y-4">
      <div className="border-border/70 bg-muted/20 rounded-lg border p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold">审阅包概览</h2>
            <p className="text-muted-foreground mt-1 text-sm">
              workspace: {summary.workspace} · profile: {summary.profile || '未指定'}
            </p>
          </div>
          <div className="text-muted-foreground text-right text-xs">
            <div>Run: {summary.latestRunId}</div>
            <div>{summary.generatedAt || '暂无时间戳'}</div>
          </div>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <Metric label="当前阶段" value={summary.phase || 'unknown'} />
        <Metric label="质量分数" value={`${summary.quality?.overall ?? 0}`} />
        <Metric label="待审批 proposal" value={`${summary.pendingApprovalCount}`} />
        <Metric label="节点 / 关系" value={`${summary.counts.nodes} / ${summary.counts.edges}`} />
      </div>

      <div className="grid gap-3 lg:grid-cols-2 xl:grid-cols-3">
        {REQUIRED_ARTIFACTS.map((artifact) => (
          <button
            key={artifact.key}
            type="button"
            onClick={() => onOpenSection(artifact.section)}
            className="border-border/70 hover:bg-accent/40 rounded-lg border bg-background p-3 text-left transition-colors"
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <div className="text-sm font-semibold">{artifact.label}</div>
                <div className="text-muted-foreground mt-1 font-mono text-xs">
                  {artifact.fileName}
                </div>
              </div>
              <span className="bg-muted rounded-md px-2 py-1 text-xs">
                {existing.has(artifact.key) ? '已生成' : '缺失'}
              </span>
            </div>
            <p className="text-muted-foreground mt-3 text-sm">{artifact.description}</p>
          </button>
        ))}
      </div>
    </section>
  )
}

export function IterationStagePanel({ iterationLog }: { iterationLog: string }) {
  return (
    <MarkdownArtifactPanel
      icon={<HistoryIcon className="size-4 text-emerald-600" />}
      title="当前阶段"
      fileName="iteration_log.md"
      content={iterationLog}
      emptyText="暂无当前阶段记录。运行审阅包后生成。"
    />
  )
}

export function KBSummaryPanel({ kbContext }: { kbContext: string }) {
  return (
    <MarkdownArtifactPanel
      icon={<BookOpenIcon className="size-4 text-emerald-600" />}
      title="当前 KB 摘要"
      fileName="kb_context.md"
      content={kbContext}
      emptyText="暂无 KB 摘要。运行审阅包后生成。"
    />
  )
}

export function BacklogPanel({ improvementBacklog }: { improvementBacklog: string }) {
  return (
    <MarkdownArtifactPanel
      icon={<ClipboardCheckIcon className="size-4 text-amber-600" />}
      title="改进 backlog"
      fileName="improvement_backlog.md"
      content={improvementBacklog}
      emptyText="暂无 backlog。"
    />
  )
}

export function DecisionMemoryPanel({
  acceptedChanges,
  rejectedChanges
}: {
  acceptedChanges: string
  rejectedChanges: string
}) {
  return (
    <section className="grid gap-4 xl:grid-cols-2">
      <MarkdownArtifactPanel
        title="已接受变更记忆"
        fileName="accepted_changes.md"
        content={acceptedChanges}
        emptyText="暂无已接受变更记忆。"
      />
      <MarkdownArtifactPanel
        title="已拒绝变更记忆"
        fileName="rejected_changes.md"
        content={rejectedChanges}
        emptyText="暂无已拒绝变更记忆。"
      />
    </section>
  )
}

export function SnapshotReviewPanel({ snapshot }: { snapshot: unknown }) {
  const payload = asRecord(snapshot)
  const nodes = Array.isArray(payload.nodes) ? payload.nodes.length : Number(payload.node_count ?? 0)
  const edges = Array.isArray(payload.edges) ? payload.edges.length : Number(payload.edge_count ?? 0)

  return (
    <JsonArtifactPanel
      title="图谱快照"
      fileName="snapshots/kg_snapshot.json"
      payload={snapshot}
      summaryRows={[
        ['节点数', String(nodes || 0)],
        ['关系数', String(edges || 0)],
        ['workspace', String(payload.workspace ?? 'unknown')],
        ['snapshot id', String(payload.snapshot_id ?? payload.generated_at ?? 'unknown')]
      ]}
      emptyText="暂无图谱快照。运行审阅包后生成。"
    />
  )
}

export function QualityScoreJsonPanel({ qualityScore }: { qualityScore: unknown }) {
  const payload = asRecord(qualityScore)
  return (
    <JsonArtifactPanel
      title="质量分数"
      fileName="snapshots/quality_score.json"
      payload={qualityScore}
      summaryRows={[
        ['overall', String(payload.overall ?? 0)],
        ['findings', String(Array.isArray(payload.findings) ? payload.findings.length : 0)],
        ['critical blockers', String(Array.isArray(payload.critical_blockers) ? payload.critical_blockers.length : 0)]
      ]}
      emptyText="暂无质量分数。运行审阅包后生成。"
    />
  )
}

export function JsonArtifactPanel({
  title,
  fileName,
  payload,
  summaryRows,
  emptyText
}: {
  title: string
  fileName: string
  payload: unknown
  summaryRows: Array<[string, string]>
  emptyText: string
}) {
  const hasPayload = payload !== null && payload !== undefined && payload !== ''
  const jsonText = hasPayload ? JSON.stringify(payload, null, 2) : emptyText

  return (
    <section className="space-y-4">
      <PanelHeader icon={<FileJsonIcon className="size-4 text-sky-600" />} title={title} fileName={fileName} />
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {summaryRows.map(([label, value]) => (
          <Metric key={label} label={label} value={value} />
        ))}
      </div>
      <pre className="border-border/70 bg-muted/20 text-muted-foreground max-h-[520px] overflow-auto rounded-lg border p-3 text-xs">
        {jsonText}
      </pre>
    </section>
  )
}

export function MarkdownArtifactPanel({
  icon,
  title,
  fileName,
  content,
  emptyText
}: {
  icon?: ReactNode
  title: string
  fileName: string
  content: string
  emptyText: string
}) {
  return (
    <section className="space-y-4">
      <PanelHeader icon={icon} title={title} fileName={fileName} />
      <pre className="border-border/70 bg-background max-h-[640px] overflow-auto whitespace-pre-wrap rounded-lg border p-4 text-sm">
        {content.trim() ? content : emptyText}
      </pre>
    </section>
  )
}

export function IterationReviewAside({
  phase,
  pendingApprovalCount,
  highRiskFindingCount
}: {
  phase?: string
  pendingApprovalCount?: number
  highRiskFindingCount?: number
}) {
  return (
    <aside className="space-y-3">
      <h2 className="text-sm font-semibold">审阅侧栏</h2>
      <AsideBlock title="当前阶段">{phase || '暂无阶段信息'}</AsideBlock>
      <AsideBlock title="待审批 proposal">{String(pendingApprovalCount ?? 0)}</AsideBlock>
      <AsideBlock title="高风险发现">{String(highRiskFindingCount ?? 0)}</AsideBlock>
      <AsideBlock title="安全边界">
        所有事实、规则、prompt、rebuild 相关变更都需要人工审批，不自动修改 KG。
      </AsideBlock>
    </aside>
  )
}

function PanelHeader({
  icon,
  title,
  fileName
}: {
  icon?: ReactNode
  title: string
  fileName: string
}) {
  return (
    <div>
      <div className="flex items-center gap-2">
        {icon}
        <h2 className="text-base font-semibold">{title}</h2>
      </div>
      <div className="text-muted-foreground mt-1 font-mono text-xs">{fileName}</div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-border/70 rounded-lg border bg-background p-3">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="mt-1 break-words text-lg font-semibold">{value}</div>
    </div>
  )
}

function AsideBlock({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="border-border/70 bg-background rounded-lg border p-3">
      <div className="text-muted-foreground text-xs">{title}</div>
      <div className="mt-1 break-words text-sm">{children}</div>
    </div>
  )
}

function EmptyState({ title, message }: { title: string; message: string }) {
  return (
    <section className="border-border/70 bg-muted/20 rounded-lg border p-6">
      <h2 className="text-sm font-semibold">{title}</h2>
      <p className="text-muted-foreground mt-2 text-sm">{message}</p>
    </section>
  )
}

function asRecord(value: unknown): Record<string, any> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, any>)
    : {}
}
```

- [ ] **Step 4: Run panel tests and verify GREEN**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx
```

Expected:

- PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
cd D:\LightRAG
git add lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.tsx lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx
git commit -m "feat: add kg iteration artifact workbench panels"
```

---

### Task 3: Console Data Loading And Section Routing

**Files:**

- Modify: `D:\LightRAG\lightrag_webui\src\features\KGMaintenanceConsole.tsx`
- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.test.tsx`

- [ ] **Step 1: Update the MainPanel test first**

In the `renderMainPanel` helper in `KGMaintenanceShell.test.tsx`, add these props:

```tsx
kbContext="# 当前 KB 摘要"
kgSnapshot={{
  workspace: 'influenza_medical_v1',
  snapshot_id: 'snapshot-1',
  nodes: [{ id: 'flu' }],
  edges: [{ source: 'flu', target: 'fever' }]
}}
qualityScore={{
  overall: 82,
  findings: [{ severity: 'medium' }]
}}
```

Replace the old MainPanel tests with:

```tsx
test('renders overview with required review package artifacts', async () => {
  const markup = await renderMainPanel('overview')

  expect(markup).toContain('审阅包概览')
  expect(markup).toContain('kb_context.md')
  expect(markup).toContain('quality_report.md')
  expect(markup).toContain('snapshots/kg_snapshot.json')
  expect(markup).toContain('snapshots/quality_score.json')
  expect(markup).toContain('approval_queue.md')
  expect(markup).toContain('improvement_backlog.md')
  expect(markup).toContain('accepted_changes.md')
  expect(markup).toContain('rejected_changes.md')
  expect(markup).toContain('iteration_log.md')
})

test('routes current KB summary to kb_context.md content', async () => {
  const markup = await renderMainPanel('kb-summary')

  expect(markup).toContain('当前 KB 摘要')
  expect(markup).toContain('kb_context.md')
  expect(markup).toContain('# 当前 KB 摘要')
})

test('routes snapshot review to raw JSON summary without graph canvas', async () => {
  const markup = await renderMainPanel('snapshot')

  expect(markup).toContain('图谱快照')
  expect(markup).toContain('snapshots/kg_snapshot.json')
  expect(markup).toContain('&quot;snapshot_id&quot;')
  expect(markup).not.toContain('Medical knowledge graph hierarchy')
  expect(markup).not.toContain('<svg')
})

test('routes decision memory to accepted and rejected artifacts', async () => {
  const markup = await renderMainPanel('memory')

  expect(markup).toContain('已接受变更记忆')
  expect(markup).toContain('Accepted changes marker')
  expect(markup).toContain('已拒绝变更记忆')
  expect(markup).toContain('Rejected changes marker')
})

test('renders LLM review materials without falling through to decision memory', async () => {
  const markup = await renderMainPanel('llm-review')

  expect(markup).toContain('LLM 审阅材料')
  expect(markup).toContain('pending_human_review')
  expect(markup).toContain('# LLM Review Report')
  expect(markup).not.toContain('Accepted changes marker')
})
```

- [ ] **Step 2: Run MainPanel tests and verify RED**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/KGMaintenanceShell.test.tsx
```

Expected:

- FAIL because `MainPanelProps` does not include `kbContext`, `kgSnapshot`, or `qualityScore`, and routes still use old sections.

- [ ] **Step 3: Add missing artifact state and loaders**

In `D:\LightRAG\lightrag_webui\src\features\KGMaintenanceConsole.tsx`, add state:

```ts
const [kbContext, setKbContext] = useState('')
const [kgSnapshot, setKgSnapshot] = useState<Record<string, any> | null>(null)
const [qualityScore, setQualityScore] = useState<Record<string, any> | null>(null)
```

Add helpers near `markdownContent`:

```ts
const artifactPayload = (artifact: Awaited<ReturnType<typeof getKBIterationArtifact>>) =>
  'payload' in artifact ? artifact.payload : null
```

In the `Promise.all` call, add:

```ts
kbContextArtifact,
kgSnapshotArtifact,
qualityScoreArtifact,
```

and include requests:

```ts
getKBIterationArtifact(selectedWorkspace, 'kb_context'),
getKBIterationArtifact(selectedWorkspace, 'kg_snapshot'),
getKBIterationArtifact(selectedWorkspace, 'quality_score'),
```

After setting existing artifacts, add:

```ts
setKbContext(markdownContent(kbContextArtifact))
setKgSnapshot(artifactPayload(kgSnapshotArtifact))
setQualityScore(artifactPayload(qualityScoreArtifact))
```

Use `optionalArtifactContent` for these three only if a missing artifact should not fail the whole page. If using optional loading, normalize missing values to `''` and `null`, and keep the error banner for core summary failures.

- [ ] **Step 4: Stop rendering the graph canvas from MainPanel**

Remove these rendered routes from `MainPanel`:

```tsx
if (activeSection === 'graph' || activeSection === 'evidence') {
  return <MedicalHierarchyGraph graph={graph} onSelectItem={onSelectItem} />
}
if (activeSection === 'entities') {
  return <EntityCatalogPanel catalog={entities} onSelect={onSelectItem} />
}
if (activeSection === 'relations') {
  return <RelationCatalogPanel catalog={relations} onSelect={onSelectItem} />
}
if (activeSection === 'runs') {
  return <RunLogPanel runsText={iterationLog} summary={summary} />
}
if (activeSection === 'diff') {
  return <DiffPanel diff={diff} />
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
if (activeSection === 'rules') {
  return <RuleMemoryPanel rules={rules} />
}
```

Replace routing with:

```tsx
if (activeSection === 'overview') {
  return (
    <IterationOverviewPanel summary={summary} loading={loading} onOpenSection={onOpenSection} />
  )
}
if (activeSection === 'stage') {
  return <IterationStagePanel iterationLog={iterationLog} />
}
if (activeSection === 'kb-summary') {
  return <KBSummaryPanel kbContext={kbContext} />
}
if (activeSection === 'quality') {
  return (
    <section className="space-y-4">
      <QualityPanel quality={quality} />
      <QualityScoreJsonPanel qualityScore={qualityScore} />
    </section>
  )
}
if (activeSection === 'snapshot') {
  return <SnapshotReviewPanel snapshot={kgSnapshot} />
}
if (activeSection === 'approval') {
  return (
    <ApprovalPanel
      approvalQueue={approvalQueue}
      improvementBacklog={improvementBacklog}
      onDecision={onProposalDecision}
    />
  )
}
if (activeSection === 'backlog') {
  return <BacklogPanel improvementBacklog={improvementBacklog} />
}
if (activeSection === 'memory') {
  return (
    <DecisionMemoryPanel
      acceptedChanges={rules?.acceptedChanges || ''}
      rejectedChanges={rules?.rejectedChanges || ''}
    />
  )
}
if (activeSection === 'llm-review') {
  return (
    <section className="space-y-4">
      <LLMReviewPanel
        trace={llmTrace}
        report={llmReport}
        proposals={llmProposals}
        running={llmRunning || running}
        onRun={onRunLLMReview}
      />
      <PatchCandidatesPanel
        proposals={llmProposals}
        patchText={patchText}
        onLoadPatch={onLoadPatch}
      />
      <LLMJudgePanel report={llmJudgeReport} />
    </section>
  )
}
return <IterationOverviewPanel summary={summary} loading={loading} onOpenSection={onOpenSection} />
```

- [ ] **Step 5: Update imports and props**

Remove imports that are no longer used from `KGMaintenanceConsole.tsx`:

```ts
EntityCatalogPanel
RelationCatalogPanel
EvidenceInspector
findEdgeByIdAcrossSources
findNodeByIdAcrossSources
KGMaintenanceOverview
MedicalHierarchyGraph
DiffPanel
RuleMemoryPanel
RunLogPanel
```

Add imports:

```ts
import {
  BacklogPanel,
  DecisionMemoryPanel,
  IterationOverviewPanel,
  IterationReviewAside,
  IterationStagePanel,
  KBSummaryPanel,
  QualityScoreJsonPanel,
  SnapshotReviewPanel
} from '@/components/kg-maintenance/IterationWorkbenchPanels'
```

Pass the new aside to `KGMaintenanceShell`:

```tsx
inspector={
  <IterationReviewAside
    phase={summary?.phase}
    pendingApprovalCount={summary?.pendingApprovalCount}
    highRiskFindingCount={summary?.highRiskFindingCount}
  />
}
```

Remove `selectedItem`, `setSelectedItem`, `selectedNode`, and `selectedEdge` if no remaining references need them.

- [ ] **Step 6: Run MainPanel tests and verify GREEN**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/KGMaintenanceShell.test.tsx src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx
```

Expected:

- PASS.

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
cd D:\LightRAG
git add lightrag_webui/src/features/KGMaintenanceConsole.tsx lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx
git commit -m "feat: load kg iteration workbench artifacts"
```

---

### Task 4: Chinese Copy For Approval, Quality, And LLM Materials

**Files:**

- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\QualityAndApprovalPanels.tsx`
- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx`
- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.test.tsx`
- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.test.tsx`

- [ ] **Step 1: Write failing LLM copy tests**

Replace assertions in `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.test.tsx` with real Chinese labels:

```tsx
expect(markup).toContain('LLM 审阅材料')
expect(markup).toContain('辅助材料，不会自动修改 KG')
expect(markup).toContain('pending_human_review')
expect(markup).toContain('generic_relation')
expect(markup).toContain('# LLM Review Report')
```

For patches:

```tsx
expect(markup).toContain('候选 Patch')
expect(markup).toContain('proposal')
expect(markup).toContain('--- a/file')
expect(markup).toContain('p1')
```

For judge:

```tsx
expect(markup).toContain('Judge 评判')
expect(markup).toContain('人工复核')
expect(markup).toContain('needs_human')
```

- [ ] **Step 2: Add a focused approval copy assertion**

In `KGMaintenanceShell.test.tsx`, add a MainPanel route test:

```tsx
test('renders approval queue with Chinese gated decision copy', async () => {
  const markup = await renderMainPanel('approval', {
    approvalQueue: `
- id: p1
  type: kg_fact_correction
  target: flu
  proposed_change: add evidence
  reason: source grounded
  confidence: medium
  risk: medium
  requires_approval: true
`
  })

  expect(markup).toContain('待审批 proposal')
  expect(markup).toContain('需要人工审批')
  expect(markup).toContain('审阅理由')
  expect(markup).toContain('影响范围')
  expect(markup).toContain('验证 / 回滚说明')
  expect(markup).toContain('接受')
  expect(markup).toContain('拒绝')
  expect(markup).toContain('延后')
})
```

If the existing helper does not accept overrides, update it to merge overrides:

```tsx
const renderMainPanel = async (
  activeSection: KGMaintenanceSection,
  overrides: Partial<Parameters<typeof MainPanel>[0]> = {}
) => {
  const props: Parameters<typeof MainPanel>[0] = {
    activeSection,
    summary: null,
    kbContext: '# 当前 KB 摘要',
    kgSnapshot: {
      workspace: 'influenza_medical_v1',
      snapshot_id: 'snapshot-1',
      nodes: [{ id: 'flu' }],
      edges: [{ source: 'flu', target: 'fever' }]
    },
    qualityScore: {
      overall: 82,
      findings: [{ severity: 'medium' }]
    },
    quality: null,
    diff: null,
    rules: {
      workspace: 'influenza_medical_v1',
      qualityRules: 'Quality rules marker',
      knownIssues: 'Known issues marker',
      acceptedChanges: 'Accepted changes marker',
      rejectedChanges: 'Rejected changes marker'
    },
    approvalQueue: '',
    improvementBacklog: '',
    iterationLog: '# Iteration Log',
    llmTrace: { stop_reason: 'pending_human_review', rounds: [] },
    llmReport: '# LLM Review Report',
    llmProposals: 'proposals:\n- id: p1\n',
    llmJudgeReport: '# Judge\nneeds_human',
    patchText: '--- a/file\n+++ b/file\n',
    llmRunning: false,
    running: false,
    loading: false,
    onOpenSection: () => undefined,
    onProposalDecision: () => undefined,
    onRunLLMReview: () => undefined,
    onLoadPatch: () => undefined,
    ...overrides
  }
  return renderToStaticMarkup(<MainPanel {...props} />)
}
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/KGMaintenanceShell.test.tsx
```

Expected:

- FAIL on old mojibake or English labels.

- [ ] **Step 4: Replace LLM panel copy**

In `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx`, use these labels:

```tsx
title="LLM 审阅材料"
subtitle={stopReason ? `停止原因：${stopReason}。辅助材料，不会自动修改 KG。` : '尚未生成 LLM 审阅 trace。辅助材料，不会自动修改 KG。'}
{running ? '运行中' : '运行 LLM 审阅'}
Trace stop reason -> 停止原因
审阅轮次
暂无审阅轮次。
LLM Review Report -> LLM 审阅报告
Generated Proposals -> 生成的 proposal
```

For patch panel:

```tsx
title="候选 Patch"
subtitle="从 proposal 加载 patch 候选，仅供人工检查，不会自动应用。"
Selected Patch -> 已选择 Patch
Proposal Source -> proposal 来源
```

For judge:

```tsx
title="Judge 评判"
subtitle="LLM judge 报告与人工复核状态"
```

- [ ] **Step 5: Replace approval and quality copy**

In `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\QualityAndApprovalPanels.tsx`, update visible labels:

```tsx
Quality Report -> 质量报告
Overall score -> 总分
Search findings -> 搜索质量发现
All severities -> 全部严重级别
Critical -> critical
High -> high
Medium -> medium
Low -> low
Critical Blockers -> 阻塞问题
Evidence -> 证据
Quality Markdown -> quality_report.md
Approval Queue -> 待审批 proposal
approval-gated proposals -> 需要人工审批
No proposal is waiting for review. -> 暂无待审批 proposal。
Proposed change -> 建议变更
Reason -> 原因
Confidence -> 置信度
Expected metric change -> 预期指标变化
Review reason -> 审阅理由
Impact scope -> 影响范围
Verification / rollback notes -> 验证 / 回滚说明
accept -> 接受
reject -> 拒绝
defer -> 延后
Approval Queue Markdown -> approval_queue.md
Improvement Backlog Markdown -> improvement_backlog.md
```

Also replace `MEDICAL_REVIEW_CONFIRMATION` in `kgMaintenanceData.ts` with valid Chinese if it still contains mojibake:

```ts
export const MEDICAL_REVIEW_CONFIRMATION =
  '该操作会改变知识库行为或重建结果。请确认已检查来源证据、影响范围和回滚方式。'
```

- [ ] **Step 6: Run tests and verify GREEN**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/KGMaintenanceShell.test.tsx src/components/kg-maintenance/kgMaintenanceData.test.ts
```

Expected:

- PASS.

- [ ] **Step 7: Commit Task 4**

Run:

```powershell
cd D:\LightRAG
git add lightrag_webui/src/components/kg-maintenance/QualityAndApprovalPanels.tsx lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.test.tsx lightrag_webui/src/components/kg-maintenance/KGMaintenanceShell.test.tsx lightrag_webui/src/components/kg-maintenance/kgMaintenanceData.ts
git commit -m "feat: localize kg iteration review panels"
```

---

### Task 5: Final Verification And Browser Review

**Files:**

- Modify only if verification finds a concrete issue:
  - `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\KGMaintenanceShell.tsx`
  - `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\IterationWorkbenchPanels.tsx`
  - `D:\LightRAG\lightrag_webui\src\features\KGMaintenanceConsole.tsx`

- [ ] **Step 1: Run focused frontend tests**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/api/lightrag.test.ts src/components/kg-maintenance/IterationWorkbenchPanels.test.tsx src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/KGMaintenanceShell.test.tsx src/components/kg-maintenance/kgMaintenanceData.test.ts src/stores/kgMaintenance.test.ts
```

Expected:

- All listed tests pass.

- [ ] **Step 2: Run lint**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun run lint
```

Expected:

- Pass with no ESLint errors.

- [ ] **Step 3: Run build**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun run build
```

Expected:

- Pass.
- `lightrag/api/webui/` build output may change but is generated/gitignored.

- [ ] **Step 4: Start or reuse a local WebUI server**

If a dev server is already running, reuse it. Otherwise run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun run dev --host 127.0.0.1
```

Expected:

- Vite reports a localhost URL, usually `http://localhost:5173`.

- [ ] **Step 5: Browser review**

Open the KG Maintenance tab in the running WebUI and verify:

- First viewport says `知识库迭代 Agent`.
- No main navigation item says `Medical Graph`, `Entity Catalog`, `Relation Catalog`, or `图谱画布`.
- The overview shows all 9 requested artifacts.
- `snapshots/kg_snapshot.json` appears in the snapshot section as JSON/text, not as a canvas or SVG graph.
- Narrow viewport does not cause top bar or artifact names to overflow.
- Running/loading banners do not obscure content.

- [ ] **Step 6: Fix any browser issues with tests first**

If browser review finds text overflow, layout overlap, or missing artifact copy:

1. Add or update a Bun render test that captures the issue where feasible.
2. Run the test to confirm RED.
3. Patch the component.
4. Re-run the focused test, lint, and build.

- [ ] **Step 7: Commit final polish if any files changed**

If Step 6 changed files, run:

```powershell
cd D:\LightRAG
git add lightrag_webui/src/components/kg-maintenance lightrag_webui/src/features/KGMaintenanceConsole.tsx lightrag_webui/src/stores/kgMaintenance.ts
git commit -m "fix: polish kg iteration workbench layout"
```

If Step 6 made no changes, do not create an empty commit.

---

## Completion Checklist

- [ ] KG Maintenance first screen is a Chinese `知识库迭代 Agent` workbench.
- [ ] The 9 required artifact labels and filenames are visible.
- [ ] `kb_context.md` is loaded and displayed.
- [ ] `snapshots/kg_snapshot.json` is loaded and displayed as JSON/summary.
- [ ] `snapshots/quality_score.json` is loaded and displayed as JSON/summary.
- [ ] No graph canvas is rendered from the main KG Maintenance workflow.
- [ ] Proposal decisions still require explicit human action.
- [ ] LLM review output is labeled as auxiliary material.
- [ ] Focused Bun tests pass.
- [ ] WebUI lint passes.
- [ ] WebUI build passes.
- [ ] Browser review checks desktop and narrow viewport behavior.
