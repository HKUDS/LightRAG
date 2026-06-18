import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type { KBIterationSummaryResponse } from '@/api/lightrag'
import type { KGMaintenanceSection } from '@/stores/kgMaintenance'

if (!('localStorage' in globalThis)) {
  const storage = new Map<string, string>()
  Object.defineProperty(globalThis, 'localStorage', {
    value: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
      removeItem: (key: string) => storage.delete(key),
      clear: () => storage.clear()
    }
  })
}

const { default: KGMaintenanceShell } = await import('./KGMaintenanceShell')
const { MainPanel } = await import('@/features/KGMaintenanceConsole')
const {
  normalizeOptionalMarkdown,
  optionalMissingResponse,
  shouldApplyWorkspaceResponse
} = await import('./kgIterationLoadUtils')

const summary: KBIterationSummaryResponse = {
  workspace: 'influenza_medical_v1',
  latestRunId: 'snapshot-1',
  phase: 'pending_human_review',
  counts: {
    nodes: 1,
    edges: 1,
    sources: 1
  },
  quality: {
    overall: 82
  },
  pendingApprovalCount: 1,
  highRiskFindingCount: 0,
  artifacts: [
    'kb_context',
    'quality_report',
    'kg_snapshot',
    'quality_score',
    'approval_queue',
    'improvement_backlog',
    'accepted_changes',
    'rejected_changes',
    'iteration_log'
  ].map((key) => ({
    key,
    contentType:
      key === 'kg_snapshot' || key === 'quality_score' ? 'application/json' : 'text/markdown',
    exists: true
  }))
}

function renderMainPanel(activeSection: KGMaintenanceSection) {
  return renderToStaticMarkup(
    <MainPanel
      activeSection={activeSection}
      summary={summary}
      quality={{
        workspace: 'influenza_medical_v1',
        runId: 'snapshot-1',
        quality: {
          overall: 82,
          findings: [
            {
              severity: 'medium',
              category: 'coverage',
              message: 'Need more evidence',
              evidence: [],
              suggested_fix_type: 'manual_review',
              requires_approval: true
            }
          ]
        },
        report: '# 质量报告'
      }}
      rules={{
        workspace: 'influenza_medical_v1',
        qualityRules: '',
        knownIssues: '',
        acceptedChanges: 'accepted content marker',
        rejectedChanges: 'rejected content marker'
      }}
      kbContext="# 当前 KB 摘要"
      kgSnapshot={{
        workspace: 'influenza_medical_v1',
        snapshot_id: 'snapshot-1',
        nodes: [{ id: 'flu' }],
        edges: [{ source: 'flu', target: 'fever' }]
      }}
      qualityScore={{ overall: 82, findings: [{ severity: 'medium' }] }}
      approvalQueue="# 待审批 proposal\n\n- id: proposal-1\n  type: add_edge"
      improvementBacklog="backlog content marker"
      iterationLog="iteration log marker"
      llmTrace={{
        stop_reason: 'pending_human_review',
        rounds: [{ round_id: 'round-1', state: 'pending_human_review' }]
      }}
      llmReport="# LLM Review Report"
      llmProposals="- id: proposal-1"
      llmJudgeReport="# Judge Report"
      patchText="patch content marker"
      llmRunning={false}
      running={false}
      loading={false}
      onOpenSection={() => undefined}
      onProposalDecision={() => undefined}
      onRunLLMReview={() => undefined}
      onLoadPatch={() => undefined}
    />
  )
}

function renderEmptyMainPanel(activeSection: KGMaintenanceSection) {
  return renderToStaticMarkup(
    <MainPanel
      activeSection={activeSection}
      summary={null}
      quality={null}
      rules={null}
      kbContext=""
      kgSnapshot={null}
      qualityScore={null}
      approvalQueue=""
      improvementBacklog=""
      iterationLog=""
      llmTrace={null}
      llmReport=""
      llmProposals=""
      llmJudgeReport=""
      patchText=""
      llmRunning={false}
      running={false}
      loading={false}
      onOpenSection={() => undefined}
      onProposalDecision={() => undefined}
      onRunLLMReview={() => undefined}
      onLoadPatch={() => undefined}
    />
  )
}

describe('KGMaintenanceShell responsive layout', () => {
  test('allows the workspace toolbar to wrap on narrow screens', () => {
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
        inspector={<div>Inspector</div>}
      >
        <div>Console body</div>
      </KGMaintenanceShell>
    )

    expect(markup).toContain('flex-wrap')
    expect(markup).toContain('min-w-0 rounded-md border px-3 text-sm sm:min-w-52')
    expect(markup).toContain('grid-cols-1')
    expect(markup).toContain('lg:grid-cols-[220px_minmax(0,1fr)_320px]')
  })

  test('renders Chinese workflow navigation without graph labels', () => {
    const markup = renderToStaticMarkup(
      <KGMaintenanceShell
        activeSection="snapshot"
        onSectionChange={() => undefined}
        workspaces={['influenza_medical_v1']}
        selectedWorkspace="influenza_medical_v1"
        onWorkspaceChange={() => undefined}
        onRefresh={() => undefined}
        onRunReview={() => undefined}
        loading={false}
        running={false}
        error={null}
        inspector={<div>Inspector</div>}
      >
        <div>Console body</div>
      </KGMaintenanceShell>
    )

    expect(markup).toContain('知识库迭代')
    expect(markup).toContain('审阅包概览')
    expect(markup).toContain('当前阶段')
    expect(markup).toContain('当前 KB 摘要')
    expect(markup).toContain('质量与快照')
    expect(markup).toContain('质量检查')
    expect(markup).toContain('快照审阅')
    expect(markup).toContain('人工审阅')
    expect(markup).toContain('Proposal 审批')
    expect(markup).toContain('改进 backlog')
    expect(markup).toContain('决策记忆')
    expect(markup).toContain('辅助材料')
    expect(markup).toContain('LLM 审阅材料')
    expect(markup).not.toContain('Medical Graph')
    expect(markup).not.toContain('图谱画布')
  })

  test('accepts the LLM review section and renders children', () => {
    const markup = renderToStaticMarkup(
      <KGMaintenanceShell
        activeSection="llm-review"
        onSectionChange={() => undefined}
        workspaces={['influenza_medical_v1']}
        selectedWorkspace="influenza_medical_v1"
        onWorkspaceChange={() => undefined}
        onRefresh={() => undefined}
        onRunReview={() => undefined}
        loading={false}
        running={false}
        error={null}
        inspector={<div>Inspector</div>}
      >
        <div>LLM body</div>
      </KGMaintenanceShell>
    )

    expect(markup).toContain('LLM body')
  })
})

describe('MainPanel workflow routing', () => {
  test('optional response helper returns fallback for missing resources only', async () => {
    const result = await optionalMissingResponse(async () => {
      throw Object.assign(new Error('404 Not Found'), { response: { status: 404 } })
    }, null)

    expect(result).toBeNull()
  })

  test('optional response helper returns fallback for wrapper-style 404 messages', async () => {
    const result = await optionalMissingResponse(async () => {
      throw new Error('404 Not Found\n{"detail":"missing"}\n/kb-iteration/workspace/summary')
    }, null)

    expect(result).toBeNull()
  })

  test('optional response helper rethrows unexpected core failures', async () => {
    await expect(
      optionalMissingResponse(async () => {
        throw Object.assign(new Error('500 Internal Server Error'), { response: { status: 500 } })
      }, null)
    ).rejects.toThrow('500 Internal Server Error')
  })

  test('optional response helper rethrows wrapper-style 500 with not-found body text', async () => {
    await expect(
      optionalMissingResponse(async () => {
        throw new Error('500 Internal Server Error\n{"detail":"not found"}\n/kb-iteration/workspace/summary')
      }, null)
    ).rejects.toThrow('500 Internal Server Error')
  })

  test('optional response helper rethrows plain missing artifact messages without explicit 404', async () => {
    await expect(
      optionalMissingResponse(async () => {
        throw new Error('artifact not found')
      }, null)
    ).rejects.toThrow('artifact not found')

    await expect(
      optionalMissingResponse(async () => {
        throw new Error('missing artifact')
      }, null)
    ).rejects.toThrow('missing artifact')
  })

  test('workspace response guard rejects stale workspace payloads', () => {
    expect(shouldApplyWorkspaceResponse('workspace-a', () => 'workspace-a')).toBe(true)
    expect(shouldApplyWorkspaceResponse('workspace-a', () => 'workspace-b')).toBe(false)
  })

  test('markdown normalization accepts pre-normalized optional strings', () => {
    expect(normalizeOptionalMarkdown('loaded markdown')).toBe('loaded markdown')
    expect(normalizeOptionalMarkdown(null)).toBe('')
  })

  test('overview renders the review package artifact list', () => {
    const markup = renderMainPanel('overview')

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

  test('kb-summary renders the current KB context artifact', () => {
    const markup = renderMainPanel('kb-summary')

    expect(markup).toContain('当前 KB 摘要')
    expect(markup).toContain('kb_context.md')
    expect(markup).toContain('# 当前 KB 摘要')
  })

  test('stage renders the current iteration log artifact', () => {
    const markup = renderMainPanel('stage')

    expect(markup).toContain('当前阶段')
    expect(markup).toContain('iteration_log.md')
    expect(markup).toContain('iteration log marker')
  })

  test('quality renders markdown quality and JSON score artifacts', () => {
    const markup = renderMainPanel('quality')

    expect(markup).toContain('质量报告')
    expect(markup).toContain('质量分数')
    expect(markup).toContain('snapshots/quality_score.json')
  })

  test('quality tolerates missing quality responses and score artifacts', () => {
    const markup = renderEmptyMainPanel('quality')

    expect(markup).toContain('Quality Report')
    expect(markup).toContain('质量分数')
    expect(markup).toContain('snapshots/quality_score.json')
    expect(markup).toContain('暂无质量分数')
  })

  test('snapshot renders raw JSON without the graph canvas', () => {
    const markup = renderMainPanel('snapshot')

    expect(markup).toContain('图谱快照')
    expect(markup).toContain('snapshots/kg_snapshot.json')
    expect(markup).toContain('&quot;snapshot_id&quot;')
    expect(markup).not.toContain('Medical knowledge graph hierarchy')
    expect(markup).not.toContain('<svg')
  })

  test('snapshot tolerates a missing snapshot artifact', () => {
    const markup = renderEmptyMainPanel('snapshot')

    expect(markup).toContain('图谱快照')
    expect(markup).toContain('snapshots/kg_snapshot.json')
    expect(markup).toContain('暂无图谱快照')
  })

  test('approval renders proposal review content', () => {
    const markup = renderMainPanel('approval')

    expect(markup).toContain('待审批 proposal')
    expect(markup).toContain('proposal-1')
  })

  test('backlog renders the improvement backlog artifact', () => {
    const markup = renderMainPanel('backlog')

    expect(markup).toContain('改进 backlog')
    expect(markup).toContain('improvement_backlog.md')
    expect(markup).toContain('backlog content marker')
  })

  test('memory renders accepted and rejected decision memory', () => {
    const markup = renderMainPanel('memory')

    expect(markup).toContain('已接受变更记忆')
    expect(markup).toContain('accepted content marker')
    expect(markup).toContain('已拒绝变更记忆')
    expect(markup).toContain('rejected content marker')
  })

  test('llm-review renders auxiliary review materials without memory fallthrough', () => {
    const markup = renderMainPanel('llm-review')

    expect(markup).toContain('LLM 审阅材料')
    expect(markup).toContain('pending_human_review')
    expect(markup).toContain('# LLM Review Report')
    expect(markup).not.toContain('accepted content marker')
  })
})
