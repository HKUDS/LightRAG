import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type { KGMaintenanceSection } from '@/stores/kgMaintenance'
import KGMaintenanceShell from './KGMaintenanceShell'

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

describe('MainPanel section routing', () => {
  const renderMainPanel = async (activeSection: KGMaintenanceSection) => {
    const { MainPanel } = await import('@/features/KGMaintenanceConsole')

    return renderToStaticMarkup(
      <MainPanel
        activeSection={activeSection}
        summary={null}
        graph={null}
        quality={null}
        entities={null}
        relations={null}
        diff={null}
        rules={{
          workspace: 'influenza_medical_v1',
          qualityRules: 'RULE_MARKER',
          knownIssues: 'Known issues marker',
          acceptedChanges: 'Accepted changes marker',
          rejectedChanges: 'Rejected changes marker'
        }}
        approvalQueue=""
        improvementBacklog=""
        iterationLog=""
        llmTrace={{ stop_reason: 'pending_human_review', rounds: [] }}
        llmReport="# LLM Review Report"
        llmProposals={'proposals:\n- id: p1\n'}
        llmJudgeReport={'# Judge\nneeds_human'}
        patchText={'--- a/file\n+++ b/file\n'}
        llmRunning={false}
        running={false}
        loading={false}
        onOpenSection={() => undefined}
        onSelectItem={() => undefined}
        onProposalDecision={() => undefined}
        onRunLLMReview={() => undefined}
        onLoadPatch={() => undefined}
      />
    )
  }

  test('renders LLM review content without falling through to rules', async () => {
    const markup = await renderMainPanel('llm-review')

    expect(markup).toContain('LLM 审阅')
    expect(markup).toContain('pending_human_review')
    expect(markup).toContain('# LLM Review Report')
    expect(markup).not.toContain('RULE_MARKER')
  })

  test('renders patch candidate content', async () => {
    const markup = await renderMainPanel('patches')

    expect(markup).toContain('候选 Patch')
    expect(markup).toContain('--- a/file')
    expect(markup).toContain('+++ b/file')
    expect(markup).not.toContain('RULE_MARKER')
  })

  test('renders judge content', async () => {
    const markup = await renderMainPanel('judge')

    expect(markup).toContain('Judge 评判')
    expect(markup).toContain('needs_human')
    expect(markup).not.toContain('RULE_MARKER')
  })

  test('renders rule memory content for rules section', async () => {
    const markup = await renderMainPanel('rules')

    expect(markup).toContain('Rule Memory')
    expect(markup).toContain('RULE_MARKER')
  })
})
