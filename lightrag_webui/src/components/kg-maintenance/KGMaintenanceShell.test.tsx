import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
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
