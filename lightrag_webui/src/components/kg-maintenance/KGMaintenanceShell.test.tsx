import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import KGMaintenanceShell from './KGMaintenanceShell'

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
})
