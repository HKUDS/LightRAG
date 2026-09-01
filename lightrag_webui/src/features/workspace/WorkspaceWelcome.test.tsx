import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import { act, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import { renderWithProviders } from '@/test/render'
import { useAuthStore } from '@/stores/state'
import { resetCustomization, seedCustomization } from '@/test/customization'

const DESCRIPTION = 'Internal deployment — do not share externally'

/**
 * The page reconciles the session against `/auth-status` on mount, through
 * the axios-based API module. Import ORDER is what makes the stub effective:
 * `mock.module` only reaches importers that have not been evaluated yet, so
 * the page is imported dynamically AFTER it. The module is restored in
 * afterAll so later test files get the real one.
 */
let realApiModule: Record<string, unknown>
let WorkspaceWelcome: typeof import('./WorkspaceWelcome').default

beforeAll(async () => {
  realApiModule = { ...(await import('@/api/lightrag')) }
  mock.module('@/api/lightrag', () => ({
    ...realApiModule,
    getAuthStatus: mock(async () => ({ auth_configured: true }))
  }))
  WorkspaceWelcome = (await import('./WorkspaceWelcome')).default
})

afterAll(() => {
  mock.module('@/api/lightrag', () => realApiModule)
})

beforeEach(() => {
  // The welcome page is for visitors who are NOT signed in; an earlier test
  // file that activates a session (authBootstrap) would otherwise leave this
  // one rendering nothing at all. The precondition belongs to the test.
  act(() => {
    useAuthStore.setState({ isAuthenticated: false, isGuestMode: false })
  })
})

afterEach(() => {
  act(() => {
    resetCustomization()
  })
})

const renderWelcome = async () => {
  const view = renderWithProviders(
    <MemoryRouter>
      <WorkspaceWelcome />
    </MemoryRouter>
  )
  // The page renders NOTHING until the customization verdict settles
  // (workspace-entry PRD §8.8: never flash the default content).
  await waitFor(() => {
    if (!view.container.innerHTML) throw new Error('welcome page has not settled')
  }, {
    // A real (refused) /auth-status attempt can outlast the 1s default
    // when the whole suite runs in one process.
    timeout: 5000
  })
  return view
}

describe('workspace welcome branding', () => {
  test('shows the deployment title', async () => {
    seedCustomization({ title: 'Acme Knowledge Base' })
    await renderWelcome()

    expect(
      await screen.findByRole('heading', { name: 'Acme Knowledge Base' })
    ).toBeInTheDocument()
  })

  test('never puts the deployment description on the page', async () => {
    // The description is the header tooltip's text — one line of operator
    // context sized for a hover. Printing it on the welcome page would put
    // an unbounded blob of bundle text in front of every visitor, competing
    // with the welcome document that IS meant to be read there.
    seedCustomization({ title: 'Acme Knowledge Base', description: DESCRIPTION })
    await renderWelcome()

    expect(await screen.findByRole('heading', { name: 'Acme Knowledge Base' })).toBeInTheDocument()
    expect(screen.queryAllByText(DESCRIPTION)).toHaveLength(0)
  })
})
