import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import { act, cleanup, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import { renderWithProviders } from '@/test/render'
import { useAuthStore } from '@/stores/state'
import { resetCustomization, seedCustomization } from '@/test/customization'
import CustomizedCopyright from './CustomizedCopyright'

const COPYRIGHT = '© 2025 ACME Inc.'

/**
 * Both pre-login pages reconcile the session against `/auth-status` on
 * mount, through the axios-based API module. Import ORDER is what makes the
 * stub effective: `mock.module` only reaches importers that have not been
 * evaluated yet, so the pages are imported dynamically AFTER it. Restored in
 * afterAll so later test files get the real module.
 */
let realApiModule: Record<string, unknown>
let LoginPage: typeof import('@/features/LoginPage').default
let WorkspaceWelcome: typeof import('@/features/workspace/WorkspaceWelcome').default

beforeAll(async () => {
  realApiModule = { ...(await import('@/api/lightrag')) }
  mock.module('@/api/lightrag', () => ({
    ...realApiModule,
    getAuthStatus: mock(async () => ({ auth_configured: true }))
  }))
  LoginPage = (await import('@/features/LoginPage')).default
  WorkspaceWelcome = (await import('@/features/workspace/WorkspaceWelcome')).default
})

afterAll(() => {
  mock.module('@/api/lightrag', () => realApiModule)
})

beforeEach(() => {
  // These are pre-login pages; an earlier test file that activates a session
  // would otherwise leave them rendering nothing. The precondition belongs
  // to the test, not to whatever ran before it.
  act(() => {
    useAuthStore.setState({ isAuthenticated: false, isGuestMode: false })
  })
})

afterEach(() => {
  // Unmount FIRST. A file-local `afterEach` runs BEFORE the preload's
  // Testing Library `cleanup()`, so resetting the store here would update a
  // still-mounted `useCustomizedContent`, whose effect immediately calls
  // `load(locale)` and starts a real /ui/customization request during
  // teardown — one that can land during the NEXT test and overwrite its
  // seeded snapshot. `cleanup` is idempotent, so the preload's own call is
  // still fine.
  cleanup()
  resetCustomization()
})

describe('the copyright line itself', () => {
  test('renders the bundle text as a page footer', () => {
    renderWithProviders(<CustomizedCopyright copyright={COPYRIGHT} />)

    const footer = screen.getByRole('contentinfo')
    expect(footer).toHaveTextContent(COPYRIGHT)
  })

  test('renders nothing at all when there is no text to show', () => {
    // Undeclared, empty and whitespace-only are ONE state: no line, and no
    // footer element drawing padding at the foot of the page either.
    for (const value of ['', '   ', '\n\t']) {
      const { container, unmount } = renderWithProviders(
        <CustomizedCopyright copyright={value} />
      )
      expect(container.innerHTML).toBe('')
      unmount()
    }
  })
})

const renderPage = async (page: 'login' | 'welcome') => {
  const Page = page === 'login' ? LoginPage : WorkspaceWelcome
  const view = renderWithProviders(
    <MemoryRouter>
      <Page />
    </MemoryRouter>
  )
  await waitFor(() => {
    if (!view.container.innerHTML) throw new Error('page has not settled')
  }, {
    // A real (refused) /auth-status attempt can outlast the 1s default
    // when the whole suite runs in one process.
    timeout: 5000
  })
  return view
}

describe.each(['login', 'welcome'] as const)('%s page copyright', (page) => {
  test('shows the deployment copyright at the foot of the page', async () => {
    seedCustomization({ title: 'Acme KB', copyright: COPYRIGHT })
    await renderPage(page)

    expect(await screen.findByText(COPYRIGHT)).toBeInTheDocument()
  })

  test('keeps it outside the card, not inside the scrolling panel', async () => {
    seedCustomization({ title: 'Acme KB', copyright: COPYRIGHT })
    const { container } = await renderPage(page)

    const footer = await screen.findByText(COPYRIGHT)
    // The card owns its own max width and can scroll its own content; a
    // footer inside it would scroll away with the form instead of holding
    // the page's bottom edge.
    const card = container.querySelector('[class*="max-w-[480px]"], [class*="max-w-[520px]"]')
    expect(card === null).toBe(false)
    expect(card!.contains(footer)).toBe(false)
  })

  test('shows no copyright at all for an uncustomized deployment', async () => {
    // LightRAG ships no default text here: the line is the deployment's own
    // legal assertion, so a customer's page never carries LightRAG's and an
    // uncustomized one carries none.
    seedCustomization({ title: 'Acme KB' }, { customized: false })
    await renderPage(page)

    expect(screen.queryAllByRole('contentinfo')).toHaveLength(0)
  })
})
