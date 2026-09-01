import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import { act, fireEvent, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import { renderWithProviders } from '@/test/render'
import { useAuthStore } from '@/stores/state'
import { resetCustomization, seedCustomization } from '@/test/customization'

const BUNDLE_LOGO = 'https://cdn.example.test/acme-logo.png'

/**
 * The page reconciles the session against `/auth-status` on mount, through
 * the axios-based API module. Import ORDER is what makes the stub effective:
 * `mock.module` only reaches importers that have not been evaluated yet, so
 * the page is imported dynamically AFTER it — a static `import LoginPage`
 * at the top of this file would bind the real module first and the request
 * would go out for real (`stores/documentTitleSync.test.ts` uses the same
 * ordering). The module is restored in afterAll so later test files get the
 * real one.
 */
let realApiModule: Record<string, unknown>
let LoginPage: typeof import('./LoginPage').default

beforeAll(async () => {
  realApiModule = { ...(await import('@/api/lightrag')) }
  mock.module('@/api/lightrag', () => ({
    ...realApiModule,
    getAuthStatus: mock(async () => ({ auth_configured: true }))
  }))
  LoginPage = (await import('./LoginPage')).default
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
  act(() => {
    resetCustomization()
  })
})

/**
 * Render the login page and wait for the customization verdict to settle.
 *
 * The page deliberately renders NOTHING until then (workspace-entry PRD
 * §8.8: never flash the default content before knowing whether a bundle is
 * active), so every assertion here has to wait first.
 */
const renderLogin = async () => {
  const view = renderWithProviders(
    <MemoryRouter>
      <LoginPage />
    </MemoryRouter>
  )
  await waitFor(() => {
    if (!view.container.innerHTML) throw new Error('login page has not settled')
  }, {
    // A real (refused) /auth-status attempt can outlast the 1s default
    // when the whole suite runs in one process.
    timeout: 5000
  })
  return view
}

describe('login page branding', () => {
  test('shows the bundle logo with the bundle alt text', async () => {
    seedCustomization({ logo_url: BUNDLE_LOGO, logo_alt: 'Acme Corp' })
    await renderLogin()

    // The deployment's own logo, not LightRAG's built-in asset and not a
    // lightning glyph — a customer's login page carries their brand.
    const logo = await screen.findByRole('img', { name: 'Acme Corp' })
    expect(logo).toHaveAttribute('src', BUNDLE_LOGO)
  })

  test('shows the resolved deployment title as the page heading', async () => {
    seedCustomization({ title: 'Acme Knowledge Base' })
    await renderLogin()

    expect(
      await screen.findByRole('heading', { name: 'Acme Knowledge Base' })
    ).toBeInTheDocument()
  })

  test('drops a logo that fails to load rather than showing a broken image', async () => {
    seedCustomization({ logo_url: BUNDLE_LOGO, logo_alt: 'Acme Corp' })
    await renderLogin()

    const logo = await screen.findByRole('img', { name: 'Acme Corp' })
    // A bundle can name a URL this browser cannot fetch (expired CDN, blocked
    // host). The alternative to removing it is a permanently broken image in
    // the middle of the login card.
    fireEvent.error(logo)

    await waitFor(() => {
      expect(screen.queryAllByRole('img', { name: 'Acme Corp' })).toHaveLength(0)
    })
  })

  test('a failed logo does not block a different logo from a later locale', async () => {
    seedCustomization({ logo_url: BUNDLE_LOGO, logo_alt: 'Acme Corp' })
    const view = await renderLogin()

    fireEvent.error(await screen.findByRole('img', { name: 'Acme Corp' }))
    await waitFor(() => {
      expect(screen.queryAllByRole('img', { name: 'Acme Corp' })).toHaveLength(0)
    })

    // The failure is latched against the URL that failed, not against
    // "logos": switching language loads a different bundle whose logo has
    // never been tried, and it must still render.
    act(() => {
      seedCustomization({ logo_url: 'https://cdn.example.test/acme-fr.png', logo_alt: 'Acme France' })
    })

    expect(await screen.findByRole('img', { name: 'Acme France' })).toHaveAttribute(
      'src',
      'https://cdn.example.test/acme-fr.png'
    )
    expect(view.container).toBeTruthy()
  })
})
