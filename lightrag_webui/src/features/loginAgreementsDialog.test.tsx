import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import { act, cleanup, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'

import { renderWithProviders } from '@/test/render'
import { useAuthStore } from '@/stores/state'
import { resetCustomization, seedCustomization } from '@/test/customization'

const DOCUMENT_HEADING = 'Acme Terms of Service'
const DOCUMENT_BODY = 'You agree to the acceptable use policy.'
const LINK_TEXT = 'Terms and Privacy Policy'

/**
 * The page reconciles the session against `/auth-status` on mount, through
 * the axios-based API module. Import ORDER is what makes the stub effective:
 * `mock.module` only reaches importers that have not been evaluated yet, so
 * the page is imported dynamically AFTER it. Restored in afterAll.
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

const seedConsentBundle = (consentDocuments: string | null = LINK_TEXT): void => {
  seedCustomization(
    { title: 'Acme KB' },
    {
      consent_required: true,
      consent_documents: consentDocuments,
      agreements: {
        format: 'markdown',
        content: `# ${DOCUMENT_HEADING}\n\n${DOCUMENT_BODY}\n`
      }
    }
  )
}

const renderLogin = async () => {
  const view = renderWithProviders(
    <MemoryRouter>
      <LoginPage />
    </MemoryRouter>
  )
  // Renders nothing until the customization verdict settles.
  await waitFor(() => {
    if (!view.container.innerHTML) throw new Error('login page has not settled')
  }, {
    // A real (refused) /auth-status attempt can outlast the 1s default
    // when the whole suite runs in one process.
    timeout: 5000
  })
  return view
}

const openAgreements = async () => {
  const user = userEvent.setup()
  await user.click(await screen.findByRole('button', { name: LINK_TEXT }))
  return screen.findByRole('dialog')
}

describe('login agreements dialog', () => {
  test('the consent link opens the agreement document', async () => {
    seedConsentBundle()
    await renderLogin()

    const dialog = await openAgreements()
    // The document itself is what the visitor came to read; it is rendered,
    // not summarised or linked away to.
    expect(within(dialog).getByText(DOCUMENT_BODY)).toBeInTheDocument()
  })

  test('is named after what the visitor is agreeing to', async () => {
    seedConsentBundle()
    await renderLogin()

    const dialog = await openAgreements()
    // The accessible name is the link text the bundle maintains beside the
    // file — not a title parsed out of the Markdown, which would drift from
    // the wording on the checkbox the moment either changed.
    expect(dialog).toHaveAccessibleName(LINK_TEXT)
  })

  test('prints no second title above the document', async () => {
    seedConsentBundle()
    await renderLogin()

    const dialog = await openAgreements()
    // The dialog's own title element exists ONLY to give the dialog its
    // accessible name, so it is screen-reader-only; the one title a sighted
    // visitor sees belongs to the document. A printed one would sit above
    // the document's own heading and disagree with it.
    const namingTitle = within(dialog).getByText(LINK_TEXT)
    expect(namingTitle.closest('.sr-only') !== null).toBe(true)

    const visibleHeadings = within(dialog)
      .getAllByRole('heading')
      .filter((heading) => heading.closest('.sr-only') === null)
    expect(visibleHeadings.map((heading) => heading.textContent)).toEqual([DOCUMENT_HEADING])
  })

  test('falls back to the translated wording when the bundle names no documents', async () => {
    seedConsentBundle(null)
    await renderLogin()

    // 'login.consentDocuments' in locales/en.json. The gate never depends on
    // the bundle supplying wording — only on consent_required.
    const trigger = await screen.findByRole('button', { name: 'Privacy Policy Agreement' })
    expect(trigger).toBeInTheDocument()
  })
})
