/// <reference types="bun" />
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import type { ComponentType } from 'react'
import { act, cleanup, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { renderWithProviders } from '@/test/render'
import { resetCustomization, seedCustomization } from '@/test/customization'
import { useAuthStore } from '@/stores/state'

/**
 * The workspace shell's brand link doubles as the deployment-description
 * tooltip trigger. `asChild` is what makes the FOCUSABLE anchor the trigger
 * rather than a wrapper Radix would otherwise render — a keyboard or screen
 * reader user reaches the description only if that wiring holds, and it is
 * invisible to anything that reads the source.
 */

/** Radix's default `delayDuration`; the shell does not override it. */
const TOOLTIP_OPEN_DELAY_MS = 700

const TITLE = 'Research Cluster'
const DESCRIPTION = 'Production cluster — read only'

let realApiModule: Record<string, unknown>
let WorkspaceApp: ComponentType

beforeAll(async () => {
  realApiModule = { ...(await import('@/api/lightrag')) }
  // The shell probes credentials and starts a health check on mount. Both are
  // stubbed BEFORE the dynamic import below, because `stores/state` binds
  // `checkHealth` at import time.
  mock.module('@/api/lightrag', () => ({
    ...realApiModule,
    getAuthStatus: mock(async () => ({
      auth_configured: false,
      access_token: 'test-token',
      token_type: 'bearer',
      auth_mode: 'disabled',
      core_version: '1.0.0',
      api_version: '1.0.0',
      webui_title: TITLE,
      webui_description: DESCRIPTION
    })),
    checkHealth: mock(async () => ({
      status: 'healthy',
      core_version: '1.0.0',
      api_version: '1.0.0',
      webui_title: TITLE,
      webui_description: DESCRIPTION
    })),
    verifyCredentials: mock(async () => ({ ok: true }))
  }))

  WorkspaceApp = (await import('./WorkspaceApp')).default
})

afterAll(() => {
  mock.module('@/api/lightrag', () => realApiModule)
})

const seedDeployment = (title: string | null, description: string | null): void => {
  act(() => {
    useAuthStore.setState({ webuiTitle: title, webuiDescription: description })
  })
}

const renderShell = async (
  title: string | null = TITLE,
  description: string | null = DESCRIPTION
): Promise<void> => {
  seedCustomization()
  renderWithProviders(<WorkspaceApp />)
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50))
  })
  // After the mount probes settle, so the auth store's own writes cannot
  // overwrite what this test is asserting on.
  seedDeployment(title, description)
  await act(async () => {})
}

const brandLink = (): HTMLAnchorElement =>
  within(screen.getByRole('banner')).getByRole('link', { name: /.+/ }) as HTMLAnchorElement

beforeEach(() => {
  act(() => {
    useAuthStore.setState({ isGuestMode: false, username: null })
  })
})

afterEach(() => {
  cleanup()
  resetCustomization()
})

describe('workspace header brand link', () => {
  test('resolves to this entry root, not to the admin index', async () => {
    await renderShell()

    // Document-relative on purpose: it must land on THIS entry's own root
    // under any proxy prefix, never on /webui and never on the admin
    // index.html the dev server answers `/` with.
    expect(brandLink().getAttribute('href')).toBe('./')
    expect(brandLink().textContent).toContain(TITLE)
  })

  test('falls back to the product name when the deployment sets no title', async () => {
    await renderShell('', DESCRIPTION)

    expect(brandLink().textContent).toContain('LightRAG')
  })
})

describe('workspace header deployment description', () => {
  test('the focusable link itself is the tooltip trigger', async () => {
    const user = userEvent.setup()
    await renderShell()

    const link = brandLink()
    // `asChild` means Radix must wire the ANCHOR up. A wrapper span carrying
    // the trigger would leave the description unreachable by keyboard, and
    // both would look identical in the source.
    expect(link.getAttribute('data-state') === null).toBe(false)

    expect(screen.queryAllByRole('tooltip')).toHaveLength(0)
    await user.hover(link)

    await waitFor(() => expect(screen.queryAllByRole('tooltip')).toHaveLength(1))
    expect(screen.getByRole('tooltip').textContent).toContain(DESCRIPTION)
  })

  test('no tooltip is attached when the deployment declares no description', async () => {
    const user = userEvent.setup()
    await renderShell(TITLE, '')

    await user.hover(brandLink())
    // Longer than Radix's open delay on purpose: a shorter wait would report
    // "no tooltip" for every deployment, description or not, and this test
    // would pass against a shell that always renders one.
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, TOOLTIP_OPEN_DELAY_MS + 300))
    })

    // An empty tooltip that opens on every hover over the brand is worse than
    // no tooltip: it covers the header for nothing.
    expect(screen.queryAllByRole('tooltip')).toHaveLength(0)
  })

  test('touch users get the description through a button, not a hover', async () => {
    const user = userEvent.setup()
    await renderShell()

    // Hover does not exist on touch, so the same text is also behind an
    // explicit control — named after what it does, not after what it reveals.
    const trigger = within(screen.getByRole('banner')).getByRole('button', {
      name: 'Deployment information'
    })
    expect(screen.queryAllByText(DESCRIPTION)).toHaveLength(0)

    await user.click(trigger)

    expect(await screen.findByText(DESCRIPTION)).toBeVisible()
  })

  test('no touch control is offered when there is nothing to show', async () => {
    await renderShell(TITLE, '')

    expect(
      within(screen.getByRole('banner')).queryAllByRole('button', {
        name: 'Deployment information'
      })
    ).toHaveLength(0)
  })
})
