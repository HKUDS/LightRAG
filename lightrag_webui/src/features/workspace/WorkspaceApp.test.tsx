/// <reference types="bun" />
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import type { ComponentType } from 'react'
import { act, cleanup, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { renderWithProviders } from '@/test/render'
import { resetCustomization, seedCustomization } from '@/test/customization'
import { useAuthStore } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { useWebuiRetrievalHistoryStore } from '@/stores/webuiRetrievalHistory'
import { useWorkspaceRetrievalHistoryStore } from '@/stores/workspaceRetrievalHistory'
import { useIdentityEpochStore } from '@/lib/loginIdentity'
import { useBackendState } from '@/stores/state'
import { resetVersionCheckCache } from '@/lib/versionCheckCache'
import {
  captureProcessState,
  restoreProcessState,
  type ProcessStateSnapshot
} from '@/test/processState'

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
let authStatusFetches = 0
/**
 * Everything a mounted shell writes OUTSIDE React.
 *
 * `activateSessionFromAuthStatus` runs for real here — that is the point of
 * the test — so every render logs a session in: the auth store ends up
 * authenticated with this file's fake title and version, and seven
 * `LIGHTRAG-*` keys land in localStorage, `LIGHTRAG-API-TOKEN` among them.
 * `cleanup()` unmounts the tree and touches none of it.
 */
const sharedStores = [
  useAuthStore,
  useSettingsStore,
  // Reconciling an identity CLEARS both retrieval histories — the whole point
  // of `activateLoginIdentityFromToken` — so a shell mount empties whatever a
  // previous file left in these singletons. Restoring localStorage alone puts
  // the persisted envelopes back and leaves the hydrated stores empty, which
  // is the harder half to notice.
  useWebuiRetrievalHistoryStore,
  useWorkspaceRetrievalHistoryStore,
  // The epoch does NOT move under this file today, and the comment above is
  // why it is captured anyway: the same reconciliation that clears those two
  // histories is one line away from bumping it. Only `handleIdentityStorageEvent`
  // does so now — the CROSS-TAB listener, installed by the entry bootstraps this
  // file never imports — while `applyLoginIdentity`, the same-tab path a shell
  // mount actually runs, leaves it alone. That split is a fact about today's
  // implementation, not about the state's kind: the shell reads the epoch as a
  // remount key, so an advanced one leaking into a later file would remount its
  // query view against a session key nothing in that file set.
  useIdentityEpochStore,
  // Unlike the epoch above, this one DOES move under this file. The shell's
  // startup `runCredentialProbe` resolves against the stubbed
  // `verifyCredentials`, and a probe that succeeds while an API-key failure is
  // still recorded calls `useBackendState.clear()` -- `health: true`,
  // `message: null`. So a mount here does not merely read the backend store,
  // it ERASES a credential error some earlier file was relying on, and does it
  // silently: every assertion in this file passes either way.
  //
  // Measured with an `Invalid API Key` seeded at file entry, as a previous
  // file would leave it: without this entry 1 of 6 tests starts with the error
  // and the other 5 start clean, and the file hands a healthy store to
  // everything after it. With it, all 6 start with the error and it is still
  // there at file exit.
  useBackendState
]
let processSnapshot: ProcessStateSnapshot
/** As in `queryEntryAlignment`: mount-time reconciliation is async, so a write
 * can land after the test that triggered it was restored. */
let fileSnapshot: ProcessStateSnapshot

beforeAll(async () => {
  realApiModule = { ...(await import('@/api/lightrag')) }
  // The shell probes credentials and starts a health check on mount. Both are
  // stubbed BEFORE the dynamic import below, because `stores/state` binds
  // `checkHealth` at import time.
  mock.module('@/api/lightrag', () => ({
    ...realApiModule,
    getAuthStatus: mock(async () => ({
      ...((authStatusFetches += 1), {}),
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
  restoreProcessState(sharedStores, fileSnapshot)
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
  // Asserted on EVERY mount, not once: `versionCheckCache` is module-level, so
  // the shell that reconciles first would otherwise leave the flag set and
  // every later mount — here or in a later file — would take the early-return
  // path and skip reconciliation while still passing. Measured without the
  // reset in `beforeEach`: 1 fetch across the whole file instead of one per
  // test. Placing the check here is what makes that visible, since the first
  // test alone looks correct either way.
  expect(authStatusFetches).toBe(1)

  // After the mount probes settle, so the auth store's own writes cannot
  // overwrite what this test is asserting on.
  seedDeployment(title, description)
  await act(async () => {})
}

const brandLink = (): HTMLAnchorElement =>
  within(screen.getByRole('banner')).getByRole('link', { name: /.+/ }) as HTMLAnchorElement

beforeEach(() => {
  // Module-level and therefore process-wide: the shell sets it once its
  // `/auth-status` request settles, and every later mount — in this file or a
  // later one — would then take the early-return path and skip the auth
  // reconciliation entirely. A real page load starts a fresh module, which is
  // what this seam stands in for.
  resetVersionCheckCache()
  authStatusFetches = 0
  // First, before anything below mutates a store.
  fileSnapshot ??= captureProcessState(sharedStores)
  processSnapshot = captureProcessState(sharedStores)
  act(() => {
    useAuthStore.setState({ isGuestMode: false, username: null })
  })
})

afterEach(() => {
  cleanup()
  resetCustomization()
  // Again on the way out, for the next test FILE. Nothing in this file can
  // observe that one — no later file mounts a shell today — so it is
  // deliberately defensive rather than pinned.
  resetVersionCheckCache()
  act(() => {
    restoreProcessState(sharedStores, processSnapshot)
  })
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
