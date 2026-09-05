import { afterAll, beforeAll, describe, expect, spyOn, test } from 'bun:test'
import { QUERY_SETTINGS_STORAGE_KEY, WEBUI_RETRIEVAL_HISTORY_KEY } from '@/lib/storageKeys'
import type { navigationService as NavigationServiceType } from '@/services/navigation'

/**
 * Workspace entry bootstrap: configures the navigation singleton for THIS
 * entry (unauthenticated → /welcome) and re-hydrates the shared query
 * settings on cross-entry `storage` events so /webui parameter saves take
 * effect from the next query without a reload.
 *
 * Isolation rules this file follows (learned the hard way):
 * - NEVER replace the `window` GLOBAL, not even with a stub carrying just
 *   the one method the bootstrap calls. The preloaded happy-dom window is
 *   process-wide, and the import graph reached from `./workspaceEntry` pulls
 *   in axios, which reads `window.location.href` while its platform module
 *   is being evaluated. A stub without `location` therefore threw here — but
 *   only when this file happened to run before whichever other file
 *   evaluates axios first, which left `@/api/lightrag` half-evaluated in the
 *   registry and cascaded into TDZ errors across ten unrelated files. The
 *   bootstrap's listener registration is observed with a call-through spy on
 *   the REAL window instead, and the captured listeners are removed again in
 *   afterAll so nothing survives this file.
 * - NO `mock.module` on shared singletons: bun's module mocks live in the
 *   process-wide registry for the rest of the run and would hand every later
 *   test file a gutted navigationService. The bootstrap's configureEntry call
 *   is observed with a RESTORABLE spy on the REAL service instead
 *   (mockRestore + resetForTests in afterAll).
 * - Persisted state is read/written through the store's OWN persist storage
 *   (zustand v5's createJSONStorage captures the Storage object once at
 *   store creation, so swapping globalThis.localStorage after the singleton
 *   was created elsewhere would read/write a different object).
 */

const storageMock = () => {
  const data = new Map<string, string>()
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value)
    },
    removeItem: (key: string) => {
      data.delete(key)
    },
    clear: () => {
      data.clear()
    }
  }
}

type StorageListener = (event: {
  key: string | null
  oldValue?: string | null
  newValue?: string | null
}) => void
const storageListeners: StorageListener[] = []

let navigationService: typeof NavigationServiceType
let configureSpy: ReturnType<typeof spyOn> | undefined
let addEventListenerSpy: ReturnType<typeof spyOn> | undefined
let querySettingsModule: typeof import('@/stores/querySettings')
let workspaceHistoryModule: typeof import('@/stores/workspaceRetrievalHistory')

beforeAll(async () => {
  if (typeof localStorage === 'undefined') {
    Object.defineProperty(globalThis, 'localStorage', {
      value: storageMock(),
      configurable: true
    })
  }
  if (typeof sessionStorage === 'undefined') {
    Object.defineProperty(globalThis, 'sessionStorage', {
      value: storageMock(),
      configurable: true
    })
  }
  // Call-through spy on the real window: records what the bootstrap
  // registers AND still registers it, so no window property goes missing
  // from the modules imported below. See the isolation rules above.
  addEventListenerSpy = spyOn(window, 'addEventListener')

  navigationService = (await import('@/services/navigation')).navigationService
  // Call-through spy: records the bootstrap's configuration AND still applies
  // it to the real singleton (so the navigate assertion below is end-to-end).
  configureSpy = spyOn(navigationService, 'configureEntry')

  querySettingsModule = await import('@/stores/querySettings')
  workspaceHistoryModule = await import('@/stores/workspaceRetrievalHistory')
  await import('./workspaceEntry')

  for (const [type, listener] of addEventListenerSpy.mock.calls) {
    if (type === 'storage') storageListeners.push(listener as StorageListener)
  }
})

afterAll(() => {
  // The listeners went onto the process-wide window, so they would answer a
  // real `storage` event dispatched by any later test file.
  for (const listener of storageListeners) {
    window.removeEventListener('storage', listener as never)
  }
  addEventListenerSpy?.mockRestore()
  configureSpy?.mockRestore()
  navigationService.resetForTests()
})

const recordedConfig = () => {
  expect(configureSpy!.mock.calls).toHaveLength(1)
  return configureSpy!.mock.calls[0][0] as import('@/services/navigation').NavigationEntryConfig
}

describe('workspace bootstrap', () => {
  test('configures the unauthenticated target to /welcome (never the admin /login)', () => {
    const config = recordedConfig()
    expect(config.unauthenticatedRoute).toBe('/welcome')
    // No graph reset adapter on this entry — registering one would pull
    // stores/graph (and graphology) into the workspace first-load closure.
    expect(config.resetAdapters ?? []).toEqual([])
  })

  test('an unauthenticated navigation on the configured singleton lands on /welcome', () => {
    const targets: string[] = []
    navigationService.setNavigate(((to: string) => targets.push(to)) as never)
    navigationService.navigateToUnauthenticated()
    expect(targets).toEqual(['/welcome'])
  })

  test('the injected history clearer clears the WORKSPACE history store', () => {
    workspaceHistoryModule.useWorkspaceRetrievalHistoryStore
      .getState()
      .setHistory([{ id: 'w', role: 'user', content: 'q' }])
    recordedConfig().clearRetrievalHistory()
    expect(
      workspaceHistoryModule.useWorkspaceRetrievalHistoryStore.getState().history
    ).toEqual([])
  })

  // Write through the store's OWN persist storage — see the isolation rules
  // in the module docs above.
  const writePersisted = (querySettings: Record<string, unknown>) => {
    const storage = querySettingsModule.useQuerySettingsStore.persist.getOptions()
      .storage!
    void storage.setItem(QUERY_SETTINGS_STORAGE_KEY, {
      state: { querySettings },
      version: 1
    } as never)
  }

  test('a storage event for the query-settings key re-hydrates the store', async () => {
    // /webui (another tab) saves new parameters directly into localStorage.
    writePersisted({ mode: 'global', top_k: 7 })
    expect(storageListeners.length).toBeGreaterThan(0)
    for (const listener of storageListeners) {
      listener({ key: QUERY_SETTINGS_STORAGE_KEY })
    }
    await querySettingsModule.useQuerySettingsStore.persist.rehydrate()
    const settings = querySettingsModule.useQuerySettingsStore.getState().querySettings
    expect(settings.mode).toBe('global')
    expect(settings.top_k).toBe(7)
  })

  test('a cross-tab identity change clears the workspace history and bumps the session epoch', async () => {
    const { useIdentityEpochStore } = await import('@/lib/loginIdentity')
    workspaceHistoryModule.useWorkspaceRetrievalHistoryStore
      .getState()
      .setHistory([{ id: 'w-old', role: 'user', content: 'previous identity' }])
    const epochBefore = useIdentityEpochStore.getState().epoch

    // Another tab records a different identity (storage events only fire on
    // a genuine value change, and only in OTHER tabs).
    for (const listener of storageListeners) {
      listener({ key: 'LIGHTRAG-PREVIOUS-USER', oldValue: 'alice', newValue: 'guest' })
    }

    expect(
      workspaceHistoryModule.useWorkspaceRetrievalHistoryStore.getState().history
    ).toEqual([])
    // The epoch bump remounts the query view, dropping its live message state.
    expect(useIdentityEpochStore.getState().epoch).toBe(epochBefore + 1)
  })

  test('storage events for OTHER keys do not touch the settings store', () => {
    const before = querySettingsModule.useQuerySettingsStore.getState().querySettings
    writePersisted({ mode: 'naive', top_k: 1 })
    for (const listener of storageListeners) {
      listener({ key: WEBUI_RETRIEVAL_HISTORY_KEY })
    }
    expect(querySettingsModule.useQuerySettingsStore.getState().querySettings).toEqual(
      before
    )
  })
})
