import { afterAll, beforeAll, describe, expect, mock, test } from 'bun:test'
import { QUERY_SETTINGS_STORAGE_KEY, WEBUI_RETRIEVAL_HISTORY_KEY } from '@/lib/storageKeys'
import type { NavigationEntryConfig } from '@/services/navigation'

/**
 * Workspace entry bootstrap: configures the navigation singleton for THIS
 * entry (unauthenticated → /welcome) and re-hydrates the shared query
 * settings on cross-entry `storage` events so /webui parameter saves take
 * effect from the next query without a reload.
 *
 * The navigation service is replaced with a RECORDING stub before the
 * bootstrap module is imported — mock.module is process-global in bun test,
 * so relying on the real singleton would make this file order-dependent on
 * other files' navigation mocks.
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

type StorageListener = (event: { key: string | null }) => void
const storageListeners: StorageListener[] = []
const recordedConfigs: NavigationEntryConfig[] = []

let querySettingsModule: typeof import('@/stores/querySettings')
let workspaceHistoryModule: typeof import('@/stores/workspaceRetrievalHistory')

beforeAll(async () => {
  Object.defineProperty(globalThis, 'localStorage', {
    value: storageMock(),
    configurable: true
  })
  Object.defineProperty(globalThis, 'sessionStorage', {
    value: storageMock(),
    configurable: true
  })
  Object.defineProperty(globalThis, 'window', {
    value: {
      addEventListener: (type: string, listener: StorageListener) => {
        if (type === 'storage') storageListeners.push(listener)
      }
    },
    configurable: true
  })

  mock.module('@/services/navigation', () => ({
    navigationService: {
      configureEntry: (config: NavigationEntryConfig) => {
        recordedConfigs.push(config)
      },
      setNavigate: () => {},
      navigateToUnauthenticated: () => {},
      navigateToHome: () => {},
      resetAllApplicationState: () => {},
      resetForTests: () => {}
    }
  }))

  querySettingsModule = await import('@/stores/querySettings')
  workspaceHistoryModule = await import('@/stores/workspaceRetrievalHistory')
  await import('./workspaceEntry')
})

afterAll(() => {
  delete (globalThis as Record<string, unknown>).window
})

describe('workspace bootstrap', () => {
  test('configures the unauthenticated target to /welcome (never the admin /login)', () => {
    expect(recordedConfigs).toHaveLength(1)
    expect(recordedConfigs[0].unauthenticatedRoute).toBe('/welcome')
    // No graph reset adapter on this entry — registering one would pull
    // stores/graph (and graphology) into the workspace first-load closure.
    expect(recordedConfigs[0].resetAdapters ?? []).toEqual([])
  })

  test('the injected history clearer clears the WORKSPACE history store', () => {
    workspaceHistoryModule.useWorkspaceRetrievalHistoryStore
      .getState()
      .setHistory([{ id: 'w', role: 'user', content: 'q' }])
    recordedConfigs[0].clearRetrievalHistory()
    expect(
      workspaceHistoryModule.useWorkspaceRetrievalHistoryStore.getState().history
    ).toEqual([])
  })

  // Write through the store's OWN persist storage: zustand v5's
  // createJSONStorage captures the Storage object once at store creation,
  // and other test files swap the global localStorage stub — writing to the
  // current global would race with whichever stub the singleton captured.
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
