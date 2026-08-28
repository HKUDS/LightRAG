import { beforeAll, describe, expect, test } from 'bun:test'
import { SETTINGS_STORAGE_VERSION_AFTER_SPLIT } from '@/migrations/splitSettingsStorage'

/**
 * Rollback safety: an envelope written by a NEWER client must never be
 * reported as successfully migrated — zustand persists the migrate result
 * back stamped with THIS build's version (22), downgrading the newer
 * client's envelope and discarding its versioning. The migrate callback
 * throws instead, which aborts hydration WITHOUT the write-back (the
 * session-only storage guard in stores/settings.ts covers the creation-time
 * path; this pins the callback itself).
 *
 * Isolation: the module-cached singleton is only READ (persist options); no
 * state or module mock is touched. localStorage must exist BEFORE the store
 * module is imported — zustand omits the `persist` API entirely when its
 * storage resolves to undefined — so a stub is installed first and the store
 * imported dynamically (matching the other store test files).
 */

const storageStub = () => {
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

type Migrate = (state: unknown, version: number) => unknown
let migrate: Migrate

beforeAll(async () => {
  if (typeof localStorage === 'undefined') {
    Object.defineProperty(globalThis, 'localStorage', {
      value: storageStub(),
      configurable: true
    })
  }
  const { useSettingsStore } = await import('./settings')
  migrate = useSettingsStore.persist.getOptions().migrate as Migrate
})

describe('settings store migrate', () => {
  test('a FUTURE envelope version throws instead of being restamped as v22', () => {
    expect(() =>
      migrate({ theme: 'dark', futureField: 1 }, SETTINGS_STORAGE_VERSION_AFTER_SPLIT + 1)
    ).toThrow(/newer than this build/)
  })

  test('a legacy envelope (<= 21) still migrates and drops the split fields', () => {
    const result = migrate(
      {
        theme: 'dark',
        querySettings: { mode: 'hybrid', history_turns: 0 },
        retrievalHistory: [{ id: 'h1', role: 'user', content: 'q' }]
      },
      21
    ) as Record<string, unknown>
    expect(result.querySettings).toBeUndefined()
    expect(result.retrievalHistory).toBeUndefined()
    expect(result.theme).toBe('dark')
  })

  test('the current version (22) passes through unchanged', () => {
    const state = { theme: 'light' }
    expect(migrate(state, SETTINGS_STORAGE_VERSION_AFTER_SPLIT)).toBe(state)
  })
})
