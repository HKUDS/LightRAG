import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import {
  getSettingsMigrationError,
  resetSettingsMigrationErrorForTests,
  runSettingsStorageSplitMigration,
  SETTINGS_STORAGE_VERSION_AFTER_SPLIT
} from '@/migrations/splitSettingsStorage'
import { LEGACY_SETTINGS_STORAGE_KEY } from '@/lib/storageKeys'

/**
 * Rollback safety: an envelope written by a NEWER client must never be
 * hydrated-and-restamped as v22 NOR clobbered by this tab's later settings
 * writes. Two mechanisms are pinned here:
 * - the `migrate` callback throws for future versions (zustand would
 *   otherwise persist the migrate result back stamped v22);
 * - the storage adapter re-checks the envelope on EVERY operation, so a v23
 *   envelope arriving mid-session from a newer tab diverts this tab's writes
 *   to a session-only fallback instead of overwriting it.
 *
 * Isolation: the singleton's storage adapter reads globalThis.localStorage
 * dynamically at call time, so this file installs its own stub and RESTORES
 * the previous binding in afterAll. localStorage must exist BEFORE the store
 * module is imported — zustand omits the `persist` API entirely when its
 * storage resolves to undefined — hence the dynamic import.
 */

const data = new Map<string, string>()
const stub = {
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

let previousDescriptor: PropertyDescriptor | undefined
type Migrate = (state: unknown, version: number) => unknown
let migrate: Migrate
let store: typeof import('./settings').useSettingsStore

beforeAll(async () => {
  previousDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', {
    value: stub,
    configurable: true
  })
  store = (await import('./settings')).useSettingsStore
  migrate = store.persist.getOptions().migrate as Migrate
})

afterAll(() => {
  if (previousDescriptor) {
    Object.defineProperty(globalThis, 'localStorage', previousDescriptor)
  } else {
    delete (globalThis as Record<string, unknown>).localStorage
  }
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
        language: 'fr',
        querySettings: { mode: 'hybrid', history_turns: 0 },
        retrievalHistory: [{ id: 'h1', role: 'user', content: 'q' }]
      },
      21
    ) as Record<string, unknown>
    expect(result.querySettings).toBeUndefined()
    expect(result.retrievalHistory).toBeUndefined()
    expect(result.theme).toBe('dark')
    // A non-default legacy language proves an explicit choice — synthesized
    // so the browser language cannot silently replace it after the upgrade.
    expect(result.languageUserSelected).toBe(true)
  })

  test('a legacy envelope with the default \'en\' stays non-explicit', () => {
    const result = migrate({ language: 'en' }, 21) as Record<string, unknown>
    expect(result.languageUserSelected).toBe(false)
  })

  test('the current version (22) passes through unchanged', () => {
    const state = { theme: 'light' }
    expect(migrate(state, SETTINGS_STORAGE_VERSION_AFTER_SPLIT)).toBe(state)
  })
})

describe('settings storage guard (per-operation, cross-tab safe)', () => {
  const persistStorage = () => store.persist.getOptions().storage!

  test('a write AFTER a newer tab stored v23 diverts — the future envelope survives byte-for-byte', async () => {
    const futureEnvelope = JSON.stringify({
      state: { theme: 'dark', futureField: 1 },
      version: SETTINGS_STORAGE_VERSION_AFTER_SPLIT + 1
    })
    stub.setItem(LEGACY_SETTINGS_STORAGE_KEY, futureEnvelope)

    await persistStorage().setItem(LEGACY_SETTINGS_STORAGE_KEY, {
      state: { theme: 'light' },
      version: SETTINGS_STORAGE_VERSION_AFTER_SPLIT
    } as never)

    // localStorage still holds the newer client's envelope untouched…
    expect(stub.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(futureEnvelope)
    // …while the diverted write stays readable within this session.
    const diverted = (await persistStorage().getItem(LEGACY_SETTINGS_STORAGE_KEY)) as {
      version: number
      state: { theme: string }
    }
    expect(diverted.version).toBe(SETTINGS_STORAGE_VERSION_AFTER_SPLIT)
    expect(diverted.state.theme).toBe('light')
  })

  test('without a future envelope, writes reach localStorage as before', async () => {
    stub.removeItem(LEGACY_SETTINGS_STORAGE_KEY)

    await persistStorage().setItem(LEGACY_SETTINGS_STORAGE_KEY, {
      state: { theme: 'system' },
      version: SETTINGS_STORAGE_VERSION_AFTER_SPLIT
    } as never)

    const written = JSON.parse(stub.getItem(LEGACY_SETTINGS_STORAGE_KEY)!)
    expect(written.version).toBe(SETTINGS_STORAGE_VERSION_AFTER_SPLIT)
    expect(written.state.theme).toBe('system')
    stub.removeItem(LEGACY_SETTINGS_STORAGE_KEY)
  })

  test('after a FAILED split migration, writes divert instead of replacing the legacy envelope', async () => {
    // Drive the REAL failure path: a storage whose writes throw (quota-style)
    // records the migration error exactly as production would.
    const legacyEnvelope = JSON.stringify({
      state: {
        theme: 'dark',
        querySettings: { mode: 'hybrid', history_turns: 0 },
        retrievalHistory: [{ id: 'h1', role: 'user', content: 'unmigrated' }]
      },
      version: 21
    })
    const failingStorage = {
      getItem: (key: string) =>
        key === LEGACY_SETTINGS_STORAGE_KEY ? legacyEnvelope : null,
      setItem: () => {
        throw new Error('quota exceeded (injected)')
      },
      removeItem: () => {},
      clear: () => {},
      key: () => null,
      length: 0
    } as Storage
    runSettingsStorageSplitMigration(failingStorage)
    expect(getSettingsMigrationError()).not.toBeNull()

    try {
      // The unmigrated envelope sits in localStorage; a set() (e.g. the i18n
      // language sync) must NOT replace it with a default-heavy v22 envelope
      // — that would make the next load treat the split as completed and
      // lose the legacy query settings/history forever.
      stub.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope)
      await persistStorage().setItem(LEGACY_SETTINGS_STORAGE_KEY, {
        state: { theme: 'light' },
        version: SETTINGS_STORAGE_VERSION_AFTER_SPLIT
      } as never)

      expect(stub.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(legacyEnvelope)
    } finally {
      resetSettingsMigrationErrorForTests()
      stub.removeItem(LEGACY_SETTINGS_STORAGE_KEY)
    }
  })
})
