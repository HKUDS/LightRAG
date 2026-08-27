import { beforeEach, describe, expect, test } from 'bun:test'
import {
  runSettingsStorageSplitMigration,
  getSettingsMigrationError,
  resetSettingsMigrationErrorForTests,
  SETTINGS_STORAGE_VERSION_AFTER_SPLIT
} from './splitSettingsStorage'
import {
  LEGACY_SETTINGS_STORAGE_KEY,
  QUERY_SETTINGS_STORAGE_KEY,
  WEBUI_RETRIEVAL_HISTORY_KEY,
  WORKSPACE_RETRIEVAL_HISTORY_KEY
} from '@/lib/storageKeys'

/** In-memory Storage double; `failAfterWrites` injects a mid-run crash. */
class FakeStorage implements Storage {
  data = new Map<string, string>()
  writeCount = 0
  failAfterWrites: number | null = null

  get length() {
    return this.data.size
  }
  clear() {
    this.data.clear()
  }
  getItem(key: string) {
    return this.data.get(key) ?? null
  }
  key(index: number) {
    return [...this.data.keys()][index] ?? null
  }
  removeItem(key: string) {
    this.data.delete(key)
  }
  setItem(key: string, value: string) {
    if (this.failAfterWrites !== null && this.writeCount >= this.failAfterWrites) {
      throw new Error('quota exceeded (injected)')
    }
    this.writeCount += 1
    this.data.set(key, value)
  }
}

const HISTORY = [
  { id: 'h1', role: 'user', content: 'admin debug question' },
  { id: 'h2', role: 'assistant', content: 'answer' }
]

const legacyEnvelope = (version: number, state: Record<string, unknown>) =>
  JSON.stringify({ state, version })

const v21State = () => ({
  theme: 'dark',
  language: 'zh',
  apiKey: 'secret',
  querySettings: { mode: 'hybrid', top_k: 7, history_turns: 0 },
  retrievalHistory: HISTORY
})

let storage: FakeStorage

beforeEach(() => {
  storage = new FakeStorage()
  resetSettingsMigrationErrorForTests()
})

const parsed = (key: string) => JSON.parse(storage.getItem(key)!)

describe('field mapping (fixed and exhaustive)', () => {
  test('querySettings → query-settings key, retrievalHistory → webui history, workspace history = []', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    runSettingsStorageSplitMigration(storage)

    expect(getSettingsMigrationError()).toBeNull()
    expect(parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
    // The admin history is copied verbatim…
    expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(HISTORY)
    // …and the workspace history has NO legacy source: it starts EMPTY.
    // Copying the admin debug record here is exactly the defect the split
    // removes (it would even feed bypass-mode conversation_history).
    expect(parsed(WORKSPACE_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])

    // Legacy envelope: migrated fields stripped, version bumped, rest kept.
    const legacy = parsed(LEGACY_SETTINGS_STORAGE_KEY)
    expect(legacy.version).toBe(SETTINGS_STORAGE_VERSION_AFTER_SPLIT)
    expect(legacy.state.querySettings).toBeUndefined()
    expect(legacy.state.retrievalHistory).toBeUndefined()
    expect(legacy.state.apiKey).toBe('secret')
    expect(legacy.state.theme).toBe('dark')
  })

  test('no legacy envelope → nothing written', () => {
    runSettingsStorageSplitMigration(storage)
    expect(storage.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()
    expect(getSettingsMigrationError()).toBeNull()
  })

  test('corrupt envelope treated as no legacy data', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, '{not json')
    runSettingsStorageSplitMigration(storage)
    expect(getSettingsMigrationError()).toBeNull()
    expect(storage.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()
    // Broken blob left in place.
    expect(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe('{not json')
  })
})

describe('version chain reuse (normalize before split)', () => {
  // Expected outcomes prove the REUSED chain ran with its version guards:
  // v1/v6 pass the v15 step (mode forced to 'mix') and the v12 step (history
  // cleared: pre-MessageWithError format); a v20 envelope passes neither, so
  // its own mode and history survive intact.
  test.each([
    [1, 'mix', []],
    [6, 'mix', []],
    [20, 'local', HISTORY]
  ] as const)(
    'v%i envelopes go through the legacy chain, not "no data"',
    (version, expectedMode, expectedHistory) => {
      const state: Record<string, unknown> = {
        theme: 'light',
        querySettings: { mode: 'local', history_turns: 0 },
        retrievalHistory: HISTORY
      }
      storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(version, state))
      runSettingsStorageSplitMigration(storage)

      expect(getSettingsMigrationError()).toBeNull()
      const qs = parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings
      expect(qs.mode).toBe(expectedMode)
      expect(qs.history_turns).toBe(0)
      expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(
        expectedHistory as unknown[]
      )
    }
  )

  test('version > cap: nothing read, cleaned or overwritten', () => {
    const envelope = legacyEnvelope(99, v21State())
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, envelope)
    runSettingsStorageSplitMigration(storage)

    expect(getSettingsMigrationError()).toBeNull()
    expect(storage.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()
    expect(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(envelope)
  })

  test('our own post-split version (22) is a no-op', () => {
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      legacyEnvelope(SETTINGS_STORAGE_VERSION_AFTER_SPLIT, { theme: 'dark' })
    )
    runSettingsStorageSplitMigration(storage)
    expect(storage.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()
  })
})

describe('new keys always win', () => {
  test('existing new keys are never overwritten by legacy values', () => {
    const userEdited = { state: { querySettings: { mode: 'naive' } }, version: 1 }
    storage.setItem(QUERY_SETTINGS_STORAGE_KEY, JSON.stringify(userEdited))
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))

    runSettingsStorageSplitMigration(storage)

    expect(parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('naive')
    // The other targets were still filled, and legacy was cleaned.
    expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(HISTORY)
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).state.querySettings).toBeUndefined()
  })

  test('triple execution: runs 2 and 3 are no-ops and do not clobber user edits', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    runSettingsStorageSplitMigration(storage)

    // User edits the new key after migration.
    storage.setItem(
      QUERY_SETTINGS_STORAGE_KEY,
      JSON.stringify({ state: { querySettings: { mode: 'global' } }, version: 1 })
    )
    const snapshotAfterEdit = new Map(storage.data)

    runSettingsStorageSplitMigration(storage)
    runSettingsStorageSplitMigration(storage)

    expect(storage.data).toEqual(snapshotAfterEdit)
  })

  test('old-tab rewrite: a full v21 envelope written back later is cleaned, never merged', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    runSettingsStorageSplitMigration(storage)

    // A tab still running pre-split code rewrites the whole legacy envelope
    // with DIFFERENT settings and history.
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      legacyEnvelope(21, {
        ...v21State(),
        querySettings: { mode: 'naive', top_k: 1, history_turns: 0 },
        retrievalHistory: [{ id: 'x', role: 'user', content: 'from old tab' }]
      })
    )

    runSettingsStorageSplitMigration(storage)

    // Cleaned…
    const legacy = parsed(LEGACY_SETTINGS_STORAGE_KEY)
    expect(legacy.version).toBe(SETTINGS_STORAGE_VERSION_AFTER_SPLIT)
    expect(legacy.state.querySettings).toBeUndefined()
    expect(legacy.state.retrievalHistory).toBeUndefined()
    // …but the new keys are unchanged: the old tab's post-migration edits
    // are explicitly allowed to be lost (this is a cleanup, not an absorb).
    expect(parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
    expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(HISTORY)
  })
})

describe('crash and partial-failure behavior', () => {
  test('write failure mid-copy: legacy untouched, error recorded, retry converges', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    storage.failAfterWrites = 2 // legacy write counted as 1; allow one target write

    runSettingsStorageSplitMigration(storage)

    // Physically partial — that is allowed…
    expect(getSettingsMigrationError()).not.toBeNull()
    // …but the legacy envelope keeps ALL its fields (cleanup never ran).
    const legacy = parsed(LEGACY_SETTINGS_STORAGE_KEY)
    expect(legacy.version).toBe(21)
    expect(legacy.state.querySettings).toBeDefined()
    expect(legacy.state.retrievalHistory).toBeDefined()
    // The interrupted run wrote NO defaults into not-yet-migrated keys:
    // whatever keys exist carry legacy-derived values, and at least one
    // target is still missing.
    const missing = [
      QUERY_SETTINGS_STORAGE_KEY,
      WEBUI_RETRIEVAL_HISTORY_KEY,
      WORKSPACE_RETRIEVAL_HISTORY_KEY
    ].filter((key) => storage.getItem(key) == null)
    expect(missing.length).toBeGreaterThan(0)

    // Retry (next startup): converges to the same result as one clean run.
    storage.failAfterWrites = null
    resetSettingsMigrationErrorForTests()
    runSettingsStorageSplitMigration(storage)

    expect(getSettingsMigrationError()).toBeNull()
    expect(parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
    expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(HISTORY)
    expect(parsed(WORKSPACE_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(
      SETTINGS_STORAGE_VERSION_AFTER_SPLIT
    )
  })

  test('crash after copy but before cleanup: re-run cleans without overwriting', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    storage.failAfterWrites = 4 // 1 legacy + 3 targets; fail on the cleanup write

    runSettingsStorageSplitMigration(storage)
    expect(getSettingsMigrationError()).not.toBeNull()
    // All targets written, legacy still v21 with fields.
    expect(storage.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)).not.toBeNull()
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(21)

    storage.failAfterWrites = null
    resetSettingsMigrationErrorForTests()
    runSettingsStorageSplitMigration(storage)

    expect(getSettingsMigrationError()).toBeNull()
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(
      SETTINGS_STORAGE_VERSION_AFTER_SPLIT
    )
    expect(parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
  })

  test('workspace entry may execute the one-time migration and writes ALL targets', () => {
    // The migration is entry-agnostic: whichever entry runs first writes the
    // full set (including the webui history key).
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    runSettingsStorageSplitMigration(storage)
    expect(storage.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)).not.toBeNull()
    expect(storage.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)).not.toBeNull()
    expect(storage.getItem(QUERY_SETTINGS_STORAGE_KEY)).not.toBeNull()
  })
})
