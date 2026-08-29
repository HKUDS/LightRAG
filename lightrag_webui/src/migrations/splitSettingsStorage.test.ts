import { beforeEach, describe, expect, test } from 'bun:test'
import {
  runSettingsStorageSplitMigration,
  getSettingsMigrationError,
  hasFutureSettingsEnvelope,
  resetSettingsMigrationErrorForTests,
  wasHistoryDroppedDuringMigration,
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
  // Quota-realistic mode: values LARGER than this fail, small ones succeed
  // (a real quota keeps accepting small writes after rejecting a big one).
  failValueLongerThan: number | null = null
  // 'Error' simulates a generic crash; 'QuotaExceededError' the quota path.
  failureName = 'Error'

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
    const overQuota =
      this.failValueLongerThan !== null && value.length > this.failValueLongerThan
    const crashed =
      this.failAfterWrites !== null && this.writeCount >= this.failAfterWrites
    if (overQuota || crashed) {
      const error = new Error('write failed (injected)')
      error.name = this.failureName
      throw error
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

describe('hasFutureSettingsEnvelope (rule 3 extended to the settings store)', () => {
  test('an envelope from a NEWER client (version > 22) is detected', () => {
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      legacyEnvelope(SETTINGS_STORAGE_VERSION_AFTER_SPLIT + 1, { theme: 'dark' })
    )
    expect(hasFutureSettingsEnvelope(storage)).toBe(true)
  })

  test('current, legacy, absent and corrupt envelopes are NOT future', () => {
    // Current (22)
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      legacyEnvelope(SETTINGS_STORAGE_VERSION_AFTER_SPLIT, { theme: 'dark' })
    )
    expect(hasFutureSettingsEnvelope(storage)).toBe(false)
    // Legacy (21)
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    expect(hasFutureSettingsEnvelope(storage)).toBe(false)
    // Non-numeric version
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      JSON.stringify({ state: {}, version: 'later' })
    )
    expect(hasFutureSettingsEnvelope(storage)).toBe(false)
    // Corrupt JSON
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, '{not json')
    expect(hasFutureSettingsEnvelope(storage)).toBe(false)
    // Absent
    storage.removeItem(LEGACY_SETTINGS_STORAGE_KEY)
    expect(hasFutureSettingsEnvelope(storage)).toBe(false)
    expect(hasFutureSettingsEnvelope(null)).toBe(false)
  })
})

describe('legacy language choice synthesis (languageUserSelected)', () => {
  // Legacy `language` only ever changed through the selector, so a persisted
  // NON-DEFAULT value proves an explicit choice; the marker is synthesized
  // during the split so an upgrade cannot let the browser language silently
  // replace it. A persisted 'en' is ambiguous and stays non-explicit.
  test('a non-default legacy language is marked as an explicit choice', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State())) // language: 'zh'
    runSettingsStorageSplitMigration(storage)
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).state.languageUserSelected).toBe(true)
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).state.language).toBe('zh')
  })

  test('the default \'en\' (or a missing language) stays non-explicit', () => {
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      legacyEnvelope(21, { ...v21State(), language: 'en' })
    )
    runSettingsStorageSplitMigration(storage)
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).state.languageUserSelected).toBe(false)

    const withoutLanguage: Record<string, unknown> = { ...v21State() }
    delete withoutLanguage.language
    storage = new FakeStorage()
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, withoutLanguage))
    runSettingsStorageSplitMigration(storage)
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).state.languageUserSelected).toBe(false)
  })
})

describe('quota-aware degradation (history is non-critical test data)', () => {
  // Product decision: a legacy history near the browser quota cannot coexist
  // with its migrated copy. Rather than locking the user into a migration
  // error whose retry can never succeed, the split retries WITHOUT the
  // history copy, records the drop (surfaced as a one-time toast by the
  // entry shells), and everything else migrates normally.
  test('QuotaExceededError on the history copy drops the history and completes', () => {
    // A history too large to duplicate: the quota rejects its serialized
    // copy while every other (small) write keeps succeeding — which is what
    // lets the degraded retry converge WITHOUT any quota being freed.
    const bigHistory = [{ id: 'h1', role: 'user', content: 'x'.repeat(4000) }]
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      legacyEnvelope(21, { ...v21State(), retrievalHistory: bigHistory })
    )
    storage.failValueLongerThan = 2000
    storage.failureName = 'QuotaExceededError'

    runSettingsStorageSplitMigration(storage)

    expect(getSettingsMigrationError()).toBeNull()
    expect(wasHistoryDroppedDuringMigration()).toBe(true)

    // Everything except the history migrated normally…
    expect(parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(
      SETTINGS_STORAGE_VERSION_AFTER_SPLIT
    )
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).state.retrievalHistory).toBeUndefined()
    // …and the history keys hold EMPTY histories, not the oversized copy.
    expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])
    expect(parsed(WORKSPACE_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])
  })

  test('a GENERIC write failure keeps the original fail-and-retry semantics', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    storage.failAfterWrites = 2
    storage.failureName = 'Error'

    runSettingsStorageSplitMigration(storage)

    expect(getSettingsMigrationError()).not.toBeNull()
    expect(wasHistoryDroppedDuringMigration()).toBe(false)
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(21)
  })

  test('a clean run reports no drop', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    runSettingsStorageSplitMigration(storage)
    expect(wasHistoryDroppedDuringMigration()).toBe(false)
    expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(HISTORY)
  })
})

/**
 * Total-byte quota — the model in which a large write can SUCCEED and a
 * LATER write then fails because the budget is exhausted. `freeOnReplace`
 * covers both real browser behaviors: some free a replaced key's old bytes
 * before checking, some check against the current total.
 */
class ByteQuotaStorage implements Storage {
  data = new Map<string, string>()
  removed: string[] = []
  constructor(
    public budget: number,
    private freeOnReplace = true
  ) {}

  get length() {
    return this.data.size
  }
  private used(exceptKey?: string) {
    let total = 0
    for (const [key, value] of this.data) {
      if (key !== exceptKey) total += key.length + value.length
    }
    return total
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
    this.removed.push(key)
    this.data.delete(key)
  }
  setItem(key: string, value: string) {
    const used = this.freeOnReplace ? this.used(key) : this.used()
    if (used + key.length + value.length > this.budget) {
      const error = new Error('quota exceeded (byte budget)')
      error.name = 'QuotaExceededError'
      throw error
    }
    this.data.set(key, value)
  }
}

describe('quota degradation under a byte budget', () => {
  const bigHistory = [{ id: 'h1', role: 'user', content: 'x'.repeat(1200) }]
  const legacyWithBigHistory = () =>
    legacyEnvelope(21, { ...v21State(), retrievalHistory: bigHistory })

  const expectCompletedWithEmptyHistory = (quotaStorage: ByteQuotaStorage) => {
    const read = (key: string) => JSON.parse(quotaStorage.getItem(key)!)
    // No dead end: the split completed under the degradation policy…
    expect(getSettingsMigrationError()).toBeNull()
    expect(wasHistoryDroppedDuringMigration()).toBe(true)
    // …every target exists, with the history dropped to empty…
    expect(read(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
    expect(read(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])
    expect(read(WORKSPACE_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])
    // …and the legacy envelope was cleaned and bumped, so the next reload is
    // a no-op instead of repeating the failure forever.
    expect(read(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(
      SETTINGS_STORAGE_VERSION_AFTER_SPLIT
    )
    expect(read(LEGACY_SETTINGS_STORAGE_KEY).state.retrievalHistory).toBeUndefined()
  }

  test('the history copy itself does not fit: the retry writes an empty one', () => {
    const quotaStorage = new ByteQuotaStorage(2100)
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyWithBigHistory())

    runSettingsStorageSplitMigration(quotaStorage)

    expectCompletedWithEmptyHistory(quotaStorage)
    // Nothing to reclaim — the oversized write never landed.
    expect(quotaStorage.removed).toEqual([])
  })

  test('the copy LANDS and the cleanup hits the quota: the history is KEPT', () => {
    // Regression: this used to reclaim the migrated copy — the one piece of
    // data the split exists to carry across — and rewrite it EMPTY, even
    // though it had provably fit. The copy is phase 1's last target and
    // `writeIfAbsent` records a key only once its write returned, so a
    // failure with the copy present pins the blame on the ONE write after
    // it: the legacy-envelope cleanup. That payload is SMALLER than what it
    // replaces (it sheds querySettings and the history), so the rejection is
    // an engine that cannot shrink a key in place, not a shortage — removing
    // the key first completes the split with nothing lost.
    //
    // Budget calibrated so the history copy still fits (total 2908 B) while
    // the cleanup write that follows it does not (needs 3025 B); after the
    // remove it needs only 1614 B.
    const quotaStorage = new ByteQuotaStorage(2950, false)
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyWithBigHistory())

    runSettingsStorageSplitMigration(quotaStorage)

    const read = (key: string) => JSON.parse(quotaStorage.getItem(key)!)
    // The legacy envelope is the key being rewritten, so it is the one that
    // gets removed; the migrated copy must NOT be (the scratch key of the
    // headroom probe is bookkeeping, so assert the intent, not the list)…
    expect(quotaStorage.removed).toContain(LEGACY_SETTINGS_STORAGE_KEY)
    expect(quotaStorage.removed).not.toContain(WEBUI_RETRIEVAL_HISTORY_KEY)
    // …the migrated history survived in full, and no degradation was claimed…
    expect(read(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(bigHistory)
    expect(wasHistoryDroppedDuringMigration()).toBe(false)
    expect(getSettingsMigrationError()).toBeNull()
    // …and the split still completed: every target exists and the legacy
    // envelope is cleaned and bumped, so the next reload is a no-op.
    expect(read(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
    expect(read(WORKSPACE_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])
    expect(read(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(
      SETTINGS_STORAGE_VERSION_AFTER_SPLIT
    )
    expect(read(LEGACY_SETTINGS_STORAGE_KEY).state.retrievalHistory).toBeUndefined()
  })

  test('the cleanup stays impossible even after the remove: the raw envelope returns', () => {
    // The other side of the shrink-safe write: when removing the key does
    // NOT make room either (another origin ate the budget between the two
    // statements), the exact bytes just removed go back. A run that left NO
    // legacy envelope would let the NEXT run read "no legacy data" and
    // declare the split done over fields it never migrated.
    const quotaStorage = new ByteQuotaStorage(2950, false)
    const raw = legacyWithBigHistory()
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, raw)
    const realSetItem = quotaStorage.setItem.bind(quotaStorage)
    // Reject every legacy-envelope write, including the post-remove retry;
    // the restore of the original bytes must still be allowed through.
    quotaStorage.setItem = (key: string, value: string) => {
      if (key === LEGACY_SETTINGS_STORAGE_KEY && value !== raw) {
        const error = new Error('quota exceeded (byte budget)')
        error.name = 'QuotaExceededError'
        throw error
      }
      realSetItem(key, value)
    }

    runSettingsStorageSplitMigration(quotaStorage)

    // Rules 6/7: the failure is recorded so dependent stores refuse to
    // hydrate and the entry offers a retry…
    expect(getSettingsMigrationError()).not.toBeNull()
    // …and the legacy envelope is back, byte for byte, rather than absent.
    expect(quotaStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(raw)
  })

  test('the FIRST target write already fails: the legacy history is freed', () => {
    // The dead end this covers: when the legacy envelope alone leaves less
    // room than the small query-settings target needs, nothing of ours ever
    // lands, so there is nothing to reclaim and the degraded retry repeats
    // the identical failing write — on this run and on every reload, pinning
    // the user on MigrationErrorScreen with a retry that can never succeed.
    // The only bytes left to free are the legacy history's own, and by then
    // the degradation has already decided to discard it.
    const quotaStorage = new ByteQuotaStorage(1450, false)
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyWithBigHistory())

    runSettingsStorageSplitMigration(quotaStorage)

    // Nothing of ours had landed, so nothing of ours was reclaimed…
    expect(quotaStorage.removed).not.toContain(WEBUI_RETRIEVAL_HISTORY_KEY)
    // …and the split still completed under the degradation policy.
    expectCompletedWithEmptyHistory(quotaStorage)
  })

  test('the same case on an engine that CAN shrink a key in place', () => {
    // freeOnReplace: the legacy envelope is rewritten smaller directly, so
    // the key is never removed and no crash window exists at all.
    const quotaStorage = new ByteQuotaStorage(1450)
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyWithBigHistory())

    runSettingsStorageSplitMigration(quotaStorage)

    expect(quotaStorage.removed).toEqual([])
    expectCompletedWithEmptyHistory(quotaStorage)
  })

  test('a quota that stays exhausted is reported, not silently declared done', () => {
    // Freeing the legacy history cannot always be enough — here another key
    // holds the store. The run must record the failure so dependent stores
    // refuse to hydrate and the entry offers a retry (rules 6/7), and it must
    // leave a legacy envelope behind: a run that found NO envelope would
    // declare the split done over data it never migrated.
    const quotaStorage = new ByteQuotaStorage(10000, false)
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyWithBigHistory())
    quotaStorage.setItem('unrelated::blob', 'y'.repeat(1000))
    // The quota tightens under us (other origins' data, another tab's writes).
    quotaStorage.budget = 1250

    runSettingsStorageSplitMigration(quotaStorage)

    expect(getSettingsMigrationError()).not.toBeNull()
    expect(wasHistoryDroppedDuringMigration()).toBe(false)
    expect(quotaStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).not.toBeNull()
  })

  test('an ALREADY-over-quota store keeps its envelope instead of losing it', () => {
    // Regression. The rewrite path removes the key first so an engine that
    // cannot replace a value in place can still shrink it — sound while the
    // store has headroom, because writing back something smaller lowers the
    // total. On a store already OVER quota it was not: the retry failed, the
    // restore of the exact bytes just removed failed too (the comment
    // claiming it "always fits" was the wrong premise), and the envelope was
    // gone. The reload below is the real damage — the next run reads "no
    // legacy data", returns early with NO error, and the app starts on
    // defaults having migrated nothing.
    //
    // freeOnReplace: true is the engine that DOES release the replaced value
    // before checking, i.e. the one that never needed the removal at all.
    const quotaStorage = new ByteQuotaStorage(100000, true)
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyWithBigHistory())
    quotaStorage.setItem('unrelated::blob', 'y'.repeat(1500))
    // The quota shrinks below what the store already holds.
    quotaStorage.budget = 1411

    runSettingsStorageSplitMigration(quotaStorage)

    // Nothing could be written, so the run reports rather than pretending…
    expect(getSettingsMigrationError()).not.toBeNull()
    // …and the envelope is still there, unchanged.
    expect(quotaStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(
      legacyWithBigHistory()
    )

    // The reload: it must still find legacy data to work on, never conclude
    // the split is done. (Space is still short, so it reports again.)
    resetSettingsMigrationErrorForTests()
    runSettingsStorageSplitMigration(quotaStorage)

    expect(getSettingsMigrationError()).not.toBeNull()
    expect(quotaStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(
      legacyWithBigHistory()
    )
  })

  test('a PRE-EXISTING history key is never reclaimed (rule 4 still wins)', () => {
    const quotaStorage = new ByteQuotaStorage(2950, false)
    const userHistory = JSON.stringify({
      state: { history: [{ id: 'u1', role: 'user', content: 'the user own record' }] },
      version: 1
    })
    quotaStorage.setItem(WEBUI_RETRIEVAL_HISTORY_KEY, userHistory)
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyWithBigHistory())

    runSettingsStorageSplitMigration(quotaStorage)

    // A key this run did NOT create survives untouched, whatever the quota did.
    expect(quotaStorage.removed).toEqual([])
    expect(quotaStorage.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)).toBe(userHistory)
  })
})
