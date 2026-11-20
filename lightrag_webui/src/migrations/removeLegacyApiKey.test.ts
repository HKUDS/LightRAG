import { afterEach, describe, expect, test } from 'bun:test'
import { removeLegacyApiKey } from './removeLegacyApiKey'
import { getSettingsMigrationError, resetSettingsMigrationErrorForTests,
  runSettingsStorageMigration, SETTINGS_STORAGE_VERSION_AFTER_SPLIT } from './splitSettingsStorage'
import { LEGACY_SETTINGS_STORAGE_KEY } from '@/lib/storageKeys'
import { useSettingsStore } from '@/stores/settings'

const version = SETTINGS_STORAGE_VERSION_AFTER_SPLIT
class MemoryStorage implements Storage {
  data = new Map<string, string>()
  writes = 0
  fail = false
  get length() { return this.data.size }
  clear() { this.data.clear() }
  key(index: number) { return [...this.data.keys()][index] ?? null }
  getItem(key: string) { return this.data.get(key) ?? null }
  removeItem(key: string) { this.data.delete(key) }
  setItem(key: string, value: string) {
    if (this.fail) throw new Error('storage unavailable')
    this.writes++
    this.data.set(key, value)
  }
}
const seed = (v = version) => {
  const storage = new MemoryStorage()
  storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, JSON.stringify({ version: v,
    metadata: 'keep-envelope-metadata', state: { apiKey: 'legacy-key', theme: 'dark',
      language: 'zh', queryLabel: 'keep', userPromptHistory: ['prompt'] } }))
  storage.setItem('LIGHTRAG-API-TOKEN', 'keep-token')
  storage.setItem('workspace-query-settings-storage', 'keep-workspace')
  return storage
}
afterEach(() => resetSettingsMigrationErrorForTests())

describe('legacy API-key removal', () => {
  test('same-version cleanup removes only apiKey and is idempotent', () => {
    const storage = seed()
    const before = JSON.parse(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)!)
    runSettingsStorageMigration(storage)
    const after = JSON.parse(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)!)
    delete before.state.apiKey
    expect(after).toEqual(before)
    expect(storage.getItem('LIGHTRAG-API-TOKEN')).toBe('keep-token')
    expect(storage.getItem('workspace-query-settings-storage')).toBe('keep-workspace')
    const writes = storage.writes
    runSettingsStorageMigration(storage)
    expect(storage.writes).toBe(writes)
    expect(getSettingsMigrationError()).toBeNull()
  })

  test('future envelopes remain byte-for-byte unchanged', () => {
    const storage = seed(version + 1)
    const original = storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)
    runSettingsStorageMigration(storage)
    expect(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(original)
  })

  test('failed atomic cleanup preserves the original and blocks hydration until retry', () => {
    const storage = seed()
    const original = storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)
    storage.fail = true
    runSettingsStorageMigration(storage)
    expect(getSettingsMigrationError()).toBeInstanceOf(Error)
    expect(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(original)
    storage.fail = false
    runSettingsStorageMigration(storage)
    expect(getSettingsMigrationError()).toBeNull()
    expect(JSON.parse(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)!).state.apiKey).toBeUndefined()
  })

  test('an existing data-loss marker remains a manual-acceptance barrier', () => {
    const storage = seed()
    const envelope = JSON.parse(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)!)
    envelope.lightragSettingsLostToQuota = true
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, JSON.stringify(envelope))
    const original = storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)
    runSettingsStorageMigration(storage)
    expect(getSettingsMigrationError()).toBeInstanceOf(Error)
    expect(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(original)
  })

  test('cleanup composes after the legacy storage split', () => {
    const storage = seed(21)
    runSettingsStorageMigration(storage)
    expect(getSettingsMigrationError()).toBeNull()
    const envelope = JSON.parse(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)!)
    expect(envelope.version).toBe(version)
    expect(envelope.state.apiKey).toBeUndefined()
    expect(envelope.state.theme).toBe('dark')
  })

  test.each(['invalid json', 'null', '{"state":null}', '{"state":[]}'])(
    'does not destructively rewrite malformed storage: %s', (raw) => {
      const storage = new MemoryStorage()
      storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, raw)
      removeLegacyApiKey(storage, version)
      expect(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(raw)
    })

  test('same-version hydration and subsequent persistence cannot resurrect a legacy key', () => {
    const options = useSettingsStore.persist.getOptions()
    const merged = options.merge!({ apiKey: 'old-tab-key', theme: 'dark' }, useSettingsStore.getState())
    expect('apiKey' in merged).toBe(false)
    expect(merged.theme).toBe('dark')
    const persisted = options.partialize!({ ...merged, apiKey: 'old-tab-key' } as typeof merged)
    expect('apiKey' in (persisted as object)).toBe(false)
  })
})
