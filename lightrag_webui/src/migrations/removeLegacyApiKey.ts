import { LEGACY_SETTINGS_STORAGE_KEY } from '@/lib/storageKeys'

export function withoutLegacyApiKey<T>(state: T): T {
  if (!state || typeof state !== 'object' || Array.isArray(state) || !('apiKey' in state)) return state
  const copy = { ...state } as T & { apiKey?: unknown }
  delete copy.apiKey
  return copy
}

/** Shrink only known envelopes, atomically; failed writes leave the original intact. */
export function removeLegacyApiKey(storage: Storage, maxVersion: number): void {
  const raw = storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)
  if (raw === null) return
  let envelope
  try { envelope = JSON.parse(raw) } catch { return }
  if (!envelope || typeof envelope !== 'object') return
  if (typeof envelope.version === 'number' && envelope.version > maxVersion) return
  const state = envelope.state
  if (!state || typeof state !== 'object' || Array.isArray(state) || !('apiKey' in state)) return
  storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, JSON.stringify({
    ...envelope, state: withoutLegacyApiKey(state)
  }))
}
