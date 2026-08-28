/**
 * Rollback-safe persist storage shared by every persisted store.
 *
 * An envelope written by a NEWER client (its `version` above this build's)
 * must be neither hydrated nor overwritten: hydrating would let zustand
 * restamp it with this build's version, and any later set() would clobber
 * fields this build does not know. The check runs on EVERY storage
 * operation — a newer client's tab can write its envelope while this tab is
 * already running — and while a future envelope is present, writes divert
 * to a session-only fallback (kept readable so a same-session rehydrate
 * does not silently reset to defaults), leaving the envelope byte-for-byte
 * intact for the client that wrote it.
 */

/** True when `key` holds an envelope whose version exceeds `currentVersion`. */
export function hasFutureEnvelope(
  key: string,
  currentVersion: number,
  storage: Pick<Storage, 'getItem'> | null = typeof localStorage === 'undefined'
    ? null
    : localStorage
): boolean {
  if (!storage) return false
  try {
    const raw = storage.getItem(key)
    if (raw == null) return false
    const version: unknown = JSON.parse(raw)?.version
    return typeof version === 'number' && version > currentVersion
  } catch {
    // Unreadable / corrupt envelope: not evidence of a newer client.
    return false
  }
}

/**
 * zustand `createJSONStorage` target implementing the guard above for a
 * store whose current envelope version is `currentVersion`. The key checked
 * is the one zustand passes in (the store's own persist name), and
 * `localStorage` is read at call time, never captured.
 */
export function createFutureGuardedStorage(currentVersion: number) {
  const sessionFallback = new Map<string, string>()
  const isFuture = (key: string) => hasFutureEnvelope(key, currentVersion)
  return {
    getItem: (key: string): string | null =>
      isFuture(key) ? (sessionFallback.get(key) ?? null) : localStorage.getItem(key),
    setItem: (key: string, value: string): void => {
      if (isFuture(key)) {
        sessionFallback.set(key, value)
        return
      }
      localStorage.setItem(key, value)
    },
    removeItem: (key: string): void => {
      sessionFallback.delete(key)
      if (!isFuture(key)) {
        localStorage.removeItem(key)
      }
    }
  }
}
