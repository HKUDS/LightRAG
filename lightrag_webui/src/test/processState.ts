/**
 * Snapshot and restore the process-wide state a mounted component writes
 * OUTSIDE React: zustand store singletons and `localStorage`.
 *
 * Bun evaluates every test file in one process, so a page that logs a session
 * in, records a prompt in the settings history, or persists a store hands all
 * of that to whichever file runs next — and `cleanup()` unmounts the tree and
 * touches none of it. The resulting failures surface somewhere else entirely
 * and depend on discovery order, which is the worst kind to debug.
 *
 * Restoring a SNAPSHOT rather than deleting by key prefix is deliberate: the
 * persist middleware writes keys that existed before the file ran, and
 * clearing those would trade one file's leak for another's.
 */

interface SnapshotableStore {
  getState: () => object
  setState: (state: never, replace: true) => void
}

export interface ProcessStateSnapshot {
  storage: Record<string, string>
  stores: object[]
}

const readStorage = (): Record<string, string> =>
  Object.fromEntries(
    Object.keys(localStorage).map((key) => [key, localStorage.getItem(key) ?? ''])
  )

/** Take this before the test mutates anything. */
export const captureProcessState = (stores: SnapshotableStore[]): ProcessStateSnapshot => ({
  storage: readStorage(),
  stores: stores.map((store) => ({ ...store.getState() }))
})

/** Put it back — after `cleanup()`, so no still-mounted effect writes again. */
export const restoreProcessState = (
  stores: SnapshotableStore[],
  snapshot: ProcessStateSnapshot
): void => {
  stores.forEach((store, index) => {
    store.setState(snapshot.stores[index] as never, true)
  })

  for (const key of Object.keys(localStorage)) {
    if (!(key in snapshot.storage)) localStorage.removeItem(key)
  }
  for (const [key, value] of Object.entries(snapshot.storage)) {
    if (localStorage.getItem(key) !== value) localStorage.setItem(key, value)
  }
}
