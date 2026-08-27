import { create, type UseBoundStore, type StoreApi } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { getSettingsMigrationError } from '@/migrations/splitSettingsStorage'
import type { MessageWithError } from '@/types/retrieval'

/**
 * Per-entry retrieval history, split out of the legacy `settings-storage`
 * envelope. The two entries have SEPARATE histories by product semantics:
 * the admin history is a parameter-debugging record, the workspace history
 * is an end user's normal conversation — mixing them into each other's
 * display (or, in bypass mode, into the LLM context) is a defect.
 *
 * The shared query-session layer never chooses a history storage itself;
 * the page composition layer injects one of these stores (see
 * `RetrievalHistoryStore` and `useQuerySession`).
 */

export interface RetrievalHistoryState {
  history: MessageWithError[]
  setHistory: (history: MessageWithError[]) => void
  clearHistory: () => void
}

export type RetrievalHistoryStore = UseBoundStore<StoreApi<RetrievalHistoryState>>

export function createRetrievalHistoryStore(storageKey: string): RetrievalHistoryStore {
  return create<RetrievalHistoryState>()(
    persist(
      (set) => ({
        history: [],
        setHistory: (history: MessageWithError[]) => set({ history }),
        clearHistory: () => set({ history: [] })
      }),
      {
        name: storageKey,
        storage: createJSONStorage(() => localStorage),
        version: 1,
        // See splitSettingsStorage rule 7: never hydrate on a half-migrated
        // storage state.
        skipHydration: getSettingsMigrationError() != null
      }
    )
  )
}
