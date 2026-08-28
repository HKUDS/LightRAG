import { create, type UseBoundStore, type StoreApi } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { toast } from 'sonner'
import i18n from '@/i18n'
import {
  getSettingsMigrationError,
  RETRIEVAL_HISTORY_STORE_VERSION
} from '@/migrations/splitSettingsStorage'
import { createFutureGuardedStorage } from '@/lib/guardedStorage'
import { isQuotaExceededError } from '@/lib/storageQuota'
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

// One warning per key per session is enough — the condition repeats on every
// persisted write until the conversation is cleared.
const quotaNoticeShown = new Set<string>()

const notifyHistoryDropped = (key: string): void => {
  if (quotaNoticeShown.has(key)) return
  quotaNoticeShown.add(key)
  try {
    toast.warning(i18n.t('retrievalHistory.quotaDropped'))
  } catch (error) {
    console.warn('Failed to show history-quota warning:', error)
  }
}

/**
 * Bounded-capacity persistence (product decision: the chat history is a
 * non-critical test record). When a persisted write exceeds the browser's
 * storage quota, the write is ABANDONED instead of silently failing forever:
 * the stored envelope is reset to an empty history (reclaiming the space so
 * the key stays bounded), the in-memory conversation stays visible for the
 * session, and a one-time toast tells the user. Everything else delegates to
 * the future-envelope/migration guard.
 */
function createBoundedHistoryStorage() {
  const guarded = createFutureGuardedStorage(
    RETRIEVAL_HISTORY_STORE_VERSION,
    () => getSettingsMigrationError() != null
  )
  return {
    getItem: guarded.getItem,
    removeItem: guarded.removeItem,
    setItem: (key: string, value: string): void => {
      try {
        guarded.setItem(key, value)
        return
      } catch (error) {
        if (!isQuotaExceededError(error)) throw error
      }
      try {
        guarded.setItem(
          key,
          JSON.stringify({
            state: { history: [] },
            version: RETRIEVAL_HISTORY_STORE_VERSION
          })
        )
      } catch {
        // Even the empty envelope does not fit: free the key outright.
        try {
          localStorage.removeItem(key)
        } catch {
          // Nothing left to do — the next write attempt starts over.
        }
      }
      notifyHistoryDropped(key)
    }
  }
}

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
        // Rollback safety: a NEWER client's envelope is neither hydrated nor
        // overwritten, even arriving mid-session; after a failed split no
        // set() may plant a defaults envelope either (lib/guardedStorage.ts).
        // Quota safety: an over-quota write abandons persistence instead of
        // failing silently forever (createBoundedHistoryStorage above).
        storage: createJSONStorage(() => createBoundedHistoryStorage()),
        version: RETRIEVAL_HISTORY_STORE_VERSION,
        // See splitSettingsStorage rule 7: never hydrate on a half-migrated
        // storage state.
        skipHydration: getSettingsMigrationError() != null,
        migrate: (state: unknown, version: number) => {
          if (version > RETRIEVAL_HISTORY_STORE_VERSION) {
            // Never report a newer client's envelope as migrated — zustand
            // would persist it back restamped with this build's version.
            throw new Error(
              `${storageKey} version ${version} is newer than this build ` +
                `supports (${RETRIEVAL_HISTORY_STORE_VERSION}); refusing to downgrade it`
            )
          }
          return state
        }
      }
    )
  )
}
