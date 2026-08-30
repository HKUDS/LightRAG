import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { createSelectors } from '@/lib/utils'
import { QUERY_SETTINGS_STORAGE_KEY } from '@/lib/storageKeys'
import {
  getSettingsMigrationError,
  QUERY_SETTINGS_STORE_VERSION
} from '@/migrations/splitSettingsStorage'
import { createFutureGuardedStorage } from '@/lib/guardedStorage'
import type { QueryRequest } from '@/api/lightrag'

/**
 * `querySettings`, split out of the legacy `settings-storage` envelope into
 * its own persist key (`lightrag::query-settings-storage`).
 *
 * Write ownership: at runtime ONLY the /webui entry writes this store (via
 * the QuerySettings panel). The /workspace entry is strictly read-only: it
 * never calls `updateQuerySettings`, and it follows cross-entry updates by
 * re-hydrating on `storage` events (see the workspace bootstrap). Keeping
 * the settings in their own key removes the whole-envelope write-back race
 * that used to let one entry's writes clobber the other's saved parameters.
 */

export type QuerySettings = Omit<QueryRequest, 'query'>

interface QuerySettingsState {
  querySettings: QuerySettings
  updateQuerySettings: (settings: Partial<QueryRequest>) => void
}

export const defaultQuerySettings: QuerySettings = {
  mode: 'mix',
  top_k: 40,
  chunk_top_k: 20,
  max_entity_tokens: 6000,
  max_relation_tokens: 8000,
  max_total_tokens: 30000,
  only_need_context: false,
  only_need_prompt: false,
  stream: true,
  history_turns: 0,
  user_prompt: '',
  enable_rerank: true
}

const useQuerySettingsStoreBase = create<QuerySettingsState>()(
  persist(
    (set) => ({
      querySettings: { ...defaultQuerySettings },

      updateQuerySettings: (settings: Partial<QueryRequest>) => {
        // Filter out history_turns to prevent changes, always keep it as 0
        const filteredSettings = { ...settings }
        delete filteredSettings.history_turns
        set((state) => ({
          querySettings: {
            ...state.querySettings,
            ...filteredSettings,
            history_turns: 0
          }
        }))
      }
    }),
    {
      name: QUERY_SETTINGS_STORAGE_KEY,
      // Rollback safety: an envelope written by a NEWER client (version above
      // this build's) is neither hydrated nor overwritten — even when it
      // arrives mid-session from a newer tab. The extra divert extends
      // rule 7 to WRITES: after a failed split, no set() may plant a
      // defaults envelope that writeIfAbsent would later treat as
      // authoritative (guard semantics in lib/guardedStorage.ts).
      storage: createJSONStorage(() =>
        createFutureGuardedStorage(
          QUERY_SETTINGS_STORE_VERSION,
          () => getSettingsMigrationError() != null
        )
      ),
      version: QUERY_SETTINGS_STORE_VERSION,
      // Rule 7 of the split migration: when the split did not complete, the
      // store must not hydrate (running on defaults would let the first user
      // write shadow the legacy value forever).
      skipHydration: getSettingsMigrationError() != null,
      migrate: (state: unknown, version: number) => {
        if (version > QUERY_SETTINGS_STORE_VERSION) {
          // Never report a newer client's envelope as migrated — zustand
          // would persist it back restamped with this build's version.
          throw new Error(
            `query-settings-storage version ${version} is newer than this build ` +
              `supports (${QUERY_SETTINGS_STORE_VERSION}); refusing to downgrade it`
          )
        }
        return state
      }
    }
  )
)

export const useQuerySettingsStore = createSelectors(useQuerySettingsStoreBase)
