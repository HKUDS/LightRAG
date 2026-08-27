import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { createSelectors } from '@/lib/utils'
import { QUERY_SETTINGS_STORAGE_KEY } from '@/lib/storageKeys'
import { getSettingsMigrationError } from '@/migrations/splitSettingsStorage'
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
      storage: createJSONStorage(() => localStorage),
      version: 1,
      // Rule 7 of the split migration: when the split did not complete, the
      // store must not hydrate (running on defaults would let the first user
      // write shadow the legacy value forever).
      skipHydration: getSettingsMigrationError() != null
    }
  )
)

export const useQuerySettingsStore = createSelectors(useQuerySettingsStoreBase)
