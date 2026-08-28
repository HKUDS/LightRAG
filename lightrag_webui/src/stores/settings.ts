import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { createSelectors } from '@/lib/utils'
import { defaultQueryLabel, suggestedUserPrompts } from '@/lib/queryDefaults'
import { LEGACY_SETTINGS_STORAGE_KEY } from '@/lib/storageKeys'
import { migrateLegacySettingsState } from '@/migrations/legacySettingsChain'
import {
  getSettingsMigrationError,
  hasFutureSettingsEnvelope,
  SETTINGS_STORAGE_VERSION_AFTER_SPLIT
} from '@/migrations/splitSettingsStorage'

type Theme = 'dark' | 'light' | 'system'
type Language = 'en' | 'zh' | 'fr' | 'ar' | 'zh_TW' | 'ru' | 'ja' | 'de' | 'uk' | 'ko' | 'vi'
type Tab = 'documents' | 'knowledge-graph' | 'retrieval'

// NOTE: `querySettings` and `retrievalHistory` were split out of this store
// into their own persist keys — see stores/querySettings.ts,
// stores/webuiRetrievalHistory.ts, stores/workspaceRetrievalHistory.ts and
// migrations/splitSettingsStorage.ts. This store keeps theme/language, graph
// display preferences and the not-yet-partitioned site-scoped state
// (apiKey, userPromptHistory, queryLabel, backendMaxGraphNodes, ...).
interface SettingsState {
  // Document manager settings
  showFileName: boolean
  setShowFileName: (show: boolean) => void

  documentsPageSize: number
  setDocumentsPageSize: (size: number) => void

  // User prompt history
  userPromptHistory: string[]
  addUserPromptToHistory: (prompt: string) => void
  setUserPromptHistory: (history: string[]) => void

  // Graph viewer settings
  showPropertyPanel: boolean
  showNodeSearchBar: boolean
  showLegend: boolean
  setShowLegend: (show: boolean) => void

  showNodeLabel: boolean
  enableNodeDrag: boolean

  showEdgeLabel: boolean
  enableHideUnselectedEdges: boolean
  enableEdgeEvents: boolean

  minEdgeSize: number
  setMinEdgeSize: (size: number) => void

  maxEdgeSize: number
  setMaxEdgeSize: (size: number) => void

  graphQueryMaxDepth: number
  setGraphQueryMaxDepth: (depth: number) => void

  graphMaxNodes: number
  setGraphMaxNodes: (nodes: number, triggerRefresh?: boolean) => void

  backendMaxGraphNodes: number | null
  setBackendMaxGraphNodes: (maxNodes: number | null) => void

  // Retrieval settings
  queryLabel: string
  setQueryLabel: (queryLabel: string) => void

  // Auth settings
  apiKey: string | null
  setApiKey: (key: string | null) => void

  // App settings
  theme: Theme
  setTheme: (theme: Theme) => void

  language: Language
  /**
   * True only after the user has picked a language in the UI. Distinguishes
   * an explicit choice from the initial default ('en') that persistence
   * writes back like any other field — the language priority contract
   * (explicit choice > browser language > bundle default) needs exactly this
   * distinction. While false, the effective language is re-resolved from the
   * browser on every load (see src/i18n.ts) and the customization request
   * may omit its locale entirely.
   */
  languageUserSelected: boolean
  setLanguage: (lang: Language) => void

  enableHealthCheck: boolean
  setEnableHealthCheck: (enable: boolean) => void

  currentTab: Tab
  setCurrentTab: (tab: Tab) => void

  // Search label dropdown refresh trigger (non-persistent, runtime only)
  searchLabelDropdownRefreshTrigger: number
  triggerSearchLabelDropdownRefresh: () => void
}

// Rollback safety (split-migration rule 3): an envelope written by a NEWER
// client (version > 22) is neither hydrated nor overwritten. Hydration would
// restamp it as v22 (zustand writes the migrate result back) and any set()
// would clobber fields this build does not know. The guard is evaluated on
// EVERY storage operation, not once at creation: a newer tab can write a v23
// envelope while this v22 tab is already running, and this tab's next
// settings write must not clobber it either — such writes divert to a
// session-only fallback (kept readable so a same-session rehydrate does not
// silently reset to defaults), leaving the future envelope byte-for-byte
// intact for the client that wrote it.
const createGuardedSettingsStorage = () => {
  const sessionFallback = new Map<string, string>()
  return {
    getItem: (key: string): string | null =>
      hasFutureSettingsEnvelope()
        ? (sessionFallback.get(key) ?? null)
        : localStorage.getItem(key),
    setItem: (key: string, value: string): void => {
      if (hasFutureSettingsEnvelope()) {
        sessionFallback.set(key, value)
        return
      }
      localStorage.setItem(key, value)
    },
    removeItem: (key: string): void => {
      sessionFallback.delete(key)
      if (!hasFutureSettingsEnvelope()) {
        localStorage.removeItem(key)
      }
    }
  }
}

const useSettingsStoreBase = create<SettingsState>()(
  persist(
    (set) => ({
      theme: 'system',
      language: 'en',
      languageUserSelected: false,
      showPropertyPanel: true,
      showNodeSearchBar: true,
      showLegend: false,

      showNodeLabel: true,
      enableNodeDrag: true,

      showEdgeLabel: false,
      enableHideUnselectedEdges: true,
      enableEdgeEvents: false,

      minEdgeSize: 1,
      maxEdgeSize: 1,

      graphQueryMaxDepth: 3,
      graphMaxNodes: 1000,
      backendMaxGraphNodes: null,

      queryLabel: defaultQueryLabel,

      enableHealthCheck: true,

      apiKey: null,

      currentTab: 'documents',
      showFileName: false,
      documentsPageSize: 10,

      userPromptHistory: [...suggestedUserPrompts],

      setTheme: (theme: Theme) => set({ theme }),

      setLanguage: (language: Language) => {
        // An explicit pick outranks the browser language from now on.
        set({ language, languageUserSelected: true })
      },

      setQueryLabel: (queryLabel: string) =>
        set({
          queryLabel
        }),

      setGraphQueryMaxDepth: (depth: number) => set({ graphQueryMaxDepth: depth }),

      setGraphMaxNodes: (nodes: number, triggerRefresh: boolean = false) => {
        const state = useSettingsStore.getState();
        if (state.graphMaxNodes === nodes) {
          return;
        }

        if (triggerRefresh) {
          const currentLabel = state.queryLabel;
          // Atomically update both the node count and the query label to trigger a refresh.
          set({ graphMaxNodes: nodes, queryLabel: '' });

          // Restore the label after a short delay.
          setTimeout(() => {
            set({ queryLabel: currentLabel });
          }, 300);
        } else {
          set({ graphMaxNodes: nodes });
        }
      },

      setBackendMaxGraphNodes: (maxNodes: number | null) => set({ backendMaxGraphNodes: maxNodes }),

      setMinEdgeSize: (size: number) => set({ minEdgeSize: size }),

      setMaxEdgeSize: (size: number) => set({ maxEdgeSize: size }),

      setEnableHealthCheck: (enable: boolean) => set({ enableHealthCheck: enable }),

      setApiKey: (apiKey: string | null) => set({ apiKey }),

      setCurrentTab: (tab: Tab) => set({ currentTab: tab }),

      setShowFileName: (show: boolean) => set({ showFileName: show }),
      setShowLegend: (show: boolean) => set({ showLegend: show }),
      setDocumentsPageSize: (size: number) => set({ documentsPageSize: size }),

      // User prompt history methods
      addUserPromptToHistory: (prompt: string) => {
        if (!prompt.trim()) return

        set((state) => {
          const newHistory = [...state.userPromptHistory]

          // Remove existing occurrence if found
          const existingIndex = newHistory.indexOf(prompt)
          if (existingIndex !== -1) {
            newHistory.splice(existingIndex, 1)
          }

          // Add to beginning
          newHistory.unshift(prompt)

          // Keep only last 12 items
          if (newHistory.length > 12) {
            newHistory.splice(12)
          }

          return { userPromptHistory: newHistory }
        })
      },

      setUserPromptHistory: (history: string[]) => set({ userPromptHistory: history }),

      // Search label dropdown refresh trigger (not persisted)
      searchLabelDropdownRefreshTrigger: 0,
      triggerSearchLabelDropdownRefresh: () =>
        set((state) => ({
          searchLabelDropdownRefreshTrigger: state.searchLabelDropdownRefreshTrigger + 1
        }))
    }),
    {
      name: LEGACY_SETTINGS_STORAGE_KEY,
      storage: createJSONStorage(() => createGuardedSettingsStorage()),
      version: SETTINGS_STORAGE_VERSION_AFTER_SPLIT,
      // See splitSettingsStorage rule 7: never hydrate on a half-migrated
      // storage state.
      skipHydration: getSettingsMigrationError() != null,
      migrate: (state: any, version: number) => {
        if (version > SETTINGS_STORAGE_VERSION_AFTER_SPLIT) {
          // Belt over the session-only storage guard above (e.g. a newer
          // client's tab writes the envelope after this store was created and
          // something calls rehydrate()): a future envelope must NEVER be
          // reported as successfully migrated — zustand would persist the
          // result back stamped v22, downgrading the newer client's data.
          // Throwing aborts hydration without that write-back.
          throw new Error(
            `settings-storage version ${version} is newer than this build supports ` +
              `(${SETTINGS_STORAGE_VERSION_AFTER_SPLIT}); refusing to downgrade it`
          )
        }
        // The split migrator normally bumps the envelope to v22 before this
        // store is even evaluated. This branch remains for envelopes written
        // back by tabs still running pre-split code (their version is <= 21):
        // normalize through the REUSED legacy chain, then drop the fields
        // that were split into their own keys — clean, never merge.
        if (version <= 21) {
          state = migrateLegacySettingsState(state, version)
          delete state.querySettings
          delete state.retrievalHistory
        }
        return state
      }
    }
  )
)

const useSettingsStore = createSelectors(useSettingsStoreBase)

export { useSettingsStore, type Theme }
