import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { createSelectors } from '@/lib/utils'
import { defaultQueryLabel } from '@/lib/constants'
import { Message, QueryRequest } from '@/api/lightrag'

type Theme = 'dark' | 'light' | 'system'
type Tab = 'documents' | 'knowledge-graph' | 'retrieval' | 'api'

interface SettingsState {
  theme: Theme
  setTheme: (theme: Theme) => void

  showPropertyPanel: boolean
  showNodeSearchBar: boolean

  showNodeLabel: boolean
  enableNodeDrag: boolean

  showEdgeLabel: boolean
  enableHideUnselectedEdges: boolean
  enableEdgeEvents: boolean

  queryLabel: string
  setQueryLabel: (queryLabel: string) => void

  enableHealthCheck: boolean
  setEnableHealthCheck: (enable: boolean) => void

  apiKey: string | null
  setApiKey: (key: string | null) => void

  currentTab: Tab
  setCurrentTab: (tab: Tab) => void

  retrievalHistory: Message[]
  setRetrievalHistory: (history: Message[]) => void

  querySettings: Omit<QueryRequest, 'query'>
  updateQuerySettings: (settings: Partial<QueryRequest>) => void
}

const useSettingsStoreBase = create<SettingsState>()(
  persist(
    (set) => ({
      theme: 'system',

      showPropertyPanel: true,
      showNodeSearchBar: true,

      showNodeLabel: true,
      enableNodeDrag: true,

      showEdgeLabel: false,
      enableHideUnselectedEdges: true,
      enableEdgeEvents: false,

      queryLabel: defaultQueryLabel,
      enableHealthCheck: true,

      apiKey: null,

      currentTab: 'documents',

      retrievalHistory: [],

      querySettings: {
        mode: 'global',
        response_type: 'Multiple Paragraphs',
        top_k: 10,
        max_token_for_text_unit: 4000,
        max_token_for_global_context: 4000,
        max_token_for_local_context: 4000,
        only_need_context: false,
        only_need_prompt: false,
        stream: true,
        history_turns: 3,
        hl_keywords: [],
        ll_keywords: []
      },

      setTheme: (theme: Theme) => set({ theme }),

      setQueryLabel: (queryLabel: string) =>
        set({
          queryLabel
        }),

      setEnableHealthCheck: (enable: boolean) => set({ enableHealthCheck: enable }),

      setApiKey: (apiKey: string | null) => set({ apiKey }),

      setCurrentTab: (tab: Tab) => set({ currentTab: tab }),

      setRetrievalHistory: (history: Message[]) => set({ retrievalHistory: history }),

      updateQuerySettings: (settings: Partial<QueryRequest>) =>
        set((state) => ({
          querySettings: { ...state.querySettings, ...settings }
        }))
    }),
    {
      name: 'settings-storage',
      storage: createJSONStorage(() => localStorage),
      version: 6,
      migrate: (state: any, version: number) => {
        if (version < 2) {
          state.showEdgeLabel = false
        }
        if (version < 3) {
          state.queryLabel = defaultQueryLabel
        }
        if (version < 4) {
          state.showPropertyPanel = true
          state.showNodeSearchBar = true
          state.showNodeLabel = true
          state.enableHealthCheck = true
          state.apiKey = null
        }
        if (version < 5) {
          state.currentTab = 'documents'
        }
        if (version < 6) {
          state.querySettings = {
            mode: 'global',
            response_type: 'Multiple Paragraphs',
            top_k: 10,
            max_token_for_text_unit: 4000,
            max_token_for_global_context: 4000,
            max_token_for_local_context: 4000,
            only_need_context: false,
            only_need_prompt: false,
            stream: true,
            history_turns: 3,
            hl_keywords: [],
            ll_keywords: []
          }
          state.retrievalHistory = []
        }
        return state
      }
    }
  )
)

const useSettingsStore = createSelectors(useSettingsStoreBase)

export { useSettingsStore, type Theme }
