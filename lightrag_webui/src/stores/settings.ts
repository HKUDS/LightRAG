import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { createSelectors } from '@/lib/utils'
import { defaultQueryLabel } from '@/lib/constants'
import { Message, QueryRequest } from '@/api/lightrag'

type Theme = 'dark' | 'light' | 'system'
type Tab = 'documents' | 'knowledge-graph' | 'retrieval' | 'api'

interface SettingsState {
  // Graph viewer settings
  showPropertyPanel: boolean
  showNodeSearchBar: boolean

  showNodeLabel: boolean
  enableNodeDrag: boolean

  showEdgeLabel: boolean
  enableHideUnselectedEdges: boolean
  enableEdgeEvents: boolean

  graphQueryMaxDepth: number
  setGraphQueryMaxDepth: (depth: number) => void

  graphMinDegree: number
  setGraphMinDegree: (degree: number) => void

  graphLayoutMaxIterations: number
  setGraphLayoutMaxIterations: (iterations: number) => void

  // Retrieval settings
  queryLabel: string
  setQueryLabel: (queryLabel: string) => void

  retrievalHistory: Message[]
  setRetrievalHistory: (history: Message[]) => void

  querySettings: Omit<QueryRequest, 'query'>
  updateQuerySettings: (settings: Partial<QueryRequest>) => void

  // Auth settings
  apiKey: string | null
  setApiKey: (key: string | null) => void

  // App settings
  theme: Theme
  setTheme: (theme: Theme) => void

  enableHealthCheck: boolean
  setEnableHealthCheck: (enable: boolean) => void

  currentTab: Tab
  setCurrentTab: (tab: Tab) => void
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

      graphQueryMaxDepth: 3,
      graphMinDegree: 0,
      graphLayoutMaxIterations: 10,

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

      setGraphLayoutMaxIterations: (iterations: number) =>
        set({
          graphLayoutMaxIterations: iterations
        }),

      setQueryLabel: (queryLabel: string) =>
        set({
          queryLabel
        }),

      setGraphQueryMaxDepth: (depth: number) => set({ graphQueryMaxDepth: depth }),

      setGraphMinDegree: (degree: number) => set({ graphMinDegree: degree }),

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
      version: 7,
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
        if (version < 7) {
          state.graphQueryMaxDepth = 3
          state.graphLayoutMaxIterations = 10
        }
        return state
      }
    }
  )
)

const useSettingsStore = createSelectors(useSettingsStoreBase)

export { useSettingsStore, type Theme }
