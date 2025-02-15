import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { createSelectors } from '@/lib/utils'
import { defaultQueryLabel } from '@/lib/constants'

type Theme = 'dark' | 'light' | 'system'
type Tab = 'documents' | 'knowledge-graph' | 'retrieval'

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

  retrievalHistory: any[]
  setRetrievalHistory: (history: any[]) => void
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

      setTheme: (theme: Theme) => set({ theme }),

      setQueryLabel: (queryLabel: string) =>
        set({
          queryLabel
        }),

      setEnableHealthCheck: (enable: boolean) => set({ enableHealthCheck: enable }),

      setApiKey: (apiKey: string | null) => set({ apiKey }),

      setCurrentTab: (tab: Tab) => set({ currentTab: tab }),

      setRetrievalHistory: (history: any[]) => set({ retrievalHistory: history })
    }),
    {
      name: 'settings-storage',
      storage: createJSONStorage(() => localStorage),
      version: 5,
      partialize(state) {
        return { ...state, retrievalHistory: undefined }
      },
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
      }
    }
  )
)

const useSettingsStore = createSelectors(useSettingsStoreBase)

export { useSettingsStore, type Theme }
