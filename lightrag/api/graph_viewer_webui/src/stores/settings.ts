import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { createSelectors } from '@/lib/utils'
import { defaultQueryLabel } from '@/lib/constants'

type Theme = 'dark' | 'light' | 'system'

interface SettingsState {
  theme: Theme
  enableNodeDrag: boolean
  enableEdgeEvents: boolean
  enableHideUnselectedEdges: boolean
  showEdgeLabel: boolean

  setTheme: (theme: Theme) => void

  queryLabel: string
  setQueryLabel: (queryLabel: string) => void
}

const useSettingsStoreBase = create<SettingsState>()(
  persist(
    (set) => ({
      theme: 'system',
      enableNodeDrag: true,
      enableEdgeEvents: false,
      enableHideUnselectedEdges: true,
      showEdgeLabel: false,

      queryLabel: defaultQueryLabel,

      setTheme: (theme: Theme) => set({ theme }),

      setQueryLabel: (queryLabel: string) =>
        set({
          queryLabel
        })
    }),
    {
      name: 'settings-storage',
      storage: createJSONStorage(() => localStorage),
      version: 3,
      migrate: (state: any, version: number) => {
        if (version < 2) {
          state.showEdgeLabel = false
        }
        if (version < 3) {
          state.queryLabel = defaultQueryLabel
        }
      }
    }
  )
)

const useSettingsStore = createSelectors(useSettingsStoreBase)

export { useSettingsStore, type Theme }
