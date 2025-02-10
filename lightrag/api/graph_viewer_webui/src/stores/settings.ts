import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { createSelectors } from '@/lib/utils'

type Theme = 'dark' | 'light' | 'system'

interface SettingsState {
  theme: Theme
  enableNodeDrag: boolean
  enableEdgeEvents: boolean
  enableHideUnselectedEdges: boolean
  showEdgeLabel: boolean
  setTheme: (theme: Theme) => void
}

const useSettingsStoreBase = create<SettingsState>()(
  persist(
    (set) => ({
      theme: 'system',
      enableNodeDrag: true,
      enableEdgeEvents: false,
      enableHideUnselectedEdges: true,
      showEdgeLabel: false,

      setTheme: (theme: Theme) => set({ theme })
    }),
    {
      name: 'settings-storage',
      storage: createJSONStorage(() => localStorage),
      version: 2,
      migrate: (state: any, version: number) => {
        if (version < 2) {
          state.showEdgeLabel = false
        }
      }
    }
  )
)

const useSettingsStore = createSelectors(useSettingsStoreBase)

export { useSettingsStore, type Theme }
