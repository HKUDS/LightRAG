import { create, StoreApi, UseBoundStore } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

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

type WithSelectors<S> = S extends { getState: () => infer T }
  ? S & { use: { [K in keyof T]: () => T[K] } }
  : never

const createSelectors = <S extends UseBoundStore<StoreApi<object>>>(_store: S) => {
  const store = _store as WithSelectors<typeof _store>
  store.use = {}
  for (const k of Object.keys(store.getState())) {
    ;(store.use as any)[k] = () => store((s) => s[k as keyof typeof s])
  }

  return store
}

const useSettingsStore = createSelectors(useSettingsStoreBase)

export { useSettingsStore, type Theme }
