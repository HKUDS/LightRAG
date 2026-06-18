import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'

export type KGMaintenanceSection =
  | 'overview'
  | 'graph'
  | 'entities'
  | 'relations'
  | 'evidence'
  | 'quality'
  | 'llm-review'
  | 'patches'
  | 'judge'
  | 'approval'
  | 'runs'
  | 'diff'
  | 'rules'

export type KGMaintenanceSelectedItem =
  | { kind: 'node'; id: string }
  | { kind: 'edge'; id: string }
  | null

interface KGMaintenanceState {
  activeSection: KGMaintenanceSection
  selectedItem: KGMaintenanceSelectedItem
  selectedWorkspace: string | null
  latestRunId: string
  setActiveSection: (section: KGMaintenanceSection) => void
  setSelectedItem: (item: KGMaintenanceSelectedItem) => void
  setSelectedWorkspace: (workspace: string | null) => void
  setLatestRunId: (runId: string) => void
}

const useKGMaintenanceStoreBase = create<KGMaintenanceState>()((set) => ({
  activeSection: 'overview',
  selectedItem: null,
  selectedWorkspace: null,
  latestRunId: 'latest',
  setActiveSection: (activeSection) => set({ activeSection }),
  setSelectedItem: (selectedItem) => set({ selectedItem }),
  setSelectedWorkspace: (selectedWorkspace) => set({ selectedWorkspace }),
  setLatestRunId: (latestRunId) => set({ latestRunId })
}))

export const useKGMaintenanceStore = createSelectors(useKGMaintenanceStoreBase)
