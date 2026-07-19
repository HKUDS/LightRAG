import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { createSelectors } from '@/lib/utils'

export type WorkspaceStatus = 'provisioning' | 'active' | 'deleting' | 'deleted' | 'failed'

export type Workspace = {
  id: string
  display_name: string
  description: string | null
  status: WorkspaceStatus
  is_default: boolean
  created_at: string
  updated_at: string
  deleted_at: string | null
}

type WorkspaceState = {
  selectedWorkspace: Workspace | null
  selectWorkspace: (workspace: Workspace) => void
  clearSelectedWorkspace: () => void
}

const useWorkspaceStoreBase = create<WorkspaceState>()(
  persist(
    (set) => ({
      selectedWorkspace: null,
      selectWorkspace: (workspace) => set({ selectedWorkspace: workspace }),
      clearSelectedWorkspace: () => set({ selectedWorkspace: null })
    }),
    {
      name: 'workspace-storage',
      storage: createJSONStorage(() => localStorage),
      version: 1
    }
  )
)

export const useWorkspaceStore = createSelectors(useWorkspaceStoreBase)
