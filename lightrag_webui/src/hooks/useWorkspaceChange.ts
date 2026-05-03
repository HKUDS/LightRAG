import { useEffect, useRef } from 'react'
import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'

/**
 * Hook to handle workspace change events.
 * When the workspace changes, this hook clears all workspace-dependent data:
 * - Graph store state
 * - Graph fetch attempt flags
 * - Retrieval history
 *
 * This should be called once at the App level.
 */
const useWorkspaceChange = () => {
  const currentWorkspace = useSettingsStore.use.currentWorkspace()
  const triggerWorkspaceRefresh = useSettingsStore.use.triggerWorkspaceRefresh()

  // Use ref to track previous workspace value to avoid infinite loops
  const previousWorkspaceRef = useRef<string | null>(currentWorkspace)

  useEffect(() => {
    // Only trigger refresh if workspace actually changed
    if (previousWorkspaceRef.current !== currentWorkspace) {
      const previousWorkspace = previousWorkspaceRef.current
      previousWorkspaceRef.current = currentWorkspace

      console.log(`Workspace changed from "${previousWorkspace}" to "${currentWorkspace}", refreshing...`)

      // 1. Clear graph store state
      const graphState = useGraphStore.getState()
      graphState.reset()
      graphState.setGraphDataFetchAttempted(false)
      graphState.setLabelsFetchAttempted(false)
      graphState.incrementGraphDataVersion()

      // 2. Clear retrieval history
      useSettingsStore.getState().setRetrievalHistory([])

      // 3. Trigger workspace refresh for DocumentManager and other components
      triggerWorkspaceRefresh()
    }
  }, [currentWorkspace, triggerWorkspaceRefresh])
}

export default useWorkspaceChange
