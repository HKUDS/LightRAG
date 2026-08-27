/**
 * Admin (/webui) entry bootstrap: configures the shared navigation singleton
 * for THIS entry. Evaluated at module scope so the policy is in place before
 * anything can hit a 401 path. This is the only module that may wire the
 * graph store into navigation — the navigation core itself must never import
 * `stores/graph` (that would drag graphology into the workspace entry's
 * first-load closure).
 */

import { navigationService } from '@/services/navigation'
import { useGraphStore } from '@/stores/graph'
import { useWebuiRetrievalHistoryStore } from '@/stores/webuiRetrievalHistory'

function resetGraphState() {
  const graphStore = useGraphStore.getState()
  const sigma = graphStore.sigmaInstance
  graphStore.reset()
  graphStore.setGraphDataFetchAttempted(false)
  graphStore.setLabelsFetchAttempted(false)
  graphStore.setSigmaInstance(null)
  graphStore.setIsFetching(false) // Reset isFetching state to prevent data loading issues

  if (sigma) {
    sigma.getGraph().clear()
    sigma.kill()
    useGraphStore.getState().setSigmaInstance(null)
  }
}

navigationService.configureEntry({
  unauthenticatedRoute: '/login',
  resetAdapters: [resetGraphState],
  clearRetrievalHistory: () =>
    useWebuiRetrievalHistoryStore.getState().clearHistory()
})
