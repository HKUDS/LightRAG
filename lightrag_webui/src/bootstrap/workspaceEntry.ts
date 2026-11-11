/**
 * Workspace (/workspace) entry bootstrap: configures the shared navigation
 * singleton for THIS entry (unauthenticated users land on the welcome page,
 * not the admin login), and wires the cross-entry `storage` listener that
 * re-hydrates the shared query settings when /webui saves new parameters in
 * another tab — so they take effect from the NEXT query without a reload.
 *
 * No graph reset adapter is registered here: the workspace entry never
 * mounts the graph viewer, and registering one would pull `stores/graph`
 * (and graphology) into this entry's first-load closure.
 */

import { navigationService } from '@/services/navigation'
import { useQuerySettingsStore } from '@/stores/querySettings'
import { useWorkspaceRetrievalHistoryStore } from '@/stores/workspaceRetrievalHistory'
import { QUERY_SETTINGS_STORAGE_KEY } from '@/lib/storageKeys'
import { watchCrossTabIdentityChanges } from '@/lib/loginIdentity'

navigationService.configureEntry({
  unauthenticatedRoute: '/welcome',
  clearRetrievalHistory: () =>
    useWorkspaceRetrievalHistoryStore.getState().clearHistory()
})

if (typeof window !== 'undefined') {
  window.addEventListener('storage', (event) => {
    // Re-hydrate ONLY the shared query settings. In-flight streaming queries
    // are unaffected: the session controller reads one consistent snapshot
    // per submit, so a mid-stream hydration only changes the NEXT query.
    if (event.key === QUERY_SETTINGS_STORAGE_KEY) {
      void useQuerySettingsStore.persist.rehydrate()
    }
  })
}

// Another tab switching identity must also reset THIS tab's live session —
// not just the persisted history the other tab already cleared.
watchCrossTabIdentityChanges()
