import { createRetrievalHistoryStore } from './retrievalHistory'
import { WORKSPACE_RETRIEVAL_HISTORY_KEY } from '@/lib/storageKeys'

/** The workspace (/workspace) entry's own retrieval history. */
export const useWorkspaceRetrievalHistoryStore = createRetrievalHistoryStore(
  WORKSPACE_RETRIEVAL_HISTORY_KEY
)
