import { createRetrievalHistoryStore } from './retrievalHistory'
import { WEBUI_RETRIEVAL_HISTORY_KEY } from '@/lib/storageKeys'

/** The admin (/webui) entry's own retrieval history. */
export const useWebuiRetrievalHistoryStore = createRetrievalHistoryStore(
  WEBUI_RETRIEVAL_HISTORY_KEY
)
