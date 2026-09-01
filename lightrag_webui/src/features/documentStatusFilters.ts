import type { DocStatus, DocumentsRequest } from '@/api/lightrag'

export type StatusBucket = 'completed' | 'parse' | 'analyze' | 'process' | 'failed'
export type StatusFilter = StatusBucket | 'all'

// Each filter bucket maps to exactly one DocStatus. `pending` and the deprecated
// `preprocessed` intentionally have no dedicated bucket — they only surface under
// the "all" tab.
const BUCKET_TO_STATUS: Record<StatusBucket, DocStatus> = {
  completed: 'processed',
  parse: 'parsing',
  analyze: 'analyzing',
  process: 'processing',
  failed: 'failed'
}

export const getStatusRequestFilters = (
  statusFilter: StatusFilter
): Pick<DocumentsRequest, 'status_filter' | 'status_filters'> => {
  if (statusFilter === 'all') {
    return {
      status_filter: null,
      status_filters: null
    }
  }

  return {
    status_filter: BUCKET_TO_STATUS[statusFilter],
    status_filters: null
  }
}
