import type { DocStatus, DocumentsRequest } from '@/api/lightrag'

export type StatusBucket = 'processed' | 'analyzing' | 'processing' | 'pending' | 'failed'
export type StatusFilter = StatusBucket | 'all'

const ANALYZING_STATUS_FILTERS: DocStatus[] = ['preprocessed', 'parsing', 'analyzing']

export const getStatusBucket = (status: DocStatus): StatusBucket => {
  if (ANALYZING_STATUS_FILTERS.includes(status)) {
    return 'analyzing'
  }
  if (status === 'processing') {
    return 'processing'
  }
  return status as Exclude<DocStatus, 'parsing' | 'analyzing' | 'preprocessed'>
}

export const getGroupedStatusesForFilter = (statusFilter: StatusFilter): DocStatus[] | null => {
  switch (statusFilter) {
  case 'analyzing':
    return ANALYZING_STATUS_FILTERS
  case 'processed':
  case 'processing':
  case 'pending':
  case 'failed':
    return [statusFilter]
  case 'all':
  default:
    return null
  }
}

export const getStatusRequestFilters = (
  statusFilter: StatusFilter
): Pick<DocumentsRequest, 'status_filter' | 'status_filters'> => {
  const groupedStatuses = getGroupedStatusesForFilter(statusFilter)

  if (groupedStatuses === null) {
    return {
      status_filter: null,
      status_filters: null
    }
  }

  if (statusFilter === 'analyzing') {
    return {
      status_filter: null,
      status_filters: groupedStatuses
    }
  }

  return {
    status_filter: groupedStatuses[0],
    status_filters: null
  }
}

export const matchesStatusFilter = (status: DocStatus, statusFilter: StatusFilter): boolean => {
  const groupedStatuses = getGroupedStatusesForFilter(statusFilter)
  return groupedStatuses === null || groupedStatuses.includes(status)
}
