import { describe, expect, test } from 'bun:test'

import { getStatusRequestFilters } from '@/features/documentStatusFilters'

describe('documentStatusFilters', () => {
  test('builds exact single-status request filters for each tab', () => {
    expect(getStatusRequestFilters('completed')).toEqual({
      status_filter: 'processed',
      status_filters: null
    })
    expect(getStatusRequestFilters('parse')).toEqual({
      status_filter: 'parsing',
      status_filters: null
    })
    expect(getStatusRequestFilters('analyze')).toEqual({
      status_filter: 'analyzing',
      status_filters: null
    })
    expect(getStatusRequestFilters('process')).toEqual({
      status_filter: 'processing',
      status_filters: null
    })
    expect(getStatusRequestFilters('failed')).toEqual({
      status_filter: 'failed',
      status_filters: null
    })
  })

  test('builds empty request filters for the all tab', () => {
    expect(getStatusRequestFilters('all')).toEqual({
      status_filter: null,
      status_filters: null
    })
  })
})
