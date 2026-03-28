import { describe, expect, test } from 'bun:test'

import { getStatusRequestFilters, matchesStatusFilter } from '@/features/documentStatusFilters'

describe('documentStatusFilters', () => {
  test('builds grouped request filters for analyzing tab', () => {
    expect(getStatusRequestFilters('analyzing')).toEqual({
      status_filter: null,
      status_filters: ['preprocessed', 'parsing', 'analyzing']
    })
  })

  test('builds exact request filters for non-grouped tabs', () => {
    expect(getStatusRequestFilters('processing')).toEqual({
      status_filter: 'processing',
      status_filters: null
    })
  })

  test('matches grouped statuses for analyzing tab', () => {
    expect(matchesStatusFilter('parsing', 'analyzing')).toBe(true)
    expect(matchesStatusFilter('preprocessed', 'analyzing')).toBe(true)
    expect(matchesStatusFilter('processing', 'analyzing')).toBe(false)
  })
})
