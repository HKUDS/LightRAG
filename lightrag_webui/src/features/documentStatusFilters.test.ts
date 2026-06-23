import { describe, expect, test } from 'bun:test'

import { getStatusRequestFilters, matchesStatusFilter } from '@/features/documentStatusFilters'

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

  test('matches a single status per non-all tab', () => {
    expect(matchesStatusFilter('parsing', 'parse')).toBe(true)
    expect(matchesStatusFilter('analyzing', 'analyze')).toBe(true)
    expect(matchesStatusFilter('processing', 'process')).toBe(true)
    expect(matchesStatusFilter('analyzing', 'parse')).toBe(false)
    expect(matchesStatusFilter('preprocessed', 'analyze')).toBe(false)
  })

  test('the all tab matches every status, including deprecated and hidden ones', () => {
    expect(matchesStatusFilter('pending', 'all')).toBe(true)
    expect(matchesStatusFilter('preprocessed', 'all')).toBe(true)
    expect(matchesStatusFilter('processed', 'all')).toBe(true)
  })
})
