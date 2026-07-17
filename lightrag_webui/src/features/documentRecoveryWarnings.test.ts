import { describe, expect, test } from 'bun:test'

import { hasDocumentWarning, hasKgRecoveryWarnings } from './documentRecoveryWarnings'

describe('document recovery warning detection', () => {
  test('detects a processed document with structured recovery warnings', () => {
    expect(
      hasDocumentWarning({
        error_msg: '',
        metadata: {
          kg_recovery_warnings: [
            { code: 'degraded_custom_chunk_rollback', operation_id: 'op-1' }
          ]
        }
      })
    ).toBe(true)
  })

  test('keeps error messages as warnings', () => {
    expect(hasDocumentWarning({ error_msg: 'failed', metadata: {} })).toBe(true)
  })

  test('ignores absent, empty, and malformed recovery warnings', () => {
    expect(hasKgRecoveryWarnings(undefined)).toBe(false)
    expect(hasKgRecoveryWarnings({ kg_recovery_warnings: [] })).toBe(false)
    expect(hasKgRecoveryWarnings({ kg_recovery_warnings: ['not-structured'] })).toBe(false)
  })
})
