import { describe, expect, test } from 'bun:test'

import { computeStatusCountsFingerprint } from '@/features/documentStatusCounts'

describe('computeStatusCountsFingerprint', () => {
  test('sees a document moving between the two bucket-less statuses', () => {
    // The property the removed per-page bucket structure could not express.
    // `pending` and `preprocessed` have no filter bucket, so a pending document
    // becoming preprocessed changed none of that structure's keys and the
    // health check never fired. Every other count is held constant here, so
    // only sensitivity to those two statuses can make this pass.
    const before = computeStatusCountsFingerprint({
      all: 5,
      processed: 4,
      pending: 1,
      preprocessed: 0
    })
    const after = computeStatusCountsFingerprint({
      all: 5,
      processed: 4,
      pending: 0,
      preprocessed: 1
    })

    expect(after).not.toBe(before)
  })

  test('sees the pending count alone change', () => {
    expect(computeStatusCountsFingerprint({ all: 5, processed: 4, pending: 1 })).not.toBe(
      computeStatusCountsFingerprint({ all: 5, processed: 4, pending: 2 })
    )
  })

  test('sees the preprocessed count alone change', () => {
    expect(computeStatusCountsFingerprint({ all: 5, processed: 4, preprocessed: 1 })).not.toBe(
      computeStatusCountsFingerprint({ all: 5, processed: 4, preprocessed: 2 })
    )
  })

  test('is stable for an unchanged map, regardless of key order and key case', () => {
    const reference = computeStatusCountsFingerprint({ all: 3, pending: 1, processed: 2 })

    expect(computeStatusCountsFingerprint({ processed: 2, all: 3, pending: 1 })).toBe(reference)
    expect(computeStatusCountsFingerprint({ PROCESSED: 2, all: 3, PENDING: 1 })).toBe(reference)
  })

  test('distinguishes a status appearing from it being absent', () => {
    expect(computeStatusCountsFingerprint({ all: 1, failed: 1 })).not.toBe(
      computeStatusCountsFingerprint({ all: 1 })
    )
  })
})
