import { beforeEach, describe, expect, test } from 'bun:test'

import { applyAiContentNoticeFlag, useAiContentNoticeStore } from './aiContentNotice'

describe('applyAiContentNoticeFlag', () => {
  beforeEach(() => {
    useAiContentNoticeStore.setState({ enabled: false })
  })

  test('adopts the server flag in both directions', () => {
    applyAiContentNoticeFlag(true)
    expect(useAiContentNoticeStore.getState().enabled).toBe(true)

    applyAiContentNoticeFlag(false)
    expect(useAiContentNoticeStore.getState().enabled).toBe(false)
  })

  test('a missing field leaves the last known value alone', () => {
    // A server too old to send the field, and the hard-coded fallbacks
    // getAuthStatus returns when the request fails, both arrive here as
    // undefined. Treating that as "off" would blank the notice on a single
    // failed poll of a deployment that requires it.
    applyAiContentNoticeFlag(true)
    applyAiContentNoticeFlag(undefined)

    expect(useAiContentNoticeStore.getState().enabled).toBe(true)
  })

  test('a non-boolean payload is ignored rather than coerced', () => {
    applyAiContentNoticeFlag('true')
    expect(useAiContentNoticeStore.getState().enabled).toBe(false)

    applyAiContentNoticeFlag(1)
    expect(useAiContentNoticeStore.getState().enabled).toBe(false)

    applyAiContentNoticeFlag(null)
    expect(useAiContentNoticeStore.getState().enabled).toBe(false)
  })
})
