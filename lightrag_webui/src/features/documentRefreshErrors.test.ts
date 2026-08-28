import { describe, expect, test } from 'bun:test'

import { classifyDocumentRefreshError } from '@/features/documentRefreshErrors'
import { documentFetchTimeoutMessage } from '@/lib/constants'

// Shaped like what the API layer's response interceptor throws.
const httpError = (status: number): Error & { status?: number } => {
  const error = new Error(`${status} Error\n{}\n/documents/paginated`) as Error & {
    status?: number
  }
  error.status = status
  return error
}

describe('classifyDocumentRefreshError', () => {
  test('an aborted request is neither retried nor reported', () => {
    const error = new DOMException('Aborted', 'AbortError')

    expect(classifyDocumentRefreshError(error)).toEqual({
      type: 'cancelled',
      shouldRetry: false,
      shouldShowToast: false
    })
  })

  test('the paginated fetch timeout is retryable', () => {
    // Pinned against the actual message getDocumentsPaginatedWithTimeout
    // rejects with; the previous inline classifier matched 'Request timeout',
    // which never occurs, so timeouts silently fell through to 'unknown'.
    expect(classifyDocumentRefreshError(new Error(documentFetchTimeoutMessage))).toEqual({
      type: 'timeout',
      shouldRetry: true,
      shouldShowToast: true
    })
  })

  test('permanent client errors must not count against the breaker', () => {
    // The interceptor used to throw a bare Error with the status only inside
    // the message, so these landed in 'unknown' -> shouldRetry: true. With a
    // breaker that actually works, three of them in a row would open it and
    // stall the document list on a condition retrying can never fix.
    for (const status of [400, 403, 404, 422]) {
      const classification = classifyDocumentRefreshError(httpError(status))
      expect(classification.type).toBe('client')
      expect(classification.shouldRetry).toBe(false)
      expect(classification.shouldShowToast).toBe(true)
    }
  })

  test('transient client errors keep the retry treatment', () => {
    for (const status of [408, 429]) {
      const classification = classifyDocumentRefreshError(httpError(status))
      expect(classification.type).toBe('client')
      expect(classification.shouldRetry).toBe(true)
    }
  })

  test('server errors are retryable', () => {
    for (const status of [500, 502, 503, 504]) {
      const classification = classifyDocumentRefreshError(httpError(status))
      expect(classification.type).toBe('server')
      expect(classification.shouldRetry).toBe(true)
    }
  })

  test('network failures are retryable', () => {
    expect(classifyDocumentRefreshError(new Error('Network Error')).type).toBe('network')

    const coded = new Error('boom') as Error & { code?: string }
    coded.code = 'NETWORK_ERROR'
    expect(classifyDocumentRefreshError(coded).type).toBe('network')
  })

  test('anything unrecognised stays retryable', () => {
    expect(classifyDocumentRefreshError(new Error('boom'))).toEqual({
      type: 'unknown',
      shouldRetry: true,
      shouldShowToast: true
    })
    expect(classifyDocumentRefreshError(null).type).toBe('unknown')
    expect(classifyDocumentRefreshError(undefined).type).toBe('unknown')
  })
})
