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
      shouldShowToast: false,
      provesBackendResponsive: false
    })
  })

  test('the paginated fetch timeout is retryable', () => {
    // Pinned against the actual message getDocumentsPaginatedWithTimeout
    // rejects with; the previous inline classifier matched 'Request timeout',
    // which never occurs, so timeouts silently fell through to 'unknown'.
    expect(classifyDocumentRefreshError(new Error(documentFetchTimeoutMessage))).toEqual({
      type: 'timeout',
      shouldRetry: true,
      shouldShowToast: true,
      provesBackendResponsive: false
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
      // The application answered, so the reachability breaker must be cleared
      // rather than left holding a stale failure count.
      expect(classification.provesBackendResponsive).toBe(true)
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
      // A 502/504 is the standard signal of a stalled backend or the proxy in
      // front of it — exactly what the breaker exists to detect. It must never
      // read as proof of life, however much it is "an HTTP response".
      expect(classification.provesBackendResponsive).toBe(false)
    }
  })

  test('network failures are retryable', () => {
    expect(classifyDocumentRefreshError(new Error('Network Error')).type).toBe('network')

    const coded = new Error('boom') as Error & { code?: string }
    coded.code = 'NETWORK_ERROR'
    expect(classifyDocumentRefreshError(coded).type).toBe('network')
  })

  test('the code axios actually uses is recognised', () => {
    // AxiosError.ERR_NETWORK, not 'NETWORK_ERROR'. The message fallback covered
    // this in practice, but only while the browser's ProgressEvent carries no
    // message of its own — adapters/xhr.js prefers event.message when present.
    const coded = new Error('') as Error & { code?: string }
    coded.code = 'ERR_NETWORK'

    const classification = classifyDocumentRefreshError(coded)
    expect(classification.type).toBe('network')
    expect(classification.shouldRetry).toBe(true)
    expect(classification.provesBackendResponsive).toBe(false)
  })

  test('anything unrecognised stays retryable', () => {
    expect(classifyDocumentRefreshError(new Error('boom'))).toEqual({
      type: 'unknown',
      shouldRetry: true,
      shouldShowToast: true,
      provesBackendResponsive: false
    })
    expect(classifyDocumentRefreshError(null).type).toBe('unknown')
    expect(classifyDocumentRefreshError(undefined).type).toBe('unknown')
  })
})
