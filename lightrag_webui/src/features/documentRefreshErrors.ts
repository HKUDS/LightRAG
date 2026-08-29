import { documentFetchTimeoutMessage } from '@/lib/constants'

/**
 * Classification of a failed `/documents/paginated` refresh.
 *
 * `shouldRetry` doubles as "counts against the circuit breaker": a permanent
 * client error is not evidence that the backend is unreachable, so it must not
 * push the document list into backoff.
 */
export type RefreshErrorClassification = {
  type: 'cancelled' | 'timeout' | 'network' | 'server' | 'client' | 'unknown'
  shouldRetry: boolean
  shouldShowToast: boolean
  /**
   * The application parsed and routed this request and answered definitively.
   * That is positive proof the backend is healthy, so the reachability breaker
   * must be CLEARED, not merely left alone: `failureCount` only ever resets on
   * proof of life, and keeping a stale count across such an answer lets three
   * NON-consecutive failures open a breaker whose threshold is documented as
   * consecutive.
   *
   * Deliberately false for 5xx: a 502/504 is the standard signal of a stalled
   * backend or the proxy in front of it, which is exactly what this breaker
   * exists to detect. False for a cancellation or a transport failure too — no
   * answer was received at all.
   *
   * `shouldRetry` outranks this at the consumer: a 429 does prove the backend
   * answered, but it is also an instruction to slow down, so backoff wins.
   */
  provesBackendResponsive: boolean
}

// 408 Request Timeout and 429 Too Many Requests are the two 4xx codes that
// describe a transient condition, so they keep the retry/backoff treatment.
const RETRYABLE_CLIENT_STATUSES = new Set([408, 429])

/**
 * Classify a document-refresh failure.
 *
 * Extracted from DocumentManager so the status handling is testable. It reads
 * `status` off the error, which the API layer's response interceptor attaches
 * (see toHttpRequestError). Before that existed the interceptor threw a bare
 * Error, so every 4xx fell through to `unknown` and was treated as retryable —
 * harmless while the breaker was inert, but with a working breaker three
 * permanent 403/404/422 responses in a row would have opened it.
 */
export const classifyDocumentRefreshError = (
  error: unknown
): RefreshErrorClassification => {
  const candidate = error as {
    name?: unknown
    message?: unknown
    code?: unknown
    status?: unknown
  } | null

  if (candidate?.name === 'AbortError') {
    return {
      type: 'cancelled',
      shouldRetry: false,
      shouldShowToast: false,
      provesBackendResponsive: false
    }
  }

  const message = typeof candidate?.message === 'string' ? candidate.message : ''

  if (message === documentFetchTimeoutMessage) {
    return {
      type: 'timeout',
      shouldRetry: true,
      shouldShowToast: true,
      provesBackendResponsive: false
    }
  }

  const status = typeof candidate?.status === 'number' ? candidate.status : null

  if (status !== null) {
    if (status >= 500) {
      return {
        type: 'server',
        shouldRetry: true,
        shouldShowToast: true,
        provesBackendResponsive: false
      }
    }
    if (status >= 400) {
      return {
        type: 'client',
        shouldRetry: RETRYABLE_CLIENT_STATUSES.has(status),
        shouldShowToast: true,
        provesBackendResponsive: true
      }
    }
  }

  if (message.includes('Network Error') || candidate?.code === 'NETWORK_ERROR') {
    return {
      type: 'network',
      shouldRetry: true,
      shouldShowToast: true,
      provesBackendResponsive: false
    }
  }

  return {
    type: 'unknown',
    shouldRetry: true,
    shouldShowToast: true,
    provesBackendResponsive: false
  }
}
