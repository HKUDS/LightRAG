/**
 * Circuit breaker state machine for the document list refresh path
 * (`POST /documents/paginated`).
 *
 * Extracted from DocumentManager so the transitions are unit-testable — the
 * project has no React component test setup, so inline component state cannot
 * be covered. Same pattern as documentStatusFilters / documentRecoveryWarnings.
 *
 * Two invariants this module exists to enforce:
 *
 * 1. **Backoff is bounded.** The previous inline implementation used a raw
 *    `2 ** failureCount * 1000` with no cap on either the exponent or the
 *    counter, so a backend that stayed down long enough could silence the
 *    document list for tens of minutes (failureCount=10 -> ~17 min). The
 *    counter is capped at CIRCUIT_BREAKER_MAX_FAILURE_COUNT and the delay at
 *    CIRCUIT_BREAKER_MAX_BACKOFF_MS.
 * 2. **Success is success.** Callers must record a success only when a
 *    response actually came back. The previous implementation called its
 *    success hook right after a fire-and-forget refresh, so every polling tick
 *    cleared the failure counter and the breaker never opened at all.
 */

export type CircuitBreakerState = {
  isOpen: boolean
  failureCount: number
  lastFailureTime: number | null
  nextRetryTime: number | null
}

/** Consecutive failures required to open the breaker. */
export const CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3

/** Hard cap on the failure counter, which bounds the exponent. */
export const CIRCUIT_BREAKER_MAX_FAILURE_COUNT = 6

/** Hard cap on the backoff delay. */
export const CIRCUIT_BREAKER_MAX_BACKOFF_MS = 60_000

export const initialCircuitBreakerState: CircuitBreakerState = {
  isOpen: false,
  failureCount: 0,
  lastFailureTime: null,
  nextRetryTime: null
}

/** Exponential backoff for `failureCount`, clamped to the max delay. */
export const circuitBreakerBackoffMs = (failureCount: number): number =>
  Math.min(2 ** failureCount * 1000, CIRCUIT_BREAKER_MAX_BACKOFF_MS)

/**
 * Record one failed request. Opens the breaker once the threshold is reached
 * and stamps the next retry time.
 */
export const recordCircuitBreakerFailure = (
  state: CircuitBreakerState,
  now: number
): CircuitBreakerState => {
  const failureCount = Math.min(
    state.failureCount + 1,
    CIRCUIT_BREAKER_MAX_FAILURE_COUNT
  )
  const isOpen = failureCount >= CIRCUIT_BREAKER_FAILURE_THRESHOLD

  return {
    isOpen,
    failureCount,
    lastFailureTime: now,
    nextRetryTime: isOpen ? now + circuitBreakerBackoffMs(failureCount) : null
  }
}

/** A real response came back: clear everything. */
export const resetCircuitBreaker = (): CircuitBreakerState => ({
  ...initialCircuitBreakerState
})

/**
 * Decide whether a request may go out now, applying the half-open transition.
 *
 * Returns the (possibly updated) state alongside the decision so the caller
 * can store it — the half-open step mutates the counter, and dropping that
 * write would re-open the breaker on the very next evaluation.
 */
export const evaluateCircuitBreaker = (
  state: CircuitBreakerState,
  now: number
): { state: CircuitBreakerState; isOpen: boolean } => {
  if (!state.isOpen) {
    return { state, isOpen: false }
  }

  if (state.nextRetryTime !== null && now >= state.nextRetryTime) {
    // Half-open: let exactly one request through. The counter is decremented
    // rather than cleared so a backend that is still down re-opens with a
    // comparable delay instead of restarting the ramp from zero.
    return {
      state: {
        ...state,
        isOpen: false,
        failureCount: Math.max(0, state.failureCount - 1),
        nextRetryTime: null
      },
      isOpen: false
    }
  }

  return { state, isOpen: true }
}
