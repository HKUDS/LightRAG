/**
 * Circuit breaker state machine for the document list refresh path
 * (`POST /documents/paginated`).
 *
 * Extracted from DocumentManager so the transitions are unit-testable — the
 * project has no React component test setup, so inline component state cannot
 * be covered. Same pattern as documentStatusFilters / documentRecoveryWarnings.
 *
 * Three invariants this module exists to enforce:
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
 * 3. **Half-open admits exactly one probe.** The breaker stays OPEN while the
 *    probe is in flight; only its settlement (recordSuccess / recordFailure)
 *    moves it on. Simply flipping `isOpen` to false at the deadline would let
 *    every subsequent poll tick through for as long as the probe takes to
 *    fail — with a 5s poll cadence and a 30s request timeout, that is several
 *    ticks, and the request they queue up would bypass the breaker entirely.
 */

export type CircuitBreakerState = {
  isOpen: boolean
  failureCount: number
  lastFailureTime: number | null
  nextRetryTime: number | null
  /**
   * When the half-open probe was admitted, or null when no probe is in
   * flight. The breaker keeps `isOpen: true` for its duration: a probe is an
   * exception granted to one request, not a state change.
   */
  halfOpenSince: number | null
}

/** Consecutive failures required to open the breaker. */
export const CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3

/** Hard cap on the failure counter, which bounds the exponent. */
export const CIRCUIT_BREAKER_MAX_FAILURE_COUNT = 6

/** Hard cap on the backoff delay. */
export const CIRCUIT_BREAKER_MAX_BACKOFF_MS = 60_000

/**
 * How long a half-open probe may stay unsettled before the breaker admits
 * another one. Every admitted probe should settle through recordSuccess /
 * recordFailure (the paginated fetch has its own 30s timeout), but an admission
 * that never turns into a request — the caller's throttle gate can coalesce it
 * into an already-running refresh — would otherwise wedge the breaker closed
 * forever. Must stay above the request timeout so it never races a live probe.
 */
export const CIRCUIT_BREAKER_PROBE_TIMEOUT_MS = 45_000

export const initialCircuitBreakerState: CircuitBreakerState = {
  isOpen: false,
  failureCount: 0,
  lastFailureTime: null,
  nextRetryTime: null,
  halfOpenSince: null
}

/** Exponential backoff for `failureCount`, clamped to the max delay. */
export const circuitBreakerBackoffMs = (failureCount: number): number =>
  Math.min(2 ** failureCount * 1000, CIRCUIT_BREAKER_MAX_BACKOFF_MS)

/**
 * Record one failed request, closing any half-open probe. Opens the breaker
 * once the threshold is reached and stamps the next retry time.
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
    nextRetryTime: isOpen ? now + circuitBreakerBackoffMs(failureCount) : null,
    halfOpenSince: null
  }
}

/** A real response came back: clear everything, probe included. */
export const resetCircuitBreaker = (): CircuitBreakerState => ({
  ...initialCircuitBreakerState
})


/**
 * Ask the breaker to admit one request, applying the half-open transition.
 *
 * Returns the (possibly updated) state alongside the decision so the caller
 * can store it — admitting a probe stamps `halfOpenSince`, and dropping that
 * write would admit an unbounded number of them.
 */
export const admitCircuitBreakerRequest = (
  state: CircuitBreakerState,
  now: number
): { state: CircuitBreakerState; admitted: boolean } => {
  if (!state.isOpen) {
    return { state, admitted: true }
  }

  if (state.halfOpenSince !== null) {
    // A probe is in flight. Admit a replacement only once it is so old that it
    // cannot still be a live request (see CIRCUIT_BREAKER_PROBE_TIMEOUT_MS).
    if (now - state.halfOpenSince < CIRCUIT_BREAKER_PROBE_TIMEOUT_MS) {
      return { state, admitted: false }
    }
    return { state: { ...state, halfOpenSince: now }, admitted: true }
  }

  if (state.nextRetryTime !== null && now >= state.nextRetryTime) {
    // Half-open: admit exactly one probe. `isOpen` deliberately stays true —
    // the breaker closes only when the probe reports success.
    return { state: { ...state, halfOpenSince: now }, admitted: true }
  }

  return { state, admitted: false }
}

/**
 * Read-only "would a request be admitted right now?" check, for callers that
 * must not consume the half-open slot — e.g. a polling tick deciding whether
 * it is worth walking the throttle gate at all. The admission itself must
 * still go through admitCircuitBreakerRequest at the point the request is
 * issued.
 */
export const isCircuitBreakerBlocked = (
  state: CircuitBreakerState,
  now: number
): boolean => !admitCircuitBreakerRequest(state, now).admitted
