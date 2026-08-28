import { describe, expect, test } from 'bun:test'

import {
  CIRCUIT_BREAKER_FAILURE_THRESHOLD,
  CIRCUIT_BREAKER_MAX_BACKOFF_MS,
  CIRCUIT_BREAKER_MAX_FAILURE_COUNT,
  CIRCUIT_BREAKER_PROBE_TIMEOUT_MS,
  admitCircuitBreakerRequest,
  circuitBreakerBackoffMs,
  initialCircuitBreakerState,
  isCircuitBreakerBlocked,
  recordCircuitBreakerFailure,
  resetCircuitBreaker,
  type CircuitBreakerState
} from '@/features/documentRefreshCircuitBreaker'

const failTimes = (
  count: number,
  now: number,
  state: CircuitBreakerState = initialCircuitBreakerState
): CircuitBreakerState => {
  let next = state
  for (let i = 0; i < count; i += 1) {
    next = recordCircuitBreakerFailure(next, now)
  }
  return next
}

const openBreaker = (now: number): CircuitBreakerState =>
  failTimes(CIRCUIT_BREAKER_FAILURE_THRESHOLD, now)

describe('documentRefreshCircuitBreaker', () => {
  test('stays closed below the threshold and opens at it', () => {
    const now = 1_000_000

    expect(failTimes(1, now).isOpen).toBe(false)
    expect(failTimes(2, now).isOpen).toBe(false)
    expect(openBreaker(now).isOpen).toBe(true)
  })

  test('backoff is capped instead of growing without bound', () => {
    const now = 1_000_000
    // The inline implementation this module replaced computed
    // `Math.pow(2, failureCount) * 1000` with no cap on the exponent or the
    // counter, so ten consecutive failures produced 2 ** 10 * 1000 =
    // 1_024_000 ms (~17 minutes) of silence on the document list.
    const state = failTimes(10, now)

    expect(state.nextRetryTime).not.toBeNull()
    expect(state.nextRetryTime! - now).toBe(CIRCUIT_BREAKER_MAX_BACKOFF_MS)
    expect(state.nextRetryTime! - now).not.toBe(2 ** 10 * 1000)
  })

  test('failure counter is capped', () => {
    expect(failTimes(50, 1_000_000).failureCount).toBe(
      CIRCUIT_BREAKER_MAX_FAILURE_COUNT
    )
  })

  test('backoff helper clamps at the max delay', () => {
    expect(circuitBreakerBackoffMs(3)).toBe(8_000)
    expect(circuitBreakerBackoffMs(5)).toBe(32_000)
    expect(circuitBreakerBackoffMs(6)).toBe(CIRCUIT_BREAKER_MAX_BACKOFF_MS)
    expect(circuitBreakerBackoffMs(30)).toBe(CIRCUIT_BREAKER_MAX_BACKOFF_MS)
  })

  test('admits everything while closed', () => {
    const decision = admitCircuitBreakerRequest(initialCircuitBreakerState, 1)

    expect(decision.admitted).toBe(true)
    expect(decision.state).toBe(initialCircuitBreakerState)
  })

  test('blocks requests while the backoff window is open', () => {
    const now = 1_000_000
    const open = openBreaker(now)

    const decision = admitCircuitBreakerRequest(open, now + 1)

    expect(decision.admitted).toBe(false)
    expect(decision.state).toEqual(open)
  })

  test('half-open admits exactly one probe and stays open until it settles', () => {
    const now = 1_000_000
    const open = openBreaker(now)
    const deadline = open.nextRetryTime!

    const first = admitCircuitBreakerRequest(open, deadline)
    expect(first.admitted).toBe(true)
    // The breaker must NOT close on admission: closing here is what let every
    // later poll tick through while the probe was still in flight.
    expect(first.state.isOpen).toBe(true)
    expect(first.state.halfOpenSince).toBe(deadline)
    expect(isCircuitBreakerBlocked(first.state, deadline)).toBe(true)

    // Poll ticks arriving while the probe is unsettled get nothing.
    for (const offset of [1, 5_000, 10_000, 30_000]) {
      const next = admitCircuitBreakerRequest(first.state, deadline + offset)
      expect(next.admitted).toBe(false)
      expect(next.state).toEqual(first.state)
    }
  })

  test('a probe that never settles does not wedge the breaker shut', () => {
    const now = 1_000_000
    const open = openBreaker(now)
    const deadline = open.nextRetryTime!
    const probing = admitCircuitBreakerRequest(open, deadline).state

    const stillBlocked = admitCircuitBreakerRequest(
      probing,
      deadline + CIRCUIT_BREAKER_PROBE_TIMEOUT_MS - 1
    )
    expect(stillBlocked.admitted).toBe(false)

    const readmitted = admitCircuitBreakerRequest(
      probing,
      deadline + CIRCUIT_BREAKER_PROBE_TIMEOUT_MS
    )
    expect(readmitted.admitted).toBe(true)
    expect(readmitted.state.halfOpenSince).toBe(
      deadline + CIRCUIT_BREAKER_PROBE_TIMEOUT_MS
    )
  })

  test('a failed probe re-arms the backoff and clears the probe slot', () => {
    const now = 1_000_000
    const open = openBreaker(now)
    const deadline = open.nextRetryTime!
    const probing = admitCircuitBreakerRequest(open, deadline).state

    const failed = recordCircuitBreakerFailure(probing, deadline + 30_000)

    expect(failed.isOpen).toBe(true)
    expect(failed.halfOpenSince).toBeNull()
    expect(failed.failureCount).toBe(open.failureCount + 1)
    expect(failed.nextRetryTime).toBe(
      deadline + 30_000 + circuitBreakerBackoffMs(failed.failureCount)
    )
    expect(admitCircuitBreakerRequest(failed, deadline + 30_001).admitted).toBe(false)
  })

  test('a successful probe clears the whole state', () => {
    expect(resetCircuitBreaker()).toEqual(initialCircuitBreakerState)
    expect(resetCircuitBreaker()).not.toBe(initialCircuitBreakerState)
    expect(isCircuitBreakerBlocked(resetCircuitBreaker(), 1)).toBe(false)
  })

  test('isCircuitBreakerBlocked reports the open state without admitting', () => {
    const now = 1_000_000
    const open = openBreaker(now)

    // Blocked inside the backoff window, and unblocked at the deadline — but
    // either way the read must not hand out the half-open slot.
    expect(isCircuitBreakerBlocked(open, now + 1)).toBe(true)
    expect(isCircuitBreakerBlocked(open, open.nextRetryTime!)).toBe(false)
    expect(open.halfOpenSince).toBeNull()
  })
})
