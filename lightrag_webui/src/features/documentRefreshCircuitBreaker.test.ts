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
  settleCircuitBreakerProbe,
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

  describe('settleCircuitBreakerProbe', () => {
    const openWithProbe = (now: number) => {
      let state = initialCircuitBreakerState
      for (let i = 0; i < CIRCUIT_BREAKER_FAILURE_THRESHOLD; i += 1) {
        state = recordCircuitBreakerFailure(state, now)
      }
      const admission = admitCircuitBreakerRequest(state, state.nextRetryTime!)
      expect(admission.admitted).toBe(true)
      return admission.state
    }

    test('returns the probe slot without moving the failure counter', () => {
      const probing = openWithProbe(1_000)
      const settledAt = probing.halfOpenSince! + 50

      const settled = settleCircuitBreakerProbe(probing, settledAt)

      expect(settled.halfOpenSince).toBeNull()
      expect(settled.failureCount).toBe(probing.failureCount)
      expect(settled.isOpen).toBe(true)
    })

    test('re-stamps the backoff instead of leaving nextRetryTime in the past', () => {
      // Returning the slot alone is not enough: nextRetryTime is already past
      // by the time a probe runs, so the very next poll tick would be admitted
      // as another probe immediately.
      const probing = openWithProbe(1_000)
      const settledAt = probing.halfOpenSince! + 50

      const settled = settleCircuitBreakerProbe(probing, settledAt)
      const expected = settledAt + circuitBreakerBackoffMs(settled.failureCount)

      expect(settled.nextRetryTime).toBe(expected)
      expect(admitCircuitBreakerRequest(settled, expected - 1).admitted).toBe(false)
      expect(admitCircuitBreakerRequest(settled, expected).admitted).toBe(true)
    })

    test('an unsettled probe blocks automatic refreshes for the whole probe timeout', () => {
      // The regression: a request that recorded neither side left halfOpenSince
      // stamped, so every tick was refused until CIRCUIT_BREAKER_PROBE_TIMEOUT_MS
      // — 45s of stalled polling bought by a request that took 50ms.
      const probing = openWithProbe(1_000)
      const respondedAt = probing.halfOpenSince! + 50

      expect(admitCircuitBreakerRequest(probing, respondedAt).admitted).toBe(false)
      expect(
        admitCircuitBreakerRequest(
          probing,
          probing.halfOpenSince! + CIRCUIT_BREAKER_PROBE_TIMEOUT_MS - 1
        ).admitted
      ).toBe(false)

      // Settling it releases the breaker on its own backoff instead.
      const settled = settleCircuitBreakerProbe(probing, respondedAt)
      expect(settled.nextRetryTime).toBeLessThan(
        probing.halfOpenSince! + CIRCUIT_BREAKER_PROBE_TIMEOUT_MS
      )
    })

    test('is a no-op when no probe is in flight', () => {
      // The precondition for calling it unconditionally from a finally block.
      const closed = initialCircuitBreakerState
      expect(settleCircuitBreakerProbe(closed, 5_000)).toBe(closed)

      const failedOnce = recordCircuitBreakerFailure(closed, 5_000)
      expect(settleCircuitBreakerProbe(failedOnce, 6_000)).toBe(failedOnce)
    })
  })

  test('an answer from the application does not leave a stale failure count', () => {
    // Consumer contract, pinned here because DocumentManager has no component
    // test: a permanent 4xx resets the breaker, so failures separated by one
    // cannot accumulate. CIRCUIT_BREAKER_FAILURE_THRESHOLD documents itself as
    // CONSECUTIVE failures, and only a reset makes that true.
    let state = initialCircuitBreakerState
    state = recordCircuitBreakerFailure(state, 1_000)
    state = recordCircuitBreakerFailure(state, 2_000)
    expect(state.isOpen).toBe(false)

    // ... a 403 arrives: the application answered, so the breaker resets.
    state = resetCircuitBreaker()

    state = recordCircuitBreakerFailure(state, 4_000)
    expect(state.failureCount).toBe(1)
    expect(state.isOpen).toBe(false)
  })
})
