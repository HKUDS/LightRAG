import { describe, expect, test } from 'bun:test'

import {
  CIRCUIT_BREAKER_FAILURE_THRESHOLD,
  CIRCUIT_BREAKER_MAX_BACKOFF_MS,
  CIRCUIT_BREAKER_MAX_FAILURE_COUNT,
  circuitBreakerBackoffMs,
  evaluateCircuitBreaker,
  initialCircuitBreakerState,
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

describe('documentRefreshCircuitBreaker', () => {
  test('stays closed below the threshold and opens at it', () => {
    const now = 1_000_000

    expect(failTimes(1, now).isOpen).toBe(false)
    expect(failTimes(2, now).isOpen).toBe(false)
    expect(failTimes(CIRCUIT_BREAKER_FAILURE_THRESHOLD, now).isOpen).toBe(true)
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
    const state = failTimes(50, 1_000_000)

    expect(state.failureCount).toBe(CIRCUIT_BREAKER_MAX_FAILURE_COUNT)
  })

  test('backoff helper clamps at the max delay', () => {
    expect(circuitBreakerBackoffMs(3)).toBe(8_000)
    expect(circuitBreakerBackoffMs(5)).toBe(32_000)
    expect(circuitBreakerBackoffMs(6)).toBe(CIRCUIT_BREAKER_MAX_BACKOFF_MS)
    expect(circuitBreakerBackoffMs(30)).toBe(CIRCUIT_BREAKER_MAX_BACKOFF_MS)
  })

  test('blocks requests while the backoff window is open', () => {
    const now = 1_000_000
    const open = failTimes(CIRCUIT_BREAKER_FAILURE_THRESHOLD, now)

    const decision = evaluateCircuitBreaker(open, now + 1)

    expect(decision.isOpen).toBe(true)
    expect(decision.state).toEqual(open)
  })

  test('half-opens once the backoff window elapses, decrementing the counter', () => {
    const now = 1_000_000
    const open = failTimes(CIRCUIT_BREAKER_FAILURE_THRESHOLD, now)

    const decision = evaluateCircuitBreaker(open, open.nextRetryTime!)

    expect(decision.isOpen).toBe(false)
    expect(decision.state.isOpen).toBe(false)
    expect(decision.state.failureCount).toBe(open.failureCount - 1)
    expect(decision.state.nextRetryTime).toBeNull()
  })

  test('a closed breaker is returned untouched', () => {
    const state = failTimes(1, 1_000_000)
    const decision = evaluateCircuitBreaker(state, 2_000_000)

    expect(decision.isOpen).toBe(false)
    expect(decision.state).toBe(state)
  })

  test('a real response clears the whole state', () => {
    expect(resetCircuitBreaker()).toEqual(initialCircuitBreakerState)
    expect(resetCircuitBreaker()).not.toBe(initialCircuitBreakerState)
  })
})
