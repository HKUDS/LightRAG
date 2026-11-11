import { describe, expect, test } from 'bun:test'

import {
  CIRCUIT_BREAKER_FAILURE_THRESHOLD,
  admitCircuitBreakerRequest,
  initialCircuitBreakerState,
  recordCircuitBreakerFailure,
  type CircuitBreakerState
} from '@/features/documentRefreshCircuitBreaker'
import { createRefreshQueue } from '@/features/documentRefreshQueue'

type Req = { id: string; auto: boolean }

/**
 * Mirrors how DocumentManager wires the breaker into the queue: only `auto`
 * requests (polling timer, activity probe, pipelineActive callback) ask for
 * admission; manual refresh and page/filter changes always run.
 */
const createHarness = (startAt = 1_000_000) => {
  let now = startAt
  let breaker: CircuitBreakerState = initialCircuitBreakerState
  const ran: string[] = []

  const queue = createRefreshQueue<Req>()

  const handlers = {
    runRequest: async (request: Req) => {
      ran.push(request.id)
      // Yield so a caller enqueueing synchronously after this one lands in the
      // pending slot, the way a real fetch would.
      await Promise.resolve()
    },
    admit: (request: Req) => {
      if (!request.auto) return true
      const decision = admitCircuitBreakerRequest(breaker, now)
      breaker = decision.state
      return decision.admitted
    },
    priority: (request: Req) => (request.auto ? 0 : 1)
  }

  return {
    ran,
    enqueue: (request: Req) => queue.enqueue(request, handlers),
    openBreaker: () => {
      for (let i = 0; i < CIRCUIT_BREAKER_FAILURE_THRESHOLD; i += 1) {
        breaker = recordCircuitBreakerFailure(breaker, now)
      }
    },
    advanceToRetryDeadline: () => {
      now = breaker.nextRetryTime ?? now
    },
    breakerState: () => breaker
  }
}

describe('documentRefreshQueue', () => {
  test('an automatic refresh is dropped when the breaker is open and the queue is idle', async () => {
    // The regression this pins: admission used to happen at the polling tick,
    // so any entrance that found the queue idle — the throttle gate's trailing
    // timer, the activity probe, the pipelineActive callback — ran its request
    // as the loop's FIRST request, with nothing consulting the breaker.
    const harness = createHarness()
    harness.openBreaker()

    await harness.enqueue({ id: 'poll-1', auto: true })

    expect(harness.ran).toEqual([])
  })

  test('manual and user-driven refreshes still run while the breaker is open', async () => {
    const harness = createHarness()
    harness.openBreaker()

    await harness.enqueue({ id: 'manual', auto: false })
    await harness.enqueue({ id: 'page-change', auto: false })

    expect(harness.ran).toEqual(['manual', 'page-change'])
  })

  test('half-open admits exactly one automatic request', async () => {
    const harness = createHarness()
    harness.openBreaker()
    harness.advanceToRetryDeadline()

    await harness.enqueue({ id: 'probe', auto: true })
    await harness.enqueue({ id: 'follow-up', auto: true })

    expect(harness.ran).toEqual(['probe'])
    // The breaker is still open with the probe outstanding — only its
    // settlement (success or failure) moves it on.
    expect(harness.breakerState().isOpen).toBe(true)
    expect(harness.breakerState().halfOpenSince).not.toBeNull()
  })

  test('a queued automatic refresh is re-checked before it runs', async () => {
    const harness = createHarness()

    // Closed breaker: the first request runs and, while it is in flight, a
    // second one queues up behind it. The breaker opens in between (as it
    // would when the first request fails), so the queued one must be dropped.
    const first = harness.enqueue({ id: 'first', auto: true })
    const queued = harness.enqueue({ id: 'queued', auto: true })
    harness.openBreaker()

    await Promise.all([first, queued])

    expect(harness.ran).toEqual(['first'])
  })

  test('requests arriving during a run collapse into a single pending slot', async () => {
    const harness = createHarness()

    const first = harness.enqueue({ id: 'first', auto: false })
    const second = harness.enqueue({ id: 'second', auto: false })
    const third = harness.enqueue({ id: 'third', auto: false })

    await Promise.all([first, second, third])

    // 'second' is superseded by 'third' in the single pending slot.
    expect(harness.ran).toEqual(['first', 'third'])
  })

  test('an automatic refresh cannot displace a queued user-intent one', async () => {
    // The regression this pins: the pending slot took the newest request
    // unconditionally, while admission moved to the point a request is issued.
    // So an `auto` request landing behind a page change discarded it and was
    // then refused by the open breaker — neither ran, and the caller saw no
    // fetch and no error.
    const harness = createHarness()
    harness.openBreaker()

    const active = harness.enqueue({ id: 'active', auto: false })
    const pageChange = harness.enqueue({ id: 'page-change', auto: false })
    const poll = harness.enqueue({ id: 'poll', auto: true })

    await Promise.all([active, pageChange, poll])

    expect(harness.ran).toEqual(['active', 'page-change'])
  })

  test('a user-intent refresh displaces a queued automatic one', async () => {
    const harness = createHarness()

    const active = harness.enqueue({ id: 'active', auto: false })
    const poll = harness.enqueue({ id: 'poll', auto: true })
    const manual = harness.enqueue({ id: 'manual', auto: false })

    await Promise.all([active, poll, manual])

    expect(harness.ran).toEqual(['active', 'manual'])
  })

  test('equal priority still means newest wins', async () => {
    const harness = createHarness()

    const active = harness.enqueue({ id: 'active', auto: true })
    const first = harness.enqueue({ id: 'poll-1', auto: true })
    const second = harness.enqueue({ id: 'poll-2', auto: true })

    await Promise.all([active, first, second])

    expect(harness.ran).toEqual(['active', 'poll-2'])
  })

  test('a dropped request does not stall the queue', async () => {
    // A rejected request runs no async work, so the loop can complete without
    // ever awaiting. The queue must not leave its busy marker behind in that
    // window, or this manual refresh would attach to a finished loop and be
    // discarded instead of running.
    const harness = createHarness()
    harness.openBreaker()

    const dropped = harness.enqueue({ id: 'poll', auto: true })
    const manual = harness.enqueue({ id: 'manual', auto: false })

    await Promise.all([dropped, manual])

    expect(harness.ran).toEqual(['manual'])
  })
})
