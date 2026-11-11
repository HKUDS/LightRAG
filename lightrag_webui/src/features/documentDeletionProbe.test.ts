import { describe, expect, test } from 'bun:test'

import {
  DELETION_PROBE_SCHEDULE,
  startDeletionProbe,
  type DeletionProbeHandle
} from '@/features/documentDeletionProbe'

type Observation = boolean | null

/**
 * Manual timer queue: the probe chains its ticks, so the harness has to run
 * them one at a time and let the awaited observation settle in between.
 */
const createHarness = (
  observations: Observation[],
  options: { schedule?: readonly number[]; mounted?: () => boolean } = {}
) => {
  const timers: { callback: () => void; delayMs: number }[] = []
  const delays: number[] = []
  const observed: number[] = []
  let refreshes = 0
  let settled = 0

  const deps = {
    observePipelineBusy: async () => {
      const index = observed.length
      observed.push(index)
      // Two microtask hops, so a test that flushes only once would show up as
      // a missing tick rather than passing by accident.
      await Promise.resolve()
      await Promise.resolve()
      return observations[Math.min(index, observations.length - 1)]
    },
    refreshDocuments: () => {
      refreshes += 1
    },
    isMounted: options.mounted ?? (() => true),
    onSettled: () => {
      settled += 1
    },
    schedule: options.schedule,
    setTimer: (callback: () => void, delayMs: number) => {
      delays.push(delayMs)
      timers.push({ callback, delayMs })
      return timers.length - 1
    },
    clearTimer: (handle: unknown) => {
      const index = handle as number
      if (timers[index]) timers[index].callback = () => {}
    }
  }

  const flush = async () => {
    for (let i = 0; i < 8; i += 1) await Promise.resolve()
  }

  /** Fire the timer the probe is currently waiting on, then settle its await. */
  const tick = async () => {
    const next = timers.shift()
    if (!next) throw new Error('no timer pending')
    next.callback()
    await flush()
  }

  return {
    deps,
    tick,
    flush,
    delays,
    pendingTimers: () => timers.length,
    refreshes: () => refreshes,
    settledCount: () => settled,
    observations: () => observed.length
  }
}

describe('startDeletionProbe', () => {
  test('idle on the very first observation confirms the deletion', async () => {
    // The fix-proof case at module level: the busy window of a sub-second
    // deletion is never observed, so the FIRST reading is already idle. A
    // transition-based rule ("refresh when busy flips to idle") sees nothing
    // happen here and refreshes nothing.
    const harness = createHarness([false])
    const probe = startDeletionProbe(harness.deps)

    await harness.tick()

    expect(harness.refreshes()).toBe(1)
    expect(harness.observations()).toBe(1)
    expect(probe.isActive()).toBe(false)
    expect(harness.pendingTimers()).toBe(0)
  })

  test('keeps observing while the pipeline is still busy', async () => {
    const harness = createHarness([true, true, false], {
      schedule: [0, 500, 1000]
    })
    startDeletionProbe(harness.deps)

    await harness.tick()
    expect(harness.refreshes()).toBe(0)

    await harness.tick()
    expect(harness.refreshes()).toBe(0)

    await harness.tick()
    expect(harness.refreshes()).toBe(1)
    expect(harness.observations()).toBe(3)
  })

  test('an exhausted schedule still issues the one confirming refresh', async () => {
    // "Idle means finished" rests on the endpoint's ordering contract, which is
    // not self-evidencing. Whatever was observed, the list gets confirmed on
    // the way out — the failure mode must never be a stale list.
    const harness = createHarness([true], { schedule: [0, 500] })
    startDeletionProbe(harness.deps)

    await harness.tick()
    await harness.tick()

    expect(harness.refreshes()).toBe(1)
    expect(harness.settledCount()).toBe(1)
  })

  test('a failed observation is unknown, not idle', async () => {
    const harness = createHarness([null, false], { schedule: [0, 500] })
    startDeletionProbe(harness.deps)

    await harness.tick()
    // `null` must not be read as "finished": an unreachable backend would be
    // reported to the user as a completed deletion.
    expect(harness.refreshes()).toBe(0)

    await harness.tick()
    expect(harness.refreshes()).toBe(1)
  })

  test('ticks are chained by inter-tick gaps, not absolute delays', async () => {
    const harness = createHarness([true, true, true], {
      schedule: [0, 500, 2000]
    })
    startDeletionProbe(harness.deps)

    await harness.tick()
    await harness.tick()

    // Scheduled after the previous observation settled, so two health requests
    // can never overlap.
    expect(harness.delays).toEqual([0, 500, 1500])
  })

  test('cancel stops the probe without refreshing', async () => {
    const harness = createHarness([true], { schedule: [0, 500] })
    const probe = startDeletionProbe(harness.deps)

    await harness.tick()
    probe.cancel()
    await harness.flush()

    expect(harness.refreshes()).toBe(0)
    expect(probe.isActive()).toBe(false)
    expect(harness.settledCount()).toBe(1)

    // The cancelled tick must not resurrect the probe.
    await harness.tick()
    expect(harness.refreshes()).toBe(0)
    expect(harness.observations()).toBe(1)
  })

  test('cancel during an in-flight observation is honoured', async () => {
    let release: (value: boolean) => void = () => {}
    let refreshes = 0
    const handle: { probe: DeletionProbeHandle | null } = { probe: null }
    const timers: (() => void)[] = []

    handle.probe = startDeletionProbe({
      observePipelineBusy: () =>
        new Promise<boolean>((resolve) => {
          release = resolve
        }),
      refreshDocuments: () => {
        refreshes += 1
      },
      isMounted: () => true,
      schedule: [0, 500],
      setTimer: (callback) => {
        timers.push(callback)
        return timers.length - 1
      },
      clearTimer: () => {}
    })

    timers.shift()?.()
    handle.probe.cancel()
    release(false)
    await Promise.resolve()
    await Promise.resolve()

    expect(refreshes).toBe(0)
  })

  test('an unmounted component ends the probe without a request', async () => {
    let mounted = true
    const harness = createHarness([false], {
      schedule: [0, 500],
      mounted: () => mounted
    })
    startDeletionProbe(harness.deps)

    mounted = false
    await harness.tick()

    expect(harness.observations()).toBe(0)
    expect(harness.refreshes()).toBe(0)
  })

  test('unmounting while the observation is in flight refreshes nothing', async () => {
    let mounted = true
    const harness = createHarness([false], {
      schedule: [0, 500],
      mounted: () => mounted
    })
    startDeletionProbe(harness.deps)

    // Fire the tick, drop the mount before its observation settles.
    const pending = harness.tick()
    mounted = false
    await pending

    expect(harness.observations()).toBe(1)
    expect(harness.refreshes()).toBe(0)
  })

  test('the shipped schedule spans the fast and the slow deletion', () => {
    expect(DELETION_PROBE_SCHEDULE[0]).toBe(0)
    // Monotonic, so the chained inter-tick gaps stay non-negative.
    const gaps = DELETION_PROBE_SCHEDULE.map((delay, index) =>
      index === 0 ? delay : delay - DELETION_PROBE_SCHEDULE[index - 1]
    )
    expect(gaps.every((gap) => gap >= 0)).toBe(true)
    expect(DELETION_PROBE_SCHEDULE.length).toBeGreaterThan(1)
  })
})
