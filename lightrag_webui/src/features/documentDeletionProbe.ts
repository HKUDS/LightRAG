/**
 * Completion probe for an asynchronous document deletion.
 *
 * `/documents/delete_document` answers `deletion_started`, not `deleted`: the
 * rows disappear later, from a background task. The backend does give a hard
 * completion signal, but it is a LEVEL, not an edge — the endpoint reserves
 * `busy` + `destructive_busy` inside the pipeline-status lock and awaits the
 * background task's start barrier BEFORE answering, and the background task
 * releases both only in its finally, after every `doc_status` row is gone.
 * So any `pipeline_busy === false` observed AFTER the response proves the
 * deletion has finished; there is no "not started yet" window to confuse it
 * with. What that requires of this module: every observation must come from a
 * FRESH /health request issued after the response — a snapshot cached from
 * before the delete is idle for the trivial reason.
 *
 * Reading the level is what DocumentManager could not do on its own. Its
 * `pipelineActive` effect fires on a TRANSITION, so a sub-second deletion —
 * whose whole busy window falls between two 15s health checks — is observed as
 * `false → false` and refreshes nothing, leaving deleted rows on screen until
 * the 30s idle poll. Nor can the fix ride on `startPollingInterval`: every
 * successful refresh hands `setStatusCounts` a fresh object, the polling effect
 * re-runs and reinstates the idle cadence, so an imperatively set faster
 * interval is destroyed before its first tick. This probe therefore owns its
 * own timers.
 *
 * Every terminal path except an unmount issues exactly ONE refresh — including
 * the exhausted-schedule path. That is deliberate: the "idle means finished"
 * reading rests on the endpoint's ordering contract, which is not
 * self-evidencing, so a refresh is issued on the way out whatever was
 * observed. If the contract ever breaks, the cost is one early refresh, and
 * the 5s active cadence plus the genuine busy→idle transition still cover the
 * list — the failure mode is not a stale list.
 */

/** Cumulative delays, in ms, measured from the probe's start. */
export const DELETION_PROBE_SCHEDULE = [0, 500, 1000, 2000, 4000, 8000] as const

export type DeletionProbeDeps = {
  /**
   * Issue a fresh health request and report `pipeline_busy` as observed by it.
   * Return `null` when the observation failed (unreachable backend, timeout):
   * unknown must never be read as idle, or an outage would be reported to the
   * user as a completed deletion.
   */
  observePipelineBusy: () => Promise<boolean | null>
  /**
   * Refresh the document list. Called at most once per probe run, with user
   * intent (the delete was a user action, so the circuit breaker must not be
   * able to refuse the refresh it earned).
   */
  refreshDocuments: () => void
  /** False once the owning component is gone; stops the probe without a refresh. */
  isMounted: () => boolean
  /** Called once when the probe stops, for any reason (including cancel). */
  onSettled?: () => void
  /** Override the schedule; tests use a shorter one. */
  schedule?: readonly number[]
  setTimer?: (callback: () => void, delayMs: number) => unknown
  clearTimer?: (handle: unknown) => void
}

export type DeletionProbeHandle = {
  /** Stop the probe without refreshing. Idempotent. */
  cancel: () => void
  isActive: () => boolean
}

/**
 * Start probing. The returned handle is single-use; re-entry (a second delete)
 * is expressed by cancelling the previous handle and starting a new probe, so
 * the latest action gets a fresh observation window.
 */
export const startDeletionProbe = (
  deps: DeletionProbeDeps
): DeletionProbeHandle => {
  const schedule = deps.schedule ?? DELETION_PROBE_SCHEDULE
  const setTimer =
    deps.setTimer ?? ((callback, delayMs) => setTimeout(callback, delayMs))
  const clearTimer =
    deps.clearTimer ?? ((handle) => clearTimeout(handle as never))

  let stopped = false
  let pendingTimer: unknown = null

  const settle = (refresh: boolean) => {
    if (stopped) return
    stopped = true
    pendingTimer = null
    if (refresh) deps.refreshDocuments()
    deps.onSettled?.()
  }

  const cancel = () => {
    if (stopped) return
    if (pendingTimer !== null) {
      clearTimer(pendingTimer)
      pendingTimer = null
    }
    settle(false)
  }

  const runTick = async (index: number) => {
    pendingTimer = null
    if (stopped) return
    // Not mounted: nothing to refresh, and a refresh would be a request nobody
    // reads. This is the only exit that does not confirm the list.
    if (!deps.isMounted()) {
      settle(false)
      return
    }

    const busy = await deps.observePipelineBusy()

    if (stopped) return
    if (!deps.isMounted()) {
      settle(false)
      return
    }

    const isLastTick = index === schedule.length - 1
    // `busy === false` is the completion proof (see the module comment).
    // `null` is an unusable observation, so it only ends the probe by
    // exhausting the schedule.
    if (busy === false || isLastTick) {
      settle(true)
      return
    }

    // Chained rather than scheduled up front: the next delay is measured from
    // the end of this observation, so two health requests can never overlap.
    const delay = schedule[index + 1] - schedule[index]
    pendingTimer = setTimer(() => {
      void runTick(index + 1)
    }, delay)
  }

  if (schedule.length === 0) {
    // Degenerate configuration; still honour the one-refresh contract.
    settle(true)
    return { cancel, isActive: () => false }
  }

  pendingTimer = setTimer(() => {
    void runTick(0)
  }, schedule[0])

  return { cancel, isActive: () => !stopped }
}
