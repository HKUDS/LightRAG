/**
 * Serializing queue for document list refreshes.
 *
 * At most one refresh runs at a time; anything arriving while one is in flight
 * collapses into a single pending slot (the newest wins) and runs once the
 * active one settles.
 *
 * `admit` is consulted immediately before a request would run — for the first
 * request and for every queued one alike. That placement is the point of this
 * module: the circuit breaker hands out a half-open probe to exactly one
 * request, so the admission has to happen where the request is actually
 * issued. Checking it at the call site instead leaves every entrance that can
 * find the queue idle (the throttle gate's trailing timer, the activity probe,
 * the pipelineActive callback) free to walk straight past an open breaker.
 *
 * Handlers ride along with each enqueue rather than being captured at
 * construction, so the queue instance stays stable for the life of the
 * component while still calling the current render's closures.
 */
export type RefreshQueueHandlers<T> = {
  runRequest: (request: T) => Promise<void>
  /** Return false to drop the request without running it. */
  admit: (request: T) => boolean
}

export type RefreshQueue<T> = {
  /**
   * Run `request`, or collapse it into the pending slot when another refresh
   * is already running. Resolves once the queue is idle again.
   */
  enqueue: (request: T, handlers: RefreshQueueHandlers<T>) => Promise<void>
}

export const createRefreshQueue = <T>(): RefreshQueue<T> => {
  let activeLoop: Promise<void> | null = null
  let pendingRequest: T | null = null
  let currentHandlers: RefreshQueueHandlers<T> | null = null

  const enqueue = async (
    request: T,
    handlers: RefreshQueueHandlers<T>
  ): Promise<void> => {
    currentHandlers = handlers

    if (activeLoop) {
      pendingRequest = request
      await activeLoop
      return
    }

    // The busy marker is published BEFORE the loop body and retired in its
    // finally, both synchronously with respect to the loop. Publishing it
    // afterwards (`const loop = (async () => {...})()` then assigning) leaves a
    // window when the body completes without ever awaiting — which is exactly
    // what a request `admit` rejects does: a caller enqueueing in that window
    // would attach to a loop that has already finished, and its request would
    // be cleared by the finally below instead of running.
    let releaseLoop: () => void = () => {}
    const loop = new Promise<void>((resolve) => {
      releaseLoop = resolve
    })
    activeLoop = loop

    try {
      let nextRequest: T | null = request

      while (nextRequest) {
        // Clear before running: anything enqueued during the request belongs
        // to the next iteration, not to this one.
        pendingRequest = null

        // Read per iteration — a request queued behind this one may have
        // brought newer handlers with it.
        const active = currentHandlers as RefreshQueueHandlers<T>

        if (active.admit(nextRequest)) {
          await active.runRequest(nextRequest)
        }

        nextRequest = pendingRequest
      }
    } finally {
      if (activeLoop === loop) {
        activeLoop = null
      }
      pendingRequest = null
      releaseLoop()
    }
  }

  return { enqueue }
}
