import { afterAll, afterEach, beforeAll, describe, expect, spyOn, test } from 'bun:test'

/**
 * Workspace credential probe: the entry has no manual API-key control, so
 * every path that can discover a bad key must be able to REQUEST the dialog —
 * the startup probe, a dialog close, and (the case pinned here) a query the
 * server rejected on credential grounds after the key was rotated.
 *
 * The request signal is a monotonic COUNTER, not a boolean or a message, so
 * two consecutive failures with the identical message still reopen the
 * dialog. Isolation: `verifyCredentials` is stubbed with a restorable spy on
 * the real module namespace; no module mocks.
 */

const storageStub = () => ({
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
  clear: () => {}
})

let probe: typeof import('./credentialProbe')
let useBackendState: typeof import('@/stores/state').useBackendState
let verifySpy: ReturnType<typeof spyOn>

beforeAll(async () => {
  for (const key of ['localStorage', 'sessionStorage'] as const) {
    if (typeof globalThis[key] === 'undefined') {
      Object.defineProperty(globalThis, key, {
        value: storageStub(),
        configurable: true
      })
    }
  }
  const apiModule = await import('@/api/lightrag')
  verifySpy = spyOn(apiModule, 'verifyCredentials')
  probe = await import('./credentialProbe')
  useBackendState = (await import('@/stores/state')).useBackendState
})

afterAll(() => {
  verifySpy.mockRestore()
})

afterEach(() => {
  useBackendState.getState().clear()
})

const apiKeyRejection = () =>
  Promise.reject(new Error('403 Forbidden\n{"detail":"Invalid API Key"}\n/auth/verify'))

describe('isApiKeyFailure', () => {
  test('recognizes the server details the dialog keys on', () => {
    expect(probe.isApiKeyFailure('403 Forbidden\n{"detail":"Invalid API Key"}')).toBe(true)
    expect(probe.isApiKeyFailure('403 Forbidden\n{"detail":"API Key required"}')).toBe(true)
  })

  test('leaves model, network and server failures alone', () => {
    expect(probe.isApiKeyFailure('Network connection error')).toBe(false)
    expect(probe.isApiKeyFailure('Server error, please try again later (500)')).toBe(false)
    expect(probe.isApiKeyFailure('LLM returned an empty response')).toBe(false)
  })
})

describe('runCredentialProbe', () => {
  test('a rejected key requests the dialog and surfaces the message', async () => {
    const errorSpy = spyOn(console, 'error').mockImplementation(() => {})
    try {
      const before = probe.useCredentialProbeStore.getState().apiKeyDialogRequests
      verifySpy.mockImplementationOnce(apiKeyRejection)

      probe.runCredentialProbe()
      await Promise.resolve()
      await Promise.resolve()

      expect(probe.useCredentialProbeStore.getState().apiKeyDialogRequests).toBe(before + 1)
      expect(useBackendState.getState().message).toContain('Invalid API Key')
    } finally {
      errorSpy.mockRestore()
    }
  })

  test('a SECOND identical failure requests the dialog again (counter, not flag)', async () => {
    const before = probe.useCredentialProbeStore.getState().apiKeyDialogRequests
    verifySpy.mockImplementationOnce(apiKeyRejection)

    probe.runCredentialProbe()
    await Promise.resolve()
    await Promise.resolve()

    expect(probe.useCredentialProbeStore.getState().apiKeyDialogRequests).toBe(before + 1)
  })

  test('valid credentials request nothing and clear a leftover key error', async () => {
    useBackendState.getState().setErrorMessage('Invalid API Key', 'API Key required')
    const before = probe.useCredentialProbeStore.getState().apiKeyDialogRequests
    verifySpy.mockImplementationOnce(() => Promise.resolve())

    probe.runCredentialProbe()
    await Promise.resolve()
    await Promise.resolve()

    expect(probe.useCredentialProbeStore.getState().apiKeyDialogRequests).toBe(before)
    expect(useBackendState.getState().message).toBeNull()
  })

  test('a non-credential failure requests nothing', async () => {
    const before = probe.useCredentialProbeStore.getState().apiKeyDialogRequests
    verifySpy.mockImplementationOnce(() => Promise.reject(new Error('Failed to fetch')))

    probe.runCredentialProbe()
    await Promise.resolve()
    await Promise.resolve()

    expect(probe.useCredentialProbeStore.getState().apiKeyDialogRequests).toBe(before)
  })
})

describe('handleApiKeyDialogClose', () => {
  test('a DISMISSAL probes nothing, so the dialog it would reopen stays closed', async () => {
    // Regression (inescapable modal): every close used to re-probe. The
    // dialog is modal and its overlay covers the header, its only button
    // saved, and a probe that fails REQUESTS the dialog again — so a user
    // without a valid key could not close it, log out, or get back to the
    // welcome page. A dismissal submits nothing, so its probe could only
    // reproduce the failure that opened the dialog.
    useBackendState.getState().setErrorMessage('Invalid API Key', 'API Key required')
    const requestsBefore = probe.useCredentialProbeStore.getState().apiKeyDialogRequests
    const callsBefore = verifySpy.mock.calls.length

    probe.handleApiKeyDialogClose('dismiss')
    await Promise.resolve()
    await Promise.resolve()

    // The server was never asked, so nothing can request the dialog again…
    expect(verifySpy.mock.calls.length).toBe(callsBefore)
    expect(probe.useCredentialProbeStore.getState().apiKeyDialogRequests).toBe(
      requestsBefore
    )
    // …and the message the dialog showed is left as it was.
    expect(useBackendState.getState().message).toContain('Invalid API Key')
  })

  test('an undefined reason is treated as a dismissal, not a save', async () => {
    const callsBefore = verifySpy.mock.calls.length

    probe.handleApiKeyDialogClose()
    await Promise.resolve()
    await Promise.resolve()

    expect(verifySpy.mock.calls.length).toBe(callsBefore)
  })

  test('a SAVE re-verifies, and a still-bad key reopens the dialog', async () => {
    const errorSpy = spyOn(console, 'error').mockImplementation(() => {})
    try {
      const before = probe.useCredentialProbeStore.getState().apiKeyDialogRequests
      verifySpy.mockImplementationOnce(apiKeyRejection)

      probe.handleApiKeyDialogClose('save')
      await Promise.resolve()
      await Promise.resolve()

      expect(probe.useCredentialProbeStore.getState().apiKeyDialogRequests).toBe(before + 1)
      expect(useBackendState.getState().message).toContain('Invalid API Key')
    } finally {
      errorSpy.mockRestore()
    }
  })

  test('a SAVE that fixes the key clears the stale error and requests nothing', async () => {
    useBackendState.getState().setErrorMessage('Invalid API Key', 'API Key required')
    const before = probe.useCredentialProbeStore.getState().apiKeyDialogRequests
    verifySpy.mockImplementationOnce(() => Promise.resolve())

    probe.handleApiKeyDialogClose('save')
    await Promise.resolve()
    await Promise.resolve()

    expect(probe.useCredentialProbeStore.getState().apiKeyDialogRequests).toBe(before)
    expect(useBackendState.getState().message).toBeNull()
  })
})

/**
 * Concurrency: only the NEWEST probe may write.
 *
 * The entry re-probes on every dialog save, so a user who saves a wrong key
 * and immediately saves the right one leaves two requests in flight. Whoever
 * settles LAST would otherwise decide the outcome, and the orders that hurt
 * are the ones where the stale probe settles second — the reason
 * `runCredentialProbe` stamps a generation and both continuations check it.
 *
 * Neither direction was covered: deleting either guard, or simply not
 * incrementing the counter, left this suite green. Both are asserted here,
 * because they fail differently — one reopens a dialog over a key that now
 * works, the other silently erases the error that should have opened it.
 *
 * The sibling implementations of this same pattern (`stores/customization`'s
 * requestCounter, `stores/state`'s healthCheckGeneration) are already pinned;
 * this closes the third.
 */
describe('runCredentialProbe generation guard', () => {
  /** A promise whose settlement this test controls. */
  const deferred = () => {
    let resolve!: () => void
    let reject!: (error: unknown) => void
    const promise = new Promise<void>((res, rej) => {
      resolve = res as () => void
      reject = rej
    })
    return { promise, resolve, reject }
  }

  // The probe chains .then/.catch onto the stubbed promise, so a settlement
  // needs several microtask turns to reach the store.
  const flush = async () => {
    for (let turn = 0; turn < 5; turn++) await Promise.resolve()
  }

  test('a stale FAILURE cannot reopen the dialog over a key that now works', async () => {
    const errorSpy = spyOn(console, 'error').mockImplementation(() => {})
    try {
      const stale = deferred()
      const newest = deferred()
      verifySpy.mockImplementationOnce(() => stale.promise)
      verifySpy.mockImplementationOnce(() => newest.promise)

      probe.runCredentialProbe() // wrong key
      probe.runCredentialProbe() // corrected key, supersedes the first
      const before = probe.useCredentialProbeStore.getState().apiKeyDialogRequests

      newest.resolve()
      await flush()
      // The superseded probe only now comes back, rejected.
      stale.reject(new Error('403 Forbidden\n{"detail":"Invalid API Key"}'))
      await flush()

      expect(probe.useCredentialProbeStore.getState().apiKeyDialogRequests).toBe(before)
      expect(useBackendState.getState().message).toBeNull()
    } finally {
      errorSpy.mockRestore()
    }
  })

  test('a stale SUCCESS cannot erase the error a newer probe just raised', async () => {
    const errorSpy = spyOn(console, 'error').mockImplementation(() => {})
    try {
      const stale = deferred()
      const newest = deferred()
      verifySpy.mockImplementationOnce(() => stale.promise)
      verifySpy.mockImplementationOnce(() => newest.promise)

      probe.runCredentialProbe() // key still valid at this point
      probe.runCredentialProbe() // key rotated server-side, supersedes it

      newest.reject(new Error('403 Forbidden\n{"detail":"API Key required"}'))
      await flush()
      const requested = probe.useCredentialProbeStore.getState().apiKeyDialogRequests
      expect(useBackendState.getState().message).toContain('API Key required')

      // The superseded probe returns successfully afterwards; its "credentials
      // accepted" cleanup must not run, or the dialog opens on a blank slate
      // with nothing explaining why.
      stale.resolve()
      await flush()

      expect(useBackendState.getState().message).toContain('API Key required')
      expect(probe.useCredentialProbeStore.getState().apiKeyDialogRequests).toBe(requested)
    } finally {
      errorSpy.mockRestore()
    }
  })
})
