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
