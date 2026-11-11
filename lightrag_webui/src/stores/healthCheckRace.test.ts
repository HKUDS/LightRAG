import { afterAll, beforeAll, describe, expect, spyOn, test } from 'bun:test'
import type { LightragStatus } from '@/api/lightrag'

/**
 * Last-caller-wins on useBackendState.check(): concurrent health checks (a
 * credential probe racing a save-triggered re-probe, or overlapping periodic
 * checks) must not let COMPLETION ORDER decide the shared state. The pinned
 * race: an older check carrying the previous API key's failure finishes
 * AFTER the newest check already validated the replacement — its result must
 * be discarded, not written over health/message (where ApiKeyAlert's watch
 * would wrongly reopen the dialog).
 *
 * Isolation: checkHealth is stubbed with a RESTORABLE spy on the real module
 * namespace (no mock.module); the backend store's touched fields are reset
 * afterwards.
 */

const storageStub = () => {
  const data = new Map<string, string>()
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value)
    },
    removeItem: (key: string) => {
      data.delete(key)
    },
    clear: () => {
      data.clear()
    }
  }
}

interface Deferred {
  promise: Promise<unknown>
  resolve: (value: unknown) => void
}
const deferred = (): Deferred => {
  let resolve!: (value: unknown) => void
  const promise = new Promise((r) => {
    resolve = r
  })
  return { promise, resolve }
}

let useBackendState: typeof import('./state').useBackendState
let checkHealthSpy: ReturnType<typeof spyOn>

beforeAll(async () => {
  if (typeof localStorage === 'undefined') {
    Object.defineProperty(globalThis, 'localStorage', {
      value: storageStub(),
      configurable: true
    })
  }
  if (typeof sessionStorage === 'undefined') {
    Object.defineProperty(globalThis, 'sessionStorage', {
      value: storageStub(),
      configurable: true
    })
  }
  const apiModule = await import('@/api/lightrag')
  useBackendState = (await import('./state')).useBackendState
  checkHealthSpy = spyOn(apiModule, 'checkHealth')
})

afterAll(() => {
  checkHealthSpy.mockRestore()
  useBackendState.setState({
    health: true,
    message: null,
    messageTitle: null,
    status: null,
    apiDocsCapability: 'unknown'
  })
})

describe('useBackendState.check() generation guard', () => {
  test('a STALE check finishing last cannot overwrite the newest result', async () => {
    const staleResponse = deferred()
    const newestResponse = deferred()
    checkHealthSpy
      .mockImplementationOnce(() => staleResponse.promise)
      .mockImplementationOnce(() => newestResponse.promise)

    // Old probe (previous, invalid API key) goes out first…
    const staleCheck = useBackendState.getState().check()
    // …then the save-triggered probe for the VALID replacement key.
    const newestCheck = useBackendState.getState().check()

    // The newest check completes first: healthy.
    newestResponse.resolve({ status: 'healthy' } as LightragStatus)
    await newestCheck
    expect(useBackendState.getState().health).toBe(true)
    expect(useBackendState.getState().message).toBeNull()

    // The stale check completes LAST with the old key's failure — discarded.
    staleResponse.resolve({ status: 'error', message: 'Invalid API Key' })
    const staleResult = await staleCheck
    expect(staleResult).toBe(true) // reports the state the newest check set
    expect(useBackendState.getState().health).toBe(true)
    expect(useBackendState.getState().message).toBeNull()
  })

  test('a single (newest) failing check still writes its result', async () => {
    const response = deferred()
    checkHealthSpy.mockImplementationOnce(() => response.promise)

    const check = useBackendState.getState().check()
    response.resolve({ status: 'error', message: 'Invalid API Key' })

    expect(await check).toBe(false)
    expect(useBackendState.getState().health).toBe(false)
    expect(useBackendState.getState().message).toBe('Invalid API Key')
  })
})
