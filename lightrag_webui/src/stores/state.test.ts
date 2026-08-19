import { afterAll, beforeAll, describe, expect, mock, test } from 'bun:test'

type StateModule = typeof import('./state')

const storageMock = () => {
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

// checkHealth is swapped per step; the store must never see the network.
let healthResponse:
  | { status: 'healthy'; pipeline_busy: boolean; api_docs_available?: boolean }
  | { status: 'error'; message: string }

const checkHealth = mock(async () => healthResponse)

let stateModule: StateModule
// mock.module rewires the module registry for the whole bun test process, so
// the real exports are captured up front and restored in afterAll — otherwise
// later test files importing '@/api/lightrag' would see the stub.
let realApiModule: Record<string, unknown>

beforeAll(async () => {
  Object.defineProperty(globalThis, 'localStorage', {
    value: storageMock(),
    configurable: true
  })
  Object.defineProperty(globalThis, 'sessionStorage', {
    value: storageMock(),
    configurable: true
  })
  realApiModule = { ...(await import('@/api/lightrag')) }
  mock.module('@/api/lightrag', () => ({ ...realApiModule, checkHealth }))

  stateModule = await import('./state')
})

afterAll(() => {
  mock.module('@/api/lightrag', () => realApiModule)
})

const healthy = (
  extra: { api_docs_available?: boolean } = {}
): { status: 'healthy'; pipeline_busy: boolean; api_docs_available?: boolean } => ({
  status: 'healthy',
  pipeline_busy: false,
  ...extra
})

describe('apiDocsCapability state transitions', () => {
  // One sequential walk through the machine: the retention semantics only
  // show up across consecutive checks, so the steps share store state on
  // purpose (RFC #3671).
  test('unknown → retained on failure → set only by successful responses', async () => {
    const backend = () => stateModule.useBackendState.getState()

    // Initial state: no successful response yet — the docs entry point must
    // stay hidden rather than guessed.
    expect(backend().apiDocsCapability).toBe('unknown')

    // A failed check before any success is NOT an old backend: stay unknown.
    healthResponse = { status: 'error', message: 'connection refused' }
    expect(await backend().check()).toBe(false)
    expect(backend().apiDocsCapability).toBe('unknown')
    expect(backend().status).toBeNull()

    // A successful response with the field false marks docs unavailable.
    healthResponse = healthy({ api_docs_available: false })
    expect(await backend().check()).toBe(true)
    expect(backend().apiDocsCapability).toBe('unavailable')

    // A later failure resets the transient status but must retain the last
    // known capability — a blip must not resurface a disabled entry point.
    healthResponse = { status: 'error', message: 'timeout' }
    expect(await backend().check()).toBe(false)
    expect(backend().status).toBeNull()
    expect(backend().apiDocsCapability).toBe('unavailable')

    // Re-enabled backend: the field flips the capability back.
    healthResponse = healthy({ api_docs_available: true })
    expect(await backend().check()).toBe(true)
    expect(backend().apiDocsCapability).toBe('available')

    // Field absent on a successful response = older backend, which always
    // serves /docs. Only valid here, after a real response.
    healthResponse = healthy()
    expect(await backend().check()).toBe(true)
    expect(backend().apiDocsCapability).toBe('available')
  })
})
