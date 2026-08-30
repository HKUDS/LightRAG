import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import { createFutureGuardedStorage, hasFutureEnvelope } from './guardedStorage'

/**
 * Rollback safety shared by ALL split persist keys (settings-storage,
 * query-settings-storage, both retrieval histories): a NEWER client's
 * envelope must be neither hydrated nor overwritten — including when it
 * arrives mid-session from a newer tab — so the guard is evaluated on every
 * storage operation and diverts writes to a session-only fallback.
 *
 * Isolation: globalThis.localStorage is swapped for a stub and RESTORED in
 * afterAll (the guard reads it dynamically at call time); the store
 * singletons touched below are only READ (persist options).
 */

const data = new Map<string, string>()
const stub = {
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

let previousDescriptor: PropertyDescriptor | undefined

beforeAll(() => {
  previousDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', {
    value: stub,
    configurable: true
  })
})

afterAll(() => {
  if (previousDescriptor) {
    Object.defineProperty(globalThis, 'localStorage', previousDescriptor)
  } else {
    delete (globalThis as Record<string, unknown>).localStorage
  }
})

const envelope = (version: unknown) => JSON.stringify({ state: {}, version })

describe('hasFutureEnvelope', () => {
  test('detects only versions strictly above the current one', () => {
    stub.setItem('k', envelope(2))
    expect(hasFutureEnvelope('k', 1, stub)).toBe(true)
    stub.setItem('k', envelope(1))
    expect(hasFutureEnvelope('k', 1, stub)).toBe(false)
    stub.setItem('k', envelope(0))
    expect(hasFutureEnvelope('k', 1, stub)).toBe(false)
  })

  test('absent, corrupt and non-numeric envelopes are not future', () => {
    stub.removeItem('k')
    expect(hasFutureEnvelope('k', 1, stub)).toBe(false)
    stub.setItem('k', '{not json')
    expect(hasFutureEnvelope('k', 1, stub)).toBe(false)
    stub.setItem('k', envelope('later'))
    expect(hasFutureEnvelope('k', 1, stub)).toBe(false)
    expect(hasFutureEnvelope('k', 1, null)).toBe(false)
    stub.removeItem('k')
  })
})

describe('createFutureGuardedStorage', () => {
  test('writes divert while a future envelope is present, per key', () => {
    const storage = createFutureGuardedStorage(1)
    const future = envelope(2)
    stub.setItem('a', future)

    storage.setItem('a', envelope(1))
    // The newer envelope survives byte-for-byte…
    expect(stub.getItem('a')).toBe(future)
    // …and the diverted write stays readable within this session.
    expect(storage.getItem('a')).toBe(envelope(1))

    // A DIFFERENT key without a future envelope writes through normally.
    storage.setItem('b', envelope(1))
    expect(stub.getItem('b')).toBe(envelope(1))
    expect(storage.getItem('b')).toBe(envelope(1))

    // removeItem never deletes a future envelope.
    storage.removeItem('a')
    expect(stub.getItem('a')).toBe(future)

    stub.removeItem('a')
    stub.removeItem('b')
  })

  test('extraDivert diverts every operation while its condition holds', () => {
    let migrating = true
    const storage = createFutureGuardedStorage(1, () => migrating)

    stub.setItem('m', envelope(1))
    storage.setItem('m', envelope(1).replace('{}', '{"x":1}'))
    // The pre-existing value survives; the write stayed session-local.
    expect(stub.getItem('m')).toBe(envelope(1))
    expect(storage.getItem('m')).toContain('"x":1')

    migrating = false
    storage.setItem('m', envelope(1))
    expect(stub.getItem('m')).toBe(envelope(1))
    expect(storage.getItem('m')).toBe(envelope(1))
    stub.removeItem('m')
  })

  test('once the future envelope is gone, reads and writes hit localStorage again', () => {
    const storage = createFutureGuardedStorage(1)
    stub.setItem('a', envelope(2))
    storage.setItem('a', envelope(1)) // diverted
    stub.removeItem('a') // the newer tab's envelope disappears

    expect(storage.getItem('a')).toBeNull()
    storage.setItem('a', envelope(1))
    expect(stub.getItem('a')).toBe(envelope(1))
    stub.removeItem('a')
  })
})

describe('split stores reject future envelopes in migrate', () => {
  test('query-settings store', async () => {
    const { useQuerySettingsStore } = await import('@/stores/querySettings')
    const migrate = useQuerySettingsStore.persist.getOptions().migrate!
    expect(() => migrate({ querySettings: {} }, 2)).toThrow(/newer than this build/)
    // The current version passes through.
    const state = { querySettings: {} }
    expect(migrate(state, 1)).toBe(state)
  })

  test('retrieval history stores', async () => {
    const { createRetrievalHistoryStore } = await import('@/stores/retrievalHistory')
    // The factory's declared return type omits the persist API zustand adds.
    const store = createRetrievalHistoryStore(
      'lightrag::guarded-storage-test-history'
    ) as unknown as {
      persist: { getOptions: () => { migrate?: (s: unknown, v: number) => unknown } }
    }
    const migrate = store.persist.getOptions().migrate!
    expect(() => migrate({ history: [] }, 2)).toThrow(/newer than this build/)
    const state = { history: [] }
    expect(migrate(state, 1)).toBe(state)
  })
})
