import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import { RETRIEVAL_HISTORY_STORE_VERSION } from '@/migrations/splitSettingsStorage'

/**
 * Bounded-capacity persistence for the retrieval histories (product
 * decision: the chat history is a non-critical test record). When a
 * persisted write exceeds the browser quota the write is ABANDONED — the
 * stored envelope resets to an empty history so the key stays bounded, the
 * in-memory conversation survives for the session, and a one-time toast
 * warns the user — instead of zustand's default of silently failing on
 * every subsequent write forever.
 *
 * Isolation: globalThis.localStorage is swapped for a quota-simulating stub
 * and RESTORED in afterAll; the store is a FRESH factory instance created
 * after the stub is installed (its storage adapter reads localStorage
 * dynamically either way).
 */

const QUOTA_LIMIT = 500
const TEST_KEY = 'lightrag::quota-test-history'

const data = new Map<string, string>()
const stub = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => {
    if (value.length > QUOTA_LIMIT) {
      const error = new Error('quota exceeded (simulated)')
      error.name = 'QuotaExceededError'
      throw error
    }
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
let store: import('./retrievalHistory').RetrievalHistoryStore

beforeAll(async () => {
  previousDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', {
    value: stub,
    configurable: true
  })
  if (typeof sessionStorage === 'undefined') {
    Object.defineProperty(globalThis, 'sessionStorage', {
      value: { getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {} },
      configurable: true
    })
  }
  const { createRetrievalHistoryStore } = await import('./retrievalHistory')
  store = createRetrievalHistoryStore(TEST_KEY)
})

afterAll(() => {
  if (previousDescriptor) {
    Object.defineProperty(globalThis, 'localStorage', previousDescriptor)
  } else {
    delete (globalThis as Record<string, unknown>).localStorage
  }
})

describe('bounded history persistence under quota pressure', () => {
  test('an over-quota write abandons persistence but keeps the live conversation', () => {
    const bigHistory = [{ id: 'b1', role: 'user' as const, content: 'x'.repeat(2000) }]

    store.getState().setHistory(bigHistory)

    // The live session still shows the conversation…
    expect(store.getState().history).toHaveLength(1)
    // …while the persisted envelope was reset to an EMPTY history (bounded),
    // not left holding a stale value and not thrown out of setHistory.
    const persisted = JSON.parse(stub.getItem(TEST_KEY)!)
    expect(persisted.version).toBe(RETRIEVAL_HISTORY_STORE_VERSION)
    expect(persisted.state.history).toEqual([])
  })

  test('small writes persist normally again afterwards', () => {
    const smallHistory = [{ id: 's1', role: 'user' as const, content: 'short' }]

    store.getState().setHistory(smallHistory)

    expect(JSON.parse(stub.getItem(TEST_KEY)!).state.history).toHaveLength(1)
  })
})
