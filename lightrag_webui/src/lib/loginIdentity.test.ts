import { afterAll, afterEach, beforeAll, describe, expect, test } from 'bun:test'
import {
  applyLoginIdentity,
  handleIdentityStorageEvent,
  loginIdentityFromToken,
  useIdentityEpochStore
} from './loginIdentity'
import { useWebuiRetrievalHistoryStore } from '@/stores/webuiRetrievalHistory'
import { useWorkspaceRetrievalHistoryStore } from '@/stores/workspaceRetrievalHistory'
import { WORKSPACE_RETRIEVAL_HISTORY_KEY } from '@/lib/storageKeys'

/**
 * Identity-change cleanup contract: logout/401 preserve the histories, so
 * EVERY activation path (login form, auto-guest, welcome-page guest
 * activation) must clear both entries' histories when the incoming identity
 * differs from the stored one — the P1 case being a guest activation on a
 * browser last used by a named user.
 *
 * Isolation: the REAL store singletons are driven via setState and reset in
 * afterEach; globalThis.localStorage is swapped for a stub and RESTORED in
 * afterAll (the helper reads it dynamically at call time).
 */

const PREVIOUS_USER_KEY = 'LIGHTRAG-PREVIOUS-USER'

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

afterEach(() => {
  data.clear()
  useWebuiRetrievalHistoryStore.setState({ history: [] })
  useWorkspaceRetrievalHistoryStore.setState({ history: [] })
})

const seedHistories = () => {
  useWebuiRetrievalHistoryStore
    .getState()
    .setHistory([{ id: 'a1', role: 'user', content: 'admin question' }])
  useWorkspaceRetrievalHistoryStore
    .getState()
    .setHistory([{ id: 'w1', role: 'user', content: 'workspace question' }])
}

// A syntactically valid JWT whose payload is {"sub":"guest","role":"guest"}.
const guestToken = `x.${btoa(JSON.stringify({ sub: 'guest', role: 'guest' }))}.y`

describe('applyLoginIdentity', () => {
  test('a DIFFERENT stored identity clears BOTH histories and records the new one', () => {
    seedHistories()
    stub.setItem(PREVIOUS_USER_KEY, 'alice')

    const changed = applyLoginIdentity('guest')

    expect(changed).toBe(true)
    expect(useWebuiRetrievalHistoryStore.getState().history).toEqual([])
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toEqual([])
    expect(stub.getItem(PREVIOUS_USER_KEY)).toBe('guest')
  })

  test('the SAME identity preserves both histories', () => {
    seedHistories()
    stub.setItem(PREVIOUS_USER_KEY, 'guest')

    const changed = applyLoginIdentity('guest')

    expect(changed).toBe(false)
    expect(useWebuiRetrievalHistoryStore.getState().history).toHaveLength(1)
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toHaveLength(1)
    expect(stub.getItem(PREVIOUS_USER_KEY)).toBe('guest')
  })

  test('a MISSING stored identity counts as different (anonymous use before first login)', () => {
    seedHistories()

    const changed = applyLoginIdentity('alice')

    expect(changed).toBe(true)
    expect(useWebuiRetrievalHistoryStore.getState().history).toEqual([])
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toEqual([])
    expect(stub.getItem(PREVIOUS_USER_KEY)).toBe('alice')
  })

  test('a NEWER client\'s history envelope is destroyed too, not diverted around', () => {
    // Rollback interaction: a v2 (future) envelope holds the previous
    // identity's conversations. The future-envelope guard would DIVERT the
    // stores' clear-writes and preserve it — while the identity commit below
    // would make the newer client skip its own cleanup later. The identity
    // transition therefore removes the persisted keys outright.
    const futureEnvelope = JSON.stringify({
      state: { history: [{ id: 'a9', role: 'user', content: 'alice secret' }] },
      version: 2
    })
    stub.setItem(WORKSPACE_RETRIEVAL_HISTORY_KEY, futureEnvelope)
    stub.setItem(PREVIOUS_USER_KEY, 'alice')

    applyLoginIdentity('bob')

    const remaining = stub.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)
    // Either fully removed, or re-created EMPTY at this build's version by
    // the store's own clear-write — never the newer envelope's content.
    if (remaining !== null) {
      const parsed = JSON.parse(remaining)
      expect(parsed.version).toBe(1)
      expect(parsed.state.history).toEqual([])
    }
    expect(stub.getItem(PREVIOUS_USER_KEY)).toBe('bob')
  })

  test('the P1 sequence end-to-end: named user → logout preserves → guest activation clears', () => {
    // Named user's session left history behind; navigation stored the name.
    seedHistories()
    stub.setItem(PREVIOUS_USER_KEY, 'alice')
    // Welcome-page guest activation derives its identity from the token.
    applyLoginIdentity(loginIdentityFromToken(guestToken))
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toEqual([])
    // A second guest activation (token refresh cycle) then PRESERVES.
    seedHistories()
    const changed = applyLoginIdentity(loginIdentityFromToken(guestToken))
    expect(changed).toBe(false)
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toHaveLength(1)
  })
})

describe('handleIdentityStorageEvent (cross-tab identity change)', () => {
  test('a DIFFERENT identity recorded by another tab clears LIVE histories and bumps the epoch', () => {
    seedHistories()
    const before = useIdentityEpochStore.getState().epoch

    handleIdentityStorageEvent({
      key: PREVIOUS_USER_KEY,
      oldValue: 'alice',
      newValue: 'guest'
    })

    expect(useWebuiRetrievalHistoryStore.getState().history).toEqual([])
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toEqual([])
    // The epoch bump is what remounts mounted query sessions (their message
    // state was copied from the store at initialization).
    expect(useIdentityEpochStore.getState().epoch).toBe(before + 1)
  })

  test('other keys, unchanged values and key removals are ignored', () => {
    seedHistories()
    const before = useIdentityEpochStore.getState().epoch

    handleIdentityStorageEvent({ key: 'some-other-key', oldValue: 'a', newValue: 'b' })
    handleIdentityStorageEvent({
      key: PREVIOUS_USER_KEY,
      oldValue: 'guest',
      newValue: 'guest'
    })
    handleIdentityStorageEvent({ key: PREVIOUS_USER_KEY, oldValue: 'guest', newValue: null })

    expect(useWebuiRetrievalHistoryStore.getState().history).toHaveLength(1)
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toHaveLength(1)
    expect(useIdentityEpochStore.getState().epoch).toBe(before)
  })
})

describe('loginIdentityFromToken', () => {
  test('reads the JWT sub', () => {
    expect(loginIdentityFromToken(guestToken)).toBe('guest')
    expect(
      loginIdentityFromToken(`x.${btoa(JSON.stringify({ sub: 'alice' }))}.y`)
    ).toBe('alice')
  })

  test('unreadable tokens fall back to "guest"', () => {
    expect(loginIdentityFromToken('not-a-jwt')).toBe('guest')
    expect(loginIdentityFromToken('a.!!!.c')).toBe('guest')
    expect(loginIdentityFromToken(`x.${btoa('{}')}.y`)).toBe('guest')
  })
})
