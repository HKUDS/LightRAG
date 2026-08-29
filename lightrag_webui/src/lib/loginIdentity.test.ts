import { afterAll, afterEach, beforeAll, describe, expect, test } from 'bun:test'
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
 * afterAll. The module under test (and the stores it pulls in — including
 * stores/state, which derives auth state from localStorage AT IMPORT) is
 * imported dynamically AFTER the stub is installed.
 */

const PREVIOUS_USER_KEY = 'LIGHTRAG-PREVIOUS-USER'
const TOKEN_STORAGE_KEY = 'LIGHTRAG-API-TOKEN'

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

let applyLoginIdentity: typeof import('./loginIdentity').applyLoginIdentity
let handleIdentityStorageEvent: typeof import('./loginIdentity').handleIdentityStorageEvent
let loginIdentityFromToken: typeof import('./loginIdentity').loginIdentityFromToken
let useIdentityEpochStore: typeof import('./loginIdentity').useIdentityEpochStore
let useWebuiRetrievalHistoryStore: typeof import('@/stores/webuiRetrievalHistory').useWebuiRetrievalHistoryStore
let useWorkspaceRetrievalHistoryStore: typeof import('@/stores/workspaceRetrievalHistory').useWorkspaceRetrievalHistoryStore
let useAuthStore: typeof import('@/stores/state').useAuthStore

beforeAll(async () => {
  previousDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', {
    value: stub,
    configurable: true
  })
  if (typeof sessionStorage === 'undefined') {
    Object.defineProperty(globalThis, 'sessionStorage', {
      value: {
        getItem: () => null,
        setItem: () => {},
        removeItem: () => {},
        clear: () => {}
      },
      configurable: true
    })
  }

  const identityModule = await import('./loginIdentity')
  applyLoginIdentity = identityModule.applyLoginIdentity
  handleIdentityStorageEvent = identityModule.handleIdentityStorageEvent
  loginIdentityFromToken = identityModule.loginIdentityFromToken
  useIdentityEpochStore = identityModule.useIdentityEpochStore
  useWebuiRetrievalHistoryStore = (
    await import('@/stores/webuiRetrievalHistory')
  ).useWebuiRetrievalHistoryStore
  useWorkspaceRetrievalHistoryStore = (
    await import('@/stores/workspaceRetrievalHistory')
  ).useWorkspaceRetrievalHistoryStore
  useAuthStore = (await import('@/stores/state')).useAuthStore
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
  useAuthStore.setState({ isAuthenticated: false, isGuestMode: false, username: null })
})

const seedHistories = () => {
  useWebuiRetrievalHistoryStore
    .getState()
    .setHistory([{ id: 'a1', role: 'user', content: 'admin question' }])
  useWorkspaceRetrievalHistoryStore
    .getState()
    .setHistory([{ id: 'w1', role: 'user', content: 'workspace question' }])
}

const makeToken = (payload: Record<string, unknown>) =>
  `x.${btoa(JSON.stringify(payload))}.y`

// A syntactically valid JWT whose payload is {"sub":"guest","role":"guest"}.
const guestToken = makeToken({ sub: 'guest', role: 'guest' })

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

  test('a MISSING stored identity does NOT clear for an incoming GUEST', () => {
    // Regression: the marker is missing on exactly one load — the first
    // after an upgrade from a build that wrote it only on a login-form
    // submit or a logout. In an auth-DISABLED deployment neither ever runs
    // (guests bypass the form and have no logout button), so every such
    // browser arrives here with a null marker and a guest identity. Treating
    // that as a change destroyed both histories on the single load where the
    // storage split had just finished migrating them.
    //
    // It is also unnecessary: on that older build a NAMED user could not
    // accumulate history without writing the marker, so a history sitting
    // beside a missing marker was written by guest use. The incoming guest
    // owns it. (The named-user case above still clears — that one IS a
    // transition.)
    seedHistories()

    const changed = applyLoginIdentity(loginIdentityFromToken(guestToken))

    expect(changed).toBe(false)
    expect(useWebuiRetrievalHistoryStore.getState().history).toHaveLength(1)
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toHaveLength(1)
    // The marker is still committed, so the NEXT named login clears normally.
    expect(stub.getItem(PREVIOUS_USER_KEY)).toBe('guest')
    expect(applyLoginIdentity('alice')).toBe(true)
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toEqual([])
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

  test('the marker\'s FIRST write by a GUEST tab is not a transition', () => {
    // Regression, and the cross-tab half of the same-tab rule above: the
    // marker's first write (oldValue null) is another tab recording what
    // this browser already was. In an auth-disabled deployment that is the
    // ONLY way it ever gets written, so the sequence is ordinary — /workspace
    // sits on the welcome page, its migration having just moved the legacy
    // history into place, while a second tab loads /webui and its guest
    // activation records the marker. Treating that as a transition cleared
    // the live stores here and persisted the empty result over the migrated
    // history.
    seedHistories()
    const before = useIdentityEpochStore.getState().epoch

    handleIdentityStorageEvent({
      key: PREVIOUS_USER_KEY,
      oldValue: null,
      newValue: 'guest'
    })

    expect(useWebuiRetrievalHistoryStore.getState().history).toHaveLength(1)
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toHaveLength(1)
    // No remount either: the live session belongs to that same guest.
    expect(useIdentityEpochStore.getState().epoch).toBe(before)
  })

  test('the marker\'s FIRST write by a NAMED tab IS a transition', () => {
    // The other side: guest use before the first login. A named identity
    // appearing beside a missing marker is a real change, so the guest's
    // conversations must not survive into it.
    seedHistories()
    const before = useIdentityEpochStore.getState().epoch

    handleIdentityStorageEvent({
      key: PREVIOUS_USER_KEY,
      oldValue: null,
      newValue: 'alice'
    })

    expect(useWebuiRetrievalHistoryStore.getState().history).toEqual([])
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toEqual([])
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

  test('a token written by another tab RESYNCS this tab\'s auth store', () => {
    // The token is the LAST thing a login writes (the identity marker the
    // first), so its event is the completion signal: this tab must stop
    // describing the previous identity while its requests already send the
    // new tab's token (the API layer reads localStorage per request).
    const exp = Math.floor(Date.now() / 1000) + 3600
    const bobToken = makeToken({ sub: 'bob', role: 'user', exp })
    stub.setItem(TOKEN_STORAGE_KEY, bobToken)

    handleIdentityStorageEvent({
      key: TOKEN_STORAGE_KEY,
      oldValue: null,
      newValue: bobToken
    })

    expect(useAuthStore.getState().isAuthenticated).toBe(true)
    expect(useAuthStore.getState().username).toBe('bob')
    expect(useAuthStore.getState().isGuestMode).toBe(false)
  })

  test('a token REMOVED by another tab (logout) de-authenticates this tab', () => {
    useAuthStore.setState({ isAuthenticated: true, username: 'alice' })
    stub.removeItem(TOKEN_STORAGE_KEY)

    handleIdentityStorageEvent({
      key: TOKEN_STORAGE_KEY,
      oldValue: 'old-token',
      newValue: null
    })

    expect(useAuthStore.getState().isAuthenticated).toBe(false)
    expect(useAuthStore.getState().username).toBeNull()
  })
})

describe('loginIdentityFromToken', () => {
  test('reads the JWT sub', () => {
    expect(loginIdentityFromToken(guestToken)).toBe('guest')
    expect(loginIdentityFromToken(makeToken({ sub: 'alice' }))).toBe('alice')
  })

  test('unreadable tokens fall back to "guest"', () => {
    expect(loginIdentityFromToken('not-a-jwt')).toBe('guest')
    expect(loginIdentityFromToken('a.!!!.c')).toBe('guest')
    expect(loginIdentityFromToken(`x.${btoa('{}')}.y`)).toBe('guest')
  })
})
