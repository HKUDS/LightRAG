import { afterAll, afterEach, beforeAll, describe, expect, test } from 'bun:test'
import type { AuthStatusResponse } from '@/api/lightrag'
import { WORKSPACE_RETRIEVAL_HISTORY_KEY } from '@/lib/storageKeys'

/**
 * Workspace boot reconciliation.
 *
 * The P1 case being pinned: a browser holding an UNEXPIRED named-user token
 * meets a server restarted with authentication DISABLED. Local token
 * validation admits the session (so the welcome page never runs), and the
 * old behaviour re-activated that stale named token with the guest flag —
 * leaving the named user's name, its workspace history and its Authorization
 * header on what is really a guest session, until the first 401 silently
 * swapped in a guest token behind the retry. The fresh `access_token` the
 * status response carries must be activated instead, through the shared
 * identity cleanup.
 *
 * Isolation: the REAL store singletons are used; globalThis.localStorage is
 * swapped for a stub and RESTORED in afterAll. The module under test (and
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

/** A JWT-shaped token: only the payload is ever read by the client. */
const tokenFor = (sub: string, role: string): string => {
  const payload = btoa(
    JSON.stringify({ sub, role, exp: Math.floor(Date.now() / 1000) + 3600 })
  )
  return `header.${payload}.signature`
}

const NAMED_TOKEN = tokenFor('alice', 'user')
const FRESH_GUEST_TOKEN = tokenFor('guest', 'guest')

let previousDescriptor: PropertyDescriptor | undefined

let activateSessionFromAuthStatus: typeof import('./authBootstrap').activateSessionFromAuthStatus
let useAuthStore: typeof import('@/stores/state').useAuthStore
let useWorkspaceRetrievalHistoryStore: typeof import('@/stores/workspaceRetrievalHistory').useWorkspaceRetrievalHistoryStore

beforeAll(async () => {
  previousDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', { value: stub, configurable: true })

  activateSessionFromAuthStatus = (await import('./authBootstrap'))
    .activateSessionFromAuthStatus
  useAuthStore = (await import('@/stores/state')).useAuthStore
  useWorkspaceRetrievalHistoryStore = (
    await import('@/stores/workspaceRetrievalHistory')
  ).useWorkspaceRetrievalHistoryStore
})

afterAll(() => {
  if (previousDescriptor) {
    Object.defineProperty(globalThis, 'localStorage', previousDescriptor)
  } else {
    // @ts-expect-error - removing the stub installed for this file
    delete globalThis.localStorage
  }
})

afterEach(() => {
  data.clear()
  useWorkspaceRetrievalHistoryStore.getState().clearHistory()
})

/** The state a browser is in after a named user used it. */
const seedNamedUserSession = () => {
  data.set(TOKEN_STORAGE_KEY, NAMED_TOKEN)
  data.set(PREVIOUS_USER_KEY, 'alice')
  data.set(
    WORKSPACE_RETRIEVAL_HISTORY_KEY,
    JSON.stringify({
      state: { history: [{ id: 'alice-chat', role: 'user', content: 'private' }] },
      version: 0
    })
  )
  useAuthStore.setState({ isAuthenticated: true, isGuestMode: false, username: 'alice' })
}

const authDisabledStatus: AuthStatusResponse = {
  auth_configured: false,
  access_token: FRESH_GUEST_TOKEN,
  auth_mode: 'disabled',
  core_version: '1.0.0',
  api_version: '1.0.0'
}

describe('activateSessionFromAuthStatus', () => {
  test('an auth-disabled server REPLACES a stale named token with the fresh guest one', () => {
    seedNamedUserSession()

    activateSessionFromAuthStatus(authDisabledStatus, NAMED_TOKEN)

    // The token actually sent on the wire is the guest one…
    expect(data.get(TOKEN_STORAGE_KEY)).toBe(FRESH_GUEST_TOKEN)
    const state = useAuthStore.getState()
    expect(state.isGuestMode).toBe(true)
    // …and the session no longer claims to be alice.
    expect(state.username).toBe('guest')
  })

  test('…and runs the identity cleanup, so alice\'s history is not shown to the guest', () => {
    seedNamedUserSession()
    useWorkspaceRetrievalHistoryStore
      .getState()
      .setHistory([{ id: 'alice-chat', role: 'user', content: 'private' } as never])

    activateSessionFromAuthStatus(authDisabledStatus, NAMED_TOKEN)

    expect(data.get(PREVIOUS_USER_KEY)).toBe('guest')
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toHaveLength(0)
    // The persisted envelope is removed outright and only ever rewritten by
    // the store's own clear — nothing of alice's survives for a later
    // hydration to pick up.
    const persisted = data.get(WORKSPACE_RETRIEVAL_HISTORY_KEY)
    expect(persisted == null || JSON.parse(persisted).state.history.length === 0).toBe(true)
    expect(persisted ?? '').not.toContain('alice-chat')
  })

  test('a guest reload keeps its own history (same identity, no cleanup)', () => {
    data.set(TOKEN_STORAGE_KEY, tokenFor('guest', 'guest'))
    data.set(PREVIOUS_USER_KEY, 'guest')
    useAuthStore.setState({ isAuthenticated: true, isGuestMode: true, username: 'guest' })
    useWorkspaceRetrievalHistoryStore
      .getState()
      .setHistory([{ id: 'guest-chat', role: 'user', content: 'kept' } as never])

    activateSessionFromAuthStatus(authDisabledStatus, data.get(TOKEN_STORAGE_KEY)!)

    expect(data.get(TOKEN_STORAGE_KEY)).toBe(FRESH_GUEST_TOKEN)
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toHaveLength(1)
  })

  test('an auth-ENABLED server keeps the stored named token and its identity', () => {
    seedNamedUserSession()

    activateSessionFromAuthStatus(
      {
        auth_configured: true,
        auth_mode: 'enabled',
        core_version: '1.0.0',
        api_version: '1.0.0'
      },
      NAMED_TOKEN
    )

    expect(data.get(TOKEN_STORAGE_KEY)).toBe(NAMED_TOKEN)
    expect(data.get(PREVIOUS_USER_KEY)).toBe('alice')
    expect(data.get(WORKSPACE_RETRIEVAL_HISTORY_KEY)).toBeDefined()
    const state = useAuthStore.getState()
    expect(state.username).toBe('alice')
    expect(state.isGuestMode).toBe(false)
  })

  test('a status response with nothing to apply leaves the session untouched', () => {
    seedNamedUserSession()

    activateSessionFromAuthStatus({ auth_configured: true }, NAMED_TOKEN)

    expect(data.get(TOKEN_STORAGE_KEY)).toBe(NAMED_TOKEN)
    expect(useAuthStore.getState().username).toBe('alice')
  })
})
