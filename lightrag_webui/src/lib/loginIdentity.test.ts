import { afterAll, afterEach, beforeAll, describe, expect, test } from 'bun:test'
import { readdirSync, readFileSync } from 'fs'
import { join } from 'path'
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

/** Every non-test source file under `src/`, for the tree-wide prohibition below. */
const sourceFiles = (dir: string): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) return sourceFiles(path)
    if (!entry.isFile()) return []
    if (!/\.(ts|tsx)$/.test(path) || /\.test\.tsx?$/.test(path)) return []
    return [path]
  })

let previousDescriptor: PropertyDescriptor | undefined

let applyLoginIdentity: typeof import('./loginIdentity').applyLoginIdentity
let activateLoginIdentityFromToken: typeof import('./loginIdentity').activateLoginIdentityFromToken
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
  activateLoginIdentityFromToken = identityModule.activateLoginIdentityFromToken
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

  test('a marker written with the OLD mangled spelling is the same person', () => {
    // Correcting the JWT payload decoding (latin-1 -> UTF-8) changes what a
    // non-ASCII identity string looks like: `كظدج` used to read as
    // `ÙØ¸Ø¯Ø¬`, and that mangled form is what the previous build persisted.
    // Treating the two as different would clear both histories once, on the
    // upgrade load, for exactly the users the decoding fix is for.
    const identity = 'كظدج'
    // Derived, not typed: the mangled spelling contains bytes that do not
    // survive being written as a source literal. This is literally what the
    // old decoder returned — atob over the payload's base64.
    const mangled = atob(Buffer.from(identity, 'utf8').toString('base64'))
    expect(mangled).not.toBe(identity)
    seedHistories()
    stub.setItem(PREVIOUS_USER_KEY, mangled)

    const changed = applyLoginIdentity(identity)

    expect(changed).toBe(false)
    expect(useWebuiRetrievalHistoryStore.getState().history).toHaveLength(1)
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toHaveLength(1)
    // …and the marker self-heals to the corrected spelling on that same
    // activation, so the comparison is needed exactly once.
    expect(stub.getItem(PREVIOUS_USER_KEY)).toBe(identity)
  })

  test('a DIFFERENT non-ASCII user still clears', () => {
    // The equivalence above must not collapse distinct names: it maps one
    // spelling of ONE identity, not any pair of non-ASCII strings.
    seedHistories()
    stub.setItem(
      PREVIOUS_USER_KEY,
      atob(Buffer.from('كظدج', 'utf8').toString('base64'))
    )

    const changed = applyLoginIdentity('محمد')

    expect(changed).toBe(true)
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toEqual([])
    expect(stub.getItem(PREVIOUS_USER_KEY)).toBe('محمد')
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
  test('reads a base64url payload, instead of silently reporting a guest', () => {
    // Regression: bare `atob` rejects the `-`/`_` alphabet a JWT is entitled
    // to use, and this function's failure mode is SILENT — it returned the
    // literal 'guest' for a named user, and applyLoginIdentity would then
    // have preserved that user's conversations for whoever came next.
    const payload = JSON.stringify({ sub: 'alice?~', role: 'user' })
    const base64url = btoa(payload)
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '')
    // The vector is only meaningful if it actually needs the url alphabet.
    expect(base64url).toMatch(/[-_]/)
    expect(() => atob(base64url)).toThrow()

    expect(loginIdentityFromToken(`x.${base64url}.y`)).toBe('alice?~')
  })

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

describe('activateLoginIdentityFromToken (the login form\'s only entry point)', () => {
  test('a guest token after a named user CLEARS, even when the typed name matched', () => {
    // Regression (P1). The submit handler used to record the TYPED username.
    // If the operator disables authentication while the form is open, /login
    // answers with a guest token whatever was submitted — and a user typing
    // the identity the browser last held made `isIdentityChange` see no
    // transition, so alice's conversations survived into a session that is
    // really a guest's (and, in bypass mode, would be resent to the LLM).
    seedHistories()
    stub.setItem(PREVIOUS_USER_KEY, 'alice')

    // What the form knows: the typed name. What the server granted: guest.
    const typedUsername = 'alice'
    expect(applyLoginIdentity(typedUsername)).toBe(false) // the old decision…

    seedHistories()
    stub.setItem(PREVIOUS_USER_KEY, 'alice')
    expect(activateLoginIdentityFromToken(guestToken)).toBe(true) // …the right one
    expect(useWebuiRetrievalHistoryStore.getState().history).toEqual([])
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toEqual([])
    expect(stub.getItem(PREVIOUS_USER_KEY)).toBe('guest')
  })

  test('an ordinary named login still records the granted identity', () => {
    seedHistories()
    stub.setItem(PREVIOUS_USER_KEY, 'alice')
    expect(activateLoginIdentityFromToken(makeToken({ sub: 'alice' }))).toBe(false)
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toHaveLength(1)

    expect(activateLoginIdentityFromToken(makeToken({ sub: 'bob' }))).toBe(true)
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toEqual([])
    expect(stub.getItem(PREVIOUS_USER_KEY)).toBe('bob')
  })

  test('no activation path decides the identity from anything but its token', () => {
    // Structural pin, and deliberately not a rendered one: `applyLoginIdentity`
    // takes a bare name, so any caller COULD pass a typed username again and
    // every behavioral test above would still pass whenever the typed name and
    // the token's `sub` happen to agree — which is most of the time, and never
    // in the cases the cleanup exists for.
    const ACTIVATION_SITES = [
      'features/LoginPage.tsx',
      'features/workspace/WorkspaceWelcome.tsx',
      'features/workspace/authBootstrap.ts',
      'App.tsx'
    ]

    for (const file of ACTIVATION_SITES) {
      const source = readFileSync(join(import.meta.dir, '..', file), 'utf8')
      expect({ file, activates: source.includes('activateLoginIdentityFromToken(') }).toEqual(
        { file, activates: true }
      )
    }

    // The prohibition is scanned across the WHOLE tree rather than over the
    // four known sites: a fifth activation path added tomorrow is the case
    // this test exists for, and a fixed list cannot see it. `loginIdentity.ts`
    // is the one legitimate caller — it is where the token-derived entry point
    // delegates.
    const callers = sourceFiles(join(import.meta.dir, '..'))
      .filter((file) => !file.endsWith('/lib/loginIdentity.ts'))
      .filter((file) => /\bapplyLoginIdentity\(/.test(readFileSync(file, 'utf8')))
      .map((file) => file.slice(join(import.meta.dir, '..').length + 1))

    expect(callers).toEqual([])
  })
})
