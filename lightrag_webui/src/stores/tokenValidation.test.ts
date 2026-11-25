import { afterAll, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'

/**
 * Local token validity (client-state partitioning doc §5). These cases pin
 * the REPLACEMENT of the old "token exists ⇒ authenticated" rule — an
 * implementation that only checks existence fails every rejection case. They
 * go through `initAuthState`, the one caller of `isTokenLocallyValid`, so each
 * also pins what a rejected token leaves behind in storage.
 */

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

let stateModule: StateModule
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
  mock.module('@/api/lightrag', () => ({
    ...realApiModule,
    checkHealth: mock(async () => ({ status: 'error', message: 'offline' }))
  }))
  stateModule = await import('./state')
})

afterAll(() => {
  mock.module('@/api/lightrag', () => realApiModule)
})

const b64url = (value: string) =>
  Buffer.from(value, 'utf8').toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')

const jwt = (payload: unknown) =>
  `${b64url('{"alg":"HS256"}')}.${b64url(JSON.stringify(payload))}.signature`

const futureExp = Math.floor(Date.now() / 1000) + 3600
const pastExp = Math.floor(Date.now() / 1000) - 3600

describe('initAuthState local validation', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  const seed = (token: string) => {
    localStorage.setItem('LIGHTRAG-API-TOKEN', token)
    localStorage.setItem('LIGHTRAG-LAST-TOKEN-RENEWAL', '2026-01-01 00:00:00 (UTC+0)')
  }

  test.each([
    ['not three segments', 'broken-token'],
    ['payload not Base64URL', 'head.@@@@@.sig'],
    ['payload not JSON', `head.${b64url('nope')}.sig`],
    ['missing exp', jwt({ sub: 'alice' })],
    ['expired exp', jwt({ sub: 'alice', exp: pastExp })]
  ])('%s ⇒ unauthenticated AND token + companion keys cleared', (_label, token) => {
    seed(token)
    const state = stateModule.initAuthState()
    expect(state.isAuthenticated).toBe(false)
    expect(state.lastTokenRenewal).toBeNull()
    expect(localStorage.getItem('LIGHTRAG-API-TOKEN')).toBeNull()
    expect(localStorage.getItem('LIGHTRAG-LAST-TOKEN-RENEWAL')).toBeNull()
  })

  test('valid unexpired token ⇒ authenticated, nothing cleared', () => {
    seed(jwt({ sub: 'alice', role: 'user', exp: futureExp }))
    const state = stateModule.initAuthState()
    expect(state.isAuthenticated).toBe(true)
    expect(state.username).toBe('alice')
    expect(localStorage.getItem('LIGHTRAG-API-TOKEN')).not.toBeNull()
    expect(localStorage.getItem('LIGHTRAG-LAST-TOKEN-RENEWAL')).not.toBeNull()
  })

  test('a base64url payload yields the CLAIMS, not a blank authenticated session', () => {
    // Regression: validation used the url-safe decoder while the claims
    // parser used bare `atob`, so the two disagreed about the SAME token —
    // it was admitted, then read as `{}`. The session then ran with no
    // username, and `navigationService` records LIGHTRAG-PREVIOUS-USER only
    // `if (currentUsername)`: that user's logout wrote no identity marker at
    // all, so the retrieval-history cleanup never fired for them and their
    // conversations were left to whoever logged in next.
    //
    // Whether a payload needs the url alphabet depends on its exact bytes,
    // `exp` included, so it varies per issuance for the SAME user — which is
    // what made this intermittent. Hence the fixed vector below.
    //
    // The sub is ASCII on purpose: `atob` yields one character per BYTE, so
    // a non-ASCII sub comes back mojibake even once it decodes. That is a
    // separate defect from this one and is not what this test pins.
    const token = jwt({ sub: '~~h', role: 'user', exp: 9999999999 })
    const segment = token.split('.')[1]
    // Only meaningful if the payload really needs the url alphabet.
    expect(segment).toMatch(/[-_]/)
    expect(() => atob(segment)).toThrow()

    seed(token)
    const state = stateModule.initAuthState()

    expect(state.isAuthenticated).toBe(true)
    expect(state.username).toBe('~~h')
    expect(state.isGuestMode).toBe(false)
    expect(state.tokenExpiresAt).toBe(9999999999 * 1000)
  })

  test('a locally-valid guest token counts as a valid login (welcome page is skipped by it)', () => {
    seed(jwt({ sub: 'guest', role: 'guest', exp: futureExp }))
    const state = stateModule.initAuthState()
    expect(state.isAuthenticated).toBe(true)
    expect(state.isGuestMode).toBe(true)
  })
})
