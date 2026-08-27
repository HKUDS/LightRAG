import { afterAll, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'

/**
 * Local token validity (client-state partitioning doc §5). These cases pin
 * the REPLACEMENT of the old "token exists ⇒ authenticated" rule — an
 * implementation that only checks existence fails five of the six cases.
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

describe('isTokenLocallyValid', () => {
  test('malformed: not three segments', () => {
    expect(stateModule.isTokenLocallyValid('justonestring')).toBe(false)
    expect(stateModule.isTokenLocallyValid('a.b')).toBe(false)
    expect(stateModule.isTokenLocallyValid('a.b.c.d')).toBe(false)
  })

  test('payload is not valid Base64URL', () => {
    expect(stateModule.isTokenLocallyValid('head.@@@@@.sig')).toBe(false)
  })

  test('payload decodes but is not JSON', () => {
    expect(stateModule.isTokenLocallyValid(`head.${b64url('not json at all')}.sig`)).toBe(false)
  })

  test('missing exp', () => {
    expect(stateModule.isTokenLocallyValid(jwt({ sub: 'alice', role: 'user' }))).toBe(false)
  })

  test('expired exp', () => {
    expect(stateModule.isTokenLocallyValid(jwt({ sub: 'alice', exp: pastExp }))).toBe(false)
  })

  test('valid and unexpired — the ONLY case counted as logged in', () => {
    expect(stateModule.isTokenLocallyValid(jwt({ sub: 'alice', exp: futureExp }))).toBe(true)
  })

  test('base64url payloads (with - and _) parse correctly', () => {
    // A payload whose base64 contains '+'/'/' in standard encoding.
    const payload = { sub: '???>>>???>>>', exp: futureExp }
    expect(stateModule.isTokenLocallyValid(jwt(payload))).toBe(true)
  })
})

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

  test('a locally-valid guest token counts as a valid login (welcome page is skipped by it)', () => {
    seed(jwt({ sub: 'guest', role: 'guest', exp: futureExp }))
    const state = stateModule.initAuthState()
    expect(state.isAuthenticated).toBe(true)
    expect(state.isGuestMode).toBe(true)
  })
})
