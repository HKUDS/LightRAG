import { afterAll, beforeAll, describe, expect, mock, test } from 'bun:test'

type StateModule = typeof import('./state')

/**
 * The auth store keeps the browser tab title on the deployment's WEBUI_TITLE.
 *
 * The server stamps the title into the entry HTML for the first paint, but
 * that value is frozen at page load: `/health` (via `setCustomTitle`) and the
 * login response are what carry a restarted server's new title into an open
 * tab, and under `bun run dev` they are the ONLY source. This pins that the
 * store actually drives `document.title` — the subscription is registered at
 * module import, so a regression here is silent in every other test.
 */

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

// Cast through `unknown`: the DOM lib types `globalThis.document` as required
// and fully-featured, which neither a stub nor `delete` can satisfy.
const g = globalThis as unknown as { document?: { title: string } }

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
  // The store must never reach the network; only the title paths are exercised.
  realApiModule = { ...(await import('@/api/lightrag')) }
  mock.module('@/api/lightrag', () => ({
    ...realApiModule,
    checkHealth: mock(async () => ({ status: 'error', message: 'offline' }))
  }))

  // `document` is read at call time, so installing it here works whether or
  // not another test file imported the store first.
  g.document = { title: 'LightRAG' }
  stateModule = await import('./state')
})

afterAll(() => {
  mock.module('@/api/lightrag', () => realApiModule)
  delete g.document
})

const b64url = (value: string) =>
  Buffer.from(value, 'utf8')
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '')

const jwt = (payload: unknown) =>
  `${b64url('{"alg":"HS256"}')}.${b64url(JSON.stringify(payload))}.signature`

const token = jwt({ sub: 'alice', exp: Math.floor(Date.now() / 1000) + 3600 })

describe('tab title follows the auth store', () => {
  test('health-driven title updates, and reverts to the default when unset', () => {
    const auth = () => stateModule.useAuthStore.getState()

    auth().setCustomTitle('My Graph KB', 'Simple and Fast Graph Based RAG System')
    expect(g.document!.title).toBe('My Graph KB')

    // A server restarted with a different WEBUI_TITLE reaches an open tab on
    // the next poll.
    auth().setCustomTitle('Renamed KB', null)
    expect(g.document!.title).toBe('Renamed KB')

    // Cleared on the server: back to the generic name, not the stale one.
    auth().setCustomTitle(null, null)
    expect(g.document!.title).toBe('LightRAG')
  })

  test('login applies the title, and logout keeps it', () => {
    const auth = () => stateModule.useAuthStore.getState()

    auth().login(token, false, '1.0.0', '1.0.0', 'My Graph KB', null)
    expect(g.document!.title).toBe('My Graph KB')

    // The title is deployment-level, not per-session: the login page belongs
    // to the same deployment, so signing out must not rename the tab.
    auth().logout()
    expect(g.document!.title).toBe('My Graph KB')
  })
})
