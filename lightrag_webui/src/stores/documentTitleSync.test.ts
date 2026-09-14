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

// Cast through `unknown`: the DOM lib types `globalThis.document`/`window` as
// required and fully-featured, which neither a stub nor `delete` can satisfy.
const g = globalThis as unknown as {
  document?: { title: string }
  window?: { __LIGHTRAG_CONFIG__?: { webuiTitle?: string | null } }
}

/**
 * Run `body` as if the page had been served with `WEBUI_TITLE=injected`.
 *
 * `window` is installed only for the duration of the call, then the previous
 * value is PUT BACK rather than deleted. Since the DOM preload
 * (`src/test/happydom.ts`) installs a real `window` for the whole process,
 * deleting it here would strip the DOM from every test file Bun runs after
 * this one — a failure that would surface in an unrelated component test.
 */
const withInjectedTitle = (injected: string, body: () => void): void => {
  const previous = g.window
  g.window = { __LIGHTRAG_CONFIG__: { webuiTitle: injected } }
  try {
    body()
  } finally {
    g.window = previous
  }
}

let stateModule: StateModule
let realApiModule: Record<string, unknown>
let realDocument: typeof g.document

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
  // not another test file imported the store first. The real (preloaded) one
  // is kept so `afterAll` can restore it for later files.
  realDocument = g.document
  g.document = { title: 'LightRAG' }
  stateModule = await import('./state')
})

afterAll(() => {
  mock.module('@/api/lightrag', () => realApiModule)
  g.document = realDocument
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

  test('a cleared title is not overridden by the injected page-load value', () => {
    // Regression (Codex review on PR #3763): the page was served while the
    // deployment had a title, so that title is frozen into the runtime config
    // in <head>. The server then restarts with WEBUI_TITLE unset and /health
    // reports null. The store is authoritative — the tab must revert to the
    // default like the header does, not resurrect the injected snapshot and
    // stay wrong until a reload.
    const auth = () => stateModule.useAuthStore.getState()

    withInjectedTitle('Old KB', () => {
      auth().setCustomTitle('Old KB', null)
      expect(g.document!.title).toBe('Old KB')

      auth().setCustomTitle(null, null)
      expect(g.document!.title).toBe('LightRAG')
    })
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
