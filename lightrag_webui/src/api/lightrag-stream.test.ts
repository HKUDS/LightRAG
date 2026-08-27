import { afterAll, afterEach, beforeAll, describe, expect, mock, spyOn, test } from 'bun:test'
// Dependency-free module — safe to import statically before the mocks below.
import { isAuthenticationRequiredError } from './errors'

// ---------------------------------------------------------------------------
// Mock dependencies BEFORE importing the module under test
// ---------------------------------------------------------------------------

// Loaded BEFORE any mock.module call, because the real '@/lib/constants' pulls
// '@/lib/utils' and the constants overlay below must spread the real exports.
const realConstants = await import('@/lib/constants')

const storageData = new Map<string, string>()
const storageMock = {
  getItem: (key: string) => storageData.get(key) ?? null,
  setItem: (key: string, value: string) => { storageData.set(key, value) },
  removeItem: (key: string) => { storageData.delete(key) },
  clear: () => { storageData.clear() },
}

Object.defineProperty(globalThis, 'localStorage', {
  value: storageMock,
  configurable: true,
})
Object.defineProperty(globalThis, 'sessionStorage', {
  value: storageMock,
  configurable: true,
})

// NOTE on isolation: `mock.module` lives in the process-wide module registry
// for the REST of the bun test run — a module mock installed here poisons
// every later test file that imports the same module (we shipped exactly that
// bug twice: navigationService replaced by a no-op stub, stores bound to fake
// implementations). So shared singletons are NOT module-mocked here. Instead:
// - the real zustand stores are driven via setState (reset in afterEach);
// - navigationService.navigateToUnauthenticated is neutralized with a
//   RESTORABLE spy (mockRestore in afterAll) so the 401 paths under test
//   don't navigate, while later files still see the real service.
// The remaining mock.module targets are leaf-ish dependencies where an
// overlay is sufficient: '@/lib/constants' (spread-real, pins the base URL)
// and 'axios' (feeds the guest-token refresh; api/lightrag is imported
// dynamically AFTER this mock so its axios instance is the mocked one).
// An OVERLAY on the real module, not a replacement. `mock.module` is global for
// the whole `bun test` run and is never undone, so a factory returning only the
// three exports this file needs also deletes every OTHER export of that module
// for every test file that runs afterwards — `src/lib/fileTypes.test.ts` imports
// `supportedFileTypes` from here and failed with "Export named
// 'supportedFileTypes' not found" whenever it ran after this file (and passed on
// its own, which is what made it look like a product bug). Spreading the real
// module keeps the rest of the surface intact while still pinning the values
// these tests assert on.
mock.module('@/lib/constants', () => ({
  ...realConstants,
  backendBaseUrl: 'http://localhost:9621',
  popularLabelsDefaultLimit: 300,
  searchLabelsDefaultLimit: 50,
}))

// Mock axios — the module calls axios.create() at top level and
// axios.get() in silentRefreshGuestToken
mock.module('axios', () => {
  const instance = {
    get: () =>
      Promise.resolve({
        data: {
          access_token: 'mock-guest-token',
          auth_configured: false,
          core_version: '1.0',
          api_version: '1.0',
        },
        headers: {},
      }),
    post: () => Promise.resolve({ data: {}, headers: {} }),
    interceptors: {
      request: { use: () => {} },
      response: { use: () => {} },
    },
  }
  // The default export from axios is the main axios function, which also has
  // .create, .get, .post, etc. as static methods.
  const axiosFn: any = () => Promise.resolve({ data: {}, headers: {} })
  axiosFn.create = () => instance
  axiosFn.get = instance.get
  axiosFn.post = instance.post
  axiosFn.interceptors = instance.interceptors
  return { default: axiosFn, __esModule: true }
})

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal QueryRequest payload. */
const makeQueryRequest = (overrides = {}) => ({
  query: 'test query',
  mode: 'mix' as const,
  ...overrides,
})

/**
 * Create a ReadableStream from an array of Uint8Array chunks. Bun's Response
 * constructor accepts a ReadableStream directly.
 */
function makeNdjsonResponse(lines: string[], status = 200): Response {
  const encoder = new TextEncoder()
  const body = lines.map((l) => encoder.encode(l + '\n'))
  // Build a simple async iterable readable stream
  const stream = new ReadableStream({
    start(controller) {
      for (const chunk of body) {
        controller.enqueue(chunk)
      }
      controller.close()
    },
  })
  return new Response(stream, { status })
}

/** Build a non-streaming error response (text body). */
function makeTextResponse(body: string, status: number): Response {
  return new Response(body, { status })
}

/**
 * Install a mocked implementation onto globalThis.fetch. Bun's `mock()` returns
 * a `Mock` that lacks the DOM `fetch.preconnect` static, so the assignment is
 * cast through `unknown` to satisfy the `typeof fetch` type.
 */
function installFetchMock(
  impl: (url: string, init?: RequestInit) => Response | Promise<Response>
): void {
  globalThis.fetch = mock(impl) as unknown as typeof fetch
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

let apiModule: typeof import('./lightrag')
let settingsStore: typeof import('@/stores/settings').useSettingsStore
let authStore: typeof import('@/stores/state').useAuthStore

// Several suites below intentionally drive queryTextStream's failure branches
// (HTTP errors, network errors, truncated streams). By design the module logs
// those via console.error / console.warn, which floods the test output with
// red lines that look like failures but are not. The tests already assert the
// observable contract through the onError callback, so silence the logs for
// the whole file and restore the originals afterwards.
let restoreMocks: (() => void) | undefined

beforeAll(async () => {
  // Dynamic imports AFTER the axios/constants mocks above, so the api module's
  // axios instance is the mocked one. These are the REAL store and navigation
  // modules — see the isolation note at the top of this file.
  settingsStore = (await import('@/stores/settings')).useSettingsStore
  authStore = (await import('@/stores/state')).useAuthStore
  const { navigationService } = await import('@/services/navigation')
  apiModule = await import('./lightrag')

  // Restorable, not module-level: the 401 paths under test call this; a
  // module mock would leak a no-op service into every later test file.
  const navSpy = spyOn(navigationService, 'navigateToUnauthenticated')
    .mockImplementation(() => {})
  const errorSpy = spyOn(console, 'error').mockImplementation(() => {})
  const warnSpy = spyOn(console, 'warn').mockImplementation(() => {})
  restoreMocks = () => {
    navSpy.mockRestore()
    errorSpy.mockRestore()
    warnSpy.mockRestore()
  }
})

afterAll(() => {
  restoreMocks?.()
})

afterEach(() => {
  storageData.clear()
  // Reset the REAL stores' state this file drives (the guest-retry tests go
  // through the real login(), which flips isAuthenticated/username too).
  settingsStore.setState({ apiKey: null })
  authStore.setState({
    isGuestMode: false,
    isAuthenticated: false,
    username: null
  })
})

describe('queryTextStream — normal path', () => {
  test('parses NDJSON response chunks and calls onChunk', async () => {
    const chunks: string[] = []
    const errors: string[] = []

    installFetchMock(() =>
      makeNdjsonResponse([
        '{"response": "Hello"}',
        '{"response": " World"}',
        '{"response": "!"}',
      ])
    )

    await apiModule.queryTextStream(
      makeQueryRequest(),
      (c) => chunks.push(c),
      (e) => errors.push(e)
    )

    expect(chunks).toEqual(['Hello', ' World', '!'])
    expect(errors).toEqual([])
  })

  test('forwards error lines to onError', async () => {
    const chunks: string[] = []
    const errors: string[] = []

    installFetchMock(() =>
      makeNdjsonResponse([
        '{"response": "ok"}',
        '{"error": "Something went wrong"}',
        '{"response": "more"}',
      ])
    )

    await apiModule.queryTextStream(
      makeQueryRequest(),
      (c) => chunks.push(c),
      (e) => errors.push(e)
    )

    expect(chunks).toEqual(['ok', 'more'])
    expect(errors).toEqual(['Something went wrong'])
  })

  test('skips malformed JSON lines', async () => {
    const chunks: string[] = []

    installFetchMock(() =>
      makeNdjsonResponse([
        'not valid json',
        '{"response": "valid"}',
        '{broken',
      ])
    )

    await apiModule.queryTextStream(
      makeQueryRequest(),
      (c) => chunks.push(c),
      () => {}
    )

    expect(chunks).toEqual(['valid'])
  })

  test('handles multi-byte characters split across chunks', async () => {
    const chunks: string[] = []
    const encoder = new TextEncoder()
    // "Hello 😀🌍" — split the emoji bytes across chunks
    const fullJson = '{"response": "Hello 😀🌍"}\n'
    const fullBytes = encoder.encode(fullJson)
    // Split at an arbitrary point inside the emoji sequence
    const splitAt = 28
    const part1 = fullBytes.slice(0, splitAt)
    const part2 = fullBytes.slice(splitAt)

    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(part1)
        controller.enqueue(part2)
        controller.close()
      },
    })

    installFetchMock(() => new Response(stream, { status: 200 }))

    await apiModule.queryTextStream(
      makeQueryRequest(),
      (c) => chunks.push(c),
      () => {}
    )

    expect(chunks).toEqual(['Hello 😀🌍'])
  })

  test('calls onError when final buffer is unparseable (truncated stream)', async () => {
    // Simulate a stream cut off mid-line with no trailing newline.
    // The residual buffer after the stream ends will be an incomplete JSON
    // object that can't be parsed.
    const encoder = new TextEncoder()
    const jsonLine = '{"response": "ok"}\n'
    const truncatedLine = '{"response": "incom'
    const body = new Uint8Array([
      ...encoder.encode(jsonLine),
      ...encoder.encode(truncatedLine),
    ])

    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(body)
        controller.close()
      },
    })

    installFetchMock(() => new Response(stream, { status: 200 }))

    const chunks: string[] = []
    const errors: string[] = []

    await apiModule.queryTextStream(
      makeQueryRequest(),
      (c) => chunks.push(c),
      (e) => errors.push(e)
    )

    expect(chunks).toEqual(['ok'])
    expect(errors).toEqual([
      'Response stream ended with incomplete data — the response may be truncated.',
    ])
  })
})

describe('queryTextStream — abort / stop button', () => {
  test('exits silently on user abort (signal.aborted)', async () => {
    const controller = new AbortController()
    controller.abort()

    installFetchMock(() => {
      // fetch should reject since the signal is already aborted
      return Promise.reject(new DOMException('Aborted', 'AbortError'))
    })

    let errorCalled = false
    await apiModule.queryTextStream(
      makeQueryRequest(),
      () => {},
      () => { errorCalled = true },
      controller.signal
    )

    expect(errorCalled).toBe(false)
  })

  test('exits silently when AbortError is thrown without a signal', async () => {
    installFetchMock(() =>
      Promise.reject(new DOMException('Aborted', 'AbortError'))
    )

    let errorCalled = false
    await apiModule.queryTextStream(
      makeQueryRequest(),
      () => {},
      () => { errorCalled = true }
    )

    expect(errorCalled).toBe(false)
  })
})

describe('queryTextStream — HTTP errors', () => {
  const errorCases = [
    {
      status: 403,
      expectedMessage:
        'You do not have permission to access this resource (403 Forbidden)',
    },
    {
      status: 404,
      expectedMessage: 'The requested resource does not exist (404 Not Found)',
    },
    {
      status: 429,
      expectedMessage:
        'Too many requests, please try again later (429 Too Many Requests)',
    },
    {
      status: 500,
      expectedMessage:
        'Server error, please try again later (500)',
    },
    {
      status: 502,
      expectedMessage:
        'Server error, please try again later (502)',
    },
    {
      status: 503,
      expectedMessage:
        'Server error, please try again later (503)',
    },
  ]

  for (const { status, expectedMessage } of errorCases) {
    test(`shows friendly message for HTTP ${status}`, async () => {
      installFetchMock(() =>
        makeTextResponse('{"error":"details"}', status)
      )

      let capturedError = ''
      await apiModule.queryTextStream(
        makeQueryRequest(),
        () => {},
        (e) => { capturedError = e }
      )

      expect(capturedError).toBe(expectedMessage)
    })
  }
})

describe('queryTextStream — network errors', () => {
  test('shows network error for Failed to fetch', async () => {
    installFetchMock(() =>
      Promise.reject(new TypeError('Failed to fetch'))
    )

    let capturedError = ''
    await apiModule.queryTextStream(
      makeQueryRequest(),
      () => {},
      (e) => { capturedError = e }
    )

    expect(capturedError).toBe(
      'Network connection error, please check your internet connection'
    )
  })

  test('shows network error for NetworkError', async () => {
    installFetchMock(() =>
      Promise.reject(new Error('NetworkError: connection refused'))
    )

    let capturedError = ''
    await apiModule.queryTextStream(
      makeQueryRequest(),
      () => {},
      (e) => { capturedError = e }
    )

    expect(capturedError).toBe(
      'Network connection error, please check your internet connection'
    )
  })
})

describe('queryTextStream — auth headers', () => {
  test('includes Bearer token when stored', async () => {
    storageData.set('LIGHTRAG-API-TOKEN', 'test-jwt-token')

    let capturedHeaders: HeadersInit | undefined
    installFetchMock((_url: string, init?: RequestInit) => {
      capturedHeaders = init?.headers
      return makeNdjsonResponse(['{"response": "ok"}'])
    })

    await apiModule.queryTextStream(
      makeQueryRequest(),
      () => {},
      () => {}
    )

    // The module builds headers as a plain object literal, so the captured
    // RequestInit.headers is a Record we can assert against directly.
    const sentHeaders = capturedHeaders as Record<string, string>
    expect(sentHeaders).toBeDefined()
    expect(sentHeaders['Authorization']).toBe('Bearer test-jwt-token')
    expect(sentHeaders['Accept']).toBe('application/x-ndjson')
    expect(sentHeaders['Content-Type']).toBe('application/json')
  })

  test('omits Bearer token when none is stored', async () => {
    let capturedHeaders: HeadersInit | undefined
    installFetchMock((_url: string, init?: RequestInit) => {
      capturedHeaders = init?.headers
      return makeNdjsonResponse(['{"response": "ok"}'])
    })

    await apiModule.queryTextStream(
      makeQueryRequest(),
      () => {},
      () => {}
    )

    const sentHeaders = capturedHeaders as Record<string, string>
    expect(sentHeaders['Authorization']).toBeUndefined()
  })

  test('calls /query/stream endpoint', async () => {
    let capturedUrl = ''
    installFetchMock((url: string) => {
      capturedUrl = url
      return makeNdjsonResponse(['{"response": "ok"}'])
    })

    await apiModule.queryTextStream(
      makeQueryRequest(),
      () => {},
      () => {}
    )

    expect(capturedUrl).toBe('http://localhost:9621/query/stream')
  })
})

describe('queryTextStream — guest-token 401 retry', () => {
  test('retries with refreshed guest token on 401', async () => {
    storageData.set('LIGHTRAG-API-TOKEN', 'expired-guest-token')
    authStore.setState({ isGuestMode: true })

    let callCount = 0

    installFetchMock(() => {
      callCount++
      if (callCount === 1) {
        return makeTextResponse('{"error":"unauthorized"}', 401)
      }
      return makeNdjsonResponse(['{"response": "retry ok"}'])
    })

    const chunks: string[] = []
    await apiModule.queryTextStream(
      makeQueryRequest(),
      (c) => chunks.push(c),
      () => {}
    )

    expect(callCount).toBe(2)
    expect(chunks).toEqual(['retry ok'])
  })

  test('classifies a non-auth HTTP error on the retried stream (e.g. 429)', async () => {
    storageData.set('LIGHTRAG-API-TOKEN', 'expired-guest-token')
    authStore.setState({ isGuestMode: true })

    let callCount = 0

    installFetchMock(() => {
      callCount++
      if (callCount === 1) {
        return makeTextResponse('{"error":"unauthorized"}', 401)
      }
      // Refresh succeeded, but the retried request is rate-limited.
      return makeTextResponse('{"error":"rate limited"}', 429)
    })

    let capturedError = ''
    await apiModule.queryTextStream(
      makeQueryRequest(),
      () => {},
      (e) => {
        capturedError = e
      }
    )

    // The retry's 429 must be classified like any other HTTP error, NOT
    // reported as an auth-refresh failure.
    expect(callCount).toBe(2)
    expect(capturedError).toBe(
      'Too many requests, please try again later (429 Too Many Requests)'
    )
  })
})

describe('queryTextStream — auth termination (401)', () => {
  // PRD error split: a 401 means the session's authentication is gone, not
  // that the answer failed. The API layer navigates to the unauthenticated
  // route and queryTextStream REJECTS with the typed error — it must never
  // surface through onError, which the query session renders as an assistant
  // error bubble and persists into history.
  test('non-guest 401 rejects with AuthenticationRequiredError, never calling onError', async () => {
    installFetchMock(() => makeTextResponse('{"error":"unauthorized"}', 401))

    let capturedError: string | null = null
    let caught: unknown = null
    try {
      await apiModule.queryTextStream(
        makeQueryRequest(),
        () => {},
        (e) => {
          capturedError = e
        }
      )
    } catch (error) {
      caught = error
    }

    expect(isAuthenticationRequiredError(caught)).toBe(true)
    expect(capturedError).toBeNull()
  })

  test('guest whose refreshed token is still rejected also gets the typed error', async () => {
    storageData.set('LIGHTRAG-API-TOKEN', 'expired-guest-token')
    authStore.setState({ isGuestMode: true })

    let callCount = 0
    installFetchMock(() => {
      callCount++
      // First request AND the refreshed retry are both rejected.
      return makeTextResponse('{"error":"unauthorized"}', 401)
    })

    let capturedError: string | null = null
    let caught: unknown = null
    try {
      await apiModule.queryTextStream(
        makeQueryRequest(),
        () => {},
        (e) => {
          capturedError = e
        }
      )
    } catch (error) {
      caught = error
    }

    expect(callCount).toBe(2)
    expect(isAuthenticationRequiredError(caught)).toBe(true)
    expect(capturedError).toBeNull()
  })
})
