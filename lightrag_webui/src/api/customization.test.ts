import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import {
  CUSTOMIZATION_REQUEST_TIMEOUT_MS,
  fetchUICustomization,
  toBcp47Locale
} from './customization'

/**
 * The customization request is OPTIONAL content whose loading state gates a
 * REQUIRED one: `WorkspaceWelcome` shows nothing but a spinner — no "Sign
 * in", no "Enter workspace" — until it settles. `fetch` has no default
 * timeout, so a route that accepts the connection and never answers (a
 * reverse proxy that has not been told about this new path) used to leave
 * the workspace entry permanently unreachable. These pin the bound.
 *
 * The stub records the init it was given and never settles unless a test
 * settles it, which is exactly the blackholed-route shape.
 */

interface StubRequest {
  url: string
  init: RequestInit | undefined
  settle: (response: Response) => void
  fail: (error: unknown) => void
}

const requests: StubRequest[] = []
const realFetch = globalThis.fetch

const takeRequest = (): StubRequest => {
  const next = requests.shift()
  if (!next) throw new Error('no customization request was issued')
  return next
}

/**
 * The rejection assertions below are exactly the ones a REGRESSION makes
 * hang: an unbounded request settles never, and `await expect(...).rejects`
 * would then stall the whole run instead of failing it. Race a watchdog so
 * losing the bound fails in milliseconds, with a message that names it.
 */
async function rejectionWithin(promise: Promise<unknown>, ms: number): Promise<unknown> {
  const stalled = Symbol('never settled')
  const outcome = await Promise.race([
    promise.then(
      (value) => ({ settled: 'resolved' as const, value }),
      (error) => ({ settled: 'rejected' as const, value: error })
    ),
    new Promise<typeof stalled>((resolve) => setTimeout(() => resolve(stalled), ms))
  ])
  if (outcome === stalled) {
    throw new Error(`the request was still pending after ${ms}ms — it is not bounded`)
  }
  if (outcome.settled !== 'rejected') {
    throw new Error(`expected a rejection, got a resolved value: ${String(outcome.value)}`)
  }
  return outcome.value
}

beforeAll(() => {
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    return new Promise<Response>((resolve, reject) => {
      const entry: StubRequest = { url, init, settle: resolve, fail: reject }
      requests.push(entry)
      // Honour the signal the way a real fetch does: an already-aborted
      // signal rejects at once, and a later abort rejects the pending
      // request with the signal's reason.
      const signal = init?.signal
      if (signal?.aborted) {
        reject(signal.reason)
        return
      }
      signal?.addEventListener('abort', () => reject(signal.reason), { once: true })
    })
  }) as typeof fetch
})

afterAll(() => {
  globalThis.fetch = realFetch
  requests.length = 0
})

describe('fetchUICustomization is bounded', () => {
  test('a request that never answers REJECTS instead of hanging forever', async () => {
    // Regression (P2). Without the timeout this promise never settles and
    // the welcome page keeps its spinner — and its only way in — forever.
    const promise = fetchUICustomization('en', undefined, 20)
    expect(takeRequest().url).toContain('/ui/customization?locale=en')

    const error = await rejectionWithin(promise, 500)
    expect(String(error)).toMatch(/timed out after 20ms/)
  })

  test('a response that stalls MID-BODY is bounded too, not just a silent socket', async () => {
    // The timer is cleared only after the body has been read: headers can
    // arrive promptly and the stream then stall, which hangs just as hard.
    // Modelled the way a real fetch behaves — aborting the signal errors the
    // body stream, which is what makes `response.json()` reject.
    const promise = fetchUICustomization('zh_TW', undefined, 20)
    const request = takeRequest()
    expect(request.url).toContain('locale=zh-TW') // BCP 47 on the wire
    let streamController: ReadableStreamDefaultController<Uint8Array> | null = null
    request.init?.signal?.addEventListener(
      'abort',
      () => streamController?.error(request.init?.signal?.reason),
      { once: true }
    )
    request.settle(
      new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            // Headers are complete; the body never arrives.
            streamController = controller
          }
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    )

    const error = await rejectionWithin(promise, 500)
    expect(String(error)).toMatch(/timed out after 20ms/)
  })

  test('a prompt response is returned, and the timer never fires on it', async () => {
    const promise = fetchUICustomization(null, undefined, 20)
    const request = takeRequest()
    // `language: null` sends no locale parameter at all.
    expect(request.url.endsWith('/ui/customization')).toBe(true)
    request.settle(
      new Response(JSON.stringify({ customized: true, brand: { title: 'Acme' } }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      })
    )

    expect((await promise).brand.title).toBe('Acme')
    // Past the timeout with the request already settled: a leaked timer would
    // abort a controller nothing is listening to, and an unhandled rejection
    // here would fail the run.
    await new Promise((resolve) => setTimeout(resolve, 40))
  })

  test('a caller\'s own signal still aborts, composed with the timeout', async () => {
    // The timeout replaces nothing: whichever fires first wins.
    const controller = new AbortController()
    const promise = fetchUICustomization('en', controller.signal, 10_000)
    takeRequest()
    controller.abort(new Error('caller went away'))

    await expect(promise).rejects.toThrow('caller went away')
  })

  test('a signal already aborted before the call never leaves a request out', async () => {
    const promise = fetchUICustomization('en', AbortSignal.abort(new Error('stale')), 10_000)
    await expect(promise).rejects.toThrow('stale')
  })

  test('the default bound matches the sibling /auth-status request', () => {
    // Both fire from the welcome page; a customization request outliving the
    // auth check would gate the page on the slower of the two.
    expect(CUSTOMIZATION_REQUEST_TIMEOUT_MS).toBe(5000)
    expect(toBcp47Locale('zh_TW')).toBe('zh-TW')
  })
})
