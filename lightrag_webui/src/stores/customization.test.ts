import { afterAll, beforeAll, describe, expect, spyOn, test } from 'bun:test'
import type { UICustomization } from '@/api/customization'
import { needsCustomizationLoad } from './customization'

/**
 * Customization store: correctness under fast language switching.
 *
 * The race being pinned (user report): with only a "newest request id" check,
 * A → B (request in flight) → A issues no new request for A (its snapshot is
 * already loaded), so B's id stays the newest and B's late response is
 * APPLIED while the user has selected A. The fix: re-targeting bumps the
 * request counter even without a network request, and a response is applied
 * only when it is BOTH the newest request AND for the still-current target
 * locale.
 *
 * Isolation: the real global fetch is swapped for a controllable deferred
 * stub in beforeAll and RESTORED in afterAll (no mock.module — the store and
 * API modules stay real for later test files).
 */

interface PendingFetch {
  url: string
  resolve: (response: Response) => void
  reject: (error: unknown) => void
}

const pendingFetches: PendingFetch[] = []
const fetchUrls: string[] = []
const realFetch = globalThis.fetch

const takePending = (): PendingFetch => {
  const next = pendingFetches.shift()
  if (!next) throw new Error('no pending customization fetch')
  return next
}

const jsonResponse = (body: unknown): Response =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' }
  })

const snapshotFor = (title: string): UICustomization => ({
  customized: true,
  brand: { title }
})

let store: typeof import('./customization').useCustomizationStore
let SERVER_DEFAULT_LOCALE: string

beforeAll(async () => {
  globalThis.fetch = ((input: RequestInfo | URL) => {
    const url = String(input)
    fetchUrls.push(url)
    return new Promise<Response>((resolve, reject) => {
      pendingFetches.push({ url, resolve, reject })
    })
  }) as typeof fetch

  const mod = await import('./customization')
  store = mod.useCustomizationStore
  SERVER_DEFAULT_LOCALE = mod.SERVER_DEFAULT_LOCALE
})

afterAll(() => {
  globalThis.fetch = realFetch
})

// The store is a module singleton — these tests run in order and continue
// from each other's state, exactly like a real page session.
describe('useCustomizationStore locale targeting', () => {
  test('first load applies the response atomically', async () => {
    const loading = store.getState().load('en')
    expect(store.getState().status).toBe('loading')
    expect(fetchUrls[fetchUrls.length - 1]).toContain('locale=en')

    takePending().resolve(jsonResponse(snapshotFor('English bundle')))
    await loading

    const state = store.getState()
    expect(state.status).toBe('ready')
    expect(state.snapshot?.brand.title).toBe('English bundle')
    expect(state.loadedLocale).toBe('en')
    expect(state.targetLocale).toBe('en')
  })

  test('A → B (in flight) → A: B\'s late response is discarded, without refetching A', async () => {
    const callsBefore = fetchUrls.length

    // Switch to B: request goes out and stays in flight.
    const loadingB = store.getState().load('ar')
    expect(fetchUrls.length).toBe(callsBefore + 1)

    // Switch back to A before B returns: no new request (A is loaded)…
    await store.getState().load('en')
    expect(fetchUrls.length).toBe(callsBefore + 1)
    expect(store.getState().targetLocale).toBe('en')

    // …and B's response, arriving late, must be dropped.
    takePending().resolve(jsonResponse(snapshotFor('Arabic bundle')))
    await loadingB

    const state = store.getState()
    expect(state.snapshot?.brand.title).toBe('English bundle')
    expect(state.loadedLocale).toBe('en')
    expect(state.targetLocale).toBe('en')
    expect(state.status).toBe('ready')
  })

  test('a late FAILURE for a stale target is dropped just as silently', async () => {
    const loadingB = store.getState().load('ar')
    await store.getState().load('en')

    takePending().reject(new Error('network down'))
    await loadingB

    const state = store.getState()
    expect(state.snapshot?.brand.title).toBe('English bundle')
    expect(state.status).toBe('ready')
    expect(state.loadedLocale).toBe('en')
  })

  test('a genuine switch still applies the new locale (BCP 47 on the wire)', async () => {
    const loading = store.getState().load('zh_TW')
    // Internal underscore id converts to the hyphen form on the way out.
    expect(fetchUrls[fetchUrls.length - 1]).toContain('locale=zh-TW')

    takePending().resolve(jsonResponse(snapshotFor('Traditional bundle')))
    await loading

    const state = store.getState()
    expect(state.snapshot?.brand.title).toBe('Traditional bundle')
    expect(state.loadedLocale).toBe('zh_TW')
  })

  test('a failing re-request for the CURRENT target keeps the last snapshot', async () => {
    const errorSpy = spyOn(console, 'error').mockImplementation(() => {})
    try {
      const loading = store.getState().load('fr')
      takePending().reject(new Error('temporarily unreachable'))
      await loading

      const state = store.getState()
      expect(state.status).toBe('ready')
      expect(state.snapshot?.brand.title).toBe('Traditional bundle')
      // The target moved on; the loaded locale honestly reflects the snapshot.
      expect(state.targetLocale).toBe('fr')
      expect(state.loadedLocale).toBe('zh_TW')
    } finally {
      errorSpy.mockRestore()
    }
  })

  test('SERVER_DEFAULT_LOCALE sends NO locale parameter', async () => {
    const loading = store.getState().load(SERVER_DEFAULT_LOCALE)
    const url = fetchUrls[fetchUrls.length - 1]
    expect(url).toContain('/ui/customization')
    expect(url).not.toContain('locale=')

    takePending().resolve(
      jsonResponse({ customized: true, locale: 'ko', brand: { title: 'Bundle default' } })
    )
    await loading

    const state = store.getState()
    expect(state.snapshot?.brand.title).toBe('Bundle default')
    expect(state.loadedLocale).toBe(SERVER_DEFAULT_LOCALE)
  })
})

describe('needsCustomizationLoad (the hook\'s gate)', () => {
  // Keying on the TARGET alone was the defect: a failed request leaves the
  // target already equal to the locale, so the hook never tried again — a
  // transient first-load failure stuck the SPA on default branding and a
  // failed switch stuck it on the previous locale until a reload.
  test('a locale that is targeted but NOT loaded still asks for a load (retry)', () => {
    expect(needsCustomizationLoad('fr', 'fr', null)).toBe(true)
    expect(needsCustomizationLoad('fr', 'fr', 'en')).toBe(true)
  })

  test('a locale that is neither targeted nor loaded asks for a load', () => {
    expect(needsCustomizationLoad('fr', 'en', 'en')).toBe(true)
    expect(needsCustomizationLoad('fr', null, null)).toBe(true)
  })

  test('the A → B → A re-target still asks (it invalidates B in flight)', () => {
    // Target is B while A is what is displayed: switching back must call
    // load('A') so the store re-targets and B's response is dropped.
    expect(needsCustomizationLoad('en', 'ar', 'en')).toBe(true)
  })

  test('the steady state (targeted AND loaded) asks for nothing', () => {
    expect(needsCustomizationLoad('en', 'en', 'en')).toBe(false)
    expect(needsCustomizationLoad('', '', '')).toBe(false)
  })
})

describe('retry and in-flight dedupe', () => {
  test('a FAILED locale is retried by a later load, with a real request', async () => {
    const errorSpy = spyOn(console, 'error').mockImplementation(() => {})
    try {
      const failing = store.getState().load('ja')
      const callsBeforeFailure = fetchUrls.length
      takePending().reject(new Error('transient network failure'))
      await failing
      expect(store.getState().loadedLocale).not.toBe('ja')

      // The gate says retry…
      expect(
        needsCustomizationLoad('ja', store.getState().targetLocale, store.getState().loadedLocale)
      ).toBe(true)
      // …and the retry issues a NEW request that can succeed.
      const retry = store.getState().load('ja')
      expect(fetchUrls.length).toBe(callsBeforeFailure + 1)
      takePending().resolve(jsonResponse(snapshotFor('Japanese bundle')))
      await retry

      expect(store.getState().snapshot?.brand.title).toBe('Japanese bundle')
      expect(store.getState().loadedLocale).toBe('ja')
    } finally {
      errorSpy.mockRestore()
    }
  })

  test('concurrent loads for the SAME locale issue ONE request', async () => {
    const callsBefore = fetchUrls.length

    // Two components mounting (or the gate re-firing while the request is
    // still out) must not storm the endpoint — nor invalidate each other.
    const first = store.getState().load('de')
    const second = store.getState().load('de')
    expect(fetchUrls.length).toBe(callsBefore + 1)

    takePending().resolve(jsonResponse(snapshotFor('German bundle')))
    await Promise.all([first, second])

    // The single request still lands (the duplicate did not invalidate it).
    expect(store.getState().snapshot?.brand.title).toBe('German bundle')
    expect(store.getState().loadedLocale).toBe('de')
    expect(store.getState().pendingLocale).toBeNull()
  })
})
