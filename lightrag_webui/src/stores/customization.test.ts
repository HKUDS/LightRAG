import { afterAll, beforeAll, describe, expect, spyOn, test } from 'bun:test'
import type { UICustomization } from '@/api/customization'
import {
  MAX_CUSTOMIZATION_ATTEMPTS,
  needsCustomizationLoad
} from './customization'

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
let settingsStore: typeof import('./settings').useSettingsStore
let SERVER_DEFAULT_LOCALE: string

// The settings store persists on every set(); give it somewhere to write and
// a navigator whose languages match no UI locale (the only situation in which
// the bundle's locale may be adopted for the chrome).
const localData = new Map<string, string>()
let previousStorage: PropertyDescriptor | undefined
let previousNavigator: PropertyDescriptor | undefined

beforeAll(async () => {
  previousStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  previousNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator')
  Object.defineProperty(globalThis, 'localStorage', {
    value: {
      getItem: (key: string) => localData.get(key) ?? null,
      setItem: (key: string, value: string) => {
        localData.set(key, value)
      },
      removeItem: (key: string) => {
        localData.delete(key)
      },
      clear: () => {
        localData.clear()
      }
    },
    configurable: true
  })
  Object.defineProperty(globalThis, 'navigator', {
    value: { languages: ['th-TH'], language: 'th-TH' },
    configurable: true
  })

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
  settingsStore = (await import('./settings')).useSettingsStore
})

afterAll(() => {
  globalThis.fetch = realFetch
  if (previousStorage) {
    Object.defineProperty(globalThis, 'localStorage', previousStorage)
  } else {
    // @ts-expect-error - removing the stub installed for this file
    delete globalThis.localStorage
  }
  if (previousNavigator) {
    Object.defineProperty(globalThis, 'navigator', previousNavigator)
  } else {
    // @ts-expect-error - removing the stub installed for this file
    delete globalThis.navigator
  }
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

  test('the automatic retry is BOUNDED — a hot loop would storm the endpoint', () => {
    // The hook re-arms on every failure (failedAttempts is a dependency), so
    // without this bound the sequence fail → state change → gate → fail …
    // would hammer the endpoint forever.
    expect(needsCustomizationLoad('fr', 'fr', null, 0)).toBe(true)
    expect(needsCustomizationLoad('fr', 'fr', null, MAX_CUSTOMIZATION_ATTEMPTS - 1)).toBe(
      true
    )
    expect(needsCustomizationLoad('fr', 'fr', null, MAX_CUSTOMIZATION_ATTEMPTS)).toBe(false)
    expect(needsCustomizationLoad('fr', 'fr', null, MAX_CUSTOMIZATION_ATTEMPTS + 5)).toBe(
      false
    )
  })

  test('a NEW target is always requested, however many attempts failed before', () => {
    // The budget is per target: switching language must never be blocked by
    // the previous locale's failures.
    expect(needsCustomizationLoad('de', 'fr', null, 99)).toBe(true)
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

  test('failures accumulate per target and reset on success / re-target', async () => {
    const errorSpy = spyOn(console, 'error').mockImplementation(() => {})
    try {
      const first = store.getState().load('uk')
      takePending().reject(new Error('down'))
      await first
      expect(store.getState().failedAttempts).toBe(1)

      const second = store.getState().load('uk')
      takePending().reject(new Error('still down'))
      await second
      expect(store.getState().failedAttempts).toBe(MAX_CUSTOMIZATION_ATTEMPTS)
      // The gate now stops the automatic retry for this locale.
      expect(
        needsCustomizationLoad(
          'uk',
          store.getState().targetLocale,
          store.getState().loadedLocale,
          store.getState().failedAttempts
        )
      ).toBe(false)

      // Re-targeting resets the budget…
      const other = store.getState().load('ru')
      expect(store.getState().failedAttempts).toBe(0)
      takePending().resolve(jsonResponse(snapshotFor('Russian bundle')))
      await other
      // …and a success leaves it at zero.
      expect(store.getState().failedAttempts).toBe(0)
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


/**
 * Chrome/branding alignment: asking for a concrete locale is not enough,
 * because a bundle that does not DECLARE it answers with its own
 * `default_locale`. Nothing used to move i18next to that locale, so English
 * controls stood beside (for example) Korean welcome content.
 *
 * The navigator stub in beforeAll matches no UI locale, so the chrome's
 * language here is the last-resort fallback — the only rank the bundle may
 * override (see bundleLocaleToAdopt).
 */
describe('adopting the locale a bundle actually answered with', () => {
  test('a non-English bundle default becomes the UI language too', async () => {
    settingsStore.setState({ language: 'en', languageUserSelected: false })

    const loading = store.getState().load('en')
    takePending().resolve(
      jsonResponse({
        customized: true,
        // The server fell back: English is not declared by this bundle.
        requested_locale: 'en',
        locale: 'ko',
        fallback_used: true,
        brand: { title: '한국어 번들' }
      })
    )
    await loading

    expect(settingsStore.getState().language).toBe('ko')
    // A fallback is not a choice: the next visit re-resolves from scratch.
    expect(settingsStore.getState().languageUserSelected).toBe(false)
  })

  test('an EXPLICIT choice keeps the chrome, whatever the bundle answers', async () => {
    settingsStore.setState({ language: 'en', languageUserSelected: true })

    const loading = store.getState().load('fr')
    takePending().resolve(
      jsonResponse({ customized: true, locale: 'ko', brand: { title: '한국어 번들' } })
    )
    await loading

    expect(settingsStore.getState().language).toBe('en')
    settingsStore.setState({ languageUserSelected: false })
  })
})
