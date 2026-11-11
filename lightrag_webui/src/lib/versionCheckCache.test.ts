import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import {
  markVersionCheckedFromLogin,
  resetVersionCheckCache,
  wasVersionCheckedThisPageLoad
} from './versionCheckCache'

/**
 * The "already fetched the auth status" cache must be scoped to the PAGE
 * LOAD, not to the tab.
 *
 * The P2 case being pinned: this flag used to live in sessionStorage under
 * `VERSION_CHECKED_FROM_LOGIN`. Tab storage survives a reload, so the app
 * shell's early return kept firing for the rest of the tab's life and the
 * `/auth-status` reconciliation never ran again. A server that flipped from
 * auth-enabled to auth-disabled while an unexpired named-user token sat in
 * the browser was therefore met, on every reload, with that user's identity
 * and history — the fresh guest token that replaces the stale one only
 * arrives in a status response the shell no longer requested.
 */

const LEGACY_SESSION_KEY = 'VERSION_CHECKED_FROM_LOGIN'

/** A page load re-evaluates the module: the query string defeats the module
 * cache, giving a genuinely fresh instance (a template literal keeps the
 * specifier dynamic, as TS cannot resolve a suffixed path statically). */
let reload = 0
const freshPageLoad = async (): Promise<typeof import('./versionCheckCache')> =>
  (await import(`./versionCheckCache?reload=${++reload}`)) as typeof import('./versionCheckCache')

let previousDescriptor: PropertyDescriptor | undefined
const sessionData = new Map<string, string>()

beforeAll(() => {
  previousDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'sessionStorage')
  Object.defineProperty(globalThis, 'sessionStorage', {
    value: {
      getItem: (key: string) => sessionData.get(key) ?? null,
      setItem: (key: string, value: string) => {
        sessionData.set(key, value)
      },
      removeItem: (key: string) => {
        sessionData.delete(key)
      },
      clear: () => {
        sessionData.clear()
      }
    },
    configurable: true
  })
})

afterAll(() => {
  if (previousDescriptor) {
    Object.defineProperty(globalThis, 'sessionStorage', previousDescriptor)
  } else {
    // @ts-expect-error - removing the stub installed for this file
    delete globalThis.sessionStorage
  }
  resetVersionCheckCache()
})

describe('versionCheckCache', () => {
  test('a fresh page load starts UNCHECKED, so the shell reconciles', () => {
    resetVersionCheckCache()
    expect(wasVersionCheckedThisPageLoad()).toBe(false)
  })

  test('marking suppresses the duplicate request within the same page load', () => {
    resetVersionCheckCache()
    markVersionCheckedFromLogin()
    expect(wasVersionCheckedThisPageLoad()).toBe(true)
    // Idempotent: the shell may mark again after its own fetch.
    markVersionCheckedFromLogin()
    expect(wasVersionCheckedThisPageLoad()).toBe(true)
  })

  test('a RELOAD (fresh module instance) forgets the mark', async () => {
    resetVersionCheckCache()
    markVersionCheckedFromLogin()
    expect(wasVersionCheckedThisPageLoad()).toBe(true)

    const reloaded = await freshPageLoad()
    expect(reloaded.wasVersionCheckedThisPageLoad()).toBe(false)
  })

  test('a value left in TAB storage never suppresses the reconciliation', async () => {
    // Exactly the pre-fix state: a previous page load in this tab wrote the
    // legacy flag. The cache must not read it back.
    sessionData.set(LEGACY_SESSION_KEY, 'true')

    const reloaded = await freshPageLoad()
    expect(reloaded.wasVersionCheckedThisPageLoad()).toBe(false)
  })

  test('marking does NOT write the flag into tab storage', () => {
    sessionData.clear()
    resetVersionCheckCache()
    markVersionCheckedFromLogin()
    expect(sessionData.has(LEGACY_SESSION_KEY)).toBe(false)
  })
})
