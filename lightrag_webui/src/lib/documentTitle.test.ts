import { afterEach, describe, expect, test } from 'bun:test'
import {
  DEFAULT_DOCUMENT_TITLE,
  applyInitialDocumentTitle,
  applyServerDocumentTitle,
  resolveInitialDocumentTitle,
  resolveServerDocumentTitle
} from './documentTitle'
import { restoreDomGlobals, withoutDomGlobals } from '@/test/domGlobals'

/**
 * Browser tab title resolution (WEBUI_TITLE).
 *
 * Bun's runner has no DOM, so `window` / `document` are installed per test as
 * the minimal shapes this module touches and removed again afterwards — the
 * `typeof … === 'undefined'` guards are themselves part of what is pinned
 * here, since the store module runs at import time in every entry.
 */

// Cast through `unknown`: the DOM lib types `globalThis.window`/`document` as
// required and fully-featured, which neither a stub nor `delete` can satisfy.
const g = globalThis as unknown as {
  window?: { __LIGHTRAG_CONFIG__?: { webuiTitle?: string | null } }
  document?: { title: string }
}

const setInjectedTitle = (webuiTitle: string | null | undefined): void => {
  g.window = { __LIGHTRAG_CONFIG__: { webuiTitle } }
}

afterEach(() => {
  // Restore rather than delete: the DOM preload installs one window and
  // document for the whole process (see src/test/domGlobals.ts).
  restoreDomGlobals()
})

describe('resolveInitialDocumentTitle', () => {
  test('prefers the injected title over the cached one', () => {
    // The injected value came from the response that served THIS page; the
    // cached one is from some earlier visit, so it may name a title the
    // deployment has since changed.
    setInjectedTitle('Current KB')
    expect(resolveInitialDocumentTitle('Stale KB')).toBe('Current KB')
  })

  test('an injected null overrides the cache — the deployment has no title', () => {
    // Regression: the backend ALWAYS writes the field (WEBUI_TITLE or None),
    // so a null is the server stating there is no title, not a missing
    // injection. Reloading after the deployment cleared WEBUI_TITLE must keep
    // the static default rather than restoring the previous visit's title,
    // which would otherwise stand until a /health response landed — or
    // forever, if the backend is unreachable.
    setInjectedTitle(null)
    expect(resolveInitialDocumentTitle('Stale KB')).toBe(DEFAULT_DOCUMENT_TITLE)
  })

  test('uses the cached title when nothing was injected (dev server)', () => {
    // `undefined` is the other state: the dev server publishes only the
    // prefixes, and an older build may predate the field entirely. There the
    // cache is the only evidence the page has.
    g.window = { __LIGHTRAG_CONFIG__: {} }
    expect(resolveInitialDocumentTitle('My Graph KB')).toBe('My Graph KB')

    g.window = {}
    expect(resolveInitialDocumentTitle('My Graph KB')).toBe('My Graph KB')
  })

  test('falls back to the default when neither source has a title', () => {
    setInjectedTitle(null)
    expect(resolveInitialDocumentTitle(null)).toBe(DEFAULT_DOCUMENT_TITLE)
    expect(resolveInitialDocumentTitle(undefined)).toBe(DEFAULT_DOCUMENT_TITLE)
    expect(resolveInitialDocumentTitle('   ')).toBe(DEFAULT_DOCUMENT_TITLE)
  })

  test('a blank injected title is a published value, not a missing one', () => {
    setInjectedTitle('  ')
    expect(resolveInitialDocumentTitle(null)).toBe(DEFAULT_DOCUMENT_TITLE)
    expect(resolveInitialDocumentTitle('Stale KB')).toBe(DEFAULT_DOCUMENT_TITLE)
  })

  test('tolerates a missing window (SSR / non-browser import)', () => {
    // Actually remove the global rather than leaving a config-less window
    // behind: the `typeof window === 'undefined'` guard is the thing pinned
    // here, and a present-but-empty window never exercises it.
    withoutDomGlobals(() => {
      expect(resolveInitialDocumentTitle('My Graph KB')).toBe('My Graph KB')
      expect(resolveInitialDocumentTitle(null)).toBe(DEFAULT_DOCUMENT_TITLE)
    })
  })
})

describe('resolveServerDocumentTitle', () => {
  test('uses what the server reported', () => {
    setInjectedTitle('Old KB')
    expect(resolveServerDocumentTitle('New KB')).toBe('New KB')
  })

  test('a cleared title reverts to the default, NOT to the injected one', () => {
    // Regression: the server restarted with WEBUI_TITLE unset, so /health
    // reports null. The injected value is a snapshot of the old config and
    // must not resurrect it — the header beside the tab already shows the
    // default, and the tab would stay wrong until a reload.
    setInjectedTitle('Old KB')
    expect(resolveServerDocumentTitle(null)).toBe(DEFAULT_DOCUMENT_TITLE)
    expect(resolveServerDocumentTitle(undefined)).toBe(DEFAULT_DOCUMENT_TITLE)
    expect(resolveServerDocumentTitle('   ')).toBe(DEFAULT_DOCUMENT_TITLE)
  })
})

describe('applying the title', () => {
  test('the initial pass keeps the injected title while the store is empty', () => {
    // First visit: the tab already shows the deployment's name (the snippet in
    // <head> set it) and the store has nothing cached yet. Overwriting it with
    // the generic default would visibly rename the tab a moment after load.
    setInjectedTitle('My Graph KB')
    g.document = { title: 'My Graph KB' }

    applyInitialDocumentTitle(null)
    expect(g.document.title).toBe('My Graph KB')
  })

  test('a server update writes through, and a cleared title resets', () => {
    setInjectedTitle('My Graph KB')
    g.document = { title: 'My Graph KB' }

    applyServerDocumentTitle('Renamed KB')
    expect(g.document.title).toBe('Renamed KB')

    applyServerDocumentTitle(null)
    expect(g.document.title).toBe(DEFAULT_DOCUMENT_TITLE)
  })

  test('does not write when the title is already correct', () => {
    setInjectedTitle(null)
    let writes = 0
    let stored = 'My Graph KB'
    g.document = {
      get title() {
        return stored
      },
      set title(value: string) {
        writes += 1
        stored = value
      }
    }

    applyServerDocumentTitle('My Graph KB')
    expect(writes).toBe(0)

    applyServerDocumentTitle('Other KB')
    expect(writes).toBe(1)
    expect(stored).toBe('Other KB')
  })

  test('is a no-op without a document', () => {
    setInjectedTitle('My Graph KB')
    // Only `document` goes: with an injected title still readable from
    // `window`, both entry points resolve a real title and then have
    // nowhere to write it — which is the guard being pinned.
    withoutDomGlobals(() => {
      expect(() => applyInitialDocumentTitle(null)).not.toThrow()
      expect(() => applyServerDocumentTitle(null)).not.toThrow()
    }, ['document'])
  })
})
