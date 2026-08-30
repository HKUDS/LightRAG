import { afterEach, describe, expect, test } from 'bun:test'
import {
  DEFAULT_DOCUMENT_TITLE,
  applyDocumentTitle,
  resolveDocumentTitle
} from './documentTitle'

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
  delete g.window
  delete g.document
})

describe('resolveDocumentTitle', () => {
  test('uses the store title when the deployment set one', () => {
    setInjectedTitle(null)
    expect(resolveDocumentTitle('My Graph KB')).toBe('My Graph KB')
  })

  test('falls back to the default when nothing is configured', () => {
    setInjectedTitle(null)
    expect(resolveDocumentTitle(null)).toBe(DEFAULT_DOCUMENT_TITLE)
    expect(resolveDocumentTitle(undefined)).toBe(DEFAULT_DOCUMENT_TITLE)
    expect(resolveDocumentTitle('   ')).toBe(DEFAULT_DOCUMENT_TITLE)
  })

  test('falls back to the server-injected title before the store has one', () => {
    // First visit: the tab already shows the deployment's name (the snippet in
    // <head> set it), and the store is still empty because no authenticated
    // request has returned a title yet. Falling back to the generic default
    // here would visibly rename the tab a moment after it loaded.
    setInjectedTitle('My Graph KB')
    expect(resolveDocumentTitle(null)).toBe('My Graph KB')
  })

  test('prefers the store title over the injected one', () => {
    // The injected value is frozen at page load; the store is refreshed by the
    // /health poll, so a server restarted with a new WEBUI_TITLE wins.
    setInjectedTitle('Old Name')
    expect(resolveDocumentTitle('New Name')).toBe('New Name')
  })

  test('ignores a blank injected title', () => {
    setInjectedTitle('  ')
    expect(resolveDocumentTitle(null)).toBe(DEFAULT_DOCUMENT_TITLE)
  })

  test('tolerates a page served without the runtime-config snippet', () => {
    g.window = {}
    expect(resolveDocumentTitle('My Graph KB')).toBe('My Graph KB')
    expect(resolveDocumentTitle(null)).toBe(DEFAULT_DOCUMENT_TITLE)
  })

  test('tolerates a missing window (SSR / non-browser import)', () => {
    expect(resolveDocumentTitle('My Graph KB')).toBe('My Graph KB')
    expect(resolveDocumentTitle(null)).toBe(DEFAULT_DOCUMENT_TITLE)
  })
})

describe('applyDocumentTitle', () => {
  test('writes the resolved title to the document', () => {
    setInjectedTitle(null)
    g.document = { title: 'LightRAG' }

    applyDocumentTitle('My Graph KB')
    expect(g.document.title).toBe('My Graph KB')

    applyDocumentTitle(null)
    expect(g.document.title).toBe(DEFAULT_DOCUMENT_TITLE)
  })

  test('leaves the injected title in place when the store is empty', () => {
    setInjectedTitle('My Graph KB')
    g.document = { title: 'My Graph KB' }

    applyDocumentTitle(null)
    expect(g.document.title).toBe('My Graph KB')
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

    applyDocumentTitle('My Graph KB')
    expect(writes).toBe(0)

    applyDocumentTitle('Other KB')
    expect(writes).toBe(1)
    expect(stored).toBe('Other KB')
  })

  test('is a no-op without a document', () => {
    setInjectedTitle('My Graph KB')
    expect(() => applyDocumentTitle('My Graph KB')).not.toThrow()
  })
})
