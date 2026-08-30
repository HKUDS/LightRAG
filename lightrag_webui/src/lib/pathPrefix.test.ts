/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { entryHomeHref, normalizeApiPrefix, normalizeWebuiPrefix } from './pathPrefix'

describe('normalizeApiPrefix', () => {
  test('empty / undefined / null collapse to ""', () => {
    expect(normalizeApiPrefix(undefined)).toBe('')
    expect(normalizeApiPrefix(null)).toBe('')
    expect(normalizeApiPrefix('')).toBe('')
    expect(normalizeApiPrefix('   ')).toBe('')
  })

  test('"/" collapses to "" — avoids `${"/"} + "/x"` producing protocol-relative `//x`', () => {
    expect(normalizeApiPrefix('/')).toBe('')
  })

  test('strips trailing slashes (one or many)', () => {
    expect(normalizeApiPrefix('/api/v1/')).toBe('/api/v1')
    expect(normalizeApiPrefix('/api/v1//')).toBe('/api/v1')
  })

  test('adds leading slash if missing', () => {
    expect(normalizeApiPrefix('api/v1')).toBe('/api/v1')
  })

  test('passes canonical form through unchanged', () => {
    expect(normalizeApiPrefix('/api/v1')).toBe('/api/v1')
  })

  test('result is safe for fetch template concat: never starts with `//` and never ends with `/`', () => {
    for (const input of ['', '/', undefined, '/api', '/api/', 'api', '/api/v1/']) {
      const out = normalizeApiPrefix(input)
      const fetchUrl = `${out}/query/stream`
      expect(fetchUrl.startsWith('//')).toBe(false)
      expect(fetchUrl).not.toContain('//')
    }
  })
})

describe('normalizeWebuiPrefix', () => {
  test('empty / undefined / null fall back to default with trailing slash', () => {
    expect(normalizeWebuiPrefix(undefined)).toBe('/webui/')
    expect(normalizeWebuiPrefix(null)).toBe('/webui/')
    expect(normalizeWebuiPrefix('')).toBe('/webui/')
    expect(normalizeWebuiPrefix('   ')).toBe('/webui/')
  })

  test('"/" falls back to default — degenerate value rejected', () => {
    expect(normalizeWebuiPrefix('/')).toBe('/webui/')
  })

  test('always ends with exactly one trailing slash (Vite `base` requirement)', () => {
    expect(normalizeWebuiPrefix('/admin/ui')).toBe('/admin/ui/')
    expect(normalizeWebuiPrefix('/admin/ui/')).toBe('/admin/ui/')
    expect(normalizeWebuiPrefix('/admin/ui//')).toBe('/admin/ui/')
  })

  test('adds leading slash if missing', () => {
    expect(normalizeWebuiPrefix('admin/ui')).toBe('/admin/ui/')
  })

  test('respects custom fallback', () => {
    expect(normalizeWebuiPrefix(undefined, '/custom')).toBe('/custom/')
    expect(normalizeWebuiPrefix('/', '/custom')).toBe('/custom/')
    expect(normalizeWebuiPrefix('', '/custom/')).toBe('/custom/')
  })
})


/**
 * The brand link's "go to this entry's root" href.
 *
 * The defect it closes: a plain `./` is right only where the entry is served
 * as a DIRECTORY index, which is a production-mount property. The Vite dev
 * server serves each entry as a file, so the workspace entry sits at
 * `/workspace.html`, `./` resolves to `/`, and `/` answers with the ADMIN
 * index.html — clicking the workspace brand during `bun run dev` silently
 * switched products.
 */
describe('entryHomeHref', () => {
  test('production mounts (directory index) keep the plain relative root', () => {
    expect(entryHomeHref('/workspace/')).toBe('./')
    expect(entryHomeHref('/webui/')).toBe('./')
    // Under a proxy prefix, and at the dev server's own root.
    expect(entryHomeHref('/site01/workspace/')).toBe('./')
    expect(entryHomeHref('/')).toBe('./')
  })

  test('a file-named entry anchors on its own filename, NOT the directory', () => {
    // Vite dev: `./` here would resolve to `/` — the admin entry.
    expect(entryHomeHref('/workspace.html')).toBe('workspace.html')
    // The explicit same-entry filename the production mounts still allow.
    expect(entryHomeHref('/workspace/workspace.html')).toBe('workspace.html')
    expect(entryHomeHref('/index.html')).toBe('index.html')
    expect(entryHomeHref('/webui/index.html')).toBe('index.html')
  })

  test('non-HTML paths and missing input fall back to the relative root', () => {
    expect(entryHomeHref('/workspace/some-path')).toBe('./')
    expect(entryHomeHref(undefined)).toBe('./')
    expect(entryHomeHref(null)).toBe('./')
    expect(entryHomeHref('')).toBe('./')
  })
})
