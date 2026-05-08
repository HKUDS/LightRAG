/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { normalizeApiPrefix, normalizeWebuiPrefix } from './pathPrefix'

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
