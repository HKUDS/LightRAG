/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { normalizeEntityName } from './entityName'

describe('normalizeEntityName', () => {
  // Bun runs these unit tests without a DOM, so HTML decoding assertions
  // exercise the explicit non-browser fallback. Production browsers use a
  // detached textarea for the full native HTML5 entity decoder.
  test('mirrors extraction cleanup for HTML, full-width characters, quotes and CJK spaces', () => {
    expect(normalizeEntityName('  <p>“Ａ 公 司”</p>  ')).toBe('A公司')
    expect(normalizeEntityName('《北 京》')).toBe('北京')
    expect(normalizeEntityName('"《Ａ 公 司》"')).toBe('A公司')
    expect(normalizeEntityName('项 目（Ａ－１）')).toBe('项目(A-1)')
  })

  test('unescapes HTML entities and strips invalid control and surrogate characters', () => {
    expect(normalizeEntityName('&quot;ACME&amp;Co&quot;')).toBe('ACME&Co')
    expect(normalizeEntityName('A\u0000B\ud800C')).toBe('ABC')
  })

  test('preserves meaningful ASCII spacing and inner quotes away from CJK text', () => {
    expect(normalizeEntityName('Entity "A"')).toBe('Entity "A"')
    expect(normalizeEntityName('New York')).toBe('New York')
    expect(normalizeEntityName('C++')).toBe('C++')
  })

  test('rejects the same short numeric and dotted names as extraction', () => {
    expect(normalizeEntityName('1')).toBe('')
    expect(normalizeEntityName('12')).toBe('')
    expect(normalizeEntityName('1.2')).toBe('')
    expect(normalizeEntityName('123.4')).toBe('')
    expect(normalizeEntityName('123')).toBe('123')
    expect(normalizeEntityName('123.45')).toBe('123.45')
  })
})
