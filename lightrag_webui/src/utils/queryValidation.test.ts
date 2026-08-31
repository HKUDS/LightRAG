/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { isRagQueryTooShort, ragQueryWeight } from './queryValidation'

describe('RAG query validation', () => {
  test('counts each Han character as two English-equivalent characters', () => {
    expect(ragQueryWeight('ab')).toBe(2)
    expect(ragQueryWeight('中')).toBe(2)
    expect(ragQueryWeight('中a')).toBe(3)
    expect(ragQueryWeight('中文')).toBe(4)
    expect(ragQueryWeight('  abc  ')).toBe(3)
  })

  test('requires weight three for retrieval modes', () => {
    expect(isRagQueryTooShort('ab', 'mix')).toBe(true)
    expect(isRagQueryTooShort('中', 'naive')).toBe(true)
    expect(isRagQueryTooShort('abc', 'local')).toBe(false)
    expect(isRagQueryTooShort('中a', 'global')).toBe(false)
    expect(isRagQueryTooShort('中文', 'hybrid')).toBe(false)
  })

  test('does not impose the RAG minimum on bypass mode', () => {
    expect(isRagQueryTooShort('', 'bypass')).toBe(false)
    expect(isRagQueryTooShort('a', 'bypass')).toBe(false)
  })
})
