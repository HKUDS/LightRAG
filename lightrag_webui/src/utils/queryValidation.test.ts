/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import {
  isQueryEmpty,
  isRagQueryTooShort,
  meetsMinRagQueryWeight,
  ragQueryWeight,
  WIDE_CODEPOINT_RANGES
} from './queryValidation'

describe('RAG query validation', () => {
  test('counts each East Asian character as two English-equivalent ones', () => {
    expect(ragQueryWeight('ab')).toBe(2)
    expect(ragQueryWeight('中')).toBe(2)
    expect(ragQueryWeight('中a')).toBe(3)
    expect(ragQueryWeight('中文')).toBe(4)
    expect(ragQueryWeight('  abc  ')).toBe(3)
  })

  test('weighs kana, Hangul and Bopomofo like Han', () => {
    // Doubling Han alone rejected ordinary two-character Japanese, Korean and
    // zhuyin queries while accepting their Han equivalents.
    expect(ragQueryWeight('ねこ')).toBe(4)
    expect(ragQueryWeight('データ')).toBe(6)
    expect(ragQueryWeight('한글')).toBe(4)
    expect(ragQueryWeight('ㄅㄆ')).toBe(4)
  })

  test('weighs halfwidth forms like their fullwidth counterparts', () => {
    // Halfwidth katakana renders narrow but is the same word in a legacy
    // encoding, still emitted by Japanese IMEs.
    expect(ragQueryWeight('ﾈｺ')).toBe(ragQueryWeight('ネコ'))
    expect(ragQueryWeight('ｶﾅ')).toBe(ragQueryWeight('カナ'))
    expect(isRagQueryTooShort('ﾈｺ', 'mix')).toBe(false)
  })

  test('covers every CJK ideograph plane without an edit', () => {
    // Planes 2 and 3 are allocated to CJK ideographs, so one range covers every
    // extension there is and will be — Extension J arrived in Unicode 17 and
    // needed no change here.
    for (const codePoint of [0x4e00, 0x3400, 0x20000, 0x2a700, 0x31350, 0x323b0, 0x3fffd]) {
      expect(ragQueryWeight(String.fromCodePoint(codePoint).repeat(2))).toBe(4)
    }
  })

  test('over-counts punctuation and unassigned code points, deliberately', () => {
    // The ranges are block- and plane-granular, so these weigh 2 apiece and
    // clear the minimum. Chasing that per character is what made the table need
    // an edit, in two languages, on every Unicode release; the cost of getting
    // it wrong is one retrieval that finds nothing.
    for (const query of ['・・', '゛゜', 'ㅤㅤ', '\u{1afff}\u{1afff}']) {
      expect(ragQueryWeight(query)).toBe(4)
    }
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

  test('flags an empty query independently of the mode', () => {
    // Exemption from the minimum is not exemption from carrying a prompt.
    expect(isQueryEmpty('')).toBe(true)
    expect(isQueryEmpty('   ')).toBe(true)
    expect(isQueryEmpty('a')).toBe(false)
  })

  test('the threshold check agrees with the full walk', () => {
    for (const query of ['', ' ', 'a', 'ab', 'abc', '中', '中a', 'ねこ', '한a']) {
      expect(meetsMinRagQueryWeight(query)).toBe(ragQueryWeight(query) >= 3)
    }
  })

  test('the threshold check stops as soon as the minimum is reached', () => {
    // Walking the whole 64 KiB a query may carry is pure waste: the answer is
    // settled by the third character at the latest.
    let read = 0
    const counted = {
      trim: () => ({
        *[Symbol.iterator]() {
          for (const character of 'a'.repeat(65536)) {
            read += 1
            yield character
          }
        }
      })
    } as unknown as string

    expect(meetsMinRagQueryWeight(counted)).toBe(true)
    expect(read).toBe(3)
  })

  test('the range table is sorted and disjoint', () => {
    // The backend owns the table's content and asserts the two files match;
    // this catches a malformed hand-edit here first.
    for (const [start, end] of WIDE_CODEPOINT_RANGES) {
      expect(start).toBeLessThanOrEqual(end)
    }
    for (let i = 1; i < WIDE_CODEPOINT_RANGES.length; i++) {
      expect(WIDE_CODEPOINT_RANGES[i - 1][1]).toBeLessThan(WIDE_CODEPOINT_RANGES[i][0])
    }
  })
})
