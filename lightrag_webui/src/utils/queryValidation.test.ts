/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { WIDE_CODEPOINT_RANGES } from './queryValidation'
import {
  isQueryEmpty,
  isRagQueryTooShort,
  meetsMinRagQueryWeight,
  ragQueryWeight
} from './queryValidation'

describe('RAG query validation', () => {
  test('counts each CJK character as two English-equivalent characters', () => {
    expect(ragQueryWeight('ab')).toBe(2)
    expect(ragQueryWeight('中')).toBe(2)
    expect(ragQueryWeight('中a')).toBe(3)
    expect(ragQueryWeight('中文')).toBe(4)
    expect(ragQueryWeight('  abc  ')).toBe(3)
  })

  test('weighs kana and Hangul like Han', () => {
    // Doubling Han alone rejected ordinary two-character Japanese and Korean
    // words while accepting their Han equivalents.
    expect(ragQueryWeight('ねこ')).toBe(4)
    expect(ragQueryWeight('データ')).toBe(6)
    expect(ragQueryWeight('한글')).toBe(4)
    expect(ragQueryWeight('오늘')).toBe(4)
  })

  test('weighs halfwidth kana and Hangul like their fullwidth forms', () => {
    // Halfwidth katakana renders narrow but is the same word in a legacy
    // encoding, still emitted by Japanese IMEs — the rule counts retrieval
    // signal, not display width.
    expect(ragQueryWeight('ﾈｺ')).toBe(ragQueryWeight('ネコ'))
    expect(ragQueryWeight('ｶﾅ')).toBe(ragQueryWeight('カナ'))
    expect(isRagQueryTooShort('ﾈｺ', 'mix')).toBe(false)
    expect(isRagQueryTooShort('ﾡﾢ', 'mix')).toBe(false)
  })

  test('weighs iteration and repeat marks in both writing directions', () => {
    // The vertical marks were missing while the horizontal ones at
    // U+309D-U+309F were already doubled.
    expect(ragQueryWeight('〳〵')).toBe(4) // vertical kana repeat marks
    expect(ragQueryWeight('々々')).toBe(4) // U+3005 ideographic iteration mark
    expect(ragQueryWeight('〆〆')).toBe(4) // U+3006 ideographic closing mark
    expect(ragQueryWeight('〻〼')).toBe(4) // U+303B, U+303C
    expect(ragQueryWeight('〡〢')).toBe(4) // Hangzhou numerals, Nl like 〇
    for (const query of ['〳〵', '々々', '〡〢']) {
      expect(isRagQueryTooShort(query, 'mix')).toBe(false)
    }
  })

  test('weighs the supplementary Chinese iteration marks', () => {
    // U+16FE1 belongs with Nushu and U+16FE3 with Old Chinese, both already
    // weighted as scripts; U+16FE2 between them is punctuation and U+16FE0 is
    // the Tangut mark, which writes another language.
    expect(ragQueryWeight('\u{16fe1}\u{16fe1}')).toBe(4)
    expect(ragQueryWeight('\u{16fe3}\u{16fe3}')).toBe(4)
    expect(ragQueryWeight('\u{16fe2}\u{16fe2}')).toBe(2)
    expect(ragQueryWeight('\u{16fe0}\u{16fe0}')).toBe(2)
  })

  test('leaves the halfwidth sound marks at one', () => {
    // Lm, so the letters rule alone would double them — but halfwidth ｶﾞ is the
    // single syllable fullwidth writes as ガ, and doubling the mark would make
    // that syllable cost more halfwidth than fullwidth.
    expect(ragQueryWeight('ﾞ')).toBe(1)
    expect(ragQueryWeight('ﾟ')).toBe(1)
    expect(ragQueryWeight('ｶﾞ')).toBe(3)
  })

  test('weighs modifier letters that spell real words', () => {
    expect(ragQueryWeight('コーヒー')).toBe(8) // U+30FC prolonged sound mark (Lm)
    expect(ragQueryWeight('\u{1aff0}\u{1aff1}')).toBe(4) // Kana Extended-B
  })

  test('leaves punctuation, sound marks and fillers at one', () => {
    // Doubling whole Unicode blocks let a query of pure punctuation clear the
    // minimum; the ranges are letters only, so these weigh 1 each.
    for (const character of [
      '｡', // halfwidth ideographic full stop
      '･', // halfwidth katakana middle dot
      'ﾞ', // halfwidth voiced sound mark
      'ﾟ',
      '・', // U+30FB KATAKANA MIDDLE DOT (Po)
      '゛', // U+309B (Sk)
      '゜', // U+309C (Sk)
      '゠', // U+30A0 KATAKANA-HIRAGANA DOUBLE HYPHEN (Pd)
      'ㅤ' // U+3164 HANGUL FILLER — renders as nothing
    ]) {
      expect(ragQueryWeight(character)).toBe(1)
    }
  })

  test('a query of pure punctuation never clears the minimum', () => {
    for (const query of ['・・', '゛゜', '゠゠', 'ㅤㅤ']) {
      expect(isRagQueryTooShort(query, 'mix')).toBe(true)
    }
  })

  test('requires weight three for retrieval modes', () => {
    expect(isRagQueryTooShort('ab', 'mix')).toBe(true)
    expect(isRagQueryTooShort('中', 'naive')).toBe(true)
    expect(isRagQueryTooShort('abc', 'local')).toBe(false)
    expect(isRagQueryTooShort('中a', 'global')).toBe(false)
    expect(isRagQueryTooShort('中文', 'hybrid')).toBe(false)
    expect(isRagQueryTooShort('ねこ', 'mix')).toBe(false)
    expect(isRagQueryTooShort('한글', 'mix')).toBe(false)
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
  test('the range table matches the backend, entry for entry', () => {
    // The backend owns the invariant (its `unicodedata` is the oracle; this
    // runtime's \p{L} is NOT — Bun 1.3.11 calls U+2B73A and even U+323B0,
    // past the end of CJK Extension H, assigned letters). What this side can
    // usefully assert is shape, so a hand-edit here is caught before the
    // cross-language sync test in the Python suite runs.
    for (const [start, end] of WIDE_CODEPOINT_RANGES) {
      expect(start).toBeLessThanOrEqual(end)
      expect(Number.isInteger(start) && Number.isInteger(end)).toBe(true)
    }
    for (let i = 1; i < WIDE_CODEPOINT_RANGES.length; i++) {
      expect(WIDE_CODEPOINT_RANGES[i - 1][1]).toBeLessThan(WIDE_CODEPOINT_RANGES[i][0])
    }
  })

  test('unassigned code points are not weighted', () => {
    // Kana Extended-B gaps, Hiragana block gaps, and the CJK Extension C
    // and E tails — every one of them weighed 2 apiece and cleared the
    // minimum while the table doubled whole blocks.
    for (const codePoint of [
      0x1aff4, 0x1affc, 0x1afff, 0x3040, 0x3097, 0x2b73a, 0x2b73f, 0x2cea2, 0x2cead
    ]) {
      const query = String.fromCodePoint(codePoint).repeat(2)
      expect(ragQueryWeight(query)).toBe(2)
      expect(isRagQueryTooShort(query, 'mix')).toBe(true)
    }
  })

  test('the last assigned ideograph of each trimmed block still weighs two', () => {
    // The trim must stop at the tail, not eat the block.
    for (const codePoint of [0x2b739, 0x2cea1, 0x2ebf0, 0x31350]) {
      expect(ragQueryWeight(String.fromCodePoint(codePoint).repeat(2))).toBe(4)
    }
  })

  test('the ideographic zero is weighted despite not being a letter', () => {
    // U+3007 is category Nl — a deliberate exception, because 二〇二五年.
    expect(ragQueryWeight('二〇')).toBe(4)
  })
})
