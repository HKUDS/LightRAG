import type { QueryMode } from '@/api/lightrag'

export const MIN_RAG_QUERY_WEIGHT = 3

// East Asian scripts count as two English-equivalent characters. Han ideographs,
// Japanese kana and Korean Hangul are all included — doubling Han alone rejected
// ordinary two-character Japanese and Korean words ('ねこ', '오늘') while
// accepting their Han equivalents. The halfwidth forms are the same letters in a
// legacy encoding — 'ﾈｺ' is the word 'ネコ' — so they weigh the same; the rule is
// about how much a character says, not how wide it renders.
//
// ASSIGNED LETTERS ONLY (Unicode category L*, minus the three fillers), which is
// why blocks appear split: doubling whole blocks let '・・', '゛゜', two HANGUL
// FILLERs and unassigned code points such as U+1AFFF clear the minimum.
//
// GENERATED, not hand-edited — from CPython's unicodedata plus an explicit
// post-UCD-14 allowlist, and verified there. This runtime's \p{L} is NOT the
// oracle: Bun 1.3.11 calls U+2B73A-U+2B73F, U+2CEA2-U+2CEAD and U+323B0
// assigned letters, which is how unassigned tails got into this table.
// Keep in sync with lightrag/query_validation.py.
export const WIDE_CODEPOINT_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0x1100, 0x115e], // Hangul Jamo
  [0x1161, 0x11ff], // Hangul Jamo
  [0x3005, 0x3006], // Ideographic iteration mark 々 and closing mark 〆
  [0x3031, 0x3035], // Vertical kana repeat marks 〱〲〳〴〵
  [0x303b, 0x303c], // Vertical ideographic iteration mark 〻, masu mark 〼
  [0x3041, 0x3096], // Hiragana
  [0x309d, 0x309f], // Hiragana
  [0x30a1, 0x30fa], // Katakana
  [0x30fc, 0x30ff], // Katakana
  [0x3105, 0x312f], // Bopomofo (zhuyin) — the phonetic notation used in Taiwan
  [0x3131, 0x3163], // Hangul Compatibility Jamo
  [0x3165, 0x318e], // Hangul Compatibility Jamo
  [0x31a0, 0x31bf], // Bopomofo Extended
  [0x31f0, 0x31ff], // Katakana Phonetic Extensions
  [0x3400, 0x4dbf], // CJK Ext A
  [0x4e00, 0x9fff], // CJK Unified Ideographs
  [0xa960, 0xa97c], // Hangul Jamo Extended-A
  [0xac00, 0xd7a3], // Hangul Syllables
  [0xd7b0, 0xd7c6], // Hangul Jamo Extended-B
  [0xd7cb, 0xd7fb], // Hangul Jamo Extended-B
  [0xf900, 0xfa6d], // CJK Compatibility Ideographs
  [0xfa70, 0xfad9], // CJK Compatibility Ideographs
  [0xff66, 0xff9d], // Halfwidth Katakana
  [0xffa1, 0xffbe], // Halfwidth Hangul
  [0xffc2, 0xffc7], // Halfwidth Hangul
  [0xffca, 0xffcf], // Halfwidth Hangul
  [0xffd2, 0xffd7], // Halfwidth Hangul
  [0xffda, 0xffdc], // Halfwidth Hangul
  [0x1aff0, 0x1aff3], // Kana Extended-B
  [0x1aff5, 0x1affb], // Kana Extended-B
  [0x1affd, 0x1affe], // Kana Extended-B
  [0x1b000, 0x1b122], // Kana Supplement / Extended-A / Small Kana Ext
  [0x1b132, 0x1b132], // Kana Supplement / Extended-A / Small Kana Ext
  [0x1b150, 0x1b152], // Kana Supplement / Extended-A / Small Kana Ext
  [0x1b155, 0x1b155], // Kana Supplement / Extended-A / Small Kana Ext
  [0x1b164, 0x1b167], // Kana Supplement / Extended-A / Small Kana Ext
  [0x1b170, 0x1b2fb], // Nushu — a script for writing Chinese
  [0x20000, 0x2a6df], // CJK Ext B
  [0x2a700, 0x2b739], // CJK Ext C
  [0x2b740, 0x2b81d], // CJK Ext D
  [0x2b820, 0x2cea1], // CJK Ext E
  [0x2ceb0, 0x2ebe0], // CJK Ext F
  [0x2ebf0, 0x2ee5d], // CJK Ext I
  [0x2f800, 0x2fa1d], // CJK Compatibility Ideographs Supplement
  [0x30000, 0x3134a], // CJK Ext G
  [0x31350, 0x323af] // CJK Ext H
]

// CJK numerals written as characters rather than digits — general category Nl,
// not L*, so they are a second table rather than entries in the one above.
// 〇 is how a Chinese year is written (二〇二五年); the Suzhou/Hangzhou numerals
// are the same idea. Mirrors _WIDE_NUMERAL_RANGES in lightrag/query_validation.py.
export const WIDE_NUMERAL_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0x3007, 0x3007], // IDEOGRAPHIC NUMBER ZERO 〇
  [0x3021, 0x3029], // HANGZHOU NUMERAL ONE..NINE
  [0x3038, 0x303a] // HANGZHOU NUMERAL TEN, TWENTY, THIRTY
]

function isWideCharacter(character: string): boolean {
  const codePoint = character.codePointAt(0) ?? 0
  const inRange = ([start, end]: readonly [number, number]) =>
    codePoint >= start && codePoint <= end
  return WIDE_CODEPOINT_RANGES.some(inRange) || WIDE_NUMERAL_RANGES.some(inRange)
}

/**
 * English-equivalent length of the trimmed query. Prefer
 * `meetsMinRagQueryWeight` when only the comparison matters — this walks the
 * whole string, which is up to 64 KiB.
 */
export function ragQueryWeight(query: string): number {
  return Array.from(query.trim()).reduce(
    (weight, character) => weight + (isWideCharacter(character) ? 2 : 1),
    0
  )
}

/** Short-circuits once the threshold is reached: at most two iterations. */
export function meetsMinRagQueryWeight(query: string): boolean {
  let weight = 0
  for (const character of query.trim()) {
    weight += isWideCharacter(character) ? 2 : 1
    if (weight >= MIN_RAG_QUERY_WEIGHT) return true
  }
  return false
}

/** Every mode must carry a prompt — `bypass` and direct-LLM paths included. */
export function isQueryEmpty(query: string): boolean {
  return query.trim().length === 0
}

export function isRagQueryTooShort(query: string, mode: QueryMode): boolean {
  return mode !== 'bypass' && !meetsMinRagQueryWeight(query)
}
