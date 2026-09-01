import type { QueryMode } from '@/api/lightrag'

export const MIN_RAG_QUERY_WEIGHT = 3

// East Asian characters count as two English-equivalent ones.
//
// DELIBERATELY COARSE — block and plane granularity, not per assigned
// character. U+20000-U+3FFFD is the whole of planes 2 and 3, which Unicode
// allocates to CJK ideographs, so every extension from B through J and every
// future one is covered without an edit. The precision given up is worth
// nothing here: an unassigned or punctuation code point inside these ranges
// weighs 2 instead of 1, and the cost of that is one retrieval that finds
// nothing.
//
// Keep in sync with lightrag/query_validation.py.
export const WIDE_CODEPOINT_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0x1100, 0x115f], // Hangul Jamo
  [0x2e80, 0x303e], // CJK Radicals, Kangxi, CJK Symbols and Punctuation
  [0x3041, 0x33ff], // Kana, Bopomofo, Hangul Compat Jamo, Kanbun, Enclosed CJK
  [0x3400, 0x4dbf], // CJK Unified Ideographs Extension A
  [0x4e00, 0x9fff], // CJK Unified Ideographs
  [0xa000, 0xa4cf], // Yi Syllables and Radicals
  [0xa960, 0xa97f], // Hangul Jamo Extended-A
  [0xac00, 0xd7a3], // Hangul Syllables
  [0xf900, 0xfaff], // CJK Compatibility Ideographs
  [0xfe30, 0xfe4f], // CJK Compatibility Forms
  [0xff00, 0xff60], // Fullwidth Forms
  [0xff66, 0xff9d], // Halfwidth Katakana — East Asian NARROW, but the
  [0xffa1, 0xffdc], // Halfwidth Hangul — same letters as the wide forms
  [0xffe0, 0xffe6], // Fullwidth signs
  [0x16fe0, 0x16fe4], // Ideographic Symbols and Punctuation
  [0x17000, 0x18aff], // Tangut
  [0x1aff0, 0x1b2ff], // Kana Extended-B/Supplement/Extended-A, Small Kana, Nushu
  [0x20000, 0x3fffd] // Planes 2 and 3 — every CJK extension, present and future
]

function isWideCharacter(character: string): boolean {
  const codePoint = character.codePointAt(0) ?? 0
  return WIDE_CODEPOINT_RANGES.some(([start, end]) => codePoint >= start && codePoint <= end)
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
