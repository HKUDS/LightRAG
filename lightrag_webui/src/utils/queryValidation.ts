import type { QueryMode } from '@/api/lightrag'

export const MIN_RAG_QUERY_WEIGHT = 3

// East Asian scripts count as two English-equivalent characters. Han ideographs,
// Japanese kana and Korean Hangul are all included — doubling Han alone rejected
// ordinary two-character Japanese and Korean words ('ねこ', '오늘') while
// accepting their Han equivalents.
// Keep these ranges in sync with lightrag/query_validation.py.
const WIDE_CODEPOINT_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0x1100, 0x11ff], // Hangul Jamo
  [0x3040, 0x309f], // Hiragana
  [0x30a0, 0x30ff], // Katakana
  [0x3130, 0x318f], // Hangul Compatibility Jamo
  [0x31f0, 0x31ff], // Katakana Phonetic Extensions
  [0x3400, 0x4dbf], // CJK Unified Ideographs Extension A
  [0x4e00, 0x9fff], // CJK Unified Ideographs
  [0xa960, 0xa97f], // Hangul Jamo Extended-A
  [0xac00, 0xd7a3], // Hangul Syllables
  [0xd7b0, 0xd7ff], // Hangul Jamo Extended-B
  [0xf900, 0xfaff], // CJK Compatibility Ideographs
  [0x1b000, 0x1b16f], // Kana Supplement / Extended-A / Small Kana Extension
  [0x20000, 0x2a6df], // CJK Unified Ideographs Extension B
  [0x2a700, 0x2b73f], // Extension C
  [0x2b740, 0x2b81f], // Extension D
  [0x2b820, 0x2ceaf], // Extension E
  [0x2ceb0, 0x2ebef], // Extension F
  [0x2ebf0, 0x2ee5f], // Extension I
  [0x2f800, 0x2fa1f], // CJK Compatibility Ideographs Supplement
  [0x30000, 0x3134f], // Extension G
  [0x31350, 0x323af] // Extension H
]

function isWideCharacter(character: string): boolean {
  const codePoint = character.codePointAt(0) ?? 0
  return (
    codePoint === 0x3007 ||
    WIDE_CODEPOINT_RANGES.some(([start, end]) => codePoint >= start && codePoint <= end)
  )
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
