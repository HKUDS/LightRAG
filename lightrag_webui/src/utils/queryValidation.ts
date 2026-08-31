import type { QueryMode } from '@/api/lightrag'

export const MIN_RAG_QUERY_WEIGHT = 3

// Keep these ranges in sync with lightrag/query_validation.py.
const HAN_CODEPOINT_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0x3400, 0x4dbf],
  [0x4e00, 0x9fff],
  [0xf900, 0xfaff],
  [0x20000, 0x2a6df],
  [0x2a700, 0x2b73f],
  [0x2b740, 0x2b81f],
  [0x2b820, 0x2ceaf],
  [0x2ceb0, 0x2ebef],
  [0x2ebf0, 0x2ee5f],
  [0x2f800, 0x2fa1f],
  [0x30000, 0x3134f],
  [0x31350, 0x323af]
]

function isHanCharacter(character: string): boolean {
  const codePoint = character.codePointAt(0) ?? 0
  return (
    codePoint === 0x3007 ||
    HAN_CODEPOINT_RANGES.some(([start, end]) => codePoint >= start && codePoint <= end)
  )
}

export function ragQueryWeight(query: string): number {
  return Array.from(query.trim()).reduce(
    (weight, character) => weight + (isHanCharacter(character) ? 2 : 1),
    0
  )
}

export function isRagQueryTooShort(query: string, mode: QueryMode): boolean {
  return mode !== 'bypass' && ragQueryWeight(query) < MIN_RAG_QUERY_WEIGHT
}
