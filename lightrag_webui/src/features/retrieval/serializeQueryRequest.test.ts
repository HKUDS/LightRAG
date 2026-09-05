import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'
import { serializeQueryRequest } from './serializeQueryRequest'
import type { QuerySettings } from '@/stores/querySettings'
import type { MessageWithError } from '@/types/retrieval'

const ALL_MODES = ['naive', 'local', 'global', 'hybrid', 'mix', 'bypass'] as const

const baseSettings = (mode: (typeof ALL_MODES)[number]): QuerySettings => ({
  mode,
  top_k: 40,
  chunk_top_k: 20,
  max_entity_tokens: 6000,
  max_relation_tokens: 8000,
  max_total_tokens: 30000,
  only_need_context: false,
  only_need_prompt: false,
  stream: true,
  history_turns: 0,
  user_prompt: 'be concise',
  disable_user_prompt_prefix: false,
  enable_rerank: true
})

// The explicit history fixture is REQUIRED — the bypass case's request body
// is sliced from it.
const historyFixture: MessageWithError[] = [
  { id: '1', role: 'user', content: 'q1' },
  { id: '2', role: 'assistant', content: 'a1' },
  { id: '3', role: 'user', content: 'broken', isError: true },
  { id: '4', role: 'user', content: 'q2' },
  { id: '5', role: 'assistant', content: 'a2' },
  { id: '6', role: 'user', content: 'q3' },
  { id: '7', role: 'assistant', content: 'a3' },
  { id: '8', role: 'user', content: 'q4' },
  { id: '9', role: 'assistant', content: 'a4' }
]

describe('serializer equivalence', () => {
  test.each([...ALL_MODES])(
    'mode=%s: same settings + same explicit history + same question ⇒ field-identical bodies',
    (mode) => {
      const a = serializeQueryRequest(baseSettings(mode), 'what is RAG', historyFixture)
      const b = serializeQueryRequest(baseSettings(mode), 'what is RAG', historyFixture)
      expect(a).toEqual(b)
      // history_turns is a client-side expansion knob the server never
      // declared: it must be stripped from every request body.
      expect('history_turns' in a).toBe(false)
    }
  )

  test('non-bypass modes send an empty conversation_history (history_turns=0)', () => {
    for (const mode of ALL_MODES.filter((m) => m !== 'bypass')) {
      const body = serializeQueryRequest(baseSettings(mode), 'q', historyFixture)
      expect(body.conversation_history).toEqual([])
    }
  })

  test('bypass with history_turns=0 takes the last 3 turns, excluding error messages', () => {
    const body = serializeQueryRequest(baseSettings('bypass'), 'q', historyFixture)
    expect(body.conversation_history).toEqual([
      { role: 'user', content: 'q2' },
      { role: 'assistant', content: 'a2' },
      { role: 'user', content: 'q3' },
      { role: 'assistant', content: 'a3' },
      { role: 'user', content: 'q4' },
      { role: 'assistant', content: 'a4' }
    ])
  })

  test('the snapshot is an INPUT: only_need_* pass through unmodified (clamping lives in the page layer)', () => {
    const settings = {
      ...baseSettings('mix'),
      only_need_context: true,
      only_need_prompt: true
    }
    const body = serializeQueryRequest(settings, 'q', [])
    // The serializer itself performs no clamping — a snapshot that carries
    // true keeps true (the ADMIN page passes its snapshot unmodified; the
    // workspace page clamps before calling).
    expect(body.only_need_context).toBe(true)
    expect(body.only_need_prompt).toBe(true)
  })

  test('response_type and include_progress are set uniformly', () => {
    const body = serializeQueryRequest(baseSettings('mix'), 'q', [])
    expect(body.response_type).toBe('Multiple Paragraphs')
    expect(body.include_progress).toBe(true)
    expect(body.query).toBe('q')
  })
})

describe('serializer source purity', () => {
  test('the shared serializer contains no only_need_* branch (clamping must stay in the composition layer)', () => {
    // If the serializer grew a branch on these fields, the workspace could
    // bypass the equivalence contract while every behavioral test still
    // passes — so pin it at the source level.
    const source = readFileSync(join(import.meta.dir, 'serializeQueryRequest.ts'), 'utf8')
    expect(source.includes('only_need')).toBe(false)
  })
})

describe('disable_user_prompt_prefix', () => {
  test.each([true, false])('forwards the switch verbatim (%s)', (value) => {
    const body = serializeQueryRequest(
      { ...baseSettings('hybrid'), disable_user_prompt_prefix: value },
      'what is RAG',
      historyFixture
    )
    expect(body.disable_user_prompt_prefix).toBe(value)
  })

  test('an unset switch is omitted from the body entirely', () => {
    // The upgrade path: settings persisted before this field existed hydrate
    // with it undefined. JSON.stringify drops undefined, the server field is
    // Optional[bool]=None, exclude_none drops it, and QueryParam supplies
    // False — so the prefix applies, which is the right default for an
    // existing install.
    const settings = { ...baseSettings('hybrid') }
    delete (settings as Partial<QuerySettings>).disable_user_prompt_prefix

    const body = serializeQueryRequest(settings, 'what is RAG', historyFixture)
    expect('disable_user_prompt_prefix' in JSON.parse(JSON.stringify(body))).toBe(false)
  })
})
