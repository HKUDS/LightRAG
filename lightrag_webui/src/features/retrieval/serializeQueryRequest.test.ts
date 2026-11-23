import { describe, expect, test } from 'bun:test'
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

  // The snapshot is an INPUT: the serializer performs no clamping, so a
  // snapshot that carries a value keeps it in every mode (the ADMIN page passes
  // its snapshot unmodified; the workspace page clamps before calling). This
  // replaces a source scan for the substring `only_need`. The two switches take
  // UNEQUAL values too, so copying one into the other is caught; a branch keyed
  // on something other than the mode (the question, the history) would escape.
  test.each(
    ALL_MODES.flatMap((mode) =>
      [[true, true], [true, false], [false, true], [false, false]].map(
        ([context, prompt]) => [mode, `context=${context} prompt=${prompt}`, context, prompt] as const
      )
    )
  )(
    'mode=%s: only_need_* (%s) pass through unmodified',
    (mode, _label, context, prompt) => {
      const settings = {
        ...baseSettings(mode),
        only_need_context: context,
        only_need_prompt: prompt
      }
      const body = serializeQueryRequest(settings, 'q', historyFixture)
      expect({ context: body.only_need_context, prompt: body.only_need_prompt }).toEqual({
        context,
        prompt
      })
    }
  )

  test('response_type and include_progress are set uniformly', () => {
    const body = serializeQueryRequest(baseSettings('mix'), 'q', [])
    expect(body.response_type).toBe('Multiple Paragraphs')
    expect(body.include_progress).toBe(true)
    expect(body.query).toBe('q')
    // history_turns is a client-side expansion knob the server never
    // declared: it must be stripped from every request body.
    expect('history_turns' in body).toBe(false)
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

describe('conversation history turns', () => {
  const bypass = (history: MessageWithError[], historyTurns = 0) =>
    serializeQueryRequest(
      { ...baseSettings('bypass'), history_turns: historyTurns },
      'next',
      history
    ).conversation_history

  const turn = (n: number, answer: Partial<MessageWithError> = {}): MessageWithError[] => [
    { id: `u${n}`, role: 'user', content: `q${n}` },
    { id: `a${n}`, role: 'assistant', content: `a${n}`, ...answer }
  ]

  test.each([
    ['a failed', { isError: true }],
    ['a stopped', { isAborted: true }],
    ['an empty', { content: '' }],
    ['a whitespace-only', { content: '  \n' }]
  ] as const)('%s answer drops its whole turn, question included', (_label, answer) => {
    expect(bypass([...turn(1), ...turn(2, answer), ...turn(3)])).toEqual([
      { role: 'user', content: 'q1' },
      { role: 'assistant', content: 'a1' },
      { role: 'user', content: 'q3' },
      { role: 'assistant', content: 'a3' }
    ])
  })

  test('dropped turns do not consume the window: it still holds N completed turns', () => {
    const history = [
      ...turn(1),
      ...turn(2),
      ...turn(3, { isError: true }),
      ...turn(4, { isAborted: true }),
      ...turn(5)
    ]
    expect(bypass(history, 2)).toEqual([
      { role: 'user', content: 'q2' },
      { role: 'assistant', content: 'a2' },
      { role: 'user', content: 'q5' },
      { role: 'assistant', content: 'a5' }
    ])
  })

  test('messages outside a question/answer pair are dropped, so the window alternates', () => {
    const history: MessageWithError[] = [
      { id: 'a0', role: 'assistant', content: 'orphan answer' },
      ...turn(1),
      { id: 's', role: 'system', content: 'Error loading message', isError: true },
      { id: 'u-lone', role: 'user', content: 'unanswered' },
      ...turn(2)
    ]
    expect(bypass(history, 1)).toEqual([
      { role: 'user', content: 'q2' },
      { role: 'assistant', content: 'a2' }
    ])
    expect(bypass(history)?.map((m) => m.role)).toEqual(['user', 'assistant', 'user', 'assistant'])
  })

  test('a partial answer with only unfinished reasoning is dropped once stopped', () => {
    expect(bypass(turn(1, { content: '<think>half a thought', isAborted: true }))).toEqual([])
  })
})
