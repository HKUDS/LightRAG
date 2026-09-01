/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { prepareQueryInput } from './queryInput'
import { BYPASS_DEFAULT_HISTORY_TURNS, serializeQueryRequest } from './serializeQueryRequest'
import type { QuerySettings } from '@/stores/querySettings'
import type { MessageWithError } from '@/types/retrieval'

/**
 * `/bypass` reaches the LLM directly, and its usefulness depends entirely on
 * the conversation history travelling with it — a bypass turn with no history
 * is a one-shot prompt, not a chat. The workspace entry gained `/mode` prefix
 * parsing in #3776, so it can now issue `/bypass` too; this pins that it does
 * so through the SAME path as `/webui` rather than a second implementation
 * that could quietly drop the history.
 *
 * Both entries do exactly three things with an input, and they are replayed
 * here in that order:
 *   1. `prepareQueryInput(input, snapshot.mode)`
 *   2. apply `modeOverride` onto their settings snapshot
 *   3. `serializeQueryRequest(snapshot, query, ownHistory)`
 */

const sharedSettings: QuerySettings = {
  mode: 'mix',
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
  enable_rerank: true
}

/** `/webui` uses the shared settings verbatim. */
const adminSnapshot = (): QuerySettings => ({ ...sharedSettings })

/** The workspace clamps its two debug-only switches; nothing else differs. */
const workspaceSnapshot = (): QuerySettings => ({
  ...sharedSettings,
  only_need_context: false,
  only_need_prompt: false
})

const history: MessageWithError[] = [
  { id: '1', role: 'user', content: 'q1' },
  { id: '2', role: 'assistant', content: 'a1' },
  { id: '3', role: 'user', content: 'q2' },
  { id: '4', role: 'assistant', content: 'a2' },
  { id: '5', role: 'user', content: 'q3' },
  { id: '6', role: 'assistant', content: 'a3' },
  { id: '7', role: 'user', content: 'q4' },
  { id: '8', role: 'assistant', content: 'a4' }
]

/** The whole client-side send path for one entry, as the view runs it. */
function submit(buildSnapshot: () => QuerySettings, input: string) {
  const base = buildSnapshot()
  const prepared = prepareQueryInput(input, base.mode)
  if (!prepared.ok) return prepared

  const snapshot: QuerySettings = prepared.modeOverride
    ? { ...base, mode: prepared.modeOverride }
    : base

  return { ok: true as const, request: serializeQueryRequest(snapshot, prepared.query, history) }
}

describe('/bypass behaves identically on both query entries', () => {
  test('produces byte-identical request bodies', () => {
    const admin = submit(adminSnapshot, '/bypass summarise the thread')
    const workspace = submit(workspaceSnapshot, '/bypass summarise the thread')

    expect(admin).toEqual(workspace)
  })

  test('sends the conversation history the mode depends on', () => {
    const workspace = submit(workspaceSnapshot, '/bypass summarise the thread')
    expect(workspace.ok).toBe(true)
    if (!workspace.ok) return

    // history_turns is 0, so bypass expands its own default window.
    expect(workspace.request.mode).toBe('bypass')
    expect(workspace.request.query).toBe('summarise the thread')
    expect(workspace.request.conversation_history).toHaveLength(
      BYPASS_DEFAULT_HISTORY_TURNS * 2
    )
    expect(workspace.request.conversation_history).toEqual([
      { role: 'user', content: 'q2' },
      { role: 'assistant', content: 'a2' },
      { role: 'user', content: 'q3' },
      { role: 'assistant', content: 'a3' },
      { role: 'user', content: 'q4' },
      { role: 'assistant', content: 'a4' }
    ])
  })

  test('an explicit history_turns setting wins on both entries', () => {
    const withTurns = (build: () => QuerySettings) => () => ({
      ...build(),
      history_turns: 1
    })

    const admin = submit(withTurns(adminSnapshot), '/bypass and now?')
    const workspace = submit(withTurns(workspaceSnapshot), '/bypass and now?')

    expect(admin).toEqual(workspace)
    expect(admin.ok).toBe(true)
    if (!admin.ok) return
    expect(admin.request.conversation_history).toEqual([
      { role: 'user', content: 'q4' },
      { role: 'assistant', content: 'a4' }
    ])
  })

  test('the RAG minimum stays off for bypass on both entries', () => {
    expect(submit(adminSnapshot, '/bypass a')).toEqual(submit(workspaceSnapshot, '/bypass a'))
    expect(submit(workspaceSnapshot, '/bypass a').ok).toBe(true)
  })

  test('an empty bypass query is refused on both entries', () => {
    expect(submit(adminSnapshot, '/bypass  ')).toEqual({ ok: false, error: 'queryEmpty' })
    expect(submit(workspaceSnapshot, '/bypass  ')).toEqual({ ok: false, error: 'queryEmpty' })
  })
})
