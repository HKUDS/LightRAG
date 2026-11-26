/// <reference types="bun" />
import { afterEach, beforeAll, describe, expect, test } from 'bun:test'
import { act, cleanup, screen } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import type { MessageWithError } from '@/types/retrieval'
import { ChatMessage } from './ChatMessage'

/**
 * The message list is ONE polite live region (MessageList), and the query
 * session's stopwatch republishes the messages array every 100ms. A ticking
 * time text inside that region would be announced ten times a second and
 * drown the streamed answer, so the timing row must be hidden from the a11y
 * tree while the query runs — and exposed with the final static time once it
 * ends, so finished and historical messages stay readable.
 */

const ANSWER = 'LightRAG is a graph-based retrieval framework.'

// ChatMessage loads its KaTeX plugins through a dynamic import on mount and
// then sets state. Warm those modules so the import resolves from cache, and
// flush the resulting state update inside act below — otherwise it lands in a
// LATER test (the DOM is process-wide) and React logs an act() warning there.
beforeAll(async () => {
  await import('katex/contrib/mhchem')
  await import('katex/contrib/copy-tex')
  await import('rehype-katex')
})

const flushKatex = (): Promise<void> => act(async () => {})

const answer = (responseTime: number): MessageWithError => ({
  id: 'a1',
  role: 'assistant',
  content: ANSWER,
  displayContent: ANSWER,
  responseTime
})

// Whether the timing row carrying `timeLabel` is hidden from the a11y tree.
// The label sits in a span whose closest div IS the timing row. Extract a
// boolean instead of asserting on the node: a failing DOM-node assertion
// makes Bun serialise the whole element and the run appears to hang. Both an
// absent attribute and aria-hidden="false" mean exposed, so only 'true'
// counts as hidden.
const timingRowHidden = (timeLabel: string): boolean => {
  const row = screen.getByText(timeLabel).closest('div')
  return row?.getAttribute('aria-hidden') === 'true'
}

afterEach(() => {
  cleanup()
})

describe('timing row inside the conversation live region', () => {
  test('the ticking stopwatch is hidden while querying, exposed with the final time', async () => {
    const { rerender } = renderWithProviders(
      <ChatMessage message={answer(1.2)} isQuerying={true} />
    )
    await flushKatex()

    // Hidden from the a11y tree only — getByText still finds the label, so
    // the stopwatch stays painted for sighted users while it ticks.
    expect(timingRowHidden('Response time 1.2s')).toBe(true)

    // The query ends: the same message settles on a DIFFERENT final time, so
    // the exposed-phase assertion cannot pass on the stale querying text.
    rerender(<ChatMessage message={answer(3.4)} isQuerying={false} />)
    await flushKatex()
    expect(timingRowHidden('Response time 3.4s')).toBe(false)
  })

  test('a historical message rendered without isQuerying keeps its time readable', async () => {
    // Restored conversations render ChatMessage without the prop (see
    // useQuerySession.test.tsx); the default must leave the row exposed.
    renderWithProviders(<ChatMessage message={answer(3.4)} />)
    await flushKatex()
    expect(timingRowHidden('Response time 3.4s')).toBe(false)
  })
})
