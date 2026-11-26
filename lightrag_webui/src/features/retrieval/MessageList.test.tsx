/// <reference types="bun" />
import { afterEach, describe, expect, test } from 'bun:test'
import { cleanup, screen, within } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import type { MessageWithError } from '@/types/retrieval'
import MessageList from './MessageList'

/**
 * The conversation is a live region: a screen reader must announce appended
 * messages and streamed answer text without the user going looking for them.
 * ONE POLITE log region carries the whole conversation — assertive would
 * interrupt the user mid-sentence, and a region per message would announce
 * the same tokens twice. The region is the scroll container itself, so the
 * follow-scroll behavior keeps working on the same element.
 */

const USER_QUESTION = 'What is LightRAG?'
const STREAMED_ANSWER = 'LightRAG is a graph-based retrieval framework.'

const conversation: MessageWithError[] = [
  { id: 'u1', role: 'user', content: USER_QUESTION },
  {
    id: 'a1',
    role: 'assistant',
    content: STREAMED_ANSWER,
    displayContent: STREAMED_ANSWER
  }
]

afterEach(() => {
  cleanup()
})

describe('message list live region', () => {
  test('the conversation is one polite log region containing the streamed answer', () => {
    renderWithProviders(
      <MessageList messages={conversation} isLoading={true} queryProgress={null} />
    )

    const logs = screen.getAllByRole('log')
    expect(logs).toHaveLength(1)
    const log = logs[0]
    expect(log.getAttribute('aria-live')).toBe('polite')

    // Both sides of the conversation sit inside the region, so a new user
    // message and the streaming assistant answer are both announced.
    expect(within(log).getByText(USER_QUESTION)).toBeInTheDocument()
    expect(within(log).getByText(STREAMED_ANSWER)).toBeInTheDocument()

    // The log region IS the scroll container: vertical-only scrolling is
    // unchanged (happy-dom computes no layout, so the classes are the claim).
    const classes = (log.getAttribute('class') ?? '').split(/\s+/)
    expect(classes).toContain('overflow-y-auto')
    expect(classes).toContain('overflow-x-hidden')
  })

  test('the empty conversation keeps the region for the first answer', () => {
    renderWithProviders(
      <MessageList
        messages={[]}
        isLoading={false}
        queryProgress={null}
        emptyState={<div>Start a retrieval</div>}
      />
    )

    const log = screen.getByRole('log')
    expect(within(log).getByText('Start a retrieval')).toBeInTheDocument()
  })
})
