/// <reference types="bun" />
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { act, cleanup } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import MessageList from './MessageList'
import { streamedContentKey } from './streamedContentKey'
import type { MessageWithError } from '@/types/retrieval'

/**
 * The scroll-follow trigger during streaming. Its whole job is to be immune
 * to `useQuerySession`'s response timer — which replaces the `messages` array
 * every 100 ms while a query is in flight, touching only `responseTime`.
 *
 * The regression this pins: keying the follow effect on the array meant a
 * 150 ms debounce restarted every 100 ms and never elapsed, so a long
 * streamed answer scrolled off-screen and the view only jumped to the bottom
 * after the query completed.
 */

const message = (over: Partial<MessageWithError> = {}): MessageWithError =>
  ({
    id: 'a1',
    role: 'assistant',
    content: 'hello',
    ...over
  }) as MessageWithError

describe('streamedContentKey', () => {
  test('a responseTime-only tick leaves the key UNCHANGED', () => {
    const before = streamedContentKey([message({ responseTime: 0.1 })])
    const after = streamedContentKey([message({ responseTime: 12.9 })])

    expect(after).toBe(before)
  })

  test('an arriving content chunk changes the key', () => {
    const before = streamedContentKey([message({ content: 'hel' })])
    const after = streamedContentKey([message({ content: 'hello' })])

    expect(after).not.toBe(before)
  })

  test('content revealed only through displayContent changes the key', () => {
    // While a <think> block is open the visible text lives in
    // displayContent/thinkingContent, and `content` can be identical across
    // two renders — following `content` alone would stall the scroll there.
    const raw = '<think>reasoning'
    const before = streamedContentKey([
      message({ content: raw, displayContent: '', thinkingContent: 'reason' })
    ])
    const after = streamedContentKey([
      message({ content: raw, displayContent: '', thinkingContent: 'reasoning' })
    ])

    expect(after).not.toBe(before)
  })

  test('a newly appended message changes the key', () => {
    const before = streamedContentKey([message()])
    const after = streamedContentKey([
      message(),
      message({ id: 'a2', content: 'hello' })
    ])

    expect(after).not.toBe(before)
  })

  test('an empty list is a stable key, not a crash', () => {
    expect(streamedContentKey([])).toBe(streamedContentKey([]))
  })
})

/**
 * The wiring, exercised rather than read. Every rerender below happens well
 * inside the component's own 150 ms debounce window, which is the point: the
 * regression was a follow effect that ONLY keyed on the debounced array, and
 * the response timer's 100 ms tick starved that debounce forever. A test that
 * waited for the debounce to elapse would pass against the broken version.
 */

/** Element.prototype is process-wide, so the original is put back in afterEach. */
let scrollIntoViewCalls = 0
let originalScrollIntoView: Element['scrollIntoView']

const streaming = (
  content: string,
  responseTime: number,
  over: Partial<MessageWithError> = {}
): MessageWithError[] =>
  [
    { id: 'u1', role: 'user', content: 'question' },
    { id: 'a1', role: 'assistant', content, responseTime, ...over }
  ] as MessageWithError[]

const list = (messages: MessageWithError[]) => (
  <MessageList messages={messages} isLoading={true} queryProgress={null} isActive={true} />
)

/**
 * Flush the effects and their `requestAnimationFrame`, while staying under the
 * 150 ms debounce — so anything that scrolls here did so on the content key.
 */
const settle = async (): Promise<void> => {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 20))
  })
}

beforeEach(() => {
  originalScrollIntoView = Element.prototype.scrollIntoView
  scrollIntoViewCalls = 0
  Element.prototype.scrollIntoView = function scrollIntoView() {
    scrollIntoViewCalls += 1
  }
})

afterEach(() => {
  // Unmount before restoring: a pending rAF from the component would otherwise
  // land on the real implementation mid-teardown.
  cleanup()
  Element.prototype.scrollIntoView = originalScrollIntoView
})

describe('MessageList follow-scroll', () => {
  test('follows an arriving chunk while the response timer keeps ticking', async () => {
    const { rerender } = renderWithProviders(list(streaming('hel', 0.1)))
    await settle()
    const afterMount = scrollIntoViewCalls

    rerender(list(streaming('hello world', 0.2)))
    await settle()

    expect(scrollIntoViewCalls).toBeGreaterThan(afterMount)
  })

  test('a responseTime-only tick does not scroll', async () => {
    const { rerender } = renderWithProviders(list(streaming('hel', 0.1)))
    await settle()
    const afterMount = scrollIntoViewCalls

    // A new array every time, exactly as the timer produces it — and nothing
    // the user can see has changed. Scrolling here would fight a user who has
    // deliberately scrolled up mid-answer.
    for (const responseTime of [0.2, 0.3, 0.4]) {
      rerender(list(streaming('hel', responseTime)))
      await settle()
    }

    expect(scrollIntoViewCalls).toBe(afterMount)
  })

  test('follows text revealed only inside an open think block', async () => {
    // `content` is byte-identical across these two renders; only the parsed
    // thinking text grows. Following `content` alone would stall here.
    const raw = '<think>reasoning'
    const { rerender } = renderWithProviders(
      list(streaming(raw, 0.1, { displayContent: '', thinkingContent: 'reason' }))
    )
    await settle()
    const afterMount = scrollIntoViewCalls

    rerender(list(streaming(raw, 0.2, { displayContent: '', thinkingContent: 'reasoning' })))
    await settle()

    expect(scrollIntoViewCalls).toBeGreaterThan(afterMount)
  })
})
