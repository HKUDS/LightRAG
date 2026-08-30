import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'
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

describe('MessageList follow-scroll wiring', () => {
  test('the follow effect is keyed on the content, not only on the debounced array', () => {
    // Behavioral coverage would need a DOM this suite does not have, and the
    // failure is invisible to the pure key above: the regression was that
    // the ONLY per-content trigger keyed on `debouncedMessages`, which the
    // response timer starves forever. Pin the wiring at the source level.
    const source = readFileSync(join(import.meta.dir, 'MessageList.tsx'), 'utf8')
    expect(source.includes('streamedContentKey(messages)')).toBe(true)
    expect(source.includes('[contentKey, scrollToBottom]')).toBe(true)
  })
})
