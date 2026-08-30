import { describe, expect, test } from 'bun:test'

import { streamErrorChunk } from './streamErrorChunk'

describe('streamErrorChunk', () => {
  test('never repeats the answer already streamed into the message', () => {
    // Regression: the caller used to pass `content + '\n' + error` into an
    // APPEND-only updater, so a stream that failed midway rendered the partial
    // answer twice.
    const partial = 'LightRAG builds a knowledge graph'
    const chunk = streamErrorChunk(partial, 'upstream LLM connection reset')

    expect(chunk).not.toContain(partial)
    expect(partial + chunk).toBe(
      'LightRAG builds a knowledge graph\n\nupstream LLM connection reset'
    )
  })

  test('separates the error from the answer with a blank line', () => {
    // A single newline leaves the error glued to the last sentence once the
    // markdown is rendered.
    expect(streamErrorChunk('answer', 'boom')).toBe('\n\nboom')
  })

  test('a failure before any answer text is the whole message', () => {
    expect(streamErrorChunk('', 'boom')).toBe('boom')
  })

  test('no error means nothing to append', () => {
    expect(streamErrorChunk('answer', '')).toBe('')
  })
})
