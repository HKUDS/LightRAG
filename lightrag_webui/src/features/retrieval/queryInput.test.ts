import { describe, expect, test } from 'bun:test'
import { prepareQueryInput, SUPPORTED_QUERY_MODES } from './queryInput'

describe('prepareQueryInput', () => {
  test('keeps a plain query and uses the configured mode', () => {
    expect(prepareQueryInput('What is LightRAG?', 'mix')).toEqual({
      ok: true,
      query: 'What is LightRAG?',
      modeOverride: undefined
    })
  })

  test('extracts every supported mode prefix', () => {
    for (const mode of SUPPORTED_QUERY_MODES) {
      expect(prepareQueryInput(`/${mode} explain`, 'mix')).toEqual({
        ok: true,
        query: 'explain',
        modeOverride: mode
      })
    }
  })

  test('rejects a prefix without a space and query', () => {
    expect(prepareQueryInput('/mix', 'mix')).toEqual({
      ok: false,
      error: 'queryModePrefixInvalid'
    })
  })

  test('rejects an unsupported query mode', () => {
    expect(prepareQueryInput('/unknown explain', 'mix')).toEqual({
      ok: false,
      error: 'queryModeError'
    })
  })

  test('validates the prefix-stripped RAG query', () => {
    expect(prepareQueryInput('/local a', 'bypass')).toEqual({
      ok: false,
      error: 'queryTooShort'
    })
  })

  test('allows short queries in the effective bypass mode', () => {
    expect(prepareQueryInput('/bypass a', 'mix')).toEqual({
      ok: true,
      query: 'a',
      modeOverride: 'bypass'
    })
    expect(prepareQueryInput('a', 'bypass')).toEqual({
      ok: true,
      query: 'a',
      modeOverride: undefined
    })
  })

  test('accepts two-character Japanese and Korean queries', () => {
    // Weighing only Han as two rejected these while accepting '猫' and '今日'.
    for (const query of ['ねこ', '한글', '오늘']) {
      expect(prepareQueryInput(query, 'mix')).toEqual({
        ok: true,
        query,
        modeOverride: undefined
      })
    }
  })

  test('rejects an empty query in every mode, bypass included', () => {
    // A prefix can consume the whole input, so the effective query can be
    // blank even when the raw input is not.
    expect(prepareQueryInput('/bypass  ', 'mix')).toEqual({
      ok: false,
      error: 'queryEmpty'
    })
    expect(prepareQueryInput('   ', 'bypass')).toEqual({
      ok: false,
      error: 'queryEmpty'
    })
    expect(prepareQueryInput('   ', 'mix')).toEqual({
      ok: false,
      error: 'queryEmpty'
    })
  })
})
