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
})
