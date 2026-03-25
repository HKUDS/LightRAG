import { describe, expect, test } from 'bun:test'
import { pruneEmptyPromptOverrides, setExampleItem } from './promptOverrides'

describe('promptOverrides', () => {
  test('drops empty nested prompt override objects', () => {
    expect(pruneEmptyPromptOverrides({ query: { rag_response: '' } })).toBeUndefined()
  })

  test('returns explicit query-only structure when query values are present', () => {
    expect(
      pruneEmptyPromptOverrides({
        query: { rag_response: '  {context_data}  ', naive_query_context: '' }
      })
    ).toEqual({
      query: { rag_response: '{context_data}' }
    })
  })

  test('returns explicit keywords-only structure when only keywords values are present', () => {
    expect(
      pruneEmptyPromptOverrides({
        keywords: {
          keywords_extraction: '  {query}\n{examples}\n{language}  ',
          keywords_extraction_examples: ['  example-a  ', '']
        }
      })
    ).toEqual({
      keywords: {
        keywords_extraction: '{query}\n{examples}\n{language}',
        keywords_extraction_examples: ['example-a']
      }
    })
  })

  test('preserves query and keywords together when both contain values', () => {
    expect(
      pruneEmptyPromptOverrides({
        query: { kg_query_context: '{entities_str}' },
        keywords: { keywords_extraction_examples: ['example-a', 'example-b'] }
      })
    ).toEqual({
      query: { kg_query_context: '{entities_str}' },
      keywords: { keywords_extraction_examples: ['example-a', 'example-b'] }
    })
  })

  test('preserves list-style example fields as string arrays', () => {
    expect(setExampleItem(['A'], 1, 'B')).toEqual(['A', 'B'])
  })
})
