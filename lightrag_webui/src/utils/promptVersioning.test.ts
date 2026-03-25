import { describe, expect, test } from 'vitest'

import { projectRetrievalVersionToOverrides } from './promptVersioning'

describe('promptVersioning', () => {
  test('projects retrieval payload to query-time overrides only', () => {
    expect(projectRetrievalVersionToOverrides({
      query: { rag_response: '{context_data}' },
      keywords: { keywords_extraction: '{query}' },
      entity_extraction: { system_prompt: 'ignored' }
    })).toEqual({
      query: { rag_response: '{context_data}' },
      keywords: { keywords_extraction: '{query}' }
    })
  })
})
