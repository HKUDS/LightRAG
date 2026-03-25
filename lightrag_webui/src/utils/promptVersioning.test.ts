import { describe, expect, test } from 'vitest'

import {
  buildPromptEditorSections,
  formatVersionLineageLabel,
  getPromptFieldPreview,
  projectRetrievalVersionToOverrides
} from './promptVersioning'

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

  test('formatVersionLineageLabel falls back to Deleted when source metadata is missing', () => {
    expect(formatVersionLineageLabel({ source_version_id: 'missing' }, {})).toBe('Deleted')
  })

  test('buildPromptEditorSections returns indexing entity types first', () => {
    expect(buildPromptEditorSections('indexing')[0].key).toBe('entity_types')
  })

  test('getPromptFieldPreview summarizes long text and lists for collapsed editor rows', () => {
    expect(getPromptFieldPreview(['Person', 'Organization', 'Event'])).toBe('Person, Organization, Event')
    expect(getPromptFieldPreview('第一行\n第二行')).toBe('第一行')
  })
})
