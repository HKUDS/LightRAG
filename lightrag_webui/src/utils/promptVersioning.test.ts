import { describe, expect, test } from 'vitest'

import {
  buildPromptEditorSections,
  formatVersionLineageLabel,
  getPromptFieldEditorValue,
  getPromptFieldPreview,
  getPreferredPromptVersionId,
  projectRetrievalVersionToOverrides
} from './promptVersioning'

describe('promptVersioning', () => {
  test('buildPromptEditorSections exposes help metadata for descriptions and variables', () => {
    const keywordsExamplesSection = buildPromptEditorSections('retrieval').find(
      (section) => section.key === 'keywords.keywords_extraction_examples'
    )

    expect(keywordsExamplesSection).toMatchObject({
      helpKey: 'keywordsExtractionExamples',
      itemPlaceholderKey: 'promptManagement.fieldHelp.keywordsExtractionExamples.itemPlaceholder'
    })
    expect(keywordsExamplesSection?.variables.map((variable) => variable.label)).toEqual([
      '{examples}'
    ])
  })

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

  test('formats entity_types arrays as comma separated editor text', () => {
    const entityTypesSection = buildPromptEditorSections('indexing')[0]

    expect(entityTypesSection.type).toBe('csv')
    expect(getPromptFieldEditorValue(entityTypesSection, ['Person', 'Organization'])).toBe(
      'Person,Organization'
    )
  })

  test('prefers the locale matching default version when no active version is set', () => {
    expect(getPreferredPromptVersionId({
      groupType: 'retrieval',
      locale: 'zh',
      activeVersionId: null,
      selectedVersionId: null,
      versions: [
        {
          version_id: 'retrieval-en-default-v1',
          group_type: 'retrieval',
          version_name: 'retrieval-en-default',
          version_number: 1,
          comment: '',
          created_at: '2026-03-26T00:00:00Z',
          payload: {}
        },
        {
          version_id: 'retrieval-zh-default-v1',
          group_type: 'retrieval',
          version_name: 'retrieval-zh-default',
          version_number: 1,
          comment: '',
          created_at: '2026-03-26T00:00:00Z',
          payload: {}
        }
      ]
    })).toBe('retrieval-zh-default-v1')
  })

  test('replaces a stale locale-mismatched default selection when no active version is set', () => {
    expect(getPreferredPromptVersionId({
      groupType: 'indexing',
      locale: 'zh',
      activeVersionId: null,
      selectedVersionId: 'indexing-en-default-v1',
      versions: [
        {
          version_id: 'indexing-en-default-v1',
          group_type: 'indexing',
          version_name: 'indexing-en-default',
          version_number: 1,
          comment: '',
          created_at: '2026-03-26T00:00:00Z',
          payload: {}
        },
        {
          version_id: 'indexing-zh-default-v1',
          group_type: 'indexing',
          version_name: 'indexing-zh-default',
          version_number: 1,
          comment: '',
          created_at: '2026-03-26T00:00:00Z',
          payload: {}
        }
      ]
    })).toBe('indexing-zh-default-v1')
  })

  test('prefers the locale matching default version over an active mismatched seed on initial load', () => {
    expect(getPreferredPromptVersionId({
      groupType: 'retrieval',
      locale: 'zh',
      activeVersionId: 'retrieval-en-default-v1',
      selectionMode: 'automatic',
      selectedVersionId: null,
      versions: [
        {
          version_id: 'retrieval-en-default-v1',
          group_type: 'retrieval',
          version_name: 'retrieval-en-default',
          version_number: 1,
          comment: '',
          created_at: '2026-03-26T00:00:00Z',
          payload: {}
        },
        {
          version_id: 'retrieval-zh-default-v1',
          group_type: 'retrieval',
          version_name: 'retrieval-zh-default',
          version_number: 1,
          comment: '',
          created_at: '2026-03-26T00:00:00Z',
          payload: {}
        }
      ]
    })).toBe('retrieval-zh-default-v1')
  })

  test('keeps an explicitly selected locale-mismatched seed version in manual mode', () => {
    expect(getPreferredPromptVersionId({
      groupType: 'retrieval',
      locale: 'zh',
      activeVersionId: null,
      selectedVersionId: 'retrieval-en-default-v1',
      selectionMode: 'manual',
      versions: [
        {
          version_id: 'retrieval-en-default-v1',
          group_type: 'retrieval',
          version_name: 'retrieval-en-default',
          version_number: 1,
          comment: '',
          created_at: '2026-03-26T00:00:00Z',
          payload: {}
        },
        {
          version_id: 'retrieval-zh-default-v1',
          group_type: 'retrieval',
          version_name: 'retrieval-zh-default',
          version_number: 1,
          comment: '',
          created_at: '2026-03-26T00:00:00Z',
          payload: {}
        }
      ]
    })).toBe('retrieval-en-default-v1')
  })

  test('getPromptFieldPreview summarizes long text and lists for collapsed editor rows', () => {
    expect(getPromptFieldPreview(['Person', 'Organization', 'Event'])).toBe('Person, Organization, Event')
    expect(getPromptFieldPreview('第一行\n第二行')).toBe('第一行')
  })
})
