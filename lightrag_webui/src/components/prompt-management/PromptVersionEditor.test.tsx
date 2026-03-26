import React from 'react'
import { afterEach, describe, expect, test, vi } from 'vitest'
import { renderToString } from 'react-dom/server'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => key
  })
}))

afterEach(() => {
  vi.restoreAllMocks()
})

describe('PromptVersionEditor', () => {
  test('renders section descriptions and variable badges for prompt categories', async () => {
    const module = await import('./PromptVersionEditor')
    const PromptVersionEditor = module.default

    const html = renderToString(
      <PromptVersionEditor
        groupType="retrieval"
        version={{
          version_id: 'retrieval-seed',
          group_type: 'retrieval',
          version_name: 'retrieval-seed',
          version_number: 1,
          comment: '',
          created_at: '2026-03-25T00:00:00Z',
          payload: {
            query: {
              rag_response: 'Use {context_data}'
            }
          }
        }}
        versionsById={{}}
        activeVersionId={null}
        onSaveCurrentVersion={async () => {}}
        onSaveAsNewVersion={async () => {}}
        onActivateVersion={async () => {}}
        onDeleteVersion={async () => {}}
        onShowDiff={async () => {}}
        onRebuildFromVersion={async () => {}}
      />
    )

    expect(html).toContain('promptManagement.fieldHelp.ragResponse.description')
    expect(html).toContain('{context_data}')
    expect(html).toContain('{response_type}')
  })

  test('renders entity types as a full width textarea editor when expanded', async () => {
    const ReactModule = await import('react')
    const noop = vi.fn()
    vi.spyOn(ReactModule, 'useState')
      .mockImplementationOnce((() => ['indexing-seed-copy', noop]) as never)
      .mockImplementationOnce((() => ['', noop]) as never)
      .mockImplementationOnce((() => [{ entity_types: ['Person', 'Organization'] }, noop]) as never)
      .mockImplementationOnce((() => [false, noop]) as never)
      .mockImplementationOnce((() => ['entity_types', noop]) as never)

    const module = await import('./PromptVersionEditor')
    const PromptVersionEditor = module.default

    const html = renderToString(
      <PromptVersionEditor
        groupType="indexing"
        version={{
          version_id: 'indexing-seed',
          group_type: 'indexing',
          version_name: 'indexing-seed',
          version_number: 1,
          comment: '',
          created_at: '2026-03-26T00:00:00Z',
          payload: {
            entity_types: ['Person', 'Organization']
          }
        }}
        versionsById={{}}
        activeVersionId={null}
        onSaveCurrentVersion={async () => {}}
        onSaveAsNewVersion={async () => {}}
        onActivateVersion={async () => {}}
        onDeleteVersion={async () => {}}
        onShowDiff={async () => {}}
        onRebuildFromVersion={async () => {}}
      />
    )

    expect(html).toContain('<textarea')
    expect(html).toContain('Person,Organization')
    expect(html).toContain('min-h-[120px]')
  })

  test('renders save current, save as new, and indexing rebuild actions', async () => {
    const module = await import('./PromptVersionEditor')
    const PromptVersionEditor = module.default

    const html = renderToString(
      <PromptVersionEditor
        groupType="indexing"
        version={{
          version_id: 'indexing-seed',
          group_type: 'indexing',
          version_name: 'indexing-seed',
          version_number: 1,
          comment: 'seed',
          created_at: '2026-03-26T00:00:00Z',
          payload: {
            entity_types: ['Person']
          }
        }}
        versionsById={{}}
        activeVersionId={null}
        onSaveCurrentVersion={async () => {}}
        onSaveAsNewVersion={async () => {}}
        onActivateVersion={async () => {}}
        onDeleteVersion={async () => {}}
        onShowDiff={async () => {}}
        onRebuildFromVersion={async () => {}}
      />
    )

    expect(html).toContain('promptManagement.saveCurrentVersion')
    expect(html).toContain('promptManagement.saveAsNewVersion')
    expect(html).toContain('promptManagement.rebuildFromSelectedVersion')
  })
})
