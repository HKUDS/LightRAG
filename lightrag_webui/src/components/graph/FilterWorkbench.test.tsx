import React from 'react'
import { beforeEach, describe, expect, test } from 'vitest'
import { renderToString } from 'react-dom/server'

import i18n from '@/i18n'
import type { GraphWorkbenchQueryRequest } from '@/api/lightrag'
import en from '@/locales/en.json'
import zh from '@/locales/zh.json'
import { useGraphWorkbenchStore, getDefaultGraphWorkbenchFilterDraft } from '@/stores/graphWorkbench'
import {
  FilterWorkbench,
  applyWorkbenchFilters,
  resetWorkbenchFilters,
  updateDraftFromInput
} from './FilterWorkbench'
import GraphWorkbenchSummary from './GraphWorkbenchSummary'

const cloneDraft = (draft: GraphWorkbenchQueryRequest): GraphWorkbenchQueryRequest =>
  JSON.parse(JSON.stringify(draft))

const getValueAtPath = (obj: Record<string, unknown>, path: string): unknown => {
  return path.split('.').reduce<unknown>((current, segment) => {
    if (current && typeof current === 'object') {
      return (current as Record<string, unknown>)[segment]
    }
    return undefined
  }, obj)
}

describe('FilterWorkbench', () => {
  beforeEach(() => {
    void i18n.changeLanguage('en')
    useGraphWorkbenchStore.getState().reset()
  })

  test('可渲染五类筛选区块', () => {
    const html = renderToString(<FilterWorkbench />)

    expect(html).toContain('Node Filters')
    expect(html).toContain('Edge Filters')
    expect(html).toContain('Scope Filters')
    expect(html).toContain('Source Filters')
    expect(html).toContain('View Controls')
  })

  test('apply / reset 行为会更新 appliedQuery', () => {
    const store = useGraphWorkbenchStore.getState()
    const beforeVersion = store.queryVersion
    const draft = getDefaultGraphWorkbenchFilterDraft()
    draft.scope.label = 'Tesla'
    draft.scope.max_depth = 2
    draft.node_filters.entity_types = ['ORGANIZATION']
    draft.view_options.highlight_matches = true
    store.setFilterDraft(draft)

    applyWorkbenchFilters()
    const applied = useGraphWorkbenchStore.getState().appliedQuery
    expect(applied?.scope.label).toBe('Tesla')
    expect(applied?.node_filters.entity_types).toEqual(['ORGANIZATION'])
    expect(useGraphWorkbenchStore.getState().queryVersion).toBe(beforeVersion + 1)

    resetWorkbenchFilters()
    const state = useGraphWorkbenchStore.getState()
    const defaults = getDefaultGraphWorkbenchFilterDraft()
    expect(state.appliedQuery).toEqual(defaults)
    expect(state.filterDraft).toEqual(defaults)
    expect(state.queryVersion).toBe(beforeVersion + 2)
  })

  test('summary metadata 展示 applied 状态与统计信息', () => {
    const draft = getDefaultGraphWorkbenchFilterDraft()
    draft.scope.label = 'OpenAI'
    draft.scope.max_depth = 4

    const applied = cloneDraft(draft)
    applied.node_filters.entity_types = ['ORGANIZATION']
    applied.edge_filters.relation_types = ['cooperate']
    applied.view_options.highlight_matches = true

    const html = renderToString(
      <GraphWorkbenchSummary
        draft={draft}
        appliedQuery={applied}
        queryVersion={3}
        nodeCount={12}
        edgeCount={18}
      />
    )
    const normalizedHtml = html.replaceAll('<!-- -->', '')

    expect(normalizedHtml).toContain('Applied')
    expect(normalizedHtml).toContain('Version 3')
    expect(normalizedHtml).toContain('Scope OpenAI · D4 · N1000')
    expect(normalizedHtml).toContain('Result 12 nodes / 18 edges')
    expect(normalizedHtml).toContain('Active Groups 3')
  })

  test('输入变化会驱动 structured payload 更新', () => {
    const draft = getDefaultGraphWorkbenchFilterDraft()

    const withEntityTypes = updateDraftFromInput(
      draft,
      'node_filters',
      'entity_types',
      'PERSON, ORGANIZATION'
    )
    expect(withEntityTypes.node_filters.entity_types).toEqual(['PERSON', 'ORGANIZATION'])

    const withDepth = updateDraftFromInput(withEntityTypes, 'scope', 'max_depth', '5')
    expect(withDepth.scope.max_depth).toBe(5)

    const withWeight = updateDraftFromInput(withDepth, 'edge_filters', 'weight_min', '0.75')
    expect(withWeight.edge_filters.weight_min).toBe(0.75)

    const withTime = updateDraftFromInput(
      withWeight,
      'source_filters',
      'time_from',
      '2026-03-25T09:30'
    )
    expect(withTime.source_filters.time_from).toBe('2026-03-25T09:30')

    const withToggle = updateDraftFromInput(withTime, 'view_options', 'highlight_matches', true)
    expect(withToggle.view_options.highlight_matches).toBe(true)

    const clearedWeight = updateDraftFromInput(withToggle, 'edge_filters', 'weight_min', '')
    expect(clearedWeight.edge_filters.weight_min).toBeNull()

    const keptDepth = updateDraftFromInput(withToggle, 'scope', 'max_depth', '')
    expect(keptDepth.scope.max_depth).toBe(withToggle.scope.max_depth)

    const clampedNodes = updateDraftFromInput(withToggle, 'scope', 'max_nodes', '-1')
    expect(clampedNodes.scope.max_nodes).toBe(1)
  })

  test('graph workbench 关键 i18n key 在 en 与 zh 中存在', () => {
    const keyPaths = [
      'graphPanel.workbench.summary.draftStatus',
      'graphPanel.workbench.summary.appliedStatus',
      'graphPanel.workbench.filter.sections.nodeFilters',
      'graphPanel.workbench.filter.actions.apply',
      'graphPanel.workbench.actionInspector.title',
      'graphPanel.workbench.actionInspector.tabs.merge',
      'graphPanel.workbench.merge.manual.title',
      'graphPanel.workbench.merge.suggestions.title'
    ]

    keyPaths.forEach((path) => {
      expect(getValueAtPath(en as Record<string, unknown>, path)).toBeTruthy()
      expect(getValueAtPath(zh as Record<string, unknown>, path)).toBeTruthy()
    })
  })
})
