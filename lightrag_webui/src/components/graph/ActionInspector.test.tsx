import React from 'react'
import { beforeEach, describe, expect, test, vi } from 'vitest'
import { renderToString } from 'react-dom/server'
import {
  getDefaultGraphWorkbenchFilterDraft,
  normalizeWorkbenchMutationError,
  useGraphWorkbenchStore
} from '@/stores/graphWorkbench'
import type { GraphMergeSuggestionCandidate } from '@/api/lightrag'

Object.defineProperty(globalThis, 'localStorage', {
  value: {
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {}
  },
  configurable: true
})

vi.mock('@/hooks/useLightragGraph', () => ({
  default: () => ({
    getNode: () => null,
    getEdge: () => null
  })
}))

vi.mock('./PropertiesView', () => ({
  default: () => <div>Mock PropertiesView</div>
}))

type ActionInspectorSelection = {
  kind: 'node' | 'edge'
  node?: any
  edge?: any
}

const nodeSelection: ActionInspectorSelection = {
  kind: 'node',
  node: {
    id: 'n-1',
    labels: ['Elon Musk'],
    properties: {
      entity_id: 'Elon Musk',
      description: 'CEO'
    },
    size: 10,
    x: 0,
    y: 0,
    color: '#000000',
    degree: 2
  }
}

const relationSelection: ActionInspectorSelection = {
  kind: 'edge',
  edge: {
    id: 'r-1',
    source: 'Elon Musk',
    target: 'Tesla',
    type: 'works_for',
    properties: {
      description: 'Elon Musk works for Tesla',
      keywords: 'works_for'
    },
    dynamicId: 'r-1-d',
    revision_token: 'edge-token-1'
  }
}

describe('ActionInspector', () => {
  beforeEach(() => {
    useGraphWorkbenchStore.getState().reset()
  })

  test('Inspect / Create / Delete / merge 四个 tab 可切换', async () => {
    const { ActionInspector, resolveActionInspectorTab } = await import('./ActionInspector')

    const inspectHtml = renderToString(
      <ActionInspector
        initialTab="inspect"
        selection={nodeSelection as any}
        inspectPane={<div>Inspect Pane</div>}
      />
    )
    expect(inspectHtml).toContain('Inspect Pane')

    const createTab = resolveActionInspectorTab('inspect', 'create')
    const createHtml = renderToString(
      <ActionInspector
        initialTab={createTab}
        selection={nodeSelection as any}
        inspectPane={<div>Inspect Pane</div>}
      />
    )
    expect(createHtml).toContain('Create Node')
    expect(createHtml).toContain('Create Relation')

    const deleteTab = resolveActionInspectorTab(createTab, 'delete')
    const deleteHtml = renderToString(
      <ActionInspector
        initialTab={deleteTab}
        selection={relationSelection as any}
        inspectPane={<div>Inspect Pane</div>}
      />
    )
    expect(deleteHtml).toContain('Delete Selection')

    const mergeTab = resolveActionInspectorTab(deleteTab, 'merge')
    const mergeHtml = renderToString(
      <ActionInspector
        initialTab={mergeTab}
        selection={relationSelection as any}
        inspectPane={<div>Inspect Pane</div>}
      />
    )
    expect(mergeHtml).toContain('Manual Merge')
    expect(mergeHtml).toContain('Merge Suggestions')
  })

  test('Create Relation 从当前选中节点预填 source', async () => {
    const { deriveCreateRelationDraftFromSelection } = await import('./CreateRelationForm')
    const draft = deriveCreateRelationDraftFromSelection(nodeSelection as any)
    expect(draft.sourceEntity).toBe('Elon Musk')
    expect(draft.targetEntity).toBe('')
  })

  test('delete confirmation copy 与错误保留', async () => {
    const { buildDeleteConfirmationCopy, reduceDeletePanelStateAfterFailure } = await import(
      './DeleteGraphObjectPanel'
    )
    const copy = buildDeleteConfirmationCopy(relationSelection as any)
    expect(copy).toContain('Elon Musk')
    expect(copy).toContain('Tesla')
    expect(copy).toContain('works_for')

    const next = reduceDeletePanelStateAfterFailure(
      { confirmationInput: 'DELETE', errorMessage: null },
      'Delete failed'
    )
    expect(next.confirmationInput).toBe('DELETE')
    expect(next.errorMessage).toBe('Delete failed')
  })

  test('stale-write conflict 会转成显式冲突反馈', () => {
    const conflict = normalizeWorkbenchMutationError(
      new Error('409 Conflict {"detail":"Stale relation revision token"}'),
      'Delete failed'
    )

    expect(conflict.isConflict).toBe(true)
    expect(conflict.message).toContain('Stale revision conflict')
  })

  test('merge: manual source/target entity selection 解析与去重', async () => {
    const { buildManualMergeDraftFromInput } = await import('./MergeEntityPanel')

    const draft = buildManualMergeDraftFromInput(
      ' Elon Msk , Ellon Musk\nElon Msk ',
      'Elon Musk'
    )

    expect(draft.targetEntity).toBe('Elon Musk')
    expect(draft.sourceEntities).toEqual(['Elon Msk', 'Ellon Musk'])
  })

  test('merge: suggested candidate evidence 可展示', async () => {
    const { buildMergeCandidateEvidence } = await import('./MergeSuggestionList')
    const candidate: GraphMergeSuggestionCandidate = {
      target_entity: 'Elon Musk',
      source_entities: ['Elon Msk', 'Ellon Musk'],
      score: 0.97,
      reasons: [
        { code: 'name_similarity', score: 0.97 },
        { code: 'description_overlap', score: 0.81 }
      ]
    }

    const evidence = buildMergeCandidateEvidence(candidate)
    expect(evidence).toContain('name_similarity')
    expect(evidence).toContain('description_overlap')
    expect(evidence).toContain('0.97')
  })

  test('merge: suggested candidate load 使用 applied scope 构建请求', async () => {
    const { buildMergeSuggestionsRequest } = await import('./MergeEntityPanel')
    const filterDraft = getDefaultGraphWorkbenchFilterDraft()
    const appliedQuery = getDefaultGraphWorkbenchFilterDraft()
    appliedQuery.scope.label = 'Tesla'
    appliedQuery.scope.max_depth = 2
    appliedQuery.scope.max_nodes = 128
    appliedQuery.scope.only_matched_neighborhood = true

    const request = buildMergeSuggestionsRequest(appliedQuery, filterDraft, 12, 0.45)
    expect(request.scope.label).toBe('Tesla')
    expect(request.scope.max_depth).toBe(2)
    expect(request.limit).toBe(12)
    expect(request.min_score).toBe(0.45)
  })

  test('merge: one-click candidate import into merge form', async () => {
    const { importMergeCandidate } = useGraphWorkbenchStore.getState()
    const candidate: GraphMergeSuggestionCandidate = {
      target_entity: 'OpenAI',
      source_entities: ['Open AI'],
      score: 0.93,
      reasons: [{ code: 'alias_overlap', score: 0.93 }]
    }

    importMergeCandidate(candidate)
    const state = useGraphWorkbenchStore.getState()
    expect(state.mergeDraft.targetEntity).toBe('OpenAI')
    expect(state.mergeDraft.sourceEntities).toEqual(['Open AI'])
  })

  test('merge: expected revision tokens 会从当前 selection 映射到 merge 请求实体', async () => {
    const { buildExpectedRevisionTokensForMerge } = await import('./MergeEntityPanel')
    const tokens = buildExpectedRevisionTokensForMerge(
      {
        sourceEntities: ['Elon Musk'],
        targetEntity: 'Tesla'
      },
      relationSelection as any
    )

    expect(tokens).toEqual({
      'Elon Musk': 'edge-token-1',
      Tesla: 'edge-token-1'
    })
  })

  test('merge: post-merge 后续动作映射（focus / refresh / continue）', async () => {
    const { resolvePostMergeFollowUp } = await import('./MergeEntityPanel')

    const focus = resolvePostMergeFollowUp('focus_target', 'Elon Musk')
    const refresh = resolvePostMergeFollowUp('refresh_results', 'Elon Musk')
    const continueReview = resolvePostMergeFollowUp('continue_review', 'Elon Musk')

    expect(focus.focusTarget).toBe('Elon Musk')
    expect(focus.shouldRefresh).toBe(false)
    expect(focus.dismissActions).toBe(false)

    expect(refresh.focusTarget).toBeNull()
    expect(refresh.shouldRefresh).toBe(true)
    expect(refresh.dismissActions).toBe(false)

    expect(continueReview.focusTarget).toBeNull()
    expect(continueReview.shouldRefresh).toBe(false)
    expect(continueReview.dismissActions).toBe(true)
  })
})
