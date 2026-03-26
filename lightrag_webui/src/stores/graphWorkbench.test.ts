import { beforeEach, describe, expect, test } from 'bun:test'

import {
  getDefaultGraphWorkbenchFilterDraft,
  useGraphWorkbenchStore
} from './graphWorkbench'

describe('graphWorkbench store', () => {
  beforeEach(() => {
    useGraphWorkbenchStore.getState().reset()
  })

  test('维护 filter draft 与 applied query 的分离状态', () => {
    const store = useGraphWorkbenchStore.getState()
    const draft = getDefaultGraphWorkbenchFilterDraft()
    const beforeVersion = useGraphWorkbenchStore.getState().queryVersion

    draft.scope.label = 'Tesla'
    draft.scope.max_depth = 2
    draft.scope.max_nodes = 200
    draft.node_filters.entity_types = ['ORGANIZATION']

    store.setFilterDraft(draft)
    expect(useGraphWorkbenchStore.getState().appliedQuery).toBeNull()

    store.applyFilterDraft()
    const applied = useGraphWorkbenchStore.getState().appliedQuery
    expect(applied).not.toBeNull()
    expect(applied?.scope.label).toBe('Tesla')
    expect(applied?.scope.max_depth).toBe(2)
    expect(applied?.node_filters.entity_types).toEqual(['ORGANIZATION'])
    expect(useGraphWorkbenchStore.getState().queryVersion).toBe(beforeVersion + 1)
  })

  test('维护 merge candidate 列表与选择队列', () => {
    const store = useGraphWorkbenchStore.getState()
    const candidates = [
      {
        target_entity: 'Elon Musk',
        source_entities: ['Elon Msk'],
        score: 0.95,
        reasons: [{ code: 'name_similarity', score: 0.95 }]
      },
      {
        target_entity: 'OpenAI',
        source_entities: ['Open AI'],
        score: 0.91,
        reasons: [{ code: 'alias_overlap', score: 0.91 }]
      }
    ]

    store.setMergeCandidates(candidates)
    store.selectMergeCandidate('Elon Musk')
    store.selectMergeCandidate('OpenAI')

    expect(useGraphWorkbenchStore.getState().selectedMergeCandidateTargets).toEqual([
      'Elon Musk',
      'OpenAI'
    ])

    store.clearSelection()
    expect(useGraphWorkbenchStore.getState().selectedMergeCandidateTargets).toEqual([])
  })

  test('applyScopeLabel 会直接更新 applied query 并递增 refetch version', () => {
    const store = useGraphWorkbenchStore.getState()
    const beforeVersion = useGraphWorkbenchStore.getState().queryVersion

    store.applyScopeLabel('OpenAI')

    const state = useGraphWorkbenchStore.getState()
    expect(state.filterDraft.scope.label).toBe('OpenAI')
    expect(state.appliedQuery?.scope.label).toBe('OpenAI')
    expect(state.queryVersion).toBe(beforeVersion + 1)
  })

  test('维护 mutationError 与 conflictError 状态', () => {
    const store = useGraphWorkbenchStore.getState()

    store.setMutationError('删除失败', true)
    expect(useGraphWorkbenchStore.getState().mutationError).toBe('删除失败')
    expect(useGraphWorkbenchStore.getState().conflictError).toBe('删除失败')

    store.clearMutationError()
    expect(useGraphWorkbenchStore.getState().mutationError).toBeNull()
    expect(useGraphWorkbenchStore.getState().conflictError).toBeNull()
  })

  test('requestRefresh 可触发 refetch version 递增', () => {
    const store = useGraphWorkbenchStore.getState()
    const before = useGraphWorkbenchStore.getState().queryVersion

    store.requestRefresh()

    expect(useGraphWorkbenchStore.getState().queryVersion).toBe(before + 1)
  })
})
