import { create } from 'zustand'

import type {
  GraphMergeSuggestionCandidate,
  GraphWorkbenchQueryRequest
} from '@/api/lightrag'
import { createSelectors } from '@/lib/utils'

const cloneQuery = (query: GraphWorkbenchQueryRequest): GraphWorkbenchQueryRequest => {
  return {
    scope: { ...query.scope },
    node_filters: {
      ...query.node_filters,
      entity_types: [...query.node_filters.entity_types]
    },
    edge_filters: {
      ...query.edge_filters,
      relation_types: [...query.edge_filters.relation_types],
      source_entity_types: [...query.edge_filters.source_entity_types],
      target_entity_types: [...query.edge_filters.target_entity_types]
    },
    source_filters: {
      ...query.source_filters,
      file_paths: [...query.source_filters.file_paths]
    },
    view_options: { ...query.view_options }
  }
}

export const getDefaultGraphWorkbenchFilterDraft = (): GraphWorkbenchQueryRequest => ({
  scope: {
    label: '*',
    max_depth: 3,
    max_nodes: 1000,
    only_matched_neighborhood: false
  },
  node_filters: {
    entity_types: [],
    name_query: '',
    description_query: '',
    degree_min: null,
    degree_max: null,
    isolated_only: false
  },
  edge_filters: {
    relation_types: [],
    keyword_query: '',
    weight_min: null,
    weight_max: null,
    source_entity_types: [],
    target_entity_types: []
  },
  source_filters: {
    source_id_query: '',
    file_paths: [],
    time_from: null,
    time_to: null
  },
  view_options: {
    show_nodes_only: false,
    show_edges_only: false,
    hide_low_weight_edges: false,
    hide_empty_description: false,
    highlight_matches: false
  }
})

interface GraphWorkbenchState {
  filterDraft: GraphWorkbenchQueryRequest
  appliedQuery: GraphWorkbenchQueryRequest | null
  mergeCandidates: GraphMergeSuggestionCandidate[]
  selectedMergeCandidateTargets: string[]
  mutationError: string | null
  conflictError: string | null
  queryVersion: number

  setFilterDraft: (draft: GraphWorkbenchQueryRequest) => void
  applyFilterDraft: () => void
  applyScopeLabel: (label: string) => void
  setMergeCandidates: (candidates: GraphMergeSuggestionCandidate[]) => void
  selectMergeCandidate: (targetEntity: string) => void
  clearSelection: () => void
  setMutationError: (message: string | null, isConflict?: boolean) => void
  clearMutationError: () => void
  requestRefresh: () => void
  reset: () => void
}

const useGraphWorkbenchStoreBase = create<GraphWorkbenchState>()((set, get) => ({
  filterDraft: getDefaultGraphWorkbenchFilterDraft(),
  appliedQuery: null,
  mergeCandidates: [],
  selectedMergeCandidateTargets: [],
  mutationError: null,
  conflictError: null,
  queryVersion: 0,

  setFilterDraft: (draft) => set({ filterDraft: cloneQuery(draft) }),
  applyFilterDraft: () => {
    const draft = get().filterDraft
    set((state) => ({
      appliedQuery: cloneQuery(draft),
      queryVersion: state.queryVersion + 1
    }))
  },
  applyScopeLabel: (label) =>
    set((state) => {
      const nextDraft = cloneQuery(state.appliedQuery ?? state.filterDraft)
      nextDraft.scope.label = label
      return {
        filterDraft: cloneQuery(nextDraft),
        appliedQuery: nextDraft,
        queryVersion: state.queryVersion + 1
      }
    }),
  setMergeCandidates: (candidates) => set({ mergeCandidates: [...candidates] }),
  selectMergeCandidate: (targetEntity) =>
    set((state) => {
      if (state.selectedMergeCandidateTargets.includes(targetEntity)) {
        return {
          selectedMergeCandidateTargets: state.selectedMergeCandidateTargets.filter(
            (item) => item !== targetEntity
          )
        }
      }
      return {
        selectedMergeCandidateTargets: [
          ...state.selectedMergeCandidateTargets,
          targetEntity
        ]
      }
    }),
  clearSelection: () => set({ selectedMergeCandidateTargets: [] }),
  setMutationError: (message, isConflict = false) =>
    set({
      mutationError: message,
      conflictError: isConflict ? message : null
    }),
  clearMutationError: () => set({ mutationError: null, conflictError: null }),
  requestRefresh: () => set((state) => ({ queryVersion: state.queryVersion + 1 })),
  reset: () =>
    set({
      filterDraft: getDefaultGraphWorkbenchFilterDraft(),
      appliedQuery: null,
      mergeCandidates: [],
      selectedMergeCandidateTargets: [],
      mutationError: null,
      conflictError: null,
      queryVersion: 0
    })
}))

const useGraphWorkbenchStore = createSelectors(useGraphWorkbenchStoreBase)

export { useGraphWorkbenchStore }
