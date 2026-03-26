import { create } from 'zustand'

import type {
  GraphMergeSuggestionCandidate,
  GraphWorkbenchQueryRequest
} from '@/api/lightrag'
import { createSelectors } from '@/lib/utils'

export type GraphMergeDraft = {
  sourceEntities: string[]
  targetEntity: string
}

export type GraphMergeFollowUpState = {
  targetEntity: string
  sourceEntities: string[]
  mergedAt: number
}

export type GraphWorkbenchMutationError = {
  message: string
  isConflict: boolean
}

const isRevisionConflictMessage = (message: string): boolean => {
  const normalized = message.toLowerCase()
  return normalized.includes('revision token') || normalized.includes('stale')
}

const extractErrorMessage = (error: unknown, fallback: string): string => {
  if (error && typeof error === 'object') {
    const response = (error as { response?: { data?: { detail?: unknown } } }).response
    const detail = response?.data?.detail
    if (typeof detail === 'string' && detail.trim()) {
      return detail
    }
  }

  if (error instanceof Error) {
    const detailMatch = error.message.match(/"detail"\s*:\s*"([^"]+)"/)
    if (detailMatch?.[1]) {
      return detailMatch[1]
    }
    return error.message
  }

  return fallback
}

export const normalizeWorkbenchMutationError = (
  error: unknown,
  fallbackMessage: string
): GraphWorkbenchMutationError => {
  const message = extractErrorMessage(error, fallbackMessage)
  const isConflict =
    (error &&
      typeof error === 'object' &&
      (error as { response?: { status?: number } }).response?.status === 409) ||
    (message.includes('409') && isRevisionConflictMessage(message)) ||
    isRevisionConflictMessage(message)

  if (isConflict) {
    return {
      message: `Stale revision conflict: ${message}. Please refresh and retry.`,
      isConflict: true
    }
  }

  return {
    message,
    isConflict: false
  }
}

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

export const getDefaultMergeDraft = (): GraphMergeDraft => ({
  sourceEntities: [],
  targetEntity: ''
})

interface GraphWorkbenchState {
  filterDraft: GraphWorkbenchQueryRequest
  appliedQuery: GraphWorkbenchQueryRequest | null
  mergeCandidates: GraphMergeSuggestionCandidate[]
  selectedMergeCandidateTargets: string[]
  mergeDraft: GraphMergeDraft
  mergeFollowUp: GraphMergeFollowUpState | null
  mutationError: string | null
  conflictError: string | null
  queryVersion: number

  setFilterDraft: (draft: GraphWorkbenchQueryRequest) => void
  applyFilterDraft: () => void
  applyScopeLabel: (label: string) => void
  setMergeCandidates: (candidates: GraphMergeSuggestionCandidate[]) => void
  selectMergeCandidate: (targetEntity: string) => void
  setMergeDraft: (draft: GraphMergeDraft) => void
  importMergeCandidate: (candidate: GraphMergeSuggestionCandidate) => void
  clearMergeDraft: () => void
  clearSelection: () => void
  setMergeFollowUp: (targetEntity: string, sourceEntities: string[]) => void
  clearMergeFollowUp: () => void
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
  mergeDraft: getDefaultMergeDraft(),
  mergeFollowUp: null,
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
  setMergeDraft: (draft) =>
    set({
      mergeDraft: {
        sourceEntities: [...draft.sourceEntities],
        targetEntity: draft.targetEntity
      }
    }),
  importMergeCandidate: (candidate) =>
    set({
      mergeDraft: {
        sourceEntities: [...candidate.source_entities],
        targetEntity: candidate.target_entity
      },
      selectedMergeCandidateTargets: [candidate.target_entity]
    }),
  clearMergeDraft: () => set({ mergeDraft: getDefaultMergeDraft() }),
  clearSelection: () => set({ selectedMergeCandidateTargets: [] }),
  setMergeFollowUp: (targetEntity, sourceEntities) =>
    set({
      mergeFollowUp: {
        targetEntity,
        sourceEntities: [...sourceEntities],
        mergedAt: Date.now()
      }
    }),
  clearMergeFollowUp: () => set({ mergeFollowUp: null }),
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
      mergeDraft: getDefaultMergeDraft(),
      mergeFollowUp: null,
      mutationError: null,
      conflictError: null,
      queryVersion: 0
    })
}))

const useGraphWorkbenchStore = createSelectors(useGraphWorkbenchStoreBase)

export { useGraphWorkbenchStore }
