import { FormEvent, useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import {
  fetchMergeSuggestions,
  mergeGraphEntities
} from '@/api/lightrag'
import type {
  GraphMergeSuggestionsRequest,
  GraphWorkbenchQueryRequest
} from '@/api/lightrag'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { useGraphStore } from '@/stores/graph'
import {
  normalizeWorkbenchMutationError,
  useGraphWorkbenchStore
} from '@/stores/graphWorkbench'
import type { GraphMergeDraft } from '@/stores/graphWorkbench'
import { useSettingsStore } from '@/stores/settings'
import type { ActionInspectorSelection } from './ActionInspector'
import MergeSuggestionList from './MergeSuggestionList'

type PostMergeFollowUpAction = 'focus_target' | 'refresh_results' | 'continue_review'

type PostMergeFollowUpOutcome = {
  focusTarget: string | null
  shouldRefresh: boolean
  dismissActions: boolean
}

const DEFAULT_SUGGESTION_LIMIT = 20
const DEFAULT_SUGGESTION_MIN_SCORE = 0.6

const normalizeEntityName = (value: string): string => value.trim()

const dedupeEntities = (values: string[]): string[] => {
  const deduped: string[] = []
  const seen = new Set<string>()
  for (const value of values) {
    const normalized = normalizeEntityName(value)
    if (!normalized) continue
    const key = normalized.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    deduped.push(normalized)
  }
  return deduped
}

const parseEntityListInput = (input: string): string[] => {
  const raw = input
    .split(/[,\n]/g)
    .map((part) => normalizeEntityName(part))
    .filter((part) => part.length > 0)
  return dedupeEntities(raw)
}

export const buildManualMergeDraftFromInput = (
  sourceEntitiesInput: string,
  targetEntityInput: string
): GraphMergeDraft => {
  const targetEntity = normalizeEntityName(targetEntityInput)
  const targetKey = targetEntity.toLowerCase()
  const sourceEntities = parseEntityListInput(sourceEntitiesInput).filter(
    (entity) => entity.toLowerCase() !== targetKey
  )
  return {
    sourceEntities,
    targetEntity
  }
}

export const buildMergeSuggestionsRequest = (
  appliedQuery: GraphWorkbenchQueryRequest | null,
  filterDraft: GraphWorkbenchQueryRequest,
  limit: number = DEFAULT_SUGGESTION_LIMIT,
  minScore: number = DEFAULT_SUGGESTION_MIN_SCORE
): GraphMergeSuggestionsRequest => {
  const scope = appliedQuery?.scope ?? filterDraft.scope
  return {
    scope: {
      label: scope.label,
      max_depth: scope.max_depth,
      max_nodes: scope.max_nodes,
      only_matched_neighborhood: scope.only_matched_neighborhood
    },
    limit,
    min_score: minScore
  }
}

const buildMergeDraftFromSelection = (
  selection: ActionInspectorSelection | null | undefined
): GraphMergeDraft => {
  if (!selection) {
    return {
      sourceEntities: [],
      targetEntity: ''
    }
  }

  if (selection.kind === 'node') {
    const targetEntity = String(selection.node.properties?.entity_id ?? selection.node.id ?? '')
    return {
      sourceEntities: [],
      targetEntity
    }
  }

  const source = String(selection.edge.sourceNode?.properties?.entity_id ?? selection.edge.source ?? '')
  const target = String(selection.edge.targetNode?.properties?.entity_id ?? selection.edge.target ?? '')
  return {
    sourceEntities: source ? [source] : [],
    targetEntity: target
  }
}

const buildVisibleRevisionTokenMap = (
  selection: ActionInspectorSelection | null | undefined
): Record<string, string> => {
  if (!selection) {
    return {}
  }

  if (selection.kind === 'node') {
    const entityId = String(selection.node.properties?.entity_id ?? selection.node.id ?? '')
    const revisionToken = selection.node.revision_token
    return entityId && revisionToken ? { [entityId]: revisionToken } : {}
  }

  const tokens: Record<string, string> = {}
  const sourceEntity = String(
    selection.edge.sourceNode?.properties?.entity_id ?? selection.edge.source ?? ''
  )
  const targetEntity = String(
    selection.edge.targetNode?.properties?.entity_id ?? selection.edge.target ?? ''
  )
  const edgeToken = selection.edge.revision_token

  if (sourceEntity && edgeToken) {
    tokens[sourceEntity] = edgeToken
  }
  if (targetEntity && edgeToken) {
    tokens[targetEntity] = edgeToken
  }
  return tokens
}

export const buildExpectedRevisionTokensForMerge = (
  draft: GraphMergeDraft,
  selection: ActionInspectorSelection | null | undefined
): Record<string, string> | undefined => {
  const visibleTokens = buildVisibleRevisionTokenMap(selection)
  const mergedTokens: Record<string, string> = {}

  for (const entity of [...draft.sourceEntities, draft.targetEntity]) {
    const normalized = normalizeEntityName(entity)
    const token = visibleTokens[normalized]
    if (normalized && token) {
      mergedTokens[normalized] = token
    }
  }

  return Object.keys(mergedTokens).length > 0 ? mergedTokens : undefined
}

export const resolvePostMergeFollowUp = (
  action: PostMergeFollowUpAction,
  targetEntity: string
): PostMergeFollowUpOutcome => {
  if (action === 'focus_target') {
    return {
      focusTarget: targetEntity,
      shouldRefresh: false,
      dismissActions: false
    }
  }

  if (action === 'refresh_results') {
    return {
      focusTarget: null,
      shouldRefresh: true,
      dismissActions: false
    }
  }

  return {
    focusTarget: null,
    shouldRefresh: false,
    dismissActions: true
  }
}

type MergeEntityPanelProps = {
  selection?: ActionInspectorSelection | null
}

const MergeEntityPanel = ({ selection = null }: MergeEntityPanelProps) => {
  const [sourceEntitiesInput, setSourceEntitiesInput] = useState('')
  const [targetEntityInput, setTargetEntityInput] = useState('')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [suggestionError, setSuggestionError] = useState<string | null>(null)
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false)
  const [isSubmittingMerge, setIsSubmittingMerge] = useState(false)

  const filterDraft = useGraphWorkbenchStore.use.filterDraft()
  const appliedQuery = useGraphWorkbenchStore.use.appliedQuery()
  const mergeCandidates = useGraphWorkbenchStore.use.mergeCandidates()
  const selectedMergeCandidateTargets = useGraphWorkbenchStore.use.selectedMergeCandidateTargets()
  const mergeDraft = useGraphWorkbenchStore.use.mergeDraft()
  const mergeFollowUp = useGraphWorkbenchStore.use.mergeFollowUp()
  const setMergeCandidates = useGraphWorkbenchStore.use.setMergeCandidates()
  const setMergeDraft = useGraphWorkbenchStore.use.setMergeDraft()
  const importMergeCandidate = useGraphWorkbenchStore.use.importMergeCandidate()
  const setMergeFollowUp = useGraphWorkbenchStore.use.setMergeFollowUp()
  const clearMergeFollowUp = useGraphWorkbenchStore.use.clearMergeFollowUp()
  const setMutationError = useGraphWorkbenchStore.use.setMutationError()
  const clearMutationError = useGraphWorkbenchStore.use.clearMutationError()
  const requestRefresh = useGraphWorkbenchStore.use.requestRefresh()
  const applyScopeLabel = useGraphWorkbenchStore.use.applyScopeLabel()
  const setQueryLabel = useSettingsStore.use.setQueryLabel()

  useEffect(() => {
    setSourceEntitiesInput(mergeDraft.sourceEntities.join(', '))
    setTargetEntityInput(mergeDraft.targetEntity)
  }, [mergeDraft])

  useEffect(() => {
    if (mergeDraft.sourceEntities.length > 0 || mergeDraft.targetEntity) {
      return
    }
    const prefilled = buildMergeDraftFromSelection(selection)
    if (!prefilled.sourceEntities.length && !prefilled.targetEntity) {
      return
    }
    setMergeDraft(prefilled)
  }, [selection, mergeDraft, setMergeDraft])

  const draftPreview = useMemo(
    () => buildManualMergeDraftFromInput(sourceEntitiesInput, targetEntityInput),
    [sourceEntitiesInput, targetEntityInput]
  )

  const canSubmitMerge = draftPreview.sourceEntities.length > 0 && !!draftPreview.targetEntity

  const handleLoadSuggestions = async () => {
    if (isLoadingSuggestions) return

    setSuggestionError(null)
    setIsLoadingSuggestions(true)
    clearMutationError()

    try {
      const request = buildMergeSuggestionsRequest(appliedQuery, filterDraft)
      const response = await fetchMergeSuggestions(request)
      setMergeCandidates(response.candidates)
      if (!response.candidates.length) {
        toast.info('No merge suggestions found for current scope.')
      }
    } catch (error) {
      const normalized = normalizeWorkbenchMutationError(error, 'Load merge suggestions failed')
      setSuggestionError(normalized.message)
      setMutationError(normalized.message, normalized.isConflict)
      toast.error(normalized.message)
    } finally {
      setIsLoadingSuggestions(false)
    }
  }

  const handleImportCandidate = (candidate: (typeof mergeCandidates)[number]) => {
    importMergeCandidate(candidate)
    setErrorMessage(null)
    setSuggestionError(null)
    clearMutationError()
  }

  const handleSubmitMerge = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (isSubmittingMerge) return

    const draft = buildManualMergeDraftFromInput(sourceEntitiesInput, targetEntityInput)
    setMergeDraft(draft)

    if (!draft.targetEntity) {
      const message = 'Target entity is required.'
      setErrorMessage(message)
      setMutationError(message, false)
      return
    }

    if (!draft.sourceEntities.length) {
      const message = 'At least one source entity is required.'
      setErrorMessage(message)
      setMutationError(message, false)
      return
    }

    setIsSubmittingMerge(true)
    setErrorMessage(null)
    clearMutationError()

    try {
      const expectedRevisionTokens = buildExpectedRevisionTokensForMerge(
        draft,
        selection
      )
      await mergeGraphEntities(
        draft.sourceEntities,
        draft.targetEntity,
        expectedRevisionTokens
      )
      setMergeFollowUp(draft.targetEntity, draft.sourceEntities)
      toast.success(
        `Merged ${draft.sourceEntities.length} source entities into "${draft.targetEntity}".`
      )
    } catch (error) {
      const normalized = normalizeWorkbenchMutationError(error, 'Merge entities failed')
      setErrorMessage(normalized.message)
      setMutationError(normalized.message, normalized.isConflict)
      toast.error(normalized.message)
    } finally {
      setIsSubmittingMerge(false)
    }
  }

  const handlePostMergeAction = (action: PostMergeFollowUpAction) => {
    if (!mergeFollowUp) return

    const outcome = resolvePostMergeFollowUp(action, mergeFollowUp.targetEntity)
    const graphStore = useGraphStore.getState()
    graphStore.setGraphDataFetchAttempted(false)
    if (outcome.focusTarget) {
      const target = outcome.focusTarget
      graphStore.setFocusedNode(target)
      graphStore.setSelectedNode(target, true)
      if (appliedQuery) {
        applyScopeLabel(target)
      } else {
        setQueryLabel(target)
      }
    }

    if (outcome.shouldRefresh) {
      requestRefresh()
      graphStore.incrementGraphDataVersion()
    }

    if (outcome.dismissActions) {
      clearMergeFollowUp()
    }
  }

  return (
    <div className="space-y-3">
      <form onSubmit={handleSubmitMerge} className="bg-background/60 space-y-3 rounded-lg border p-3">
        <div>
          <h3 className="text-sm font-semibold">Manual Merge</h3>
          <p className="text-muted-foreground mt-1 text-xs">
            Provide source entities and one target entity, then submit merge.
          </p>
        </div>

        <div className="space-y-1">
          <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
            Source Entities
          </label>
          <Input
            value={sourceEntitiesInput}
            onChange={(event) => setSourceEntitiesInput(event.target.value)}
            placeholder="Elon Msk, Ellon Musk"
          />
          <p className="text-muted-foreground text-[11px]">
            Comma/newline separated. Target entity will be excluded automatically.
          </p>
        </div>

        <div className="space-y-1">
          <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
            Target Entity
          </label>
          <Input
            value={targetEntityInput}
            onChange={(event) => setTargetEntityInput(event.target.value)}
            placeholder="Elon Musk"
          />
        </div>

        <div className="text-muted-foreground rounded-md border border-dashed px-2 py-2 text-[11px]">
          Draft preview: [{draftPreview.sourceEntities.join(', ')}] →{' '}
          {draftPreview.targetEntity || '(target required)'}
        </div>

        {errorMessage && <p className="text-xs text-red-600 dark:text-red-300">{errorMessage}</p>}

        <div className="flex items-center justify-end gap-2">
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={handleLoadSuggestions}
            disabled={isLoadingSuggestions}
          >
            {isLoadingSuggestions ? 'Loading...' : 'Load Suggestions'}
          </Button>
          <Button type="submit" size="sm" disabled={!canSubmitMerge || isSubmittingMerge}>
            {isSubmittingMerge ? 'Merging...' : 'Merge Entities'}
          </Button>
        </div>
      </form>

      <MergeSuggestionList
        candidates={mergeCandidates}
        selectedTargets={selectedMergeCandidateTargets}
        isLoading={isLoadingSuggestions}
        errorMessage={suggestionError}
        onImportCandidate={handleImportCandidate}
      />

      {mergeFollowUp && (
        <section className="space-y-2 rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-3">
          <p className="text-xs font-medium text-emerald-700 dark:text-emerald-300">
            Merge completed: {mergeFollowUp.sourceEntities.join(', ')} → {mergeFollowUp.targetEntity}
          </p>
          <p className="text-muted-foreground text-[11px]">
            Next actions: focus merged target, refresh result graph, or continue review.
          </p>
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 px-2 text-[11px]"
              onClick={() => handlePostMergeAction('focus_target')}
            >
              Focus merged target
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 px-2 text-[11px]"
              onClick={() => handlePostMergeAction('refresh_results')}
            >
              Refresh results
            </Button>
            <Button
              type="button"
              size="sm"
              className="h-7 px-2 text-[11px]"
              onClick={() => handlePostMergeAction('continue_review')}
            >
              Continue review
            </Button>
          </div>
        </section>
      )}
    </div>
  )
}

export default MergeEntityPanel
