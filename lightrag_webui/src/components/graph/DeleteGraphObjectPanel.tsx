import { FormEvent, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { deleteGraphEntity, deleteGraphRelation } from '@/api/lightrag'
import { useGraphStore } from '@/stores/graph'
import {
  normalizeWorkbenchMutationError,
  useGraphWorkbenchStore
} from '@/stores/graphWorkbench'
import type { ActionInspectorSelection } from './ActionInspector'

export type DeletePanelState = {
  confirmationInput: string
  errorMessage: string | null
}

const DELETE_CONFIRM_KEYWORD = 'DELETE'
type DeleteCopyTranslator = (key: string, options?: Record<string, unknown>) => string

export const buildDeleteConfirmationCopy = (
  selection: ActionInspectorSelection | null | undefined,
  t?: DeleteCopyTranslator
): string => {
  const translate = (key: string, options?: Record<string, unknown>): string => {
    if (t) {
      return t(key, options)
    }
    return key
  }

  if (!selection) {
    return t
      ? translate('graphPanel.workbench.deleteObject.confirmation.emptySelection')
      : 'Select a node or relation first.'
  }

  if (selection.kind === 'node') {
    const entityName = String(selection.node.properties?.entity_id ?? selection.node.id)
    return t
      ? translate('graphPanel.workbench.deleteObject.confirmation.node', { entity: entityName })
      : `You are deleting entity "${entityName}". Related relations will also be removed.`
  }

  const source = String(
    selection.edge.sourceNode?.properties?.entity_id ?? selection.edge.source
  )
  const target = String(
    selection.edge.targetNode?.properties?.entity_id ?? selection.edge.target
  )
  const summary = selection.edge.type || selection.edge.properties?.keywords
  if (summary) {
    return t
      ? translate('graphPanel.workbench.deleteObject.confirmation.relationWithSummary', {
          source,
          target,
          summary
        })
      : `You are deleting relation "${source} -> ${target}" (${summary}).`
  }
  return t
    ? translate('graphPanel.workbench.deleteObject.confirmation.relation', { source, target })
    : `You are deleting relation "${source} -> ${target}".`
}

export const reduceDeletePanelStateAfterFailure = (
  state: DeletePanelState,
  message: string
): DeletePanelState => ({
  ...state,
  errorMessage: message
})

type DeleteGraphObjectPanelProps = {
  selection?: ActionInspectorSelection | null
}

const DeleteGraphObjectPanel = ({ selection = null }: DeleteGraphObjectPanelProps) => {
  const { t } = useTranslation()
  const [state, setState] = useState<DeletePanelState>({
    confirmationInput: '',
    errorMessage: null
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const setMutationError = useGraphWorkbenchStore.use.setMutationError()
  const clearMutationError = useGraphWorkbenchStore.use.clearMutationError()
  const requestRefresh = useGraphWorkbenchStore.use.requestRefresh()

  const confirmationCopy = useMemo(() => buildDeleteConfirmationCopy(selection, t), [selection, t])
  const canSubmit =
    !!selection && state.confirmationInput.trim().toUpperCase() === DELETE_CONFIRM_KEYWORD

  useEffect(() => {
    setState((prev) => ({ ...prev, errorMessage: null }))
  }, [selection])

  const handleDelete = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!selection || !canSubmit || isSubmitting) return

    setIsSubmitting(true)
    clearMutationError()
    setState((prev) => ({ ...prev, errorMessage: null }))

    try {
      if (selection.kind === 'node') {
        const entityName = String(selection.node.properties?.entity_id ?? selection.node.id)
        await deleteGraphEntity(entityName)
        toast.success(t('graphPanel.workbench.deleteObject.messages.entityDeleted', { entity: entityName }))
      } else {
        const source = String(
          selection.edge.sourceNode?.properties?.entity_id ?? selection.edge.source
        )
        const target = String(
          selection.edge.targetNode?.properties?.entity_id ?? selection.edge.target
        )
        await deleteGraphRelation(source, target, selection.edge.revision_token)
        toast.success(t('graphPanel.workbench.deleteObject.messages.relationDeleted', { source, target }))
      }

      setState({ confirmationInput: '', errorMessage: null })
      useGraphStore.getState().setGraphDataFetchAttempted(false)
      requestRefresh()
      useGraphStore.getState().incrementGraphDataVersion()
    } catch (error) {
      const normalized = normalizeWorkbenchMutationError(
        error,
        t('graphPanel.workbench.deleteObject.errors.deleteFailed')
      )
      setState((prev) => reduceDeletePanelStateAfterFailure(prev, normalized.message))
      setMutationError(normalized.message, normalized.isConflict)
      toast.error(normalized.message)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleDelete} className="bg-background/60 space-y-3 rounded-lg border p-3">
      <div>
        <h3 className="text-sm font-semibold">{t('graphPanel.workbench.deleteObject.title')}</h3>
        <p className="text-muted-foreground mt-1 text-xs">{confirmationCopy}</p>
      </div>

      <div className="space-y-1">
        <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
          {t('graphPanel.workbench.deleteObject.confirmation.inputLabel', {
            keyword: DELETE_CONFIRM_KEYWORD
          })}
        </label>
        <Input
          value={state.confirmationInput}
          onChange={(event) =>
            setState((prev) => ({ ...prev, confirmationInput: event.target.value }))
          }
          placeholder={t('graphPanel.workbench.deleteObject.confirmation.inputPlaceholder', {
            keyword: DELETE_CONFIRM_KEYWORD
          })}
        />
      </div>

      {state.errorMessage && <p className="text-xs text-red-600 dark:text-red-300">{state.errorMessage}</p>}

      <div className="flex justify-end">
        <Button type="submit" size="sm" variant="destructive" disabled={!canSubmit || isSubmitting}>
          {isSubmitting
            ? t('graphPanel.workbench.deleteObject.actions.deleting')
            : t('graphPanel.workbench.deleteObject.actions.delete')}
        </Button>
      </div>
    </form>
  )
}

export default DeleteGraphObjectPanel
