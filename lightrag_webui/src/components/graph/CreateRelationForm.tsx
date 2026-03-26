import { FormEvent, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import { createGraphRelation } from '@/api/lightrag'
import { useGraphStore } from '@/stores/graph'
import {
  normalizeWorkbenchMutationError,
  useGraphWorkbenchStore
} from '@/stores/graphWorkbench'
import type { ActionInspectorSelection } from './ActionInspector'

export type CreateRelationDraft = {
  sourceEntity: string
  targetEntity: string
  description: string
  keywords: string
  weight: string
}

export const deriveCreateRelationDraftFromSelection = (
  selection: ActionInspectorSelection | null | undefined
): CreateRelationDraft => {
  if (!selection) {
    return {
      sourceEntity: '',
      targetEntity: '',
      description: '',
      keywords: '',
      weight: '1'
    }
  }

  if (selection.kind === 'node') {
    return {
      sourceEntity: String(selection.node.properties?.entity_id ?? selection.node.id),
      targetEntity: '',
      description: '',
      keywords: '',
      weight: '1'
    }
  }

  return {
    sourceEntity: String(
      selection.edge.sourceNode?.properties?.entity_id ?? selection.edge.source
    ),
    targetEntity: String(
      selection.edge.targetNode?.properties?.entity_id ?? selection.edge.target
    ),
    description: '',
    keywords: '',
    weight: '1'
  }
}

type CreateRelationFormProps = {
  selection?: ActionInspectorSelection | null
}

const CreateRelationForm = ({ selection = null }: CreateRelationFormProps) => {
  const { t } = useTranslation()
  const initialDraft = deriveCreateRelationDraftFromSelection(selection)
  const [sourceEntity, setSourceEntity] = useState(initialDraft.sourceEntity)
  const [targetEntity, setTargetEntity] = useState(initialDraft.targetEntity)
  const [description, setDescription] = useState('')
  const [keywords, setKeywords] = useState('')
  const [weight, setWeight] = useState('1')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const setMutationError = useGraphWorkbenchStore.use.setMutationError()
  const clearMutationError = useGraphWorkbenchStore.use.clearMutationError()
  const requestRefresh = useGraphWorkbenchStore.use.requestRefresh()

  useEffect(() => {
    const prefilled = deriveCreateRelationDraftFromSelection(selection)
    if (!sourceEntity.trim()) {
      setSourceEntity(prefilled.sourceEntity)
    }
    if (!targetEntity.trim()) {
      setTargetEntity(prefilled.targetEntity)
    }
  }, [selection, sourceEntity, targetEntity])

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (isSubmitting) return

    const source = sourceEntity.trim()
    const target = targetEntity.trim()
    if (!source || !target) {
      const message = t('graphPanel.workbench.createRelation.errors.required')
      setErrorMessage(message)
      setMutationError(message, false)
      return
    }

    if (source === target) {
      const message = t('graphPanel.workbench.createRelation.errors.sameEntity')
      setErrorMessage(message)
      setMutationError(message, false)
      return
    }

    setIsSubmitting(true)
    setErrorMessage(null)
    clearMutationError()

    const relationData: Record<string, unknown> = {
      description: description.trim(),
      keywords: keywords.trim()
    }
    const parsedWeight = Number(weight)
    if (Number.isFinite(parsedWeight)) {
      relationData.weight = parsedWeight
    }

    try {
      await createGraphRelation(source, target, relationData)
      toast.success(t('graphPanel.workbench.createRelation.messages.created', { source, target }))
      setDescription('')
      setKeywords('')
      setWeight('1')
      useGraphStore.getState().setGraphDataFetchAttempted(false)
      requestRefresh()
      useGraphStore.getState().incrementGraphDataVersion()
    } catch (error) {
      const normalized = normalizeWorkbenchMutationError(
        error,
        t('graphPanel.workbench.createRelation.errors.createFailed')
      )
      setErrorMessage(normalized.message)
      setMutationError(normalized.message, normalized.isConflict)
      toast.error(normalized.message)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-background/60 space-y-3 rounded-lg border p-3">
      <div>
        <h3 className="text-sm font-semibold">{t('graphPanel.workbench.createRelation.title')}</h3>
        <p className="text-muted-foreground mt-1 text-xs">
          {t('graphPanel.workbench.createRelation.description')}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
            {t('graphPanel.workbench.createRelation.fields.source')}
          </label>
          <Input
            value={sourceEntity}
            onChange={(event) => setSourceEntity(event.target.value)}
            placeholder={t('graphPanel.workbench.createRelation.placeholders.source')}
          />
        </div>
        <div className="space-y-1">
          <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
            {t('graphPanel.workbench.createRelation.fields.target')}
          </label>
          <Input
            value={targetEntity}
            onChange={(event) => setTargetEntity(event.target.value)}
            placeholder={t('graphPanel.workbench.createRelation.placeholders.target')}
          />
        </div>
      </div>

      <div className="space-y-1">
        <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
          {t('graphPanel.workbench.createRelation.fields.description')}
        </label>
        <Textarea
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          placeholder={t('graphPanel.workbench.createRelation.placeholders.description')}
          rows={2}
        />
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
            {t('graphPanel.workbench.createRelation.fields.keywords')}
          </label>
          <Input
            value={keywords}
            onChange={(event) => setKeywords(event.target.value)}
            placeholder={t('graphPanel.workbench.createRelation.placeholders.keywords')}
          />
        </div>
        <div className="space-y-1">
          <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
            {t('graphPanel.workbench.createRelation.fields.weight')}
          </label>
          <Input value={weight} onChange={(event) => setWeight(event.target.value)} type="number" step="0.1" />
        </div>
      </div>

      {errorMessage && <p className="text-xs text-red-600 dark:text-red-300">{errorMessage}</p>}

      <div className="flex justify-end">
        <Button type="submit" size="sm" disabled={isSubmitting}>
          {isSubmitting
            ? t('graphPanel.workbench.createRelation.actions.creating')
            : t('graphPanel.workbench.createRelation.actions.create')}
        </Button>
      </div>
    </form>
  )
}

export default CreateRelationForm
