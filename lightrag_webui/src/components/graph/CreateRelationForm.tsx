import { FormEvent, useEffect, useState } from 'react'
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
      const message = 'Source and target entity are required.'
      setErrorMessage(message)
      setMutationError(message, false)
      return
    }

    if (source === target) {
      const message = 'Source and target cannot be the same entity.'
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
      toast.success(`Relation "${source} -> ${target}" created.`)
      setDescription('')
      setKeywords('')
      setWeight('1')
      useGraphStore.getState().setGraphDataFetchAttempted(false)
      requestRefresh()
      useGraphStore.getState().incrementGraphDataVersion()
    } catch (error) {
      const normalized = normalizeWorkbenchMutationError(error, 'Create relation failed')
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
        <h3 className="text-sm font-semibold">Create Relation</h3>
        <p className="text-muted-foreground mt-1 text-xs">
          Source/target prefer current selection and can be overridden manually.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
            Source
          </label>
          <Input
            value={sourceEntity}
            onChange={(event) => setSourceEntity(event.target.value)}
            placeholder="Elon Musk"
          />
        </div>
        <div className="space-y-1">
          <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
            Target
          </label>
          <Input
            value={targetEntity}
            onChange={(event) => setTargetEntity(event.target.value)}
            placeholder="Tesla"
          />
        </div>
      </div>

      <div className="space-y-1">
        <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
          Description
        </label>
        <Textarea
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          placeholder="Elon Musk works for Tesla"
          rows={2}
        />
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
            Keywords
          </label>
          <Input
            value={keywords}
            onChange={(event) => setKeywords(event.target.value)}
            placeholder="works_for"
          />
        </div>
        <div className="space-y-1">
          <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
            Weight
          </label>
          <Input value={weight} onChange={(event) => setWeight(event.target.value)} type="number" step="0.1" />
        </div>
      </div>

      {errorMessage && <p className="text-xs text-red-600 dark:text-red-300">{errorMessage}</p>}

      <div className="flex justify-end">
        <Button type="submit" size="sm" disabled={isSubmitting}>
          {isSubmitting ? 'Creating...' : 'Create Relation'}
        </Button>
      </div>
    </form>
  )
}

export default CreateRelationForm
