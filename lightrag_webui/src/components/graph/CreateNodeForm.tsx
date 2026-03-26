import { FormEvent, useState } from 'react'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import { createGraphEntity } from '@/api/lightrag'
import { useGraphStore } from '@/stores/graph'
import {
  normalizeWorkbenchMutationError,
  useGraphWorkbenchStore
} from '@/stores/graphWorkbench'

const CreateNodeForm = () => {
  const [entityName, setEntityName] = useState('')
  const [description, setDescription] = useState('')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const setMutationError = useGraphWorkbenchStore.use.setMutationError()
  const clearMutationError = useGraphWorkbenchStore.use.clearMutationError()
  const requestRefresh = useGraphWorkbenchStore.use.requestRefresh()

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (isSubmitting) return

    const trimmedEntity = entityName.trim()
    const trimmedDescription = description.trim()
    if (!trimmedEntity || !trimmedDescription) {
      const message = 'Entity name and description are required.'
      setErrorMessage(message)
      setMutationError(message, false)
      return
    }

    setIsSubmitting(true)
    setErrorMessage(null)
    clearMutationError()

    try {
      await createGraphEntity(trimmedEntity, { description: trimmedDescription })
      toast.success(`Entity "${trimmedEntity}" created.`)
      setEntityName('')
      setDescription('')
      useGraphStore.getState().setGraphDataFetchAttempted(false)
      requestRefresh()
      useGraphStore.getState().incrementGraphDataVersion()
    } catch (error) {
      const normalized = normalizeWorkbenchMutationError(error, 'Create node failed')
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
        <h3 className="text-sm font-semibold">Create Node</h3>
        <p className="text-muted-foreground mt-1 text-xs">Minimal payload: entity name + description.</p>
      </div>

      <div className="space-y-1">
        <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
          Entity Name
        </label>
        <Input value={entityName} onChange={(event) => setEntityName(event.target.value)} placeholder="Tesla" />
      </div>

      <div className="space-y-1">
        <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
          Description
        </label>
        <Textarea
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          placeholder="Electric vehicle manufacturer"
          rows={3}
        />
      </div>

      {errorMessage && <p className="text-xs text-red-600 dark:text-red-300">{errorMessage}</p>}

      <div className="flex justify-end">
        <Button type="submit" size="sm" disabled={isSubmitting}>
          {isSubmitting ? 'Creating...' : 'Create Node'}
        </Button>
      </div>
    </form>
  )
}

export default CreateNodeForm
