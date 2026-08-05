import { useState, useCallback } from 'react'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter
} from '@/components/ui/Dialog'
import Input from '@/components/ui/Input'
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'
import { deleteDocuments, deleteKnowledgeBaseDocuments } from '@/api/lightrag'

import { TrashIcon, AlertTriangleIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

// Simple Label component
const Label = ({
  htmlFor,
  className,
  children,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement>) => (
  <label
    htmlFor={htmlFor}
    className={className}
    {...props}
  >
    {children}
  </label>
)

interface DeleteDocumentsDialogProps {
  selectedDocIds: string[]
  /** Per-document KB id mapping so deletes route to the correct workspace */
  docKbIds?: Record<string, string | undefined>
  onDocumentsDeleted?: () => Promise<void>
  /** Knowledge base id to scope the delete to; global delete when omitted */
  kbId?: string
}

export default function DeleteDocumentsDialog({
  selectedDocIds,
  docKbIds,
  onDocumentsDeleted,
  kbId
}: DeleteDocumentsDialogProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [confirmText, setConfirmText] = useState('')
  const [deleteFile, setDeleteFile] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const [deleteLLMCache, setDeleteLLMCache] = useState(false)
  const isConfirmEnabled = confirmText.toLowerCase() === 'yes' && !isDeleting

  // Reset state when dialog closes - handled in onOpenChange to avoid setState in effect
  const handleOpenChange = useCallback((newOpen: boolean) => {
    setOpen(newOpen)
    if (!newOpen) {
      setConfirmText('')
      setDeleteFile(false)
      setDeleteLLMCache(false)
      setIsDeleting(false)
    }
  }, [])

  const handleDelete = useCallback(async () => {
    if (!isConfirmEnabled || selectedDocIds.length === 0) return

    setIsDeleting(true)
    try {
      // Scoped deletes route through the KB endpoint so the destructive
      // pipeline slot is reserved for that KB's workspace; global deletes keep
      // the original /documents/delete_document behaviour.
      if (kbId) {
        const result = await deleteKnowledgeBaseDocuments(kbId, selectedDocIds, false, {
          deleteFile,
          deleteLlmCache: deleteLLMCache
        })
        if (result.status === 'deletion_started') {
          toast.success(t('documentPanel.deleteDocuments.success', { count: selectedDocIds.length }))
          if (onDocumentsDeleted) {
            onDocumentsDeleted().catch(console.error)
          }
          handleOpenChange(false)
          return
        }
        toast.error(t('documentPanel.deleteDocuments.failed', { message: result.message }))
        setConfirmText('')
        setIsDeleting(false)
        return
      }

      // Global view: group selected docs by kb_id, send a KB-scoped
      // delete for each group.
      const byKb = new Map<string, string[]>()
      for (const docId of selectedDocIds) {
        const docKb = (docKbIds || {})[docId] || ''
        const existing = byKb.get(docKb)
        if (existing) {
          existing.push(docId)
        } else {
          byKb.set(docKb, [docId])
        }
      }
      let anyFailed = false
      for (const [workspace, ids] of byKb) {
        const result = workspace
          ? await deleteKnowledgeBaseDocuments(workspace, ids, false, {
              deleteFile,
              deleteLlmCache: deleteLLMCache
            })
          : await deleteDocuments(ids, deleteFile, deleteLLMCache, workspace)
        if (result.status !== 'deletion_started' && result.status !== 'busy') {
          anyFailed = true
          toast.error(t('documentPanel.deleteDocuments.failed', { message: result.message }))
        }
      }
      if (anyFailed) {
        setConfirmText('')
        setIsDeleting(false)
        return
      }
      toast.success(t('documentPanel.deleteDocuments.success', { count: selectedDocIds.length }))
      if (onDocumentsDeleted) {
        onDocumentsDeleted().catch(console.error)
      }
      handleOpenChange(false)
      return
    } catch (err) {
      toast.error(t('documentPanel.deleteDocuments.error', { error: errorMessage(err) }))
      setConfirmText('')
    } finally {
      setIsDeleting(false)
    }
  }, [isConfirmEnabled, selectedDocIds, deleteFile, deleteLLMCache, kbId, handleOpenChange, t, onDocumentsDeleted])

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button
          variant="destructive"
          side="bottom"
          tooltip={t('documentPanel.deleteDocuments.tooltip', { count: selectedDocIds.length })}
          size="sm"
        >
          <TrashIcon/> {t('documentPanel.deleteDocuments.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-red-500 dark:text-red-400 font-bold">
            <AlertTriangleIcon className="h-5 w-5" />
            {t('documentPanel.deleteDocuments.title')}
          </DialogTitle>
          <DialogDescription className="pt-2">
            {t('documentPanel.deleteDocuments.description', { count: selectedDocIds.length })}
          </DialogDescription>
        </DialogHeader>

        <div className="text-red-500 dark:text-red-400 font-semibold mb-4">
          {t('documentPanel.deleteDocuments.warning')}
        </div>

        <div className="mb-4">
          {t('documentPanel.deleteDocuments.confirm', { count: selectedDocIds.length })}
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="confirm-text" className="text-sm font-medium">
              {t('documentPanel.deleteDocuments.confirmPrompt')}
            </Label>
            <Input
              id="confirm-text"
              value={confirmText}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setConfirmText(e.target.value)}
              placeholder={t('documentPanel.deleteDocuments.confirmPlaceholder')}
              className="w-full"
              disabled={isDeleting}
            />
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="delete-file"
              checked={deleteFile}
              onChange={(e) => setDeleteFile(e.target.checked)}
              disabled={isDeleting}
              className="h-4 w-4 text-red-600 focus:ring-red-500 border-gray-300 rounded"
            />
            <Label htmlFor="delete-file" className="text-sm font-medium cursor-pointer">
              {t('documentPanel.deleteDocuments.deleteFileOption')}
            </Label>
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="delete-llm-cache"
              checked={deleteLLMCache}
              onChange={(e) => setDeleteLLMCache(e.target.checked)}
              disabled={isDeleting}
              className="h-4 w-4 text-red-600 focus:ring-red-500 border-gray-300 rounded"
            />
            <Label htmlFor="delete-llm-cache" className="text-sm font-medium cursor-pointer">
              {t('documentPanel.deleteDocuments.deleteLLMCacheOption')}
            </Label>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => handleOpenChange(false)} disabled={isDeleting}>
            {t('common.cancel')}
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={!isConfirmEnabled}
          >
            {isDeleting ? t('documentPanel.deleteDocuments.deleting') : t('documentPanel.deleteDocuments.confirmButton')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
