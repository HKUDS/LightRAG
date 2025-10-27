import { useState, useCallback, useEffect } from 'react'
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
import { deleteDocuments } from '@/api/lightrag'

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
  onDocumentsDeleted?: () => Promise<void>
}

export default function DeleteDocumentsDialog({ selectedDocIds, onDocumentsDeleted }: DeleteDocumentsDialogProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [confirmText, setConfirmText] = useState('')
  const [deleteFile, setDeleteFile] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const [deleteLLMCache, setDeleteLLMCache] = useState(false)
  const isConfirmEnabled = confirmText.toLowerCase() === 'yes' && !isDeleting

  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      setConfirmText('')
      setDeleteFile(false)
      setDeleteLLMCache(false)
      setIsDeleting(false)
    }
  }, [open])

  const handleDelete = useCallback(async () => {
    if (!isConfirmEnabled || selectedDocIds.length === 0) return

    setIsDeleting(true)
    try {
      const result = await deleteDocuments(selectedDocIds, deleteFile, deleteLLMCache)

      if (result.status === 'deletion_started') {
        toast.success(t('documentPanel.deleteDocuments.success', { count: selectedDocIds.length }))
      } else if (result.status === 'busy') {
        toast.error(t('documentPanel.deleteDocuments.busy'))
        setConfirmText('')
        setIsDeleting(false)
        return
      } else if (result.status === 'not_allowed') {
        toast.error(t('documentPanel.deleteDocuments.notAllowed'))
        setConfirmText('')
        setIsDeleting(false)
        return
      } else {
        toast.error(t('documentPanel.deleteDocuments.failed', { message: result.message }))
        setConfirmText('')
        setIsDeleting(false)
        return
      }

      // Refresh document list if provided
      if (onDocumentsDeleted) {
        onDocumentsDeleted().catch(console.error)
      }

      // Close dialog after successful operation
      setOpen(false)
    } catch (err) {
      toast.error(t('documentPanel.deleteDocuments.error', { error: errorMessage(err) }))
      setConfirmText('')
    } finally {
      setIsDeleting(false)
    }
  }, [isConfirmEnabled, selectedDocIds, deleteFile, deleteLLMCache, setOpen, t, onDocumentsDeleted])

  return (
    <Dialog open={open} onOpenChange={setOpen}>
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
          <Button variant="outline" onClick={() => setOpen(false)} disabled={isDeleting}>
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
