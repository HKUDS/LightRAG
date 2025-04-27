import { useState, useCallback } from 'react'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter
} from '@/components/ui/Dialog'
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'
import { deleteDocument } from '@/api/lightrag'
import { AlertTriangleIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

interface DeleteDocumentDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  docId: string
  onDocumentDeleted?: () => Promise<void>
}

export default function DeleteDocumentDialog({
  open,
  onOpenChange,
  docId,
  onDocumentDeleted
}: DeleteDocumentDialogProps) {
  const { t } = useTranslation()
  const [isDeleting, setIsDeleting] = useState(false)

  const handleDelete = useCallback(async () => {
    if (!docId || isDeleting) return

    try {
      setIsDeleting(true)
      const result = await deleteDocument(docId)

      if (result.status !== 'success') {
        toast.error(t('documentPanel.deleteDocument.failed', { message: result.message }))
        return
      }

      toast.success(t('documentPanel.deleteDocument.success'))

      // Refresh document list if provided
      if (onDocumentDeleted) {
        try {
          await onDocumentDeleted()
        } catch (err) {
          console.error('Error refreshing documents:', err)
          toast.error(t('documentPanel.deleteDocument.refreshFailed'))
        }
      }

      // Close dialog
      onOpenChange(false)
    } catch (err) {
      toast.error(t('documentPanel.deleteDocument.error', { error: errorMessage(err) }))
    } finally {
      setIsDeleting(false)
    }
  }, [docId, isDeleting, onDocumentDeleted, onOpenChange, t])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-red-500 dark:text-red-400 font-bold">
            <AlertTriangleIcon className="h-5 w-5" />
            {t('documentPanel.deleteDocument.title')}
          </DialogTitle>
          <DialogDescription className="pt-2">
            {t('documentPanel.deleteDocument.description')}
          </DialogDescription>
        </DialogHeader>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isDeleting}
          >
            {t('common.cancel')}
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={isDeleting}
          >
            {isDeleting
              ? t('documentPanel.deleteDocument.deleting')
              : t('documentPanel.deleteDocument.confirmButton')
            }
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
