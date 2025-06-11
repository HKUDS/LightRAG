import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/Dialog'
import { deleteDocument, type DocStatusResponse } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { TrashIcon } from 'lucide-react'

interface DeleteDocumentDialogProps {
  document: DocStatusResponse | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onDocumentDeleted: () => void
}

export default function DeleteDocumentDialog({
  document,
  open,
  onOpenChange,
  onDocumentDeleted
}: DeleteDocumentDialogProps) {
  const { t } = useTranslation()
  const [isDeleting, setIsDeleting] = useState(false)

  const handleDelete = async () => {
    if (!document) return

    setIsDeleting(true)
    try {
      const result = await deleteDocument(document.id, document.file_path)
      
      if (result.status === 'success') {
        toast.success(t('documentPanel.deleteDocument.success', { 
          fileName: document.file_path || document.id 
        }))
        onDocumentDeleted()
        onOpenChange(false)
      } else if (result.status === 'not_found') {
        toast.warning(t('documentPanel.deleteDocument.notFound'))
        onDocumentDeleted() // Refresh the list since document doesn't exist
        onOpenChange(false)
      } else if (result.status === 'busy') {
        toast.error(t('documentPanel.deleteDocument.busy'))
      } else {
        toast.error(t('documentPanel.deleteDocument.error', { 
          error: result.message 
        }))
      }
    } catch (error) {
      toast.error(t('documentPanel.deleteDocument.error', { 
        error: errorMessage(error) 
      }))
    } finally {
      setIsDeleting(false)
    }
  }

  const getDisplayFileName = (doc: DocStatusResponse): string => {
    if (!doc.file_path || doc.file_path.trim() === '') {
      return doc.id
    }
    const parts = doc.file_path.split('/')
    return parts[parts.length - 1] || doc.id
  }

  if (!document) return null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px] overflow-hidden p-6">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <TrashIcon className="h-5 w-5 text-red-500" />
            {t('documentPanel.deleteDocument.title')}
          </DialogTitle>
          <DialogDescription>
            {t('documentPanel.deleteDocument.description')}
          </DialogDescription>
        </DialogHeader>

        <div className="py-4">
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 border overflow-hidden">
            <div className="space-y-2">
              <div>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {t('documentPanel.deleteDocument.fileName')}:
                </span>
                <span className="ml-2 text-sm text-gray-900 dark:text-gray-100 font-mono break-all">
                  {getDisplayFileName(document)}
                </span>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {t('documentPanel.deleteDocument.documentId')}:
                </span>
                <span className="ml-2 text-sm text-gray-900 dark:text-gray-100 font-mono truncate block max-w-full">
                  {document.id}
                </span>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {t('documentPanel.deleteDocument.status')}:
                </span>
                <span className="ml-2 text-sm">
                  {document.status === 'processed' && (
                    <span className="text-green-600">{t('documentPanel.documentManager.status.completed')}</span>
                  )}
                  {document.status === 'processing' && (
                    <span className="text-blue-600">{t('documentPanel.documentManager.status.processing')}</span>
                  )}
                  {document.status === 'pending' && (
                    <span className="text-yellow-600">{t('documentPanel.documentManager.status.pending')}</span>
                  )}
                  {document.status === 'failed' && (
                    <span className="text-red-600">{t('documentPanel.documentManager.status.failed')}</span>
                  )}
                </span>
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 rounded-lg">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <strong>{t('documentPanel.deleteDocument.warning')}:</strong>{' '}
              {t('documentPanel.deleteDocument.warningText')}
            </p>
          </div>
        </div>

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
            {isDeleting ? (
              <>
                <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                {t('documentPanel.deleteDocument.deleting')}
              </>
            ) : (
              <>
                <TrashIcon className="mr-2 h-4 w-4" />
                {t('documentPanel.deleteDocument.confirm')}
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}