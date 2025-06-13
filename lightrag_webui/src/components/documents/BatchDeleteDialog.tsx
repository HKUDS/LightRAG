import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import Progress from '@/components/ui/Progress'
import { deleteDocumentsBatch, type DocStatusResponse } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { TrashIcon, AlertTriangleIcon } from 'lucide-react'

interface BatchDeleteDialogProps {
  documents: DocStatusResponse[]
  open: boolean
  onOpenChange: (open: boolean) => void
  onDocumentsDeleted: () => void
}

export default function BatchDeleteDialog({
  documents,
  open,
  onOpenChange,
  onDocumentsDeleted
}: BatchDeleteDialogProps) {
  const { t } = useTranslation()
  const [isDeleting, setIsDeleting] = useState(false)
  const [confirmationText, setConfirmationText] = useState('')
  const [progress, setProgress] = useState(0)

  const handleDelete = async () => {
    if (confirmationText !== 'DELETE') {
      toast.error(t('documentPanel.batchDelete.confirmationError'))
      return
    }

    setIsDeleting(true)
    setProgress(0)

    try {
      const documentsToDelete = documents.map((doc) => ({
        doc_id: doc.id,
        file_name: doc.file_path
      }))
      const result = await deleteDocumentsBatch(documentsToDelete)

      // Update progress to 100% when complete
      setProgress(100)

      if (result.overall_status === 'success') {
        toast.success(
          t('documentPanel.batchDelete.success', {
            count: result.deleted_count
          })
        )
      } else if (result.overall_status === 'partial_success') {
        toast.warning(
          t('documentPanel.batchDelete.partialSuccess', {
            deleted: result.deleted_count,
            total: documents.length,
            failed: result.failed_count
          })
        )
      } else {
        toast.error(
          t('documentPanel.batchDelete.failure', {
            message: result.message
          })
        )
      }

      // Show detailed results if there were any failures
      if (result.failed_count > 0) {
        const failedDocs = result.results
          .filter((r) => r.status !== 'success')
          .map((r) => r.doc_id)
          .slice(0, 3) // Show first 3 failed docs

        if (failedDocs.length > 0) {
          const more = result.failed_count > 3 ? ` (+${result.failed_count - 3} more)` : ''
          toast.error(
            t('documentPanel.batchDelete.failedDocs', {
              docs: failedDocs.join(', ') + more
            })
          )
        }
      }

      onDocumentsDeleted()
      onOpenChange(false)
    } catch (error) {
      toast.error(
        t('documentPanel.batchDelete.error', {
          error: errorMessage(error)
        })
      )
      setProgress(0)
    } finally {
      setIsDeleting(false)
      setConfirmationText('')
    }
  }

  const getDisplayFileName = (doc: DocStatusResponse): string => {
    if (!doc.file_path || doc.file_path.trim() === '') {
      return doc.id
    }
    const parts = doc.file_path.split('/')
    return parts[parts.length - 1] || doc.id
  }

  const isConfirmationValid = confirmationText === 'DELETE'

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="overflow-hidden p-6 sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <TrashIcon className="h-5 w-5 text-red-500" />
            {t('documentPanel.batchDelete.title')}
          </DialogTitle>
          <DialogDescription>
            {t('documentPanel.batchDelete.description', { count: documents.length })}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Document List Preview */}
          <div className="overflow-hidden rounded-lg border bg-gray-50 p-4 dark:bg-gray-800">
            <div className="max-h-32 space-y-1 overflow-y-auto rounded border p-2">
              {documents.slice(0, 5).map((doc) => (
                <div key={doc.id} className="flex items-center justify-between text-sm">
                  <span className="mr-2 ml-2 flex-1 font-mono text-sm break-all text-gray-900 dark:text-gray-100">
                    {getDisplayFileName(doc)}
                  </span>
                  <span className="text-xs text-gray-500">
                    {doc.status === 'processed' && (
                      <span className="text-green-600">
                        {t('documentPanel.documentManager.status.completed')}
                      </span>
                    )}
                    {doc.status === 'processing' && (
                      <span className="text-blue-600">
                        {t('documentPanel.documentManager.status.processing')}
                      </span>
                    )}
                    {doc.status === 'pending' && (
                      <span className="text-yellow-600">
                        {t('documentPanel.documentManager.status.pending')}
                      </span>
                    )}
                    {doc.status === 'failed' && (
                      <span className="text-red-600">
                        {t('documentPanel.documentManager.status.failed')}
                      </span>
                    )}
                  </span>
                </div>
              ))}
              {documents.length > 5 && (
                <div className="text-sm text-gray-500 italic">
                  {t('documentPanel.batchDelete.andMore', { count: documents.length - 5 })}
                </div>
              )}
            </div>
          </div>

          {/* Warning */}
          <div className="rounded-lg border border-red-200 bg-red-50 p-3 dark:border-red-700 dark:bg-red-900/20">
            <div className="flex items-start gap-2">
              <AlertTriangleIcon className="mt-0.5 h-5 w-5 flex-shrink-0 text-red-600 dark:text-red-400" />
              <div className="space-y-1">
                <p className="text-sm font-medium text-red-800 dark:text-red-200">
                  {t('documentPanel.batchDelete.warning')}
                </p>
                <p className="text-sm text-red-700 dark:text-red-300">
                  {t('documentPanel.batchDelete.warningText')}
                </p>
              </div>
            </div>
          </div>

          {/* Confirmation Input */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {t('documentPanel.batchDelete.confirmationLabel')}
            </label>
            <Input
              type="text"
              value={confirmationText}
              onChange={(e) => setConfirmationText(e.target.value)}
              placeholder="DELETE"
              disabled={isDeleting}
              className={confirmationText && !isConfirmationValid ? 'border-red-300' : ''}
            />
            <p className="text-xs text-gray-500">
              {t('documentPanel.batchDelete.confirmationHint')}
            </p>
          </div>

          {/* Progress Bar */}
          {isDeleting && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>{t('documentPanel.batchDelete.deleting')}</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="w-full" />
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={isDeleting}>
            {t('common.cancel')}
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={isDeleting || !isConfirmationValid}
          >
            {isDeleting ? (
              <>
                <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                {t('documentPanel.batchDelete.deleting')}
              </>
            ) : (
              <>
                <TrashIcon className="mr-2 h-4 w-4" />
                {t('documentPanel.batchDelete.confirm', { count: documents.length })}
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
