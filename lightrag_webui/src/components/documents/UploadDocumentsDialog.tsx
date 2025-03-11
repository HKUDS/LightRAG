import { useState, useCallback } from 'react'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/Dialog'
import FileUploader from '@/components/ui/FileUploader'
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'
import { uploadDocument } from '@/api/lightrag'

import { UploadIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

export default function UploadDocumentsDialog() {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [progresses, setProgresses] = useState<Record<string, number>>({})

  const handleDocumentsUpload = useCallback(
    async (filesToUpload: File[]) => {
      setIsUploading(true)

      try {
        await Promise.all(
          filesToUpload.map(async (file) => {
            try {
              const result = await uploadDocument(file, (percentCompleted: number) => {
                console.debug(t('documentPanel.uploadDocuments.uploading', { name: file.name, percent: percentCompleted }))
                setProgresses((pre) => ({
                  ...pre,
                  [file.name]: percentCompleted
                }))
              })
              if (result.status === 'success') {
                toast.success(t('documentPanel.uploadDocuments.success', { name: file.name }))
              } else {
                toast.error(t('documentPanel.uploadDocuments.failed', { name: file.name, message: result.message }))
              }
            } catch (err) {
              toast.error(t('documentPanel.uploadDocuments.error', { name: file.name, error: errorMessage(err) }))
            }
          })
        )
      } catch (err) {
        toast.error(t('documentPanel.uploadDocuments.generalError', { error: errorMessage(err) }))
      } finally {
        setIsUploading(false)
        // setOpen(false)
      }
    },
    [setIsUploading, setProgresses]
  )

  return (
    <Dialog
      open={open}
      onOpenChange={(open) => {
        if (isUploading && !open) {
          return
        }
        setOpen(open)
      }}
    >
      <DialogTrigger asChild>
        <Button variant="default" side="bottom" tooltip={t('documentPanel.uploadDocuments.tooltip')} size="sm">
          <UploadIcon /> {t('documentPanel.uploadDocuments.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{t('documentPanel.uploadDocuments.title')}</DialogTitle>
          <DialogDescription>
            {t('documentPanel.uploadDocuments.description')}
          </DialogDescription>
        </DialogHeader>
        <FileUploader
          maxFileCount={Infinity}
          maxSize={200 * 1024 * 1024}
          description={t('documentPanel.uploadDocuments.fileTypes')}
          onUpload={handleDocumentsUpload}
          progresses={progresses}
          disabled={isUploading}
        />
      </DialogContent>
    </Dialog>
  )
}
