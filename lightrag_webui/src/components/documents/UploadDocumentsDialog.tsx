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
  // Track upload errors for each file
  const [fileErrors, setFileErrors] = useState<Record<string, string>>({})

  const handleDocumentsUpload = useCallback(
    async (filesToUpload: File[]) => {
      setIsUploading(true)

      // Reset error states before new upload
      setFileErrors({})
      
      try {
        // Use a single toast for the entire batch upload process
        toast.promise(
          (async () => {
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
                    
                    // Store error message if upload failed
                    if (result.status !== 'success') {
                      setFileErrors(prev => ({
                        ...prev,
                        [file.name]: result.message
                      }))
                    }
                  } catch (err) {
                    // Store error message from exception
                    setFileErrors(prev => ({
                      ...prev,
                      [file.name]: errorMessage(err)
                    }))
                  }
                })
              )
              // Keep dialog open to show final status
              // User needs to close dialog manually
            } catch (error) {
              console.error('Upload failed:', error)
            }
          })(),
          {
            loading: t('documentPanel.uploadDocuments.uploading.batch'),
            success: t('documentPanel.uploadDocuments.success.batch'),
            error: t('documentPanel.uploadDocuments.error.batch')
          }
        )
      } catch (err) {
        // Handle general upload errors
        toast.error(`Upload error: ${errorMessage(err)}`)
      } finally {
        setIsUploading(false)
      }
    },
    [setIsUploading, setProgresses, t]
  )

  return (
    <Dialog
      open={open}
      onOpenChange={(open) => {
        // Prevent closing dialog during upload
        if (isUploading) {
          return
        }
        if (!open) {
          // Reset states when dialog is closed
          setProgresses({})
          setFileErrors({})
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
          fileErrors={fileErrors}
          disabled={isUploading}
        />
      </DialogContent>
    </Dialog>
  )
}
