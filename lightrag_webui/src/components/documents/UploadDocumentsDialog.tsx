import { useState, useCallback } from 'react'
import { FileRejection } from 'react-dropzone'
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
import EntityTypeConfig from '@/components/documents/EntityTypeConfig'
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'
import { uploadDocument } from '@/api/lightrag'

import { UploadIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

interface UploadDocumentsDialogProps {
  onDocumentsUploaded?: () => Promise<void>
  defaultEntityTypes?: string[]
}

export default function UploadDocumentsDialog({
  onDocumentsUploaded,
  defaultEntityTypes = []
}: UploadDocumentsDialogProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [progresses, setProgresses] = useState<Record<string, number>>({})
  const [fileErrors, setFileErrors] = useState<Record<string, string>>({})
  const [entityTypes, setEntityTypes] = useState<string[]>([])
  const [pendingFiles, setPendingFiles] = useState<File[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<Set<string>>(new Set())

  // 当对话框关闭时，重置实体类型配置
  const handleOpenChange = (newOpen: boolean) => {
    if (!newOpen && !isUploading) {
      // 对话框关闭时重置所有状态
      setProgresses({})
      setFileErrors({})
      setEntityTypes([])
      setPendingFiles([])
      setUploadedFiles(new Set())
    }
    setOpen(newOpen)
  }

  const handleRejectedFiles = useCallback(
    (rejectedFiles: FileRejection[]) => {
      // Process rejected files and add them to fileErrors
      rejectedFiles.forEach(({ file, errors }) => {
        // Get the first error message
        let errorMsg = errors[0]?.message || t('documentPanel.uploadDocuments.fileUploader.fileRejected', { name: file.name })

        // Simplify error message for unsupported file types
        if (errorMsg.includes('file-invalid-type')) {
          errorMsg = t('documentPanel.uploadDocuments.fileUploader.unsupportedType')
        }

        // Set progress to 100% to display error message
        setProgresses((pre) => ({
          ...pre,
          [file.name]: 100
        }))

        // Add error message to fileErrors
        setFileErrors(prev => ({
          ...prev,
          [file.name]: errorMsg
        }))
      })
    },
    [setProgresses, setFileErrors, t]
  )

  const handleDocumentsUpload = useCallback(
    async (filesToUpload: File[]) => {
      setIsUploading(true)
      let hasSuccessfulUpload = false

      // Only clear errors for files that are being uploaded, keep errors for rejected files
      setFileErrors(prev => {
        const newErrors = { ...prev };
        filesToUpload.forEach(file => {
          delete newErrors[file.name];
        });
        return newErrors;
      });

      // Show uploading toast
      const toastId = toast.loading(t('documentPanel.uploadDocuments.batch.uploading'))

      try {
        // Track errors locally to ensure we have the final state
        const uploadErrors: Record<string, string> = {}

        // Create a collator that supports Chinese sorting
        const collator = new Intl.Collator(['zh-CN', 'en'], {
          sensitivity: 'accent',  // consider basic characters, accents, and case
          numeric: true           // enable numeric sorting, e.g., "File 10" will be after "File 2"
        });
        const sortedFiles = [...filesToUpload].sort((a, b) =>
          collator.compare(a.name, b.name)
        );

        // Upload files in sequence, not parallel
        for (const file of sortedFiles) {
          try {
            // Initialize upload progress
            setProgresses((pre) => ({
              ...pre,
              [file.name]: 0
            }))

            const result = await uploadDocument(file, (percentCompleted: number) => {
              console.debug(t('documentPanel.uploadDocuments.single.uploading', { name: file.name, percent: percentCompleted }))
              setProgresses((pre) => ({
                ...pre,
                [file.name]: percentCompleted
              }))
            }, entityTypes.length > 0 ? entityTypes : undefined)

            if (result.status === 'duplicated') {
              uploadErrors[file.name] = t('documentPanel.uploadDocuments.fileUploader.duplicateFile')
              setFileErrors(prev => ({
                ...prev,
                [file.name]: t('documentPanel.uploadDocuments.fileUploader.duplicateFile')
              }))
            } else if (result.status !== 'success') {
              uploadErrors[file.name] = result.message
              setFileErrors(prev => ({
                ...prev,
                [file.name]: result.message
              }))
            } else {
              // Mark that we had at least one successful upload
              hasSuccessfulUpload = true
              // Mark file as uploaded
              setUploadedFiles(prev => new Set(prev).add(file.name))
            }
          } catch (err) {
            console.error(`Upload failed for ${file.name}:`, err)

            // Handle HTTP errors, including 400 errors
            let errorMsg = errorMessage(err)

            // If it's an axios error with response data, try to extract more detailed error info
            if (err && typeof err === 'object' && 'response' in err) {
              const axiosError = err as { response?: { status: number, data?: { detail?: string } } }
              if (axiosError.response?.status === 400) {
                // Extract specific error message from backend response
                errorMsg = axiosError.response.data?.detail || errorMsg
              }

              // Set progress to 100% to display error message
              setProgresses((pre) => ({
                ...pre,
                [file.name]: 100
              }))
            }

            // Record error message in both local tracking and state
            uploadErrors[file.name] = errorMsg
            setFileErrors(prev => ({
              ...prev,
              [file.name]: errorMsg
            }))
          }
        }

        // Check if any files failed to upload using our local tracking
        const hasErrors = Object.keys(uploadErrors).length > 0

        // Update toast status
        if (hasErrors) {
          toast.error(t('documentPanel.uploadDocuments.batch.error'), { id: toastId })
        } else {
          toast.success(t('documentPanel.uploadDocuments.batch.success'), { id: toastId })
        }

        // Only update if at least one file was uploaded successfully
        if (hasSuccessfulUpload) {
          // Clear pending files after successful upload
          setPendingFiles([])
          setProgresses({})
          setFileErrors({})
          // Refresh document list
          if (onDocumentsUploaded) {
            onDocumentsUploaded().catch(err => {
              console.error('Error refreshing documents:', err)
            })
          }
        }
      } catch (err) {
        console.error('Unexpected error during upload:', err)
        toast.error(t('documentPanel.uploadDocuments.generalError', { error: errorMessage(err) }), { id: toastId })
      } finally {
        setIsUploading(false)
      }
    },
    [setIsUploading, setProgresses, setFileErrors, t, onDocumentsUploaded, entityTypes]
  )

  // Handle files selected (not yet uploaded)
  const handleFilesSelected = useCallback((files: File[]) => {
    setPendingFiles(files)
    setProgresses({})
    setFileErrors({})
  }, [])

  // Handle upload button click
  const handleUploadButtonClick = useCallback(() => {
    if (pendingFiles.length > 0) {
      handleDocumentsUpload(pendingFiles)
    }
  }, [pendingFiles, handleDocumentsUpload])

  return (
    <Dialog
      open={open}
      onOpenChange={handleOpenChange}
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
        <div className="space-y-4">
          <EntityTypeConfig
            entityTypes={entityTypes}
            onEntityTypesChange={setEntityTypes}
            defaultEntityTypes={defaultEntityTypes}
          />
          <FileUploader
            value={pendingFiles}
            onValueChange={handleFilesSelected}
            maxFileCount={Infinity}
            maxSize={200 * 1024 * 1024}
            description={t('documentPanel.uploadDocuments.fileTypes')}
            onReject={handleRejectedFiles}
            progresses={progresses}
            fileErrors={fileErrors}
            disabled={isUploading}
            multiple
          />
          {pendingFiles.length > 0 && (
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setPendingFiles([])
                  setProgresses({})
                  setFileErrors({})
                }}
                disabled={isUploading}
              >
                {t('documentPanel.uploadDocuments.cancel')}
              </Button>
              <Button
                type="button"
                variant="default"
                onClick={handleUploadButtonClick}
                disabled={isUploading || pendingFiles.length === 0}
              >
                {isUploading ? t('documentPanel.uploadDocuments.uploading') : t('documentPanel.uploadDocuments.startUpload')}
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
