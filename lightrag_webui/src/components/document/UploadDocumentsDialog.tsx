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

export default function UploadDocumentsDialog() {
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
                console.debug(`Uploading ${file.name}: ${percentCompleted}%`)
                setProgresses((pre) => ({
                  ...pre,
                  [file.name]: percentCompleted
                }))
              })
              if (result.status === 'success') {
                toast.success(`Upload Success:\n${file.name} uploaded successfully`)
              } else {
                toast.error(`Upload Failed:\n${file.name}\n${result.message}`)
              }
            } catch (err) {
              toast.error(`Upload Failed:\n${file.name}\n${errorMessage(err)}`)
            }
          })
        )
      } catch (err) {
        toast.error('Upload Failed\n' + errorMessage(err))
      } finally {
        setIsUploading(false)
        setOpen(false)
      }
    },
    [setIsUploading, setProgresses, setOpen]
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
        <Button variant="outline" tooltip="Upload documents" side="bottom" size="icon">
          <UploadIcon />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>Upload documents</DialogTitle>
          <DialogDescription>
            Drag and drop your documents here or click to browse.
          </DialogDescription>
        </DialogHeader>
        <FileUploader
          maxFileCount={Infinity}
          maxSize={200 * 1024 * 1024}
          description="supported types: TXT, MD, DOC, PDF, PPTX"
          onUpload={handleDocumentsUpload}
          progresses={progresses}
          disabled={isUploading}
        />
      </DialogContent>
    </Dialog>
  )
}
