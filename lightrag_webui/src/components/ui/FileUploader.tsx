/**
 * @see https://github.com/sadmann7/file-uploader
 */

import { useControllableState } from '@radix-ui/react-use-controllable-state'
import { FileText, Upload, X } from 'lucide-react'
import * as React from 'react'
import Dropzone, { type DropzoneProps, type FileRejection } from 'react-dropzone'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { supportedFileTypes } from '@/lib/constants'
import { cn } from '@/lib/utils'

interface FileUploaderProps extends React.HTMLAttributes<HTMLDivElement> {
  /**
   * Value of the uploader.
   * @type File[]
   * @default undefined
   * @example value={files}
   */
  value?: File[]

  /**
   * Function to be called when the value changes.
   * @type (files: File[]) => void
   * @default undefined
   * @example onValueChange={(files) => setFiles(files)}
   */
  onValueChange?: (files: File[]) => void

  /**
   * Function to be called when files are uploaded.
   * @type (files: File[]) => Promise<void>
   * @default undefined
   * @example onUpload={(files) => uploadFiles(files)}
   */
  onUpload?: (files: File[]) => Promise<void>

  /**
   * Function to be called when files are rejected.
   * @type (rejections: FileRejection[]) => void
   * @default undefined
   * @example onReject={(rejections) => handleRejectedFiles(rejections)}
   */
  onReject?: (rejections: FileRejection[]) => void

  /**
   * Progress of the uploaded files.
   * @type Record<string, number> | undefined
   * @default undefined
   * @example progresses={{ "file1.png": 50 }}
   */
  progresses?: Record<string, number>

  /**
   * Error messages for failed uploads.
   * @type Record<string, string> | undefined
   * @default undefined
   * @example fileErrors={{ "file1.png": "Upload failed" }}
   */
  fileErrors?: Record<string, string>

  /**
   * Accepted file types for the uploader.
   * @type { [key: string]: string[]}
   * @default
   * ```ts
   * { "text/*": [] }
   * ```
   * @example accept={["text/plain", "application/pdf"]}
   */
  accept?: DropzoneProps['accept']

  /**
   * Maximum file size for the uploader.
   * @type number | undefined
   * @default 1024 * 1024 * 200 // 200MB
   * @example maxSize={1024 * 1024 * 2} // 2MB
   */
  maxSize?: DropzoneProps['maxSize']

  /**
   * Maximum number of files for the uploader.
   * @type number | undefined
   * @default 1
   * @example maxFileCount={4}
   */
  maxFileCount?: DropzoneProps['maxFiles']

  /**
   * Whether the uploader should accept multiple files.
   * @type boolean
   * @default false
   * @example multiple
   */
  multiple?: boolean

  /**
   * Whether the uploader is disabled.
   * @type boolean
   * @default false
   * @example disabled
   */
  disabled?: boolean

  description?: string
}

function formatBytes(
  bytes: number,
  opts: {
    decimals?: number
    sizeType?: 'accurate' | 'normal'
  } = {}
) {
  const { decimals = 0, sizeType = 'normal' } = opts

  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
  const accurateSizes = ['Bytes', 'KiB', 'MiB', 'GiB', 'TiB']
  if (bytes === 0) return '0 Byte'
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${(bytes / 1024 ** i).toFixed(decimals)} ${
    sizeType === 'accurate' ? (accurateSizes[i] ?? 'Bytes') : (sizes[i] ?? 'Bytes')
  }`
}

function FileUploader(props: FileUploaderProps) {
  const { t } = useTranslation()
  const {
    value: valueProp,
    onValueChange,
    onUpload,
    onReject,
    progresses,
    fileErrors,
    accept = supportedFileTypes,
    maxSize = 1024 * 1024 * 200,
    maxFileCount = 1,
    multiple = false,
    disabled = false,
    description,
    className,
    ...dropzoneProps
  } = props

  const [files, setFiles] = useControllableState({
    prop: valueProp,
    defaultProp: [],
    onChange: onValueChange,
  })

  const onDrop = React.useCallback(
    (acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
      // Calculate total file count including both accepted and rejected files
      const totalFileCount = (files?.length ?? 0) + acceptedFiles.length + rejectedFiles.length

      // Check file count limits
      if (!multiple && maxFileCount === 1 && acceptedFiles.length + rejectedFiles.length > 1) {
        toast.error(t('documentPanel.uploadDocuments.fileUploader.singleFileLimit'))
        return
      }

      if (totalFileCount > maxFileCount) {
        toast.error(
          t('documentPanel.uploadDocuments.fileUploader.maxFilesLimit', { count: maxFileCount })
        )
        return
      }

      // Handle rejected files first - this will set error states
      if (rejectedFiles.length > 0) {
        if (onReject) {
          // Use the onReject callback if provided
          onReject(rejectedFiles)
        } else {
          // Fall back to toast notifications if no callback is provided
          rejectedFiles.forEach(({ file }) => {
            toast.error(
              t('documentPanel.uploadDocuments.fileUploader.fileRejected', { name: file.name })
            )
          })
        }
      }

      // Process accepted files
      const newAcceptedFiles = acceptedFiles.map((file) =>
        Object.assign(file, {
          preview: URL.createObjectURL(file),
        })
      )

      // Process rejected files for UI display
      const newRejectedFiles = rejectedFiles.map(({ file }) =>
        Object.assign(file, {
          preview: URL.createObjectURL(file),
          rejected: true,
        })
      )

      // Combine all files for display
      const allNewFiles = [...newAcceptedFiles, ...newRejectedFiles]
      const updatedFiles = files ? [...files, ...allNewFiles] : allNewFiles

      // Update the files state with all files
      setFiles(updatedFiles)

      // Only upload accepted files - make sure we're not uploading rejected files
      if (onUpload && acceptedFiles.length > 0) {
        // Filter out any files that might have been rejected by our custom validator
        const validFiles = acceptedFiles.filter((file) => {
          // Skip files without a name
          if (!file.name) {
            return false
          }

          // Check if file type is accepted
          const fileExt = `.${file.name.split('.').pop()?.toLowerCase() || ''}`
          const isAccepted = Object.entries(accept || {}).some(([mimeType, extensions]) => {
            return (
              file.type === mimeType || (Array.isArray(extensions) && extensions.includes(fileExt))
            )
          })

          // Check file size
          const isSizeValid = file.size <= maxSize

          return isAccepted && isSizeValid
        })

        if (validFiles.length > 0) {
          onUpload(validFiles)
        }
      }
    },
    [files, maxFileCount, multiple, onUpload, onReject, setFiles, t, accept, maxSize]
  )

  function onRemove(index: number) {
    if (!files) return
    const newFiles = files.filter((_, i) => i !== index)
    setFiles(newFiles)
    onValueChange?.(newFiles)
  }

  // Revoke preview url when component unmounts
  React.useEffect(() => {
    return () => {
      if (!files) return
      files.forEach((file) => {
        if (isFileWithPreview(file)) {
          URL.revokeObjectURL(file.preview)
        }
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files])

  const isDisabled = disabled || (files?.length ?? 0) >= maxFileCount

  return (
    <div className="relative flex flex-col gap-6 overflow-hidden">
      <Dropzone
        onDrop={onDrop}
        // remove accept，use customizd validator
        noClick={false}
        noKeyboard={false}
        maxSize={maxSize}
        maxFiles={maxFileCount}
        multiple={maxFileCount > 1 || multiple}
        disabled={isDisabled}
        validator={(file) => {
          // Ensure file name exists
          if (!file.name) {
            return {
              code: 'invalid-file-name',
              message: t('documentPanel.uploadDocuments.fileUploader.invalidFileName', {
                fallback: 'Invalid file name',
              }),
            }
          }

          // Safely extract file extension
          const fileExt = `.${file.name.split('.').pop()?.toLowerCase() || ''}`

          // Ensure accept object exists and has correct format
          const isAccepted = Object.entries(accept || {}).some(([mimeType, extensions]) => {
            // Ensure extensions is an array before calling includes
            return (
              file.type === mimeType || (Array.isArray(extensions) && extensions.includes(fileExt))
            )
          })

          if (!isAccepted) {
            return {
              code: 'file-invalid-type',
              message: t('documentPanel.uploadDocuments.fileUploader.unsupportedType'),
            }
          }

          // Check file size
          if (file.size > maxSize) {
            return {
              code: 'file-too-large',
              message: t('documentPanel.uploadDocuments.fileUploader.fileTooLarge', {
                maxSize: formatBytes(maxSize),
              }),
            }
          }

          return null
        }}
      >
        {({ getRootProps, getInputProps, isDragActive }) => (
          <div
            {...getRootProps()}
            className={cn(
              'group border-muted-foreground/25 hover:bg-muted/25 relative grid h-52 w-full cursor-pointer place-items-center rounded-lg border-2 border-dashed px-5 py-2.5 text-center transition',
              'ring-offset-background focus-visible:ring-ring focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:outline-none',
              isDragActive && 'border-muted-foreground/50',
              isDisabled && 'pointer-events-none opacity-60',
              className
            )}
            {...dropzoneProps}
          >
            <input {...getInputProps()} />
            {isDragActive ? (
              <div className="flex flex-col items-center justify-center gap-4 sm:px-5">
                <div className="rounded-full border border-dashed p-3">
                  <Upload className="text-muted-foreground size-7" aria-hidden="true" />
                </div>
                <p className="text-muted-foreground font-medium">
                  {t('documentPanel.uploadDocuments.fileUploader.dropHere')}
                </p>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center gap-4 sm:px-5">
                <div className="rounded-full border border-dashed p-3">
                  <Upload className="text-muted-foreground size-7" aria-hidden="true" />
                </div>
                <div className="flex flex-col gap-px">
                  <p className="text-muted-foreground font-medium">
                    {t('documentPanel.uploadDocuments.fileUploader.dragAndDrop')}
                  </p>
                  {description ? (
                    <p className="text-muted-foreground/70 text-sm">{description}</p>
                  ) : (
                    <p className="text-muted-foreground/70 text-sm">
                      {t('documentPanel.uploadDocuments.fileUploader.uploadDescription', {
                        count: maxFileCount,
                        isMultiple: maxFileCount === Number.POSITIVE_INFINITY,
                        maxSize: formatBytes(maxSize),
                      })}
                      {t('documentPanel.uploadDocuments.fileTypes')}
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </Dropzone>
      {files?.length ? (
        <ScrollArea className="h-fit w-full px-3">
          <div className="flex max-h-48 flex-col gap-4">
            {files?.map((file, index) => (
              <FileCard
                key={`${file.name}-${file.size}-${index}`}
                file={file}
                onRemove={() => onRemove(index)}
                progress={progresses?.[file.name]}
                error={fileErrors?.[file.name]}
              />
            ))}
          </div>
        </ScrollArea>
      ) : null}
    </div>
  )
}

interface ProgressProps {
  value: number
  error?: boolean
  showIcon?: boolean // New property to control icon display
}

function Progress({ value, error }: ProgressProps) {
  return (
    <div className="relative h-2 w-full">
      <div className="h-full w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={cn('h-full transition-all', error ? 'bg-red-400' : 'bg-primary')}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  )
}

interface FileCardProps {
  file: File
  onRemove: () => void
  progress?: number
  error?: string
}

function FileCard({ file, progress, error, onRemove }: FileCardProps) {
  const { t } = useTranslation()
  return (
    <div className="relative flex items-center gap-2.5">
      <div className="flex flex-1 gap-2.5">
        {error ? (
          <FileText className="text-red-400 size-10" aria-hidden="true" />
        ) : isFileWithPreview(file) ? (
          <FilePreview file={file} />
        ) : null}
        <div className="flex w-full flex-col gap-2">
          <div className="flex flex-col gap-px">
            <p className="text-foreground/80 line-clamp-1 text-sm font-medium">{file.name}</p>
            <p className="text-muted-foreground text-xs">{formatBytes(file.size)}</p>
          </div>
          {error ? (
            <div className="text-red-400 text-sm">
              <div className="relative mb-2">
                <Progress value={100} error={true} />
              </div>
              <p>{error}</p>
            </div>
          ) : progress ? (
            <Progress value={progress} />
          ) : null}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Button type="button" variant="outline" size="icon" className="size-7" onClick={onRemove}>
          <X className="size-4" aria-hidden="true" />
          <span className="sr-only">
            {t('documentPanel.uploadDocuments.fileUploader.removeFile')}
          </span>
        </Button>
      </div>
    </div>
  )
}

function isFileWithPreview(file: File): file is File & { preview: string } {
  return 'preview' in file && typeof file.preview === 'string'
}

interface FilePreviewProps {
  file: File & { preview: string }
}

function FilePreview({ file }: FilePreviewProps) {
  if (file.type.startsWith('image/')) {
    return <div className="aspect-square shrink-0 rounded-md object-cover" />
  }

  return <FileText className="text-muted-foreground size-10" aria-hidden="true" />
}

export default FileUploader
