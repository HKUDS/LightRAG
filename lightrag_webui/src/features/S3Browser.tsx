import type { S3ObjectInfo } from '@/api/lightrag'
import { s3Delete, s3Download, s3List, s3Upload } from '@/api/lightrag'
import FileViewer from '@/components/storage/FileViewer'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/AlertDialog'
import Button from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/Table'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ChevronRightIcon,
  DownloadIcon,
  EyeIcon,
  FileIcon,
  FolderIcon,
  HomeIcon,
  RefreshCwIcon,
  Trash2Icon,
  UploadIcon,
} from 'lucide-react'
import { useCallback, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'

// Format bytes to human readable size
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`
}

// Format ISO date to localized string
function formatDate(isoDate: string): string {
  try {
    return new Date(isoDate).toLocaleString()
  } catch {
    return isoDate
  }
}

// Extract display name from full key or folder path
function getDisplayName(keyOrPath: string, prefix: string): string {
  // Remove prefix to get relative path
  const relative = keyOrPath.startsWith(prefix) ? keyOrPath.slice(prefix.length) : keyOrPath
  // For folders, remove trailing slash
  return relative.endsWith('/') ? relative.slice(0, -1) : relative
}

// Parse prefix into breadcrumb segments
function parseBreadcrumbs(prefix: string): { name: string; path: string }[] {
  const segments: { name: string; path: string }[] = [{ name: 'Root', path: '' }]
  if (!prefix) return segments

  const parts = prefix.split('/').filter(Boolean)
  let currentPath = ''
  for (const part of parts) {
    currentPath += part + '/'
    segments.push({ name: part, path: currentPath })
  }
  return segments
}

export default function S3Browser() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Current prefix for navigation
  const [prefix, setPrefix] = useState('')

  // Delete confirmation state
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)

  // File viewer state
  const [viewTarget, setViewTarget] = useState<S3ObjectInfo | null>(null)

  // Fetch objects at current prefix
  const {
    data: listData,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ['s3', 'list', prefix],
    queryFn: () => s3List(prefix),
  })

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: async (file: File) => s3Upload(prefix, file),
    onSuccess: (data) => {
      toast.success(t('storagePanel.uploadSuccess', { name: getDisplayName(data.key, prefix) }))
      queryClient.invalidateQueries({ queryKey: ['s3', 'list', prefix] })
    },
    onError: (err: Error) => {
      toast.error(t('storagePanel.uploadFailed', { error: err.message }))
    },
  })

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: async (key: string) => s3Delete(key),
    onSuccess: (data) => {
      toast.success(t('storagePanel.deleteSuccess', { name: getDisplayName(data.key, prefix) }))
      queryClient.invalidateQueries({ queryKey: ['s3', 'list', prefix] })
      setDeleteTarget(null)
    },
    onError: (err: Error) => {
      toast.error(t('storagePanel.deleteFailed', { error: err.message }))
      setDeleteTarget(null)
    },
  })

  // Navigate to a folder
  const navigateToFolder = useCallback((folderPath: string) => {
    setPrefix(folderPath)
  }, [])

  // Handle download click
  const handleDownload = useCallback(async (key: string) => {
    try {
      const response = await s3Download(key)
      // Open presigned URL in new tab
      window.open(response.url, '_blank')
    } catch (err) {
      toast.error(t('storagePanel.downloadFailed', { error: err instanceof Error ? err.message : 'Unknown error' }))
    }
  }, [t])

  // Handle file upload
  const handleUpload = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  // Handle file selection
  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = event.target.files
      if (files && files.length > 0) {
        uploadMutation.mutate(files[0])
      }
      // Reset input so the same file can be uploaded again
      event.target.value = ''
    },
    [uploadMutation]
  )

  // Handle view click
  const handleView = useCallback((obj: S3ObjectInfo) => {
    setViewTarget(obj)
  }, [])

  // Handle delete click
  const handleDelete = useCallback((key: string) => {
    setDeleteTarget(key)
  }, [])

  // Confirm delete
  const confirmDelete = useCallback(() => {
    if (deleteTarget) {
      deleteMutation.mutate(deleteTarget)
    }
  }, [deleteTarget, deleteMutation])

  // Cancel delete
  const cancelDelete = useCallback(() => {
    setDeleteTarget(null)
  }, [])

  const breadcrumbs = parseBreadcrumbs(prefix)
  const folders = listData?.folders || []
  const objects = listData?.objects || []

  return (
    <div className="h-full flex flex-col p-4 gap-4 overflow-hidden">
      <Card className="flex-1 overflow-hidden flex flex-col">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-medium">{t('storagePanel.title')}</CardTitle>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleUpload}
                disabled={uploadMutation.isPending}
              >
                <UploadIcon className="h-4 w-4 mr-1" />
                {t('storagePanel.actions.upload')}
              </Button>
              <Button variant="outline" size="icon" onClick={() => refetch()}>
                <RefreshCwIcon className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </div>

          {/* Breadcrumb navigation */}
          <div className="flex items-center gap-1 text-sm text-muted-foreground mt-2 flex-wrap">
            {breadcrumbs.map((segment, index) => (
              <div key={segment.path} className="flex items-center">
                {index > 0 && <ChevronRightIcon className="h-4 w-4 mx-1" />}
                <button
                  type="button"
                  className={`hover:text-foreground transition-colors ${
                    index === breadcrumbs.length - 1
                      ? 'font-medium text-foreground'
                      : 'hover:underline'
                  }`}
                  onClick={() => navigateToFolder(segment.path)}
                >
                  {index === 0 ? (
                    <span className="flex items-center gap-1">
                      <HomeIcon className="h-4 w-4" />
                      {listData?.bucket || segment.name}
                    </span>
                  ) : (
                    segment.name
                  )}
                </button>
              </div>
            ))}
          </div>
        </CardHeader>

        <CardContent className="flex-1 p-0 overflow-auto">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <RefreshCwIcon className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : isError ? (
            <div className="flex flex-col items-center justify-center h-full text-destructive gap-2">
              <p className="font-medium">{t('storagePanel.loadFailed')}</p>
              <p className="text-sm text-muted-foreground">
                {error instanceof Error ? error.message : 'Unknown error'}
              </p>
              <Button variant="outline" size="sm" onClick={() => refetch()} className="mt-2">
                {t('storagePanel.actions.retry')}
              </Button>
            </div>
          ) : folders.length === 0 && objects.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
              <FolderIcon className="h-12 w-12 mb-2 opacity-50" />
              <p>{t('storagePanel.empty')}</p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[50%]">{t('storagePanel.table.name')}</TableHead>
                  <TableHead className="w-[15%]">{t('storagePanel.table.size')}</TableHead>
                  <TableHead className="w-[20%]">{t('storagePanel.table.modified')}</TableHead>
                  <TableHead className="w-[15%] text-right">{t('storagePanel.table.actions')}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {/* Folders */}
                {folders.map((folder) => (
                  <TableRow
                    key={folder}
                    className="cursor-pointer hover:bg-muted/50"
                    onClick={() => navigateToFolder(folder)}
                  >
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        <FolderIcon className="h-4 w-4 text-yellow-500" />
                        {getDisplayName(folder, prefix)}
                      </div>
                    </TableCell>
                    <TableCell className="text-muted-foreground">-</TableCell>
                    <TableCell className="text-muted-foreground">-</TableCell>
                    <TableCell></TableCell>
                  </TableRow>
                ))}

                {/* Objects */}
                {objects.map((obj: S3ObjectInfo) => (
                  <TableRow key={obj.key}>
                    <TableCell className="font-medium">
                      <button
                        type="button"
                        className="flex items-center gap-2 hover:text-primary transition-colors text-left"
                        onClick={() => handleView(obj)}
                      >
                        <FileIcon className="h-4 w-4 text-blue-500 flex-shrink-0" />
                        <span className="truncate">{getDisplayName(obj.key, prefix)}</span>
                      </button>
                    </TableCell>
                    <TableCell>{formatBytes(obj.size)}</TableCell>
                    <TableCell>{formatDate(obj.last_modified)}</TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          onClick={() => handleView(obj)}
                          title={t('storagePanel.actions.view')}
                        >
                          <EyeIcon className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          onClick={() => handleDownload(obj.key)}
                          title={t('storagePanel.actions.download')}
                        >
                          <DownloadIcon className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 text-destructive hover:text-destructive"
                          onClick={() => handleDelete(obj.key)}
                          title={t('storagePanel.actions.delete')}
                        >
                          <Trash2Icon className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>

        {/* Footer with object count */}
        <div className="border-t p-2 flex items-center justify-between bg-muted/20">
          <div className="text-sm text-muted-foreground">
            {t('storagePanel.count', {
              folders: folders.length,
              objects: objects.length,
            })}
          </div>
        </div>
      </Card>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* Delete confirmation dialog */}
      <AlertDialog open={!!deleteTarget} onOpenChange={(open) => !open && cancelDelete()}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('storagePanel.confirmDelete.title')}</AlertDialogTitle>
            <AlertDialogDescription>
              {t('storagePanel.confirmDelete.description', {
                name: deleteTarget ? getDisplayName(deleteTarget, prefix) : '',
              })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={cancelDelete}>
              {t('common.cancel')}
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {t('storagePanel.actions.delete')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* File viewer */}
      <FileViewer
        open={!!viewTarget}
        onOpenChange={(open) => !open && setViewTarget(null)}
        fileKey={viewTarget?.key ?? null}
        fileName={viewTarget ? getDisplayName(viewTarget.key, prefix) : ''}
        fileSize={viewTarget?.size ?? 0}
      />
    </div>
  )
}
