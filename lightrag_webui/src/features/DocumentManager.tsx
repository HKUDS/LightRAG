import { useState, useEffect, useCallback } from 'react'
import Button from '@/components/ui/Button'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/Table'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/Card'
import Progress from '@/components/ui/Progress'
import EmptyCard from '@/components/ui/EmptyCard'
import UploadDocumentsDialog from '@/components/documents/UploadDocumentsDialog'
import ClearDocumentsDialog from '@/components/documents/ClearDocumentsDialog'

import {
  getDocuments,
  // getDocumentsScanProgress,
  scanNewDocuments
  // LightragDocumentsScanProgress
} from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { toast } from 'sonner'
// import { useBackendState } from '@/stores/state'

import { RefreshCwIcon, TrashIcon } from 'lucide-react'

// type DocumentStatus = 'indexed' | 'pending' | 'indexing' | 'error'

export default function DocumentManager() {
  // const health = useBackendState.use.health()
  const [files, setFiles] = useState<string[]>([])
  const [indexedFiles, setIndexedFiles] = useState<string[]>([])
  // const [scanProgress, setScanProgress] = useState<LightragDocumentsScanProgress | null>(null)

  const fetchDocuments = useCallback(async () => {
    try {
      const docs = await getDocuments()
      setFiles(docs)
    } catch (err) {
      toast.error('Failed to load documents\n' + errorMessage(err))
    }
  }, [setFiles])

  useEffect(() => {
    fetchDocuments()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const scanDocuments = useCallback(async () => {
    try {
      const { status } = await scanNewDocuments()
      toast.message(status)
    } catch (err) {
      toast.error('Failed to load documents\n' + errorMessage(err))
    }
  }, [])

  // useEffect(() => {
  //   const interval = setInterval(async () => {
  //     try {
  //       if (!health) return
  //       const progress = await getDocumentsScanProgress()
  //       setScanProgress((pre) => {
  //         if (pre?.is_scanning === progress.is_scanning && progress.is_scanning === false) {
  //           return pre
  //         }
  //         return progress
  //       })
  //       console.log(progress)
  //     } catch (err) {
  //       toast.error('Failed to get scan progress\n' + errorMessage(err))
  //     }
  //   }, 2000)
  //   return () => clearInterval(interval)
  // }, [health])

  const handleDelete = async (fileName: string) => {
    console.log(`deleting ${fileName}`)
  }

  return (
    <Card className="!size-full !rounded-none !border-none">
      <CardHeader>
        <CardTitle className="text-lg">Document Management</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={scanDocuments}
            side="bottom"
            tooltip="Scan documents"
            size="sm"
          >
            <RefreshCwIcon /> Scan
          </Button>
          <div className="flex-1" />
          <ClearDocumentsDialog />
          <UploadDocumentsDialog />
        </div>

        {/* {scanProgress?.is_scanning && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Indexing {scanProgress.current_file}</span>
              <span>{scanProgress.progress}%</span>
            </div>
            <Progress value={scanProgress.progress} />
          </div>
        )} */}

        <Card>
          <CardHeader>
            <CardTitle>Uploaded documents</CardTitle>
            <CardDescription>view the uploaded documents here</CardDescription>
          </CardHeader>

          <CardContent>
            {files.length == 0 && (
              <EmptyCard
                title="No documents uploades"
                description="upload documents to see them here"
              />
            )}
            {files.length > 0 && (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Filename</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {files.map((file) => (
                    <TableRow key={file}>
                      <TableCell>{file}</TableCell>
                      <TableCell>
                        {indexedFiles.includes(file) ? (
                          <span className="text-green-600">Indexed</span>
                        ) : (
                          <span className="text-yellow-600">Pending</span>
                        )}
                      </TableCell>
                      <TableCell>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDelete(file)}
                          // disabled={isUploading}
                        >
                          <TrashIcon />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </CardContent>
    </Card>
  )
}
