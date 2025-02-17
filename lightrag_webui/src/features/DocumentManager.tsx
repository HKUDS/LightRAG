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
import EmptyCard from '@/components/ui/EmptyCard'
import Text from '@/components/ui/Text'
import UploadDocumentsDialog from '@/components/documents/UploadDocumentsDialog'
import ClearDocumentsDialog from '@/components/documents/ClearDocumentsDialog'

import { getDocuments, scanNewDocuments, DocsStatusesResponse } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { toast } from 'sonner'
import { useBackendState } from '@/stores/state'

import { RefreshCwIcon } from 'lucide-react'

export default function DocumentManager() {
  const health = useBackendState.use.health()
  const [docs, setDocs] = useState<DocsStatusesResponse | null>(null)

  const fetchDocuments = useCallback(async () => {
    try {
      const docs = await getDocuments()
      if (docs && docs.statuses) {
        // compose all documents count
        const numDocuments = Object.values(docs.statuses).reduce(
          (acc, status) => acc + status.length,
          0
        )
        if (numDocuments > 0) {
          setDocs(docs)
        } else {
          setDocs(null)
        }
        // console.log(docs)
      } else {
        setDocs(null)
      }
    } catch (err) {
      toast.error('Failed to load documents\n' + errorMessage(err))
    }
  }, [setDocs])

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

  useEffect(() => {
    const interval = setInterval(async () => {
      if (!health) {
        return
      }
      try {
        await fetchDocuments()
      } catch (err) {
        toast.error('Failed to get scan progress\n' + errorMessage(err))
      }
    }, 5000)
    return () => clearInterval(interval)
  }, [health, fetchDocuments])

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

        <Card>
          <CardHeader>
            <CardTitle>Uploaded documents</CardTitle>
            <CardDescription>view the uploaded documents here</CardDescription>
          </CardHeader>

          <CardContent>
            {!docs && (
              <EmptyCard
                title="No documents uploaded"
                description="upload documents to see them here"
              />
            )}
            {docs && (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>ID</TableHead>
                    <TableHead>Summary</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Length</TableHead>
                    <TableHead>Chunks</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Updated</TableHead>
                    <TableHead>Metadata</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody className="text-sm">
                  {Object.entries(docs.statuses).map(([status, documents]) =>
                    documents.map((doc) => (
                      <TableRow key={doc.id}>
                        <TableCell className="truncate font-mono">{doc.id}</TableCell>
                        <TableCell className="max-w-xs min-w-24 truncate">
                          <Text
                            text={doc.content_summary}
                            tooltip={doc.content_summary}
                            tooltipClassName="max-w-none overflow-visible block"
                          />
                        </TableCell>
                        <TableCell>
                          {status === 'processed' && (
                            <span className="text-green-600">Completed</span>
                          )}
                          {status === 'processing' && (
                            <span className="text-blue-600">Processing</span>
                          )}
                          {status === 'pending' && <span className="text-yellow-600">Pending</span>}
                          {status === 'failed' && <span className="text-red-600">Failed</span>}
                          {doc.error && (
                            <span className="ml-2 text-red-500" title={doc.error}>
                              ⚠️
                            </span>
                          )}
                        </TableCell>
                        <TableCell>{doc.content_length ?? '-'}</TableCell>
                        <TableCell>{doc.chunks_count ?? '-'}</TableCell>
                        <TableCell className="truncate">
                          {new Date(doc.created_at).toLocaleString()}
                        </TableCell>
                        <TableCell className="truncate">
                          {new Date(doc.updated_at).toLocaleString()}
                        </TableCell>
                        <TableCell className="max-w-xs truncate">
                          {doc.metadata ? JSON.stringify(doc.metadata) : '-'}
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </CardContent>
    </Card>
  )
}
