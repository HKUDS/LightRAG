import { useState, useEffect, useCallback, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { useTabVisibility } from '@/contexts/useTabVisibility'
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
  const { t } = useTranslation()
  const health = useBackendState.use.health()
  const [docs, setDocs] = useState<DocsStatusesResponse | null>(null)
  const { isTabVisible } = useTabVisibility()
  const isDocumentsTabVisible = isTabVisible('documents')
  const initialLoadRef = useRef(false)

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
      } else {
        setDocs(null)
      }
    } catch (err) {
      toast.error(t('documentPanel.documentManager.errors.loadFailed', { error: errorMessage(err) }))
    }
  }, [setDocs, t])

  // Only fetch documents when the tab becomes visible for the first time
  useEffect(() => {
    if (isDocumentsTabVisible && !initialLoadRef.current) {
      fetchDocuments()
      initialLoadRef.current = true
    }
  }, [isDocumentsTabVisible, fetchDocuments])

  const scanDocuments = useCallback(async () => {
    try {
      const { status } = await scanNewDocuments()
      toast.message(status)
    } catch (err) {
      toast.error(t('documentPanel.documentManager.errors.scanFailed', { error: errorMessage(err) }))
    }
  }, [t])

  // Only set up polling when the tab is visible and health is good
  useEffect(() => {
    if (!isDocumentsTabVisible || !health) {
      return
    }

    const interval = setInterval(async () => {
      try {
        await fetchDocuments()
      } catch (err) {
        toast.error(t('documentPanel.documentManager.errors.scanProgressFailed', { error: errorMessage(err) }))
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [health, fetchDocuments, t, isDocumentsTabVisible])

  return (
    <Card className="!size-full !rounded-none !border-none">
      <CardHeader>
        <CardTitle className="text-lg">{t('documentPanel.documentManager.title')}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={scanDocuments}
            side="bottom"
            tooltip={t('documentPanel.documentManager.scanTooltip')}
            size="sm"
          >
            <RefreshCwIcon /> {t('documentPanel.documentManager.scanButton')}
          </Button>
          <div className="flex-1" />
          <ClearDocumentsDialog />
          <UploadDocumentsDialog />
        </div>

        <Card>
          <CardHeader>
            <CardTitle>{t('documentPanel.documentManager.uploadedTitle')}</CardTitle>
            <CardDescription>{t('documentPanel.documentManager.uploadedDescription')}</CardDescription>
          </CardHeader>

          <CardContent>
            {!docs && (
              <EmptyCard
                title={t('documentPanel.documentManager.emptyTitle')}
                description={t('documentPanel.documentManager.emptyDescription')}
              />
            )}
            {docs && (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t('documentPanel.documentManager.columns.id')}</TableHead>
                    <TableHead>{t('documentPanel.documentManager.columns.summary')}</TableHead>
                    <TableHead>{t('documentPanel.documentManager.columns.status')}</TableHead>
                    <TableHead>{t('documentPanel.documentManager.columns.length')}</TableHead>
                    <TableHead>{t('documentPanel.documentManager.columns.chunks')}</TableHead>
                    <TableHead>{t('documentPanel.documentManager.columns.created')}</TableHead>
                    <TableHead>{t('documentPanel.documentManager.columns.updated')}</TableHead>
                    <TableHead>{t('documentPanel.documentManager.columns.metadata')}</TableHead>
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
                            <span className="text-green-600">{t('documentPanel.documentManager.status.completed')}</span>
                          )}
                          {status === 'processing' && (
                            <span className="text-blue-600">{t('documentPanel.documentManager.status.processing')}</span>
                          )}
                          {status === 'pending' && <span className="text-yellow-600">{t('documentPanel.documentManager.status.pending')}</span>}
                          {status === 'failed' && <span className="text-red-600">{t('documentPanel.documentManager.status.failed')}</span>}
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
