import { useState, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'
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
import UploadDocumentsDialog from '@/components/documents/UploadDocumentsDialog'
import ClearDocumentsDialog from '@/components/documents/ClearDocumentsDialog'

import { getDocuments, scanNewDocuments, DocsStatusesResponse } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { toast } from 'sonner'
import { useBackendState } from '@/stores/state'

import { RefreshCwIcon } from 'lucide-react'
import { DocStatusResponse } from '@/api/lightrag'

const getDisplayFileName = (doc: DocStatusResponse, maxLength: number = 20): string => {
  // Check if file_path exists and is a non-empty string
  if (!doc.file_path || typeof doc.file_path !== 'string' || doc.file_path.trim() === '') {
    return doc.id;
  }

  // Try to extract filename from path
  const parts = doc.file_path.split('/');
  const fileName = parts[parts.length - 1];

  // Ensure extracted filename is valid
  if (!fileName || fileName.trim() === '') {
    return doc.id;
  }

  // If filename is longer than maxLength, truncate it and add ellipsis
  return fileName.length > maxLength
    ? fileName.slice(0, maxLength) + '...'
    : fileName;
};

export default function DocumentManager() {
  const { t } = useTranslation()
  const health = useBackendState.use.health()
  const [docs, setDocs] = useState<DocsStatusesResponse | null>(null)
  const currentTab = useSettingsStore.use.currentTab()
  const showFileName = useSettingsStore.use.showFileName()
  const setShowFileName = useSettingsStore.use.setShowFileName()

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

  // Fetch documents when the tab becomes visible
  useEffect(() => {
    if (currentTab === 'documents') {
      fetchDocuments()
    }
  }, [currentTab, fetchDocuments])

  const scanDocuments = useCallback(async () => {
    try {
      const { status } = await scanNewDocuments()
      toast.message(status)
    } catch (err) {
      toast.error(t('documentPanel.documentManager.errors.scanFailed', { error: errorMessage(err) }))
    }
  }, [t])

  // Set up polling when the documents tab is active and health is good
  useEffect(() => {
    if (currentTab !== 'documents' || !health) {
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
  }, [health, fetchDocuments, t, currentTab])

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
            <div className="flex justify-between items-center">
              <CardTitle>{t('documentPanel.documentManager.uploadedTitle')}</CardTitle>
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">{t('documentPanel.documentManager.fileNameLabel')}</span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowFileName(!showFileName)}
                  className="border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800"
                >
                  {showFileName
                    ? t('documentPanel.documentManager.hideButton')
                    : t('documentPanel.documentManager.showButton')
                  }
                </Button>
              </div>
            </div>
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
                  </TableRow>
                </TableHeader>
                <TableBody className="text-sm">
                  {Object.entries(docs.statuses).map(([status, documents]) =>
                    documents.map((doc) => (
                      <TableRow key={doc.id}>
                        <TableCell className="truncate font-mono overflow-visible">
                          {showFileName ? (
                            <>
                              <div className="group relative overflow-visible">
                                <div className="truncate">
                                  {getDisplayFileName(doc, 35)}
                                </div>
                                <div className="invisible group-hover:visible fixed z-[9999] mt-1 max-w-[800px] whitespace-normal break-all rounded-md bg-black/95 px-3 py-2 text-sm text-white shadow-lg dark:bg-white/95 dark:text-black">
                                  {doc.file_path}
                                </div>
                              </div>
                              <div className="text-xs text-gray-500">{doc.id}</div>
                            </>
                          ) : (
                            <div className="group relative overflow-visible">
                              <div className="truncate">
                                {doc.id}
                              </div>
                              <div className="invisible group-hover:visible fixed z-[9999] mt-1 max-w-[800px] whitespace-normal break-all rounded-md bg-black/95 px-3 py-2 text-sm text-white shadow-lg dark:bg-white/95 dark:text-black">
                                {doc.file_path}
                              </div>
                            </div>
                          )}
                        </TableCell>
                        <TableCell className="max-w-xs min-w-24 truncate overflow-visible">
                          <div className="group relative overflow-visible">
                            <div className="truncate">
                              {doc.content_summary}
                            </div>
                            <div className="invisible group-hover:visible fixed z-[9999] mt-1 max-w-[800px] whitespace-normal break-all rounded-md bg-black/95 px-3 py-2 text-sm text-white shadow-lg dark:bg-white/95 dark:text-black">
                              {doc.content_summary}
                            </div>
                          </div>
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
