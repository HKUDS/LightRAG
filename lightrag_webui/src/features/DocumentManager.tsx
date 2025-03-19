import { useState, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs'

import {
  getDocuments,
  scanNewDocuments,
  DocsStatusesResponse,
  pipelineStatus,
  PipelineStatusResponse
} from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { toast } from 'sonner'
import { useBackendState } from '@/stores/state'

import { RefreshCwIcon, FlameIcon } from 'lucide-react'

export default function DocumentManager() {
  const { t } = useTranslation()
  const health = useBackendState.use.health()
  const [docs, setDocs] = useState<DocsStatusesResponse | null>(null)
  const [pipeline, setPipeline] = useState<PipelineStatusResponse | null>(null)
  const [activeTab, setActiveTab] = useState<string>('documents')

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
      toast.error(
        t('documentPanel.documentManager.errors.loadFailed', { error: errorMessage(err) })
      )
    }
  }, [setDocs, t])

  const fetchPipelineStatus = useCallback(async () => {
    try {
      const status = await pipelineStatus()
      setPipeline(status)
    } catch (err) {
      toast.error(
        t('documentPanel.documentManager.errors.pipelineStatusFailed', { error: errorMessage(err) })
      )
    }
  }, [t, setPipeline])

  const scanDocuments = useCallback(async () => {
    try {
      const { status } = await scanNewDocuments()
      toast.message(status)
    } catch (err) {
      toast.error(
        t('documentPanel.documentManager.errors.scanFailed', { error: errorMessage(err) })
      )
    }
  }, [t])

  const refreshData = useCallback(async () => {
    if (activeTab === 'documents') {
      await fetchDocuments()
    } else {
      await fetchPipelineStatus()
    }
  }, [fetchDocuments, fetchPipelineStatus, activeTab])

  // Set up polling when the documents tab is active and health is good
  useEffect(() => {
    // Fetch documents when the tab becomes visible
    refreshData()

    const interval = setInterval(async () => {
      try {
        await refreshData()
      } catch (err) {
        toast.error(
          t('documentPanel.documentManager.errors.refreshFailed', { error: errorMessage(err) })
        )
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [t, health, refreshData])

  return (
    <Card className="!size-full !rounded-none !border-none">
      <CardHeader>
        <CardTitle className="text-lg">{t('documentPanel.documentManager.title')}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Tabs defaultValue="documents" value={activeTab} onValueChange={setActiveTab}>
          <div className="relative flex gap-2">
            <Button
              variant="outline"
              onClick={scanDocuments}
              side="bottom"
              tooltip={t('documentPanel.documentManager.scanTooltip')}
              size="sm"
            >
              <RefreshCwIcon /> {t('documentPanel.documentManager.scanButton')}
            </Button>
            <div className="pointer-events-none absolute right-0 bottom-0 left-0 flex justify-center">
              <TabsList className="pointer-events-auto flex-shrink gap-2">
                <TabsTrigger
                  value="documents"
                  className="hover:bg-background/60 cursor-pointer px-2 py-1 transition-all"
                >
                  {t('documentPanel.documentManager.tabs.documents')}
                </TabsTrigger>
                <TabsTrigger
                  value="pipeline"
                  className="hover:bg-background/60 cursor-pointer px-2 py-1 transition-all"
                >
                  {t('documentPanel.documentManager.tabs.pipeline')}
                </TabsTrigger>
              </TabsList>
            </div>
            <div className="flex-1" />
            <ClearDocumentsDialog />
            <UploadDocumentsDialog />
          </div>

          <TabsContent value="documents">
            <Card>
              <CardHeader>
                <CardTitle>{t('documentPanel.documentManager.uploadedTitle')}</CardTitle>
                <CardDescription>
                  {t('documentPanel.documentManager.uploadedDescription')}
                </CardDescription>
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
                        <TableHead>{t('documentPanel.documentManager.columns.filepath')}</TableHead>
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
                                text={doc.file_path}
                                tooltip={doc.file_path}
                                tooltipClassName="max-w-none overflow-visible block"
                              />
                            </TableCell>
                            <TableCell className="max-w-xs min-w-24 truncate">
                              <Text
                                text={doc.content_summary}
                                tooltip={doc.content_summary}
                                tooltipClassName="max-w-none overflow-visible block"
                              />
                            </TableCell>
                            <TableCell className='truncate'>
                              {status === 'processed' && (
                                <span className="text-green-600">
                                  {t('documentPanel.documentManager.status.completed')}
                                </span>
                              )}
                              {status === 'processing' && (
                                <span className="text-blue-600">
                                  {t('documentPanel.documentManager.status.processing')}
                                </span>
                              )}
                              {status === 'pending' && (
                                <span className="text-yellow-600">
                                  {t('documentPanel.documentManager.status.pending')}
                                </span>
                              )}
                              {status === 'failed' && (
                                <span className="text-red-600">
                                  {t('documentPanel.documentManager.status.failed')}
                                </span>
                              )}
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
          </TabsContent>

          <TabsContent value="pipeline">
            <Card>
              <CardHeader>
                <CardTitle>{t('documentPanel.documentManager.pipeline.pipelineTitle')}</CardTitle>
                <CardDescription>
                  {t('documentPanel.documentManager.pipeline.pipelineDescription')}
                </CardDescription>
              </CardHeader>

              <CardContent>
                {!pipeline ? (
                  <EmptyCard
                    title={t('documentPanel.documentManager.pipeline.pipelineEmptyTitle')}
                    description={t(
                      'documentPanel.documentManager.pipeline.pipelineEmptyDescription'
                    )}
                    icon={FlameIcon}
                  />
                ) : (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <Card>
                        <CardHeader className="p-4">
                          <CardTitle className="text-sm">
                            {t('documentPanel.documentManager.pipeline.status')}
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="p-4 pt-0">
                          <div className="flex items-center">
                            <div
                              className={`mr-2 h-3 w-3 rounded-full ${pipeline.busy ? 'animate-pulse bg-blue-500' : 'bg-green-500'}`}
                            ></div>
                            <span>
                              {pipeline.busy
                                ? t('documentPanel.documentManager.pipeline.busy')
                                : t('documentPanel.documentManager.pipeline.idle')}
                            </span>
                          </div>
                        </CardContent>
                      </Card>

                      <Card>
                        <CardHeader className="p-4">
                          <CardTitle className="text-sm">
                            {t('documentPanel.documentManager.pipeline.currentJob')}
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="p-4 pt-0">
                          {pipeline.job_name ? (
                            <div>
                              <div className="font-medium">{pipeline.job_name}</div>
                              {pipeline.job_start && (
                                <div className="text-muted-foreground text-sm">
                                  {t('documentPanel.documentManager.pipeline.started')}{' '}
                                  {new Date(pipeline.job_start).toLocaleString()}
                                </div>
                              )}
                            </div>
                          ) : (
                            <span className="text-muted-foreground">
                              {t('documentPanel.documentManager.pipeline.noCurrentJob')}
                            </span>
                          )}
                        </CardContent>
                      </Card>
                    </div>

                    <Card>
                      <CardHeader className="p-4">
                        <CardTitle className="text-sm">
                          {t('documentPanel.documentManager.pipeline.progress')}
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="p-4 pt-0">
                        {pipeline.docs > 0 ? (
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span>
                                {t('documentPanel.documentManager.pipeline.batch')}:{' '}
                                {pipeline.cur_batch} / {pipeline.batchs}
                              </span>
                              <span>
                                {t('documentPanel.documentManager.pipeline.documents')}:{' '}
                                {pipeline.docs}
                              </span>
                            </div>
                            <div className="h-2.5 w-full rounded-full bg-gray-200 dark:bg-gray-700">
                              <div
                                className="h-2.5 rounded-full bg-blue-400"
                                style={{
                                  width: `${Math.round((pipeline.cur_batch / pipeline.batchs) * 100)}%`
                                }}
                              />
                            </div>
                          </div>
                        ) : (
                          <span className="text-muted-foreground">
                            {t('documentPanel.documentManager.pipeline.noProgress')}
                          </span>
                        )}
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="p-4">
                        <CardTitle className="text-sm">
                          {t('documentPanel.documentManager.pipeline.latestMessage')}
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="p-4 pt-0">
                        {pipeline.latest_message ? (
                          <div className="bg-muted rounded p-2 text-sm">
                            {pipeline.latest_message}
                          </div>
                        ) : (
                          <span className="text-muted-foreground">
                            {t('documentPanel.documentManager.pipeline.noMessages')}
                          </span>
                        )}
                      </CardContent>
                    </Card>

                    {pipeline.history_messages && pipeline.history_messages.length > 0 && (
                      <Card>
                        <CardHeader className="p-4">
                          <CardTitle className="text-sm">
                            {t('documentPanel.documentManager.pipeline.historyMessages')}
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="p-4 pt-0">
                          <div className="max-h-80 overflow-y-auto">
                            <ul className="space-y-2">
                              {pipeline.history_messages.map((message, idx) => (
                                <li key={idx} className="bg-muted rounded p-2 text-sm">
                                  {message}
                                </li>
                              ))}
                            </ul>
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    <div className="grid grid-cols-2 gap-4">
                      <Card>
                        <CardHeader className="p-4">
                          <CardTitle className="text-sm">
                            {t('documentPanel.documentManager.pipeline.autoScan')}
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="p-4 pt-0">
                          <div className="flex items-center">
                            <div
                              className={`mr-2 h-3 w-3 rounded-full ${pipeline.autoscanned ? 'bg-green-500' : 'bg-yellow-500'}`}
                            ></div>
                            <span>
                              {pipeline.autoscanned
                                ? t('documentPanel.documentManager.pipeline.autoScanExecuted')
                                : t('documentPanel.documentManager.pipeline.autoScanNotExecuted')}
                            </span>
                          </div>
                        </CardContent>
                      </Card>

                      <Card>
                        <CardHeader className="p-4">
                          <CardTitle className="text-sm">
                            {t('documentPanel.documentManager.pipeline.requestStatus')}
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="p-4 pt-0">
                          <div className="flex items-center">
                            <div
                              className={`mr-2 h-3 w-3 rounded-full ${pipeline.request_pending ? 'bg-yellow-500' : 'bg-green-500'}`}
                            ></div>
                            <span>
                              {pipeline.request_pending
                                ? t('documentPanel.documentManager.pipeline.pendingRequests')
                                : t('documentPanel.documentManager.pipeline.noPendingRequests')}
                            </span>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
