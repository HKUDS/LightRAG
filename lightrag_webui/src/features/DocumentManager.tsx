import { useState, useEffect, useCallback, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'
import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'
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

import { RefreshCwIcon, ActivityIcon } from 'lucide-react'
import { DocStatusResponse } from '@/api/lightrag'
import PipelineStatusDialog from '@/components/documents/PipelineStatusDialog'

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

const pulseStyle = `
/* Custom tooltip styles */
.tooltip-top {
  bottom: 100% !important;
  top: auto !important;
  margin-bottom: 0.25rem !important;
  margin-top: 0 !important;
}

/* Fixed tooltip styles for small tables */
.tooltip-fixed {
  position: fixed !important;
  z-index: 9999 !important;
}

/* Parent container for tooltips */
.tooltip-container {
  position: relative;
  overflow: visible !important;
}

@keyframes pulse {
  0% {
    background-color: rgb(255 0 0 / 0.1);
    border-color: rgb(255 0 0 / 0.2);
  }
  50% {
    background-color: rgb(255 0 0 / 0.2);
    border-color: rgb(255 0 0 / 0.4);
  }
  100% {
    background-color: rgb(255 0 0 / 0.1);
    border-color: rgb(255 0 0 / 0.2);
  }
}

.dark .pipeline-busy {
  animation: dark-pulse 2s infinite;
}

@keyframes dark-pulse {
  0% {
    background-color: rgb(255 0 0 / 0.2);
    border-color: rgb(255 0 0 / 0.4);
  }
  50% {
    background-color: rgb(255 0 0 / 0.3);
    border-color: rgb(255 0 0 / 0.6);
  }
  100% {
    background-color: rgb(255 0 0 / 0.2);
    border-color: rgb(255 0 0 / 0.4);
  }
}

.pipeline-busy {
  animation: pulse 2s infinite;
  border: 1px solid;
}
`;

export default function DocumentManager() {
  const [showPipelineStatus, setShowPipelineStatus] = useState(false)
  const { t } = useTranslation()
  const health = useBackendState.use.health()
  const pipelineBusy = useBackendState.use.pipelineBusy()
  const [docs, setDocs] = useState<DocsStatusesResponse | null>(null)
  const currentTab = useSettingsStore.use.currentTab()
  const showFileName = useSettingsStore.use.showFileName()
  const setShowFileName = useSettingsStore.use.setShowFileName()

  // Store previous status counts
  const prevStatusCounts = useRef({
    processed: 0,
    processing: 0,
    pending: 0,
    failed: 0
  })

  // Add pulse style to document
  useEffect(() => {
    const style = document.createElement('style')
    style.textContent = pulseStyle
    document.head.appendChild(style)
    return () => {
      document.head.removeChild(style)
    }
  }, [])

  // Reference to the card content element
  const cardContentRef = useRef<HTMLDivElement>(null);

  // Add tooltip position adjustment based on mouse position
  useEffect(() => {
    if (!docs) return;

    // Function to handle mouse movement
    const handleMouseMove = (event: MouseEvent) => {
      const cardContent = cardContentRef.current;
      if (!cardContent) return;

      // Get all visible tooltips
      const visibleTooltips = document.querySelectorAll<HTMLElement>('.group:hover > div[class*="invisible group-hover:visible absolute"]');

      visibleTooltips.forEach(tooltip => {
        // Get the parent element that triggered the tooltip
        const triggerElement = tooltip.parentElement;
        if (!triggerElement) return;

        const triggerRect = triggerElement.getBoundingClientRect();

        // Use fixed positioning for all tooltips
        tooltip.classList.add('tooltip-fixed');

        // Calculate position based on trigger element and mouse
        const tooltipHeight = tooltip.offsetHeight;
        const viewportHeight = window.innerHeight;

        // Check if tooltip would go off the bottom of the viewport
        const wouldOverflowBottom = event.clientY + tooltipHeight > viewportHeight;

        if (wouldOverflowBottom) {
          // Position above the trigger
          tooltip.style.top = `${triggerRect.top - tooltipHeight - 5}px`;
          tooltip.style.bottom = 'auto';
        } else {
          // Position below the trigger
          tooltip.style.top = `${triggerRect.bottom + 5}px`;
          tooltip.style.bottom = 'auto';
        }

        // Horizontal positioning
        tooltip.style.left = `${triggerRect.left}px`;
        tooltip.style.maxWidth = '600px';
      });
    };

    // Add mouse move listener to the document
    document.addEventListener('mousemove', handleMouseMove);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, [docs]);

  const fetchDocuments = useCallback(async () => {
    try {
      const docs = await getDocuments()

      // Get new status counts (treat null as all zeros)
      const newStatusCounts = {
        processed: docs?.statuses?.processed?.length || 0,
        processing: docs?.statuses?.processing?.length || 0,
        pending: docs?.statuses?.pending?.length || 0,
        failed: docs?.statuses?.failed?.length || 0
      }

      // Check if any status count has changed
      const hasStatusCountChange = (Object.keys(newStatusCounts) as Array<keyof typeof newStatusCounts>).some(
        status => newStatusCounts[status] !== prevStatusCounts.current[status]
      )

      // Trigger health check if changes detected
      if (hasStatusCountChange) {
        useBackendState.getState().check()
      }

      // Update previous status counts
      prevStatusCounts.current = newStatusCounts

      // Update docs state
      if (docs && docs.statuses) {
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
    <Card className="!rounded-none !overflow-hidden flex flex-col h-full min-h-0">
      <CardHeader className="py-2 px-6">
        <CardTitle className="text-lg">{t('documentPanel.documentManager.title')}</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <div className="flex gap-2 mb-2">
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
            <Button
              variant="outline"
              onClick={() => setShowPipelineStatus(true)}
              side="bottom"
              tooltip={t('documentPanel.documentManager.pipelineStatusTooltip')}
              size="sm"
              className={cn(
                pipelineBusy && 'pipeline-busy'
              )}
            >
              <ActivityIcon /> {t('documentPanel.documentManager.pipelineStatusButton')}
            </Button>
          </div>
          <div className="flex-1" />
          <ClearDocumentsDialog />
          <UploadDocumentsDialog />
          <PipelineStatusDialog
            open={showPipelineStatus}
            onOpenChange={setShowPipelineStatus}
          />
        </div>

        <Card className="flex-1 flex flex-col border rounded-md min-h-0 mb-0">
          <CardHeader className="flex-none py-2 px-4">
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
            <CardDescription aria-hidden="true" className="hidden">{t('documentPanel.documentManager.uploadedDescription')}</CardDescription>
          </CardHeader>

          <CardContent className="flex-1 relative p-0" ref={cardContentRef}>
            {!docs && (
              <div className="absolute inset-0 p-0">
                <EmptyCard
                  title={t('documentPanel.documentManager.emptyTitle')}
                  description={t('documentPanel.documentManager.emptyDescription')}
                />
              </div>
            )}
            {docs && (
              <div className="absolute inset-0 flex flex-col p-0">
                <div className="w-full h-full flex flex-col rounded-lg border border-gray-200 dark:border-gray-700">
                  <div className="flex-1 overflow-hidden flex flex-col">
                    <Table className="w-full" style={{ minHeight: '100%', height: '100%'}}>
                      <TableHeader className="sticky top-0 bg-background z-10 shadow-sm">
                        <TableRow className="border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/75 shadow-[inset_0_-1px_0_rgba(0,0,0,0.1)]">
                          <TableHead>{t('documentPanel.documentManager.columns.id')}</TableHead>
                          <TableHead>{t('documentPanel.documentManager.columns.summary')}</TableHead>
                          <TableHead>{t('documentPanel.documentManager.columns.status')}</TableHead>
                          <TableHead>{t('documentPanel.documentManager.columns.length')}</TableHead>
                          <TableHead>{t('documentPanel.documentManager.columns.chunks')}</TableHead>
                          <TableHead>{t('documentPanel.documentManager.columns.created')}</TableHead>
                          <TableHead>{t('documentPanel.documentManager.columns.updated')}</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody className="text-sm overflow-auto">
                        {Object.entries(docs.statuses).map(([status, documents]) =>
                          documents.map((doc) => (
                            <TableRow key={doc.id}>
                              <TableCell className="truncate font-mono overflow-visible max-w-[250px]">
                                {showFileName ? (
                                  <>
                                    <div className="group relative overflow-visible tooltip-container">
                                      <div className="truncate">
                                        {getDisplayFileName(doc, 30)}
                                      </div>
                                      <div className="invisible group-hover:visible absolute z-[9999] mt-1 max-w-[600px] whitespace-normal break-all rounded-md bg-black/95 px-3 py-2 text-sm text-white shadow-lg dark:bg-white/95 dark:text-black">
                                        {doc.file_path}
                                      </div>
                                    </div>
                                    <div className="text-xs text-gray-500">{doc.id}</div>
                                  </>
                                ) : (
                                  <div className="group relative overflow-visible tooltip-container">
                                    <div className="truncate">
                                      {doc.id}
                                    </div>
                                    <div className="invisible group-hover:visible absolute z-[9999] mt-1 max-w-[600px] whitespace-normal break-all rounded-md bg-black/95 px-3 py-2 text-sm text-white shadow-lg dark:bg-white/95 dark:text-black">
                                      {doc.file_path}
                                    </div>
                                  </div>
                                )}
                              </TableCell>
                              <TableCell className="max-w-xs min-w-45 truncate overflow-visible">
                                <div className="group relative overflow-visible tooltip-container">
                                  <div className="truncate">
                                    {doc.content_summary}
                                  </div>
                                  <div className="invisible group-hover:visible absolute z-[9999] mt-1 max-w-[600px] whitespace-normal break-all rounded-md bg-black/95 px-3 py-2 text-sm text-white shadow-lg dark:bg-white/95 dark:text-black">
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
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </CardContent>
    </Card>
  )
}
