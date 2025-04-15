import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
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

import { getDocuments, scanNewDocuments, DocsStatusesResponse, DocStatus, DocStatusResponse } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { toast } from 'sonner'
import { useBackendState } from '@/stores/state'

import { RefreshCwIcon, ActivityIcon, ArrowUpIcon, ArrowDownIcon, FilterIcon } from 'lucide-react'
import PipelineStatusDialog from '@/components/documents/PipelineStatusDialog'

type StatusFilter = DocStatus | 'all';


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
/* Tooltip styles */
.tooltip-container {
  position: relative;
  overflow: visible !important;
}

.tooltip {
  position: fixed; /* Use fixed positioning to escape overflow constraints */
  z-index: 9999; /* Ensure tooltip appears above all other elements */
  max-width: 600px;
  white-space: normal;
  border-radius: 0.375rem;
  padding: 0.5rem 0.75rem;
  background-color: rgba(0, 0, 0, 0.95);
  color: white;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  pointer-events: none; /* Prevent tooltip from interfering with mouse events */
  opacity: 0;
  visibility: hidden;
  transition: opacity 0.15s, visibility 0.15s;
}

.tooltip.visible {
  opacity: 1;
  visibility: visible;
}

.dark .tooltip {
  background-color: rgba(255, 255, 255, 0.95);
  color: black;
}

/* Position tooltip helper class */
.tooltip-helper {
  position: absolute;
  visibility: hidden;
  pointer-events: none;
  top: 0;
  left: 0;
  width: 100%;
  height: 0;
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

// Type definitions for sort field and direction
type SortField = 'created_at' | 'updated_at' | 'id';
type SortDirection = 'asc' | 'desc';

export default function DocumentManager() {
  // Track component mount status
  const isMountedRef = useRef(true);

  // Set up mount/unmount status tracking
  useEffect(() => {
    isMountedRef.current = true;

    // Handle page reload/unload
    const handleBeforeUnload = () => {
      isMountedRef.current = false;
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      isMountedRef.current = false;
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  const [showPipelineStatus, setShowPipelineStatus] = useState(false)
  const { t, i18n } = useTranslation()
  const health = useBackendState.use.health()
  const pipelineBusy = useBackendState.use.pipelineBusy()
  const [docs, setDocs] = useState<DocsStatusesResponse | null>(null)
  const currentTab = useSettingsStore.use.currentTab()
  const showFileName = useSettingsStore.use.showFileName()
  const setShowFileName = useSettingsStore.use.setShowFileName()

  // Sort state
  const [sortField, setSortField] = useState<SortField>('updated_at')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')

  // State for document status filter
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');


  // Handle sort column click
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      // Toggle sort direction if clicking the same field
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc')
    } else {
      // Set new sort field with default desc direction
      setSortField(field)
      setSortDirection('desc')
    }
  }

  // Sort documents based on current sort field and direction
  const sortDocuments = useCallback((documents: DocStatusResponse[]) => {
    return [...documents].sort((a, b) => {
      let valueA, valueB;

      // Special handling for ID field based on showFileName setting
      if (sortField === 'id' && showFileName) {
        valueA = getDisplayFileName(a);
        valueB = getDisplayFileName(b);
      } else if (sortField === 'id') {
        valueA = a.id;
        valueB = b.id;
      } else {
        // Date fields
        valueA = new Date(a[sortField]).getTime();
        valueB = new Date(b[sortField]).getTime();
      }

      // Apply sort direction
      const sortMultiplier = sortDirection === 'asc' ? 1 : -1;

      // Compare values
      if (typeof valueA === 'string' && typeof valueB === 'string') {
        return sortMultiplier * valueA.localeCompare(valueB);
      } else {
        return sortMultiplier * (valueA > valueB ? 1 : valueA < valueB ? -1 : 0);
      }
    });
  }, [sortField, sortDirection, showFileName]);

  // Define a new type that includes status information
  type DocStatusWithStatus = DocStatusResponse & { status: DocStatus };

  const filteredAndSortedDocs = useMemo(() => {
    if (!docs) return null;

    // Create a flat array of documents with status information
    const allDocuments: DocStatusWithStatus[] = [];

    if (statusFilter === 'all') {
      // When filter is 'all', include documents from all statuses
      Object.entries(docs.statuses).forEach(([status, documents]) => {
        documents.forEach(doc => {
          allDocuments.push({
            ...doc,
            status: status as DocStatus
          });
        });
      });
    } else {
      // When filter is specific status, only include documents from that status
      const documents = docs.statuses[statusFilter] || [];
      documents.forEach(doc => {
        allDocuments.push({
          ...doc,
          status: statusFilter
        });
      });
    }

    // Sort all documents together if sort field and direction are specified
    if (sortField && sortDirection) {
      return sortDocuments(allDocuments);
    }

    return allDocuments;
  }, [docs, sortField, sortDirection, statusFilter, sortDocuments]);

  // Calculate document counts for each status
  const documentCounts = useMemo(() => {
    if (!docs) return { all: 0 } as Record<string, number>;

    const counts: Record<string, number> = { all: 0 };

    Object.entries(docs.statuses).forEach(([status, documents]) => {
      counts[status as DocStatus] = documents.length;
      counts.all += documents.length;
    });

    return counts;
  }, [docs]);

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

  // Add tooltip position adjustment for fixed positioning
  useEffect(() => {
    if (!docs) return;

    // Function to position tooltips
    const positionTooltips = () => {
      // Get all tooltip containers
      const containers = document.querySelectorAll<HTMLElement>('.tooltip-container');

      containers.forEach(container => {
        const tooltip = container.querySelector<HTMLElement>('.tooltip');
        if (!tooltip) return;

        // Skip tooltips that aren't visible
        if (!tooltip.classList.contains('visible')) return;

        // Get container position
        const rect = container.getBoundingClientRect();

        // Position tooltip above the container
        tooltip.style.left = `${rect.left}px`;
        tooltip.style.top = `${rect.top - 5}px`;
        tooltip.style.transform = 'translateY(-100%)';
      });
    };

    // Set up event listeners
    const handleMouseOver = (e: MouseEvent) => {
      // Check if target or its parent is a tooltip container
      const target = e.target as HTMLElement;
      const container = target.closest('.tooltip-container');
      if (!container) return;

      // Find tooltip and make it visible
      const tooltip = container.querySelector<HTMLElement>('.tooltip');
      if (tooltip) {
        tooltip.classList.add('visible');
        // Position immediately without delay
        positionTooltips();
      }
    };

    const handleMouseOut = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const container = target.closest('.tooltip-container');
      if (!container) return;

      const tooltip = container.querySelector<HTMLElement>('.tooltip');
      if (tooltip) {
        tooltip.classList.remove('visible');
      }
    };

    document.addEventListener('mouseover', handleMouseOver);
    document.addEventListener('mouseout', handleMouseOut);

    return () => {
      document.removeEventListener('mouseover', handleMouseOver);
      document.removeEventListener('mouseout', handleMouseOut);
    };
  }, [docs]);

  const fetchDocuments = useCallback(async () => {
    try {
      // Check if component is still mounted before starting the request
      if (!isMountedRef.current) return;

      const docs = await getDocuments();

      // Check again if component is still mounted after the request completes
      if (!isMountedRef.current) return;

      // Only update state if component is still mounted
      if (isMountedRef.current) {
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
      }
    } catch (err) {
      // Only show error if component is still mounted
      if (isMountedRef.current) {
        toast.error(t('documentPanel.documentManager.errors.loadFailed', { error: errorMessage(err) }))
      }
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
      // Check if component is still mounted before starting the request
      if (!isMountedRef.current) return;

      const { status } = await scanNewDocuments();

      // Check again if component is still mounted after the request completes
      if (!isMountedRef.current) return;

      toast.message(status);
    } catch (err) {
      // Only show error if component is still mounted
      if (isMountedRef.current) {
        toast.error(t('documentPanel.documentManager.errors.scanFailed', { error: errorMessage(err) }));
      }
    }
  }, [t])

  // Set up polling when the documents tab is active and health is good
  useEffect(() => {
    if (currentTab !== 'documents' || !health) {
      return
    }

    const interval = setInterval(async () => {
      try {
        // Only perform fetch if component is still mounted
        if (isMountedRef.current) {
          await fetchDocuments()
        }
      } catch (err) {
        // Only show error if component is still mounted
        if (isMountedRef.current) {
          toast.error(t('documentPanel.documentManager.errors.scanProgressFailed', { error: errorMessage(err) }))
        }
      }
    }, 5000)

    return () => {
      clearInterval(interval)
    }
  }, [health, fetchDocuments, t, currentTab])

  // Monitor docs changes to check status counts and trigger health check if needed
  useEffect(() => {
    if (!docs) return;

    // Get new status counts
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

    // Trigger health check if changes detected and component is still mounted
    if (hasStatusCountChange && isMountedRef.current) {
      useBackendState.getState().check()
    }

    // Update previous status counts
    prevStatusCounts.current = newStatusCounts
  }, [docs]);

  // Add dependency on sort state to re-render when sort changes
  useEffect(() => {
    // This effect ensures the component re-renders when sort state changes
  }, [sortField, sortDirection]);

  return (
    <Card className="!rounded-none !overflow-hidden flex flex-col h-full min-h-0">
      <CardHeader className="py-2 px-6">
        <CardTitle className="text-lg">{t('documentPanel.documentManager.title')}</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0 overflow-auto">
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
          <ClearDocumentsDialog onDocumentsCleared={fetchDocuments} />
          <UploadDocumentsDialog onDocumentsUploaded={fetchDocuments} />
          <PipelineStatusDialog
            open={showPipelineStatus}
            onOpenChange={setShowPipelineStatus}
          />
        </div>

        <Card className="flex-1 flex flex-col border rounded-md min-h-0 mb-2">
          <CardHeader className="flex-none py-2 px-4">
            <div className="flex justify-between items-center">
              <CardTitle>{t('documentPanel.documentManager.uploadedTitle')}</CardTitle>
              <div className="flex items-center gap-2">
                <FilterIcon className="h-4 w-4" />
                <div className="flex gap-1" dir={i18n.dir()}>
                  <Button
                    size="sm"
                    variant={statusFilter === 'all' ? 'secondary' : 'outline'}
                    onClick={() => setStatusFilter('all')}
                    className={cn(
                      statusFilter === 'all' && 'bg-gray-100 dark:bg-gray-900 font-medium border border-gray-400 dark:border-gray-500 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.status.all')} ({documentCounts.all})
                  </Button>
                  <Button
                    size="sm"
                    variant={statusFilter === 'processed' ? 'secondary' : 'outline'}
                    onClick={() => setStatusFilter('processed')}
                    className={cn(
                      documentCounts.processed > 0 ? 'text-green-600' : 'text-gray-500',
                      statusFilter === 'processed' && 'bg-green-100 dark:bg-green-900/30 font-medium border border-green-400 dark:border-green-600 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.status.completed')} ({documentCounts.processed || 0})
                  </Button>
                  <Button
                    size="sm"
                    variant={statusFilter === 'processing' ? 'secondary' : 'outline'}
                    onClick={() => setStatusFilter('processing')}
                    className={cn(
                      documentCounts.processing > 0 ? 'text-blue-600' : 'text-gray-500',
                      statusFilter === 'processing' && 'bg-blue-100 dark:bg-blue-900/30 font-medium border border-blue-400 dark:border-blue-600 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.status.processing')} ({documentCounts.processing || 0})
                  </Button>
                  <Button
                    size="sm"
                    variant={statusFilter === 'pending' ? 'secondary' : 'outline'}
                    onClick={() => setStatusFilter('pending')}
                    className={cn(
                      documentCounts.pending > 0 ? 'text-yellow-600' : 'text-gray-500',
                      statusFilter === 'pending' && 'bg-yellow-100 dark:bg-yellow-900/30 font-medium border border-yellow-400 dark:border-yellow-600 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.status.pending')} ({documentCounts.pending || 0})
                  </Button>
                  <Button
                    size="sm"
                    variant={statusFilter === 'failed' ? 'secondary' : 'outline'}
                    onClick={() => setStatusFilter('failed')}
                    className={cn(
                      documentCounts.failed > 0 ? 'text-red-600' : 'text-gray-500',
                      statusFilter === 'failed' && 'bg-red-100 dark:bg-red-900/30 font-medium border border-red-400 dark:border-red-600 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.status.failed')} ({documentCounts.failed || 0})
                  </Button>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <label
                  htmlFor="toggle-filename-btn"
                  className="text-sm text-gray-500"
                >
                  {t('documentPanel.documentManager.fileNameLabel')}
                </label>
                <Button
                  id="toggle-filename-btn"
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
                <div className="absolute inset-[-1px] flex flex-col p-0 border rounded-md border-gray-200 dark:border-gray-700 overflow-hidden">
                  <Table className="w-full">
                    <TableHeader className="sticky top-0 bg-background z-10 shadow-sm">
                      <TableRow className="border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/75 shadow-[inset_0_-1px_0_rgba(0,0,0,0.1)]">
                        <TableHead
                          onClick={() => handleSort('id')}
                          className="cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 select-none"
                        >
                          <div className="flex items-center">
                            {t('documentPanel.documentManager.columns.id')}
                            {sortField === 'id' && (
                              <span className="ml-1">
                                {sortDirection === 'asc' ? <ArrowUpIcon size={14} /> : <ArrowDownIcon size={14} />}
                              </span>
                            )}
                          </div>
                        </TableHead>
                        <TableHead>{t('documentPanel.documentManager.columns.summary')}</TableHead>
                        <TableHead>{t('documentPanel.documentManager.columns.status')}</TableHead>
                        <TableHead>{t('documentPanel.documentManager.columns.length')}</TableHead>
                        <TableHead>{t('documentPanel.documentManager.columns.chunks')}</TableHead>
                        <TableHead
                          onClick={() => handleSort('created_at')}
                          className="cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 select-none"
                        >
                          <div className="flex items-center">
                            {t('documentPanel.documentManager.columns.created')}
                            {sortField === 'created_at' && (
                              <span className="ml-1">
                                {sortDirection === 'asc' ? <ArrowUpIcon size={14} /> : <ArrowDownIcon size={14} />}
                              </span>
                            )}
                          </div>
                        </TableHead>
                        <TableHead
                          onClick={() => handleSort('updated_at')}
                          className="cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 select-none"
                        >
                          <div className="flex items-center">
                            {t('documentPanel.documentManager.columns.updated')}
                            {sortField === 'updated_at' && (
                              <span className="ml-1">
                                {sortDirection === 'asc' ? <ArrowUpIcon size={14} /> : <ArrowDownIcon size={14} />}
                              </span>
                            )}
                          </div>
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody className="text-sm overflow-auto">
                      {filteredAndSortedDocs && filteredAndSortedDocs.map((doc) => (
                        <TableRow key={doc.id}>
                          <TableCell className="truncate font-mono overflow-visible max-w-[250px]">
                            {showFileName ? (
                              <>
                                <div className="group relative overflow-visible tooltip-container">
                                  <div className="truncate">
                                    {getDisplayFileName(doc, 30)}
                                  </div>
                                  <div className="invisible group-hover:visible tooltip">
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
                                <div className="invisible group-hover:visible tooltip">
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
                              <div className="invisible group-hover:visible tooltip">
                                {doc.content_summary}
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>
                            {doc.status === 'processed' && (
                              <span className="text-green-600">{t('documentPanel.documentManager.status.completed')}</span>
                            )}
                            {doc.status === 'processing' && (
                              <span className="text-blue-600">{t('documentPanel.documentManager.status.processing')}</span>
                            )}
                            {doc.status === 'pending' && (
                              <span className="text-yellow-600">{t('documentPanel.documentManager.status.pending')}</span>
                            )}
                            {doc.status === 'failed' && (
                              <span className="text-red-600">{t('documentPanel.documentManager.status.failed')}</span>
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
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </CardContent>
    </Card>
  )
}
