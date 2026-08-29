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
import Checkbox from '@/components/ui/Checkbox'
import UploadDocumentsDialog from '@/components/documents/UploadDocumentsDialog'
import ClearDocumentsDialog from '@/components/documents/ClearDocumentsDialog'
import DeleteDocumentsDialog from '@/components/documents/DeleteDocumentsDialog'
import PaginationControls from '@/components/ui/PaginationControls'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/Dialog'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'

import {
  scanNewDocuments,
  getDocumentsPaginatedWithTimeout,
  DocsStatusesResponse,
  DocStatus,
  DocStatusResponse,
  DocumentsRequest,
  PaginatedDocsResponse,
  PaginationInfo
} from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { toast } from 'sonner'
import { useBackendState } from '@/stores/state'
import { copyToClipboard } from '@/utils/clipboard'

import { RefreshCwIcon, ActivityIcon, ArrowUpIcon, ArrowDownIcon, RotateCcwIcon, CheckSquareIcon, XIcon, AlertTriangle, Info, CopyIcon } from 'lucide-react'
import PipelineStatusDialog from '@/components/documents/PipelineStatusDialog'
import {
  getStatusBucket,
  getStatusRequestFilters,
  matchesStatusFilter,
  type StatusBucket,
  type StatusFilter
} from '@/features/documentStatusFilters'
import { hasDocumentWarning } from '@/features/documentRecoveryWarnings'
import {
  admitCircuitBreakerRequest,
  initialCircuitBreakerState,
  isCircuitBreakerBlocked,
  recordCircuitBreakerFailure,
  resetCircuitBreaker,
  settleCircuitBreakerProbe,
  type CircuitBreakerState
} from '@/features/documentRefreshCircuitBreaker'
import { classifyDocumentRefreshError } from '@/features/documentRefreshErrors'
import { createRefreshQueue } from '@/features/documentRefreshQueue'

type StatusDisplayConfig = {
  labelKey: string
  className: string
}

const STATUS_BUCKETS: StatusBucket[] = ['completed', 'parse', 'analyze', 'process', 'failed']

// Utility functions defined outside component for better performance and to avoid dependency issues
const getCountValue = (counts: Record<string, number>, ...keys: string[]): number => {
  for (const key of keys) {
    const value = counts[key]
    if (typeof value === 'number') {
      return value
    }
  }
  return 0
}

const getAggregateCount = (counts: Record<string, number>, ...keys: string[]): number =>
  keys.reduce((total, key) => total + getCountValue(counts, key), 0)

const hasActiveDocumentsStatus = (counts: Record<string, number>): boolean =>
  getAggregateCount(counts, 'PROCESSING', 'processing', 'PARSING', 'parsing', 'ANALYZING', 'analyzing') > 0 ||
  getCountValue(counts, 'PENDING', 'pending') > 0 ||
  getCountValue(counts, 'PREPROCESSED', 'preprocessed') > 0

const buildLegacyDocs = (documents: DocStatusResponse[]): DocsStatusesResponse => {
  const statuses = STATUS_BUCKETS.reduce<Record<StatusBucket, DocStatusResponse[]>>((acc, status) => {
    acc[status] = []
    return acc
  }, {} as Record<StatusBucket, DocStatusResponse[]>)

  documents.forEach((doc) => {
    const bucket = getStatusBucket(doc.status)
    if (bucket) statuses[bucket].push(doc)
  })

  return { statuses }
}

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

const formatMetadata = (metadata: Record<string, any>): string => {
  const formattedMetadata = { ...metadata };

  if (formattedMetadata.parse_start_time && typeof formattedMetadata.parse_start_time === 'number') {
    const date = new Date(formattedMetadata.parse_start_time * 1000);
    if (!isNaN(date.getTime())) {
      formattedMetadata.parse_start_time = date.toLocaleString();
    }
  }

  if (formattedMetadata.parse_end_time && typeof formattedMetadata.parse_end_time === 'number') {
    const date = new Date(formattedMetadata.parse_end_time * 1000);
    if (!isNaN(date.getTime())) {
      formattedMetadata.parse_end_time = date.toLocaleString();
    }
  }

  if (formattedMetadata.analyzing_start_time && typeof formattedMetadata.analyzing_start_time === 'number') {
    const date = new Date(formattedMetadata.analyzing_start_time * 1000);
    if (!isNaN(date.getTime())) {
      formattedMetadata.analyzing_start_time = date.toLocaleString();
    }
  }

  if (formattedMetadata.analyzing_end_time && typeof formattedMetadata.analyzing_end_time === 'number') {
    const date = new Date(formattedMetadata.analyzing_end_time * 1000);
    if (!isNaN(date.getTime())) {
      formattedMetadata.analyzing_end_time = date.toLocaleString();
    }
  }

  if (formattedMetadata.process_start_time && typeof formattedMetadata.process_start_time === 'number') {
    const date = new Date(formattedMetadata.process_start_time * 1000);
    if (!isNaN(date.getTime())) {
      formattedMetadata.process_start_time = date.toLocaleString();
    }
  }

  if (formattedMetadata.process_end_time && typeof formattedMetadata.process_end_time === 'number') {
    const date = new Date(formattedMetadata.process_end_time * 1000);
    if (!isNaN(date.getTime())) {
      formattedMetadata.process_end_time = date.toLocaleString();
    }
  }

  const formatValue = (value: any, indent = 0): string => {
    if (value === null) return 'null';
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      return String(value);
    }

    if (Array.isArray(value)) {
      if (value.length === 0) return '[]';
      return `[${value
        .map(item => {
          if (item === null) return 'null';
          if (typeof item === 'object') return JSON.stringify(item);
          return String(item);
        })
        .join(', ')}]`;
    }

    if (typeof value === 'object') {
      const entries = Object.entries(value).sort(([a], [b]) => a.localeCompare(b));
      if (entries.length === 0) return '{}';
      return entries
        .map(([key, child]) => {
          const renderedChild = formatValue(child, indent + 1);
          const prefix = '  '.repeat(indent);
          if (typeof child === 'object' && child !== null && !Array.isArray(child)) {
            return `${prefix}${key}:\n${renderedChild}`;
          }
          return `${prefix}${key}: ${renderedChild}`;
        })
        .join('\n');
    }

    return String(value);
  };

  return Object.entries(formattedMetadata)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, value]) => {
      const rendered = formatValue(value, 1);
      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        return `${key}:\n${rendered}`;
      }
      return `${key}: ${rendered}`;
    })
    .join('\n');
};

const hasDocumentDetails = (doc: DocStatusResponse): boolean => {
  return Boolean(
    doc.track_id ||
    doc.error_msg ||
    (doc.metadata && Object.keys(doc.metadata).length > 0)
  )
}

const formatDocumentDetails = (doc: DocStatusResponse): string => {
  const details: string[] = []

  if (doc.track_id) {
    details.push(`Track ID: ${doc.track_id}`)
  }

  if (doc.metadata && Object.keys(doc.metadata).length > 0) {
    details.push(formatMetadata(doc.metadata))
  }

  if (doc.error_msg) {
    details.push(`Error Message:\n${doc.error_msg}`)
  }

  return details.join('\n\n')
}

const DocumentStatusDetailsDialog = ({ doc }: { doc: DocStatusResponse }) => {
  const { t } = useTranslation()
  const details = formatDocumentDetails(doc)

  const openLabel = t('documentPanel.documentManager.details.openTooltip')
  const copyLabel = t('documentPanel.documentManager.details.copyTooltip')

  const handleCopy = async () => {
    const result = await copyToClipboard(details)

    if (result.success) {
      toast.success(t('documentPanel.documentManager.details.copySuccess'))
    } else {
      toast.error(t('documentPanel.documentManager.details.copyFailed'))
    }
  }

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="ml-2 size-7"
          tooltip={openLabel}
          side="top"
          aria-label={openLabel}
        >
          {hasDocumentWarning(doc) ? (
            <AlertTriangle className="h-4 w-4 text-yellow-500" />
          ) : (
            <Info className="h-4 w-4 text-blue-500" />
          )}
        </Button>
      </DialogTrigger>
      <DialogContent
        className="max-w-2xl"
        onOpenAutoFocus={(e) => {
          e.preventDefault()
          ;(e.currentTarget as HTMLElement | null)?.focus()
        }}
        onCloseAutoFocus={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle>{t('documentPanel.documentManager.details.title')}</DialogTitle>
          <DialogDescription className="break-all">
            {doc.id}
          </DialogDescription>
        </DialogHeader>

        <div className="relative rounded-md border bg-muted/30">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="absolute top-2 right-2 z-10 size-7 bg-background/80 hover:bg-accent"
            onClick={handleCopy}
            tooltip={copyLabel}
            side="left"
            aria-label={copyLabel}
          >
            <CopyIcon className="h-4 w-4" />
          </Button>
          <div className="max-h-[60vh] overflow-y-auto p-3 pr-12">
            <pre className="whitespace-pre-wrap break-words text-sm">{details}</pre>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

const pulseStyle = `
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
type SortField = 'created_at' | 'updated_at' | 'id' | 'file_path';
type SortDirection = 'asc' | 'desc';
type QuerySnapshot = {
  statusFilter: StatusFilter
  page: number
  pageSize: number
  sortField: SortField
  sortDirection: SortDirection
}
type RefreshRequest =
  | {
    type: 'intelligent';
    query: QuerySnapshot;
    customTimeout?: number;
    requestVersion: number;
    // Issued by the polling timer / activity probe rather than by a user
    // action. Only these are subject to the circuit breaker: a page change or
    // a manual refresh is an explicit intent and stays the escape hatch.
    auto?: boolean;
  }
  | {
    type: 'manual';
    query: QuerySnapshot;
    requestVersion: number;
  };

/**
 * Stable id for the document-refresh failure toast, so repeated failures
 * update one notice in place instead of stacking. Dismissed by the first
 * successful response.
 */
const DOCUMENT_REFRESH_FAILURE_TOAST_ID = 'document-refresh-failed'

export default function DocumentManager() {
  // Track component mount status
  const isMountedRef = useRef(true);

  // Set up mount/unmount status tracking. Pending throttle/probe timers are NOT
  // explicitly cleared on unmount — every timer callback checks isMountedRef
  // before doing any work, so a stray fire is a no-op.
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
      // The Toaster is mounted above the router, so the outage notice outlives
      // this component: without this it would still be on screen after a
      // logout, with nothing left that could ever dismiss it.
      toast.dismiss(DOCUMENT_REFRESH_FAILURE_TOAST_ID);
    };
  }, []);

  const [showPipelineStatus, setShowPipelineStatus] = useState(false)
  const { t, i18n } = useTranslation()
  const health = useBackendState.use.health()
  const pipelineActive = useBackendState.use.pipelineActive()

  // Legacy state for backward compatibility
  const [docs, setDocs] = useState<DocsStatusesResponse | null>(null)

  const currentTab = useSettingsStore.use.currentTab()
  const showFileName = useSettingsStore.use.showFileName()
  const setShowFileName = useSettingsStore.use.setShowFileName()
  const documentsPageSize = useSettingsStore.use.documentsPageSize()
  const setDocumentsPageSize = useSettingsStore.use.setDocumentsPageSize()

  // New pagination state
  const [currentPageDocs, setCurrentPageDocs] = useState<DocStatusResponse[]>([])
  const [pagination, setPagination] = useState<PaginationInfo>({
    page: 1,
    page_size: documentsPageSize,
    total_count: 0,
    total_pages: 0,
    has_next: false,
    has_prev: false
  })
  const [statusCounts, setStatusCounts] = useState<Record<string, number>>({ all: 0 })
  // Mirror statusCounts in a ref so async callbacks (e.g. activity probe ticks)
  // can read the latest value without being tied to the closure captured at
  // schedule time. Synced via useEffect to satisfy react-hooks/refs.
  const statusCountsRef = useRef(statusCounts)
  useEffect(() => {
    statusCountsRef.current = statusCounts
  }, [statusCounts])
  const [isRefreshing, setIsRefreshing] = useState(false)

  // Sort state
  const [sortField, setSortField] = useState<SortField>('updated_at')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')

  // State for document status filter
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');

  // State to store page number for each status filter
  const [pageByStatus, setPageByStatus] = useState<Record<StatusFilter, number>>({
    all: 1,
    completed: 1,
    parse: 1,
    analyze: 1,
    process: 1,
    failed: 1,
  });

  // State for document selection
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])
  const isSelectionMode = selectedDocIds.length > 0

  // Add refs to track previous pipelineActive state and current interval
  const prevPipelineActiveRef = useRef<boolean | undefined>(undefined);
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const latestRefreshRequestVersionRef = useRef(0);
  // Throttle gate: every /documents/paginated entrance that is not a direct
  // page/filter/sort change funnels through refreshDocumentsThrottled() to
  // enforce a minimum 2s wall-clock interval.
  const lastPaginatedAtRef = useRef(0);
  const pendingPaginatedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Intent tag the pending trailing timer will fire with. Held outside the
  // closure so a user-intent call arriving during the wait can upgrade it.
  const pendingPaginatedIsAutoRef = useRef(true);
  // Activity probe: exponential-backoff burst of /health calls that stops once
  // pipelineActive flips true. Holds the pending setTimeout ids so re-entry can
  // reset the schedule to t=0.
  const probeTimersRef = useRef<ReturnType<typeof setTimeout>[] | null>(null);
  const probeActiveRef = useRef(false);

  // Circuit breaker guarding the automatic /documents/paginated polling.
  // Transitions live in features/documentRefreshCircuitBreaker (unit-tested);
  // this ref just holds the current state.
  const circuitBreakerRef = useRef<CircuitBreakerState>(initialCircuitBreakerState);


  // Handle checkbox change for individual documents
  const handleDocumentSelect = useCallback((docId: string, checked: boolean) => {
    setSelectedDocIds(prev => {
      if (checked) {
        return [...prev, docId]
      } else {
        return prev.filter(id => id !== docId)
      }
    })
  }, [])

  // Handle deselect all documents
  const handleDeselectAll = useCallback(() => {
    setSelectedDocIds([])
  }, [])

  // Handle sort column click
  const handleSort = (field: SortField) => {
    let actualField = field;

    // When clicking the first column, determine the actual sort field based on showFileName
    if (field === 'id') {
      actualField = showFileName ? 'file_path' : 'id';
    }

    const newDirection = (sortField === actualField && sortDirection === 'desc') ? 'asc' : 'desc';

    setSortField(actualField);
    setSortDirection(newDirection);

    // Reset page to 1 when sorting changes
    setPagination(prev => ({ ...prev, page: 1 }));

    // Reset all status filters' page memory since sorting affects all
    setPageByStatus({
      all: 1,
      completed: 1,
      parse: 1,
      analyze: 1,
      process: 1,
      failed: 1,
    });
  };

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

  const getStatusDisplay = useCallback((status: DocStatus): StatusDisplayConfig => {
    switch (status) {
      case 'processed':
        return {
          labelKey: 'documentPanel.documentManager.status.completed',
          className: 'text-green-600'
        }
      case 'preprocessed':
        return {
          labelKey: 'documentPanel.documentManager.status.preprocessed',
          className: 'text-purple-600'
        }
      case 'parsing':
        return {
          labelKey: 'documentPanel.documentManager.status.parsing',
          className: 'text-cyan-600'
        }
      case 'analyzing':
        return {
          labelKey: 'documentPanel.documentManager.status.analyzing',
          className: 'text-indigo-600'
        }
      case 'processing':
        return {
          labelKey: 'documentPanel.documentManager.status.processing',
          className: 'text-blue-600'
        }
      case 'pending':
        return {
          labelKey: 'documentPanel.documentManager.status.pending',
          className: 'text-yellow-600'
        }
      case 'failed':
      default:
        return {
          labelKey: 'documentPanel.documentManager.status.failed',
          className: 'text-red-600'
        }
    }
  }, [])

  const filteredAndSortedDocs = useMemo(() => {
    // Use currentPageDocs directly if available (from paginated API)
    // This preserves the backend's sort order and prevents status grouping
    if (currentPageDocs && currentPageDocs.length > 0) {
      return currentPageDocs.map(doc => ({
        ...doc,
        status: doc.status as DocStatus
      })) as DocStatusWithStatus[];
    }

    // Fallback to legacy docs structure for backward compatibility
    if (!docs) return null;

    // Create a flat array of documents with status information
    const allDocuments: DocStatusWithStatus[] = [];

    Object.entries(docs.statuses).forEach(([status, documents]) => {
      const fallbackStatus = status as DocStatus

      for (const doc of documents ?? []) {
        const documentStatus = doc.status ?? fallbackStatus

        if (matchesStatusFilter(documentStatus, statusFilter)) {
          allDocuments.push({
            ...doc,
            status: documentStatus
          })
        }
      }
    })

    // Sort all documents together if sort field and direction are specified
    if (sortField && sortDirection) {
      return sortDocuments(allDocuments);
    }

    return allDocuments;
  }, [currentPageDocs, docs, sortField, sortDirection, statusFilter, sortDocuments]);

  // Calculate current page selection state (after filteredAndSortedDocs is defined)
  const currentPageDocIds = useMemo(() => {
    return filteredAndSortedDocs?.map(doc => doc.id) || []
  }, [filteredAndSortedDocs])

  const selectedCurrentPageCount = useMemo(() => {
    return currentPageDocIds.filter(id => selectedDocIds.includes(id)).length
  }, [currentPageDocIds, selectedDocIds])

  const isCurrentPageFullySelected = useMemo(() => {
    return currentPageDocIds.length > 0 && selectedCurrentPageCount === currentPageDocIds.length
  }, [currentPageDocIds, selectedCurrentPageCount])

  const hasCurrentPageSelection = useMemo(() => {
    return selectedCurrentPageCount > 0
  }, [selectedCurrentPageCount])

  // Handle select current page
  const handleSelectCurrentPage = useCallback(() => {
    setSelectedDocIds(currentPageDocIds)
  }, [currentPageDocIds])


  // Get selection button properties
  const getSelectionButtonProps = useCallback(() => {
    if (!hasCurrentPageSelection) {
      return {
        text: t('documentPanel.selectDocuments.selectCurrentPage', { count: currentPageDocIds.length }),
        action: handleSelectCurrentPage,
        icon: CheckSquareIcon
      }
    } else if (isCurrentPageFullySelected) {
      return {
        text: t('documentPanel.selectDocuments.deselectAll', { count: currentPageDocIds.length }),
        action: handleDeselectAll,
        icon: XIcon
      }
    } else {
      return {
        text: t('documentPanel.selectDocuments.selectCurrentPage', { count: currentPageDocIds.length }),
        action: handleSelectCurrentPage,
        icon: CheckSquareIcon
      }
    }
  }, [hasCurrentPageSelection, isCurrentPageFullySelected, currentPageDocIds.length, handleSelectCurrentPage, handleDeselectAll, t])

  // Calculate document counts for each status
  const documentCounts = useMemo(() => {
    if (!docs) return { all: 0 } as Record<string, number>;

    const counts: Record<string, number> = { all: 0 };

    Object.entries(docs.statuses).forEach(([status, documents]) => {
      counts[status] = documents.length;
      counts.all += documents.length;
    });

    return counts;
  }, [docs]);

  const completedCount = getCountValue(statusCounts, 'PROCESSED', 'processed') || documentCounts.completed || 0;
  const parseCount = getCountValue(statusCounts, 'PARSING', 'parsing') || documentCounts.parse || 0;
  const analyzeCount = getCountValue(statusCounts, 'ANALYZING', 'analyzing') || documentCounts.analyze || 0;
  const processCount = getCountValue(statusCounts, 'PROCESSING', 'processing') || documentCounts.process || 0;
  const failedCount = getCountValue(statusCounts, 'FAILED', 'failed') || documentCounts.failed || 0;

  // Store previous status counts
  // Fingerprint of per-bucket counts. Kept key-agnostic on purpose: `docs.statuses`
  // may be keyed by raw DocStatus (backend) or by filter bucket (buildLegacyDocs),
  // so we compare a stable digest of all buckets rather than fixed keys.
  const prevStatusFingerprint = useRef('')

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

  const buildQuerySnapshot = useCallback((
    overrides: Partial<QuerySnapshot> = {}
  ): QuerySnapshot => ({
    statusFilter: overrides.statusFilter ?? statusFilter,
    page: overrides.page ?? pagination.page,
    pageSize: overrides.pageSize ?? pagination.page_size,
    sortField: overrides.sortField ?? sortField,
    sortDirection: overrides.sortDirection ?? sortDirection
  }), [pagination.page, pagination.page_size, sortField, sortDirection, statusFilter])

  const buildDocumentsRequest = useCallback((
    query: QuerySnapshot,
    page: number = query.page
  ): DocumentsRequest => ({
    ...getStatusRequestFilters(query.statusFilter),
    page,
    page_size: query.pageSize,
    sort_field: query.sortField,
    sort_direction: query.sortDirection
  }), [])

  // Utility function to update component state.
  //
  // `page` is passed in rather than read off the response: the displayed page
  // number is what THIS request asked for (or an explicit boundary correction),
  // never what the response echoed back. Request serialization plus the
  // requestVersion guard make a stale response unreachable today, so this
  // changes no behaviour; deriving the page locally means neither a backend
  // echo bug nor a future relaxation of either layer can surface as a page
  // number rolling backwards under the user.
  const updateComponentState = useCallback((response: any, page: number) => {
    setPagination({ ...response.pagination, page });
    setCurrentPageDocs(response.documents);
    setStatusCounts(response.status_counts);

    setDocs(response.pagination.total_count > 0 ? buildLegacyDocs(response.documents) : null);
  }, []);


  // Circuit breaker utility functions. State lives in a ref, not useState: the
  // breaker is read from timer callbacks only, and a state write would change
  // startPollingInterval's identity and make the polling useEffect tear down
  // and recreate the interval, resetting its phase on every failure.
  // Ask for permission to issue one automatic request. Past the backoff
  // deadline this admits a single half-open probe and keeps the breaker open
  // until that probe settles, so the ticks that arrive during its (up to 30s)
  // lifetime do not slip through behind it.
  const admitAutomaticRefresh = useCallback(() => {
    const { state, admitted } = admitCircuitBreakerRequest(
      circuitBreakerRef.current,
      Date.now()
    );
    circuitBreakerRef.current = state;
    return admitted;
  }, []);

  // Read-only check for a request that was already queued: it must not take
  // the half-open slot, so it only gets to run while the breaker is closed.
  const isAutomaticRefreshBlocked = useCallback(
    () => isCircuitBreakerBlocked(circuitBreakerRef.current, Date.now()),
    []
  );

  const recordFailure = useCallback(() => {
    circuitBreakerRef.current = recordCircuitBreakerFailure(
      circuitBreakerRef.current,
      Date.now()
    );
  }, []);

  // Call ONLY when a response actually came back. Recording success on a
  // fire-and-forget refresh (what the polling tick used to do) clears the
  // failure counter on every tick and the breaker can never open.
  const recordSuccess = useCallback(() => {
    circuitBreakerRef.current = resetCircuitBreaker();
  }, []);

  // Return the half-open probe slot for an outcome that proves nothing about
  // reachability. Idempotent, so it can be called unconditionally from the one
  // place every request path passes through.
  const settleProbe = useCallback(() => {
    circuitBreakerRef.current = settleCircuitBreakerProbe(
      circuitBreakerRef.current,
      Date.now()
    );
  }, []);

  // Handle page size change - update state and save to store
  const handlePageSizeChange = useCallback((newPageSize: number) => {
    if (newPageSize === pagination.page_size) return;

    // Save the new page size to the store
    setDocumentsPageSize(newPageSize);

    // Reset all status filters to page 1 when page size changes
    setPageByStatus({
      all: 1,
      completed: 1,
      parse: 1,
      analyze: 1,
      process: 1,
      failed: 1,
    });

    setPagination(prev => ({ ...prev, page: 1, page_size: newPageSize }));
  }, [pagination.page_size, setDocumentsPageSize]);

  const runRefreshRequest = useCallback(async (refreshRequest: RefreshRequest) => {
    try {
      if (!isMountedRef.current) return;

      setIsRefreshing(true);

      const { query, requestVersion } = refreshRequest
      const isStaleRequest = () => requestVersion !== latestRefreshRequestVersionRef.current

      // Single entrance for the paginated fetch so the circuit breaker is fed
      // by real responses. A 200 proves the backend is reachable even when the
      // payload is then discarded as stale, so success is recorded before any
      // staleness check.
      const fetchPage = async (
        pageRequest: DocumentsRequest,
        timeoutMs?: number
      ): Promise<PaginatedDocsResponse> => {
        const response = await getDocumentsPaginatedWithTimeout(pageRequest, timeoutMs)
        recordSuccess()
        toast.dismiss(DOCUMENT_REFRESH_FAILURE_TOAST_ID)
        return response
      }

      if (refreshRequest.type === 'manual') {
        const request = buildDocumentsRequest(query, 1)
        const response = await fetchPage(request)

        if (!isMountedRef.current || isStaleRequest()) return;

        if (response.pagination.total_count < query.pageSize && query.pageSize !== 10) {
          handlePageSizeChange(10);
        } else {
          // Manual refresh always requests page 1; same rule as
          // updateComponentState — the page shown is the page asked for.
          setPagination({ ...response.pagination, page: 1 });
          setCurrentPageDocs(response.documents);
          setStatusCounts(response.status_counts);

          const legacyDocs: DocsStatusesResponse = {
            statuses: {
              processed: response.documents.filter(doc => doc.status === 'processed'),
              preprocessed: response.documents.filter(doc => doc.status === 'preprocessed'),
              processing: response.documents.filter(doc => doc.status === 'processing'),
              pending: response.documents.filter(doc => doc.status === 'pending'),
              failed: response.documents.filter(doc => doc.status === 'failed')
            }
          };

          if (response.pagination.total_count > 0) {
            setDocs(legacyDocs);
          } else {
            setDocs(null);
          }
        }
      } else {
        const { customTimeout } = refreshRequest;
        const pageToFetch = query.page;
        const request = buildDocumentsRequest(query, pageToFetch)
        const response = await fetchPage(request, customTimeout)

        if (!isMountedRef.current || isStaleRequest()) return;

        // Boundary case handling: if target page has no data but total count > 0
        if (response.documents.length === 0 && response.pagination.total_count > 0) {
          const lastPage = Math.max(1, response.pagination.total_pages);

          if (pageToFetch !== lastPage) {
            const lastPageRequest = buildDocumentsRequest(query, lastPage)
            const lastPageResponse = await fetchPage(lastPageRequest, customTimeout)

            if (!isMountedRef.current || isStaleRequest()) return;

            setPageByStatus(prev => ({ ...prev, [query.statusFilter]: lastPage }));
            // Explicit: pageByStatus is only read when the status filter
            // changes, so this call is the only thing moving pagination.page
            // to the corrected page.
            updateComponentState(lastPageResponse, lastPage);
            return;
          }
        }

        setPageByStatus(prev => (
          prev[query.statusFilter] === pageToFetch
            ? prev
            : { ...prev, [query.statusFilter]: pageToFetch }
        ));
        updateComponentState(response, pageToFetch);
      }

    } catch (err) {
      if (isMountedRef.current) {
        const errorClassification = classifyDocumentRefreshError(err);

        if (errorClassification.shouldRetry) {
          // No answer at all, or an answer that is itself a back-off signal
          // (408/429). shouldRetry outranks provesBackendResponsive by design.
          recordFailure();
        } else if (errorClassification.provesBackendResponsive) {
          // A permanent 4xx: the application parsed and routed the request and
          // answered. Reachability is proven, so the breaker must be CLEARED
          // rather than left holding a stale failure count that would let three
          // non-consecutive failures open it.
          recordSuccess();
        }
        // Anything left — a cancellation — proves nothing and must not reset,
        // but must not keep holding the probe slot either: the finally settles
        // it below.

        if (errorClassification.shouldShowToast) {
          // One notice for an ongoing outage, not one per failed poll. Every
          // retryable failure means the list is stale and staying stale, so the
          // notice stands until a response clears it (fetchPage dismisses it on
          // success). StatusIndicator's red dot covers the plain "backend is
          // unreachable" case, but not this one: it tracks /health, so a
          // backend that answers /health while /documents/paginated keeps
          // failing shows green over a frozen list, and it is not rendered at
          // all when the user turns health checks off.
          //
          // The stable id alone is NOT enough: it merges into a toast that is
          // still on screen, and the default 4s lifetime always expires before
          // the next poll (5s at its fastest, 30s once health drops). Every
          // failure would create a fresh toast, which is the flood this is
          // meant to remove. Gating persistence on the breaker being OPEN had
          // the same hole for the failures that precede the threshold.
          //
          // A permanent 4xx is not an outage — the backend answered, and the
          // breaker has just been cleared above — so it keeps the ordinary
          // auto-dismissing toast.
          toast.error(t('documentPanel.documentManager.errors.loadFailed', { error: errorMessage(err) }), {
            id: DOCUMENT_REFRESH_FAILURE_TOAST_ID,
            duration: errorClassification.shouldRetry ? Infinity : undefined
          });
        }
      }
    } finally {
      // The probe slot is held for the lifetime of ONE real request, so every
      // exit path must return it. recordSuccess / recordFailure already clear
      // it; this covers the paths that do neither — a cancellation, and the
      // unmount early return at the top of this function, which precedes every
      // record call. Idempotent, hence unconditional: the invariant is "every
      // exit returns the slot", and finally is the only construct that states
      // it. An else-branch in the catch would cover today's paths and silently
      // reopen the hole the next time an early return is added.
      settleProbe();
      if (isMountedRef.current) {
        setIsRefreshing(false);
      }
    }
  }, [
    t,
    updateComponentState,
    recordFailure,
    recordSuccess,
    settleProbe,
    handlePageSizeChange,
    buildDocumentsRequest
  ]);

  // One queue for the component's lifetime: it owns the in-flight/pending
  // state, so rebuilding it would drop a request mid-flight. The handlers ride
  // along with each enqueue instead of being captured at construction.
  const [refreshQueue] = useState(() => createRefreshQueue<RefreshRequest>());

  const enqueueRefresh = useCallback(
    (refreshRequest: RefreshRequest) =>
      refreshQueue.enqueue(refreshRequest, {
        runRequest: runRefreshRequest,
        // The single admission point for automatic refreshes, evaluated where
        // the request is actually issued — first request and queued request
        // alike. Manual refresh and page/filter/sort changes are untagged and
        // always run: explicit user intent is the escape hatch.
        admit: (request) =>
          request.type !== 'intelligent' ||
          !request.auto ||
          admitAutomaticRefresh(),
        // Automatic refreshes rank below everything else in the pending slot.
        // Admission is evaluated where the request is issued, so an `auto`
        // request the breaker will refuse there must not be able to discard a
        // page change or a manual refresh that would have run.
        priority: (request) =>
          request.type === 'intelligent' && request.auto ? 0 : 1
      }),
    [refreshQueue, runRefreshRequest, admitAutomaticRefresh]
  );

  // Intelligent refresh function: handles all boundary cases
  const handleIntelligentRefresh = useCallback(async (options: {
    targetPage?: number
    resetToFirst?: boolean
    customTimeout?: number
    auto?: boolean
  } = {}) => {
    const { targetPage, resetToFirst, customTimeout, auto } = options
    const page = resetToFirst ? 1 : (targetPage || pagination.page)
    const query = buildQuerySnapshot({ page })
    const requestVersion = latestRefreshRequestVersionRef.current

    await enqueueRefresh({
      type: 'intelligent',
      query,
      customTimeout,
      requestVersion,
      auto
    });
  }, [buildQuerySnapshot, enqueueRefresh, pagination.page]);

  // Throttle gate: any caller wanting to refresh the document list goes through
  // here. If the wall-clock gap since the last paginated request is >= 2s, fire
  // immediately; otherwise schedule a single trailing call at the 2s boundary
  // and drop any further calls into that pending slot (natural coalescing).
  const refreshDocumentsThrottled = useCallback((options: { auto?: boolean } = {}) => {
    // Defaults to automatic: the timers are the common caller. Callers that
    // follow a user action (upload, scan, delete) pass auto: false so the
    // breaker cannot refuse the refresh their action earned.
    const auto = options.auto ?? true
    const fire = (isAuto: boolean) => {
      lastPaginatedAtRef.current = Date.now()
      handleIntelligentRefresh({ auto: isAuto }).catch((err) => {
        console.error('Throttled document refresh failed:', err)
      })
    }
    const gap = Date.now() - lastPaginatedAtRef.current
    if (gap >= 2000) {
      fire(auto)
      return
    }
    if (pendingPaginatedTimerRef.current !== null) {
      // A user-intent call arriving while an automatic trailing timer waits
      // must UPGRADE it, not be dropped — the same defect the refresh queue's
      // pending slot had, one layer up. The tag decides whether the queue's
      // admission gate may refuse this request 2s from now, so dropping the
      // call here would make the queue's priority rule unreachable.
      // Monotonic: once intent, intent for the rest of this window.
      if (!auto) pendingPaginatedIsAutoRef.current = false
      return
    }
    // Snapshot the query identity. If page/filter/sort changes while we wait,
    // the page-change useEffect bumps latestRefreshRequestVersionRef AND fires
    // its own paginated request on the new query. Our trailing closure still
    // holds the OLD handleIntelligentRefresh (capturing the old page), so we
    // must drop it — otherwise the stale request would overwrite the new list
    // (its requestVersion would be the newly-bumped value, so the in-flight
    // stale-check inside runRefreshRequest can't catch it).
    const versionAtSchedule = latestRefreshRequestVersionRef.current
    pendingPaginatedIsAutoRef.current = auto
    pendingPaginatedTimerRef.current = setTimeout(() => {
      pendingPaginatedTimerRef.current = null
      if (!isMountedRef.current) return
      if (versionAtSchedule !== latestRefreshRequestVersionRef.current) return
      fire(pendingPaginatedIsAutoRef.current)
    }, 2000 - gap)
  }, [handleIntelligentRefresh]);

  // Activity probe: short exponential-backoff burst of /health checks fired
  // after scan/upload triggers. Stops as soon as pipelineActive flips true so
  // we can hand off to the existing 5s active polling cadence. Re-entry
  // (e.g. another scan while a probe is mid-flight) cancels the current
  // schedule and restarts at t=0 so the latest action gets a fresh observation
  // window.
  const startActivityProbe = useCallback((reason: string) => {
    if (probeTimersRef.current) {
      probeTimersRef.current.forEach((id) => clearTimeout(id))
      probeTimersRef.current = null
    }
    probeActiveRef.current = true
    const timers: ReturnType<typeof setTimeout>[] = []
    const probeSchedule = [0, 1000, 2000, 4000, 8000, 16000] as const
    const refreshAt = new Set<number>([0, 2000, 4000, 8000, 16000])
    // The probe follows a user action (scan / upload), so its FIRST refresh
    // carries that intent and must not be refusable by the breaker. The rest
    // of the burst is speculative — it polls for work the action may have
    // started — so tagging all five as intent would let one click bypass the
    // breaker five times.
    const cleanup = () => {
      timers.forEach((id) => clearTimeout(id))
      if (probeTimersRef.current === timers) {
        probeTimersRef.current = null
        probeActiveRef.current = false
      }
    }
    probeSchedule.forEach((delay, index) => {
      const id = setTimeout(async () => {
        if (!isMountedRef.current) {
          cleanup()
          return
        }
        try {
          await useBackendState.getState().check()
        } catch (err) {
          console.error(`Activity probe (${reason}) check failed:`, err)
        }
        if (!isMountedRef.current) {
          cleanup()
          return
        }
        if (refreshAt.has(delay)) {
          refreshDocumentsThrottled({ auto: delay !== 0 })
        }
        // Exit conditions (in priority order):
        //  - pipelineActive=true AND the document list has caught up: the 5s
        //    active polling cadence will take over from here.
        //  - pipelineActive=false after the first tick: the scan/upload didn't
        //    actually start any work (e.g. scan found nothing new, upload was
        //    rejected) — no point continuing to burst /health.
        //  - last tick: time budget exhausted, hand off to the polling loop.
        // Note: NOT stopping on bare `pipelineActive=true` is intentional.
        // /health flips to active on scanning/pending_enqueues before the new
        // doc rows are visible in /documents/paginated, so a premature exit
        // would strand the UI in 30s idle polling while classification is
        // still running.
        const active = useBackendState.getState().pipelineActive
        const docsActive = hasActiveDocumentsStatus(statusCountsRef.current)
        const isLast = index === probeSchedule.length - 1
        const stop = (active && docsActive) || (!active && index > 0) || isLast
        if (stop) {
          cleanup()
        }
      }, delay)
      timers.push(id)
    })
    probeTimersRef.current = timers
  }, [refreshDocumentsThrottled]);

  // New paginated data fetching function
  const fetchPaginatedDocuments = useCallback(async (
    page: number,
    pageSize: number,
    currentStatusFilter: StatusFilter
  ) => {
    // Update pagination state
    setPagination(prev => ({ ...prev, page, page_size: pageSize }));

    // Use intelligent refresh
    await enqueueRefresh({
      type: 'intelligent',
      query: buildQuerySnapshot({
        page,
        pageSize,
        statusFilter: currentStatusFilter
      }),
      requestVersion: latestRefreshRequestVersionRef.current
    });
  }, [buildQuerySnapshot, enqueueRefresh]);

  // Legacy fetchDocuments function for backward compatibility
  const fetchDocuments = useCallback(async () => {
    await fetchPaginatedDocuments(pagination.page, pagination.page_size, statusFilter);
  }, [fetchPaginatedDocuments, pagination.page, pagination.page_size, statusFilter]);

  // Function to clear current polling interval
  const clearPollingInterval = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  }, []);

  // Function to start polling with given interval
  const startPollingInterval = useCallback((intervalMs: number) => {
    clearPollingInterval();

    pollingIntervalRef.current = setInterval(() => {
      if (!isMountedRef.current) return;
      // Read-only: the admission itself happens in the refresh queue, at the
      // point the request is issued. Skipping here just avoids walking the
      // throttle gate for a request that would be dropped anyway.
      if (isAutomaticRefreshBlocked()) return;
      // refreshDocumentsThrottled is fire-and-forget; the breaker is fed from
      // inside runRefreshRequest (recordSuccess on a real response,
      // recordFailure in its catch), never from this tick — a tick proves
      // nothing about the backend and clearing the counter here is what kept
      // the breaker from ever opening.
      refreshDocumentsThrottled();
    }, intervalMs);
  }, [refreshDocumentsThrottled, clearPollingInterval, isAutomaticRefreshBlocked]);

  const scanDocuments = useCallback(async () => {
    try {
      if (!isMountedRef.current) return;

      const { status, message } = await scanNewDocuments();

      if (!isMountedRef.current) return;

      toast.message(message || status);

      if (status === 'scanning_started') {
        // Activity probe drives /health bursts + throttled document refreshes.
        // It exits as soon as pipelineActive flips true, after which the
        // standard 5s polling cadence (driven by hasActiveDocumentsStatus)
        // takes over.
        startActivityProbe('scan');
      } else {
        // scanning_skipped_pipeline_busy: a single check+refresh is enough,
        // no need to start the probe (pipeline is already active).
        useBackendState.getState().check().catch(() => undefined);
        refreshDocumentsThrottled({ auto: false });
      }
    } catch (err) {
      if (isMountedRef.current) {
        toast.error(t('documentPanel.documentManager.errors.scanFailed', { error: errorMessage(err) }));
      }
    }
  }, [t, startActivityProbe, refreshDocumentsThrottled])

  // Handle manual refresh with pagination reset logic
  const handleManualRefresh = useCallback(async () => {
    await enqueueRefresh({
      type: 'manual',
      query: buildQuerySnapshot(),
      requestVersion: latestRefreshRequestVersionRef.current
    });
  }, [buildQuerySnapshot, enqueueRefresh]);

  useEffect(() => {
    latestRefreshRequestVersionRef.current += 1
  }, [pagination.page, pagination.page_size, statusFilter, sortField, sortDirection])

  // Monitor pipelineActive changes and trigger an immediate refresh. The
  // polling interval is reconciled by the main polling useEffect below
  // (which also depends on pipelineActive), so there's no need to re-call
  // startPollingInterval here.
  useEffect(() => {
    if (prevPipelineActiveRef.current !== undefined && prevPipelineActiveRef.current !== pipelineActive) {
      if (currentTab === 'documents' && health && isMountedRef.current) {
        refreshDocumentsThrottled();
      }
    }
    prevPipelineActiveRef.current = pipelineActive;
  }, [
    pipelineActive,
    currentTab,
    health,
    refreshDocumentsThrottled
  ]);

  // Set up intelligent polling with dynamic interval based on document status.
  // Treat pipelineActive=true as enough reason to stay in 5s fast polling even
  // when statusCounts hasn't surfaced pending rows yet — /health flips active
  // during scan classification / upload enqueue, well before the new doc rows
  // appear in /documents/paginated. Without this, the UI would stall in 30s
  // idle polling for several seconds after the user clicked scan/upload.
  // A failed health check no longer stops the list: it drops polling to the
  // idle cadence instead. Latching health=false used to clear the interval
  // outright, so a single 504 from a busy backend froze the document list until
  // the next successful /health probe. The circuit breaker (fed by real
  // responses) is what throttles a backend that is genuinely down.
  useEffect(() => {
    if (currentTab !== 'documents') {
      clearPollingInterval();
      // Retiring the notice with the polling that would clear it. Nothing
      // refreshes the document list while another tab is up, so a standing
      // outage notice could never be dismissed by a successful response, and
      // the Toaster renders it globally — it would sit over the graph or
      // retrieval view indefinitely. An unmount cleanup does not cover this:
      // TabsContent uses forceMount, so this component stays mounted while
      // another tab is active. Returning to Documents re-creates the notice on
      // the next failure, which is the correct behaviour.
      toast.dismiss(DOCUMENT_REFRESH_FAILURE_TOAST_ID);
      return
    }

    const hasActiveDocuments = hasActiveDocumentsStatus(statusCounts);
    const pollingInterval = health
      ? ((hasActiveDocuments || pipelineActive) ? 5000 : 30000)
      : 30000;

    startPollingInterval(pollingInterval);

    return () => {
      clearPollingInterval();
    }
  }, [health, t, currentTab, statusCounts, pipelineActive, startPollingInterval, clearPollingInterval])

  // Monitor docs changes to check status counts and trigger health check if needed
  useEffect(() => {
    if (!docs) return;

    // Build a key-agnostic digest of every status bucket's count. Sorting keeps
    // it stable regardless of object iteration order or whether buckets are keyed
    // by raw DocStatus or by filter bucket.
    const fingerprint = Object.entries(docs.statuses)
      .map(([key, list]) => `${key}:${list?.length || 0}`)
      .sort()
      .join('|')

    // Trigger health check if any bucket count changed and component is still
    // mounted. Skip when the activity probe is running — the probe already drives
    // /health on its own schedule, and double-firing would burn cache and skew rate.
    if (fingerprint !== prevStatusFingerprint.current && isMountedRef.current && !probeActiveRef.current) {
      useBackendState.getState().check()
    }

    // Always update the snapshot so the first post-probe transition still fires.
    prevStatusFingerprint.current = fingerprint
  }, [docs]);

  // Handle page change - only update state
  const handlePageChange = useCallback((newPage: number) => {
    if (newPage === pagination.page) return;

    // Save the new page for current status filter
    setPageByStatus(prev => ({ ...prev, [statusFilter]: newPage }));
    setPagination(prev => ({ ...prev, page: newPage }));
  }, [pagination.page, statusFilter]);

  // Handle status filter change - only update state
  const handleStatusFilterChange = useCallback((newStatusFilter: StatusFilter) => {
    if (newStatusFilter === statusFilter) return;

    // Save current page for the current status filter
    setPageByStatus(prev => ({ ...prev, [statusFilter]: pagination.page }));

    // Get the saved page for the new status filter
    const newPage = pageByStatus[newStatusFilter];

    // Update status filter and restore the saved page
    setStatusFilter(newStatusFilter);
    setPagination(prev => ({ ...prev, page: newPage }));
  }, [statusFilter, pagination.page, pageByStatus]);

  // Handle documents deleted callback
  const handleDocumentsDeleted = useCallback(async () => {
    setSelectedDocIds([])

    // Reset health check timer with 1 second delay to avoid race condition
    useBackendState.getState().resetHealthCheckTimerDelayed(1000)

    // A completed delete is user intent: refresh unconditionally rather than
    // waiting for a poll tick the breaker may refuse.
    refreshDocumentsThrottled({ auto: false })

    // Schedule a health check 2 seconds after successful clear
    startPollingInterval(2000)
  }, [refreshDocumentsThrottled, startPollingInterval])

  // Handle documents cleared callback with proper interval reset
  const handleDocumentsCleared = useCallback(async () => {
    // Clear current polling interval
    clearPollingInterval();

    // Reset status counts to ensure proper state
    setStatusCounts({
      all: 0,
      processed: 0,
      preprocessed: 0,
      parsing: 0,
      analyzing: 0,
      processing: 0,
      pending: 0,
      failed: 0
    });

    // Perform one immediate refresh to confirm clear operation
    if (isMountedRef.current) {
      try {
        await fetchDocuments();
      } catch (err) {
        console.error('Error fetching documents after clear:', err);
      }
    }

    // Set appropriate polling interval based on current state
    // Since documents are cleared, use idle interval (30 seconds)
    if (currentTab === 'documents' && health && isMountedRef.current) {
      startPollingInterval(30000); // 30 seconds for idle state
    }
  }, [clearPollingInterval, setStatusCounts, fetchDocuments, currentTab, health, startPollingInterval])


  // Handle showFileName change - switch sort field if currently sorting by first column.
  // Render-time comparison avoids cascading renders flagged by react-hooks/set-state-in-effect.
  const [previousShowFileName, setPreviousShowFileName] = useState(showFileName)
  if (showFileName !== previousShowFileName) {
    setPreviousShowFileName(showFileName)
    if (sortField === 'id' || sortField === 'file_path') {
      const newSortField = showFileName ? 'file_path' : 'id';
      if (sortField !== newSortField) {
        setSortField(newSortField);
      }
    }
  }

  // Reset selection state when page, status filter, or sort changes (render-time comparison).
  const [previousSelectionDeps, setPreviousSelectionDeps] = useState({
    page: pagination.page,
    statusFilter,
    sortField,
    sortDirection
  })
  if (
    previousSelectionDeps.page !== pagination.page ||
    previousSelectionDeps.statusFilter !== statusFilter ||
    previousSelectionDeps.sortField !== sortField ||
    previousSelectionDeps.sortDirection !== sortDirection
  ) {
    setPreviousSelectionDeps({
      page: pagination.page,
      statusFilter,
      sortField,
      sortDirection
    })
    setSelectedDocIds([])
  }

  // Central effect to handle all data fetching - genuine side effect (network call)
  useEffect(() => {
    if (currentTab === 'documents') {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      fetchPaginatedDocuments(pagination.page, pagination.page_size, statusFilter);
    }
  }, [
    currentTab,
    pagination.page,
    pagination.page_size,
    statusFilter,
    sortField,
    sortDirection,
    fetchPaginatedDocuments
  ]);

  return (
    <Card className="!rounded-none !overflow-hidden flex flex-col h-full min-h-0">
      <CardHeader className="py-2 px-6">
        <CardTitle className="text-lg">{t('documentPanel.documentManager.title')}</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0 overflow-auto">
        <div className="flex justify-between items-center gap-2 mb-2">
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
                pipelineActive && 'pipeline-busy'
              )}
            >
              <ActivityIcon /> {t('documentPanel.documentManager.pipelineStatusButton')}
            </Button>
          </div>

          {/* Pagination Controls in the middle */}
          {pagination.total_pages > 1 && (
            <PaginationControls
              currentPage={pagination.page}
              totalPages={pagination.total_pages}
              pageSize={pagination.page_size}
              totalCount={pagination.total_count}
              onPageChange={handlePageChange}
              onPageSizeChange={handlePageSizeChange}
              isLoading={isRefreshing}
              compact={true}
            />
          )}

          <div className="flex gap-2">
            {isSelectionMode && (
              <DeleteDocumentsDialog
                selectedDocIds={selectedDocIds}
                onDocumentsDeleted={handleDocumentsDeleted}
              />
            )}
            {isSelectionMode && hasCurrentPageSelection ? (
              (() => {
                const buttonProps = getSelectionButtonProps();
                const IconComponent = buttonProps.icon;
                return (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={buttonProps.action}
                    side="bottom"
                    tooltip={buttonProps.text}
                  >
                    <IconComponent className="h-4 w-4" />
                    {buttonProps.text}
                  </Button>
                );
              })()
            ) : !isSelectionMode ? (
              <ClearDocumentsDialog onDocumentsCleared={handleDocumentsCleared} />
            ) : null}
            <UploadDocumentsDialog
              onUploadBatchAccepted={() => startActivityProbe('upload')}
              onDocumentsUploaded={async () => { refreshDocumentsThrottled({ auto: false }) }}
            />
            <PipelineStatusDialog
              open={showPipelineStatus}
              onOpenChange={setShowPipelineStatus}
            />
          </div>
        </div>

        <Card className="flex-1 flex flex-col border rounded-md min-h-0 mb-2">
          <CardHeader className="flex-none py-2 px-4">
            <div className="flex justify-between items-center">
              <CardTitle>{t('documentPanel.documentManager.uploadedTitle')}</CardTitle>
              <div className="flex items-center gap-2">
                <div className="flex gap-1" dir={i18n.dir()}>
                  <Button
                    size="sm"
                    variant={statusFilter === 'all' ? 'secondary' : 'outline'}
                    onClick={() => handleStatusFilterChange('all')}
                    disabled={isRefreshing}
                    className={cn(
                      statusFilter === 'all' && 'bg-gray-100 dark:bg-gray-900 font-medium border border-gray-400 dark:border-gray-500 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.filters.all')} ({statusCounts.all || documentCounts.all})
                  </Button>
                  <Button
                    size="sm"
                    variant={statusFilter === 'completed' ? 'secondary' : 'outline'}
                    onClick={() => handleStatusFilterChange('completed')}
                    disabled={isRefreshing}
                    className={cn(
                      completedCount > 0 ? 'text-green-600' : 'text-gray-500',
                      statusFilter === 'completed' && 'bg-green-100 dark:bg-green-900/30 font-medium border border-green-400 dark:border-green-600 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.filters.completed')} ({completedCount})
                  </Button>
                  <Button
                    size="sm"
                    variant={statusFilter === 'parse' ? 'secondary' : 'outline'}
                    onClick={() => handleStatusFilterChange('parse')}
                    disabled={isRefreshing}
                    className={cn(
                      parseCount > 0 ? 'text-cyan-600' : 'text-gray-500',
                      statusFilter === 'parse' && 'bg-cyan-100 dark:bg-cyan-900/30 font-medium border border-cyan-400 dark:border-cyan-600 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.filters.parse')} ({parseCount})
                  </Button>
                  <Button
                    size="sm"
                    variant={statusFilter === 'analyze' ? 'secondary' : 'outline'}
                    onClick={() => handleStatusFilterChange('analyze')}
                    disabled={isRefreshing}
                    className={cn(
                      analyzeCount > 0 ? 'text-indigo-600' : 'text-gray-500',
                      statusFilter === 'analyze' && 'bg-indigo-100 dark:bg-indigo-900/30 font-medium border border-indigo-400 dark:border-indigo-600 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.filters.analyze')} ({analyzeCount})
                  </Button>
                  <Button
                    size="sm"
                    variant={statusFilter === 'process' ? 'secondary' : 'outline'}
                    onClick={() => handleStatusFilterChange('process')}
                    disabled={isRefreshing}
                    className={cn(
                      processCount > 0 ? 'text-blue-600' : 'text-gray-500',
                      statusFilter === 'process' && 'bg-blue-100 dark:bg-blue-900/30 font-medium border border-blue-400 dark:border-blue-600 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.filters.process')} ({processCount})
                  </Button>
                  <Button
                    size="sm"
                    variant={statusFilter === 'failed' ? 'secondary' : 'outline'}
                    onClick={() => handleStatusFilterChange('failed')}
                    disabled={isRefreshing}
                    className={cn(
                      failedCount > 0 ? 'text-red-600' : 'text-gray-500',
                      statusFilter === 'failed' && 'bg-red-100 dark:bg-red-900/30 font-medium border border-red-400 dark:border-red-600 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.filters.failed')} ({failedCount})
                  </Button>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleManualRefresh}
                  disabled={isRefreshing}
                  side="bottom"
                  tooltip={t('documentPanel.documentManager.refreshTooltip')}
                >
                  <RotateCcwIcon className="h-4 w-4" />
                </Button>
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

          <CardContent className="min-h-0 flex-1 relative p-0" ref={cardContentRef}>
            {!docs && (
              <div className="absolute inset-0 min-h-0 p-0">
                <EmptyCard
                  title={t('documentPanel.documentManager.emptyTitle')}
                  description={t('documentPanel.documentManager.emptyDescription')}
                />
              </div>
            )}
            {docs && (
              <div className="absolute inset-0 flex min-h-0 flex-col p-0">
                <div className="absolute inset-[-1px] flex flex-col p-0 border rounded-md border-gray-200 dark:border-gray-700 overflow-hidden">
                  <TooltipProvider>
                    <Table className="w-full">
                      <TableHeader className="sticky top-0 bg-background z-10 shadow-sm">
                        <TableRow className="border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/75 shadow-[inset_0_-1px_0_rgba(0,0,0,0.1)]">
                          <TableHead
                            onClick={() => handleSort('id')}
                            className="cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 select-none"
                          >
                            <div className="flex items-center">
                              {showFileName
                                ? t('documentPanel.documentManager.columns.fileName')
                                : t('documentPanel.documentManager.columns.id')
                              }
                              {((sortField === 'id' && !showFileName) || (sortField === 'file_path' && showFileName)) && (
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
                          <TableHead className="w-16 text-center">
                            {t('documentPanel.documentManager.columns.select')}
                          </TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody className="text-sm overflow-auto">
                        {filteredAndSortedDocs && filteredAndSortedDocs.map((doc) => (
                          <TableRow key={doc.id}>
                            <TableCell className="truncate font-mono overflow-visible max-w-[250px]">
                              {showFileName ? (
                                <>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <div className="truncate">
                                        {getDisplayFileName(doc, 30)}
                                      </div>
                                    </TooltipTrigger>
                                    <TooltipContent side="top" className="max-w-2xl">
                                      {doc.file_path}
                                    </TooltipContent>
                                  </Tooltip>
                                  <div className="text-xs text-gray-500">{doc.id}</div>
                                </>
                              ) : (
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <div className="truncate">
                                      {doc.id}
                                    </div>
                                  </TooltipTrigger>
                                  <TooltipContent side="top" className="max-w-2xl">
                                    {doc.file_path}
                                  </TooltipContent>
                                </Tooltip>
                              )}
                            </TableCell>
                            <TableCell className="max-w-xs min-w-45 truncate overflow-visible">
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <div className="truncate">
                                    {doc.content_summary}
                                  </div>
                                </TooltipTrigger>
                                <TooltipContent side="top" className="max-w-2xl">
                                  {doc.content_summary}
                                </TooltipContent>
                              </Tooltip>
                            </TableCell>
                            <TableCell>
                              <div className="flex items-center">
                                {(() => {
                                  const statusDisplay = getStatusDisplay(doc.status)
                                  return (
                                    <span className={statusDisplay.className}>
                                      {t(statusDisplay.labelKey)}
                                    </span>
                                  )
                                })()}

                                {hasDocumentDetails(doc) && <DocumentStatusDetailsDialog doc={doc} />}
                              </div>
                            </TableCell>
                            <TableCell>{doc.content_length ?? '-'}</TableCell>
                            <TableCell>{doc.chunks_count ?? '-'}</TableCell>
                            <TableCell className="truncate">
                              {new Date(doc.created_at).toLocaleString()}
                            </TableCell>
                            <TableCell className="truncate">
                              {new Date(doc.updated_at).toLocaleString()}
                            </TableCell>
                            <TableCell className="text-center">
                              <Checkbox
                                checked={selectedDocIds.includes(doc.id)}
                                onCheckedChange={(checked) => handleDocumentSelect(doc.id, checked === true)}
                                // disabled={doc.status !== 'processed'}
                                className="mx-auto"
                              />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TooltipProvider>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </CardContent>
    </Card>
  )
}
