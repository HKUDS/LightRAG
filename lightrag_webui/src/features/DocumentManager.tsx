import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'
import { useTenantState } from '@/stores/tenant'
import { useRouteState } from '@/hooks/useRouteState'
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
import Checkbox from '@/components/ui/Checkbox'
import UploadDocumentsDialog from '@/components/documents/UploadDocumentsDialog'
import ClearDocumentsDialog from '@/components/documents/ClearDocumentsDialog'
import DeleteDocumentsDialog from '@/components/documents/DeleteDocumentsDialog'
import PaginationControls from '@/components/ui/PaginationControls'

import {
  scanNewDocuments,
  reprocessFailedDocuments,
  getDocumentsPaginated,
  DocsStatusesResponse,
  DocStatus,
  DocStatusResponse,
  DocumentsRequest,
  PaginationInfo
} from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { toast } from 'sonner'
import { useBackendState } from '@/stores/state'

import { RefreshCwIcon, ActivityIcon, ArrowUpIcon, ArrowDownIcon, RotateCcwIcon, CheckSquareIcon, XIcon, AlertTriangle, Info, AlertCircle, Loader2, FilesIcon } from 'lucide-react'
import PipelineStatusDialog from '@/components/documents/PipelineStatusDialog'

type StatusFilter = DocStatus | 'all';

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

const hasActiveDocumentsStatus = (counts: Record<string, number>): boolean =>
  getCountValue(counts, 'PROCESSING', 'processing') > 0 ||
  getCountValue(counts, 'PENDING', 'pending') > 0 ||
  getCountValue(counts, 'PREPROCESSED', 'preprocessed', 'multimodal_processed') > 0

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

  if (formattedMetadata.processing_start_time && typeof formattedMetadata.processing_start_time === 'number') {
    const date = new Date(formattedMetadata.processing_start_time * 1000);
    if (!isNaN(date.getTime())) {
      formattedMetadata.processing_start_time = date.toLocaleString();
    }
  }

  if (formattedMetadata.processing_end_time && typeof formattedMetadata.processing_end_time === 'number') {
    const date = new Date(formattedMetadata.processing_end_time * 1000);
    if (!isNaN(date.getTime())) {
      formattedMetadata.processing_end_time = date.toLocaleString();
    }
  }

  // Format JSON and remove outer braces and indentation
  const jsonStr = JSON.stringify(formattedMetadata, null, 2);
  const lines = jsonStr.split('\n');
  // Remove first line ({) and last line (}), and remove leading indentation (2 spaces)
  return lines.slice(1, -1)
    .map(line => line.replace(/^ {2}/, ''))
    .join('\n');
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
  word-break: break-word;
  overflow-wrap: break-word;
  border-radius: 0.375rem;
  padding: 0.5rem 0.75rem;
  font-size: 0.75rem; /* 12px */
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

.tooltip pre {
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: break-word;
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
type SortField = 'created_at' | 'updated_at' | 'id' | 'file_path';
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

  // Legacy state for backward compatibility
  const [docs, setDocs] = useState<DocsStatusesResponse | null>(null)

  // Tenant context
  const selectedTenant = useTenantState.use.selectedTenant()
  const selectedKB = useTenantState.use.selectedKB()

  // Route state for URL synchronization (tenant-agnostic URLs)
  const routeState = useRouteState('documents')

  const currentTab = useSettingsStore.use.currentTab()
  const showFileName = useSettingsStore.use.showFileName()
  const setShowFileName = useSettingsStore.use.setShowFileName()
  const documentsPageSize = useSettingsStore.use.documentsPageSize()
  const setDocumentsPageSize = useSettingsStore.use.setDocumentsPageSize()

  // New pagination state - initialize from route state if available
  const [currentPageDocs, setCurrentPageDocs] = useState<DocStatusResponse[]>([])
  const [pagination, setPagination] = useState<PaginationInfo>(() => ({
    page: routeState.page,
    page_size: routeState.pageSize || documentsPageSize,
    total_count: 0,
    total_pages: 0,
    has_next: false,
    has_prev: false
  }))
  const [statusCounts, setStatusCounts] = useState<Record<string, number>>({ all: 0 })
  const [isRefreshing, setIsRefreshing] = useState(false)
  // Track if we've completed at least one successful load for the current tenant/KB
  const [hasLoadedOnce, setHasLoadedOnce] = useState(false)

  // Sort state - initialize from route state if available
  const [sortField, setSortField] = useState<SortField>(() => 
    (routeState.sort as SortField) || 'updated_at'
  )
  const [sortDirection, setSortDirection] = useState<SortDirection>(() => 
    routeState.sortDirection || 'desc'
  )

  // State for document status filter - initialize from route state if available
  const [statusFilter, setStatusFilter] = useState<StatusFilter>(() => {
    const urlFilter = routeState.filters?.status as StatusFilter
    return urlFilter && ['all', 'processed', 'processing', 'pending', 'failed'].includes(urlFilter)
      ? urlFilter
      : 'all'
  });

  // State to store page number for each status filter
  const [pageByStatus, setPageByStatus] = useState<Record<StatusFilter, number>>({
    all: routeState.page,
    processed: 1,
    multimodal_processed: 1,
    processing: 1,
    pending: 1,
    failed: 1,
  });
  
  // Sync state changes to URL (tenant-agnostic)
  useEffect(() => {
    if (!selectedTenant) return
    
    routeState.setState({
      page: pagination.page,
      pageSize: pagination.page_size,
      sort: sortField,
      sortDirection: sortDirection,
      filters: statusFilter !== 'all' ? { status: statusFilter } : {},
      currentKB: selectedKB?.kb_id,
    })
  }, [pagination.page, pagination.page_size, sortField, sortDirection, statusFilter, selectedKB?.kb_id])

  // State for document selection
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])
  const isSelectionMode = selectedDocIds.length > 0

  // Add refs to track previous pipelineBusy state and current interval
  const prevPipelineBusyRef = useRef<boolean | undefined>(undefined);
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Add retry mechanism state
  const [retryState, setRetryState] = useState({
    count: 0,
    lastError: null as Error | null,
    isBackingOff: false
  });

  // Add circuit breaker state
  const [circuitBreakerState, setCircuitBreakerState] = useState({
    isOpen: false,
    failureCount: 0,
    lastFailureTime: null as number | null,
    nextRetryTime: null as number | null
  });


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
      processed: 1,
      'multimodal_processed': 1,
      processing: 1,
      pending: 1,
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
      counts[status as DocStatus] = documents.length;
      counts.all += documents.length;
    });

    return counts;
  }, [docs]);

  const processedCount = getCountValue(statusCounts, 'PROCESSED', 'processed') || documentCounts.processed || 0;
  const preprocessedCount =
    getCountValue(statusCounts, 'PREPROCESSED', 'preprocessed', 'multimodal_processed') ||
    documentCounts.multimodal_processed ||
    0;
  const processingCount = getCountValue(statusCounts, 'PROCESSING', 'processing') || documentCounts.processing || 0;
  const pendingCount = getCountValue(statusCounts, 'PENDING', 'pending') || documentCounts.pending || 0;
  const failedCount = getCountValue(statusCounts, 'FAILED', 'failed') || documentCounts.failed || 0;

  // Store previous status counts
  const prevStatusCounts = useRef({
    processed: 0,
    multimodal_processed: 0,
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

  // Utility function to update component state
  const updateComponentState = useCallback((response: any) => {
    setPagination(response.pagination);
    setCurrentPageDocs(response.documents);
    setStatusCounts(response.status_counts);

    // Update legacy docs state for backward compatibility
    const legacyDocs: DocsStatusesResponse = {
      statuses: {
        processed: response.documents.filter((doc: DocStatusResponse) => doc.status === 'processed'),
        multimodal_processed: response.documents.filter((doc: DocStatusResponse) => doc.status === 'multimodal_processed'),
        processing: response.documents.filter((doc: DocStatusResponse) => doc.status === 'processing'),
        pending: response.documents.filter((doc: DocStatusResponse) => doc.status === 'pending'),
        failed: response.documents.filter((doc: DocStatusResponse) => doc.status === 'failed')
      }
    };

    setDocs(response.pagination.total_count > 0 ? legacyDocs : null);
    // Mark that we've successfully loaded at least once
    setHasLoadedOnce(true);
  }, []);

  // Utility function to create timeout wrapper for API calls
  const withTimeout = useCallback((
    promise: Promise<any>,
    timeoutMs: number = 30000,
    errorMsg: string = 'Request timeout'
  ): Promise<any> => {
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error(errorMsg)), timeoutMs)
    });
    return Promise.race([promise, timeoutPromise]);
  }, []);


  // Enhanced error classification
  const classifyError = useCallback((error: any) => {
    // Handle axios Cancel errors (from context guards) - don't show toast
    if (error.__CANCEL__ || error.name === 'CanceledError' || 
        error.message?.includes('Please select a tenant')) {
      return { type: 'context-missing', shouldRetry: false, shouldShowToast: false };
    }

    if (error.name === 'AbortError') {
      return { type: 'cancelled', shouldRetry: false, shouldShowToast: false };
    }

    if (error.message === 'Request timeout') {
      return { type: 'timeout', shouldRetry: true, shouldShowToast: true };
    }

    if (error.message?.includes('Network Error') || error.code === 'NETWORK_ERROR') {
      return { type: 'network', shouldRetry: true, shouldShowToast: true };
    }

    if (error.status >= 500) {
      return { type: 'server', shouldRetry: true, shouldShowToast: true };
    }

    if (error.status >= 400 && error.status < 500) {
      return { type: 'client', shouldRetry: false, shouldShowToast: true };
    }

    return { type: 'unknown', shouldRetry: true, shouldShowToast: true };
  }, []);

  // Circuit breaker utility functions
  const isCircuitBreakerOpen = useCallback(() => {
    if (!circuitBreakerState.isOpen) return false;

    const now = Date.now();
    if (circuitBreakerState.nextRetryTime && now >= circuitBreakerState.nextRetryTime) {
      // Reset circuit breaker to half-open state
      setCircuitBreakerState(prev => ({
        ...prev,
        isOpen: false,
        failureCount: Math.max(0, prev.failureCount - 1)
      }));
      return false;
    }

    return true;
  }, [circuitBreakerState]);

  const recordFailure = useCallback((error: Error) => {
    const now = Date.now();
    setCircuitBreakerState(prev => {
      const newFailureCount = prev.failureCount + 1;
      const shouldOpen = newFailureCount >= 3; // Open after 3 failures

      return {
        isOpen: shouldOpen,
        failureCount: newFailureCount,
        lastFailureTime: now,
        nextRetryTime: shouldOpen ? now + (Math.pow(2, newFailureCount) * 1000) : null
      };
    });

    setRetryState(prev => ({
      count: prev.count + 1,
      lastError: error,
      isBackingOff: true
    }));
  }, []);

  const recordSuccess = useCallback(() => {
    setCircuitBreakerState({
      isOpen: false,
      failureCount: 0,
      lastFailureTime: null,
      nextRetryTime: null
    });

    setRetryState({
      count: 0,
      lastError: null,
      isBackingOff: false
    });
  }, []);

  // Helper to check if tenant context is ready for API calls
  // Uses localStorage as source of truth since axios interceptor reads from there
  const isTenantContextReady = useCallback(() => {
    try {
      const storedTenant = localStorage.getItem('SELECTED_TENANT');
      const storedKB = localStorage.getItem('SELECTED_KB');
      
      if (!storedTenant || !storedKB) {
        console.log('[DocumentManager] localStorage tenant context not ready');
        return false;
      }
      
      const parsedTenant = JSON.parse(storedTenant);
      const parsedKB = JSON.parse(storedKB);
      
      if (!parsedTenant?.tenant_id || !parsedKB?.kb_id) {
        console.log('[DocumentManager] localStorage tenant/KB missing required fields');
        return false;
      }
      
      return true;
    } catch (e) {
      console.error('[DocumentManager] Error checking localStorage tenant context', e);
      return false;
    }
  }, []);

  // Intelligent refresh function: handles all boundary cases
  const handleIntelligentRefresh = useCallback(async (
    targetPage?: number, // Optional target page, defaults to current page
    resetToFirst?: boolean // Whether to force reset to first page
  ) => {
    try {
      if (!isMountedRef.current) return;
      
      // Guard: Check tenant context before making API calls
      if (!isTenantContextReady()) {
        console.log('[DocumentManager] Skipping refresh - tenant context not ready');
        return;
      }

      setIsRefreshing(true);

      // Determine target page
      const pageToFetch = resetToFirst ? 1 : (targetPage || pagination.page);

      const request: DocumentsRequest = {
        status_filter: statusFilter === 'all' ? null : statusFilter,
        page: pageToFetch,
        page_size: pagination.page_size,
        sort_field: sortField,
        sort_direction: sortDirection
      };

      // Use timeout wrapper for the API call
      const response = await withTimeout(
        getDocumentsPaginated(request),
        30000, // 30 second timeout
        'Document fetch timeout'
      );

      if (!isMountedRef.current) return;

      // Boundary case handling: if target page has no data but total count > 0
      if (response.documents.length === 0 && response.pagination.total_count > 0) {
        // Calculate last page
        const lastPage = Math.max(1, response.pagination.total_pages);

        if (pageToFetch !== lastPage) {
          // Re-request last page
          const lastPageRequest: DocumentsRequest = {
            ...request,
            page: lastPage
          };

          const lastPageResponse = await withTimeout(
            getDocumentsPaginated(lastPageRequest),
            30000,
            'Document fetch timeout'
          );

          if (!isMountedRef.current) return;

          // Update page state to last page
          setPageByStatus(prev => ({ ...prev, [statusFilter]: lastPage }));
          updateComponentState(lastPageResponse);
          return;
        }
      }

      // Normal case: update state
      if (pageToFetch !== pagination.page) {
        setPageByStatus(prev => ({ ...prev, [statusFilter]: pageToFetch }));
      }
      updateComponentState(response);

    } catch (err) {
      if (isMountedRef.current) {
        const errorClassification = classifyError(err);

        if (errorClassification.shouldShowToast) {
          toast.error(t('documentPanel.documentManager.errors.loadFailed', { error: errorMessage(err) }));
        }

        if (errorClassification.shouldRetry) {
          recordFailure(err as Error);
        }
        
        // Mark as loaded even on error to stop infinite loading spinner
        // This allows user to see the "No Documents" state and retry
        if (errorClassification.type !== 'context-missing') {
          setHasLoadedOnce(true);
        }
      }
    } finally {
      if (isMountedRef.current) {
        setIsRefreshing(false);
      }
    }
  }, [statusFilter, pagination.page, pagination.page_size, sortField, sortDirection, t, updateComponentState, withTimeout, classifyError, recordFailure]);

  // New paginated data fetching function
  const fetchPaginatedDocuments = useCallback(async (
    page: number,
    pageSize: number,
    _statusFilter: StatusFilter // eslint-disable-line @typescript-eslint/no-unused-vars
  ) => {
    // Update pagination state
    setPagination(prev => ({ ...prev, page, page_size: pageSize }));

    // Use intelligent refresh
    await handleIntelligentRefresh(page);
  }, [handleIntelligentRefresh]);

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

    pollingIntervalRef.current = setInterval(async () => {
      try {
        // Check circuit breaker before making request
        if (isCircuitBreakerOpen()) {
          return; // Skip this polling cycle
        }

        // Only perform fetch if component is still mounted
        if (isMountedRef.current) {
          await fetchDocuments();
          recordSuccess(); // Record successful operation
        }
      } catch (err) {
        // Only handle error if component is still mounted
        if (isMountedRef.current) {
          const errorClassification = classifyError(err);

          // Always reset isRefreshing state on error
          setIsRefreshing(false);

          if (errorClassification.shouldShowToast) {
            toast.error(t('documentPanel.documentManager.errors.scanProgressFailed', { error: errorMessage(err) }));
          }

          if (errorClassification.shouldRetry) {
            recordFailure(err as Error);

            // Implement exponential backoff for retries
            const backoffDelay = Math.min(Math.pow(2, retryState.count) * 1000, 30000); // Max 30s

            if (retryState.count < 3) { // Max 3 retries
              setTimeout(() => {
                if (isMountedRef.current) {
                  setRetryState(prev => ({ ...prev, isBackingOff: false }));
                }
              }, backoffDelay);
            }
          } else {
            // For non-retryable errors, stop polling
            clearPollingInterval();
          }
        }
      }
    }, intervalMs);
  }, [fetchDocuments, t, clearPollingInterval, isCircuitBreakerOpen, recordSuccess, recordFailure, classifyError, retryState.count]);

  const scanDocuments = useCallback(async () => {
    try {
      // Check if component is still mounted before starting the request
      if (!isMountedRef.current) return;

      const { status, message, track_id: _track_id } = await scanNewDocuments(); // eslint-disable-line @typescript-eslint/no-unused-vars

      // Check again if component is still mounted after the request completes
      if (!isMountedRef.current) return;

      // Note: _track_id is available for future use (e.g., progress tracking)
      toast.message(message || status);

      // Reset health check timer with 1 second delay to avoid race condition
      useBackendState.getState().resetHealthCheckTimerDelayed(1000);

      // Start fast refresh with 2-second interval immediately after scan
      startPollingInterval(2000);

      // Set recovery timer to restore normal polling interval after 15 seconds
      setTimeout(() => {
        if (isMountedRef.current && currentTab === 'documents' && health) {
          // Restore intelligent polling interval based on document status
          const hasActiveDocuments = hasActiveDocumentsStatus(statusCounts);
          const normalInterval = hasActiveDocuments ? 5000 : 30000;
          startPollingInterval(normalInterval);
        }
      }, 15000); // Restore after 15 seconds
    } catch (err) {
      // Only show error if component is still mounted
      if (isMountedRef.current) {
        toast.error(t('documentPanel.documentManager.errors.scanFailed', { error: errorMessage(err) }));
      }
    }
  }, [t, startPollingInterval, currentTab, health, statusCounts])

  const retryFailedDocuments = useCallback(async () => {
    try {
      // Check if component is still mounted before starting the request
      if (!isMountedRef.current) return;

      const { status, message, track_id: _track_id } = await reprocessFailedDocuments(); // eslint-disable-line @typescript-eslint/no-unused-vars

      // Check again if component is still mounted after the request completes
      if (!isMountedRef.current) return;

      // Note: _track_id is available for future use (e.g., progress tracking)
      toast.message(message || status);

      // Reset health check timer with 1 second delay to avoid race condition
      useBackendState.getState().resetHealthCheckTimerDelayed(1000);

      // Start fast refresh with 2-second interval immediately after retry
      startPollingInterval(2000);

      // Set recovery timer to restore normal polling interval after 15 seconds
      setTimeout(() => {
        if (isMountedRef.current && currentTab === 'documents' && health) {
          // Restore intelligent polling interval based on document status
          const hasActiveDocuments = hasActiveDocumentsStatus(statusCounts);
          const normalInterval = hasActiveDocuments ? 5000 : 30000;
          startPollingInterval(normalInterval);
        }
      }, 15000); // Restore after 15 seconds
    } catch (err) {
      // Only show error if component is still mounted
      if (isMountedRef.current) {
        toast.error(errorMessage(err));
      }
    }
  }, [startPollingInterval, currentTab, health, statusCounts])

  // Handle page size change - update state and save to store
  const handlePageSizeChange = useCallback((newPageSize: number) => {
    if (newPageSize === pagination.page_size) return;

    // Save the new page size to the store
    setDocumentsPageSize(newPageSize);

    // Reset all status filters to page 1 when page size changes
    setPageByStatus({
      all: 1,
      processed: 1,
      multimodal_processed: 1,
      processing: 1,
      pending: 1,
      failed: 1,
    });

    setPagination(prev => ({ ...prev, page: 1, page_size: newPageSize }));
  }, [pagination.page_size, setDocumentsPageSize]);

  // Handle manual refresh with pagination reset logic
  const handleManualRefresh = useCallback(async () => {
    // Guard: Check tenant context before making API calls
    if (!isTenantContextReady()) {
      console.log('[DocumentManager] Skipping manual refresh - tenant context not ready');
      return;
    }
    
    try {
      setIsRefreshing(true);

      // Fetch documents from the first page
      const request: DocumentsRequest = {
        status_filter: statusFilter === 'all' ? null : statusFilter,
        page: 1,
        page_size: pagination.page_size,
        sort_field: sortField,
        sort_direction: sortDirection
      };

      const response = await getDocumentsPaginated(request);

      if (!isMountedRef.current) return;

      // Check if total count is less than current page size and page size is not already 10
      if (response.pagination.total_count < pagination.page_size && pagination.page_size !== 10) {
        // Reset page size to 10 which will trigger a new fetch
        handlePageSizeChange(10);
      } else {
        // Update pagination state
        setPagination(response.pagination);
        setCurrentPageDocs(response.documents);
        setStatusCounts(response.status_counts);

        // Update legacy docs state for backward compatibility
        const legacyDocs: DocsStatusesResponse = {
          statuses: {
            processed: response.documents.filter(doc => doc.status === 'processed'),
            multimodal_processed: response.documents.filter(doc => doc.status === 'multimodal_processed'),
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

    } catch (err) {
      if (isMountedRef.current) {
        const errorClassification = classifyError(err);
        if (errorClassification.shouldShowToast) {
          toast.error(t('documentPanel.documentManager.errors.loadFailed', { error: errorMessage(err) }));
        }
      }
    } finally {
      if (isMountedRef.current) {
        setIsRefreshing(false);
      }
    }
  }, [statusFilter, pagination.page_size, sortField, sortDirection, handlePageSizeChange, t, isTenantContextReady, classifyError]);

  // Monitor pipelineBusy changes and trigger immediate refresh with timer reset
  useEffect(() => {
    // Skip the first render when prevPipelineBusyRef is undefined
    if (prevPipelineBusyRef.current !== undefined && prevPipelineBusyRef.current !== pipelineBusy) {
      // pipelineBusy state has changed, trigger immediate refresh
      if (currentTab === 'documents' && health && isMountedRef.current) {
        // Use intelligent refresh to preserve current page
        handleIntelligentRefresh();

        // Reset polling timer after intelligent refresh
        const hasActiveDocuments = hasActiveDocumentsStatus(statusCounts);
        const pollingInterval = hasActiveDocuments ? 5000 : 30000;
        startPollingInterval(pollingInterval);
      }
    }
    // Update the previous state
    prevPipelineBusyRef.current = pipelineBusy;
  }, [
    pipelineBusy,
    currentTab,
    health,
    handleIntelligentRefresh,
    statusCounts,
    startPollingInterval
  ]);

  // Set up intelligent polling with dynamic interval based on document status
  useEffect(() => {
    if (currentTab !== 'documents' || !health) {
      clearPollingInterval();
      return
    }

    // Determine polling interval based on document status
    const hasActiveDocuments = hasActiveDocumentsStatus(statusCounts);
    const pollingInterval = hasActiveDocuments ? 5000 : 30000; // 5s if active, 30s if idle

    startPollingInterval(pollingInterval);

    return () => {
      clearPollingInterval();
    }
  }, [health, t, currentTab, statusCounts, startPollingInterval, clearPollingInterval])

  // Monitor docs changes to check status counts and trigger health check if needed
  useEffect(() => {
    if (!docs) return;

    // Get new status counts
    const newStatusCounts = {
      processed: docs?.statuses?.processed?.length || 0,
      multimodal_processed: docs?.statuses?.multimodal_processed?.length || 0,
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

    // Schedule a health check 2 seconds after successful clear
    startPollingInterval(2000)
  }, [startPollingInterval])

  // Handle documents cleared callback with proper interval reset
  const handleDocumentsCleared = useCallback(async () => {
    // Clear current polling interval
    clearPollingInterval();

    // Reset status counts to ensure proper state
    setStatusCounts({
      all: 0,
      processed: 0,
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


  // Handle showFileName change - switch sort field if currently sorting by first column
  useEffect(() => {
    // Only switch if currently sorting by the first column (id or file_path)
    if (sortField === 'id' || sortField === 'file_path') {
      const newSortField = showFileName ? 'file_path' : 'id';
      if (sortField !== newSortField) {
        setSortField(newSortField);
      }
    }
  }, [showFileName, sortField]);

  // Reset selection state when page, status filter, or sort changes
  useEffect(() => {
    setSelectedDocIds([])
  }, [pagination.page, statusFilter, sortField, sortDirection]);

  // Reset document state when tenant or KB changes - clear old data immediately
  useEffect(() => {
    // Clear current documents to prevent showing stale data
    setCurrentPageDocs([]);
    setDocs(null);
    setStatusCounts({ all: 0 });
    setPagination(prev => ({
      ...prev,
      page: 1,
      total_count: 0,
      total_pages: 0,
      has_next: false,
      has_prev: false
    }));
    setSelectedDocIds([]);
    // Reset page memory for all status filters
    setPageByStatus({ all: 1, processed: 1, processing: 1, pending: 1, failed: 1 });
    // Reset load tracking - we haven't loaded docs for this new context yet
    setHasLoadedOnce(false);
    console.log('[DocumentManager] Reset document state due to tenant/KB change:', {
      tenant_id: selectedTenant?.tenant_id,
      kb_id: selectedKB?.kb_id
    });
  }, [selectedTenant?.tenant_id, selectedKB?.kb_id]);

  // Central effect to handle all data fetching - with debounce to avoid race conditions
  useEffect(() => {
    // Guard: Skip if not on documents tab
    if (currentTab !== 'documents') {
      console.log('[DocumentManager] Skipping fetch - not on documents tab');
      return;
    }
    
    // Guard: Must have tenant selected
    if (!selectedTenant?.tenant_id) {
      console.log('[DocumentManager] Skipping fetch - no tenant');
      return;
    }
    
    // Verify localStorage is in sync before making API calls
    // We need to check synchronously and retry a few times if needed
    let attempts = 0;
    const maxAttempts = 10;
    const checkIntervalMs = 50;
    
    const checkAndFetch = () => {
      attempts++;
      
      // Check localStorage directly - this is the source of truth
      try {
        const storedTenant = localStorage.getItem('SELECTED_TENANT');
        const storedKB = localStorage.getItem('SELECTED_KB');
        
        if (storedTenant && storedKB) {
          const parsedTenant = JSON.parse(storedTenant);
          const parsedKB = JSON.parse(storedKB);
          
          // Verify tenant matches what we expect
          if (parsedTenant?.tenant_id === selectedTenant.tenant_id && parsedKB?.kb_id) {
            // Context is ready, proceed with fetch
            console.log('[DocumentManager] Context ready, fetching documents for KB:', parsedKB.kb_id);
            fetchPaginatedDocuments(pagination.page, pagination.page_size, statusFilter);
            return;
          }
        }
      } catch (e) {
        console.error('[DocumentManager] Error checking localStorage', e);
      }
      
      // If not ready yet and we haven't exceeded max attempts, try again
      if (attempts < maxAttempts) {
        console.log(`[DocumentManager] Context not ready yet, retry ${attempts}/${maxAttempts}`);
        setTimeout(checkAndFetch, checkIntervalMs);
      } else {
        console.log('[DocumentManager] Max attempts reached, context still not ready');
        // Set hasLoadedOnce to show empty state instead of infinite spinner
        setHasLoadedOnce(true);
      }
    };
    
    // Start checking after a small initial delay
    const timeoutId = setTimeout(checkAndFetch, 30);
    
    return () => clearTimeout(timeoutId);
  }, [
    currentTab,
    pagination.page,
    pagination.page_size,
    statusFilter,
    sortField,
    sortDirection,
    fetchPaginatedDocuments,
    selectedTenant?.tenant_id, // Trigger fetch when tenant changes
    selectedKB?.kb_id // Trigger fetch when KB changes
  ]);

  // Guard: Check if tenant and KB are selected
  // Also check localStorage as fallback since Zustand state may lag behind
  const hasTenantContext = useMemo(() => {
    if (selectedTenant && selectedKB) return true;
    
    // Check localStorage as fallback
    try {
      const storedTenant = localStorage.getItem('SELECTED_TENANT');
      const storedKB = localStorage.getItem('SELECTED_KB');
      return !!(storedTenant && storedKB);
    } catch {
      return false;
    }
  }, [selectedTenant, selectedKB]);
  
  if (!hasTenantContext) {
    return (
      <Card className="!rounded-none !overflow-hidden flex flex-col h-full min-h-0">
        <CardHeader className="py-2 px-6">
          <CardTitle className="text-lg">{t('documentPanel.documentManager.title')}</CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col min-h-0 items-center justify-center">
          <div className="text-center">
            <AlertCircle className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">{t('documentPanel.selectTenant')}</h3>
            <p className="text-sm text-gray-500">
              {t('documentPanel.selectTenantDescription')}
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }

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
              onClick={retryFailedDocuments}
              side="bottom"
              tooltip={t('documentPanel.documentManager.retryFailedTooltip')}
              size="sm"
              disabled={pipelineBusy}
            >
              <RotateCcwIcon /> {t('documentPanel.documentManager.retryFailedButton')}
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
            <UploadDocumentsDialog onDocumentsUploaded={fetchDocuments} />
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
                    {t('documentPanel.documentManager.status.all')} ({statusCounts.all || documentCounts.all})
                  </Button>
                  <Button
                    size="sm"
                    variant={statusFilter === 'processed' ? 'secondary' : 'outline'}
                    onClick={() => handleStatusFilterChange('processed')}
                    disabled={isRefreshing}
                    className={cn(
                      processedCount > 0 ? 'text-green-600' : 'text-gray-500',
                      statusFilter === 'processed' && 'bg-green-100 dark:bg-green-900/30 font-medium border border-green-400 dark:border-green-600 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.status.completed')} ({processedCount})
                  </Button>
                  <Button
                    size="sm"
                    variant={statusFilter === 'multimodal_processed' ? 'secondary' : 'outline'}
                    onClick={() => handleStatusFilterChange('multimodal_processed')}
                    disabled={isRefreshing}
                    className={cn(
                      preprocessedCount > 0 ? 'text-purple-600' : 'text-gray-500',
                      statusFilter === 'multimodal_processed' && 'bg-purple-100 dark:bg-purple-900/30 font-medium border border-purple-400 dark:border-purple-600 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.status.preprocessed')} ({preprocessedCount})
                  </Button>
                  <Button
                    size="sm"
                    variant={statusFilter === 'processing' ? 'secondary' : 'outline'}
                    onClick={() => handleStatusFilterChange('processing')}
                    disabled={isRefreshing}
                    className={cn(
                      processingCount > 0 ? 'text-blue-600' : 'text-gray-500',
                      statusFilter === 'processing' && 'bg-blue-100 dark:bg-blue-900/30 font-medium border border-blue-400 dark:border-blue-600 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.status.processing')} ({processingCount})
                  </Button>
                  <Button
                    size="sm"
                    variant={statusFilter === 'pending' ? 'secondary' : 'outline'}
                    onClick={() => handleStatusFilterChange('pending')}
                    disabled={isRefreshing}
                    className={cn(
                      pendingCount > 0 ? 'text-yellow-600' : 'text-gray-500',
                      statusFilter === 'pending' && 'bg-yellow-100 dark:bg-yellow-900/30 font-medium border border-yellow-400 dark:border-yellow-600 shadow-sm'
                    )}
                  >
                    {t('documentPanel.documentManager.status.pending')} ({pendingCount})
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
                    {t('documentPanel.documentManager.status.failed')} ({failedCount})
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

          <CardContent className="flex-1 relative p-0" ref={cardContentRef}>
            {!docs && !hasLoadedOnce && (
              <div className="absolute inset-0 p-0 flex items-center justify-center">
                <div className="text-center">
                  <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto mb-3" />
                  <p className="text-sm font-medium text-foreground">{t('documentPanel.documentManager.loading')}</p>
                  <p className="text-xs text-muted-foreground mt-1">{t('documentPanel.documentManager.loadingHint')}</p>
                </div>
              </div>
            )}
            {!docs && hasLoadedOnce && pipelineBusy && (
              <div className="absolute inset-0 p-0 flex items-center justify-center">
                <div className="text-center max-w-md px-6">
                  <div className="mx-auto w-16 h-16 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center mb-4">
                    <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">{t('documentPanel.documentManager.emptyWithPipelineTitle')}</h3>
                  <p className="text-sm text-muted-foreground mb-4">{t('documentPanel.documentManager.emptyWithPipelineDescription')}</p>
                  <div className="flex gap-2 justify-center">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={scanDocuments}
                      disabled={isRefreshing}
                    >
                      {t('documentPanel.documentManager.scanForDocuments')}
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => setShowPipelineStatus(true)}
                    >
                      {t('documentPanel.documentManager.viewPipeline')}
                    </Button>
                  </div>
                </div>
              </div>
            )}
            {!docs && hasLoadedOnce && !pipelineBusy && (
              <div className="absolute inset-0 p-0 flex items-center justify-center">
                <div className="text-center max-w-md px-6">
                  <div className="mx-auto w-16 h-16 rounded-full bg-muted/50 flex items-center justify-center mb-4">
                    <FilesIcon className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">{t('documentPanel.documentManager.emptyTitle')}</h3>
                  <p className="text-sm text-muted-foreground mb-4">{t('documentPanel.documentManager.emptyDescription')}</p>
                  <div className="flex flex-col gap-2 items-center">
                    <p className="text-xs text-muted-foreground">{t('documentPanel.documentManager.emptyHint')}</p>
                  </div>
                </div>
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
                            <div className="group relative flex items-center overflow-visible tooltip-container">
                              {doc.status === 'processed' && (
                                <span className="text-green-600">{t('documentPanel.documentManager.status.completed')}</span>
                              )}
                              {doc.status === 'multimodal_processed' && (
                                <span className="text-purple-600">{t('documentPanel.documentManager.status.preprocessed')}</span>
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

                              {/* Icon rendering logic */}
                              {doc.error_msg ? (
                                <AlertTriangle className="ml-2 h-4 w-4 text-yellow-500" />
                              ) : (doc.metadata && Object.keys(doc.metadata).length > 0) && (
                                <Info className="ml-2 h-4 w-4 text-blue-500" />
                              )}

                              {/* Tooltip rendering logic */}
                              {(doc.error_msg || (doc.metadata && Object.keys(doc.metadata).length > 0) || doc.track_id) && (
                                <div className="invisible group-hover:visible tooltip">
                                  {doc.track_id && (
                                    <div className="mt-1">Track ID: {doc.track_id}</div>
                                  )}
                                  {doc.metadata && Object.keys(doc.metadata).length > 0 && (
                                    <pre>{formatMetadata(doc.metadata)}</pre>
                                  )}
                                  {doc.error_msg && (
                                    <pre>{doc.error_msg}</pre>
                                  )}
                                </div>
                              )}
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
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </CardContent>
    </Card>
  )
}
