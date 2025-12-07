import {
  backendBaseUrl,
  popularLabelsDefaultLimit,
  searchLabelsDefaultLimit,
} from '@/lib/constants'
import { errorMessage } from '@/lib/utils'
import { navigationService } from '@/services/navigation'
import { useSettingsStore } from '@/stores/settings'
import axios, { type AxiosError } from 'axios'

// Types
export type LightragNodeType = {
  id: string
  labels: string[]
  properties: Record<string, any>
}

export type LightragEdgeType = {
  id: string
  source: string
  target: string
  type: string
  properties: Record<string, any>
}

export type LightragGraphType = {
  nodes: LightragNodeType[]
  edges: LightragEdgeType[]
}

export type LightragStatus = {
  status: 'healthy'
  working_directory: string
  input_directory: string
  configuration: {
    llm_binding: string
    llm_binding_host: string
    llm_model: string
    embedding_binding: string
    embedding_binding_host: string
    embedding_model: string
    kv_storage: string
    doc_status_storage: string
    graph_storage: string
    vector_storage: string
    workspace?: string
    max_graph_nodes?: string
    enable_rerank?: boolean
    rerank_binding?: string | null
    rerank_model?: string | null
    rerank_binding_host?: string | null
    summary_language: string
    force_llm_summary_on_merge: boolean
    max_parallel_insert: number
    max_async: number
    embedding_func_max_async: number
    embedding_batch_num: number
    cosine_threshold: number
    min_rerank_score: number
    related_chunk_number: number
    auto_connect_orphans?: boolean
  }
  update_status?: Record<string, any>
  core_version?: string
  api_version?: string
  auth_mode?: 'enabled' | 'disabled'
  pipeline_busy: boolean
  keyed_locks?: {
    process_id: number
    cleanup_performed: {
      mp_cleaned: number
      async_cleaned: number
    }
    current_status: {
      total_mp_locks: number
      pending_mp_cleanup: number
      total_async_locks: number
      pending_async_cleanup: number
    }
  }
  webui_title?: string
  webui_description?: string
}

export type LightragDocumentsScanProgress = {
  is_scanning: boolean
  current_file: string
  indexed_count: number
  total_files: number
  progress: number
}

/**
 * Specifies the retrieval mode:
 * - "naive": Performs a basic search without advanced techniques.
 * - "local": Focuses on context-dependent information.
 * - "global": Utilizes global knowledge.
 * - "hybrid": Combines local and global retrieval methods.
 * - "mix": Integrates knowledge graph and vector retrieval.
 * - "bypass": Bypasses knowledge retrieval and directly uses the LLM.
 */
export type QueryMode = 'naive' | 'local' | 'global' | 'hybrid' | 'mix' | 'bypass'

/**
 * Citation marker with position data for frontend insertion
 */
export type CitationMarker = {
  marker: string              // e.g., "[1]" or "[1,2]"
  insert_position: number     // Character position to insert marker
  reference_ids: string[]     // Reference IDs this marker cites
  confidence: number          // Match confidence (0.0-1.0)
  text_preview: string        // Preview of the cited text
}

/**
 * Enhanced source metadata for hover cards
 */
export type CitationSource = {
  reference_id: string
  file_path: string
  document_title: string | null
  section_title: string | null
  page_range: string | null
  excerpt: string | null
}

/**
 * Consolidated citation metadata from backend
 */
export type CitationsMetadata = {
  markers: CitationMarker[]   // Position-based markers for insertion
  sources: CitationSource[]   // Enhanced reference metadata
  footnotes: string[]         // Pre-formatted footnote strings
  uncited_count: number       // Number of claims without citations
}

export type Message = {
  role: 'user' | 'assistant' | 'system'
  content: string
  thinkingContent?: string
  displayContent?: string
  thinkingTime?: number | null
  citationsProcessed?: boolean
  citationsMetadata?: CitationsMetadata  // New consolidated citation data
}

export type QueryRequest = {
  query: string
  /** Specifies the retrieval mode. */
  mode: QueryMode
  /** If True, only returns the retrieved context without generating a response. */
  only_need_context?: boolean
  /** If True, only returns the generated prompt without producing a response. */
  only_need_prompt?: boolean
  /** Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'. */
  response_type?: string
  /** If True, enables streaming output for real-time responses. */
  stream?: boolean
  /** Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode. */
  top_k?: number
  /** Maximum number of text chunks to retrieve and keep after reranking. */
  chunk_top_k?: number
  /** Maximum number of tokens allocated for entity context in unified token control system. */
  max_entity_tokens?: number
  /** Maximum number of tokens allocated for relationship context in unified token control system. */
  max_relation_tokens?: number
  /** Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt). */
  max_total_tokens?: number
  /**
   * Stores past conversation history to maintain context.
   * Format: [{"role": "user/assistant", "content": "message"}].
   */
  conversation_history?: Message[]
  /** Number of complete conversation turns (user-assistant pairs) to consider in the response context. */
  history_turns?: number
  /** User-provided prompt for the query. If provided, this will be used instead of the default value from prompt template. */
  user_prompt?: string
  /** Enable reranking for retrieved text chunks. If True but no rerank model is configured, a warning will be issued. Default is True. */
  enable_rerank?: boolean
  /** Citation mode for post-processing citations. 'none' = no citations, 'inline' = [n] markers only, 'footnotes' = full footnotes with document titles */
  citation_mode?: 'none' | 'inline' | 'footnotes'
  /** Minimum similarity threshold (0.0-1.0) for matching response sentences to source chunks. Higher = stricter matching. Default is 0.7 */
  citation_threshold?: number
}

export type QueryResponse = {
  response: string
}

export type EntityUpdateResponse = {
  status: string
  message: string
  data: Record<string, any>
  operation_summary?: {
    merged: boolean
    merge_status: 'success' | 'failed' | 'not_attempted'
    merge_error: string | null
    operation_status: 'success' | 'partial_success' | 'failure'
    target_entity: string | null
    final_entity?: string | null
    renamed?: boolean
  }
}

export type DocActionResponse = {
  status: 'success' | 'partial_success' | 'failure' | 'duplicated'
  message: string
  track_id?: string
}

export type ScanResponse = {
  status: 'scanning_started'
  message: string
  track_id: string
}

export type ReprocessFailedResponse = {
  status: 'reprocessing_started'
  message: string
  track_id: string
}

export type DeleteDocResponse = {
  status: 'deletion_started' | 'busy' | 'not_allowed'
  message: string
  doc_id: string
}

export type DocStatus = 'pending' | 'processing' | 'preprocessed' | 'processed' | 'failed'

export type DocStatusResponse = {
  id: string
  content_summary: string
  content_length: number
  status: DocStatus
  created_at: string
  updated_at: string
  track_id?: string
  chunks_count?: number
  error_msg?: string
  metadata?: Record<string, any>
  file_path: string
  s3_key?: string
}

export type DocsStatusesResponse = {
  statuses: Record<DocStatus, DocStatusResponse[]>
}

export type TrackStatusResponse = {
  track_id: string
  documents: DocStatusResponse[]
  total_count: number
  status_summary: Record<string, number>
}

export type DocumentsRequest = {
  status_filter?: DocStatus | null
  page: number
  page_size: number
  sort_field: 'created_at' | 'updated_at' | 'id' | 'file_path'
  sort_direction: 'asc' | 'desc'
}

export type PaginationInfo = {
  page: number
  page_size: number
  total_count: number
  total_pages: number
  has_next: boolean
  has_prev: boolean
}

export type PaginatedDocsResponse = {
  documents: DocStatusResponse[]
  pagination: PaginationInfo
  status_counts: Record<string, number>
}

export type StatusCountsResponse = {
  status_counts: Record<string, number>
}

export type AuthStatusResponse = {
  auth_configured: boolean
  access_token?: string
  token_type?: string
  auth_mode?: 'enabled' | 'disabled'
  message?: string
  core_version?: string
  api_version?: string
  webui_title?: string
  webui_description?: string
}

export type PipelineStatusResponse = {
  autoscanned: boolean
  busy: boolean
  job_name: string
  job_start?: string
  docs: number
  batchs: number
  cur_batch: number
  request_pending: boolean
  cancellation_requested?: boolean
  latest_message: string
  history_messages?: string[]
  update_status?: Record<string, any>
}

export type LoginResponse = {
  access_token: string
  token_type: string
  auth_mode?: 'enabled' | 'disabled' // Authentication mode identifier
  message?: string // Optional message
  core_version?: string
  api_version?: string
  webui_title?: string
  webui_description?: string
}

export const InvalidApiKeyError = 'Invalid API Key'
export const RequireApiKeError = 'API Key required'

// Axios instance
const axiosInstance = axios.create({
  baseURL: backendBaseUrl,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Interceptor: add api key and check authentication
axiosInstance.interceptors.request.use((config) => {
  const apiKey = useSettingsStore.getState().apiKey
  const token = localStorage.getItem('LIGHTRAG-API-TOKEN')

  // Always include token if it exists, regardless of path
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`
  }
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey
  }
  return config
})

// Interceptor：hanle error
axiosInstance.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response) {
      if (error.response?.status === 401) {
        // For login API, throw error directly
        if (error.config?.url?.includes('/login')) {
          throw error
        }
        // For other APIs, navigate to login page
        navigationService.navigateToLogin()

        // return a reject Promise
        return Promise.reject(new Error('Authentication required'))
      }
      throw new Error(
        `${error.response.status} ${error.response.statusText}\n${JSON.stringify(
          error.response.data
        )}\n${error.config?.url}`
      )
    }
    throw error
  }
)

// API methods
export const queryGraphs = async (
  label: string,
  maxDepth: number,
  maxNodes: number,
  minDegree = 0,
  includeOrphans = false
): Promise<LightragGraphType> => {
  const response = await axiosInstance.get(
    `/graphs?label=${encodeURIComponent(label)}&max_depth=${maxDepth}&max_nodes=${maxNodes}&min_degree=${minDegree}&include_orphans=${includeOrphans}`
  )
  return response.data
}

export const getGraphLabels = async (): Promise<string[]> => {
  const response = await axiosInstance.get('/graph/label/list')
  return response.data
}

export const getPopularLabels = async (
  limit: number = popularLabelsDefaultLimit
): Promise<string[]> => {
  const response = await axiosInstance.get(`/graph/label/popular?limit=${limit}`)
  return response.data
}

export const searchLabels = async (
  query: string,
  limit: number = searchLabelsDefaultLimit
): Promise<string[]> => {
  const response = await axiosInstance.get(
    `/graph/label/search?q=${encodeURIComponent(query)}&limit=${limit}`
  )
  return response.data
}

export const checkHealth = async (): Promise<
  LightragStatus | { status: 'error'; message: string }
> => {
  try {
    const response = await axiosInstance.get('/health')
    return response.data
  } catch (error) {
    return {
      status: 'error',
      message: errorMessage(error),
    }
  }
}

export const getDocuments = async (): Promise<DocsStatusesResponse> => {
  const response = await axiosInstance.get('/documents')
  return response.data
}

export const scanNewDocuments = async (): Promise<ScanResponse> => {
  const response = await axiosInstance.post('/documents/scan')
  return response.data
}

export const reprocessFailedDocuments = async (): Promise<ReprocessFailedResponse> => {
  const response = await axiosInstance.post('/documents/reprocess_failed')
  return response.data
}

export const getDocumentsScanProgress = async (): Promise<LightragDocumentsScanProgress> => {
  const response = await axiosInstance.get('/documents/scan-progress')
  return response.data
}

export const queryText = async (request: QueryRequest): Promise<QueryResponse> => {
  const response = await axiosInstance.post('/query', request)
  return response.data
}

export const queryTextStream = async (
  request: QueryRequest,
  onChunk: (chunk: string) => void,
  onError?: (error: string) => void,
  onCitations?: (metadata: CitationsMetadata) => void
) => {
  const apiKey = useSettingsStore.getState().apiKey
  const token = localStorage.getItem('LIGHTRAG-API-TOKEN')
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    Accept: 'application/x-ndjson',
  }
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  if (apiKey) {
    headers['X-API-Key'] = apiKey
  }

  try {
    const response = await fetch(`${backendBaseUrl}/query/stream`, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      // Handle 401 Unauthorized error specifically
      if (response.status === 401) {
        // For consistency with axios interceptor, navigate to login page
        navigationService.navigateToLogin()

        // Create a specific authentication error
        const authError = new Error('Authentication required')
        throw authError
      }

      // Handle other common HTTP errors with specific messages
      let errorBody = 'Unknown error'
      try {
        errorBody = await response.text() // Try to get error details from body
      } catch {
        /* ignore */
      }

      // Format error message similar to axios interceptor for consistency
      const url = `${backendBaseUrl}/query/stream`
      throw new Error(
        `${response.status} ${response.statusText}\n${JSON.stringify({ error: errorBody })}\n${url}`
      )
    }

    if (!response.body) {
      throw new Error('Response body is null')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break // Stream finished
      }

      // Decode the chunk and add to buffer
      buffer += decoder.decode(value, { stream: true }) // stream: true handles multi-byte chars split across chunks

      // Process complete lines (NDJSON)
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // Keep potentially incomplete line in buffer

      for (const line of lines) {
        if (line.trim()) {
          try {
            const parsed = JSON.parse(line)
            if (parsed.response) {
              onChunk(parsed.response)
            } else if (parsed.error && onError) {
              onError(parsed.error)
            } else if (parsed.citations_metadata && onCitations) {
              // NEW: Handle consolidated citations_metadata object
              onCitations(parsed.citations_metadata as CitationsMetadata)
            }
            // Silently ignore references and other events
          } catch (error) {
            console.error('Error parsing stream chunk:', line, error)
            if (onError) onError(`Error parsing server response: ${line}`)
          }
        }
      }
    }

    // Process any remaining data in the buffer after the stream ends
    if (buffer.trim()) {
      try {
        const parsed = JSON.parse(buffer)
        if (parsed.response) {
          onChunk(parsed.response)
        } else if (parsed.error && onError) {
          onError(parsed.error)
        } else if (parsed.citations_metadata && onCitations) {
          onCitations(parsed.citations_metadata as CitationsMetadata)
        }
      } catch (error) {
        console.error('Error parsing final chunk:', buffer, error)
        if (onError) onError(`Error parsing final server response: ${buffer}`)
      }
    }
  } catch (error) {
    const message = errorMessage(error)

    // Check if this is an authentication error
    if (message === 'Authentication required') {
      // Already navigated to login page in the response.status === 401 block
      console.error('Authentication required for stream request')
      if (onError) {
        onError('Authentication required')
      }
      return // Exit early, no need for further error handling
    }

    // Check for specific HTTP error status codes in the error message
    const statusCodeMatch = message.match(/^(\d{3})\s/)
    if (statusCodeMatch) {
      const statusCode = Number.parseInt(statusCodeMatch[1], 10)

      // Handle specific status codes with user-friendly messages
      let userMessage = message

      switch (statusCode) {
        case 403:
          userMessage = 'You do not have permission to access this resource (403 Forbidden)'
          console.error('Permission denied for stream request:', message)
          break
        case 404:
          userMessage = 'The requested resource does not exist (404 Not Found)'
          console.error('Resource not found for stream request:', message)
          break
        case 429:
          userMessage = 'Too many requests, please try again later (429 Too Many Requests)'
          console.error('Rate limited for stream request:', message)
          break
        case 500:
        case 502:
        case 503:
        case 504:
          userMessage = `Server error, please try again later (${statusCode})`
          console.error('Server error for stream request:', message)
          break
        default:
          console.error('Stream request failed with status code:', statusCode, message)
      }

      if (onError) {
        onError(userMessage)
      }
      return
    }

    // Handle network errors (like connection refused, timeout, etc.)
    if (
      message.includes('NetworkError') ||
      message.includes('Failed to fetch') ||
      message.includes('Network request failed')
    ) {
      console.error('Network error for stream request:', message)
      if (onError) {
        onError('Network connection error, please check your internet connection')
      }
      return
    }

    // Handle JSON parsing errors during stream processing
    if (message.includes('Error parsing') || message.includes('SyntaxError')) {
      console.error('JSON parsing error in stream:', message)
      if (onError) {
        onError('Error processing response data')
      }
      return
    }

    // Handle other errors
    console.error('Unhandled stream error:', message)
    if (onError) {
      onError(message)
    } else {
      console.error('No error handler provided for stream error:', message)
    }
  }
}

export const insertText = async (text: string): Promise<DocActionResponse> => {
  const response = await axiosInstance.post('/documents/text', { text })
  return response.data
}

export const insertTexts = async (texts: string[]): Promise<DocActionResponse> => {
  const response = await axiosInstance.post('/documents/texts', { texts })
  return response.data
}

export const uploadDocument = async (
  file: File,
  onUploadProgress?: (percentCompleted: number) => void
): Promise<DocActionResponse> => {
  const formData = new FormData()
  formData.append('file', file)

  const response = await axiosInstance.post('/documents/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    // prettier-ignore
    onUploadProgress:
      onUploadProgress !== undefined
        ? (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total!)
            onUploadProgress(percentCompleted)
          }
        : undefined,
  })
  return response.data
}

export const batchUploadDocuments = async (
  files: File[],
  onUploadProgress?: (fileName: string, percentCompleted: number) => void
): Promise<DocActionResponse[]> => {
  return await Promise.all(
    files.map(async (file) => {
      return await uploadDocument(file, (percentCompleted) => {
        onUploadProgress?.(file.name, percentCompleted)
      })
    })
  )
}

export const clearDocuments = async (): Promise<DocActionResponse> => {
  const response = await axiosInstance.delete('/documents')
  return response.data
}

export const clearCache = async (): Promise<{
  status: 'success' | 'fail'
  message: string
}> => {
  const response = await axiosInstance.post('/documents/clear_cache', {})
  return response.data
}

export const deleteDocuments = async (
  docIds: string[],
  deleteFile = false,
  deleteLLMCache = false
): Promise<DeleteDocResponse> => {
  const response = await axiosInstance.delete('/documents/delete_document', {
    data: { doc_ids: docIds, delete_file: deleteFile, delete_llm_cache: deleteLLMCache },
  })
  return response.data
}

export const getAuthStatus = async (): Promise<AuthStatusResponse> => {
  try {
    // Add a timeout to the request to prevent hanging
    const response = await axiosInstance.get('/auth-status', {
      timeout: 5000, // 5 second timeout
      headers: {
        Accept: 'application/json', // Explicitly request JSON
      },
    })

    // Check if response is HTML (which indicates a redirect or wrong endpoint)
    const contentType = response.headers['content-type'] || ''
    if (contentType.includes('text/html')) {
      console.warn('Received HTML response instead of JSON for auth-status endpoint')
      return {
        auth_configured: true,
        auth_mode: 'enabled',
      }
    }

    // Strict validation of the response data
    if (
      response.data &&
      typeof response.data === 'object' &&
      'auth_configured' in response.data &&
      typeof response.data.auth_configured === 'boolean'
    ) {
      // For unconfigured auth, ensure we have an access token
      if (!response.data.auth_configured) {
        if (response.data.access_token && typeof response.data.access_token === 'string') {
          return response.data
        } else {
          console.warn('Auth not configured but no valid access token provided')
        }
      } else {
        // For configured auth, just return the data
        return response.data
      }
    }

    // If response data is invalid but we got a response, log it
    console.warn('Received invalid auth status response:', response.data)

    // Default to auth configured if response is invalid
    return {
      auth_configured: true,
      auth_mode: 'enabled',
    }
  } catch (error) {
    // If the request fails, assume authentication is configured
    console.error('Failed to get auth status:', errorMessage(error))
    return {
      auth_configured: true,
      auth_mode: 'enabled',
    }
  }
}

export const getPipelineStatus = async (): Promise<PipelineStatusResponse> => {
  const response = await axiosInstance.get('/documents/pipeline_status')
  return response.data
}

export const cancelPipeline = async (): Promise<{
  status: 'cancellation_requested' | 'not_busy'
  message: string
}> => {
  const response = await axiosInstance.post('/documents/cancel_pipeline')
  return response.data
}

export const loginToServer = async (username: string, password: string): Promise<LoginResponse> => {
  const formData = new FormData()
  formData.append('username', username)
  formData.append('password', password)

  const response = await axiosInstance.post('/login', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

  return response.data
}

/**
 * Updates an entity's properties in the knowledge graph
 * @param entityName The name of the entity to update
 * @param updatedData Dictionary containing updated attributes
 * @param allowRename Whether to allow renaming the entity (default: false)
 * @param allowMerge Whether to merge into an existing entity when renaming to a duplicate name
 * @returns Promise with the updated entity information
 */
export const updateEntity = async (
  entityName: string,
  updatedData: Record<string, any>,
  allowRename = false,
  allowMerge = false
): Promise<EntityUpdateResponse> => {
  const response = await axiosInstance.post('/graph/entity/edit', {
    entity_name: entityName,
    updated_data: updatedData,
    allow_rename: allowRename,
    allow_merge: allowMerge,
  })
  return response.data
}

/**
 * Updates a relation's properties in the knowledge graph
 * @param sourceEntity The source entity name
 * @param targetEntity The target entity name
 * @param updatedData Dictionary containing updated attributes
 * @returns Promise with the updated relation information
 */
export const updateRelation = async (
  sourceEntity: string,
  targetEntity: string,
  updatedData: Record<string, any>
): Promise<DocActionResponse> => {
  const response = await axiosInstance.post('/graph/relation/edit', {
    source_id: sourceEntity,
    target_id: targetEntity,
    updated_data: updatedData,
  })
  return response.data
}

/**
 * Response from the orphan connection endpoint
 */
export type OrphanConnectionResponse = {
  status: string
  message: string
  data: {
    orphans_found: number
    connections_made: number
    connections: Array<{
      orphan: string
      connected_to: string
      relationship_type: string
      keywords: string
      confidence: number
      similarity: number
    }>
    errors: string[]
  }
}

/**
 * Status of the orphan connection background pipeline
 */
export type OrphanConnectionStatus = {
  busy: boolean
  job_name: string
  job_start: string | null
  total_orphans: number
  processed_orphans: number
  connections_made: number
  request_pending: boolean
  cancellation_requested: boolean
  latest_message: string
  history_messages: string[]
}

/**
 * Connects orphan entities (entities with no relationships) to the knowledge graph
 * @param maxCandidates Maximum number of connection candidates per orphan (1-10, default 3)
 * @param similarityThreshold Minimum vector similarity for candidates (0.0-1.0)
 * @param confidenceThreshold Minimum LLM confidence for connections (0.0-1.0)
 * @param crossConnect Whether to allow orphan-to-orphan connections
 * @returns Promise with connection results including number of connections made
 */
export const connectOrphanEntities = async (
  maxCandidates = 3,
  similarityThreshold?: number,
  confidenceThreshold?: number,
  crossConnect?: boolean
): Promise<OrphanConnectionResponse> => {
  const response = await axiosInstance.post('/graph/orphans/connect', {
    max_candidates: maxCandidates,
    similarity_threshold: similarityThreshold,
    confidence_threshold: confidenceThreshold,
    cross_connect: crossConnect,
  })
  return response.data
}

/**
 * Get the current status of the orphan connection background pipeline
 * @returns Promise with current pipeline status
 */
export const getOrphanConnectionStatus = async (): Promise<OrphanConnectionStatus> => {
  const response = await axiosInstance.get('/graph/orphans/status')
  return response.data
}

/**
 * Start orphan connection as a background job
 * @param maxCandidates Maximum candidates to evaluate per entity (default: 3)
 * @param maxDegree Maximum connection degree to target (default: 0)
 *   - 0: True orphans only (no connections)
 *   - 1: Orphans + leaf nodes (0-1 connections)
 *   - 2+: Include sparsely connected nodes
 * @returns Promise with start status
 */
export const startOrphanConnection = async (
  maxCandidates = 3,
  maxDegree = 0
): Promise<{ status: string }> => {
  const response = await axiosInstance.post('/graph/orphans/start', null, {
    params: { max_candidates: maxCandidates, max_degree: maxDegree },
  })
  return response.data
}

/**
 * Request cancellation of a running orphan connection job
 * @returns Promise with cancellation status
 */
export const cancelOrphanConnection = async (): Promise<{ status: string }> => {
  const response = await axiosInstance.post('/graph/orphans/cancel')
  return response.data
}

/**
 * Checks if an entity name already exists in the knowledge graph
 * @param entityName The entity name to check
 * @returns Promise with boolean indicating if the entity exists
 */
export const checkEntityNameExists = async (entityName: string): Promise<boolean> => {
  try {
    const response = await axiosInstance.get(
      `/graph/entity/exists?name=${encodeURIComponent(entityName)}`
    )
    return response.data.exists
  } catch (error) {
    console.error('Error checking entity name:', error)
    return false
  }
}

/**
 * Get the processing status of documents by tracking ID
 * @param trackId The tracking ID returned from upload, text, or texts endpoints
 * @returns Promise with the track status response containing documents and summary
 */
export const getTrackStatus = async (trackId: string): Promise<TrackStatusResponse> => {
  const response = await axiosInstance.get(`/documents/track_status/${encodeURIComponent(trackId)}`)
  return response.data
}

/**
 * Get documents with pagination support
 * @param request The pagination request parameters
 * @returns Promise with paginated documents response
 */
export const getDocumentsPaginated = async (
  request: DocumentsRequest
): Promise<PaginatedDocsResponse> => {
  const response = await axiosInstance.post('/documents/paginated', request)
  return response.data
}

/**
 * Get counts of documents by status
 * @returns Promise with status counts response
 */
export const getDocumentStatusCounts = async (): Promise<StatusCountsResponse> => {
  const response = await axiosInstance.get('/documents/status_counts')
  return response.data
}

export type TableSchema = {
  ddl: string
}

export type TableDataResponse = {
  data: Record<string, any>[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

// Mock data for dev mode
const mockTables = [
  'lightrag_doc_status',
  'lightrag_doc_chunks',
  'lightrag_entities',
  'lightrag_relations',
  'lightrag_entity_aliases',
  'lightrag_llm_cache',
]

const mockSchemas: Record<string, string> = {
  lightrag_doc_status: `CREATE TABLE lightrag_doc_status (
  id VARCHAR(255) PRIMARY KEY,
  workspace VARCHAR(255) NOT NULL,
  content_summary TEXT,
  content_length INTEGER,
  status VARCHAR(50),
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);`,
  lightrag_entities: `CREATE TABLE lightrag_entities (
  id SERIAL PRIMARY KEY,
  workspace VARCHAR(255) NOT NULL,
  entity_name VARCHAR(500) NOT NULL,
  entity_type VARCHAR(100),
  description TEXT,
  source_chunk_id VARCHAR(255),
  created_at TIMESTAMP DEFAULT NOW()
);`,
  lightrag_entity_aliases: `CREATE TABLE lightrag_entity_aliases (
  id SERIAL PRIMARY KEY,
  workspace VARCHAR(255) NOT NULL,
  alias VARCHAR(500) NOT NULL,
  canonical_entity VARCHAR(500) NOT NULL,
  method VARCHAR(50),
  confidence FLOAT,
  created_at TIMESTAMP DEFAULT NOW()
);`,
}

const mockTableData: Record<string, any[]> = {
  lightrag_doc_status: [
    {
      id: 'doc_001',
      workspace: 'default',
      content_summary: 'Research paper on AI...',
      content_length: 15420,
      status: 'processed',
      created_at: '2024-01-15T10:30:00Z',
    },
    {
      id: 'doc_002',
      workspace: 'default',
      content_summary: 'Technical documentation...',
      content_length: 8750,
      status: 'processed',
      created_at: '2024-01-16T14:22:00Z',
    },
    {
      id: 'doc_003',
      workspace: 'default',
      content_summary: 'Meeting notes from Q4...',
      content_length: 3200,
      status: 'pending',
      created_at: '2024-01-17T09:15:00Z',
    },
  ],
  lightrag_entities: [
    {
      id: 1,
      workspace: 'default',
      entity_name: 'OpenAI',
      entity_type: 'Organization',
      description: 'AI research company',
      created_at: '2024-01-15T10:30:00Z',
    },
    {
      id: 2,
      workspace: 'default',
      entity_name: 'GPT-4',
      entity_type: 'Product',
      description: 'Large language model',
      created_at: '2024-01-15T10:31:00Z',
    },
    {
      id: 3,
      workspace: 'default',
      entity_name: 'San Francisco',
      entity_type: 'Location',
      description: 'City in California',
      created_at: '2024-01-15T10:32:00Z',
    },
  ],
  lightrag_entity_aliases: [
    {
      id: 1,
      workspace: 'default',
      alias: 'openai',
      canonical_entity: 'OpenAI',
      method: 'exact',
      confidence: 1.0,
      created_at: '2024-01-15T10:30:00Z',
    },
    {
      id: 2,
      workspace: 'default',
      alias: 'gpt4',
      canonical_entity: 'GPT-4',
      method: 'fuzzy',
      confidence: 0.92,
      created_at: '2024-01-15T10:31:00Z',
    },
    {
      id: 3,
      workspace: 'default',
      alias: 'SF',
      canonical_entity: 'San Francisco',
      method: 'llm',
      confidence: 0.85,
      created_at: '2024-01-15T10:32:00Z',
    },
  ],
}

const SAFE_TABLE_NAME_REGEX = /^[a-zA-Z0-9_.-]+$/

export const getTableList = async (): Promise<string[]> => {
  if (import.meta.env.DEV) {
    return mockTables
  }
  const response = await axiosInstance.get('/tables/list')
  return response.data
}

export const getTableSchema = async (tableName: string): Promise<TableSchema> => {
  if (!tableName || typeof tableName !== 'string') {
    throw new Error('Invalid table name')
  }
  if (!SAFE_TABLE_NAME_REGEX.test(tableName)) {
    throw new Error('Invalid table name: contains forbidden characters')
  }
  if (import.meta.env.DEV) {
    return { ddl: mockSchemas[tableName] || `-- Schema not available for ${tableName}` }
  }
  const response = await axiosInstance.get(`/tables/${encodeURIComponent(tableName)}/schema`)
  return response.data
}

export const getTableData = async (
  tableName: string,
  page: number,
  pageSize: number
): Promise<TableDataResponse> => {
  if (!tableName || typeof tableName !== 'string') {
    throw new Error('Invalid table name')
  }
  if (!SAFE_TABLE_NAME_REGEX.test(tableName)) {
    throw new Error('Invalid table name: contains forbidden characters')
  }
  if (
    !Number.isInteger(page) ||
    !Number.isInteger(pageSize) ||
    page < 1 ||
    pageSize < 1 ||
    pageSize > 1000
  ) {
    throw new Error('Page must be >= 1 and page size must be between 1 and 1000')
  }

  if (import.meta.env.DEV) {
    const data = mockTableData[tableName] || []
    const start = (page - 1) * pageSize
    const end = start + pageSize
    const paginatedData = data.slice(start, end)
    return {
      data: paginatedData,
      total: data.length,
      page: page,
      page_size: pageSize,
      total_pages: Math.ceil(data.length / pageSize),
    }
  }
  const response = await axiosInstance.get(`/tables/${encodeURIComponent(tableName)}/data`, {
    params: { page, page_size: pageSize },
  })
  return response.data
}

// =====================
// S3 Storage Browser API
// =====================

export type S3ObjectInfo = {
  key: string
  size: number
  last_modified: string
  content_type: string | null
}

export type S3ListResponse = {
  bucket: string
  prefix: string
  folders: string[]
  objects: S3ObjectInfo[]
}

export type S3DownloadResponse = {
  key: string
  url: string
  expiry_seconds: number
}

export type S3UploadResponse = {
  key: string
  size: number
  url: string
}

export type S3DeleteResponse = {
  key: string
  status: string
}

/**
 * List objects and folders under a prefix in the S3 bucket.
 * @param prefix - S3 prefix to list (e.g., "staging/default/")
 * @returns List of folders and objects at the prefix
 */
export const s3List = async (prefix = ''): Promise<S3ListResponse> => {
  const response = await axiosInstance.get('/s3/list', {
    params: { prefix },
  })
  return response.data
}

/**
 * Get a presigned download URL for an S3 object.
 * @param key - Full S3 object key
 * @param expiry - URL expiry time in seconds (default: 3600)
 * @returns Presigned download URL
 */
export const s3Download = async (key: string, expiry = 3600): Promise<S3DownloadResponse> => {
  const response = await axiosInstance.get(`/s3/download/${encodeURIComponent(key)}`, {
    params: { expiry },
  })
  return response.data
}

/**
 * Upload a file to the S3 bucket.
 * @param prefix - S3 prefix path (e.g., "staging/default/")
 * @param file - File to upload
 * @returns Upload result with key and presigned URL
 */
export const s3Upload = async (prefix: string, file: File): Promise<S3UploadResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('prefix', prefix)
  const response = await axiosInstance.post('/s3/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

/**
 * Delete an object from the S3 bucket.
 * @param key - Full S3 object key to delete
 * @returns Deletion confirmation
 */
export const s3Delete = async (key: string): Promise<S3DeleteResponse> => {
  const response = await axiosInstance.delete(`/s3/object/${encodeURIComponent(key)}`)
  return response.data
}
