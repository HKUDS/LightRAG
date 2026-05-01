import axios, { AxiosError } from 'axios'
import type { AxiosProgressEvent } from 'axios'
import { backendBaseUrl, popularLabelsDefaultLimit, searchLabelsDefaultLimit } from '@/lib/constants'
import { errorMessage } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { useAuthStore } from '@/stores/state'
import { navigationService } from '@/services/navigation'

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

export type Message = {
  role: 'user' | 'assistant' | 'system'
  content: string
  thinkingContent?: string
  displayContent?: string
  thinkingTime?: number | null
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
}

export type QueryResponse = {
  response: string
  references?: Array<Record<string, any>> | null
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
  auth_mode?: 'enabled' | 'disabled' | 'enterprise'
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
  auth_mode?: 'enabled' | 'disabled' | 'enterprise'  // Authentication mode identifier
  message?: string                    // Optional message
  core_version?: string
  api_version?: string
  webui_title?: string
  webui_description?: string
  principal?: LittleBullPrincipal
}

export type LittleBullPrincipal = {
  user_id: string
  sub: string
  tenant_id: string | null
  is_master_global: boolean
  roles: string[]
  workspace_ids: string[]
  permission_version: number
  permissions: string[]
}

export type LittleBullArea = {
  id: string
  label: string
  slug: string
  description: string
  privacy: string
  document_count: number
  ready_count: number
  processing_count: number
  accent: string
  emoji: string
  data_plane_attached?: boolean
  chat_model_id?: string | null
  embedding_model_id?: string | null
  embedding_reindex_required?: boolean
}

export type LittleBullKnowledgeGroup = {
  group_id: string
  tenant_id?: string | null
  workspace_id: string
  slug: string
  name: string
  description: string
  privacy: string
  color: string
  metadata: Record<string, any>
  created_at?: string | null
  updated_at?: string | null
}

export type LittleBullKnowledgeSubgroup = {
  subgroup_id: string
  tenant_id?: string | null
  workspace_id: string
  group_id: string
  slug: string
  name: string
  description: string
  privacy: string
  metadata: Record<string, any>
  created_at?: string | null
  updated_at?: string | null
}

export type LittleBullDocument = {
  id: string
  file_path: string
  title: string
  status: string
  content_summary: string
  content_length: number
  group_id?: string | null
  subgroup_id?: string | null
  registry_document_id?: string | null
  updated_at?: string | null
  created_at?: string | null
  track_id?: string | null
  chunks_count?: number | null
  metadata: Record<string, any>
}

export type LittleBullDocumentsResponse = {
  documents: LittleBullDocument[]
  total_count: number
  status_counts: Record<string, number>
}

export type LittleBullQueryRequest = {
  workspace_id: string
  query: string
  mode?: QueryMode
  response_type?: string
  top_k?: number
  include_references?: boolean
  include_chunk_content?: boolean
  conversation_history?: Message[]
  confidentiality?: 'normal' | 'sensivel' | 'privado'
  model_profile?: string
  agent_id?: string | null
}

export type LittleBullQueryResponse = {
  response: string
  references: Array<Record<string, any>>
  workspace_id: string
  model_profile: string
}

export type LittleBullUploadResponse = {
  status: string
  message: string
  track_id?: string | null
  workspace_id: string
  group_id?: string | null
  subgroup_id?: string | null
  registry_document_id?: string | null
}

export type LittleBullReindexArchivedResponse = {
  status: string
  message: string
  track_id?: string | null
  workspace_id: string
  recovered_count: number
  skipped_count: number
  files: string[]
}

export type LittleBullActivityItem = {
  id: string
  action: string
  result: string
  created_at: string
  actor_user_id: string
  workspace_id?: string | null
  metadata: Record<string, any>
}

export type LittleBullAssistant = {
  id: string
  name: string
  description: string
  enabled: boolean
  response_rules: string[]
}

export type LittleBullModelUsage = 'chat' | 'embedding' | 'rerank' | 'agent'

export type LittleBullModelSetting = {
  model_setting_id?: string | null
  tenant_id?: string | null
  workspace_id?: string | null
  usage: LittleBullModelUsage
  provider: string
  binding: string
  binding_host: string
  model_id: string
  display_name: string
  enabled: boolean
  is_default: boolean
  config: Record<string, any>
  created_by?: string | null
  updated_by?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export type LittleBullEmbeddingCatalogItem = {
  model_id: string
  display_name: string
  provider: string
  binding: string
  binding_host: string
  context_length: number
  prompt_cost_per_million_tokens: number
  prompt_cost_per_token: number
  estimated_cost_100k_tokens: number
  estimated_cost_200k_tokens: number
  quality_tier: string
  recommended_chunk_tokens: number
  notes: string
}

export type LittleBullKnowledgeBase = {
  workspace_id: string
  tenant_id?: string | null
  name: string
  slug: string
  description: string
  privacy: string
  data_plane_attached: boolean
  document_count: number
  ready_count: number
  processing_count: number
  chat_model?: LittleBullModelSetting | null
  embedding_model?: LittleBullModelSetting | null
  embedding_reindex_required: boolean
  embedding_estimated_tokens: number
  embedding_estimated_cost_usd: number
}

export type LittleBullKnowledgeBaseUpsertRequest = {
  workspace_id?: string | null
  name: string
  slug?: string | null
  description?: string
  privacy?: string
  embedding_model_id?: string | null
  estimated_tokens?: number | null
}

export type LittleBullKnowledgeBaseAttachResponse = {
  status: string
  message: string
  workspace_id: string
  data_plane_attached: boolean
  input_dir?: string | null
  working_dir?: string | null
}

export type LittleBullKnowledgeBaseReindexResponse = {
  status: string
  message: string
  workspace_id: string
  track_id?: string | null
  approval?: LittleBullApproval | null
  destructive_rebuild?: boolean
  snapshot_id?: string | null
  snapshot_path?: string | null
  rollback_available?: boolean
  queued_count: number
  skipped_count: number
  files: string[]
}

export type LittleBullEmbeddingCostEstimateRequest = {
  workspace_id: string
  model_id: string
  estimated_tokens?: number | null
  page_count?: number | null
  words_per_page?: number
}

export type LittleBullEmbeddingCostEstimateResponse = {
  workspace_id: string
  model_id: string
  display_name: string
  estimated_tokens: number
  estimated_cost_usd: number
  prompt_cost_per_million_tokens: number
  context_length: number
  recommended_chunk_tokens: number
  reindex_required: boolean
  notes: string[]
}

export type LittleBullAgentStudioConfig = {
  schema_version?: number
  identity?: {
    mission?: string
    when_to_use?: string
    when_not_to_use?: string
    audience?: string
  }
  model?: {
    profile?: string
    temperature?: number
    max_tokens?: number
    cost_limit?: string
    fallback_model_setting_id?: string
  }
  knowledge?: {
    retrieval_mode?: QueryMode
    allowed_workspace_ids?: string[]
    allowed_labels?: string[]
    require_sources?: boolean
    block_without_context?: boolean
  }
  persona?: {
    tone?: string
    formality?: string
    verbosity?: string
    technical_level?: string
    humor?: string
    posture?: string
  }
  ethics?: {
    principles?: string[]
    refusal_rules?: string[]
    human_approval_triggers?: string[]
    sensitive_topics?: string[]
    privacy_rules?: string[]
  }
  vocabulary?: {
    preferred_terms?: string[]
    forbidden_terms?: string[]
    required_phrases?: string[]
    forbidden_phrases?: string[]
  }
  tools_policy?: {
    allowed_tools?: string[]
    approval_required_tools?: string[]
    disabled_tools?: string[]
  }
  memory?: {
    enabled?: boolean
    scope?: 'conversation' | 'user' | 'workspace'
    retention_days?: number
    never_save?: string[]
  }
  output?: {
    default_format?: string
    include_sources?: boolean
    include_next_steps?: boolean
    include_uncertainty?: boolean
    template?: string
  }
  tests?: Array<{
    name?: string
    input?: string
    expected_behavior?: string
    forbidden_behavior?: string
  }>
} & Record<string, any>

export type LittleBullAgentConfig = {
  agent_id?: string | null
  tenant_id?: string | null
  workspace_id?: string | null
  name: string
  description: string
  enabled: boolean
  model_setting_id?: string | null
  system_prompt: string
  response_rules: string[]
  tools: string[]
  config: LittleBullAgentStudioConfig
  created_by?: string | null
  updated_by?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export type LittleBullAgentStudioIssue = {
  severity: 'error' | 'warning'
  field: string
  message: string
}

export type LittleBullAgentStudioPreviewResponse = {
  agent: LittleBullAgentConfig
  issues: LittleBullAgentStudioIssue[]
  readiness_score: number
  ready_to_publish: boolean
  compiled_prompt: string
  test_input: string
  test_summary: string
}

export type LittleBullConversationMessage = {
  message_id?: string | null
  id?: string | null
  role: 'user' | 'assistant' | 'system'
  content: string
  references: Array<Record<string, any>>
  metadata?: Record<string, any>
  created_at?: string | null
}

export type LittleBullConversation = {
  conversation_id: string
  tenant_id?: string | null
  workspace_id: string
  user_id: string
  title: string
  agent_id?: string | null
  model_profile: string
  confidentiality: string
  message_count: number
  messages?: LittleBullConversationMessage[]
  created_at?: string | null
  updated_at?: string | null
}

export type LittleBullCorrelationSuggestion = {
  suggestion_id: string
  tenant_id?: string | null
  workspace_id: string
  user_id: string
  source_label: string
  target_label: string
  reason: string
  status: 'pending' | 'approved' | 'rejected'
  metadata: Record<string, any>
  created_at?: string | null
  decided_at?: string | null
  decided_by?: string | null
}

export type LittleBullApproval = {
  approval_id: string
  action: string
  actor_user_id: string
  tenant_id: string | null
  workspace_id: string | null
  payload_hash: string
  reason: string
  status: 'pending' | 'approved' | 'executing' | 'executed' | 'failed' | 'rejected'
  requested_at: string
  decided_at?: string | null
  decided_by?: string | null
  metadata: Record<string, any>
}

export type LittleBullDeleteDocumentResponse =
  | {
    status: 'success'
    message: string
    doc_id: string
  }
  | {
    status: 'pending_approval'
    message: string
    approval: LittleBullApproval
  }

export type LittleBullAuditEvent = {
  event_id: string
  actor_user_id: string
  action: string
  tenant_id: string | null
  workspace_id: string | null
  result: string
  approval_id?: string | null
  model?: string | null
  metadata: Record<string, any>
  created_at: string
}

export type LittleBullCostPeriodSummary = {
  name: string
  since?: string | null
  request_count: number
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  estimated_cost_usd: number
  actual_cost_usd: number
  cost_usd: number
}

export type LittleBullCostBreakdownItem = {
  key: string
  label: string
  request_count: number
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  estimated_cost_usd: number
  actual_cost_usd: number
  cost_usd: number
  metadata?: Record<string, any>
}

export type LittleBullCostSummaryResponse = {
  workspace_id: string
  currency: string
  periods: Record<string, LittleBullCostPeriodSummary>
  by_user: LittleBullCostBreakdownItem[]
  by_agent: LittleBullCostBreakdownItem[]
  by_model: LittleBullCostBreakdownItem[]
  by_group_subgroup: LittleBullCostBreakdownItem[]
  by_operation: LittleBullCostBreakdownItem[]
}

export type LittleBullKnowledgeDossier = {
  knowledge_dossier_id: string
  tenant_id?: string | null
  workspace_id: string
  group_id?: string | null
  subgroup_id?: string | null
  title: string
  slug: string
  dossier_kind: string
  status: string
  content_refs: Array<Record<string, any>>
  export_policy: Record<string, any>
  approval_id?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export type LittleBullLegalMatterExtractionRun = {
  legal_matter_extraction_run_id: string
  tenant_id?: string | null
  workspace_id: string
  group_id?: string | null
  subgroup_id?: string | null
  document_id?: string | null
  matter_reference: string
  extraction_model_id: string
  schema_version: string
  run_status: string
  extracted_payload: Record<string, any>
  source_refs: Array<Record<string, any>>
  confidence?: number | null
  review_status: 'pending' | 'approved' | 'rejected' | 'needs_changes'
  requires_human_review: boolean
  approved_by?: string | null
  approved_at?: string | null
  error_message?: string
  created_at?: string | null
  updated_at?: string | null
}

export const InvalidApiKeyError = 'Invalid API Key'
export const RequireApiKeError = 'API Key required'

// Axios instance
const axiosInstance = axios.create({
  baseURL: backendBaseUrl,
  headers: {
    'Content-Type': 'application/json'
  }
})

// ========== Token Management ==========
// Prevent multiple requests from triggering token refresh simultaneously
let isRefreshingGuestToken = false;
let refreshTokenPromise: Promise<string> | null = null;

// Silent refresh for guest token
const silentRefreshGuestToken = async (): Promise<string> => {
  // If already refreshing, return the same Promise
  if (isRefreshingGuestToken && refreshTokenPromise) {
    return refreshTokenPromise;
  }

  isRefreshingGuestToken = true;
  refreshTokenPromise = (async () => {
    try {
      // Call /auth-status to get new guest token
      const response = await axios.get('/auth-status', {
        baseURL: backendBaseUrl,
        // This request must skip the interceptor to avoid adding expired token
        headers: { 'X-Skip-Interceptor': 'true' }
      });

      if (response.data.access_token && !response.data.auth_configured) {
        const newToken = response.data.access_token;
        // Update localStorage
        localStorage.setItem('LIGHTRAG-API-TOKEN', newToken);
        // Update auth state
        useAuthStore.getState().login(
          newToken,
          true,
          response.data.core_version,
          response.data.api_version,
          response.data.webui_title || null,
          response.data.webui_description || null
        );
        return newToken;
      } else {
        throw new Error('Failed to get guest token');
      }
    } finally {
      isRefreshingGuestToken = false;
      refreshTokenPromise = null;
    }
  })();

  return refreshTokenPromise;
};

// Interceptor: add api key and check authentication
axiosInstance.interceptors.request.use((config) => {
  // Skip interceptor for token refresh requests
  if (config.headers['X-Skip-Interceptor']) {
    delete config.headers['X-Skip-Interceptor'];
    return config;
  }

  const apiKey = useSettingsStore.getState().apiKey
  const token = localStorage.getItem('LIGHTRAG-API-TOKEN');

  // Always include token if it exists, regardless of path
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`
  }
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey
  }
  return config
})

// Interceptor：handle token renewal and authentication errors
axiosInstance.interceptors.response.use(
  (response) => {
    // ========== Check for new token from backend ==========
    const newToken = response.headers['x-new-token'];
    if (newToken) {
      localStorage.setItem('LIGHTRAG-API-TOKEN', newToken);

      // Optional: log in development mode
      if (import.meta.env.DEV) {
        console.log('[Auth] Token auto-renewed by backend');
      }

      // Update auth state with renewal tracking
      try {
        const payload = JSON.parse(atob(newToken.split('.')[1]));
        const authStore = useAuthStore.getState();
        if (authStore.isAuthenticated) {
          // Track token renewal time and expiration
          const renewalTime = Date.now();
          const expiresAt = payload.exp ? payload.exp * 1000 : 0;
          authStore.setTokenRenewal(renewalTime, expiresAt);

          // Update username (usually unchanged, but just in case)
          const newUsername = payload.sub;
          if (newUsername && newUsername !== authStore.username) {
            // Need to add setUsername method or just update via login
            // For now, we'll skip username update as it's rare
          }
        }
      } catch (error) {
        console.warn('[Auth] Failed to parse renewed token:', error);
      }
    }
    // ========== End of token renewal check ==========

    return response;
  },
  async (error: AxiosError) => {
    if (error.response) {
      if (error.response?.status === 401) {
        const originalRequest = error.config;

        // 1. For login API, throw error directly
        if (originalRequest?.url?.includes('/login')) {
          throw error;
        }

        // 2. Prevent infinite retry
        if (originalRequest && (originalRequest as any)._retry) {
          navigationService.navigateToLogin();
          return Promise.reject(new Error('Authentication required'));
        }

        // 3. Check if in guest mode
        const authStore = useAuthStore.getState();
        const currentToken = localStorage.getItem('LIGHTRAG-API-TOKEN');
        const isGuest = currentToken && authStore.isGuestMode;

        // 4. Guest mode: silent refresh and retry
        if (isGuest && originalRequest) {
          try {
            const newToken = await silentRefreshGuestToken();

            // Mark as retried to prevent infinite loop
            (originalRequest as any)._retry = true;

            // Update token in request headers
            originalRequest.headers['Authorization'] = `Bearer ${newToken}`;

            // Retry original request
            return axiosInstance(originalRequest);
          } catch (refreshError) {
            console.error('Failed to refresh guest token:', refreshError);
            // Refresh failed, navigate to login
            navigationService.navigateToLogin();
            return Promise.reject(new Error('Failed to refresh authentication'));
          }
        }

        // 5. Non-guest mode: navigate to login page
        navigationService.navigateToLogin();
        return Promise.reject(new Error('Authentication required'));
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
  workspaceId?: string
): Promise<LightragGraphType> => {
  const response = workspaceId
    ? await axiosInstance.get('/little-bull/graph', {
      params: { workspace_id: workspaceId, label, max_depth: maxDepth, max_nodes: maxNodes }
    })
    : await axiosInstance.get(`/graphs?label=${encodeURIComponent(label)}&max_depth=${maxDepth}&max_nodes=${maxNodes}`)
  return response.data
}

export const getGraphLabels = async (workspaceId?: string): Promise<string[]> => {
  const response = workspaceId
    ? await axiosInstance.get('/little-bull/graph/label/list', { params: { workspace_id: workspaceId } })
    : await axiosInstance.get('/graph/label/list')
  return response.data
}

export const getPopularLabels = async (
  limit: number = popularLabelsDefaultLimit,
  workspaceId?: string
): Promise<string[]> => {
  const response = workspaceId
    ? await axiosInstance.get('/little-bull/graph/label/popular', { params: { workspace_id: workspaceId, limit } })
    : await axiosInstance.get(`/graph/label/popular?limit=${limit}`)
  return response.data
}

export const searchLabels = async (
  query: string,
  limit: number = searchLabelsDefaultLimit,
  workspaceId?: string
): Promise<string[]> => {
  const response = workspaceId
    ? await axiosInstance.get('/little-bull/graph/label/search', { params: { workspace_id: workspaceId, q: query, limit } })
    : await axiosInstance.get(`/graph/label/search?q=${encodeURIComponent(query)}&limit=${limit}`)
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
      message: errorMessage(error)
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
  onError?: (error: string) => void
) => {
  const apiKey = useSettingsStore.getState().apiKey;
  const token = localStorage.getItem('LIGHTRAG-API-TOKEN');
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    'Accept': 'application/x-ndjson',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }

  try {
    const response = await fetch(`${backendBaseUrl}/query/stream`, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      // Handle 401 Unauthorized error specifically
      if (response.status === 401) {
        // Check if in guest mode
        const authStore = useAuthStore.getState();
        const currentToken = localStorage.getItem('LIGHTRAG-API-TOKEN');
        const isGuest = currentToken && authStore.isGuestMode;

        if (isGuest) {
          try {
            // Silent refresh token for guest mode
            const newToken = await silentRefreshGuestToken();

            // Retry stream request with new token
            const retryHeaders = { ...headers };
            retryHeaders['Authorization'] = `Bearer ${newToken}`;

            const retryResponse = await fetch(`${backendBaseUrl}/query/stream`, {
              method: 'POST',
              headers: retryHeaders,
              body: JSON.stringify(request),
            });

            if (!retryResponse.ok) {
              throw new Error(`HTTP error! status: ${retryResponse.status}`);
            }

            // Retry successful, process stream response
            // Re-execute the stream processing logic with retryResponse
            if (!retryResponse.body) {
              throw new Error('Response body is null');
            }

            const reader = retryResponse.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n');
              buffer = lines.pop() || '';

              for (const line of lines) {
                if (line.trim()) {
                  try {
                    const parsed = JSON.parse(line);
                    if (parsed.response) {
                      onChunk(parsed.response);
                    } else if (parsed.error) {
                      onError?.(parsed.error);
                    }
                  } catch (parseError) {
                    console.error('Failed to parse JSON:', parseError, 'Line:', line);
                    onError?.(`JSON parse error: ${parseError}`);
                  }
                }
              }
            }

            // Process any remaining data in buffer
            if (buffer.trim()) {
              try {
                const parsed = JSON.parse(buffer);
                if (parsed.response) {
                  onChunk(parsed.response);
                } else if (parsed.error) {
                  onError?.(parsed.error);
                }
              } catch (parseError) {
                console.error('Failed to parse final buffer:', parseError);
              }
            }

            return; // Successfully completed retry
          } catch (refreshError) {
            console.error('Failed to refresh guest token for streaming:', refreshError);
            navigationService.navigateToLogin();
            throw new Error('Failed to refresh authentication', { cause: refreshError });
          }
        }

        // Non-guest mode: navigate to login page
        navigationService.navigateToLogin();

        // Create a specific authentication error
        const authError = new Error('Authentication required');
        throw authError;
      }

      // Handle other common HTTP errors with specific messages
      let errorBody = 'Unknown error';
      try {
        errorBody = await response.text(); // Try to get error details from body
      } catch { /* ignore */ }

      // Format error message similar to axios interceptor for consistency
      const url = `${backendBaseUrl}/query/stream`;
      throw new Error(
        `${response.status} ${response.statusText}\n${JSON.stringify(
          { error: errorBody }
        )}\n${url}`
      );
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break; // Stream finished
      }

      // Decode the chunk and add to buffer
      buffer += decoder.decode(value, { stream: true }); // stream: true handles multi-byte chars split across chunks

      // Process complete lines (NDJSON)
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep potentially incomplete line in buffer

      for (const line of lines) {
        if (line.trim()) {
          try {
            const parsed = JSON.parse(line);
            if (parsed.response) {
              onChunk(parsed.response);
            } else if (parsed.error && onError) {
              onError(parsed.error);
            }
          } catch (error) {
            console.error('Error parsing stream chunk:', line, error);
            if (onError) onError(`Error parsing server response: ${line}`);
          }
        }
      }
    }

    // Process any remaining data in the buffer after the stream ends
    if (buffer.trim()) {
      try {
        const parsed = JSON.parse(buffer);
        if (parsed.response) {
          onChunk(parsed.response);
        } else if (parsed.error && onError) {
          onError(parsed.error);
        }
      } catch (error) {
        console.error('Error parsing final chunk:', buffer, error);
        if (onError) onError(`Error parsing final server response: ${buffer}`);
      }
    }

  } catch (error) {
    const message = errorMessage(error);

    // Check if this is an authentication error
    if (message === 'Authentication required') {
      // Already navigated to login page in the response.status === 401 block
      console.error('Authentication required for stream request');
      if (onError) {
        onError('Authentication required');
      }
      return; // Exit early, no need for further error handling
    }

    // Check for specific HTTP error status codes in the error message
    const statusCodeMatch = message.match(/^(\d{3})\s/);
    if (statusCodeMatch) {
      const statusCode = parseInt(statusCodeMatch[1], 10);

      // Handle specific status codes with user-friendly messages
      let userMessage = message;

      switch (statusCode) {
        case 403:
          userMessage = 'You do not have permission to access this resource (403 Forbidden)';
          console.error('Permission denied for stream request:', message);
          break;
        case 404:
          userMessage = 'The requested resource does not exist (404 Not Found)';
          console.error('Resource not found for stream request:', message);
          break;
        case 429:
          userMessage = 'Too many requests, please try again later (429 Too Many Requests)';
          console.error('Rate limited for stream request:', message);
          break;
        case 500:
        case 502:
        case 503:
        case 504:
          userMessage = `Server error, please try again later (${statusCode})`;
          console.error('Server error for stream request:', message);
          break;
        default:
          console.error('Stream request failed with status code:', statusCode, message);
      }

      if (onError) {
        onError(userMessage);
      }
      return;
    }

    // Handle network errors (like connection refused, timeout, etc.)
    if (message.includes('NetworkError') ||
        message.includes('Failed to fetch') ||
        message.includes('Network request failed')) {
      console.error('Network error for stream request:', message);
      if (onError) {
        onError('Network connection error, please check your internet connection');
      }
      return;
    }

    // Handle JSON parsing errors during stream processing
    if (message.includes('Error parsing') || message.includes('SyntaxError')) {
      console.error('JSON parsing error in stream:', message);
      if (onError) {
        onError('Error processing response data');
      }
      return;
    }

    // Handle other errors
    console.error('Unhandled stream error:', message);
    if (onError) {
      onError(message);
    } else {
      console.error('No error handler provided for stream error:', message);
    }
  }
};

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
      'Content-Type': 'multipart/form-data'
    },
    // prettier-ignore
    onUploadProgress:
      onUploadProgress !== undefined
        ? (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total!)
          onUploadProgress(percentCompleted)
        }
        : undefined
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
  deleteFile: boolean = false,
  deleteLLMCache: boolean = false
): Promise<DeleteDocResponse> => {
  const response = await axiosInstance.delete('/documents/delete_document', {
    data: { doc_ids: docIds, delete_file: deleteFile, delete_llm_cache: deleteLLMCache }
  })
  return response.data
}

export const getAuthStatus = async (): Promise<AuthStatusResponse> => {
  try {
    // Add a timeout to the request to prevent hanging
    const response = await axiosInstance.get('/auth-status', {
      timeout: 5000, // 5 second timeout
      headers: {
        'Accept': 'application/json' // Explicitly request JSON
      }
    });

    // Check if response is HTML (which indicates a redirect or wrong endpoint)
    const contentType = String(response.headers['content-type'] ?? '');
    if (contentType.includes('text/html')) {
      console.warn('Received HTML response instead of JSON for auth-status endpoint');
      return {
        auth_configured: true,
        auth_mode: 'enabled'
      };
    }

    // Strict validation of the response data
    if (response.data &&
        typeof response.data === 'object' &&
        'auth_configured' in response.data &&
        typeof response.data.auth_configured === 'boolean') {

      // For unconfigured auth, ensure we have an access token
      if (!response.data.auth_configured) {
        if (response.data.access_token && typeof response.data.access_token === 'string') {
          return response.data;
        } else {
          console.warn('Auth not configured but no valid access token provided');
        }
      } else {
        // For configured auth, just return the data
        return response.data;
      }
    }

    // If response data is invalid but we got a response, log it
    console.warn('Received invalid auth status response:', response.data);

    // Default to auth configured if response is invalid
    return {
      auth_configured: true,
      auth_mode: 'enabled'
    };
  } catch (error) {
    // If the request fails, assume authentication is configured
    console.error('Failed to get auth status:', errorMessage(error));
    return {
      auth_configured: true,
      auth_mode: 'enabled'
    };
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
  const formData = new URLSearchParams();
  formData.append('username', username);
  formData.append('password', password);
  formData.append('grant_type', 'password');

  const response = await axiosInstance.post('/login', formData, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
  });

  return response.data;
}

export const getLittleBullMe = async (): Promise<LittleBullPrincipal> => {
  const response = await axiosInstance.get('/auth/me')
  return response.data
}

export const getLittleBullAreas = async (): Promise<LittleBullArea[]> => {
  const response = await axiosInstance.get('/little-bull/areas')
  return response.data.areas
}

export const getLittleBullKnowledgeGroups = async (
  workspaceId: string
): Promise<LittleBullKnowledgeGroup[]> => {
  const response = await axiosInstance.get('/little-bull/knowledge-groups', {
    params: { workspace_id: workspaceId }
  })
  return response.data.groups
}

export const getLittleBullKnowledgeSubgroups = async (
  workspaceId: string,
  groupId?: string | null
): Promise<LittleBullKnowledgeSubgroup[]> => {
  const response = await axiosInstance.get('/little-bull/knowledge-subgroups', {
    params: { workspace_id: workspaceId, group_id: groupId || undefined }
  })
  return response.data.subgroups
}

export const getLittleBullDocuments = async (
  workspaceId: string,
  page: number = 1,
  pageSize: number = 50
): Promise<LittleBullDocumentsResponse> => {
  const response = await axiosInstance.get('/little-bull/documents', {
    params: { workspace_id: workspaceId, page, page_size: pageSize }
  })
  return response.data
}

type LittleBullDocumentConfidentiality = 'normal' | 'sensivel' | 'privado'

type LittleBullUploadDocumentConfig = {
  params: {
    workspace_id: string
    group_id: string
    subgroup_id: string
    confidentiality: LittleBullDocumentConfidentiality
  }
  headers: { 'Content-Type': string }
  onUploadProgress?: (progressEvent: AxiosProgressEvent) => void
}

const defaultLittleBullUploadPost = async (
  formData: FormData,
  config: LittleBullUploadDocumentConfig
): Promise<LittleBullUploadResponse> => {
  const response = await axiosInstance.post('/little-bull/documents/upload', formData, config)
  return response.data
}

let littleBullUploadPost = defaultLittleBullUploadPost

export const uploadLittleBullDocument = async (
  workspaceId: string,
  groupId: string,
  subgroupId: string,
  file: File,
  confidentiality: LittleBullDocumentConfidentiality = 'normal',
  onUploadProgress?: (percentCompleted: number) => void
): Promise<LittleBullUploadResponse> => {
  const formData = new FormData()
  formData.append('file', file)

  return littleBullUploadPost(formData, {
    params: { workspace_id: workspaceId, group_id: groupId, subgroup_id: subgroupId, confidentiality },
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress:
      onUploadProgress !== undefined
        ? (progressEvent) => {
          const total = progressEvent.total || progressEvent.loaded || 1
          onUploadProgress(Math.round((progressEvent.loaded * 100) / total))
        }
        : undefined
  })
}

export const reindexLittleBullArchivedDocuments = async (
  workspaceId: string
): Promise<LittleBullReindexArchivedResponse> => {
  const response = await axiosInstance.post('/little-bull/documents/reindex-archived', null, {
    params: { workspace_id: workspaceId }
  })
  return response.data
}

export const deleteLittleBullDocument = async (
  workspaceId: string,
  documentId: string
): Promise<LittleBullDeleteDocumentResponse> => {
  const response = await axiosInstance.delete(`/little-bull/documents/${encodeURIComponent(documentId)}`, {
    params: { workspace_id: workspaceId }
  })
  return response.data
}

export const queryLittleBull = async (
  request: LittleBullQueryRequest
): Promise<LittleBullQueryResponse> => {
  const response = await axiosInstance.post('/little-bull/query', request)
  return response.data
}

export const getLittleBullActivity = async (
  workspaceId: string,
  limit: number = 50
): Promise<LittleBullActivityItem[]> => {
  const response = await axiosInstance.get('/little-bull/activity', {
    params: { workspace_id: workspaceId, limit }
  })
  return response.data.activity
}

export const getLittleBullAssistants = async (
  workspaceId: string
): Promise<LittleBullAssistant[]> => {
  const response = await axiosInstance.get('/little-bull/assistants', {
    params: { workspace_id: workspaceId }
  })
  return response.data.assistants
}

export const getLittleBullAdminModels = async (
  workspaceId: string
): Promise<LittleBullModelSetting[]> => {
  const response = await axiosInstance.get('/little-bull/admin/models', {
    params: { workspace_id: workspaceId }
  })
  return response.data.models
}

export const saveLittleBullAdminModel = async (
  workspaceId: string,
  model: LittleBullModelSetting
): Promise<LittleBullModelSetting> => {
  const response = await axiosInstance.post('/little-bull/admin/models', model, {
    params: { workspace_id: workspaceId }
  })
  return response.data
}

export const getLittleBullEmbeddingCatalog = async (): Promise<LittleBullEmbeddingCatalogItem[]> => {
  const response = await axiosInstance.get('/little-bull/admin/embedding-models')
  return response.data.models
}

export const getLittleBullKnowledgeBases = async (): Promise<LittleBullKnowledgeBase[]> => {
  const response = await axiosInstance.get('/little-bull/admin/knowledge-bases')
  return response.data.knowledge_bases
}

export const saveLittleBullKnowledgeBase = async (
  payload: LittleBullKnowledgeBaseUpsertRequest
): Promise<LittleBullKnowledgeBase> => {
  const response = await axiosInstance.post('/little-bull/admin/knowledge-bases', payload)
  return response.data
}

export const estimateLittleBullEmbeddingCost = async (
  payload: LittleBullEmbeddingCostEstimateRequest
): Promise<LittleBullEmbeddingCostEstimateResponse> => {
  const response = await axiosInstance.post('/little-bull/admin/embedding-cost-estimate', payload)
  return response.data
}

export const attachLittleBullKnowledgeBaseDataPlane = async (
  workspaceId: string
): Promise<LittleBullKnowledgeBaseAttachResponse> => {
  const response = await axiosInstance.post(`/little-bull/admin/knowledge-bases/${encodeURIComponent(workspaceId)}/attach-data-plane`)
  return response.data
}

export const reindexLittleBullKnowledgeBase = async (
  workspaceId: string,
  approvalId?: string | null,
  destructiveRebuild = false
): Promise<LittleBullKnowledgeBaseReindexResponse> => {
  const response = await axiosInstance.post(`/little-bull/admin/knowledge-bases/${encodeURIComponent(workspaceId)}/reindex`, {
    approval_id: approvalId ?? null,
    include_archived: true,
    include_input_root: true,
    destructive_rebuild: destructiveRebuild
  })
  return response.data
}

export const getLittleBullAdminAgents = async (
  workspaceId: string
): Promise<LittleBullAgentConfig[]> => {
  const response = await axiosInstance.get('/little-bull/admin/agents', {
    params: { workspace_id: workspaceId }
  })
  return response.data.agents
}

export const saveLittleBullAdminAgent = async (
  workspaceId: string,
  agent: LittleBullAgentConfig
): Promise<LittleBullAgentConfig> => {
  const response = await axiosInstance.post('/little-bull/admin/agents', agent, {
    params: { workspace_id: workspaceId }
  })
  return response.data
}

export const previewLittleBullAdminAgent = async (
  workspaceId: string,
  agent: LittleBullAgentConfig,
  testInput: string = ''
): Promise<LittleBullAgentStudioPreviewResponse> => {
  const response = await axiosInstance.post('/little-bull/admin/agents/preview', {
    workspace_id: workspaceId,
    agent,
    test_input: testInput
  })
  return response.data
}

export const getLittleBullConversations = async (
  workspaceId: string
): Promise<LittleBullConversation[]> => {
  const response = await axiosInstance.get('/little-bull/conversations', {
    params: { workspace_id: workspaceId }
  })
  return response.data.conversations
}

export const saveLittleBullConversation = async (
  conversation: Omit<LittleBullConversation, 'conversation_id' | 'user_id' | 'message_count'> & {
    conversation_id?: string | null
  }
): Promise<LittleBullConversation> => {
  const response = await axiosInstance.post('/little-bull/conversations', conversation)
  return response.data
}

export const exportLittleBullConversation = async (
  conversationId: string,
  format: 'md' | 'txt' | 'docx'
): Promise<Blob> => {
  const response = await axiosInstance.get(`/little-bull/conversations/${encodeURIComponent(conversationId)}/export`, {
    params: { format },
    responseType: 'blob'
  })
  return response.data
}

export const getLittleBullCorrelationSuggestions = async (
  workspaceId: string
): Promise<LittleBullCorrelationSuggestion[]> => {
  const response = await axiosInstance.get('/little-bull/correlation-suggestions', {
    params: { workspace_id: workspaceId }
  })
  return response.data.suggestions
}

export const createLittleBullCorrelationSuggestion = async (
  payload: Pick<LittleBullCorrelationSuggestion, 'workspace_id' | 'source_label' | 'target_label' | 'reason'> & {
    metadata?: Record<string, any>
  }
): Promise<LittleBullCorrelationSuggestion> => {
  const response = await axiosInstance.post('/little-bull/correlation-suggestions', payload)
  return response.data
}

export const decideLittleBullCorrelationSuggestion = async (
  suggestionId: string,
  decision: 'approve' | 'reject'
): Promise<LittleBullCorrelationSuggestion> => {
  const response = await axiosInstance.post(
    `/little-bull/correlation-suggestions/${encodeURIComponent(suggestionId)}/${decision}`
  )
  return response.data
}

export const getLittleBullApprovals = async (): Promise<LittleBullApproval[]> => {
  const response = await axiosInstance.get('/approvals')
  return response.data.approvals
}

export const approveLittleBullApproval = async (
  approvalId: string
): Promise<LittleBullApproval> => {
  const response = await axiosInstance.post(`/approvals/${encodeURIComponent(approvalId)}/approve`)
  return response.data
}

export const rejectLittleBullApproval = async (
  approvalId: string
): Promise<LittleBullApproval> => {
  const response = await axiosInstance.post(`/approvals/${encodeURIComponent(approvalId)}/reject`)
  return response.data
}

export const getLittleBullAuditEvents = async (
  limit: number = 100
): Promise<LittleBullAuditEvent[]> => {
  const response = await axiosInstance.get('/audit/events', { params: { limit } })
  return response.data.events
}

export const getLittleBullCostSummary = async (
  workspaceId: string
): Promise<LittleBullCostSummaryResponse> => {
  const response = await axiosInstance.get('/little-bull/costs/summary', {
    params: { workspace_id: workspaceId }
  })
  return response.data
}

export const getLittleBullDossiers = async (
  workspaceId: string
): Promise<LittleBullKnowledgeDossier[]> => {
  const response = await axiosInstance.get('/little-bull/dossiers', {
    params: { workspace_id: workspaceId }
  })
  return response.data.dossiers
}

export const exportLittleBullDossier = async (
  workspaceId: string,
  dossierId: string,
  payload: {
    format: 'txt' | 'md' | 'docx' | 'xlsx'
    destination: 'internal' | 'external'
    approval_id?: string | null
    include_audit?: boolean
  }
): Promise<Blob | { status: 'pending_approval'; message: string; approval: LittleBullApproval }> => {
  const response = await axiosInstance.post(
    `/little-bull/dossiers/${encodeURIComponent(dossierId)}/export`,
    payload,
    {
      params: { workspace_id: workspaceId },
      responseType: payload.destination === 'internal' || payload.approval_id ? 'blob' : 'json'
    }
  )
  return response.data
}

export const getLittleBullLegalExtractions = async (
  workspaceId: string
): Promise<LittleBullLegalMatterExtractionRun[]> => {
  const response = await axiosInstance.get('/little-bull/legal/extractions', {
    params: { workspace_id: workspaceId }
  })
  return response.data.runs
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
  allowRename: boolean = false,
  allowMerge: boolean = false
): Promise<EntityUpdateResponse> => {
  const response = await axiosInstance.post('/graph/entity/edit', {
    entity_name: entityName,
    updated_data: updatedData,
    allow_rename: allowRename,
    allow_merge: allowMerge
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
    updated_data: updatedData
  })
  return response.data
}

/**
 * Checks if an entity name already exists in the knowledge graph
 * @param entityName The entity name to check
 * @returns Promise with boolean indicating if the entity exists
 */
export const checkEntityNameExists = async (entityName: string): Promise<boolean> => {
  try {
    const response = await axiosInstance.get(`/graph/entity/exists?name=${encodeURIComponent(entityName)}`)
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

type InFlightPaginatedDocumentRequest = {
  controller: AbortController
  promise: Promise<PaginatedDocsResponse>
  subscriberCount: number
}

const getPaginatedDocumentsRequestKey = (request: DocumentsRequest): string =>
  JSON.stringify(request)

// Deduplicate in-flight paginated document requests with identical parameters.
// This prevents duplicate backend calls caused by overlapping timers/effects or
// React StrictMode double-mount behavior in development.
const inFlightPaginatedDocumentRequests = new Map<
  string,
  InFlightPaginatedDocumentRequest
>()

const releasePaginatedDocumentSubscriber = (
  requestKey: string,
  requestEntry: InFlightPaginatedDocumentRequest,
  abortIfLastSubscriber: boolean
): void => {
  requestEntry.subscriberCount = Math.max(0, requestEntry.subscriberCount - 1)

  if (requestEntry.subscriberCount !== 0) {
    return
  }

  if (inFlightPaginatedDocumentRequests.get(requestKey) === requestEntry) {
    inFlightPaginatedDocumentRequests.delete(requestKey)
  }

  if (abortIfLastSubscriber) {
    requestEntry.controller.abort()
  }
}

const subscribeToPaginatedDocumentsRequest = (
  request: DocumentsRequest
): {
  requestKey: string
  requestEntry: InFlightPaginatedDocumentRequest
  release: (abortIfLastSubscriber: boolean) => void
} => {
  const requestKey = getPaginatedDocumentsRequestKey(request)
  let requestEntry = inFlightPaginatedDocumentRequests.get(requestKey)

  if (!requestEntry) {
    const controller = new AbortController()
    requestEntry = {
      controller,
      subscriberCount: 0,
      promise: paginatedDocumentsPost(request, controller)
        .finally(() => {
          if (inFlightPaginatedDocumentRequests.get(requestKey) === requestEntry) {
            inFlightPaginatedDocumentRequests.delete(requestKey)
          }
        })
    }
    inFlightPaginatedDocumentRequests.set(requestKey, requestEntry)
  }

  requestEntry.subscriberCount += 1

  let released = false
  const release = (abortIfLastSubscriber: boolean): void => {
    if (released) {
      return
    }
    released = true
    releasePaginatedDocumentSubscriber(
      requestKey,
      requestEntry,
      abortIfLastSubscriber
    )
  }

  return {
    requestKey,
    requestEntry,
    release
  }
}

const defaultPaginatedDocumentsPost = async (
  request: DocumentsRequest,
  controller: AbortController
): Promise<PaginatedDocsResponse> => {
  const response = await axiosInstance.post('/documents/paginated', request, {
    signal: controller.signal
  })
  return response.data
}

let paginatedDocumentsPost = defaultPaginatedDocumentsPost

export const abortDocumentsPaginated = (request: DocumentsRequest): void => {
  const requestKey = getPaginatedDocumentsRequestKey(request)
  const inFlightRequest = inFlightPaginatedDocumentRequests.get(requestKey)

  if (!inFlightRequest) {
    return
  }

  inFlightPaginatedDocumentRequests.delete(requestKey)
  inFlightRequest.controller.abort()
}

export const __resetPaginatedDocumentRequestsForTests = (): void => {
  for (const { controller } of inFlightPaginatedDocumentRequests.values()) {
    controller.abort()
  }
  inFlightPaginatedDocumentRequests.clear()
  paginatedDocumentsPost = defaultPaginatedDocumentsPost
  littleBullUploadPost = defaultLittleBullUploadPost
}

export const __setPaginatedDocumentsPostForTests = (
  post: typeof defaultPaginatedDocumentsPost
): void => {
  paginatedDocumentsPost = post
}

export const __setLittleBullUploadPostForTests = (
  post: typeof defaultLittleBullUploadPost
): void => {
  littleBullUploadPost = post
}

/**
 * Get documents with pagination support
 * @param request The pagination request parameters
 * @returns Promise with paginated documents response
 */
export const getDocumentsPaginated = async (request: DocumentsRequest): Promise<PaginatedDocsResponse> => {
  const { requestEntry, release } = subscribeToPaginatedDocumentsRequest(request)

  try {
    return await requestEntry.promise
  } finally {
    release(false)
  }
}

export const getDocumentsPaginatedWithTimeout = (
  request: DocumentsRequest,
  timeoutMs: number = 30000,
  errorMsg: string = 'Document fetch timeout'
): Promise<PaginatedDocsResponse> => {
  const { requestEntry, release } = subscribeToPaginatedDocumentsRequest(request)

  return new Promise<PaginatedDocsResponse>((resolve, reject) => {
    let timedOut = false
    const timeoutId = setTimeout(() => {
      timedOut = true
      release(true)
      reject(new Error(errorMsg))
    }, timeoutMs)

    requestEntry.promise
      .then(response => {
        if (timedOut) {
          return
        }
        clearTimeout(timeoutId)
        release(false)
        resolve(response)
      })
      .catch(error => {
        if (timedOut) {
          return
        }
        clearTimeout(timeoutId)
        release(false)
        reject(error)
      })
  })
}

/**
 * Get counts of documents by status
 * @returns Promise with status counts response
 */
export const getDocumentStatusCounts = async (): Promise<StatusCountsResponse> => {
  const response = await axiosInstance.get('/documents/status_counts')
  return response.data
}
