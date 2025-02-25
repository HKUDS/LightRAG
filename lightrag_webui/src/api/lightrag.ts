import axios, { AxiosError } from 'axios'
import { backendBaseUrl } from '@/lib/constants'
import { errorMessage } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'

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
    max_tokens: number
    kv_storage: string
    doc_status_storage: string
    graph_storage: string
    vector_storage: string
  }
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
 */
export type QueryMode = 'naive' | 'local' | 'global' | 'hybrid' | 'mix'

export type Message = {
  role: 'user' | 'assistant' | 'system'
  content: string
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
  /** Maximum number of tokens allowed for each retrieved text chunk. */
  max_token_for_text_unit?: number
  /** Maximum number of tokens allocated for relationship descriptions in global retrieval. */
  max_token_for_global_context?: number
  /** Maximum number of tokens allocated for entity descriptions in local retrieval. */
  max_token_for_local_context?: number
  /** List of high-level keywords to prioritize in retrieval. */
  hl_keywords?: string[]
  /** List of low-level keywords to refine retrieval focus. */
  ll_keywords?: string[]
  /**
   * Stores past conversation history to maintain context.
   * Format: [{"role": "user/assistant", "content": "message"}].
   */
  conversation_history?: Message[]
  /** Number of complete conversation turns (user-assistant pairs) to consider in the response context. */
  history_turns?: number
}

export type QueryResponse = {
  response: string
}

export type DocActionResponse = {
  status: 'success' | 'partial_success' | 'failure'
  message: string
}

export type DocStatus = 'pending' | 'processing' | 'processed' | 'failed'

export type DocStatusResponse = {
  id: string
  content_summary: string
  content_length: number
  status: DocStatus
  created_at: string
  updated_at: string
  chunks_count?: number
  error?: string
  metadata?: Record<string, any>
}

export type DocsStatusesResponse = {
  statuses: Record<DocStatus, DocStatusResponse[]>
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

// Interceptor：add api key
axiosInstance.interceptors.request.use((config) => {
  const apiKey = useSettingsStore.getState().apiKey
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
export const queryGraphs = async (label: string, maxDepth: number): Promise<LightragGraphType> => {
  const response = await axiosInstance.get(`/graphs?label=${label}&max_depth=${maxDepth}`)
  return response.data
}

export const getGraphLabels = async (): Promise<string[]> => {
  const response = await axiosInstance.get('/graph/label/list')
  return response.data
}

export const checkHealth = async (): Promise<
  LightragStatus | { status: 'error'; message: string }
> => {
  try {
    const response = await axiosInstance.get('/health')
    return response.data
  } catch (e) {
    return {
      status: 'error',
      message: errorMessage(e)
    }
  }
}

export const getDocuments = async (): Promise<DocsStatusesResponse> => {
  const response = await axiosInstance.get('/documents')
  return response.data
}

export const scanNewDocuments = async (): Promise<{ status: string }> => {
  const response = await axiosInstance.post('/documents/scan')
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
  try {
    let buffer = ''
    await axiosInstance
      .post('/query/stream', request, {
        responseType: 'text',
        headers: {
          Accept: 'application/x-ndjson'
        },
        transformResponse: [
          (data: string) => {
            // Accumulate the data and process complete lines
            buffer += data
            const lines = buffer.split('\n')
            // Keep the last potentially incomplete line in the buffer
            buffer = lines.pop() || ''

            for (const line of lines) {
              if (line.trim()) {
                try {
                  const parsed = JSON.parse(line)
                  if (parsed.response) {
                    onChunk(parsed.response)
                  } else if (parsed.error && onError) {
                    onError(parsed.error)
                  }
                } catch (e) {
                  console.error('Error parsing stream chunk:', e)
                  if (onError) onError('Error parsing server response')
                }
              }
            }
            return data
          }
        ]
      })
      .catch((error) => {
        if (onError) onError(errorMessage(error))
      })

    // Process any remaining data in the buffer
    if (buffer.trim()) {
      try {
        const parsed = JSON.parse(buffer)
        if (parsed.response) {
          onChunk(parsed.response)
        } else if (parsed.error && onError) {
          onError(parsed.error)
        }
      } catch (e) {
        console.error('Error parsing final chunk:', e)
        if (onError) onError('Error parsing server response')
      }
    }
  } catch (error) {
    const message = errorMessage(error)
    console.error('Stream request failed:', message)
    if (onError) onError(message)
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
