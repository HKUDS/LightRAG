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
  indexed_files: string[]
  indexed_files_count: number
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

export type QueryMode = 'naive' | 'local' | 'global' | 'hybrid' | 'mix'

export type QueryRequest = {
  query: string
  mode: QueryMode
  stream?: boolean
  only_need_context?: boolean
}

export type QueryResponse = {
  response: string
}

export type DocumentActionResponse = {
  status: 'success' | 'partial_success' | 'failure'
  message: string
  document_count: number
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
export const queryGraphs = async (label: string): Promise<LightragGraphType> => {
  const response = await axiosInstance.get(`/graphs?label=${label}`)
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

export const getDocuments = async (): Promise<string[]> => {
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
    await axiosInstance.post('/query/stream', request, {
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

export const insertText = async (
  text: string,
  description?: string
): Promise<DocumentActionResponse> => {
  const response = await axiosInstance.post('/documents/text', { text, description })
  return response.data
}

export const uploadDocument = async (
  file: File,
  onUploadProgress?: (percentCompleted: number) => void
): Promise<DocumentActionResponse> => {
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
): Promise<DocumentActionResponse[]> => {
  return await Promise.all(
    files.map(async (file) => {
      return await uploadDocument(file, (percentCompleted) => {
        onUploadProgress?.(file.name, percentCompleted)
      })
    })
  )
}

export const clearDocuments = async (): Promise<DocumentActionResponse> => {
  const response = await axiosInstance.delete('/documents')
  return response.data
}
