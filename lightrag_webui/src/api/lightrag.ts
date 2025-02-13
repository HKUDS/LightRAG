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

export const InvalidApiKeyError = 'Invalid API Key'
export const RequireApiKeError = 'API Key required'

// Helper functions
const getResponseContent = async (response: Response) => {
  const contentType = response.headers.get('content-type')
  if (contentType) {
    if (contentType.includes('application/json')) {
      const data = await response.json()
      return JSON.stringify(data, undefined, 2)
    } else if (contentType.startsWith('text/')) {
      return await response.text()
    } else if (contentType.includes('application/xml') || contentType.includes('text/xml')) {
      return await response.text()
    } else if (contentType.includes('application/octet-stream')) {
      const buffer = await response.arrayBuffer()
      const decoder = new TextDecoder('utf-8', { fatal: false, ignoreBOM: true })
      return decoder.decode(buffer)
    } else {
      try {
        return await response.text()
      } catch (error) {
        console.warn('Failed to decode as text, may be binary:', error)
        return `[Could not decode response body. Content-Type: ${contentType}]`
      }
    }
  } else {
    try {
      return await response.text()
    } catch (error) {
      console.warn('Failed to decode as text, may be binary:', error)
      return '[Could not decode response body. No Content-Type header.]'
    }
  }
  return ''
}

const fetchWithAuth = async (url: string, options: RequestInit = {}): Promise<Response> => {
  const apiKey = useSettingsStore.getState().apiKey
  const headers = {
    ...(options.headers || {}),
    ...(apiKey ? { 'X-API-Key': apiKey } : {})
  }

  const response = await fetch(backendBaseUrl + url, {
    ...options,
    headers
  })

  if (!response.ok) {
    throw new Error(
      `${response.status} ${response.statusText}\n${await getResponseContent(response)}\n${response.url}`
    )
  }

  return response
}

// API methods
export const queryGraphs = async (label: string): Promise<LightragGraphType> => {
  const response = await fetchWithAuth(`/graphs?label=${label}`)
  return await response.json()
}

export const getGraphLabels = async (): Promise<string[]> => {
  const response = await fetchWithAuth('/graph/label/list')
  return await response.json()
}

export const checkHealth = async (): Promise<
  LightragStatus | { status: 'error'; message: string }
> => {
  try {
    const response = await fetchWithAuth('/health')
    return await response.json()
  } catch (e) {
    return {
      status: 'error',
      message: errorMessage(e)
    }
  }
}

export const getDocuments = async (): Promise<string[]> => {
  const response = await fetchWithAuth('/documents')
  return await response.json()
}

export const getDocumentsScanProgress = async (): Promise<LightragDocumentsScanProgress> => {
  const response = await fetchWithAuth('/documents/scan-progress')
  return await response.json()
}

export const uploadDocument = async (
  file: File
): Promise<{
  status: string
  message: string
  total_documents: number
}> => {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetchWithAuth('/documents/upload', {
    method: 'POST',
    body: formData
  })
  return await response.json()
}

export const startDocumentScan = async (): Promise<{ status: string }> => {
  const response = await fetchWithAuth('/documents/scan', {
    method: 'POST'
  })
  return await response.json()
}

export const queryText = async (request: QueryRequest): Promise<QueryResponse> => {
  const response = await fetchWithAuth('/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(request)
  })
  return await response.json()
}

export const queryTextStream = async (request: QueryRequest, onChunk: (chunk: string) => void) => {
  const response = await fetchWithAuth('/query/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(request)
  })

  const reader = response.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    const chunk = decoder.decode(value)
    const lines = chunk.split('\n')
    for (const line of lines) {
      if (line) {
        try {
          const data = JSON.parse(line)
          if (data.response) {
            onChunk(data.response)
          }
        } catch (e) {
          console.error('Error parsing stream chunk:', e)
        }
      }
    }
  }
}

// Text insertion API
export const insertText = async (
  text: string,
  description?: string
): Promise<{
  status: string
  message: string
  document_count: number
}> => {
  const response = await fetchWithAuth('/documents/text', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ text, description })
  })
  return await response.json()
}

// Batch file upload API
export const uploadBatchDocuments = async (
  files: File[]
): Promise<{
  status: string
  message: string
  document_count: number
}> => {
  const formData = new FormData()
  files.forEach((file) => {
    formData.append('files', file)
  })

  const response = await fetchWithAuth('/documents/batch', {
    method: 'POST',
    body: formData
  })
  return await response.json()
}

// Clear all documents API
export const clearDocuments = async (): Promise<{
  status: string
  message: string
  document_count: number
}> => {
  const response = await fetchWithAuth('/documents', {
    method: 'DELETE'
  })
  return await response.json()
}
