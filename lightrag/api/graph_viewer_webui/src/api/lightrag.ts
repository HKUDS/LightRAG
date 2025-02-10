import { backendBaseUrl } from '@/lib/constants'
import { errorMessage } from '@/lib/utils'

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

const checkResponse = (response: Response) => {
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} ${response.url}`)
  }
}

export const queryGraphs = async (label: string): Promise<LightragGraphType> => {
  const response = await fetch(backendBaseUrl + `/graphs?label=${label}`)
  checkResponse(response)
  return await response.json()
}

export const getGraphLabels = async (): Promise<string[]> => {
  const response = await fetch(backendBaseUrl + '/graph/label/list')
  checkResponse(response)
  return await response.json()
}

export const checkHealth = async (): Promise<
  LightragStatus | { status: 'error'; message: string }
> => {
  try {
    const response = await fetch(backendBaseUrl + '/health')
    if (!response.ok) {
      return {
        status: 'error',
        message: `Health check failed. Service is currently unavailable.\n${response.status} ${response.statusText} ${response.url}`
      }
    }
    return await response.json()
  } catch (e) {
    return {
      status: 'error',
      message: errorMessage(e)
    }
  }
}

export const getDocuments = async (): Promise<string[]> => {
  const response = await fetch(backendBaseUrl + '/documents')
  return await response.json()
}

export const getDocumentsScanProgress = async (): Promise<LightragDocumentsScanProgress> => {
  const response = await fetch(backendBaseUrl + '/documents/scan-progress')
  return await response.json()
}
