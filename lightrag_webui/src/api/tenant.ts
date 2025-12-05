import { axiosInstance as apiClient } from './client'
import { Tenant, KnowledgeBase } from '@/stores/tenant'

// Pagination response type
export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
  total_pages: number
  has_next: boolean
  has_prev: boolean
}

// Tenants API

/**
 * Fetch paginated list of tenants
 * @param page - Page number (1-indexed)
 * @param pageSize - Items per page (max 100)
 * @param search - Optional search string
 */
export async function fetchTenantsPaginated(
  page: number = 1,
  pageSize: number = 10,
  search?: string
): Promise<PaginatedResponse<Tenant>> {
  try {
    const params: Record<string, any> = { page, page_size: pageSize }
    if (search) params.search = search

    const response = await apiClient.get('/api/v1/tenants', { params })
    return response.data
  } catch (error) {
    console.error('Failed to fetch tenants paginated:', error)
    // WUI-002 FIX: Throw error instead of returning fake data
    // This allows the UI to show proper error state
    throw error
  }
}

/**
 * Fetch all tenants (deprecated - use fetchTenantsPaginated instead)
 * @deprecated Use fetchTenantsPaginated for better performance
 */
export async function fetchTenants(): Promise<Tenant[]> {
  try {
    const response = await apiClient.get('/api/v1/tenants')
    // Handle both paginated response format and legacy array format
    const data = response.data
    if (Array.isArray(data)) {
      return data
    }
    // New paginated format returns { items: [...], total: N, ... }
    return data?.items || []
  } catch (error) {
    console.error('Failed to fetch tenants:', error)
    // WUI-002 FIX: Throw error instead of returning fake data
    // This allows the UI to show proper error state
    throw error
  }
}

export async function fetchCurrentTenant(tenantId: string): Promise<Tenant | null> {
  try {
    const response = await apiClient.get('/api/v1/tenants/me', {
      headers: { 'X-Tenant-ID': tenantId }
    })
    return response.data
  } catch (error) {
    console.error(`Failed to fetch current tenant ${tenantId}:`, error)
    return null
  }
}

export async function createTenant(
  name: string,
  description?: string,
  metadata?: Record<string, any>
): Promise<Tenant> {
  const response = await apiClient.post('/api/v1/tenants', {
    name,
    description,
    metadata,
  })
  return response.data
}

// Knowledge Bases API

/**
 * Fetch paginated list of knowledge bases for a tenant
 * @param tenantId - The tenant ID
 * @param page - Page number (1-indexed)
 * @param pageSize - Items per page (max 100)
 * @param search - Optional search string
 */
export async function fetchKnowledgeBasesPaginated(
  tenantId: string,
  page: number = 1,
  pageSize: number = 10,
  search?: string
): Promise<PaginatedResponse<KnowledgeBase>> {
  try {
    const params: Record<string, any> = { page, page_size: pageSize }
    if (search) params.search = search

    const response = await apiClient.get('/api/v1/knowledge-bases', {
      params,
      headers: { 'X-Tenant-ID': tenantId }
    })
    return response.data
  } catch (error) {
    console.error(`Failed to fetch knowledge bases paginated for tenant ${tenantId}:`, error)
    // Return default paginated response if API not available
    return {
      items: [
        {
          kb_id: 'default',
          tenant_id: tenantId,
          kb_name: 'Default KB',
          description: 'Default knowledge base',
          num_documents: 0,
          num_entities: 0,
          num_relations: 0,
        }
      ],
      total: 1,
      page: 1,
      page_size: pageSize,
      total_pages: 1,
      has_next: false,
      has_prev: false
    }
  }
}

/**
 * Fetch all knowledge bases for a tenant (deprecated - use fetchKnowledgeBasesPaginated instead)
 * @deprecated Use fetchKnowledgeBasesPaginated for better performance
 */
export async function fetchKnowledgeBases(tenantId: string): Promise<KnowledgeBase[]> {
  try {
    const response = await apiClient.get('/api/v1/knowledge-bases', {
      headers: { 'X-Tenant-ID': tenantId }
    })
    return response.data?.items || []
  } catch (error) {
    console.error(`Failed to fetch knowledge bases for tenant ${tenantId}:`, error)
    // Return default KB if API is not multi-tenant enabled
    return [
      {
        kb_id: 'default',
        tenant_id: tenantId,
        kb_name: 'Default KB',
        description: 'Default knowledge base',
        num_documents: 0,
        num_entities: 0,
        num_relations: 0,
      }
    ]
  }
}

export async function fetchKnowledgeBase(
  tenantId: string,
  kbId: string
): Promise<KnowledgeBase | null> {
  try {
    const response = await apiClient.get(`/api/v1/knowledge-bases/${kbId}`, {
      headers: { 'X-Tenant-ID': tenantId }
    })
    return response.data
  } catch (error) {
    console.error(`Failed to fetch KB ${kbId}:`, error)
    return null
  }
}

export async function createKnowledgeBase(
  tenantId: string,
  name: string,
  description?: string,
  metadata?: Record<string, any>
): Promise<KnowledgeBase> {
  const response = await apiClient.post(
    '/api/v1/knowledge-bases',
    {
      name,
      description,
      metadata,
    },
    {
      headers: { 'X-Tenant-ID': tenantId }
    }
  )
  return response.data
}

export async function updateKnowledgeBase(
  tenantId: string,
  kbId: string,
  updates: Partial<KnowledgeBase>
): Promise<KnowledgeBase> {
  const response = await apiClient.put(
    `/api/v1/knowledge-bases/${kbId}`,
    updates,
    {
      headers: { 'X-Tenant-ID': tenantId }
    }
  )
  return response.data
}

export async function deleteKnowledgeBase(tenantId: string, kbId: string): Promise<void> {
  await apiClient.delete(`/api/v1/knowledge-bases/${kbId}`, {
    headers: { 'X-Tenant-ID': tenantId }
  })
}
