import { useTenantState } from '@/stores/tenant'

/**
 * Hook to get the currently selected tenant and knowledge base
 * This is used to inject tenant/KB context into API requests
 */
export function useTenantContext() {
  const selectedTenant = useTenantState.use.selectedTenant()
  const selectedKB = useTenantState.use.selectedKB()

  return {
    tenantId: selectedTenant?.tenant_id || null,
    kbId: selectedKB?.kb_id || null,
    tenant: selectedTenant,
    kb: selectedKB,
  }
}

/**
 * Get tenant context headers for API requests
 * These headers help the API identify which tenant/KB to operate on
 */
export function getTenantContextHeaders(tenantId?: string | null, kbId?: string | null) {
  const headers: Record<string, string> = {}

  if (tenantId) {
    headers['X-Tenant-ID'] = tenantId
  }

  if (kbId) {
    headers['X-KB-ID'] = kbId
  }

  return headers
}

/**
 * Build workspace name from tenant and KB
 * Format: {tenant_id}_{kb_id}_{workspace_name}
 */
export function buildWorkspaceName(tenantId: string, kbId: string, workspaceName?: string) {
  const base = `${tenantId}_${kbId}`
  return workspaceName ? `${base}_${workspaceName}` : base
}

/**
 * Hook to inject tenant context into document upload requests
 */
export function useDocumentUploadContext() {
  const { tenantId, kbId } = useTenantContext()

  const getUploadParams = () => {
    if (!tenantId || !kbId) {
      throw new Error('Tenant and KB must be selected before uploading documents')
    }

    return {
      tenantId,
      kbId,
      workspace: buildWorkspaceName(tenantId, kbId),
      headers: getTenantContextHeaders(tenantId, kbId),
    }
  }

  return {
    tenantId,
    kbId,
    getUploadParams,
    isReady: !!tenantId && !!kbId,
  }
}

/**
 * Hook to inject tenant context into query requests
 */
export function useQueryContext() {
  const { tenantId, kbId } = useTenantContext()

  const getQueryParams = () => {
    if (!tenantId || !kbId) {
      throw new Error('Tenant and KB must be selected before running queries')
    }

    return {
      tenantId,
      kbId,
      workspace: buildWorkspaceName(tenantId, kbId),
      headers: getTenantContextHeaders(tenantId, kbId),
    }
  }

  return {
    tenantId,
    kbId,
    getQueryParams,
    isReady: !!tenantId && !!kbId,
  }
}
