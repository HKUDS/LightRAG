import { useEffect, useRef, useCallback } from 'react'
import { useTenantState } from '@/stores/tenant'
import { fetchTenants, fetchKnowledgeBases } from '@/api/tenant'
import type { Tenant, KnowledgeBase } from '@/stores/tenant'

/**
 * Hook to auto-initialize tenant and KB on first app load
 * Ensures that documents are visible even on initial page load
 * This solves the "empty state on refresh" issue by automatically
 * selecting the first available tenant and KB if none are currently selected
 */
export function useTenantInitialization() {
  const selectedKB = useTenantState.use.selectedKB()
  const setSelectedTenant = useTenantState.use.setSelectedTenant()
  const setSelectedKB = useTenantState.use.setSelectedKB()
  const setTenants = useTenantState.use.setTenants()
  const setKnowledgeBases = useTenantState.use.setKnowledgeBases()
  const setLoading = useTenantState.use.setLoading()
  const setError = useTenantState.use.setError()

  const initializationAttempted = useRef(false)

  const initializeTenantContext = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)

      // Fetch list of available tenants
      console.log('[TenantInit] Fetching available tenants...')
      const availableTenants: Tenant[] = await fetchTenants()

      if (availableTenants.length === 0) {
        console.warn('[TenantInit] No tenants available')
        setError('No tenants available')
        setLoading(false)
        return
      }

      console.log(
        '[TenantInit] Available tenants:',
        availableTenants.map((t) => t.tenant_id)
      )
      setTenants(availableTenants)

      // Determine which tenant to use (existing or first available)
      // We re-read from store/localStorage here to ensure we have the latest value
      // in case it was updated during the async fetch
      let currentTenant = useTenantState.getState().selectedTenant
      
      // Fallback: Check localStorage directly if store is empty
      if (!currentTenant) {
        try {
          const stored = localStorage.getItem('SELECTED_TENANT')
          if (stored) {
            currentTenant = JSON.parse(stored)
            console.log('[TenantInit] Recovered tenant from localStorage:', currentTenant?.tenant_id)
            // Sync back to store
            if (currentTenant) {
              setSelectedTenant(currentTenant)
            }
          }
        } catch (e) {
          console.error('[TenantInit] Failed to parse localStorage fallback', e)
        }
      }

      console.log('[TenantInit] Current tenant resolved to:', currentTenant?.tenant_id)
      
      if (!currentTenant) {
        const firstTenant = availableTenants[0]
        console.log('[TenantInit] Auto-selecting first tenant:', firstTenant.tenant_id)
        setSelectedTenant(firstTenant)
        currentTenant = firstTenant
      } else {
        console.log('[TenantInit] Using existing tenant:', currentTenant.tenant_id)
      }

      // Ensure KB is selected for the current tenant
      if (currentTenant && !selectedKB) {
        try {
          console.log(
            '[TenantInit] Fetching knowledge bases for tenant:',
            currentTenant.tenant_id
          )
          const availableKBs: KnowledgeBase[] = await fetchKnowledgeBases(
            currentTenant.tenant_id
          )
          console.log('[TenantInit] Available KBs:', availableKBs.map((kb) => kb.kb_id))
          setKnowledgeBases(availableKBs)

          // Auto-select first KB
          if (availableKBs.length > 0) {
            const firstKB = availableKBs[0]
            console.log('[TenantInit] Auto-selecting first KB:', firstKB.kb_id)
            setSelectedKB(firstKB)
            console.log('[TenantInit] Initialization complete! Tenant and KB selected.')
          } else {
            console.warn('[TenantInit] No knowledge bases available for tenant')
            setError('No knowledge bases available')
          }
        } catch (error) {
          console.error('[TenantInit] Failed to fetch KBs:', error)
          setError('Failed to fetch knowledge bases')
        }
      }

      setLoading(false)
    } catch (error) {
      console.error('[TenantInit] Failed to initialize tenant context:', error)
      setError(error instanceof Error ? error.message : 'Failed to initialize tenant context')
      setLoading(false)
    }
  }, [setError, setKnowledgeBases, setLoading, setSelectedKB, setSelectedTenant, setTenants])

  useEffect(() => {
    // Prevent double initialization in strict mode
    if (initializationAttempted.current) {
      return
    }

    // Check current state directly from store to avoid stale closure issues
    const currentState = useTenantState.getState()
    if (currentState.selectedTenant && currentState.selectedKB) {
      console.log('[TenantInit] Tenant and KB already selected, skipping auto-init', {
        tenant: currentState.selectedTenant.tenant_id,
        kb: currentState.selectedKB.kb_id,
      })
      return
    }

    initializationAttempted.current = true
    initializeTenantContext()
  }, [initializeTenantContext])
}
