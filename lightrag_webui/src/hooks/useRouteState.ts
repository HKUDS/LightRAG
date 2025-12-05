/**
 * React hook for tenant state management
 *
 * Provides easy access to tenantStateManager within React components
 * with automatic subscription and cleanup.
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import { useTenantState } from '@/stores/tenant'
import { tenantStateManager, RouteState, RouteName } from '@/services/tenantStateManager'

/**
 * Hook to get and manage route state for the current tenant
 *
 * @param routeName - The route name to manage state for
 * @returns Object with state, setState, and utility functions
 */
export function useRouteState(routeName: RouteName) {
  const selectedTenant = useTenantState.use.selectedTenant()
  const tenantId = selectedTenant?.tenant_id

  // Initialize state from manager
  const [routeState, setLocalState] = useState<RouteState>(() => {
    if (!tenantId) return {}
    return tenantStateManager.mergeWithURL(tenantId, routeName)
  })

  // Set current tenant in manager
  useEffect(() => {
    tenantStateManager.setCurrentTenant(tenantId || null)
  }, [tenantId])

  // Subscribe to state changes
  useEffect(() => {
    if (!tenantId) return

    // Initial hydration with URL
    const initialState = tenantStateManager.mergeWithURL(tenantId, routeName)
    setLocalState(initialState)

    // Subscribe to changes
    const unsubscribe = tenantStateManager.subscribe(tenantId, routeName, (newState) => {
      setLocalState(newState)
    })

    return unsubscribe
  }, [tenantId, routeName])

  // Update state and persist
  const setState = useCallback((updates: Partial<RouteState>) => {
    if (!tenantId) return

    tenantStateManager.setState(tenantId, routeName, updates)

    // Also sync to URL (debounced internally)
    const newState = tenantStateManager.getState(tenantId, routeName)
    tenantStateManager.syncToURL(routeName, newState)
  }, [tenantId, routeName])

  // Push state to URL (creates history entry)
  const pushState = useCallback((updates: Partial<RouteState>) => {
    if (!tenantId) return

    tenantStateManager.setState(tenantId, routeName, updates)
    const newState = tenantStateManager.getState(tenantId, routeName)
    tenantStateManager.pushToURL(routeName, newState)
  }, [tenantId, routeName])

  // Reset state to defaults
  const resetState = useCallback(() => {
    if (!tenantId) return
    tenantStateManager.resetRouteState(tenantId, routeName)
  }, [tenantId, routeName])

  // Get specific state values with type safety
  const page = routeState.page ?? 1
  const pageSize = routeState.pageSize ?? 10
  const sort = routeState.sort
  const sortDirection = routeState.sortDirection ?? 'desc'
  const filters = routeState.filters ?? {}
  const viewMode = routeState.viewMode ?? 'list'
  const query = routeState.query ?? ''
  const currentKB = routeState.currentKB

  return {
    state: routeState,
    setState,
    pushState,
    resetState,
    // Convenience accessors
    page,
    pageSize,
    sort,
    sortDirection,
    filters,
    viewMode,
    query,
    currentKB,
    // Convenience setters
    setPage: useCallback((p: number) => setState({ page: p }), [setState]),
    setPageSize: useCallback((ps: number) => setState({ pageSize: ps, page: 1 }), [setState]),
    setSort: useCallback((s: string, dir?: 'asc' | 'desc') =>
      setState({ sort: s, sortDirection: dir }), [setState]),
    setFilters: useCallback((f: Record<string, string>) =>
      setState({ filters: f, page: 1 }), [setState]),
    setViewMode: useCallback((vm: 'list' | 'card' | 'graph') =>
      setState({ viewMode: vm }), [setState]),
    setQuery: useCallback((q: string) => setState({ query: q }), [setState]),
    setCurrentKB: useCallback((kb: string) => {
      setState({ currentKB: kb })
      if (tenantId) {
        tenantStateManager.setLastSelectedKB(tenantId, kb)
      }
    }, [setState, tenantId]),
  }
}

/**
 * Hook for managing KB selection with state persistence
 */
export function useKBState() {
  const selectedTenant = useTenantState.use.selectedTenant()
  const selectedKB = useTenantState.use.selectedKB()
  const setSelectedKB = useTenantState.use.setSelectedKB()
  const tenantId = selectedTenant?.tenant_id

  // Get last selected KB for this tenant
  const lastKBId = useMemo(() => {
    if (!tenantId) return null
    return tenantStateManager.getLastSelectedKB(tenantId)
  }, [tenantId])

  // Select KB and persist
  const selectKB = useCallback((kb: { kb_id: string; [key: string]: any }) => {
    setSelectedKB(kb as any)
    if (tenantId) {
      tenantStateManager.setLastSelectedKB(tenantId, kb.kb_id)
    }
  }, [tenantId, setSelectedKB])

  return {
    selectedKB,
    selectKB,
    lastKBId,
    kbId: selectedKB?.kb_id || null,
  }
}

/**
 * Hook to build URL query string from current state
 * Useful for creating shareable links (tenant-agnostic)
 */
export function useShareableURL(routeName: RouteName) {
  const { state } = useRouteState(routeName)

  const url = useMemo(() => {
    const params = new URLSearchParams()

    if (state.currentKB) params.set('kb', state.currentKB)
    if (state.page && state.page !== 1) params.set('page', state.page.toString())
    if (state.pageSize && state.pageSize !== 10) params.set('pageSize', state.pageSize.toString())
    if (state.sort) params.set('sort', state.sort)
    if (state.sortDirection) params.set('sortDirection', state.sortDirection)
    if (state.viewMode && state.viewMode !== 'list') params.set('view', state.viewMode)
    if (state.query) params.set('q', state.query)

    if (state.filters && Object.keys(state.filters).length > 0) {
      const filterStr = Object.entries(state.filters)
        .filter(([_, v]) => v)
        .map(([k, v]) => `${k}:${v}`)
        .join(',')
      if (filterStr) params.set('filters', filterStr)
    }

    const queryString = params.toString()
    return queryString ? `${window.location.pathname}?${queryString}` : window.location.pathname
  }, [state])

  return url
}

export default useRouteState
