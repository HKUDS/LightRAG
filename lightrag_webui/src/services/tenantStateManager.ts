/**
 * Tenant State Manager
 * 
 * Centralized module for managing tenant+route state handling.
 * Provides state persistence across tenant switches and URL synchronization.
 * 
 * Security: Tenant IDs are NEVER exposed in URLs. They are stored in
 * sessionStorage with tenant-scoped keys and provided via X-Tenant-ID header.
 */

import { debounce } from './debounce'

// Types for route state
export interface RouteState {
  currentKB?: string
  page?: number
  pageSize?: number
  filters?: Record<string, string>
  sort?: string
  sortDirection?: 'asc' | 'desc'
  viewMode?: 'list' | 'card' | 'graph'
  query?: string
  // Tab-specific state
  [key: string]: any
}

export interface TenantRouteKey {
  tenantId: string
  routeName: string
}

// Route names that support state persistence
export type RouteName = 'documents' | 'knowledge-graph' | 'retrieval' | 'api' | 'chat'

// Session storage key prefix
const STORAGE_PREFIX = 'lightrag:tenant'
const STATE_VERSION = '1.0'

// Default state for each route
const DEFAULT_ROUTE_STATE: Record<RouteName, RouteState> = {
  documents: {
    page: 1,
    pageSize: 10,
    filters: {},
    sort: 'updated_at',
    sortDirection: 'desc',
    viewMode: 'list',
  },
  'knowledge-graph': {
    viewMode: 'graph',
    filters: {},
    query: '',
  },
  retrieval: {
    page: 1,
    pageSize: 20,
    query: '',
  },
  api: {},
  chat: {
    page: 1,
    pageSize: 50,
  },
}

/**
 * Generate storage key for tenant+route state
 */
function getStorageKey(tenantId: string, routeName: string): string {
  return `${STORAGE_PREFIX}:${tenantId}:route:${routeName}`
}

/**
 * Parse URL query parameters into RouteState
 */
function parseURLParams(): Partial<RouteState> {
  const params = new URLSearchParams(window.location.search)
  const state: Partial<RouteState> = {}

  // Parse standard params
  const kb = params.get('kb')
  if (kb) state.currentKB = kb

  const page = params.get('page')
  if (page) state.page = parseInt(page, 10)

  const pageSize = params.get('pageSize')
  if (pageSize) state.pageSize = parseInt(pageSize, 10)

  const sort = params.get('sort')
  if (sort) state.sort = sort

  const sortDirection = params.get('sortDirection')
  if (sortDirection === 'asc' || sortDirection === 'desc') {
    state.sortDirection = sortDirection
  }

  const viewMode = params.get('view') || params.get('viewMode')
  if (viewMode === 'list' || viewMode === 'card' || viewMode === 'graph') {
    state.viewMode = viewMode
  }

  const query = params.get('query') || params.get('q')
  if (query) state.query = query

  // Parse filters (format: filters=key1:value1,key2:value2)
  const filtersParam = params.get('filters')
  if (filtersParam) {
    const filters: Record<string, string> = {}
    filtersParam.split(',').forEach(pair => {
      const [key, value] = pair.split(':')
      if (key && value) {
        filters[key] = value
      }
    })
    if (Object.keys(filters).length > 0) {
      state.filters = filters
    }
  }

  return state
}

/**
 * Serialize RouteState to URL query parameters
 */
function serializeToURLParams(state: RouteState): URLSearchParams {
  const params = new URLSearchParams()

  if (state.currentKB) params.set('kb', state.currentKB)
  if (state.page && state.page !== 1) params.set('page', state.page.toString())
  if (state.pageSize && state.pageSize !== 10) params.set('pageSize', state.pageSize.toString())
  if (state.sort) params.set('sort', state.sort)
  if (state.sortDirection) params.set('sortDirection', state.sortDirection)
  if (state.viewMode && state.viewMode !== 'list') params.set('view', state.viewMode)
  if (state.query) params.set('q', state.query)

  // Serialize filters
  if (state.filters && Object.keys(state.filters).length > 0) {
    const filterStr = Object.entries(state.filters)
      .filter(([_, v]) => v)
      .map(([k, v]) => `${k}:${v}`)
      .join(',')
    if (filterStr) params.set('filters', filterStr)
  }

  return params
}

/**
 * TenantStateManager class
 * 
 * Manages tenant+route state with:
 * - URL synchronization (tenant-agnostic)
 * - sessionStorage persistence (tenant-scoped)
 * - In-memory caching for fast access
 */
class TenantStateManager {
  private memoryCache: Map<string, RouteState> = new Map()
  private listeners: Map<string, Set<(state: RouteState) => void>> = new Map()
  private currentTenantId: string | null = null

  constructor() {
    // Listen for popstate events to handle browser back/forward
    if (typeof window !== 'undefined') {
      window.addEventListener('popstate', () => {
        this.hydrateFromURL()
      })
    }
  }

  /**
   * Set the current tenant ID (from X-Tenant-ID header/localStorage)
   */
  setCurrentTenant(tenantId: string | null): void {
    const oldTenantId = this.currentTenantId
    this.currentTenantId = tenantId
    
    if (oldTenantId && tenantId && oldTenantId !== tenantId) {
      this.onTenantSwitch(oldTenantId, tenantId)
    }
  }

  /**
   * Get the current tenant ID
   */
  getCurrentTenant(): string | null {
    return this.currentTenantId
  }

  /**
   * Get state for a specific tenant and route
   */
  getState(tenantId: string, routeName: RouteName): RouteState {
    const key = getStorageKey(tenantId, routeName)

    // Check memory cache first
    if (this.memoryCache.has(key)) {
      return this.memoryCache.get(key)!
    }

    // Try sessionStorage
    try {
      const stored = sessionStorage.getItem(key)
      if (stored) {
        const parsed = JSON.parse(stored)
        if (parsed.version === STATE_VERSION) {
          const state = parsed.state as RouteState
          this.memoryCache.set(key, state)
          return state
        }
      }
    } catch (e) {
      console.warn('[TenantStateManager] Failed to parse stored state:', e)
    }

    // Return default state
    const defaultState = { ...DEFAULT_ROUTE_STATE[routeName] }
    this.memoryCache.set(key, defaultState)
    return defaultState
  }

  /**
   * Set state for a specific tenant and route
   */
  setState(tenantId: string, routeName: RouteName, state: Partial<RouteState>): void {
    const key = getStorageKey(tenantId, routeName)
    const currentState = this.getState(tenantId, routeName)
    const newState = { ...currentState, ...state }

    // Update memory cache
    this.memoryCache.set(key, newState)

    // Persist to sessionStorage
    try {
      sessionStorage.setItem(key, JSON.stringify({
        version: STATE_VERSION,
        state: newState,
        updatedAt: new Date().toISOString(),
      }))
    } catch (e) {
      console.warn('[TenantStateManager] Failed to persist state:', e)
    }

    // Notify listeners
    this.notifyListeners(key, newState)
  }

  /**
   * Hydrate state from URL parameters
   * URL is tenant-agnostic, so this only restores UI state
   */
  hydrateFromURL(): Partial<RouteState> {
    return parseURLParams()
  }

  /**
   * Sync current route state to URL
   * URL remains tenant-agnostic (no tenant ID in URL)
   */
  syncToURL = debounce((routeName: RouteName, state: RouteState): void => {
    const params = serializeToURLParams(state)
    const newURL = params.toString() 
      ? `${window.location.pathname}?${params.toString()}${window.location.hash}`
      : `${window.location.pathname}${window.location.hash}`
    
    // Use replaceState to avoid polluting browser history for filter/sort changes
    window.history.replaceState({ routeName, state }, '', newURL)
  }, 300)

  /**
   * Push state to URL (creates new history entry)
   * Use for explicit user navigation actions
   */
  pushToURL(routeName: RouteName, state: RouteState): void {
    const params = serializeToURLParams(state)
    const newURL = params.toString() 
      ? `${window.location.pathname}?${params.toString()}${window.location.hash}`
      : `${window.location.pathname}${window.location.hash}`
    
    window.history.pushState({ routeName, state }, '', newURL)
  }

  /**
   * Handle tenant switch
   * Restores previously saved state for the new tenant
   */
  onTenantSwitch(oldTenantId: string, newTenantId: string): void {
    console.log(`[TenantStateManager] Tenant switch: ${oldTenantId} -> ${newTenantId}`)
    
    // Clear URL params when switching tenants (tenant-agnostic URLs)
    // The new tenant's state will be loaded from sessionStorage
    const hash = window.location.hash
    window.history.replaceState({}, '', `${window.location.pathname}${hash}`)

    // Trigger re-render by notifying all listeners for the new tenant
    for (const [key, listeners] of this.listeners.entries()) {
      if (key.includes(`:${newTenantId}:`)) {
        const routeName = key.split(':route:')[1] as RouteName
        const state = this.getState(newTenantId, routeName)
        listeners.forEach(listener => listener(state))
      }
    }
  }

  /**
   * Subscribe to state changes for a specific tenant+route
   */
  subscribe(tenantId: string, routeName: RouteName, listener: (state: RouteState) => void): () => void {
    const key = getStorageKey(tenantId, routeName)
    
    if (!this.listeners.has(key)) {
      this.listeners.set(key, new Set())
    }
    
    this.listeners.get(key)!.add(listener)

    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(key)
      if (listeners) {
        listeners.delete(listener)
        if (listeners.size === 0) {
          this.listeners.delete(key)
        }
      }
    }
  }

  /**
   * Notify all listeners for a specific key
   */
  private notifyListeners(key: string, state: RouteState): void {
    const listeners = this.listeners.get(key)
    if (listeners) {
      listeners.forEach(listener => listener(state))
    }
  }

  /**
   * Clear all state for a specific tenant
   */
  clearTenantState(tenantId: string): void {
    // Clear from memory cache
    for (const key of this.memoryCache.keys()) {
      if (key.includes(`:${tenantId}:`)) {
        this.memoryCache.delete(key)
      }
    }

    // Clear from sessionStorage
    const keysToRemove: string[] = []
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i)
      if (key && key.includes(`:${tenantId}:`)) {
        keysToRemove.push(key)
      }
    }
    keysToRemove.forEach(key => sessionStorage.removeItem(key))
  }

  /**
   * Get last selected KB for a tenant
   */
  getLastSelectedKB(tenantId: string): string | null {
    try {
      const key = `${STORAGE_PREFIX}:${tenantId}:lastKB`
      return sessionStorage.getItem(key)
    } catch {
      return null
    }
  }

  /**
   * Set last selected KB for a tenant
   */
  setLastSelectedKB(tenantId: string, kbId: string): void {
    try {
      const key = `${STORAGE_PREFIX}:${tenantId}:lastKB`
      sessionStorage.setItem(key, kbId)
    } catch (e) {
      console.warn('[TenantStateManager] Failed to save last KB:', e)
    }
  }

  /**
   * Merge URL state with stored state
   * URL parameters take precedence over stored state
   */
  mergeWithURL(tenantId: string, routeName: RouteName): RouteState {
    const storedState = this.getState(tenantId, routeName)
    const urlState = this.hydrateFromURL()
    return { ...storedState, ...urlState }
  }

  /**
   * Reset state for a route to defaults
   */
  resetRouteState(tenantId: string, routeName: RouteName): void {
    const defaultState = { ...DEFAULT_ROUTE_STATE[routeName] }
    this.setState(tenantId, routeName, defaultState)
  }
}

// Singleton instance
export const tenantStateManager = new TenantStateManager()

// Export default for convenience
export default tenantStateManager
