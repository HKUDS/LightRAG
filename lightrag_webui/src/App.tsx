import { useState, useCallback, useEffect, useRef } from 'react'
import ThemeProvider from '@/components/ThemeProvider'
import TabVisibilityProvider from '@/contexts/TabVisibilityProvider'
import ApiKeyAlert from '@/components/ApiKeyAlert'
import StatusIndicator from '@/components/status/StatusIndicator'
import { SiteInfo, webuiPrefix } from '@/lib/constants'
import { useBackendState, useAuthStore } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { useTenantState, Tenant } from '@/stores/tenant'
import { getAuthStatus } from '@/api/lightrag'
import SiteHeader from '@/features/SiteHeader'
import TenantSelectionPage from '@/features/TenantSelectionPage'
import { InvalidApiKeyError, RequireApiKeError } from '@/api/lightrag'
import { ZapIcon } from 'lucide-react'
import { useTenantInitialization } from '@/hooks/useTenantInitialization'

import GraphViewer from '@/features/GraphViewer'
import DocumentManager from '@/features/DocumentManager'
import RetrievalTesting from '@/features/RetrievalTesting'
import ApiSite from '@/features/ApiSite'

import { Tabs, TabsContent } from '@/components/ui/Tabs'

// Default tenant for single-tenant mode
const DEFAULT_SINGLE_TENANT: Tenant = {
  tenant_id: 'default',
  name: 'Default',
  description: 'Default single-tenant workspace',
}

// Default KB for single-tenant mode
const DEFAULT_SINGLE_KB = {
  kb_id: 'default',
  tenant_id: 'default',
  name: 'Default',
  description: 'Default knowledge base',
}

function App() {
  const message = useBackendState.use.message()
  const enableHealthCheck = useSettingsStore.use.enableHealthCheck()
  const currentTab = useSettingsStore.use.currentTab()
  const selectedTenant = useTenantState.use.selectedTenant()
  const setSelectedTenant = useTenantState.use.setSelectedTenant()
  const initializeTenantState = useTenantState.use.initializeFromStorage()
  const multiTenantEnabled = useAuthStore(state => state.multiTenantEnabled)
  
  // Auto-initialize tenant/KB on app load (fixes empty state on refresh)
  useTenantInitialization()
  
  const [apiKeyAlertOpen, setApiKeyAlertOpen] = useState(false)
  const [initializing, setInitializing] = useState(true) // Add initializing state
  const versionCheckRef = useRef(false); // Prevent duplicate calls in Vite dev mode
  const healthCheckInitializedRef = useRef(false); // Prevent duplicate health checks in Vite dev mode

  const handleApiKeyAlertOpenChange = useCallback((open: boolean) => {
    setApiKeyAlertOpen(open)
    if (!open) {
      useBackendState.getState().clear()
    }
  }, [])

  // Track component mount status with useRef
  const isMountedRef = useRef(true);

  // Set up mount/unmount status tracking
  useEffect(() => {
    isMountedRef.current = true;
    initializeTenantState();

    // Handle page reload/unload
    const handleBeforeUnload = () => {
      isMountedRef.current = false;
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      isMountedRef.current = false;
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  // Health check - can be disabled
  useEffect(() => {
    // Health check function
    const performHealthCheck = async () => {
      try {
        // Only perform health check if component is still mounted
        if (isMountedRef.current) {
          await useBackendState.getState().check();
        }
      } catch (error) {
        console.error('Health check error:', error);
      }
    };

    // Set health check function in the store
    useBackendState.getState().setHealthCheckFunction(performHealthCheck);

    if (!enableHealthCheck || apiKeyAlertOpen) {
      useBackendState.getState().clearHealthCheckTimer();
      return;
    }

    // On first mount or when enableHealthCheck becomes true and apiKeyAlertOpen is false,
    // perform an immediate health check and start the timer
    if (!healthCheckInitializedRef.current) {
      healthCheckInitializedRef.current = true;
    }

    // Start/reset the health check timer using the store
    useBackendState.getState().resetHealthCheckTimer();

    // Component unmount cleanup
    return () => {
      useBackendState.getState().clearHealthCheckTimer();
    };
  }, [enableHealthCheck, apiKeyAlertOpen]);

  // Version check - independent and executed only once
  useEffect(() => {
    const checkVersion = async () => {
      // Prevent duplicate calls in Vite dev mode
      if (versionCheckRef.current) return;
      versionCheckRef.current = true;

      // Check if version info was already obtained in login page
      const versionCheckedFromLogin = sessionStorage.getItem('VERSION_CHECKED_FROM_LOGIN') === 'true';
      if (versionCheckedFromLogin) {
        // Still need to check if we should auto-set default tenant/KB for single-tenant mode
        const multiTenant = localStorage.getItem('LIGHTRAG-MULTI-TENANT') === 'true';
        if (!multiTenant && !useTenantState.getState().selectedTenant) {
          useTenantState.getState().setSelectedTenant(DEFAULT_SINGLE_TENANT);
          useTenantState.getState().setSelectedKB(DEFAULT_SINGLE_KB);
        }
        setInitializing(false); // Skip initialization if already checked
        return;
      }

      try {
        setInitializing(true); // Start initialization

        // Get version info
        const token = localStorage.getItem('LIGHTRAG-API-TOKEN');
        const status = await getAuthStatus();
        const isMultiTenant = status.multi_tenant_enabled ?? false;

        // If auth is not configured and a new token is returned, use the new token
        if (!status.auth_configured && status.access_token) {
          useAuthStore.getState().login(
            status.access_token, // Use the new token
            true, // Guest mode
            status.core_version,
            status.api_version,
            status.webui_title || null,
            status.webui_description || null,
            isMultiTenant
          );
        } else if (token && (status.core_version || status.api_version || status.webui_title || status.webui_description)) {
          // Otherwise use the old token (if it exists)
          const isGuestMode = status.auth_mode === 'disabled' || useAuthStore.getState().isGuestMode;
          useAuthStore.getState().login(
            token,
            isGuestMode,
            status.core_version,
            status.api_version,
            status.webui_title || null,
            status.webui_description || null,
            isMultiTenant
          );
        } else {
          // Just update multi-tenant setting
          useAuthStore.getState().setMultiTenantEnabled(isMultiTenant);
        }

        // In single-tenant mode, auto-set the default tenant and KB if not already set
        if (!isMultiTenant && !useTenantState.getState().selectedTenant) {
          console.log('[App] Single-tenant mode detected, setting default tenant and KB');
          useTenantState.getState().setSelectedTenant(DEFAULT_SINGLE_TENANT);
          useTenantState.getState().setSelectedKB(DEFAULT_SINGLE_KB);
        }

        // Set flag to indicate version info has been checked
        sessionStorage.setItem('VERSION_CHECKED_FROM_LOGIN', 'true');
      } catch (error) {
        console.error('Failed to get version info:', error);
        // On error, assume single-tenant mode and set default tenant and KB
        if (!useTenantState.getState().selectedTenant) {
          console.log('[App] Error fetching auth status, defaulting to single-tenant mode');
          useTenantState.getState().setSelectedTenant(DEFAULT_SINGLE_TENANT);
          useTenantState.getState().setSelectedKB(DEFAULT_SINGLE_KB);
        }
      } finally {
        // Ensure initializing is set to false even if there's an error
        setInitializing(false);
      }
    };

    // Execute version check
    checkVersion();
  }, []); // Empty dependency array ensures it only runs once on mount

  const handleTabChange = useCallback(
    (tab: string) => useSettingsStore.getState().setCurrentTab(tab as any),
    []
  )

  useEffect(() => {
    if (message) {
      if (message.includes(InvalidApiKeyError) || message.includes(RequireApiKeError)) {
        setApiKeyAlertOpen(true)
      }
    }
  }, [message])

  return (
    <ThemeProvider>
      <TabVisibilityProvider>
        {initializing ? (
          // Loading state while initializing with simplified header
          <div className="flex h-screen w-screen flex-col">
            {/* Simplified header during initialization - matches SiteHeader structure */}
            <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50 flex h-10 w-full border-b px-4 backdrop-blur">
              <div className="min-w-[200px] w-auto flex items-center">
                <a href={webuiPrefix} className="flex items-center gap-2">
                  <ZapIcon className="size-4 text-emerald-400" aria-hidden="true" />
                  <span className="font-bold md:inline-block">{SiteInfo.name}</span>
                </a>
              </div>

              {/* Empty middle section to maintain layout */}
              <div className="flex h-10 flex-1 items-center justify-center">
              </div>

              {/* Empty right section to maintain layout */}
              <nav className="w-[200px] flex items-center justify-end">
              </nav>
            </header>

            {/* Loading indicator in content area */}
            <div className="flex flex-1 items-center justify-center">
              <div className="text-center">
                <div className="mb-2 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent mx-auto"></div>
                <p>Initializing...</p>
              </div>
            </div>
          </div>
        ) : multiTenantEnabled && !selectedTenant ? (
          // Only show tenant selection in multi-tenant mode when no tenant is selected
          <TenantSelectionPage onSelect={() => {}} />
        ) : (
          // Main content after initialization
          <main className="flex h-screen w-screen overflow-hidden">
            <Tabs
              defaultValue={currentTab}
              className="!m-0 flex grow flex-col !p-0 overflow-hidden"
              onValueChange={handleTabChange}
            >
              <SiteHeader />
              <div className="relative grow">
                <TabsContent value="documents" className="absolute top-0 right-0 bottom-0 left-0 overflow-auto">
                  <DocumentManager />
                </TabsContent>
                <TabsContent value="knowledge-graph" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
                  <GraphViewer />
                </TabsContent>
                <TabsContent value="retrieval" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
                  <RetrievalTesting />
                </TabsContent>
                <TabsContent value="api" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
                  <ApiSite />
                </TabsContent>
              </div>
            </Tabs>
            {enableHealthCheck && <StatusIndicator />}
            <ApiKeyAlert open={apiKeyAlertOpen} onOpenChange={handleApiKeyAlertOpenChange} />
          </main>
        )}
      </TabVisibilityProvider>
    </ThemeProvider>
  )
}

export default App
