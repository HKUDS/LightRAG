import { useState, useCallback, useEffect, useRef } from 'react'
import ThemeProvider from '@/components/ThemeProvider'
import TabVisibilityProvider from '@/contexts/TabVisibilityProvider'
import MessageAlert from '@/components/MessageAlert'
import ApiKeyAlert from '@/components/ApiKeyAlert'
import StatusIndicator from '@/components/graph/StatusIndicator'
import { healthCheckInterval } from '@/lib/constants'
import { useBackendState, useAuthStore } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { getAuthStatus } from '@/api/lightrag'
import SiteHeader from '@/features/SiteHeader'
import { InvalidApiKeyError, RequireApiKeError } from '@/api/lightrag'

import GraphViewer from '@/features/GraphViewer'
import DocumentManager from '@/features/DocumentManager'
import RetrievalTesting from '@/features/RetrievalTesting'
import ApiSite from '@/features/ApiSite'

import { Tabs, TabsContent } from '@/components/ui/Tabs'

function App() {
  const message = useBackendState.use.message()
  const enableHealthCheck = useSettingsStore.use.enableHealthCheck()
  const currentTab = useSettingsStore.use.currentTab()
  const [apiKeyInvalid, setApiKeyInvalid] = useState(false)
  const versionCheckRef = useRef(false); // Prevent duplicate calls in Vite dev mode

  // Health check - can be disabled
  useEffect(() => {
    // Only execute if health check is enabled
    if (!enableHealthCheck) return;

    // Health check function
    const performHealthCheck = async () => {
      await useBackendState.getState().check();
    };

    // Execute immediately
    performHealthCheck();

    // Set interval for periodic execution
    const interval = setInterval(performHealthCheck, healthCheckInterval * 1000);
    return () => clearInterval(interval);
  }, [enableHealthCheck]);

  // Version check - independent and executed only once
  useEffect(() => {
    const checkVersion = async () => {
      // Prevent duplicate calls in Vite dev mode
      if (versionCheckRef.current) return;
      versionCheckRef.current = true;

      // Check if version info was already obtained in login page
      const versionCheckedFromLogin = sessionStorage.getItem('VERSION_CHECKED_FROM_LOGIN') === 'true';
      if (versionCheckedFromLogin) return;

      // Get version info
      const token = localStorage.getItem('LIGHTRAG-API-TOKEN');
      if (!token) return;

      try {
        const status = await getAuthStatus();
        if (status.core_version || status.api_version) {
          // Update version info while maintaining login state
          useAuthStore.getState().login(
            token,
            useAuthStore.getState().isGuestMode,
            status.core_version,
            status.api_version
          );

          // Set flag to indicate version info has been checked
          sessionStorage.setItem('VERSION_CHECKED_FROM_LOGIN', 'true');
        }
      } catch (error) {
        console.error('Failed to get version info:', error);
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
        setApiKeyInvalid(true)
        return
      }
    }
    setApiKeyInvalid(false)
  }, [message, setApiKeyInvalid])

  return (
    <ThemeProvider>
      <TabVisibilityProvider>
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
          {message !== null && !apiKeyInvalid && <MessageAlert />}
          {apiKeyInvalid && <ApiKeyAlert />}
        </main>
      </TabVisibilityProvider>
    </ThemeProvider>
  )
}

export default App
