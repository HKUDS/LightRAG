import { useState, useCallback } from 'react'
import ThemeProvider from '@/components/ThemeProvider'
import TabVisibilityProvider from '@/contexts/TabVisibilityProvider'
import MessageAlert from '@/components/MessageAlert'
import ApiKeyAlert from '@/components/ApiKeyAlert'
import StatusIndicator from '@/components/graph/StatusIndicator'
import { healthCheckInterval } from '@/lib/constants'
import { useBackendState, useAuthStore } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { useEffect } from 'react'
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

  // Health check
  useEffect(() => {
    const { isAuthenticated } = useAuthStore.getState();
    if (!enableHealthCheck || !isAuthenticated) return

    // Check immediately
    useBackendState.getState().check()

    const interval = setInterval(async () => {
      await useBackendState.getState().check()
    }, healthCheckInterval * 1000)
    return () => clearInterval(interval)
  }, [enableHealthCheck])

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
