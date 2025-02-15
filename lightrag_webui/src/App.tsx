import ThemeProvider from '@/components/ThemeProvider'
import MessageAlert from '@/components/MessageAlert'
import StatusIndicator from '@/components/StatusIndicator'
import { healthCheckInterval } from '@/lib/constants'
import { useBackendState } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { useEffect } from 'react'
import { Toaster } from 'sonner'
import SiteHeader from '@/features/SiteHeader'

import GraphViewer from '@/features/GraphViewer'
import DocumentManager from '@/features/DocumentManager'

import { Tabs, TabsContent } from '@/components/ui/Tabs'

function App() {
  const message = useBackendState.use.message()
  const enableHealthCheck = useSettingsStore.use.enableHealthCheck()

  // health check
  useEffect(() => {
    if (!enableHealthCheck) return

    // Check immediately
    useBackendState.getState().check()

    const interval = setInterval(async () => {
      await useBackendState.getState().check()
    }, healthCheckInterval * 1000)
    return () => clearInterval(interval)
  }, [enableHealthCheck])

  return (
    <ThemeProvider>
      <div className="flex h-screen w-screen">
        <Tabs defaultValue="knowledge-graph" className="flex size-full flex-col">
          <SiteHeader />
          <TabsContent value="documents" className="flex-1">
            <DocumentManager />
          </TabsContent>
          <TabsContent value="knowledge-graph" className="flex-1">
            <GraphViewer />
          </TabsContent>
          <TabsContent value="settings" className="size-full">
            <h1> Settings </h1>
          </TabsContent>
        </Tabs>
      </div>
      {enableHealthCheck && <StatusIndicator />}
      {message !== null && <MessageAlert />}
      <Toaster />
    </ThemeProvider>
  )
}

export default App
