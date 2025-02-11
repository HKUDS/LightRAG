import ThemeProvider from '@/components/ThemeProvider'
import MessageAlert from '@/components/MessageAlert'
import StatusIndicator from '@/components/StatusIndicator'
import GraphViewer from '@/GraphViewer'
import { healthCheckInterval } from '@/lib/constants'
import { useBackendState } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { useEffect } from 'react'

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
      <div className="h-screen w-screen">
        <GraphViewer />
      </div>
      {enableHealthCheck && <StatusIndicator />}
      {message !== null && <MessageAlert />}
    </ThemeProvider>
  )
}

export default App
