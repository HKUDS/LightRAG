import ThemeProvider from '@/components/ThemeProvider'
import MessageAlert from '@/components/MessageAlert'
import { GraphViewer } from '@/GraphViewer'
import { cn } from '@/lib/utils'
import { healthCheckInterval } from '@/lib/constants'
import { useBackendState } from '@/stores/state'
import { useEffect } from 'react'

function App() {
  const message = useBackendState.use.message()

  // health check
  useEffect(() => {
    const interval = setInterval(async () => {
      await useBackendState.getState().check()
    }, healthCheckInterval * 1000)
    return () => clearInterval(interval)
  }, [])

  return (
    <ThemeProvider>
      <div className={cn('h-screen w-screen', message !== null && 'pointer-events-none')}>
        <GraphViewer />
      </div>
      {message !== null && <MessageAlert />}
    </ThemeProvider>
  )
}

export default App
