import ThemeProvider from '@/components/ThemeProvider'
import BackendMessageAlert from '@/components/BackendMessageAlert'
import { GraphViewer } from '@/GraphViewer'
import { cn } from '@/lib/utils'
import { useBackendState } from '@/stores/state'

function App() {
  const health = useBackendState.use.health()

  return (
    <ThemeProvider>
      <div className={cn('h-screen w-screen', !health && 'pointer-events-none')}>
        <GraphViewer />
      </div>
      {!health && <BackendMessageAlert />}
    </ThemeProvider>
  )
}

export default App
