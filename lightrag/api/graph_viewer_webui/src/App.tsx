import ThemeProvider from '@/components/ThemeProvider'
import { GraphViewer } from '@/GraphViewer'

function App() {
  return (
    <ThemeProvider defaultTheme="system" storageKey="lightrag-viewer-webui-theme">
      <div className="h-screen w-screen">
        <GraphViewer />
      </div>
    </ThemeProvider>
  )
}

export default App