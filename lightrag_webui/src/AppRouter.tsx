import '@/lib/extensions'; // Import all global extensions
import { HashRouter as Router, Navigate, Routes, Route, useNavigate, useParams } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { useAuthStore } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'
import { useWorkspaceStore, type Workspace } from '@/stores/workspace'
import { getWorkspace } from '@/api/lightrag'
import { navigationService } from '@/services/navigation'
import { Toaster } from 'sonner'
import App from './App'
import LoginPage from '@/features/LoginPage'
import WorkspaceManager from '@/features/WorkspaceManager'
import ThemeProvider from '@/components/ThemeProvider'

const WorkspaceScope = () => {
  const { workspaceId } = useParams<{ workspaceId: string }>()
  const navigate = useNavigate()
  const selectWorkspace = useWorkspaceStore.use.selectWorkspace()
  const [workspace, setWorkspace] = useState<Workspace | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    const loadWorkspace = async () => {
      if (!workspaceId) {
        navigate('/workspaces', { replace: true })
        return
      }

      setWorkspace(null)
      setError(null)
      try {
        const nextWorkspace = await getWorkspace(workspaceId)
        if (cancelled) return

        if (nextWorkspace.status !== 'active') {
          setError('This workspace is not ready yet.')
          return
        }

        // The data layer is isolated server-side by the workspace header. Reset
        // client-side graph and chat state too, so no content from a previous
        // workspace remains visible while the selected runtime changes.
        useGraphStore.getState().reset()
        useGraphStore.getState().setGraphDataFetchAttempted(false)
        useGraphStore.getState().setLabelsFetchAttempted(false)
        useSettingsStore.getState().setQueryLabel('')
        useSettingsStore.getState().setRetrievalHistory([])
        selectWorkspace(nextWorkspace)
        setWorkspace(nextWorkspace)
      } catch {
        if (!cancelled) setError('The requested workspace could not be loaded.')
      }
    }

    void loadWorkspace()
    return () => {
      cancelled = true
    }
  }, [navigate, selectWorkspace, workspaceId])

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center p-6">
        <div className="max-w-md space-y-4 text-center">
          <p className="text-muted-foreground">{error}</p>
          <button className="text-primary underline" onClick={() => navigate('/workspaces')}>
            Return to workspaces
          </button>
        </div>
      </div>
    )
  }

  if (!workspace) {
    return (
      <div className="flex h-screen items-center justify-center text-muted-foreground">
        Loading workspace...
      </div>
    )
  }

  return <App key={workspace.id} workspace={workspace} onWorkspaceManager={() => navigate('/workspaces')} />
}

const AppContent = () => {
  const [initializing, setInitializing] = useState(true)
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()

  // Set navigate function for navigation service
  useEffect(() => {
    navigationService.setNavigate(navigate)
  }, [navigate])

  // Token validity check
  useEffect(() => {

    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('LIGHTRAG-API-TOKEN')

        if (token && isAuthenticated) {
          setInitializing(false);
          return;
        }

        if (!token) {
          useAuthStore.getState().logout()
        }
      } catch (error) {
        console.error('Auth initialization error:', error)
        if (!isAuthenticated) {
          useAuthStore.getState().logout()
        }
      } finally {
        setInitializing(false)
      }
    }

    checkAuth()

    return () => {
    }
  }, [isAuthenticated])

  // Redirect effect for protected routes
  useEffect(() => {
    if (!initializing && !isAuthenticated) {
      const currentPath = window.location.hash.slice(1);
      if (currentPath !== '/login') {
        console.log('Not authenticated, redirecting to login');
        navigate('/login');
      }
    }
  }, [initializing, isAuthenticated, navigate]);

  // Show nothing while initializing
  if (initializing) {
    return null
  }

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/workspaces" element={isAuthenticated ? <WorkspaceManager /> : null} />
      <Route path="/workspaces/:workspaceId/*" element={isAuthenticated ? <WorkspaceScope /> : null} />
      <Route path="/*" element={isAuthenticated ? <Navigate to="/workspaces" replace /> : null} />
    </Routes>
  )
}

const AppRouter = () => {
  return (
    <ThemeProvider>
      <Router>
        <AppContent />
        <Toaster
          position="bottom-center"
          theme="system"
          closeButton
          richColors
        />
      </Router>
    </ThemeProvider>
  )
}

export default AppRouter
