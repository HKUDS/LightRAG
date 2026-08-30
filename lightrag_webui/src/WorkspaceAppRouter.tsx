import '@/lib/extensions'; // Import all global extensions
import '@/bootstrap/workspaceEntry'; // Entry-specific navigation policy — before anything can 401
import { HashRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { useEffect } from 'react'
import { useAuthStore } from '@/stores/state'
import { navigationService } from '@/services/navigation'
import {
  getSettingsMigrationError,
  wasHistoryDroppedDuringMigration
} from '@/migrations/splitSettingsStorage'
import { toast, Toaster } from 'sonner'
import i18n from '@/i18n'
import ThemeProvider from '@/components/ThemeProvider'
import MigrationErrorScreen from '@/components/MigrationErrorScreen'
import LoginPage from '@/features/LoginPage'
import WorkspaceWelcome from '@/features/workspace/WorkspaceWelcome'
import WorkspaceApp from '@/features/workspace/WorkspaceApp'

/**
 * The workspace entry's OWN router — its own route table, not a shared one
 * branched on an entry-mode flag (no such flag exists; entry identity is the
 * loaded HTML product):
 *
 *   #/welcome  public welcome page; the unauthenticated default
 *   #/login    login page; on success navigate('/') lands back in THIS entry
 *   #/         the protected query page
 *
 * Unknown hash routes fall back to this entry's own default page — never a
 * cross-entry jump.
 */
const WorkspaceAppContent = () => {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()

  // Register the navigate function for the (entry-configured) navigation
  // service. Token validity itself was already checked LOCALLY by the auth
  // store's init (expired/broken tokens were cleared before first render).
  useEffect(() => {
    navigationService.setNavigate(navigate)
  }, [navigate])

  // One-time notice: the storage-split migration had to DISCARD the legacy
  // chat history to fit within the browser quota (non-critical test data by
  // product decision — but the drop must not be silent).
  useEffect(() => {
    if (wasHistoryDroppedDuringMigration()) {
      toast.warning(i18n.t('migration.historyDropped'))
    }
  }, [])

  const fallback = isAuthenticated ? '/' : '/welcome'

  return (
    <Routes>
      <Route path="/welcome" element={<WorkspaceWelcome />} />
      {/* Guest activation happens ONLY through the welcome page's explicit
          action — an auth-disabled deployment redirects #/login there. */}
      <Route path="/login" element={<LoginPage autoActivateGuest={false} />} />
      <Route
        path="/"
        element={isAuthenticated ? <WorkspaceApp /> : <Navigate to="/welcome" replace />}
      />
      {/* Unknown hash → this entry's own default page. */}
      <Route path="*" element={<Navigate to={fallback} replace />} />
    </Routes>
  )
}

const WorkspaceAppRouter = () => {
  if (getSettingsMigrationError() != null) {
    // Storage split migration did not complete: dependent stores skipped
    // hydration, so render a retryable error instead of running on defaults.
    return (
      <ThemeProvider>
        <MigrationErrorScreen />
      </ThemeProvider>
    )
  }
  return (
    <ThemeProvider>
      <Router>
        <WorkspaceAppContent />
        <Toaster position="bottom-center" theme="system" closeButton richColors />
      </Router>
    </ThemeProvider>
  )
}

export default WorkspaceAppRouter
