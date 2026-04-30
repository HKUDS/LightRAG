import '@/lib/extensions'; // Import all global extensions
import { HashRouter as Router, Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { useAuthStore } from '@/stores/state'
import { navigationService } from '@/services/navigation'
import { Toaster } from 'sonner'
import LoginPage from '@/features/LoginPage'
import LittleBullPreview from '@/features/LittleBullPreview'
import ThemeProvider from '@/components/ThemeProvider'

const littleBullPaths = new Set(['/little-bull', '/little-bull-preview'])
const littleBullLoginRedirectKey = 'LIGHTRAG-LITTLE-BULL-LOGIN-REDIRECT'

const getSafeLittleBullRedirect = (path: string | null) => {
  return path && littleBullPaths.has(path) ? path : null
}

const AppContent = () => {
  const [initializing, setInitializing] = useState(true)
  const { isAuthenticated, isGuestMode } = useAuthStore()
  const navigate = useNavigate()
  const location = useLocation()

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
    if (initializing) return

    const currentPath = location.pathname

    if (isAuthenticated && !isGuestMode && currentPath === '/') {
      const pendingLittleBullPath = getSafeLittleBullRedirect(
        sessionStorage.getItem(littleBullLoginRedirectKey)
      )
      if (pendingLittleBullPath) {
        sessionStorage.removeItem(littleBullLoginRedirectKey)
        navigate(pendingLittleBullPath, { replace: true })
        return
      }
      navigate('/little-bull', { replace: true })
      return
    }

    if (!isAuthenticated || isGuestMode) {
      const isLittleBullPath = littleBullPaths.has(currentPath)
      const publicPaths = ['/login']
      if (!publicPaths.includes(currentPath) && (!isAuthenticated || isLittleBullPath)) {
        if (isLittleBullPath) {
          sessionStorage.setItem(littleBullLoginRedirectKey, currentPath)
        }
        console.log('Not authenticated, redirecting to login');
        navigate('/login', { replace: true });
      }
    }
  }, [initializing, isAuthenticated, isGuestMode, location.pathname, navigate]);

  // Show nothing while initializing
  if (initializing) {
    return null
  }

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/"
        element={isAuthenticated && !isGuestMode ? <Navigate to="/little-bull" replace /> : <Navigate to="/login" replace />}
      />
      <Route
        path="/little-bull-preview"
        element={<Navigate to="/little-bull" replace />}
      />
      <Route
        path="/little-bull"
        element={isAuthenticated && !isGuestMode ? <LittleBullPreview /> : null}
      />
      <Route
        path="/*"
        element={isAuthenticated && !isGuestMode ? <Navigate to="/little-bull" replace /> : <Navigate to="/login" replace />}
      />
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
