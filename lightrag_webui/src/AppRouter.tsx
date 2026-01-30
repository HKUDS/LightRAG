import '@/lib/extensions'; // Import all global extensions
import { HashRouter as Router, Routes, Route, useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { useAuthStore } from '@/stores/state'
import { navigationService } from '@/services/navigation'
import { Toaster } from 'sonner'
import App from './App'
import LoginPage from '@/features/LoginPage'
import RegisterPage from '@/features/RegisterPage'
import ThemeProvider from '@/components/ThemeProvider'

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
      if (currentPath !== '/login' && currentPath !== '/register') {
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
      <Route path="/register" element={<RegisterPage />} />
      <Route
        path="/*"
        element={isAuthenticated ? <App /> : null}
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
