import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { useAuthStore } from '@/stores/state'
import { getAuthStatus } from '@/api/lightrag'
import { toast } from 'sonner'
import { Toaster } from 'sonner'
import App from './App'
import LoginPage from '@/features/LoginPage'
import ThemeProvider from '@/components/ThemeProvider'

interface ProtectedRouteProps {
  children: React.ReactNode
}

const ProtectedRoute = ({ children }: ProtectedRouteProps) => {
  const { isAuthenticated } = useAuthStore()
  const [isChecking, setIsChecking] = useState(true)

  useEffect(() => {
    let isMounted = true; // Flag to prevent state updates after unmount
    
    // This effect will run when the component mounts
    // and will check if authentication is required
    const checkAuthStatus = async () => {
      try {
        // Skip check if already authenticated
        if (isAuthenticated) {
          if (isMounted) setIsChecking(false);
          return;
        }
        
        const status = await getAuthStatus()
        
        // Only proceed if component is still mounted
        if (!isMounted) return;
        
        if (!status.auth_configured && status.access_token) {
          // If auth is not configured, use the guest token
          useAuthStore.getState().login(status.access_token, true)
          if (status.message) {
            toast.info(status.message)
          }
        }
      } catch (error) {
        console.error('Failed to check auth status:', error)
      } finally {
        // Only update state if component is still mounted
        if (isMounted) {
          setIsChecking(false)
        }
      }
    }

    // Execute immediately
    checkAuthStatus()
    
    // Cleanup function to prevent state updates after unmount
    return () => {
      isMounted = false;
    }
  }, [isAuthenticated])

  // Show nothing while checking auth status
  if (isChecking) {
    return null
  }

  // After checking, if still not authenticated, redirect to login
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}

const AppRouter = () => {
  const [initializing, setInitializing] = useState(true)
  const { isAuthenticated } = useAuthStore()

  // Check token validity and auth configuration on app initialization
  useEffect(() => {
    let isMounted = true; // Flag to prevent state updates after unmount
    
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('LIGHTRAG-API-TOKEN')
        
        // If we have a token, we're already authenticated
        if (token && isAuthenticated) {
          if (isMounted) setInitializing(false);
          return;
        }
        
        // If no token or not authenticated, check if auth is configured
        const status = await getAuthStatus()
        
        // Only proceed if component is still mounted
        if (!isMounted) return;
        
        if (!status.auth_configured && status.access_token) {
          // If auth is not configured, use the guest token
          useAuthStore.getState().login(status.access_token, true)
          if (status.message) {
            toast.info(status.message)
          }
        } else if (!token) {
          // Only logout if we don't have a token
          useAuthStore.getState().logout()
        }
      } catch (error) {
        console.error('Auth initialization error:', error)
        if (isMounted && !isAuthenticated) {
          useAuthStore.getState().logout()
        }
      } finally {
        // Only update state if component is still mounted
        if (isMounted) {
          setInitializing(false)
        }
      }
    }

    // Execute immediately
    checkAuth()
    
    // Cleanup function to prevent state updates after unmount
    return () => {
      isMounted = false;
    }
  }, [isAuthenticated])

  // Show nothing while initializing
  if (initializing) {
    return null
  }

  return (
    <ThemeProvider>
      <Router>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <App />
              </ProtectedRoute>
            }
          />
        </Routes>
        <Toaster position="top-center" />
      </Router>
    </ThemeProvider>
  )
}

export default AppRouter
