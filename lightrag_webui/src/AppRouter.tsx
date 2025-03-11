import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
// import { useAuthStore } from '@/stores/state'
import { Toaster } from 'sonner'
import App from './App'
import LoginPage from '@/features/LoginPage'
import ThemeProvider from '@/components/ThemeProvider'

interface ProtectedRouteProps {
  children: React.ReactNode
}

const ProtectedRoute = ({ children }: ProtectedRouteProps) => {
  // const { isAuthenticated } = useAuthStore()

  // if (!isAuthenticated) {
  //   return <Navigate to="/login" replace />
  // }

  return <>{children}</>
}

const AppRouter = () => {
  return (
    <ThemeProvider>
      <BrowserRouter>
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
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default AppRouter
