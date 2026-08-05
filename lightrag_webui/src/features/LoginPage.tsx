import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore, isGuestToken } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { loginToServer, getAuthStatus } from '@/api/lightrag'
import { toast } from 'sonner'
import { useTranslation } from 'react-i18next'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import Input from '@/components/ui/Input'
import Button from '@/components/ui/Button'
import { ZapIcon } from 'lucide-react'
import AppSettings from '@/components/AppSettings'

const LoginPage = () => {
  const navigate = useNavigate()
  const { login, isAuthenticated } = useAuthStore()
  const { t } = useTranslation()
  const [loading, setLoading] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [checkingAuth, setCheckingAuth] = useState(true)
  const authCheckRef = useRef(false); // Prevent duplicate calls in Vite dev mode

  useEffect(() => {
    console.log('LoginPage mounted')
  }, []);

  // Check auth status and show login form regardless of configuration
  useEffect(() => {

    const checkAuthConfig = async () => {
      // Prevent duplicate calls in Vite dev mode
      if (authCheckRef.current) {
        return;
      }
      authCheckRef.current = true;

      try {
        // If already authenticated, redirect to home
        if (isAuthenticated) {
          navigate('/')
          return
        }

        // Check auth status for version info
        const status = await getAuthStatus()

        // Set session flag for version check to avoid duplicate checks in App component
        if (status.core_version || status.api_version) {
          sessionStorage.setItem('VERSION_CHECKED_FROM_LOGIN', 'true');
        }
      } finally {
        setCheckingAuth(false);
      }
    }

    // Execute immediately
    checkAuthConfig()

    return () => {
    }
  }, [isAuthenticated, login, navigate])

  // Don't render anything while checking auth
  if (checkingAuth) {
    return null
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!username || !password) {
      toast.error(t('login.errorEmptyFields'))
      return
    }

    try {
      setLoading(true)
      const response = await loginToServer(username, password)

      // Get previous username from localStorage
      const previousUsername = localStorage.getItem('LIGHTRAG-PREVIOUS-USER')

      // Check if it's the same user logging in again
      const isSameUser = previousUsername === username

      // If it's not the same user, clear chat history
      if (isSameUser) {
        console.log('Same user logging in, preserving chat history')
      } else {
        console.log('Different user logging in, clearing chat history')
        // Directly clear chat history instead of setting a flag
        useSettingsStore.getState().setRetrievalHistory([])
      }

      // Update previous username
      localStorage.setItem('LIGHTRAG-PREVIOUS-USER', username)

      // Determine guest mode from the JWT token role, not from the server's
      // auth_mode flag. A token with role='guest' is a guest regardless of
      // whether the server has auth enabled or disabled.
      const isGuestMode = isGuestToken(response.access_token)
      login(response.access_token, isGuestMode, response.permissions || null, response.core_version, response.api_version, response.webui_title || null, response.webui_description || null)

      // Set session flag for version check
      if (response.core_version || response.api_version) {
        sessionStorage.setItem('VERSION_CHECKED_FROM_LOGIN', 'true');
      }

      if (isGuestMode) {
        // Show authentication disabled notification
        toast.info(response.message || t('login.authDisabled', 'Authentication is disabled. Using guest access.'))
      } else {
        toast.success(t('login.successMessage'))
      }

      // Navigate to home page after successful login
      navigate('/')
    } catch (error) {
      console.error('Login failed...', error)
      toast.error(t('login.errorInvalidCredentials'))

      // Clear any existing auth state
      useAuthStore.getState().logout()
      // Clear local storage
      localStorage.removeItem('LIGHTRAG-API-TOKEN')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-gradient-to-br from-emerald-50 to-teal-100 dark:from-gray-900 dark:to-gray-800">
      <div className="absolute top-4 right-4 flex items-center gap-2">
        <AppSettings className="bg-white/30 dark:bg-gray-800/30 backdrop-blur-sm rounded-md" />
      </div>
      <Card className="w-full max-w-[480px] shadow-lg mx-4">
        <CardHeader className="flex items-center justify-center space-y-2 pb-8 pt-6">
          <div className="flex flex-col items-center space-y-4">
            <div className="flex items-center gap-3">
              <img src="logo.svg" alt="RAG知识库 Logo" className="h-12 w-12" />
              <ZapIcon className="size-10 text-emerald-400" aria-hidden="true" />
            </div>
            <div className="text-center space-y-2">
              <h1 className="text-3xl font-bold tracking-tight">RAG知识库</h1>
            </div>
          </div>
        </CardHeader>
        <CardContent className="px-8 pb-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="flex items-center gap-4">
              <label htmlFor="username-input" className="text-sm font-medium w-16 shrink-0">
                {t('login.username')}
              </label>
              <Input
                id="username-input"
                placeholder={t('login.usernamePlaceholder')}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="h-11 flex-1"
              />
            </div>
            <div className="flex items-center gap-4">
              <label htmlFor="password-input" className="text-sm font-medium w-16 shrink-0">
                {t('login.password')}
              </label>
              <Input
                id="password-input"
                type="password"
                placeholder={t('login.passwordPlaceholder')}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="h-11 flex-1"
              />
            </div>
            <Button
              type="submit"
              className="w-full h-11 text-base font-medium mt-2"
              disabled={loading}
            >
              {loading ? t('login.loggingIn') : t('login.loginButton')}
            </Button>
          </form>

        </CardContent>
      </Card>
    </div>
  )
}

export default LoginPage
