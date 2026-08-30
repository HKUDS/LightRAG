import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/stores/state'
import { activateLoginIdentityFromToken } from '@/lib/loginIdentity'
import { markVersionCheckedFromLogin } from '@/lib/versionCheckCache'
import { loginToServer, getAuthStatus } from '@/api/lightrag'
import logoUrl from '@/assets/logo.svg'
import { toast } from 'sonner'
import { useTranslation } from 'react-i18next'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import Input from '@/components/ui/Input'
import Button from '@/components/ui/Button'
import { ZapIcon } from 'lucide-react'
import AppSettings from '@/components/AppSettings'

interface LoginPageProps {
  /**
   * Whether an auth-disabled deployment may activate the guest token right
   * here. The admin entry keeps the historical auto-login; the WORKSPACE
   * entry passes false — its guest activation happens ONLY through the
   * welcome page's explicit action (PRD §6.3), so a direct or bookmarked
   * #/login visit redirects there instead of silently activating a token.
   */
  autoActivateGuest?: boolean
}

const LoginPage = ({ autoActivateGuest = true }: LoginPageProps) => {
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

  // Check if authentication is configured, skip login if not
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

        // Check auth status
        const status = await getAuthStatus()

        // Set session flag for version check to avoid duplicate checks in App component
        if (status.core_version || status.api_version) {
          markVersionCheckedFromLogin();
        }

        if (!status.auth_configured && status.access_token) {
          if (!autoActivateGuest) {
            // Workspace entry: guest activation belongs to the welcome page's
            // explicit action — never to a (possibly bookmarked) login visit.
            navigate('/welcome', { replace: true })
            return
          }
          // If auth is not configured, use the guest token and redirect.
          // Guest activation is an identity transition: a different stored
          // identity means the previous user's histories must not be shown.
          activateLoginIdentityFromToken(status.access_token)
          login(status.access_token, true, status.core_version, status.api_version, status.webui_title || null, status.webui_description || null)
          if (status.message) {
            toast.info(status.message)
          }
          navigate('/')
          return
        }

        // Only set checkingAuth to false if we need to show the login page
        setCheckingAuth(false);

      } catch (error) {
        console.error('Failed to check auth configuration:', error)
        // Also set checkingAuth to false in case of error
        setCheckingAuth(false);
      }
      // Removed finally block as we're setting checkingAuth earlier
    }

    // Execute immediately
    checkAuthConfig()

    // Cleanup function to prevent state updates after unmount
    return () => {
    }
  }, [isAuthenticated, login, navigate, autoActivateGuest])

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

      // Identity-change cleanup, shared with the guest activation paths: a
      // different user must not see the previous user's conversations in
      // EITHER entry — the rule applies to both split histories.
      const identityChanged = activateLoginIdentityFromToken(response.access_token)
      console.log(
        identityChanged
          ? 'Different user logging in, clearing chat history'
          : 'Same user logging in, preserving chat history'
      )

      // Check authentication mode
      const isGuestMode = response.auth_mode === 'disabled'
      login(response.access_token, isGuestMode, response.core_version, response.api_version, response.webui_title || null, response.webui_description || null)

      // Set session flag for version check
      if (response.core_version || response.api_version) {
        markVersionCheckedFromLogin();
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
    <div
      // min-h-dvh, not h-screen: mobile browsers include their collapsing
      // chrome in 100vh, which can push the form under the toolbar; dvh
      // tracks the actually visible viewport, and min- lets small screens
      // scroll instead of clipping.
      className="flex min-h-dvh w-screen items-center justify-center bg-gradient-to-br from-emerald-50 to-teal-100 dark:from-gray-900 dark:to-gray-800"
      style={{
        padding:
          'calc(env(safe-area-inset-top) + 1rem) calc(env(safe-area-inset-right) + 1rem) calc(env(safe-area-inset-bottom) + 1rem) calc(env(safe-area-inset-left) + 1rem)'
      }}
    >
      <div
        className="absolute flex items-center gap-2"
        style={{
          top: 'calc(env(safe-area-inset-top) + 1rem)',
          right: 'calc(env(safe-area-inset-right) + 1rem)'
        }}
      >
        <AppSettings className="bg-white/30 dark:bg-gray-800/30 backdrop-blur-sm rounded-md" />
      </div>
      <Card className="w-full max-w-[480px] shadow-lg mx-4">
        <CardHeader className="flex items-center justify-center space-y-2 pb-8 pt-6">
          <div className="flex flex-col items-center space-y-4">
            <div className="flex items-center gap-3">
              <img src={logoUrl} alt="LightRAG Logo" className="h-12 w-12" />
              <ZapIcon className="size-10 text-emerald-400" aria-hidden="true" />
            </div>
            <div className="text-center space-y-2">
              <h1 className="text-3xl font-bold tracking-tight">LightRAG</h1>
              <p className="text-muted-foreground text-sm">
                {t('login.description')}
              </p>
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
