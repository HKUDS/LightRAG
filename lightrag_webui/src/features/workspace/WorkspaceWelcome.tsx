import { useCallback, useEffect, useRef, useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import AppSettings from '@/components/AppSettings'
import CustomizedMarkdown from '@/components/customization/CustomizedMarkdown'
import CustomizedCopyright from '@/components/customization/CustomizedCopyright'
import { useCustomizedContent } from '@/components/customization/useCustomizedContent'
import { useAuthStore } from '@/stores/state'
import { getAuthStatus } from '@/api/lightrag'
import { activateLoginIdentityFromToken } from '@/lib/loginIdentity'
import { markVersionCheckedFromLogin } from '@/lib/versionCheckCache'

/**
 * The workspace entry's public welcome page — the unauthenticated default of
 * this entry ("no valid token" ⇒ welcome; a guest token is a valid login and
 * skips it entirely).
 *
 * Guest-token FETCH and ACTIVATION are deliberately separated (§6.3):
 * /auth-status necessarily returns a guest token when auth is disabled (and
 * it is requested here in parallel with the customization content to keep
 * first paint fast), but on page load that token is DISCARDED — not passed
 * to login(), not written to localStorage. Only the explicit "Enter
 * workspace" click re-requests /auth-status and activates the token;
 * otherwise the welcome page would be skipped by the very token it stored.
 */
export default function WorkspaceWelcome() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { isAuthenticated } = useAuthStore()
  const content = useCustomizedContent()
  const [authConfigured, setAuthConfigured] = useState<boolean | null>(null)
  const [entering, setEntering] = useState(false)
  // Latch the URL that failed, not a boolean: a transient failure must not
  // keep hiding the logo after a language switch supplies a different URL.
  const [failedLogoUrl, setFailedLogoUrl] = useState<string | null>(null)
  const authProbeRef = useRef(false)

  useEffect(() => {
    if (isAuthenticated) return
    if (authProbeRef.current) return
    authProbeRef.current = true
    // Requested in parallel with the customization request that
    // useCustomizedContent fires on mount — no serial first-paint cost.
    getAuthStatus()
      .then((status) => {
        // Read auth_configured ONLY; a piggybacked guest token is dropped.
        setAuthConfigured(status.auth_configured)
      })
      .catch(() => setAuthConfigured(true))
  }, [isAuthenticated])

  const handlePrimaryAction = useCallback(async () => {
    if (authConfigured !== false) {
      navigate('/login')
      return
    }
    // Auth disabled: NOW fetch and activate the guest token.
    //
    // No consent gate here, deliberately: the login-page gate binds an
    // agreement to a user, and this path has no user to bind it to. See the
    // guest branch of LoginPage.checkAuthConfig for the full reasoning.
    setEntering(true)
    try {
      const status = await getAuthStatus()
      if (!status.auth_configured && status.access_token) {
        // Guest activation is an identity transition like any login: when the
        // browser was last used by a DIFFERENT identity (e.g. a named user
        // before auth was disabled), both entries' histories are cleared so
        // the guest never sees that user's conversations.
        activateLoginIdentityFromToken(status.access_token)
        useAuthStore
          .getState()
          .login(
            status.access_token,
            true,
            status.core_version,
            status.api_version,
            status.webui_title || null,
            status.webui_description || null
          )
        markVersionCheckedFromLogin()
        navigate('/')
      } else {
        // Auth got enabled between the probe and the click.
        navigate('/login')
      }
    } catch (error) {
      console.error('Failed to enter workspace:', error)
      toast.error(t('workspace.welcome.enterFailed', 'Could not reach the server. Please try again.'))
    } finally {
      setEntering(false)
    }
  }, [authConfigured, navigate, t])

  // A valid login (guest included) never sees the welcome page.
  if (isAuthenticated) {
    return <Navigate to="/" replace />
  }

  return (
    <div
      className="flex h-dvh w-screen flex-col items-center justify-center bg-gradient-to-br from-emerald-50 to-teal-100 dark:from-gray-900 dark:to-gray-800"
      style={{
        // 1rem base padding extended by the device safe areas (notch,
        // Dynamic Island, home indicator).
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
        <AppSettings className="bg-white/30 backdrop-blur-sm dark:bg-gray-800/30 rounded-md" />
      </div>
      {/* The card centers in the space ABOVE the footer, which keeps the
          copyright line at the page's bottom edge without overlapping the
          card on short screens (min-h-0 leaves the card's own scrolling
          intact). */}
      <div className="flex min-h-0 w-full flex-1 items-center justify-center">
        <div
          dir={content.direction}
          className="flex max-h-full w-full max-w-[520px] flex-col items-center gap-6 overflow-y-auto rounded-xl bg-white/70 p-8 text-center shadow-lg backdrop-blur dark:bg-gray-900/70"
        >
          {content.loading ? (
            <div
              className="border-primary size-8 animate-spin rounded-full border-4 border-t-transparent"
              role="status"
              aria-label="Loading"
            />
          ) : (
            <>
              {content.logoUrl && failedLogoUrl !== content.logoUrl && (
                <img
                  src={content.logoUrl}
                  alt={content.logoAlt}
                  className="max-h-[88px] max-w-[88px] object-contain md:max-h-[120px] md:max-w-[120px]"
                  onError={() => setFailedLogoUrl(content.logoUrl)}
                />
              )}
              <h1 className="text-2xl font-bold tracking-tight">{content.brandTitle}</h1>
              <CustomizedMarkdown
                content={content.welcomeMarkdown}
                className="text-left text-sm"
              />
              <Button
                type="button"
                className="h-11 w-full max-w-xs text-base font-medium"
                disabled={authConfigured === null || entering}
                onClick={handlePrimaryAction}
              >
                {authConfigured === false
                  ? t('workspace.welcome.enterButton', 'Enter workspace')
                  : t('workspace.welcome.loginButton', 'Sign in')}
              </Button>
            </>
          )}
        </div>
      </div>
      <CustomizedCopyright
        copyright={content.copyright}
        direction={content.direction}
        className="pt-4"
      />
    </div>
  )
}
