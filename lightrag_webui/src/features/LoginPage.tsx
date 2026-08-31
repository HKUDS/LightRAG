import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/stores/state'
import { activateLoginIdentityFromToken } from '@/lib/loginIdentity'
import { markVersionCheckedFromLogin } from '@/lib/versionCheckCache'
import { loginToServer, getAuthStatus } from '@/api/lightrag'
import { toast } from 'sonner'
import { useTranslation } from 'react-i18next'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import Input from '@/components/ui/Input'
import Button from '@/components/ui/Button'
import Checkbox from '@/components/ui/Checkbox'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import { ScrollArea } from '@/components/ui/ScrollArea'
import AppSettings from '@/components/AppSettings'
import CustomizedMarkdown from '@/components/customization/CustomizedMarkdown'
import CustomizedCopyright from '@/components/customization/CustomizedCopyright'
import { useCustomizedContent } from '@/components/customization/useCustomizedContent'
import {
  CONSENT_LABEL_MARKER,
  isAgreedTo,
  isLoginBlockedByConsent,
  shouldShowLoginConsent,
  splitConsentLabel,
  type LoginConsentTick
} from '@/features/loginConsent'

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
  const [authStatusTitle, setAuthStatusTitle] = useState<string | null>(null)
  // The tick is stored WITH the document it was given for; a language switch
  // replaces that document, and consent must not survive the swap.
  const [consentTick, setConsentTick] = useState<LoginConsentTick>({
    document: null,
    agreed: false
  })
  const [agreementsOpen, setAgreementsOpen] = useState(false)
  // Latch the URL that failed, not a boolean: a language switch may supply
  // a different logo URL that should get its own load attempt.
  const [failedLogoUrl, setFailedLogoUrl] = useState<string | null>(null)
  const authCheckRef = useRef(false); // Prevent duplicate calls in Vite dev mode

  // Fired in PARALLEL with the /auth-status probe below (both are effects of
  // this same mount), so the consent gate costs no serial round trip.
  const content = useCustomizedContent(authStatusTitle)
  const consentAgreed = isAgreedTo(consentTick, content.agreementsMarkdown)
  const consentState = {
    // NOT content.loading: on a language switch the store keeps the previous
    // locale's snapshot on screen, and its verdict does not carry.
    pending: content.consentPending,
    consentRequired: content.consentRequired,
    agreed: consentAgreed
  }
  const showConsent = shouldShowLoginConsent(consentState)
  const consentBlocked = isLoginBlockedByConsent(consentState)
  const consentDocuments = t('login.consentDocuments')
  const consentLabelText = t('login.consentLabel', { documents: consentDocuments })
  const consentLabelParts = splitConsentLabel(
    t('login.consentLabel', { documents: CONSENT_LABEL_MARKER })
  )

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
        setAuthStatusTitle(status.webui_title || null)

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
          //
          // NOT gated by the consent checkbox, deliberately. The gate binds
          // an agreement to a user, and an auth-disabled deployment has no
          // user to bind it to -- every visitor is the same anonymous guest,
          // so a tick here would record nothing and prove nothing. It is a
          // development and demo posture, not a production one; a deployment
          // that needs the agreement accepted configures AUTH_ACCOUNTS and
          // gets the gate on the credentialed form below. The same reasoning
          // covers the workspace entry's guest activation in
          // WorkspaceWelcome.handlePrimaryAction. Documented in
          // docs/ui_templates_example/README.md.
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

  // Don't render anything while checking auth, or before the customization
  // response settles. The second wait is what keeps the consent gate honest:
  // the flag that decides whether signing in requires agreement is not known
  // until then, so a form rendered earlier would be submittable ungated on a
  // deployment that requires the agreement. Both requests are in flight
  // together, so on a reachable server this is the same single wait as before.
  if (checkingAuth || content.loading) {
    return null
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!username || !password) {
      toast.error(t('login.errorEmptyFields'))
      return
    }
    // The disabled button is the visible half of the gate; this is the half
    // that actually holds. A form submits on Enter regardless of its submit
    // button's state, so without this check the keyboard path would bypass
    // the agreement entirely.
    if (consentBlocked) {
      toast.error(t('login.errorConsentRequired'))
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
      className="flex min-h-dvh w-full flex-col items-center justify-center bg-gradient-to-br from-emerald-50 to-teal-100 dark:from-gray-900 dark:to-gray-800"
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
      {/* The card centers in the space ABOVE the footer, so the copyright
          line sits at the page's bottom edge instead of inside the card. */}
      <div className="flex w-full flex-1 items-center justify-center">
        <Card className="w-full max-w-[480px] shadow-lg mx-4">
          <CardHeader className="flex items-center justify-center space-y-2 pb-8 pt-6">
            <div className="flex flex-col items-center space-y-4">
              {content.logoUrl && failedLogoUrl !== content.logoUrl && (
                <img
                  src={content.logoUrl}
                  alt={content.logoAlt}
                  className="h-12 w-12 object-contain"
                  onError={() => setFailedLogoUrl(content.logoUrl)}
                />
              )}
              <div className="text-center space-y-2">
                <h1 className="text-3xl font-bold tracking-tight">{content.brandTitle}</h1>
                <p className="text-muted-foreground text-sm">
                  {t('login.description')}
                </p>
              </div>
            </div>
          </CardHeader>
          <CardContent className="px-8 pb-8">
            {content.loginMarkdown && (
              <CustomizedMarkdown
                content={content.loginMarkdown}
                dir={content.direction}
                className="mb-6 text-sm"
              />
            )}
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
                  className="h-11 min-w-0 flex-1"
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
                  className="h-11 min-w-0 flex-1"
                />
              </div>
              {showConsent && (
                <div className="flex items-start gap-2" dir={content.direction}>
                  <Checkbox
                    id="login-consent"
                    checked={consentAgreed}
                    onCheckedChange={(checked) =>
                      setConsentTick({
                        document: content.agreementsMarkdown,
                        agreed: checked === true
                      })
                    }
                    // The visible text is split around the document link, so the
                    // control's accessible name is spelled out here in full
                    // rather than assembled from the label fragments.
                    aria-label={consentLabelText}
                    className="mt-0.5"
                  />
                  <span className="text-muted-foreground text-sm leading-5">
                    {/* Two labels around the link, never one wrapping it: a
                        <label> that contained the button would toggle the
                        checkbox on every click meant to OPEN the document. */}
                    {consentLabelParts.before && (
                      <label htmlFor="login-consent" className="cursor-pointer">
                        {consentLabelParts.before}
                      </label>
                    )}
                    <button
                      type="button"
                      className="text-primary underline underline-offset-2 hover:opacity-80"
                      onClick={() => setAgreementsOpen(true)}
                    >
                      {consentDocuments}
                    </button>
                    {consentLabelParts.after && (
                      <label htmlFor="login-consent" className="cursor-pointer">
                        {consentLabelParts.after}
                      </label>
                    )}
                  </span>
                </div>
              )}
              <Button
                type="submit"
                className="w-full h-11 text-base font-medium mt-2"
                disabled={loading || consentBlocked}
              >
                {loading ? t('login.loggingIn') : t('login.loginButton')}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
      <CustomizedCopyright
        copyright={content.copyright}
        direction={content.direction}
        className="pt-4"
      />
      {/* Rendered only where the gate exists, so a deployment without an
          agreement document has no dialog to open. */}
      {showConsent && content.agreementsMarkdown && (
        <Dialog open={agreementsOpen} onOpenChange={setAgreementsOpen}>
          <DialogContent className="max-h-[85vh] max-w-2xl" dir={content.direction}>
            <DialogHeader>
              <DialogTitle>{consentDocuments}</DialogTitle>
            </DialogHeader>
            <ScrollArea className="max-h-[60vh] pr-4">
              {/* No text-sm here: this is a document to READ, not a form
                  caption, so it keeps the prose tier's own sizing. */}
              <CustomizedMarkdown content={content.agreementsMarkdown} />
            </ScrollArea>
          </DialogContent>
        </Dialog>
      )}
    </div>
  )
}

export default LoginPage
