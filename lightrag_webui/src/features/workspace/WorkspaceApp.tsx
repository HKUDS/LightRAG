import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ZapIcon, LogOutIcon } from 'lucide-react'
import Button from '@/components/ui/Button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import TouchDescriptionPopover from '@/components/ui/TouchDescriptionPopover'
import AppSettings from '@/components/AppSettings'
import ApiKeyAlert, { type ApiKeyAlertCloseReason } from '@/components/ApiKeyAlert'
import ErrorBoundary from '@/components/ErrorBoundary'
import { useAuthStore } from '@/stores/state'
import { getAuthStatus } from '@/api/lightrag'
import { useIdentityEpochStore } from '@/lib/loginIdentity'
import {
  markVersionCheckedFromLogin,
  wasVersionCheckedThisPageLoad
} from '@/lib/versionCheckCache'
import { activateSessionFromAuthStatus } from './authBootstrap'
import {
  handleApiKeyDialogClose,
  initialDialogAcknowledgement,
  runCredentialProbe,
  useCredentialProbeStore
} from './credentialProbe'
import { entryHomeHref } from '@/lib/pathPrefix'
import { navigationService } from '@/services/navigation'
import WorkspaceQueryView from './WorkspaceQueryView'

/**
 * The workspace entry's app shell: a slim header (brand, deployment title,
 * status & settings — username, theme, language, logout) above the query
 * view. NO document/graph/API-docs navigation, no QuerySettings sidebar, no
 * admin health widgets — this page's whole job is asking questions.
 *
 * Layout uses dynamic viewport height (h-dvh) + safe-area padding so the
 * composer stays visible above mobile browser chrome and the soft keyboard.
 */
export default function WorkspaceApp() {
  const { t } = useTranslation()
  const { isGuestMode, username, webuiTitle, webuiDescription } = useAuthStore()
  const [initializing, setInitializing] = useState(true)
  // Dialog visibility is DERIVED, not synced: the probe signals through a
  // monotonic counter and this records the count the user last dismissed, so
  // two consecutive failures with the IDENTICAL message still reopen it (and
  // no effect has to push state, which would cascade renders).
  //
  // The baseline is the counter as it stands at THIS mount, not zero: the
  // counter is module state that outlives the shell, and the shell can mount
  // again within the same page load (see `initialDialogAcknowledgement`).
  const [dismissedDialogRequests, setDismissedDialogRequests] = useState(
    initialDialogAcknowledgement
  )
  // Bumped by the cross-tab identity watch: remounting the query view drops
  // the live session state that belonged to the previous identity.
  const identityEpoch = useIdentityEpochStore((s) => s.epoch)
  const versionCheckRef = useRef(false) // Prevent duplicate calls in Vite dev mode

  // Refresh version/title info once, and reconcile the session with what the
  // server reports (mirrors the admin shell, minus the auto-login for
  // UNAUTHENTICATED visitors: on this entry the first way in is always the
  // welcome page's explicit action).
  useEffect(() => {
    const checkVersion = async () => {
      if (versionCheckRef.current) return
      versionCheckRef.current = true

      // Skip only the request the login/welcome page just made in THIS page
      // load. A reload must reconcile again — the server's auth mode can have
      // changed under a still-valid stored token.
      if (wasVersionCheckedThisPageLoad()) {
        setInitializing(false)
        return
      }

      try {
        const token = localStorage.getItem('LIGHTRAG-API-TOKEN')
        const status = await getAuthStatus()
        // Reconciliation (including the stale-token replacement an
        // auth-disabled server requires) lives in authBootstrap.
        activateSessionFromAuthStatus(status, token)
        markVersionCheckedFromLogin()
      } catch (error) {
        console.error('Failed to get version info:', error)
      } finally {
        setInitializing(false)
      }
    }

    checkVersion()
  }, [])

  // Startup probe: an API-key-only deployment rejects queries until a key is
  // stored, and this entry has no other place to enter one.
  useEffect(() => {
    if (initializing) return
    runCredentialProbe()
  }, [initializing])

  // Requests come from the startup probe AND from a query rejected on
  // credential grounds (the server key can be rotated long after startup).
  const apiKeyDialogRequests = useCredentialProbeStore((s) => s.apiKeyDialogRequests)
  const apiKeyAlertOpen = apiKeyDialogRequests > dismissedDialogRequests

  const handleApiKeyAlertOpenChange = useCallback(
    (open: boolean, reason?: ApiKeyAlertCloseReason) => {
      if (open) return // opening is driven by the probe's request counter
      // Record the dismissal so the dialog stays closed until a NEW request;
      // whether the close is worth re-verifying is the probe's decision.
      setDismissedDialogRequests(useCredentialProbeStore.getState().apiKeyDialogRequests)
      handleApiKeyDialogClose(reason)
    },
    []
  )

  const handleLogout = () => {
    navigationService.navigateToUnauthenticated()
  }

  return (
    <main
      className="flex h-dvh w-screen flex-col overflow-hidden"
      style={{
        paddingTop: 'env(safe-area-inset-top)',
        paddingLeft: 'env(safe-area-inset-left)',
        paddingRight: 'env(safe-area-inset-right)',
        paddingBottom: 'env(safe-area-inset-bottom)'
      }}
    >
      <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50 flex h-11 w-full shrink-0 items-center border-b px-3 backdrop-blur md:px-4">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          {/* Document-relative brand link: resolves to THIS entry's own root
              under any proxy prefix — never to /webui, and never to the admin
              index.html that the dev server answers `/` with. */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <a href={entryHomeHref(window.location.pathname)} className="flex shrink-0 items-center gap-2">
                  <ZapIcon className="size-4 text-emerald-400" aria-hidden="true" />
                  <span className="font-bold">{webuiTitle || 'LightRAG'}</span>
                </a>
              </TooltipTrigger>
              {webuiDescription && (
                <TooltipContent side="bottom">
                  {webuiDescription}
                </TooltipContent>
              )}
            </Tooltip>
          </TooltipProvider>
          <TouchDescriptionPopover description={webuiDescription} />
          {isGuestMode && (
            <span className="rounded-md bg-amber-100 px-2 py-0.5 text-xs text-amber-800 dark:bg-amber-900 dark:text-amber-200">
              {t('login.guestMode', 'Guest Mode')}
            </span>
          )}
        </div>
        <nav className="flex shrink-0 items-center gap-1">
          {username && !isGuestMode && (
            <span className="text-muted-foreground mr-1 hidden max-w-[10rem] truncate text-xs sm:inline">
              {username}
            </span>
          )}
          <AppSettings />
          {!isGuestMode && (
            <Button
              variant="ghost"
              size="icon"
              side="bottom"
              className="min-h-11 min-w-11"
              tooltip={`${t('header.logout')}${username ? ` (${username})` : ''}`}
              aria-label={t('header.logout')}
              onClick={handleLogout}
            >
              <LogOutIcon className="size-4" aria-hidden="true" />
            </Button>
          )}
        </nav>
      </header>

      <div className="relative min-h-0 grow">
        {initializing ? (
          <div className="flex h-full items-center justify-center">
            <div
              className="border-primary size-8 animate-spin rounded-full border-4 border-t-transparent"
              role="status"
              aria-label="Loading"
            />
          </div>
        ) : (
          <ErrorBoundary>
            <WorkspaceQueryView key={identityEpoch} />
          </ErrorBoundary>
        )}
      </div>
      <ApiKeyAlert open={apiKeyAlertOpen} onOpenChange={handleApiKeyAlertOpenChange} />
    </main>
  )
}
