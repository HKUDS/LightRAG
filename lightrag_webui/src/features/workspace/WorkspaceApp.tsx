import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ZapIcon, LogOutIcon } from 'lucide-react'
import Button from '@/components/ui/Button'
import AppSettings from '@/components/AppSettings'
import ApiKeyAlert from '@/components/ApiKeyAlert'
import ErrorBoundary from '@/components/ErrorBoundary'
import { useAuthStore, useBackendState } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { getAuthStatus } from '@/api/lightrag'
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
  const [apiKeyAlertOpen, setApiKeyAlertOpen] = useState(false)
  const apiKey = useSettingsStore.use.apiKey()
  const versionCheckRef = useRef(false) // Prevent duplicate calls in Vite dev mode

  // Refresh version/title info once (mirrors the admin shell, minus the
  // guest auto-login: on this entry guest activation happens ONLY through
  // the welcome page's explicit action).
  useEffect(() => {
    const checkVersion = async () => {
      if (versionCheckRef.current) return
      versionCheckRef.current = true

      const versionCheckedFromLogin =
        sessionStorage.getItem('VERSION_CHECKED_FROM_LOGIN') === 'true'
      if (versionCheckedFromLogin) {
        setInitializing(false)
        return
      }

      try {
        const token = localStorage.getItem('LIGHTRAG-API-TOKEN')
        const status = await getAuthStatus()
        if (
          token &&
          (status.core_version ||
            status.api_version ||
            status.webui_title ||
            status.webui_description)
        ) {
          const isGuest =
            status.auth_mode === 'disabled' || useAuthStore.getState().isGuestMode
          useAuthStore
            .getState()
            .login(
              token,
              isGuest,
              status.core_version,
              status.api_version,
              status.webui_title || null,
              status.webui_description || null
            )
        }
        sessionStorage.setItem('VERSION_CHECKED_FROM_LOGIN', 'true')
      } catch (error) {
        console.error('Failed to get version info:', error)
      } finally {
        setInitializing(false)
      }
    }

    checkVersion()
  }, [])

  // Credential probe — once after initialization and again whenever the
  // stored API key changes. In an API-key-only deployment (LIGHTRAG_API_KEY
  // without AUTH_ACCOUNTS) the guest token alone never authenticates, so a
  // fresh browser would otherwise only ever see 403s: the probe surfaces the
  // API-key error in the backend state, and the mounted ApiKeyAlert (which
  // watches that message) opens to capture the key. This is the workspace
  // counterpart of the admin shell's periodic health check — a single probe,
  // not a polling loop.
  useEffect(() => {
    if (initializing) return
    void useBackendState.getState().check()
  }, [initializing, apiKey])

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
          {/* Document-relative brand link: resolves to THIS entry's mount
              root under any proxy prefix — never to /webui. */}
          <a href="./" className="flex shrink-0 items-center gap-2">
            <ZapIcon className="size-4 text-emerald-400" aria-hidden="true" />
            <span className="font-bold">{webuiTitle || 'LightRAG'}</span>
          </a>
          {webuiDescription && (
            <span className="text-muted-foreground hidden truncate text-xs md:inline">
              {webuiDescription}
            </span>
          )}
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
            <WorkspaceQueryView />
          </ErrorBoundary>
        )}
      </div>
      <ApiKeyAlert open={apiKeyAlertOpen} onOpenChange={setApiKeyAlertOpen} />
    </main>
  )
}
