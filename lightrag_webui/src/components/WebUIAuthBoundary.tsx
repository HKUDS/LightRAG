import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { getAuthStatus } from '@/api/lightrag'
import { useAuthStore } from '@/stores/state'
import { applyWebUIAuthStatus, useWebUIAuthStore } from '@/stores/webuiAuth'
import Button from '@/components/ui/Button'

/** Discover capabilities before either entry mounts protected screens or probes. */
export default function WebUIAuthBoundary({ children }: { children: ReactNode }) {
  const { t } = useTranslation()
  const [checking, setChecking] = useState(true)
  const mode = useWebUIAuthStore((state) => state.mode)
  const isGuest = useAuthStore((state) => state.isGuestMode)
  const check = useCallback(async () => {
    try {
      const status = await getAuthStatus()
      applyWebUIAuthStatus(status)
      if (status.auth_configured && useAuthStore.getState().isGuestMode) {
        // A guest token never becomes an account identity after a restart.
        useAuthStore.getState().logout()
      }
    } catch {
      useWebUIAuthStore.setState({ mode: 'unknown' })
    } finally {
      setChecking(false)
    }
  }, [])

  useEffect(() => { void check() }, [check])

  useEffect(() => {
    if (mode === 'account' && isGuest) useAuthStore.getState().logout()
  }, [mode, isGuest])

  if (checking || (mode === 'account' && isGuest)) {
    return <main role="status" className="p-8">{t('webuiAuth.checking')}</main>
  }
  if (mode === 'api-key-only' || mode === 'unknown') {
    return (
      <main className="flex min-h-dvh items-center justify-center p-6">
        <section role="alert" className="max-w-xl space-y-4">
          <h1 className="text-xl font-semibold">{t(`webuiAuth.${mode === 'api-key-only' ? 'title' : 'unavailableTitle'}`)}</h1>
          <p>{t(`webuiAuth.${mode === 'api-key-only' ? 'guidance' : 'unavailable'}`)}</p>
          <Button onClick={() => { setChecking(true); void check() }}>
            {t('webuiAuth.retry')}
          </Button>
        </section>
      </main>
    )
  }
  return children
}
