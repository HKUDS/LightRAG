import React from 'react'
import { ErrorBoundary as ReactErrorBoundary, type FallbackProps } from 'react-error-boundary'
import { useTranslation } from 'react-i18next'
import { AlertTriangleIcon } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/Alert'
import Button from '@/components/ui/Button'

/**
 * Fallback rendered when a child subtree throws during render. It must never
 * throw itself, so every translation lookup carries an inline English default
 * (i18n keys may be missing in some locales, and the throw can happen before
 * translations are ready).
 */
const DefaultFallback: React.FC<FallbackProps> = ({ error, resetErrorBoundary }) => {
  const { t } = useTranslation()
  const message = error instanceof Error ? error.message : String(error)

  return (
    <div className="flex h-full w-full items-center justify-center p-4">
      <Alert variant="destructive" className="max-w-md">
        <AlertTriangleIcon className="h-4 w-4" />
        <AlertTitle>{t('common.errorBoundaryTitle', 'Something went wrong')}</AlertTitle>
        <AlertDescription className="flex flex-col gap-3">
          <span>
            {t(
              'common.errorBoundaryDescription',
              'This panel failed to render. Try again, or switch to another tab.'
            )}
          </span>
          {message && <span className="text-xs opacity-70 break-words">{message}</span>}
          <Button variant="outline" size="sm" className="self-start" onClick={resetErrorBoundary}>
            {t('common.errorBoundaryRetry', 'Try again')}
          </Button>
        </AlertDescription>
      </Alert>
    </div>
  )
}

interface ErrorBoundaryProps {
  children: React.ReactNode
}

/**
 * Wraps a subtree so a render-time throw degrades to a recoverable fallback
 * instead of unmounting the whole SPA (React tears down #root on an uncaught
 * render error, and there is otherwise no boundary in the app).
 */
const ErrorBoundary: React.FC<ErrorBoundaryProps> = ({ children }) => (
  <ReactErrorBoundary FallbackComponent={DefaultFallback}>{children}</ReactErrorBoundary>
)

export default ErrorBoundary
