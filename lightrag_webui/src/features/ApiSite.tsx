import { useState, useEffect, useRef } from 'react'
import { useTabVisibility } from '@/contexts/useTabVisibility'
import { backendBaseUrl } from '@/lib/constants'
import { useTranslation } from 'react-i18next'

const getRootTheme = (): 'light' | 'dark' =>
  window.document.documentElement.classList.contains('dark') ? 'dark' : 'light'

export default function ApiSite() {
  const { t } = useTranslation()
  const { isTabVisible } = useTabVisibility()
  const isApiTabVisible = isTabVisible('api')
  const [iframeLoaded, setIframeLoaded] = useState(false)
  const [docsTheme, setDocsTheme] = useState<'light' | 'dark'>(getRootTheme)
  // Freeze the initial theme so the iframe src never changes after mount;
  // subsequent theme switches are pushed via postMessage to avoid reloading
  // the entire Swagger UI on every toggle.
  const [initialTheme] = useState(docsTheme)
  const iframeRef = useRef<HTMLIFrameElement>(null)

  useEffect(() => {
    const timer = setTimeout(() => setIframeLoaded(true), 0)
    return () => clearTimeout(timer)
  }, [])

  useEffect(() => {
    const root = window.document.documentElement
    const syncDocsTheme = () => setDocsTheme(getRootTheme())

    syncDocsTheme()
    const observer = new MutationObserver(syncDocsTheme)
    observer.observe(root, { attributes: true, attributeFilter: ['class'] })

    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    iframeRef.current?.contentWindow?.postMessage(
      { type: 'lightrag:set-docs-theme', theme: docsTheme },
      '*'
    )
  }, [docsTheme])

  const handleIframeLoad = () => {
    iframeRef.current?.contentWindow?.postMessage(
      { type: 'lightrag:set-docs-theme', theme: docsTheme },
      '*'
    )
  }

  // Use CSS to hide content when tab is not visible
  return (
    <div className={`size-full ${isApiTabVisible ? '' : 'hidden'}`}>
      {iframeLoaded ? (
        <iframe
          ref={iframeRef}
          src={`${backendBaseUrl}/docs?theme=${initialTheme}`}
          className="size-full w-full h-full"
          style={{ width: '100%', height: '100%', border: 'none' }}
          onLoad={handleIframeLoad}
          key="api-docs-iframe"
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center bg-background">
          <div className="text-center">
            <div className="mb-2 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
            <p>{t('apiSite.loading')}</p>
          </div>
        </div>
      )}
    </div>
  )
}
