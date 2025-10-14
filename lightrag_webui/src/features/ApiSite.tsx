import { useState, useEffect } from 'react'
import { useTabVisibility } from '@/contexts/useTabVisibility'
import { backendBaseUrl } from '@/lib/constants'
import { useTranslation } from 'react-i18next'

export default function ApiSite() {
  const { t } = useTranslation()
  const { isTabVisible } = useTabVisibility()
  const isApiTabVisible = isTabVisible('api')
  const [iframeLoaded, setIframeLoaded] = useState(false)

  // Load the iframe once on component mount
  useEffect(() => {
    if (!iframeLoaded) {
      setIframeLoaded(true)
    }
  }, [iframeLoaded])

  // Use CSS to hide content when tab is not visible
  return (
    <div className={`size-full ${isApiTabVisible ? '' : 'hidden'}`}>
      {iframeLoaded ? (
        <iframe
          src={backendBaseUrl + '/redoc'}
          className="size-full h-full w-full"
          style={{ width: '100%', height: '100%', border: 'none' }}
          // Use key to ensure iframe doesn't reload
          key="api-docs-iframe"
        />
      ) : (
        <div className="bg-background flex h-full w-full items-center justify-center">
          <div className="text-center">
            <div className="border-primary mb-2 h-8 w-8 animate-spin rounded-full border-4 border-t-transparent"></div>
            <p>{t('apiSite.loading')}</p>
          </div>
        </div>
      )}
    </div>
  )
}
