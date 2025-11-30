import { useTabVisibility } from '@/contexts/useTabVisibility'
import { backendBaseUrl } from '@/lib/constants'
import { useEffect, useState } from 'react'
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
          src={backendBaseUrl + '/docs'}
          className="size-full w-full h-full"
          style={{ width: '100%', height: '100%', border: 'none' }}
          // Use key to ensure iframe doesn't reload
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
