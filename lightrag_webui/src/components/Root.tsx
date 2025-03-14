import { StrictMode, useEffect, useState } from 'react'
import { initializeI18n } from '@/i18n'
import App from '@/App'

export const Root = () => {
  const [isI18nInitialized, setIsI18nInitialized] = useState(false)

  useEffect(() => {
    // Initialize i18n immediately with persisted language
    initializeI18n().then(() => {
      setIsI18nInitialized(true)
    })
  }, [])

  if (!isI18nInitialized) {
    return null // or a loading spinner
  }

  return (
    <StrictMode>
      <App />
    </StrictMode>
  )
}
