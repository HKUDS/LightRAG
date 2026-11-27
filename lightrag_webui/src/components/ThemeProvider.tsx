import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { ThemeProviderContext } from '@/contexts/ThemeProviderContext'
import { useSettingsStore } from '@/stores/settings'

type ThemeProviderProps = {
  children: React.ReactNode
}

/**
 * Component that provides the theme state and setter function to its children.
 */
export default function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  const theme = useSettingsStore.use.theme()
  const setTheme = useSettingsStore.use.setTheme()
  const { i18n } = useTranslation()

  // WCAG 3.1.1: the document language must follow the active UI language,
  // not the HTML shell's static pre-hydration default. Locale ids use
  // underscores (zh_TW); BCP-47 wants hyphens (zh-TW).
  useEffect(() => {
    const syncLang = (language: string) => {
      window.document.documentElement.lang = language.replace(/_/g, '-')
    }
    syncLang(i18n.language)
    i18n.on('languageChanged', syncLang)
    return () => {
      i18n.off('languageChanged', syncLang)
    }
  }, [i18n])

  useEffect(() => {
    const root = window.document.documentElement
    root.classList.remove('light', 'dark')

    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
      const handleChange = (e: MediaQueryListEvent) => {
        root.classList.remove('light', 'dark')
        root.classList.add(e.matches ? 'dark' : 'light')
      }

      root.classList.add(mediaQuery.matches ? 'dark' : 'light')
      mediaQuery.addEventListener('change', handleChange)

      return () => mediaQuery.removeEventListener('change', handleChange)
    } else {
      root.classList.add(theme)
    }
  }, [theme])

  const value = {
    theme,
    setTheme
  }

  return (
    <ThemeProviderContext.Provider {...props} value={value}>
      {children}
    </ThemeProviderContext.Provider>
  )
}
