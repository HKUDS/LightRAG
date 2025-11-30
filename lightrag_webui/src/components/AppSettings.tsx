import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { MoonIcon, SunIcon } from 'lucide-react'
import { useCallback } from 'react'
import { useTranslation } from 'react-i18next'

interface AppSettingsProps {
  className?: string
}

export default function AppSettings({ className }: AppSettingsProps) {
  const { t } = useTranslation()

  const theme = useSettingsStore.use.theme()
  const setTheme = useSettingsStore.use.setTheme()

  // Compute effective theme for icon/tooltip display when theme is 'system'
  const effectiveTheme =
    theme === 'system'
      ? window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light'
      : theme

  const handleThemeToggle = useCallback(() => {
    if (theme === 'system') {
      // Detect actual system preference and toggle to opposite
      const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      setTheme(isDark ? 'light' : 'dark')
    } else {
      setTheme(theme === 'dark' ? 'light' : 'dark')
    }
  }, [theme, setTheme])

  return (
    <Button
      variant="ghost"
      size="icon"
      className={cn('h-9 w-9', className)}
      onClick={handleThemeToggle}
      tooltip={effectiveTheme === 'dark' ? t('settings.light') : t('settings.dark')}
    >
      {effectiveTheme === 'dark' ? (
        <MoonIcon className="h-5 w-5" />
      ) : (
        <SunIcon className="h-5 w-5" />
      )}
    </Button>
  )
}
