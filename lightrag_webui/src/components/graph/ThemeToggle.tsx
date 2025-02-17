import Button from '@/components/ui/Button'
import useTheme from '@/hooks/useTheme'
import { MoonIcon, SunIcon } from 'lucide-react'
import { useCallback } from 'react'
import { controlButtonVariant } from '@/lib/constants'

/**
 * Component that toggles the theme between light and dark.
 */
export default function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const setLight = useCallback(() => setTheme('light'), [setTheme])
  const setDark = useCallback(() => setTheme('dark'), [setTheme])

  if (theme === 'dark') {
    return (
      <Button
        onClick={setLight}
        variant={controlButtonVariant}
        tooltip="Switch to light theme"
        size="icon"
        side="bottom"
      >
        <MoonIcon />
      </Button>
    )
  }
  return (
    <Button
      onClick={setDark}
      variant={controlButtonVariant}
      tooltip="Switch to dark theme"
      size="icon"
      side="bottom"
    >
      <SunIcon />
    </Button>
  )
}
