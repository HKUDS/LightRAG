import { useSyncExternalStore } from 'react'
import { useSettingsStore } from '@/stores/settings'

// Single source of truth for "is the UI effectively in dark mode right now",
// resolving theme === 'system' against the OS preference. Reactive: re-renders
// consumers both when the theme setting changes and (in system mode) when the
// OS color scheme is toggled. This mirrors how ThemeProvider writes the
// dark/light class, but reads matchMedia directly so it does not depend on the
// class write having happened first.
const darkMql = () => window.matchMedia('(prefers-color-scheme: dark)')

const subscribe = (onChange: () => void) => {
  const mql = darkMql()
  mql.addEventListener('change', onChange)
  return () => mql.removeEventListener('change', onChange)
}

const getSnapshot = () => darkMql().matches

const useIsDarkMode = (): boolean => {
  const theme = useSettingsStore.use.theme()
  const systemDark = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
  return theme === 'dark' ? true : theme === 'light' ? false : systemDark
}

export default useIsDarkMode
