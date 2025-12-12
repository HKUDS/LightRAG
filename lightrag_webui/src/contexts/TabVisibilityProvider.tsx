import type React from 'react'
import { useEffect, useMemo, useState } from 'react'
import { useSettingsStore } from '@/stores/settings'
import { TabVisibilityContext } from './context'
import type { TabVisibilityContextType } from './types'

interface TabVisibilityProviderProps {
  children: React.ReactNode
}

/**
 * Provider component for the TabVisibility context
 * Manages the visibility state of tabs throughout the application
 */
export const TabVisibilityProvider: React.FC<TabVisibilityProviderProps> = ({ children }) => {
  // Subscribe to current tab to trigger re-render when tab changes
  const currentTab = useSettingsStore.use.currentTab()
  void currentTab

  // Initialize visibility state with all tabs visible
  const [visibleTabs, setVisibleTabs] = useState<Record<string, boolean>>(() => ({
    documents: true,
    'knowledge-graph': true,
    retrieval: true,
    api: true,
  }))

  // Keep all tabs visible because we use CSS to control TAB visibility instead of React
  useEffect(() => {
    setVisibleTabs((prev) => ({
      ...prev,
      documents: true,
      'knowledge-graph': true,
      retrieval: true,
      api: true,
    }))
  }, [])

  // Create the context value with memoization to prevent unnecessary re-renders
  const contextValue = useMemo<TabVisibilityContextType>(
    () => ({
      visibleTabs,
      setTabVisibility: (tabId: string, isVisible: boolean) => {
        setVisibleTabs((prev) => ({
          ...prev,
          [tabId]: isVisible,
        }))
      },
      isTabVisible: (tabId: string) => !!visibleTabs[tabId],
    }),
    [visibleTabs]
  )

  return (
    <TabVisibilityContext.Provider value={contextValue}>{children}</TabVisibilityContext.Provider>
  )
}

export default TabVisibilityProvider
