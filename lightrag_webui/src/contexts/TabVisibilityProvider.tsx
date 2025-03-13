import React, { useState, useEffect, useMemo } from 'react';
import { TabVisibilityContext } from './context';
import { TabVisibilityContextType } from './types';
import { useSettingsStore } from '@/stores/settings';

interface TabVisibilityProviderProps {
  children: React.ReactNode;
}

/**
 * Provider component for the TabVisibility context
 * Manages the visibility state of tabs throughout the application
 */
export const TabVisibilityProvider: React.FC<TabVisibilityProviderProps> = ({ children }) => {
  // Get current tab from settings store
  const currentTab = useSettingsStore.use.currentTab();
  
  // Initialize visibility state with current tab as visible
  const [visibleTabs, setVisibleTabs] = useState<Record<string, boolean>>(() => ({
    [currentTab]: true
  }));

  // Update visibility when current tab changes
  useEffect(() => {
    setVisibleTabs((prev) => ({
      ...prev,
      [currentTab]: true
    }));
  }, [currentTab]);

  // Create the context value with memoization to prevent unnecessary re-renders
  const contextValue = useMemo<TabVisibilityContextType>(
    () => ({
      visibleTabs,
      setTabVisibility: (tabId: string, isVisible: boolean) => {
        setVisibleTabs((prev) => ({
          ...prev,
          [tabId]: isVisible,
        }));
      },
      isTabVisible: (tabId: string) => !!visibleTabs[tabId],
    }),
    [visibleTabs]
  );

  return (
    <TabVisibilityContext.Provider value={contextValue}>
      {children}
    </TabVisibilityContext.Provider>
  );
};

export default TabVisibilityProvider;
