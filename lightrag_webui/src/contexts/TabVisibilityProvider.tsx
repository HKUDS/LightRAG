import React, { useState, useMemo } from 'react';
import { TabVisibilityContext } from './context';
import { TabVisibilityContextType } from './types';

interface TabVisibilityProviderProps {
  children: React.ReactNode;
}

/**
 * Provider component for the TabVisibility context
 * Manages the visibility state of tabs throughout the application
 */
export const TabVisibilityProvider: React.FC<TabVisibilityProviderProps> = ({ children }) => {
  const [visibleTabs, setVisibleTabs] = useState<Record<string, boolean>>({});

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
