import { createContext } from 'react';
import { TabVisibilityContextType } from './types';

// Default context value
const defaultContext: TabVisibilityContextType = {
  visibleTabs: {},
  setTabVisibility: () => {},
  isTabVisible: () => false,
};

// Create the context
export const TabVisibilityContext = createContext<TabVisibilityContextType>(defaultContext);
