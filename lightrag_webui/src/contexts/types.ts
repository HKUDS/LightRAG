export interface TabVisibilityContextType {
  visibleTabs: Record<string, boolean>
  setTabVisibility: (tabId: string, isVisible: boolean) => void
  isTabVisible: (tabId: string) => boolean
}
