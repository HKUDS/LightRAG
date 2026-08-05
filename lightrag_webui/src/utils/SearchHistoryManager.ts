import { searchHistoryMaxItems, searchHistoryVersion } from '@/lib/constants'

/**
 * SearchHistoryManager - Manages search history persistence in localStorage
 *
 * This utility class handles:
 * - Storing and retrieving search history from localStorage
 * - Managing history size limits
 * - Sorting by access time and frequency
 * - Version compatibility
 */

export interface SearchHistoryItem {
  label: string           // Label name
  lastAccessed: number   // Last access timestamp
  accessCount: number    // Access count for sorting optimization
}

export interface SearchHistoryData {
  items: SearchHistoryItem[]
  version: string        // Data version for compatibility
  workspace?: string     // Workspace isolation (if needed)
}

export class SearchHistoryManager {
  private static readonly STORAGE_KEY = 'lightrag_search_history'
  private static readonly MAX_HISTORY = searchHistoryMaxItems
  private static readonly VERSION = searchHistoryVersion

  /**
   * Get search history from localStorage
   * @returns Array of search history items sorted by last accessed time (descending)
   */
  static getHistory(): SearchHistoryItem[] {
    try {
      const data = localStorage.getItem(this.STORAGE_KEY)
      if (!data) return []

      const parsed: SearchHistoryData = JSON.parse(data)

      // Version compatibility check
      if (parsed.version !== this.VERSION) {
        console.warn(`Search history version mismatch. Expected ${this.VERSION}, got ${parsed.version}. Clearing history.`)
        this.clearHistory()
        return []
      }

      // Ensure items is an array
      if (!Array.isArray(parsed.items)) {
        console.warn('Invalid search history format. Clearing history.')
        this.clearHistory()
        return []
      }

      // Sort by last accessed time (descending) then by access count (descending)
      return parsed.items.sort((a, b) => {
        if (b.lastAccessed !== a.lastAccessed) {
          return b.lastAccessed - a.lastAccessed
        }
        return (b.accessCount || 0) - (a.accessCount || 0)
      })
    } catch (error) {
      console.error('Error reading search history:', error)
      this.clearHistory()
      return []
    }
  }

  /**
   * Add a label to search history (or update if exists)
   * @param label Label to add to history
   */
  static addToHistory(label: string): void {
    if (!label || typeof label !== 'string' || label.trim() === '') {
      return
    }

    try {
      const history = this.getHistory()
      const now = Date.now()
      const trimmedLabel = label.trim()

      // Find existing item
      const existingIndex = history.findIndex(item => item.label === trimmedLabel)

      if (existingIndex >= 0) {
        // Update existing item
        const existingItem = history[existingIndex]
        existingItem.lastAccessed = now
        existingItem.accessCount = (existingItem.accessCount || 0) + 1

        // Move to front (will be sorted properly when saved)
        history.splice(existingIndex, 1)
        history.unshift(existingItem)
      } else {
        // Add new item to the beginning
        history.unshift({
          label: trimmedLabel,
          lastAccessed: now,
          accessCount: 1
        })
      }

      // Limit history size
      if (history.length > this.MAX_HISTORY) {
        history.splice(this.MAX_HISTORY)
      }

      // Save to localStorage
      const data: SearchHistoryData = {
        items: history,
        version: this.VERSION
      }

      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(data))
    } catch (error) {
      console.error('Error saving search history:', error)
    }
  }

  /**
   * Clear all search history
   */
  static clearHistory(): void {
    try {
      localStorage.removeItem(this.STORAGE_KEY)
    } catch (error) {
      console.error('Error clearing search history:', error)
    }
  }

  /**
   * Refresh the search-history dropdown with the latest popular labels from
   * the backend, while preserving user-entered search entries (accessCount > 0).
   *
   * - On first load (empty history): initialise with popular labels only.
   * - On subsequent loads: keep user-typed entries, replace the popular-label
   *   section (accessCount === 0) with fresh API data.
   * - When the API returns an empty list (all documents deleted), the popular
   *   section becomes empty — stale entity names from deleted docs are purged.
   */
  static async initializeWithDefaults(popularLabels: string[]): Promise<void> {
    try {
      const now = Date.now()
      const popularItems: SearchHistoryItem[] = popularLabels.map((label, index) => ({
        label: label.trim(),
        lastAccessed: now - index,
        accessCount: 0, // 0 = API-populated defaults, >0 = user-entered
      }))

      const history = this.getHistory()
      if (history.length === 0) {
        // First load — just write the popular labels directly.
        const data: SearchHistoryData = { items: popularItems, version: this.VERSION }
        localStorage.setItem(this.STORAGE_KEY, JSON.stringify(data))
        return
      }

      // Merge: keep user-entered items, replace popular-label items with
      // the latest API data.  This is the key fix — stale entity names
      // from deleted documents are purged because they no longer appear
      // in the API response.
      const userItems = history.filter((item) => item.accessCount > 0)
      const merged = [...popularItems, ...userItems]
        .sort((a, b) => b.lastAccessed - a.lastAccessed)
        .slice(0, searchHistoryMaxItems)

      const data: SearchHistoryData = { items: merged, version: this.VERSION }
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(data))
    } catch (error) {
      console.error('Error initializing search history with defaults:', error)
    }
  }

  /**
   * Get recent searches (items with accessCount > 0)
   * @param limit Maximum number of recent searches to return
   * @returns Array of recent search items
   */
  static getRecentSearches(limit: number = 10): SearchHistoryItem[] {
    const history = this.getHistory()
    return history
      .filter(item => item.accessCount > 0)
      .slice(0, limit)
  }

  /**
   * Get popular recommendations (items with accessCount = 0, i.e., defaults)
   * @param limit Maximum number of recommendations to return
   * @returns Array of popular recommendation items
   */
  static getPopularRecommendations(limit?: number): SearchHistoryItem[] {
    const history = this.getHistory()
    const recommendations = history.filter(item => item.accessCount === 0)
    return limit ? recommendations.slice(0, limit) : recommendations
  }

  /**
   * Get all history items as simple string array
   * @param limit Maximum number of items to return
   * @returns Array of label strings
   */
  static getHistoryLabels(limit?: number): string[] {
    const history = this.getHistory()
    const labels = history.map(item => item.label)
    return limit ? labels.slice(0, limit) : labels
  }

  /**
   * Check if a label exists in history
   * @param label Label to check
   * @returns True if label exists in history
   */
  static hasLabel(label: string): boolean {
    if (!label || typeof label !== 'string') return false
    const history = this.getHistory()
    return history.some(item => item.label === label.trim())
  }

  /**
   * Remove a specific label from history
   * @param label Label to remove
   */
  static removeLabel(label: string): void {
    if (!label || typeof label !== 'string') return

    try {
      const history = this.getHistory()
      const trimmedLabel = label.trim()
      const filteredHistory = history.filter(item => item.label !== trimmedLabel)

      if (filteredHistory.length !== history.length) {
        const data: SearchHistoryData = {
          items: filteredHistory,
          version: this.VERSION
        }

        localStorage.setItem(this.STORAGE_KEY, JSON.stringify(data))
      }
    } catch (error) {
      console.error('Error removing label from search history:', error)
    }
  }

  /**
   * Get storage statistics
   * @returns Object with history statistics
   */
  static getStats(): {
    totalItems: number
    recentSearches: number
    popularRecommendations: number
    storageSize: number
  } {
    const history = this.getHistory()
    const recentCount = history.filter(item => item.accessCount > 0).length
    const popularCount = history.filter(item => item.accessCount === 0).length

    let storageSize = 0
    try {
      const data = localStorage.getItem(this.STORAGE_KEY)
      storageSize = data ? data.length : 0
    } catch {
      // Ignore error
    }

    return {
      totalItems: history.length,
      recentSearches: recentCount,
      popularRecommendations: popularCount,
      storageSize
    }
  }
}
