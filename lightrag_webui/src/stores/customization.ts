import { create } from 'zustand'
import { fetchUICustomization, type UICustomization } from '@/api/customization'

/**
 * In-memory snapshot of the UI customization for the current page.
 * NEVER persisted to localStorage — a deployment update must not leave stale
 * branding behind (`no-store` on the endpoint plus memory-only caching here).
 *
 * Loading rules (workspace-entry PRD §8.8):
 * - the FIRST load shows a loading placeholder; frontend default content is
 *   rendered only after a definite `customized: false` or a hard failure —
 *   a bundle-configured deployment must never flash the default first;
 * - a language switch keeps the current complete content until the new
 *   response succeeds, then swaps logo, alt text and both texts atomically;
 * - a temporarily failing re-request keeps the last successful snapshot.
 */

export type CustomizationStatus = 'loading' | 'ready' | 'error'

interface CustomizationState {
  status: CustomizationStatus
  /** Last successful response (customized or not); null before the first. */
  snapshot: UICustomization | null
  /** Language (internal id, e.g. zh_TW) the snapshot was loaded for. */
  loadedLanguage: string | null
  load: (language: string) => Promise<void>
}

let requestCounter = 0

export const useCustomizationStore = create<CustomizationState>((set, get) => ({
  status: 'loading',
  snapshot: null,
  loadedLanguage: null,

  load: async (language: string) => {
    const requestId = ++requestCounter
    if (get().snapshot === null) {
      set({ status: 'loading' })
    }
    try {
      const snapshot = await fetchUICustomization(language)
      if (requestId !== requestCounter) return // superseded by a newer request
      // Atomic swap: one set() replaces the whole locale representation.
      set({ status: 'ready', snapshot, loadedLanguage: language })
    } catch (error) {
      if (requestId !== requestCounter) return
      console.error('Failed to load UI customization:', error)
      if (get().snapshot === null) {
        // Hard failure on first load → frontend default content takes over.
        set({ status: 'error' })
      }
      // With a previous snapshot: keep it (status stays 'ready').
    }
  }
}))
