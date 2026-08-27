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
 * - a temporarily failing re-request keeps the last successful snapshot;
 * - correctness under fast switching (A → B while B is in flight → back to
 *   A): a response is applied only when it is BOTH the newest request AND for
 *   the locale the user still targets. Re-targeting a locale whose snapshot
 *   is already loaded issues no network request but still invalidates every
 *   response in flight — otherwise B's late response would land while A is
 *   selected.
 */

export type CustomizationStatus = 'loading' | 'ready' | 'error'

/**
 * Sentinel target locale: request the customization WITHOUT a `locale`
 * parameter, so the server resolves the bundle's `default_locale`. Used when
 * the user never explicitly selected a language and the browser's languages
 * match none of the UI locales.
 */
export const SERVER_DEFAULT_LOCALE = ''

interface CustomizationState {
  status: CustomizationStatus
  /** Last successful response (customized or not); null before the first. */
  snapshot: UICustomization | null
  /** Locale the user currently targets: last `load` argument (internal id,
   * e.g. zh_TW, or SERVER_DEFAULT_LOCALE); null before the first load. */
  targetLocale: string | null
  /** Locale the current snapshot was loaded for. */
  loadedLocale: string | null
  load: (locale: string) => Promise<void>
}

let requestCounter = 0

export const useCustomizationStore = create<CustomizationState>((set, get) => ({
  status: 'loading',
  snapshot: null,
  targetLocale: null,
  loadedLocale: null,

  load: async (locale: string) => {
    // Every (re-)target bumps the counter, so responses for previously
    // targeted locales can never be applied afterwards — even when this call
    // itself needs no network request.
    const requestId = ++requestCounter
    set({ targetLocale: locale })
    if (get().loadedLocale === locale && get().snapshot !== null) {
      // Already showing this locale. The counter bump above just invalidated
      // any other locale's in-flight response (the A → B → A race).
      return
    }
    if (get().snapshot === null) {
      set({ status: 'loading' })
    }
    try {
      const snapshot = await fetchUICustomization(
        locale === SERVER_DEFAULT_LOCALE ? null : locale
      )
      // Apply only when still the newest request AND for the current target.
      if (requestId !== requestCounter || get().targetLocale !== locale) return
      // Atomic swap: one set() replaces the whole locale representation.
      set({ status: 'ready', snapshot, loadedLocale: locale })
    } catch (error) {
      if (requestId !== requestCounter || get().targetLocale !== locale) return
      console.error('Failed to load UI customization:', error)
      if (get().snapshot === null) {
        // Hard failure on first load → frontend default content takes over.
        set({ status: 'error' })
      }
      // With a previous snapshot: keep it (status stays 'ready').
    }
  }
}))
