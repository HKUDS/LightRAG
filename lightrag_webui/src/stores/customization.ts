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
 * parameter, so the server resolves the bundle's `default_locale`.
 *
 * NOT what the UI surfaces send. They resolve a concrete language through
 * `resolveUiLanguage`, because the UI chrome cannot wait for an async
 * default: letting the bundle pick here left English buttons beside branding
 * in the bundle's own default locale. A caller that genuinely wants the
 * server to choose (and has nothing else to keep in sync) can still use it —
 * the server falls back to `default_locale` for an undeclared locale anyway.
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
  /** Locale of the NEWEST in-flight request, or null when none is running.
   * Lets repeated `load` calls for the same locale (several components
   * mounting, or the retry gate re-firing while the request is still out)
   * dedupe instead of issuing a request storm. */
  pendingLocale: string | null
  /** Failed attempts for `targetLocale` since it was last targeted. Bounds
   * the automatic retry: without a cap, re-arming on completion would loop
   * (fail → state change → gate re-fires → fail …) and storm the endpoint. */
  failedAttempts: number
  load: (locale: string) => Promise<void>
}

/** Automatic attempts per target locale: the first one plus ONE retry. */
export const MAX_CUSTOMIZATION_ATTEMPTS = 2

/**
 * Whether the hook should ask the store to (re-)load `locale`. Pure so the
 * retry contract is testable without a DOM:
 * - target moved (a language switch, or the A → B → A re-target whose whole
 *   job is invalidating B's in-flight response);
 * - OR this locale is not the one actually LOADED — the retry case: a failed
 *   request leaves `targetLocale` already equal to it, so keying on the
 *   target alone would never try again (transient first-load failure = stuck
 *   on default branding; failed switch = stuck on the previous locale).
 * In-flight duplicates are absorbed by `pendingLocale` inside `load`.
 */
export function needsCustomizationLoad(
  locale: string,
  targetLocale: string | null,
  loadedLocale: string | null,
  failedAttempts = 0
): boolean {
  if (targetLocale !== locale) return true
  if (loadedLocale === locale) return false
  // Targeted but not loaded: in flight (deduped in `load`) or failed. Retry
  // only while attempts remain, so a persistently failing endpoint settles
  // on the default content instead of being hammered.
  return failedAttempts < MAX_CUSTOMIZATION_ATTEMPTS
}

let requestCounter = 0

export const useCustomizationStore = create<CustomizationState>((set, get) => ({
  status: 'loading',
  snapshot: null,
  targetLocale: null,
  loadedLocale: null,
  pendingLocale: null,
  failedAttempts: 0,

  load: async (locale: string) => {
    if (get().pendingLocale === locale) {
      // The NEWEST in-flight request is already for exactly this locale
      // (pendingLocale is cleared by any newer request below, so it can only
      // describe the current one). Re-assert the target and let it finish —
      // bumping the counter here would invalidate the very request we are
      // waiting on and leave nothing to replace it.
      set({ targetLocale: locale })
      return
    }
    // Every (re-)target bumps the counter, so responses for previously
    // targeted locales can never be applied afterwards — even when this call
    // itself needs no network request. Any older in-flight request is now
    // stale, so it no longer owns `pendingLocale`.
    const requestId = ++requestCounter
    // A NEW target starts its own attempt budget.
    set({
      targetLocale: locale,
      pendingLocale: null,
      ...(get().targetLocale === locale ? {} : { failedAttempts: 0 })
    })
    if (get().loadedLocale === locale && get().snapshot !== null) {
      // Already showing this locale. The counter bump above just invalidated
      // any other locale's in-flight response (the A → B → A race).
      return
    }
    if (get().snapshot === null) {
      set({ status: 'loading' })
    }
    set({ pendingLocale: locale })
    try {
      const snapshot = await fetchUICustomization(
        locale === SERVER_DEFAULT_LOCALE ? null : locale
      )
      // Apply only when still the newest request AND for the current target.
      if (requestId !== requestCounter || get().targetLocale !== locale) return
      // Atomic swap: one set() replaces the whole locale representation.
      set({ status: 'ready', snapshot, loadedLocale: locale, failedAttempts: 0 })
    } catch (error) {
      if (requestId !== requestCounter || get().targetLocale !== locale) return
      console.error('Failed to load UI customization:', error)
      set({ failedAttempts: get().failedAttempts + 1 })
      if (get().snapshot === null) {
        // Hard failure on first load → frontend default content takes over.
        set({ status: 'error' })
      }
      // With a previous snapshot: keep it (status stays 'ready').
    } finally {
      // Only the newest request owns the flag; a superseded one already had
      // it cleared by whoever superseded it.
      if (requestId === requestCounter) set({ pendingLocale: null })
    }
  }
}))
