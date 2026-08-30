import { backendBaseUrl } from '@/lib/constants'

/**
 * Client for the public, read-only UI customization API
 * (`GET /ui/customization`). Unauthenticated by design — the welcome page
 * shows customized content BEFORE login — and served with
 * `Cache-Control: no-store`, so every page load fetches it once.
 *
 * Uses plain fetch (not the axios instance): the endpoint is public, and the
 * axios interceptors' 401 handling must never be triggered by branding
 * content.
 */

export interface UICustomizationBrand {
  title?: string | null
  description?: string | null
  logo_url?: string | null
  logo_alt?: string
}

export interface UICustomizationContent {
  format: string
  content: string
}

export interface UICustomization {
  customized: boolean
  requested_locale?: string
  locale?: string
  fallback_used?: boolean
  direction?: 'ltr' | 'rtl'
  brand: UICustomizationBrand
  welcome?: UICustomizationContent
  query_empty?: UICustomizationContent
}

/**
 * SINGLE THROAT for locale normalization (workspace-entry PRD §8.2): the
 * frontend's internal language ids use underscores (`zh_TW`, matching the
 * src/locales/*.json filenames); the customization API accepts only the
 * BCP 47 hyphen form and rejects underscores. The conversion happens here,
 * once, on the way out — never on the server side.
 */
export const toBcp47Locale = (language: string): string => language.replace(/_/g, '-')

/**
 * How long this request may hang before it is abandoned. Matches the
 * `/auth-status` timeout in `api/lightrag.ts`, the sibling request the
 * welcome page fires beside this one.
 *
 * A bound is REQUIRED here, not a nicety: `fetch` has no default timeout, and
 * customization is optional content whose loading state gates a REQUIRED one.
 * The welcome page renders nothing but its spinner until this settles — no
 * "Sign in", no "Enter workspace" — so a route that accepts the connection
 * and never answers (a reverse proxy that has not been told about this new
 * path is the realistic case) leaves the workspace entry permanently
 * unreachable. Rejecting instead lets the store's first-load failure branch
 * fall back to the frontend default content, which is what an unconfigured
 * deployment shows anyway.
 *
 * The store retries once (`MAX_CUSTOMIZATION_ATTEMPTS`) before settling, so
 * the worst case a user waits is two of these, not forever.
 */
export const CUSTOMIZATION_REQUEST_TIMEOUT_MS = 5000

/**
 * `language: null` sends NO locale parameter — the server then resolves the
 * bundle's `default_locale`. Used when the user never explicitly selected a
 * language and the browser's languages match none of the UI locales
 * (language priority: explicit choice > browser language > bundle default).
 *
 * `signal` (a caller's own cancellation) and the timeout are composed rather
 * than either replacing the other: whichever fires first aborts the request.
 * The timer is cleared only after the BODY has been read, because a
 * connection that stalls mid-response hangs exactly as hard as one that never
 * answers at all.
 */
export async function fetchUICustomization(
  language: string | null,
  signal?: AbortSignal,
  timeoutMs: number = CUSTOMIZATION_REQUEST_TIMEOUT_MS
): Promise<UICustomization> {
  const query = language
    ? `?locale=${encodeURIComponent(toBcp47Locale(language))}`
    : ''
  const controller = new AbortController()
  const abortForCaller = () => controller.abort(signal?.reason)
  if (signal?.aborted) {
    controller.abort(signal.reason)
  } else {
    signal?.addEventListener('abort', abortForCaller, { once: true })
  }
  const timer = setTimeout(
    () =>
      controller.abort(
        new Error(`customization request timed out after ${timeoutMs}ms`)
      ),
    timeoutMs
  )
  try {
    const response = await fetch(`${backendBaseUrl}/ui/customization${query}`, {
      signal: controller.signal,
      headers: { Accept: 'application/json' }
    })
    if (!response.ok) {
      throw new Error(`customization request failed: ${response.status}`)
    }
    return (await response.json()) as UICustomization
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', abortForCaller)
  }
}
