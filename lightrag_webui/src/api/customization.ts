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

export async function fetchUICustomization(
  language: string,
  signal?: AbortSignal
): Promise<UICustomization> {
  const locale = toBcp47Locale(language)
  const response = await fetch(
    `${backendBaseUrl}/ui/customization?locale=${encodeURIComponent(locale)}`,
    { signal, headers: { Accept: 'application/json' } }
  )
  if (!response.ok) {
    throw new Error(`customization request failed: ${response.status}`)
  }
  return (await response.json()) as UICustomization
}
