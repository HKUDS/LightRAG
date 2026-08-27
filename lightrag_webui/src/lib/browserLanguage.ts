/**
 * Browser-language detection for the language priority contract
 * (workspace-entry PRD): an EXPLICIT, persisted user selection wins;
 * otherwise the browser's preferred languages are matched against the UI
 * locales; otherwise callers fall back (UI text → 'en'; the customization
 * request sends NO locale so the server resolves the bundle's
 * default_locale).
 *
 * Pure module: no store imports, so it is usable from the pre-React i18n
 * bootstrap and unit-testable with injected preference lists.
 */

/** Internal ids of the UI locales — matches src/locales/*.json filenames and
 * the settings store's Language union. */
export const SUPPORTED_UI_LANGUAGES = [
  'en',
  'zh',
  'fr',
  'ar',
  'zh_TW',
  'ru',
  'ja',
  'de',
  'uk',
  'ko',
  'vi'
] as const

export type SupportedUiLanguage = (typeof SUPPORTED_UI_LANGUAGES)[number]

const SUPPORTED = new Set<string>(SUPPORTED_UI_LANGUAGES)

/** Map one BCP 47 browser tag onto a supported UI language id, or null. */
export function matchBrowserLanguageTag(tag: string): SupportedUiLanguage | null {
  if (!tag) return null
  const subtags = tag.trim().replace(/_/g, '-').toLowerCase().split('-').filter(Boolean)
  if (subtags.length === 0) return null
  const base = subtags[0]
  // Traditional-Chinese script/region subtags map onto zh_TW — the only
  // traditional variant the UI ships; other zh tags fall through to 'zh'.
  if (
    base === 'zh' &&
    subtags.slice(1).some((s) => s === 'hant' || s === 'tw' || s === 'hk' || s === 'mo')
  ) {
    return 'zh_TW'
  }
  return SUPPORTED.has(base) ? (base as SupportedUiLanguage) : null
}

/**
 * First supported match across the browser's preference list (most-preferred
 * first), or null when nothing matches. Defaults to `navigator.languages`
 * (falling back to `navigator.language`); pass a list to override in tests.
 */
export function detectBrowserLanguage(
  preferred?: readonly (string | undefined)[]
): SupportedUiLanguage | null {
  let list: readonly (string | undefined)[]
  if (preferred !== undefined) {
    list = preferred
  } else if (typeof navigator !== 'undefined') {
    list =
      navigator.languages && navigator.languages.length > 0
        ? navigator.languages
        : [navigator.language]
  } else {
    list = []
  }
  for (const tag of list) {
    const match = matchBrowserLanguageTag(tag ?? '')
    if (match) return match
  }
  return null
}
