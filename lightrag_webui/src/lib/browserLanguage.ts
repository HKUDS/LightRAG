/**
 * Browser-language detection for the language priority contract
 * (workspace-entry PRD): an EXPLICIT, persisted user selection wins;
 * otherwise the browser's preferred languages are matched against the UI
 * locales; otherwise `DEFAULT_UI_LANGUAGE`.
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


/**
 * Last resort when no explicit selection exists and the browser's languages
 * match no UI locale.
 *
 * Every surface must land on the SAME language here. Asking the server for
 * no locale instead — letting it answer with the bundle's `default_locale` —
 * looks like a closer reading of "browser language > bundle default", but it
 * makes the two surfaces disagree: the UI chrome has no async fallback and
 * is already rendering English, so a bundle defaulting to another locale
 * produced English buttons beside, say, Korean welcome text. Requesting this
 * language keeps the bundle's own fallback in charge of what it actually
 * covers (`UIBundle.resolve_locale` answers with `default_locale` when the
 * requested locale is not declared), which is the same outcome wherever the
 * bundle has no English variant — and the right one wherever it does.
 */
export const DEFAULT_UI_LANGUAGE: SupportedUiLanguage = 'en'

/**
 * The full priority chain, shared by the i18n bootstrap and the branding
 * loader so the two can never drift apart: explicit persisted choice >
 * browser language > `DEFAULT_UI_LANGUAGE`.
 *
 * `persistedLanguage` is honored ONLY when `userSelected` is true; without
 * that marker the stored value is itself a browser-derived leftover, and
 * honoring it would keep a stale language after the browser's preferences
 * changed.
 */
export function resolveUiLanguage(
  userSelected: unknown,
  persistedLanguage: unknown
): SupportedUiLanguage {
  const persisted =
    typeof persistedLanguage === 'string' &&
    (SUPPORTED_UI_LANGUAGES as readonly string[]).includes(persistedLanguage)
      ? (persistedLanguage as SupportedUiLanguage)
      : null
  if (userSelected && persisted) return persisted
  return detectBrowserLanguage() ?? DEFAULT_UI_LANGUAGE
}
