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


/**
 * The locale a customization response should ALSO apply to the UI chrome, or
 * null when the chrome keeps what it has.
 *
 * Requesting a concrete locale still cannot guarantee matching surfaces: a
 * bundle that does not declare the requested language answers with its own
 * `default_locale`, so English buttons could stand beside, say, Korean
 * welcome text. When the chrome's language was itself only the last-resort
 * `DEFAULT_UI_LANGUAGE` — no explicit choice AND no browser match — nothing
 * is lost by adopting what the bundle actually returned, and the PRD chain is
 * then honored end to end: explicit choice > browser language > bundle
 * default, for BOTH surfaces.
 *
 * The two higher ranks are never overridden: an explicit selection stands
 * even against a bundle that has no content for it, and so does a genuine
 * browser preference — a German browser must not be flipped to the bundle's
 * Korean default just because the bundle omits German.
 *
 * A bundle may also answer in a language this build ships NO interface
 * translation for (any valid BCP 47 locale is a legal bundle locale — a
 * deployment may write its welcome text in Spanish). There is nothing to
 * adopt then, and nothing this side can do about it: the content renders in
 * its own language and direction while the controls stay where they are.
 * That is a bundle-coverage decision, made visible to the operator by the
 * startup warning `locales_without_chrome_translation` drives, not a
 * client-side defect to paper over.
 */
export function bundleLocaleToAdopt(
  responseLocale: string | null | undefined,
  userSelected: unknown,
  currentLanguage: unknown
): SupportedUiLanguage | null {
  if (!responseLocale) return null
  if (userSelected) return null
  if (detectBrowserLanguage() !== null) return null
  // The response speaks BCP 47; the UI ids use underscores. The browser-tag
  // matcher is exactly that normalization (zh-Hant/zh-TW → zh_TW included).
  const adopted = matchBrowserLanguageTag(responseLocale)
  if (!adopted || adopted === currentLanguage) return null
  return adopted
}
