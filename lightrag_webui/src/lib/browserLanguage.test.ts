import { describe, expect, test } from 'bun:test'
import {
  bundleLocaleToAdopt,
  DEFAULT_UI_LANGUAGE,
  detectBrowserLanguage,
  matchBrowserLanguageTag,
  resolveUiLanguage,
  SUPPORTED_UI_LANGUAGES
} from './browserLanguage'

/**
 * Language priority contract (workspace-entry PRD): without an explicit
 * persisted selection the browser languages decide, normalized onto the UI
 * locale ids; when nothing matches, every surface falls back to
 * DEFAULT_UI_LANGUAGE. These tests pin the normalization half of that
 * contract and the shared resolver that keeps the surfaces aligned.
 */

describe('matchBrowserLanguageTag', () => {
  test('region variants normalize to their base language', () => {
    expect(matchBrowserLanguageTag('ar-SA')).toBe('ar')
    expect(matchBrowserLanguageTag('de-DE')).toBe('de')
    expect(matchBrowserLanguageTag('en-US')).toBe('en')
    expect(matchBrowserLanguageTag('fr-CA')).toBe('fr')
    expect(matchBrowserLanguageTag('id-ID')).toBe('id')
  })

  test('bare base tags match case-insensitively', () => {
    expect(matchBrowserLanguageTag('ja')).toBe('ja')
    expect(matchBrowserLanguageTag('KO')).toBe('ko')
    expect(matchBrowserLanguageTag('Vi')).toBe('vi')
  })

  test('traditional-Chinese script/region subtags map onto zh_TW', () => {
    expect(matchBrowserLanguageTag('zh-TW')).toBe('zh_TW')
    expect(matchBrowserLanguageTag('zh-Hant')).toBe('zh_TW')
    expect(matchBrowserLanguageTag('zh-Hant-HK')).toBe('zh_TW')
    expect(matchBrowserLanguageTag('zh-HK')).toBe('zh_TW')
    expect(matchBrowserLanguageTag('zh-MO')).toBe('zh_TW')
  })

  test('other Chinese tags fall back to simplified zh', () => {
    expect(matchBrowserLanguageTag('zh')).toBe('zh')
    expect(matchBrowserLanguageTag('zh-CN')).toBe('zh')
    expect(matchBrowserLanguageTag('zh-Hans-SG')).toBe('zh')
  })

  test('unsupported languages and junk return null', () => {
    expect(matchBrowserLanguageTag('pt-BR')).toBeNull()
    expect(matchBrowserLanguageTag('es')).toBeNull()
    expect(matchBrowserLanguageTag('')).toBeNull()
    expect(matchBrowserLanguageTag('   ')).toBeNull()
    expect(matchBrowserLanguageTag('-')).toBeNull()
  })

  test('underscore form is tolerated on input', () => {
    expect(matchBrowserLanguageTag('zh_TW')).toBe('zh_TW')
    expect(matchBrowserLanguageTag('de_DE')).toBe('de')
    expect(matchBrowserLanguageTag('id_ID')).toBe('id')
  })
})

describe('detectBrowserLanguage', () => {
  test('returns the FIRST supported match, honoring preference order', () => {
    expect(detectBrowserLanguage(['pt-BR', 'es-419', 'ar-SA', 'en'])).toBe('ar')
    expect(detectBrowserLanguage(['de-DE', 'ar-SA'])).toBe('de')
  })

  test('returns null when nothing matches', () => {
    expect(detectBrowserLanguage(['pt-BR', 'es'])).toBeNull()
    expect(detectBrowserLanguage([])).toBeNull()
    expect(detectBrowserLanguage([undefined])).toBeNull()
  })

  test('every supported UI language matches itself', () => {
    for (const language of SUPPORTED_UI_LANGUAGES) {
      expect(detectBrowserLanguage([language])).toBe(language)
    }
  })
})


/**
 * `resolveUiLanguage` is the WHOLE chain, and it exists to be shared: the
 * i18n bootstrap and useCustomizedContent both run it, so UI chrome and
 * branding cannot land on different languages.
 *
 * The defect it closes: the branding loader used to send NO locale when the
 * browser matched nothing, letting the server answer with the bundle's
 * `default_locale` — while the chrome, which has no async fallback, had
 * already rendered English. A bundle defaulting to Korean produced Korean
 * welcome text beside English buttons and settings.
 */
describe('resolveUiLanguage (the shared chain)', () => {
  const withBrowserLanguages = <T>(tags: string[], run: () => T): T => {
    const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator')
    Object.defineProperty(globalThis, 'navigator', {
      value: { languages: tags, language: tags[0] },
      configurable: true
    })
    try {
      return run()
    } finally {
      if (descriptor) Object.defineProperty(globalThis, 'navigator', descriptor)
      // @ts-expect-error - removing the stub when there was no navigator
      else delete globalThis.navigator
    }
  }

  test('an unsupported browser language resolves to the SAME fallback as i18n', () => {
    // Both surfaces call this, so there is one answer by construction.
    withBrowserLanguages(['th-TH', 'pt-BR'], () => {
      expect(resolveUiLanguage(false, null)).toBe(DEFAULT_UI_LANGUAGE)
      expect(resolveUiLanguage(false, 'zh')).toBe(DEFAULT_UI_LANGUAGE)
    })
  })

  test('it never returns the empty "let the server decide" locale', () => {
    // Sending no locale is what let the bundle default disagree with the
    // chrome; the chain must always name a supported language.
    withBrowserLanguages(['th-TH'], () => {
      const resolved = resolveUiLanguage(false, undefined)
      expect(resolved).not.toBe('')
      expect(SUPPORTED_UI_LANGUAGES).toContain(resolved)
    })
  })

  test('the browser language wins over a persisted value that was not chosen', () => {
    withBrowserLanguages(['de-DE'], () => {
      expect(resolveUiLanguage(false, 'zh')).toBe('de')
    })
  })

  test('an explicit persisted choice wins over the browser', () => {
    withBrowserLanguages(['de-DE'], () => {
      expect(resolveUiLanguage(true, 'zh')).toBe('zh')
      // …but only when it is a language this build ships.
      expect(resolveUiLanguage(true, 'xx')).toBe('de')
    })
  })
})


/**
 * `bundleLocaleToAdopt` closes the residual half of the same mismatch:
 * requesting a concrete locale does not guarantee matching content, because a
 * bundle that does not DECLARE that language answers with its own
 * `default_locale`. When the chrome's language was itself only the
 * last-resort fallback, following the bundle is what keeps one language on
 * the page; the two higher ranks of the chain must never be overridden.
 */
describe('bundleLocaleToAdopt', () => {
  const withBrowserLanguages = <T,>(tags: string[], run: () => T): T => {
    const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator')
    Object.defineProperty(globalThis, 'navigator', {
      value: { languages: tags, language: tags[0] },
      configurable: true
    })
    try {
      return run()
    } finally {
      if (descriptor) Object.defineProperty(globalThis, 'navigator', descriptor)
      // @ts-expect-error - removing the stub when there was no navigator
      else delete globalThis.navigator
    }
  }

  test('adopts the bundle default when the chrome only had the last-resort fallback', () => {
    // No explicit choice, browser matches nothing: 'en' was a fallback, and a
    // bundle without English answered in Korean. Follow it.
    withBrowserLanguages(['th-TH'], () => {
      expect(bundleLocaleToAdopt('ko', false, 'en')).toBe('ko')
    })
  })

  test('normalizes the BCP 47 form the response speaks', () => {
    withBrowserLanguages(['th-TH'], () => {
      expect(bundleLocaleToAdopt('zh-TW', false, 'en')).toBe('zh_TW')
      expect(bundleLocaleToAdopt('zh-Hant', false, 'en')).toBe('zh_TW')
    })
  })

  test('an EXPLICIT choice is never overridden by the bundle', () => {
    withBrowserLanguages(['th-TH'], () => {
      expect(bundleLocaleToAdopt('ko', true, 'en')).toBeNull()
    })
  })

  test('a real browser preference is never overridden by the bundle', () => {
    // German browser, bundle without German: the chrome stays German — the
    // PRD ranks the browser language above the bundle default.
    withBrowserLanguages(['de-DE'], () => {
      expect(bundleLocaleToAdopt('ko', false, 'de')).toBeNull()
    })
  })

  test('nothing to do for an absent, unsupported or already-current locale', () => {
    withBrowserLanguages(['th-TH'], () => {
      expect(bundleLocaleToAdopt(undefined, false, 'en')).toBeNull()
      expect(bundleLocaleToAdopt('', false, 'en')).toBeNull()
      // A bundle may declare any valid BCP 47 locale, including ones this
      // build ships no interface translation for. There is nothing to adopt
      // — the content renders in its own language while the controls stay
      // put — and the server warns the operator about it at startup.
      expect(bundleLocaleToAdopt('th', false, 'en')).toBeNull()
      expect(bundleLocaleToAdopt('es', false, 'en')).toBeNull()
      expect(bundleLocaleToAdopt('en', false, 'en')).toBeNull()
    })
  })
})
