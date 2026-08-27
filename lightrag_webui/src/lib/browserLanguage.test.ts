import { describe, expect, test } from 'bun:test'
import {
  detectBrowserLanguage,
  matchBrowserLanguageTag,
  SUPPORTED_UI_LANGUAGES
} from './browserLanguage'

/**
 * Language priority contract (workspace-entry PRD): without an explicit
 * persisted selection the browser languages decide, normalized onto the UI
 * locale ids; when nothing matches, callers fall back (UI → 'en',
 * customization request → server default). These tests pin the
 * normalization half of that contract.
 */

describe('matchBrowserLanguageTag', () => {
  test('region variants normalize to their base language', () => {
    expect(matchBrowserLanguageTag('ar-SA')).toBe('ar')
    expect(matchBrowserLanguageTag('de-DE')).toBe('de')
    expect(matchBrowserLanguageTag('en-US')).toBe('en')
    expect(matchBrowserLanguageTag('fr-CA')).toBe('fr')
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
