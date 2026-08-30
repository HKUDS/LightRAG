import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import { LEGACY_SETTINGS_STORAGE_KEY } from '@/lib/storageKeys'
import { resolveUiLanguage } from '@/lib/browserLanguage'

/**
 * Language priority at startup: an EXPLICIT persisted choice wins; otherwise
 * the browser decides; otherwise 'en'. The persisted value has NO say without
 * the explicit marker — it is itself a browser-derived leftover, so honoring
 * it would keep a stale language after the browser's preferences changed AND
 * disagree with useCustomizedContent, mixing two languages on one page —
 * both now resolve through the SAME `resolveUiLanguage`.
 *
 * i18n.ts resolves this once at module load, so each case re-imports the
 * module with a fresh registry (bun caches per path; the query string makes
 * each import distinct) after seeding storage and navigator.
 */

const data = new Map<string, string>()
const stub = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => {
    data.set(key, value)
  },
  removeItem: (key: string) => {
    data.delete(key)
  },
  clear: () => {
    data.clear()
  }
}

let previousStorage: PropertyDescriptor | undefined
let previousNavigator: PropertyDescriptor | undefined
let variant = 0

const seed = (persisted: Record<string, unknown> | null, languages: string[]) => {
  data.clear()
  if (persisted) {
    stub.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      JSON.stringify({ state: persisted, version: 22 })
    )
  }
  Object.defineProperty(globalThis, 'navigator', {
    value: { languages, language: languages[0] },
    configurable: true
  })
}

const resolveInitialLanguage = async (): Promise<string> => {
  const mod = await import(`@/i18n?case=${++variant}`)
  return (mod.default as { language: string }).language
}

beforeAll(() => {
  previousStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  previousNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator')
  Object.defineProperty(globalThis, 'localStorage', { value: stub, configurable: true })
})

afterAll(() => {
  if (previousStorage) {
    Object.defineProperty(globalThis, 'localStorage', previousStorage)
  }
  if (previousNavigator) {
    Object.defineProperty(globalThis, 'navigator', previousNavigator)
  }
})

describe('startup language resolution', () => {
  test('an EXPLICIT persisted choice outranks the browser', async () => {
    seed({ language: 'fr', languageUserSelected: true }, ['de-DE'])
    expect(await resolveInitialLanguage()).toBe('fr')
  })

  test('without the marker, the browser wins over the persisted leftover', async () => {
    seed({ language: 'fr', languageUserSelected: false }, ['de-DE'])
    expect(await resolveInitialLanguage()).toBe('de')
  })

  test('an unsupported browser language falls back to en, NOT the leftover', async () => {
    // The regression: a French-browser visit persisted 'fr'; the browser is
    // now Portuguese. UI text must not stay French while the customization
    // request (browser → bundle default) asks for something else.
    seed({ language: 'fr', languageUserSelected: false }, ['pt-BR'])
    expect(await resolveInitialLanguage()).toBe('en')
  })

  test('no persisted state at all: browser, else en', async () => {
    seed(null, ['ja'])
    expect(await resolveInitialLanguage()).toBe('ja')
    seed(null, ['pt-BR'])
    expect(await resolveInitialLanguage()).toBe('en')
  })

  test('an unsupported browser language: branding requests the SAME language', async () => {
    // The mismatch this pins: the branding loader used to send NO locale
    // here, letting the server answer with the bundle's `default_locale`,
    // while i18n had already settled on English — Korean welcome text beside
    // English buttons. Both surfaces now run the one chain below.
    seed(null, ['th-TH', 'pt-BR'])

    const uiLanguage = await resolveInitialLanguage()
    // What useCustomizedContent passes to the customization request:
    const brandingLocale = resolveUiLanguage(false, null)

    expect(uiLanguage).toBe('en')
    expect<string>(brandingLocale).toBe(uiLanguage)
    // Never the "let the server decide" sentinel.
    expect(brandingLocale).not.toBe('')
  })
})
