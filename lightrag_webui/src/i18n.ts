import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'
import {
  detectBrowserLanguage,
  SUPPORTED_UI_LANGUAGES,
  type SupportedUiLanguage
} from '@/lib/browserLanguage'
import { LEGACY_SETTINGS_STORAGE_KEY } from '@/lib/storageKeys'
import { getSettingsMigrationError } from '@/migrations/splitSettingsStorage'

import en from './locales/en.json'
import zh from './locales/zh.json'
import fr from './locales/fr.json'
import ar from './locales/ar.json'
import zh_TW from './locales/zh_TW.json'
import ru from './locales/ru.json'
import ja from './locales/ja.json'
import de from './locales/de.json'
import uk from './locales/uk.json'
import ko from './locales/ko.json'
import vi from './locales/vi.json'

/**
 * Language priority (workspace-entry PRD): explicit persisted choice >
 * browser language > default. The store's initial 'en' is persisted like any
 * other field, so `language` alone cannot distinguish "chose English" from
 * "never chose" — only an envelope with `languageUserSelected` set wins over
 * the browser languages.
 */
const asSupportedLanguage = (value: unknown): SupportedUiLanguage | null =>
  typeof value === 'string' && (SUPPORTED_UI_LANGUAGES as readonly string[]).includes(value)
    ? (value as SupportedUiLanguage)
    : null

const resolveInitialLanguage = (): SupportedUiLanguage => {
  let persisted: { language?: unknown; languageUserSelected?: unknown } | undefined
  try {
    const settingsString = localStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY)
    if (settingsString) {
      persisted = JSON.parse(settingsString)?.state
    }
  } catch (e) {
    console.error('Failed to get stored language:', e)
  }
  const persistedLanguage = asSupportedLanguage(persisted?.language)
  if (persisted?.languageUserSelected && persistedLanguage) {
    return persistedLanguage
  }
  // WITHOUT an explicit selection the persisted value has no say: it is
  // itself a browser-derived leftover, and honoring it here would (a) keep a
  // stale language after the browser's preferences changed to something
  // unsupported, and (b) disagree with useCustomizedContent, whose chain is
  // `browser ?? bundle default` — the page would then mix UI text in the
  // stale language with branding in the bundle's default one.
  return detectBrowserLanguage() ?? 'en'
}

const initialLanguage = resolveInitialLanguage()

i18n
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      zh: { translation: zh },
      fr: { translation: fr },
      ar: { translation: ar },
      zh_TW: { translation: zh_TW },
      ru: { translation: ru },
      ja: { translation: ja },
      de: { translation: de },
      uk: { translation: uk },
      ko: { translation: ko },
      vi: { translation: vi }
    },
    lng: initialLanguage, // Explicit choice, else browser language, else 'en'
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false
    },
    // Configuration to handle missing translations
    returnEmptyString: false,
    returnNull: false,
  })

// Sync the store to the resolved language WITHOUT marking it user-selected:
// a browser-derived language must keep re-resolving on future visits.
// Skipped entirely after a failed split migration: with skipHydration the
// store holds defaults, and persisting ANY set() would replace the
// unmigrated legacy envelope with a default-heavy v22 one (the storage
// adapter also diverts such writes — this gate just avoids even that).
// i18n itself already runs on `initialLanguage` via init() above.
if (
  getSettingsMigrationError() == null &&
  useSettingsStore.getState().language !== initialLanguage
) {
  useSettingsStore.setState({ language: initialLanguage })
}

// Subscribe to language changes
useSettingsStore.subscribe((state) => {
  const currentLanguage = state.language
  if (i18n.language !== currentLanguage) {
    i18n.changeLanguage(currentLanguage)
  }
})

export default i18n
