import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'

import ar from './locales/ar.json'
import en from './locales/en.json'
import fr from './locales/fr.json'
import zh from './locales/zh.json'
import zh_TW from './locales/zh_TW.json'

const getStoredLanguage = () => {
  try {
    const settingsString = localStorage.getItem('settings-storage')
    if (settingsString) {
      const settings = JSON.parse(settingsString)
      return settings.state?.language || 'en'
    }
  } catch (e) {
    console.error('Failed to get stored language:', e)
  }
  return 'en'
}

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    zh: { translation: zh },
    fr: { translation: fr },
    ar: { translation: ar },
    zh_TW: { translation: zh_TW },
  },
  lng: getStoredLanguage(), // Use stored language settings
  fallbackLng: 'en',
  interpolation: {
    escapeValue: false,
  },
  // Configuration to handle missing translations
  returnEmptyString: false,
  returnNull: false,
})

// Subscribe to language changes
useSettingsStore.subscribe((state) => {
  const currentLanguage = state.language
  if (i18n.language !== currentLanguage) {
    i18n.changeLanguage(currentLanguage)
  }
})

export default i18n
