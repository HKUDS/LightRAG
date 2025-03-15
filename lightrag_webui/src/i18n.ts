import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'

import en from './locales/en.json'
import zh from './locales/zh.json'

// Function to sync i18n with store state
export const initializeI18n = async (): Promise<typeof i18n> => {
  // Get initial language from store
  const initialLanguage = useSettingsStore.getState().language

  // Initialize with store language
  await i18n.use(initReactI18next).init({
    resources: {
      en: { translation: en },
      zh: { translation: zh }
    },
    lng: initialLanguage,
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false
    }
  })

  // Subscribe to language changes
  useSettingsStore.subscribe((state) => {
    const currentLanguage = state.language
    if (i18n.language !== currentLanguage) {
      i18n.changeLanguage(currentLanguage)
    }
  })

  return i18n
}

export default i18n
