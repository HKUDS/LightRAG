import Button from '@/components/ui/Button'
import { useCallback } from 'react'
import { controlButtonVariant } from '@/lib/constants'
import { useTranslation } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'

/**
 * Component that toggles the language between English and Chinese.
 */
export default function LanguageToggle() {
  const { i18n } = useTranslation()
  const currentLanguage = i18n.language
  const setLanguage = useSettingsStore.use.setLanguage()

  const setEnglish = useCallback(() => {
    i18n.changeLanguage('en')
    setLanguage('en')
  }, [i18n, setLanguage])

  const setChinese = useCallback(() => {
    i18n.changeLanguage('zh')
    setLanguage('zh')
  }, [i18n, setLanguage])

  if (currentLanguage === 'zh') {
    return (
      <Button
        onClick={setEnglish}
        variant={controlButtonVariant}
        tooltip="Switch to English"
        size="icon"
        side="bottom"
      >
        中
      </Button>
    )
  }
  return (
    <Button
      onClick={setChinese}
      variant={controlButtonVariant}
      tooltip="切换到中文"
      size="icon"
      side="bottom"
    >
      EN
    </Button>
  )
}
