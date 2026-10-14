import { useState, useCallback, useEffect } from 'react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import Button from '@/components/ui/Button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import Separator from '@/components/ui/Separator'
import { useSettingsStore } from '@/stores/settings'
import { PaletteIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'
import {
  getLangfuseTracingStatus,
  updateLangfuseTracing,
  type LangfuseTracingStatus
} from '@/api/lightrag'

interface AppSettingsProps {
  className?: string
}

export default function AppSettings({ className }: AppSettingsProps) {
  const [opened, setOpened] = useState<boolean>(false)
  const [langfuseStatus, setLangfuseStatus] = useState<LangfuseTracingStatus | null>(null)
  const [isLangfuseLoading, setIsLangfuseLoading] = useState(false)
  const [isLangfuseUpdating, setIsLangfuseUpdating] = useState(false)
  const { t } = useTranslation()

  const language = useSettingsStore.use.language()
  const setLanguage = useSettingsStore.use.setLanguage()

  const theme = useSettingsStore.use.theme()
  const setTheme = useSettingsStore.use.setTheme()

  const handleLanguageChange = useCallback((value: string) => {
    setLanguage(value as 'en' | 'zh' | 'fr' | 'ar' | 'zh_TW' | 'ru' | 'ja' | 'de' | 'uk' | 'ko' | 'vi' | 'id')
  }, [setLanguage])

  const handleThemeChange = useCallback((value: string) => {
    setTheme(value as 'light' | 'dark' | 'system')
  }, [setTheme])

  useEffect(() => {
    if (!opened) return

    let cancelled = false

    const loadLangfuseStatus = async () => {
      setIsLangfuseLoading(true)
      try {
        const status = await getLangfuseTracingStatus()
        if (!cancelled) setLangfuseStatus(status)
      } catch (error) {
        if (!cancelled) {
          console.error('Failed to load Langfuse tracing status:', error)
          toast.error(t('settings.langfuseLoadFailed'))
        }
      } finally {
        if (!cancelled) setIsLangfuseLoading(false)
      }
    }

    void loadLangfuseStatus()

    return () => {
      cancelled = true
    }
  }, [opened, t])

  const handleLangfuseTracingChange = useCallback(async () => {
    if (
      !langfuseStatus ||
      isLangfuseUpdating ||
      !langfuseStatus.installed ||
      !langfuseStatus.configured
    ) {
      return
    }

    setIsLangfuseUpdating(true)
    try {
      const status = await updateLangfuseTracing(!langfuseStatus.enabled)
      setLangfuseStatus(status)
    } catch (error) {
      console.error('Failed to update Langfuse tracing:', error)
      toast.error(t('settings.langfuseUpdateFailed'))
    } finally {
      setIsLangfuseUpdating(false)
    }
  }, [isLangfuseUpdating, langfuseStatus, t])

  return (
    <Popover open={opened} onOpenChange={setOpened}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          side="bottom"
          tooltip={t('header.appSettings')}
          // A tooltip is only a description, never the accessible name — an
          // icon-only button needs the explicit aria-label.
          aria-label={t('header.appSettings')}
          // 36px visual size; the ::after overlay widens the TOUCH target to
          // 44px (PRD: primary touch targets ≥44px) without moving layout.
          className={cn(
            'relative h-9 w-9 after:absolute after:-inset-1 after:content-[\'\']',
            className
          )}
        >
          <PaletteIcon className="h-5 w-5" aria-hidden="true" />
        </Button>
      </PopoverTrigger>
      <PopoverContent side="bottom" align="end" className="w-80">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">{t('settings.language')}</label>
            <Select value={language} onValueChange={handleLanguageChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="en">English</SelectItem>
                <SelectItem value="zh">中文</SelectItem>
                <SelectItem value="fr">Français</SelectItem>
                <SelectItem value="ar">العربية</SelectItem>
                <SelectItem value="zh_TW">繁體中文</SelectItem>
                <SelectItem value="ru">Русский</SelectItem>
                <SelectItem value="ja">日本語</SelectItem>
                <SelectItem value="de">Deutsch</SelectItem>
                <SelectItem value="uk">Українська</SelectItem>
                <SelectItem value="ko">한국어</SelectItem>
                <SelectItem value="vi">Tiếng Việt</SelectItem>
                <SelectItem value="id">Bahasa Indonesia</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">{t('settings.theme')}</label>
            <Select value={theme} onValueChange={handleThemeChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="light">{t('settings.light')}</SelectItem>
                <SelectItem value="dark">{t('settings.dark')}</SelectItem>
                <SelectItem value="system">{t('settings.system')}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Separator />

          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <label htmlFor="langfuse-tracing-toggle" className="text-sm font-medium">
                {t('settings.langfuseTracing')}
              </label>
              <p className="text-muted-foreground mt-1 text-xs">
                {langfuseStatus && (!langfuseStatus.installed || !langfuseStatus.configured)
                  ? t('settings.langfuseUnavailable')
                  : t('settings.langfuseTracingDescription')}
              </p>
            </div>
            <button
              id="langfuse-tracing-toggle"
              type="button"
              role="switch"
              aria-label={t('settings.langfuseTracing')}
              aria-checked={langfuseStatus?.enabled ?? false}
              disabled={
                isLangfuseLoading ||
                isLangfuseUpdating ||
                !langfuseStatus?.installed ||
                !langfuseStatus.configured
              }
              onClick={handleLangfuseTracingChange}
              className={cn(
                'relative mt-0.5 inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors',
                'focus-visible:ring-ring focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:outline-none',
                'disabled:cursor-not-allowed disabled:opacity-50',
                langfuseStatus?.enabled ? 'bg-primary' : 'bg-input'
              )}
            >
              <span
                aria-hidden="true"
                className={cn(
                  'bg-background pointer-events-none block h-5 w-5 rounded-full shadow-lg ring-0 transition-transform',
                  langfuseStatus?.enabled ? 'translate-x-5' : 'translate-x-0'
                )}
              />
            </button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}
