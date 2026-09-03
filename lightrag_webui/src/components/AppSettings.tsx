import { useState, useCallback, useEffect } from 'react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { useSettingsStore } from '@/stores/settings'
import { PaletteIcon, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { getLangfuseConfig, updateLangfuseConfig } from '@/api/lightrag'

interface AppSettingsProps {
  className?: string
}

export default function AppSettings({ className }: AppSettingsProps) {
  const [opened, setOpened] = useState<boolean>(false)
  const { t } = useTranslation()

  const language = useSettingsStore.use.language()
  const setLanguage = useSettingsStore.use.setLanguage()

  const theme = useSettingsStore.use.theme()
  const setTheme = useSettingsStore.use.setTheme()

  // Langfuse state
  const [langfusePublicKey, setLangfusePublicKey] = useState('')
  const [langfuseSecretKey, setLangfuseSecretKey] = useState('')
  const [langfuseHost, setLangfuseHost] = useState('')
  const [langfuseEnabled, setLangfuseEnabled] = useState(false)
  const [secretKeySet, setSecretKeySet] = useState(false)
  const [isLoadingLangfuse, setIsLoadingLangfuse] = useState(false)
  const [isSavingLangfuse, setIsSavingLangfuse] = useState(false)
  const [langfuseStatusMsg, setLangfuseStatusMsg] = useState<{
    type: 'success' | 'error'
    text: string
  } | null>(null)

  const handleLanguageChange = useCallback(
    (value: string) => {
      setLanguage(
        value as
          | 'en'
          | 'zh'
          | 'fr'
          | 'ar'
          | 'zh_TW'
          | 'ru'
          | 'ja'
          | 'de'
          | 'uk'
          | 'ko'
          | 'vi'
          | 'id'
      )
    },
    [setLanguage]
  )

  const handleThemeChange = useCallback(
    (value: string) => {
      setTheme(value as 'light' | 'dark' | 'system')
    },
    [setTheme]
  )

  const fetchLangfuseConfig = useCallback(async () => {
    try {
      setIsLoadingLangfuse(true)
      const config = await getLangfuseConfig()
      setLangfusePublicKey(config.public_key || '')
      setLangfuseHost(config.host || '')
      setSecretKeySet(config.secret_key_set)
      setLangfuseEnabled(config.enabled)
    } catch {
      // Endpoint may fail if not authorized or offline; fail gracefully
    } finally {
      setIsLoadingLangfuse(false)
    }
  }, [])

  useEffect(() => {
    if (opened) {
      fetchLangfuseConfig()
      setLangfuseStatusMsg(null)
    }
  }, [opened, fetchLangfuseConfig])

  const handleSaveLangfuse = async () => {
    setIsSavingLangfuse(true)
    setLangfuseStatusMsg(null)
    try {
      const payload: { public_key?: string; secret_key?: string; host?: string } = {
        public_key: langfusePublicKey.trim(),
        host: langfuseHost.trim()
      }
      if (langfuseSecretKey.trim()) {
        payload.secret_key = langfuseSecretKey.trim()
      }

      const res = await updateLangfuseConfig(payload)
      setLangfusePublicKey(res.public_key || '')
      setLangfuseHost(res.host || '')
      setSecretKeySet(res.secret_key_set)
      setLangfuseEnabled(res.enabled)
      setLangfuseSecretKey('')
      setLangfuseStatusMsg({ type: 'success', text: 'Langfuse settings saved' })
    } catch {
      setLangfuseStatusMsg({ type: 'error', text: 'Failed to update Langfuse settings' })
    } finally {
      setIsSavingLangfuse(false)
    }
  }

  return (
    <Popover open={opened} onOpenChange={setOpened}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          side="bottom"
          tooltip={t('header.appSettings')}
          aria-label={t('header.appSettings')}
          className={cn(
            'relative h-9 w-9 after:absolute after:-inset-1 after:content-[\'\']',
            className
          )}
        >
          <PaletteIcon className="h-5 w-5" aria-hidden="true" />
        </Button>
      </PopoverTrigger>
      <PopoverContent side="bottom" align="end" className="w-80 p-4">
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

          <div className="my-1 border-t border-border" />

          {/* Langfuse Observability Section */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold">Langfuse Integration</span>
              <span
                className={cn(
                  'rounded-full px-2 py-0.5 text-xs font-medium',
                  langfuseEnabled
                    ? 'bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300'
                    : 'bg-muted text-muted-foreground'
                )}
              >
                {langfuseEnabled ? 'Active' : 'Disabled'}
              </span>
            </div>

            {isLoadingLangfuse ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <>
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-muted-foreground">Public Key</label>
                  <Input
                    placeholder="pk-lf-..."
                    value={langfusePublicKey}
                    onChange={(e) => setLangfusePublicKey(e.target.value)}
                    className="h-8 text-xs"
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-muted-foreground">
                    Secret Key {secretKeySet && !langfuseSecretKey ? '(Configured)' : ''}
                  </label>
                  <Input
                    type="password"
                    placeholder={secretKeySet ? '••••••••••••••••' : 'sk-lf-...'}
                    value={langfuseSecretKey}
                    onChange={(e) => setLangfuseSecretKey(e.target.value)}
                    className="h-8 text-xs"
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-muted-foreground">Host URL</label>
                  <Input
                    placeholder="https://cloud.langfuse.com"
                    value={langfuseHost}
                    onChange={(e) => setLangfuseHost(e.target.value)}
                    className="h-8 text-xs"
                  />
                </div>

                {langfuseStatusMsg && (
                  <div
                    className={cn(
                      'flex items-center gap-1.5 text-xs',
                      langfuseStatusMsg.type === 'success' ? 'text-green-600' : 'text-red-500'
                    )}
                  >
                    {langfuseStatusMsg.type === 'success' ? (
                      <CheckCircle2 className="h-3.5 w-3.5" />
                    ) : (
                      <AlertCircle className="h-3.5 w-3.5" />
                    )}
                    <span>{langfuseStatusMsg.text}</span>
                  </div>
                )}

                <Button
                  size="sm"
                  onClick={handleSaveLangfuse}
                  disabled={isSavingLangfuse}
                  className="mt-1 w-full"
                >
                  {isSavingLangfuse ? (
                    <>
                      <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    'Save Langfuse Config'
                  )}
                </Button>
              </>
            )}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}