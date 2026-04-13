import { useState, useCallback, useEffect } from 'react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import Button from '@/components/ui/Button'
import { listWorkspaces, type LightragWorkspace } from '@/api/lightrag'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import Input from '@/components/ui/Input'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState } from '@/stores/state'
import { Loader2, PaletteIcon, RefreshCw } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'

interface AppSettingsProps {
  className?: string
}

export default function AppSettings({ className }: AppSettingsProps) {
  const [opened, setOpened] = useState<boolean>(false)
  const [knownWorkspaces, setKnownWorkspaces] = useState<LightragWorkspace[]>([])
  const [loadingWorkspaces, setLoadingWorkspaces] = useState<boolean>(false)
  const [workspaceListLoaded, setWorkspaceListLoaded] = useState<boolean>(false)
  const { t } = useTranslation()

  const language = useSettingsStore.use.language()
  const setLanguage = useSettingsStore.use.setLanguage()

  const theme = useSettingsStore.use.theme()
  const setTheme = useSettingsStore.use.setTheme()

  const workspace = useSettingsStore.use.workspace()
  const setWorkspace = useSettingsStore.use.setWorkspace()
  const [workspaceDraft, setWorkspaceDraft] = useState<string>(workspace)

  const handleLanguageChange = useCallback((value: string) => {
    setLanguage(value as 'en' | 'zh' | 'fr' | 'ar' | 'zh_TW' | 'ru' | 'ja' | 'de' | 'uk' | 'ko' | 'vi')
  }, [setLanguage])

  const handleThemeChange = useCallback((value: string) => {
    setTheme(value as 'light' | 'dark' | 'system')
  }, [setTheme])

  const handleWorkspaceChange = useCallback((value: string) => {
    setWorkspace(value)
    useBackendState.getState().resetHealthCheckTimerDelayed(0)
  }, [setWorkspace])

  const handleWorkspaceCandidateSelect = useCallback((value: string) => {
    setWorkspaceDraft(value)
    handleWorkspaceChange(value)
    setOpened(false)
  }, [handleWorkspaceChange])

  const commitWorkspaceDraft = useCallback(() => {
    if (workspaceDraft !== workspace) {
      handleWorkspaceChange(workspaceDraft)
    }
  }, [handleWorkspaceChange, workspace, workspaceDraft])

  const handleOpenChange = useCallback((nextOpened: boolean) => {
    if (!nextOpened) {
      commitWorkspaceDraft()
    }
    setOpened(nextOpened)
  }, [commitWorkspaceDraft])

  const handleWorkspaceKeyDown = useCallback((event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      commitWorkspaceDraft()
    }
  }, [commitWorkspaceDraft])

  const handleWorkspaceBlur = useCallback(() => {
    commitWorkspaceDraft()
  }, [commitWorkspaceDraft])

  const handleWorkspaceDraftChange = useCallback((value: string) => {
    setWorkspaceDraft(value)
  }, [])

  useEffect(() => {
    setWorkspaceDraft(workspace)
  }, [workspace])

  const refreshWorkspaceList = useCallback(async () => {
    setLoadingWorkspaces(true)
    try {
      const result = await listWorkspaces()
      setKnownWorkspaces(result.workspaces || [])
      setWorkspaceListLoaded(true)
    } catch (error) {
      console.error('Failed to load workspaces:', error)
      setKnownWorkspaces([])
      setWorkspaceListLoaded(false)
    } finally {
      setLoadingWorkspaces(false)
    }
  }, [])

  useEffect(() => {
    if (opened && !workspaceListLoaded && !loadingWorkspaces) {
      void refreshWorkspaceList()
    }
  }, [loadingWorkspaces, opened, refreshWorkspaceList, workspaceListLoaded])

  return (
    <Popover open={opened} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" className={cn('h-9 w-9', className)}>
          <PaletteIcon className="h-5 w-5" />
        </Button>
      </PopoverTrigger>
      <PopoverContent side="bottom" align="end" className="w-72">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">
              {t('settings.workspace', { defaultValue: 'Workspace' })}
            </label>
            <Input
              value={workspaceDraft}
              onChange={(event) => handleWorkspaceDraftChange(event.target.value)}
              onBlur={handleWorkspaceBlur}
              onKeyDown={handleWorkspaceKeyDown}
              placeholder={t('settings.workspacePlaceholder', {
                defaultValue: 'Leave empty to use the server default workspace'
              })}
              autoCapitalize="off"
              autoCorrect="off"
              spellCheck={false}
            />
            <p className="text-muted-foreground text-xs">
              {t('settings.workspaceDescription', {
                defaultValue: 'All document, graph, and retrieval requests use this workspace header.'
              })}
            </p>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">
                {t('settings.detectedWorkspaces', { defaultValue: 'Detected workspaces' })}
              </span>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={() => void refreshWorkspaceList()}
              >
                {loadingWorkspaces ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
              </Button>
            </div>
            <div className="flex flex-wrap gap-2">
              {knownWorkspaces.length > 0 ? knownWorkspaces.map((candidate) => (
                <Button
                  key={candidate.id || '__default__'}
                  type="button"
                  variant={workspaceDraft === candidate.id ? 'default' : 'outline'}
                  size="sm"
                  className="h-7 px-2 text-xs"
                  onClick={() => handleWorkspaceCandidateSelect(candidate.id)}
                >
                  {candidate.label}
                </Button>
              )) : (
                <span className="text-muted-foreground text-xs">
                  {loadingWorkspaces
                    ? t('settings.loadingWorkspaces', { defaultValue: 'Loading workspaces...' })
                    : t('settings.noDetectedWorkspaces', { defaultValue: 'No detected workspaces yet' })}
                </span>
              )}
            </div>
          </div>

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
        </div>
      </PopoverContent>
    </Popover>
  )
}
