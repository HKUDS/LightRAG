import Button from '@/components/ui/Button'
import { webuiPrefix } from '@/lib/constants'
import AppSettings from '@/components/AppSettings'
import StatusIndicator from '@/components/status/StatusIndicator'
import { TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { useSettingsStore } from '@/stores/settings'
import { useAuthStore } from '@/stores/state'
import { cn } from '@/lib/utils'
import { useTranslation } from 'react-i18next'
import { navigationService } from '@/services/navigation'
import { ZapIcon, LogOutIcon, BrainIcon } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'

interface NavigationTabProps {
  value: string
  currentTab: string
  children: React.ReactNode
}

function NavigationTab({ value, currentTab, children }: NavigationTabProps) {
  return (
    <TabsTrigger
      value={value}
      className={cn(
        'cursor-pointer px-2 py-1 transition-all',
        currentTab === value ? '!bg-plum !text-plum-foreground' : 'hover:bg-background/60'
      )}
    >
      {children}
    </TabsTrigger>
  )
}

function shouldShowTableExplorer(storageConfig: any) {
  // Always show for now - TODO: fix storageConfig state propagation from health check
  return true

  // Original logic (storageConfig not being populated from health endpoint):
  // if (import.meta.env.DEV) return true
  // return (
  //   storageConfig &&
  //   storageConfig.kv_storage === 'PGKVStorage' &&
  //   storageConfig.doc_status_storage === 'PGDocStatusStorage' &&
  //   storageConfig.graph_storage === 'PGGraphStorage' &&
  //   storageConfig.vector_storage === 'PGVectorStorage'
  // )
}

function TabsNavigation() {
  const currentTab = useSettingsStore.use.currentTab()
  const storageConfig = useSettingsStore.use.storageConfig()
  const { t } = useTranslation()

  return (
    <div className="flex h-8 self-center">
      <TabsList className="h-full gap-2">
        <NavigationTab value="documents" currentTab={currentTab}>
          {t('header.documents')}
        </NavigationTab>
        <NavigationTab value="knowledge-graph" currentTab={currentTab}>
          {t('header.knowledgeGraph')}
        </NavigationTab>
        <NavigationTab value="retrieval" currentTab={currentTab}>
          {t('header.retrieval')}
        </NavigationTab>
        <NavigationTab value="api" currentTab={currentTab}>
          {t('header.api')}
        </NavigationTab>
        {shouldShowTableExplorer(storageConfig) && (
          <NavigationTab value="table-explorer" currentTab={currentTab}>
            {t('header.tables')}
          </NavigationTab>
        )}
      </TabsList>
    </div>
  )
}

export default function SiteHeader() {
  const { t } = useTranslation()
  const { isGuestMode, username, webuiTitle, webuiDescription } = useAuthStore()
  const enableHealthCheck = useSettingsStore.use.enableHealthCheck()

  const handleLogout = () => {
    navigationService.navigateToLogin();
  }

  return (
    <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50 flex h-10 w-full border-b px-4 backdrop-blur">
      <div className="min-w-[200px] w-auto flex items-center">
        <a href={webuiPrefix} className="flex items-center gap-2">
          <BrainIcon className="size-4 text-plum" aria-hidden="true" />
        </a>
        {webuiTitle && (
          <div className="flex items-center">
            <span className="mx-1 text-xs text-gray-500 dark:text-gray-400">|</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="font-medium text-sm cursor-default">
                    {webuiTitle}
                  </span>
                </TooltipTrigger>
                {webuiDescription && (
                  <TooltipContent side="bottom">
                    {webuiDescription}
                  </TooltipContent>
                )}
              </Tooltip>
            </TooltipProvider>
          </div>
        )}
      </div>

      <div className="flex h-10 flex-1 items-center justify-center">
        <TabsNavigation />
      </div>

      <nav className="w-[200px] flex items-center justify-end">
        <div className="flex items-center gap-2">
          {enableHealthCheck && <StatusIndicator />}
          <AppSettings />
          {!isGuestMode && (
            <Button
              variant="ghost"
              size="icon"
              side="bottom"
              tooltip={`${t('header.logout')} (${username})`}
              onClick={handleLogout}
            >
              <LogOutIcon className="size-4" aria-hidden="true" />
            </Button>
          )}
        </div>
      </nav>
    </header>
  )
}
