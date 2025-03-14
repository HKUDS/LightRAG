import Button from '@/components/ui/Button'
import { SiteInfo } from '@/lib/constants'
import ThemeToggle from '@/components/ThemeToggle'
import { TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { useSettingsStore } from '@/stores/settings'
import { cn } from '@/lib/utils'
import { useTranslation } from 'react-i18next'

import { ZapIcon, GithubIcon } from 'lucide-react'

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
        'pointer-events-auto cursor-pointer px-2 py-1 transition-all',
        currentTab === value ? '!bg-emerald-400 !text-zinc-50' : 'hover:bg-background/60'
      )}
    >
      {children}
    </TabsTrigger>
  )
}

function TabsNavigation() {
  const currentTab = useSettingsStore.use.currentTab()
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
      </TabsList>
    </div>
  )
}

export default function SiteHeader() {
  const { t } = useTranslation()
  return (
    <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 relative sticky top-0 z-50 flex h-10 w-full border-b px-4 backdrop-blur">
      <a href="/" className="mr-6 flex items-center gap-2">
        <ZapIcon className="size-4 text-emerald-400" aria-hidden="true" />
        {/* <img src='/logo.png' className="size-4" /> */}
        <span className="font-bold md:inline-block">{SiteInfo.name}</span>
      </a>

      <div className="pointer-events-none absolute right-0 bottom-0 left-0 flex h-10 justify-center">
        <TabsNavigation />
      </div>

      <div className="flex-grow" />

      <nav className="flex items-center">
        <Button variant="ghost" size="icon" side="bottom" tooltip={t('header.projectRepository')}>
          <a href={SiteInfo.github} target="_blank" rel="noopener noreferrer">
            <GithubIcon className="size-4" aria-hidden="true" />
          </a>
        </Button>
        <ThemeToggle />
      </nav>
    </header>
  )
}
