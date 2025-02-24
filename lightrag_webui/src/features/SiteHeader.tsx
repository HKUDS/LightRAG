import Button from '@/components/ui/Button'
import { SiteInfo } from '@/lib/constants'
import ThemeToggle from '@/components/ThemeToggle'
import { TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { useSettingsStore } from '@/stores/settings'
import { cn } from '@/lib/utils'

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
        'cursor-pointer px-2 py-1 transition-all',
        currentTab === value ? '!bg-emerald-400 !text-zinc-50' : 'hover:bg-background/60'
      )}
    >
      {children}
    </TabsTrigger>
  )
}

function TabsNavigation() {
  const currentTab = useSettingsStore.use.currentTab()

  return (
    <div className="flex h-8 self-center">
      <TabsList className="h-full gap-2">
        <NavigationTab value="documents" currentTab={currentTab}>
          Documents
        </NavigationTab>
        <NavigationTab value="knowledge-graph" currentTab={currentTab}>
          Knowledge Graph
        </NavigationTab>
        <NavigationTab value="retrieval" currentTab={currentTab}>
          Retrieval
        </NavigationTab>
        <NavigationTab value="api" currentTab={currentTab}>
          API
        </NavigationTab>
      </TabsList>
    </div>
  )
}

export default function SiteHeader() {
  return (
    <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50 flex h-10 w-full border-b px-4 backdrop-blur">
      <a href="/" className="mr-6 flex items-center gap-2">
        <ZapIcon className="size-4 text-emerald-400" aria-hidden="true" />
        {/* <img src='/logo.png' className="size-4" /> */}
        <span className="font-bold md:inline-block">{SiteInfo.name}</span>
      </a>

      <div className="flex h-10 flex-1 justify-center">
        <TabsNavigation />
      </div>

      <nav className="flex items-center">
        <Button variant="ghost" size="icon" side="bottom" tooltip="Project Repository">
          <a href={SiteInfo.github} target="_blank" rel="noopener noreferrer">
            <GithubIcon className="size-4" aria-hidden="true" />
          </a>
        </Button>
        <ThemeToggle />
      </nav>
    </header>
  )
}
