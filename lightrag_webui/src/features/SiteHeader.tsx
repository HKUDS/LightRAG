import Button from '@/components/ui/Button'
import { SiteInfo } from '@/lib/constants'
import ThemeToggle from '@/components/ThemeToggle'
import { TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { useSettingsStore } from '@/stores/settings'
import { cn } from '@/lib/utils'

import { ZapIcon, GithubIcon } from 'lucide-react'

export default function SiteHeader() {
  const currentTab = useSettingsStore.use.currentTab()

  return (
    <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50 flex h-10 w-full border-b px-4 backdrop-blur">
      <a href="/" className="mr-6 flex items-center gap-2">
        <ZapIcon className="size-4 text-emerald-400" aria-hidden="true" />
        <span className="font-bold md:inline-block">{SiteInfo.name}</span>
      </a>

      <div className="flex h-10 flex-1 justify-center">
        <div className="flex h-8 self-center">
          <TabsList className="h-full gap-2">
            <TabsTrigger
              value="documents"
              className={cn(
                'cursor-pointer px-2 py-1 transition-all',
                currentTab === 'documents'
                  ? '!bg-emerald-400 !text-zinc-50'
                  : 'hover:bg-background/60'
              )}
            >
              Documents
            </TabsTrigger>
            <TabsTrigger
              value="knowledge-graph"
              className={cn(
                'cursor-pointer px-2 py-1 transition-all',
                currentTab === 'knowledge-graph'
                  ? '!bg-emerald-400 !text-zinc-50'
                  : 'hover:bg-background/60'
              )}
            >
              Knowledge Graph
            </TabsTrigger>
            <TabsTrigger
              value="retrieval"
              className={cn(
                'cursor-pointer px-2 py-1 transition-all',
                currentTab === 'retrieval'
                  ? '!bg-emerald-400 !text-zinc-50'
                  : 'hover:bg-background/60'
              )}
            >
              Retrieval
            </TabsTrigger>
          </TabsList>
        </div>
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
