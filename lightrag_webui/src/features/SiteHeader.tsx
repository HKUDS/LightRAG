import Button from '@/components/ui/Button'
import { SiteInfo } from '@/lib/constants'
import ThemeToggle from '@/components/ThemeToggle'
import { TabsList, TabsTrigger } from '@/components/ui/Tabs'

import { ZapIcon, GithubIcon } from 'lucide-react'

export default function SiteHeader() {
  return (
    <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50 flex h-10 w-full border-b px-4 backdrop-blur">
      <a href="/" className="mr-6 flex items-center gap-2">
        <ZapIcon className="size-4 text-teal-400" aria-hidden="true" />
        <span className="font-bold md:inline-block">{SiteInfo.name}</span>
      </a>

      <div className="flex h-10 flex-1 justify-center">
        <div className="flex h-8 self-center">
          <TabsList className="h-full gap-2">
            <TabsTrigger
              value="documents"
              className="hover:bg-background/60 cursor-pointer px-2 py-1 transition-all"
            >
              Documents
            </TabsTrigger>
            <TabsTrigger
              value="knowledge-graph"
              className="hover:bg-background/60 cursor-pointer px-2 py-1 transition-all"
            >
              Knowledge Graph
            </TabsTrigger>
            {/* <TabsTrigger
              value="settings"
              className="hover:bg-background/60 cursor-pointer px-2 py-1 transition-all"
            >
              Settings
            </TabsTrigger> */}
          </TabsList>
        </div>
      </div>

      <nav className="flex items-center">
        <Button variant="ghost" size="icon">
          <a href={SiteInfo.github} target="_blank" rel="noopener noreferrer">
            <GithubIcon className="size-4" aria-hidden="true" />
          </a>
        </Button>
        <ThemeToggle />
      </nav>
    </header>
  )
}
