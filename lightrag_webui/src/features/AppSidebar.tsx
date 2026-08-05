import { useCallback, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import {
  LayoutDashboardIcon,
  FileTextIcon,
  DatabaseIcon,
  SearchIcon,
  NetworkIcon,
  UsersIcon,
  PanelLeftCloseIcon,
  PanelLeftOpenIcon,
  ZapIcon,
  LogOutIcon,
  type LucideIcon
} from 'lucide-react'

import { cn } from '@/lib/utils'
import { SiteInfo, webuiPrefix } from '@/lib/constants'
import { useSettingsStore } from '@/stores/settings'
import { useAuthStore } from '@/stores/state'
import { navigationService } from '@/services/navigation'
import Button from '@/components/ui/Button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import GithubIcon from '@/components/icons/GithubIcon'

type NavItem = {
  value: string
  labelKey: string
  fallback: string
  icon: LucideIcon
}

/**
 * Primary navigation entries, matching the product's mental model: overview
 * first, then content, then exploration, then administration.
 */
const NAV_ITEMS: NavItem[] = [
  {
    value: 'dashboard',
    labelKey: 'header.dashboard',
    fallback: 'Dashboard',
    icon: LayoutDashboardIcon
  },
  {
    value: 'knowledge-base',
    labelKey: 'header.knowledgeBase',
    fallback: 'Knowledge Base',
    icon: DatabaseIcon
  },
  {
    value: 'documents',
    labelKey: 'header.documents',
    fallback: 'Document Management',
    icon: FileTextIcon
  },
  {
    value: 'knowledge-graph',
    labelKey: 'header.knowledgeGraph',
    fallback: 'Knowledge Graph',
    icon: NetworkIcon
  },
  {
    value: 'retrieval',
    labelKey: 'header.retrieval',
    fallback: 'Retrieval',
    icon: SearchIcon
  },
  {
    value: 'users',
    labelKey: 'header.users',
    fallback: 'User Management',
    icon: UsersIcon
  }
]

interface SidebarItemProps {
  item: NavItem
  active: boolean
  collapsed: boolean
  onSelect: (value: string) => void
}

function SidebarItem({ item, active, collapsed, onSelect }: SidebarItemProps) {
  const { t } = useTranslation()
  const Icon = item.icon
  const label = t(item.labelKey, item.fallback)

  const button = (
    <button
      type="button"
      onClick={() => onSelect(item.value)}
      aria-current={active ? 'page' : undefined}
      className={cn(
        'group relative flex w-full items-center rounded-lg text-sm font-medium transition-colors duration-150',
        'focus-visible:ring-ring focus-visible:ring-2 focus-visible:ring-offset-0 focus-visible:outline-none',
        collapsed ? 'justify-center px-0 py-2.5' : 'gap-3 px-3 py-2.5',
        active
          ? 'bg-primary/12 text-foreground'
          : 'text-muted-foreground hover:bg-foreground/5 hover:text-foreground'
      )}
    >
      {/* Active indicator rail */}
      <span
        aria-hidden="true"
        className={cn(
          'bg-primary absolute top-1/2 left-0 h-5 w-0.5 -translate-y-1/2 rounded-r-full transition-opacity',
          active ? 'opacity-100' : 'opacity-0'
        )}
      />
      <Icon
        className={cn('size-4.5 shrink-0', active ? 'text-primary' : 'text-current')}
        aria-hidden="true"
      />
      {!collapsed && <span className="truncate">{label}</span>}
    </button>
  )

  if (!collapsed) return button

  return (
    <Tooltip>
      <TooltipTrigger asChild>{button}</TooltipTrigger>
      <TooltipContent side="right">{label}</TooltipContent>
    </Tooltip>
  )
}

/**
 * Fixed left navigation rail.
 *
 * Owns primary navigation plus identity/utility actions that previously lived
 * in the top header, so the header can stay a thin contextual bar.
 */
export default function AppSidebar() {
  const { t } = useTranslation()
  const currentTab = useSettingsStore.use.currentTab()
  const collapsed = useSettingsStore.use.sidebarCollapsed()
  const toggleSidebar = useSettingsStore.use.toggleSidebar()
  const { isGuestMode, coreVersion, apiVersion, username, permissions } = useAuthStore()

  const handleSelect = useCallback((value: string) => {
    useSettingsStore.getState().setCurrentTab(value as any)
  }, [])

  const handleLogout = useCallback(() => {
    navigationService.navigateToLogin()
  }, [])

  const versionDisplay = useMemo(
    () => (coreVersion && apiVersion ? `${coreVersion}/${apiVersion}` : null),
    [coreVersion, apiVersion]
  )

  return (
    <TooltipProvider delayDuration={200}>
      <aside
        className={cn(
          'glass-panel relative z-30 flex h-full shrink-0 flex-col rounded-none border-y-0 border-l-0',
          'transition-[width] duration-200 ease-out',
          collapsed ? 'w-16' : 'w-60'
        )}
      >
        {/* Brand */}
        <div
          className={cn(
            'flex h-14 items-center border-b border-[var(--glass-border)]',
            collapsed ? 'justify-center px-2' : 'gap-2.5 px-4'
          )}
        >
          <a
            href={webuiPrefix}
            className="flex min-w-0 items-center gap-2.5"
            aria-label={SiteInfo.name}
          >
            <span className="from-primary flex size-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br to-blue-500 shadow-lg shadow-cyan-500/20">
              <ZapIcon className="size-4.5 text-slate-900" aria-hidden="true" />
            </span>
            {!collapsed && (
              <span className="text-gradient-brand truncate text-base font-semibold tracking-tight">
                {SiteInfo.name}
              </span>
            )}
          </a>
        </div>

        {/* Navigation */}
        <nav
          className="flex flex-1 flex-col gap-1 overflow-y-auto p-2"
          aria-label={t('header.mainNavigation', 'Main navigation')}
        >
          {NAV_ITEMS.filter(item => !permissions || permissions.includes(item.value)).map((item) => (
            <SidebarItem
              key={item.value}
              item={item}
              active={currentTab === item.value}
              collapsed={collapsed}
              onSelect={handleSelect}
            />
          ))}
        </nav>

        {/* Footer utilities */}
        <div className="flex flex-col gap-2 border-t border-[var(--glass-border)] p-2">
          {isGuestMode && !collapsed && (
            <div className="rounded-md bg-amber-500/12 px-2 py-1 text-center text-xs text-amber-400 ring-1 ring-amber-500/20 ring-inset">
              {t('login.guestMode', 'Guest Mode')}
            </div>
          )}

          <div
            className={cn(
              'flex items-center gap-1',
              collapsed ? 'flex-col' : 'justify-between'
            )}
          >
            <div className={cn('flex items-center gap-1', collapsed && 'flex-col')}>
              <Button
                variant="ghost"
                size="icon"
                side="right"
                tooltip={t('header.projectRepository')}
                asChild
              >
                <a href={SiteInfo.github} target="_blank" rel="noopener noreferrer">
                  <GithubIcon className="size-4" />
                </a>
              </Button>

              <Button
                variant="ghost"
                size="icon"
                side="right"
                tooltip={`${t('header.logout')} (${username})`}
                onClick={handleLogout}
              >
                <LogOutIcon className="size-4" aria-hidden="true" />
              </Button>
            </div>

            <Button
              variant="ghost"
              size="icon"
              side="right"
              tooltip={
                collapsed
                  ? t('header.expandSidebar', 'Expand sidebar')
                  : t('header.collapseSidebar', 'Collapse sidebar')
              }
              onClick={toggleSidebar}
              aria-expanded={!collapsed}
            >
              {collapsed ? (
                <PanelLeftOpenIcon className="size-4" aria-hidden="true" />
              ) : (
                <PanelLeftCloseIcon className="size-4" aria-hidden="true" />
              )}
            </Button>
          </div>

          {versionDisplay && !collapsed && (
            <p className="text-muted-foreground/70 px-1 text-center text-[10px] tabular-nums">
              v{versionDisplay}
            </p>
          )}
        </div>
      </aside>
    </TooltipProvider>
  )
}
