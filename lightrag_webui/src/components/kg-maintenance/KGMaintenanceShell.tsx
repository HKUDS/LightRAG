import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'
import type { KGMaintenanceSection } from '@/stores/kgMaintenance'
import {
  BookOpenIcon,
  ClipboardCheckIcon,
  FileSearchIcon,
  GitCompareIcon,
  HistoryIcon,
  LayoutDashboardIcon,
  ListTreeIcon,
  NetworkIcon,
  RefreshCwIcon,
  ShieldCheckIcon
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

type SectionItem = {
  id: KGMaintenanceSection
  label: string
  group: string
  icon: LucideIcon
}

const sections: SectionItem[] = [
  { id: 'overview', label: 'Overview', group: 'Workbench', icon: LayoutDashboardIcon },
  { id: 'graph', label: 'Medical Graph', group: 'Workbench', icon: NetworkIcon },
  { id: 'entities', label: 'Entity Catalog', group: 'Workbench', icon: ListTreeIcon },
  { id: 'relations', label: 'Relation Catalog', group: 'Workbench', icon: GitCompareIcon },
  { id: 'evidence', label: 'Evidence Review', group: '审阅', icon: FileSearchIcon },
  { id: 'quality', label: 'Quality Report', group: '审阅', icon: ShieldCheckIcon },
  { id: 'llm-review', label: 'LLM 审阅', group: '审阅', icon: FileSearchIcon },
  { id: 'patches', label: '候选 Patch', group: '审阅', icon: GitCompareIcon },
  { id: 'judge', label: 'Judge 评判', group: '审阅', icon: ShieldCheckIcon },
  { id: 'approval', label: 'Approval Queue', group: '审阅', icon: ClipboardCheckIcon },
  { id: 'runs', label: 'Run Log', group: 'Iteration', icon: HistoryIcon },
  { id: 'diff', label: 'Diff Review', group: 'Iteration', icon: GitCompareIcon },
  { id: 'rules', label: 'Rule Memory', group: 'Iteration', icon: BookOpenIcon }
]

interface KGMaintenanceShellProps {
  activeSection: KGMaintenanceSection
  onSectionChange: (section: KGMaintenanceSection) => void
  workspaces: string[]
  selectedWorkspace: string | null
  onWorkspaceChange: (workspace: string) => void
  onRefresh: () => void
  onRunReview: () => void
  loading: boolean
  running: boolean
  error: string | null
  children: ReactNode
  inspector: ReactNode
}

export default function KGMaintenanceShell({
  activeSection,
  onSectionChange,
  workspaces,
  selectedWorkspace,
  onWorkspaceChange,
  onRefresh,
  onRunReview,
  loading,
  running,
  error,
  children,
  inspector
}: KGMaintenanceShellProps) {
  const grouped = sections.reduce<Record<string, SectionItem[]>>((acc, section) => {
    acc[section.group] = [...(acc[section.group] || []), section]
    return acc
  }, {})

  return (
    <section className="bg-background flex h-full min-h-0 flex-col">
      <div className="border-border/70 flex min-h-14 shrink-0 flex-wrap items-center justify-between gap-2 border-b px-4 py-2">
        <div className="min-w-0">
          <h1 className="text-base font-semibold">KG Maintenance</h1>
          <p className="text-muted-foreground truncate text-xs">
            {selectedWorkspace || 'No workspace selected'}
          </p>
        </div>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <select
            value={selectedWorkspace || ''}
            onChange={(event) => onWorkspaceChange(event.target.value)}
            className="border-input bg-background h-9 min-w-0 rounded-md border px-3 text-sm sm:min-w-52"
            aria-label="Workspace"
          >
            {workspaces.length === 0 && <option value="">No workspace</option>}
            {workspaces.map((workspace) => (
              <option key={workspace} value={workspace}>
                {workspace}
              </option>
            ))}
          </select>
          <Button variant="outline" size="sm" onClick={onRefresh} disabled={loading}>
            <RefreshCwIcon className={cn('size-4', loading && 'animate-spin')} />
            Refresh
          </Button>
          <Button size="sm" onClick={onRunReview} disabled={!selectedWorkspace || running}>
            <ShieldCheckIcon className={cn('size-4', running && 'animate-pulse')} />
            {running ? 'Running' : 'Run Review'}
          </Button>
        </div>
      </div>

      {error && (
        <div className="border-destructive/30 bg-destructive/10 text-destructive mx-4 mt-3 flex flex-wrap items-center justify-between gap-2 rounded-md border px-3 py-2 text-sm">
          <span>{error}</span>
          <button type="button" className="underline" onClick={onRefresh}>
            Retry / check server logs
          </button>
        </div>
      )}

      {(running || loading) && !error && (
        <div className="border-border/70 bg-muted/30 mx-4 mt-3 rounded-md border px-3 py-2 text-sm">
          {running
            ? 'Running KB iteration review. The package will remain pending user review.'
            : 'Loading review artifacts...'}
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-0 overflow-hidden lg:grid-cols-[220px_minmax(0,1fr)_320px]">
        <aside className="border-border/70 bg-muted/20 min-h-0 overflow-auto border-b p-3 lg:border-r lg:border-b-0">
          {Object.entries(grouped).map(([group, items]) => (
            <div key={group} className="mb-4">
              <div className="text-muted-foreground mb-2 px-2 text-xs font-medium">{group}</div>
              <div className="space-y-1">
                {items.map((item) => {
                  const Icon = item.icon
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => onSectionChange(item.id)}
                      className={cn(
                        'flex h-9 w-full items-center gap-2 rounded-md px-2 text-left text-sm transition-colors',
                        activeSection === item.id
                          ? 'bg-emerald-500 text-white'
                          : 'hover:bg-accent text-foreground'
                      )}
                    >
                      <Icon className="size-4 shrink-0" />
                      <span className="truncate">{item.label}</span>
                    </button>
                  )
                })}
              </div>
            </div>
          ))}
        </aside>

        <main className="min-h-0 overflow-auto p-4">{children}</main>

        <aside className="border-border/70 bg-muted/10 min-h-0 overflow-auto border-t p-4 lg:border-t-0 lg:border-l">
          {inspector}
        </aside>
      </div>
    </section>
  )
}
